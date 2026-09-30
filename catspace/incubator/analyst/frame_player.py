"""THE FRAME PLAYER (Kaveh 2026-09-27: "you look at the board, you see a frame of pieces ... you pick the right frames,
you analyze the frames ... these frames may interact with the rest of the board ... you resolve each frame and out of
the resolution ... there's one stable output ... try some positions and see if you make the best move ... if not,
let's dig in"). A move is chosen from the frames alone and its reasoning is kept:

  1. the frames: around every contested square, the pieces that attack or defend it, and those that would once the
     frame's pieces are lifted off (fates.square_frame);
  2. each frame resolved exactly: the capture sequence on its square, each side capturing or passing (alpha-beta);
  3. the stable output: a move's value is what it wins, minus the opponent's best frame after it, plus their best
     frame if we had passed (the ledger); the best value is chosen (ties: every tied move is listed).

Scored on held-out positions against Stockfish (its evaluation of every move, cached): does the choice lose 1.5 pawns
or more against the best move? The misses are printed with their trace, for the dig.

    python -m catspace.incubator.analyst.frame_player --n 200
"""
from __future__ import annotations

import argparse
import json
import time

import chess

from catspace.io import paths

from .fates import captured_value, changed_squares, joint_threats, joint_threats_book, ledger_move_value, passed, square_frame, threats
from .gauge_verdicts import LOSS_CP, positions, truths
from .engine import Engine


JOINT = True                                                        # frames resolved jointly with the pieces they pull in (2026-09-28)
SELECT = None                                                       # the prior's choice of pulled-in pieces (None: all of them)
PLANS = "none"                                                      # the tactical plan layer that breaks ties in the top set (plan_layers)
LEDGER = "one_ply"                                                  # "one_ply": captured - their best frame after + their best if we pass;
                                                                    # "two_ply": captured + min over their replies of (- their capture + our best frame after)
_LAYER = None


def _layer():
    global _LAYER
    if _LAYER is None or _LAYER.name != PLANS:
        from .plan_layers import get
        _LAYER = get(PLANS)
    return _LAYER


def _our_zone(b: chess.Board, book: dict) -> set:
    """The squares our winning frames live on: their squares, pieces, and the lines between (the replies that touch
    none of them cannot change what those frames win)."""
    z = set()
    for fr, rel, v, ln in book.values():
        if v > 0:
            z |= set(fr[0]) | set(fr[2])
    zs = list(z)
    for i, x in enumerate(zs):
        for y in zs[i + 1:]:
            z |= set(chess.SquareSet(chess.between(x, y)))
    return z


def _adds_defender(b: chess.Board, r: chess.Move, targets: set) -> bool:
    nb = b.copy(stack=False); nb.push(r)
    return any(len(nb.attackers(b.turn, t)) > len(b.attackers(b.turn, t)) for t in targets if b.piece_at(t) is not None and b.piece_at(t).color == b.turn)


def two_ply_value(b: chess.Board, m: chess.Move, opponent: str = "all") -> tuple:
    """THE TWO-PLY FRAME LEDGER (2026-09-28, from the own misses: a4 in r4b1r/1ppkqNpp/.. -- Nxh8 walks into axb3;
    Rxg4+ in r5rk/pbp2Q2/.. -- Nxg4 deflects the f7 guard; Bd3 in 8/5pk1/.. -- a fork is our threat): our move, then the
    opponent's reply, then our best frame. Their replies are those that can matter -- every capture, check or evasion,
    a move of a piece our frames target, a move onto their squares, a move that adds a defender -- and an irrelevant
    move (the pass). -> (value, their best reply uci or '--', our best frame line after it)."""
    nb = b.copy(stack=False); nb.push(m)
    if nb.is_checkmate():
        return 1e6, None, []
    got = captured_value(b, m)
    ours = lambda p: (joint_threats(p, square_frame)[0], joint_threats(p, square_frame)[2]) if p.turn == b.turn else (-joint_threats(p, square_frame)[0], [])
    worst = None
    pb = passed(nb)
    if pb is not None:                                                # the pass: our frames after the move stand
        v, ln = ours(pb); worst = (v, "--", ln)
        zone = _our_zone(pb, joint_threats_book(pb, square_frame)[1]); targets = {q for q in zone if nb.piece_at(q) is not None and nb.piece_at(q).color == nb.turn}
    else:
        zone = set(); targets = set()
    if opponent == "instinct":                                           # the instinctive opponent (Kaveh): only first-instinct replies, or a pass
        from .instincts import instincts
        replies = [i.move for i in instincts(nb)]
    else:
        replies = [r for r in nb.legal_moves if nb.is_check() or nb.is_capture(r) or nb.gives_check(r) or r.from_square in zone
                   or r.to_square in zone or _adds_defender(nb, r, targets)]
    for r in replies:
        ab = nb.copy(stack=False); ab.push(r)
        if ab.is_checkmate():
            return -1000.0, r.uci(), []
        v, ln = ours(ab)
        v -= captured_value(nb, r)
        if worst is None or v < worst[0]:
            worst = (v, r.uci(), ln)
    if worst is None:
        return 0.0, None, []                                          # stalemate
    return got + worst[0], worst[1], worst[2]


CANDIDATES = 6                                                      # two_ply_candidates: the one-ply top moves that get the reply ply
TOTAL_CAP = [None]                                                  # a hard node total for a position (equal-compute gauges); None: off
KSTATS: dict = {}                                                   # the knowledge's firing counts, summed over choose calls
NODES = [0]                                                         # the zone search's nodes for the last choose (for equal-compute gauges)
BUDGET_POLICY = "fixed"                                            # zone_candidates: "fixed" (equal shares) | "watch" (the king watches' policy)
DECIDER_THRESHOLD = [0.5]                                           # BUDGET_POLICY "decider": go on past a rung when the learned chance that it
                                                                    # changes the decision is at least this (depth_decider; the planner's knob)
DECIDER_PRICE = [None]                                              # ... or, when set, a price on search: go on when the chance is worth
                                                                    # price x the next rung's expected nodes (depth_decider.stopper)
KING_WATCH = False                                                  # zone_candidates: the king watches' moves join the candidates
PROPOSE = 0                                                         # zone_candidates: the moves realizing the proposer's top-k what-if edits join (0: off)
JOBS = False                                                        # zone_candidates: the job ledger's operator moves join the candidates
PLANNER = None                                                      # the planner (frame_planner): None -> MaterialPlanner() defaults
ZONE_BUDGET = 20000                                                 # zone_candidates: nodes per candidate per depth (all deepen together: 4, 6, 8 plies)


def _one_ply_values(b: chess.Board, before, book) -> dict:
    out = {}
    for m in b.legal_moves:
        nb = b.copy(stack=False); nb.push(m)
        if nb.is_checkmate():
            out[m.uci()] = 1e6; continue
        after = joint_threats(nb, square_frame, select=SELECT, parent=book, changed=changed_squares(b, m))
        out[m.uci()] = captured_value(b, m) - after[0] + before[0]
    return out


def candidates(b: chess.Board, before=None, book=None, ledger: str = "zone_candidates", count: int | None = None) -> list:
    """THE CANDIDATE MOVES the reply ply or the zone search looks at -- one function for the player and every harness
    (bug 2026-09-30: the ladder's own copy lacked the quiet threats): the one-ply top `count` (CANDIDATES) and its ties,
    every capture, check and promotion, the quiet threats (the null-move what-if), and with the zone ledger the king
    watches' moves and the proposer's constructions when those are switched on."""
    if before is None:
        stand = passed(b)
        before, book = joint_threats_book(stand, square_frame, select=SELECT) if stand is not None else ((0.0, None, []), None)
    moves = list(b.legal_moves)
    one = _one_ply_values(b, before, book)                            # every capture and check (Kaveh: which what-ifs to consider)
    ranked = sorted(one, key=lambda u: -one[u])
    top1 = max(one.values())
    keep = set(ranked[:(CANDIDATES if count is None else count)]) | {u for u in one if one[u] == top1} \
        | {m.uci() for m in moves if b.is_capture(m) or b.gives_check(m) or m.promotion}   # (the one-ply top set: its quiet ties stay in)
    now = joint_threats(b, square_frame)[0]                           # QUIET THREATS (puzzle dig 2026-09-28: 20 of 41 misses were quiet
    for m in moves:                                                   # moves never given the reply ply): the null-move what-if -- if they
        if m.uci() in keep:                                           # did nothing after it, would our frames win more than they do now?
            continue
        nb = b.copy(stack=False); nb.push(m); pb = passed(nb)
        if pb is not None and joint_threats(pb, square_frame)[0] >= now + 0.5:
            keep.add(m.uci())
    if KING_WATCH and ledger == "zone_candidates":                     # the standing king frames: moves toward the enemy king (a member's
        from .king_watch import candidate_moves                         # route, a condition piece taken) and our king's answers
        keep |= set(candidate_moves(b))
    if JOBS and ledger == "zone_candidates":                           # the job ledger's moves: pressure on holders, deflection of the
        from .jobs import ledger as _jobs, operators                    # overloaded, a double-attacking discovery, our pins relieved
        keep |= set(operators(b, _jobs(b)))
    if PROPOSE and ledger == "zone_candidates":                        # the what-if proposer's top edits: the moves that realize them join
        from .whatif_proposer import propose_moves                      # (AlphaGeometry-style constructions; the zone search verifies)
        keep |= set(propose_moves(b, one, PROPOSE))
    return [m for m in moves if m.uci() in keep]


def choose(b: chess.Board) -> dict:
    """-> {'values': {uci: value}, 'best': [ucis with the top value], 'trace': {uci: (their best after, their best if
    we pass)}} -- the frames' own reasoning for every legal move."""
    stand = passed(b)
    if JOINT:
        before, book = joint_threats_book(stand, square_frame, select=SELECT) if stand is not None else ((0.0, None, []), None)
    else:
        before = threats(stand, square_frame) if stand is not None else (0.0, None, [])
    values = {}; trace = {}
    if LEDGER in ("two_ply", "two_ply_candidates", "two_ply_instinct", "zone_candidates"):
        moves = list(b.legal_moves)
        if LEDGER in ("two_ply_candidates", "two_ply_instinct", "zone_candidates"):
            moves = candidates(b, before, book, LEDGER)
        if LEDGER == "zone_candidates":                                     # the zone search verifies the candidates, all at the same depth
            from .zone_search import ZoneSearch                             # (Kaveh: plies are cheap in the right frame)
            from .frame_planner import MaterialPlanner                      # the PLANNER sets every limit and names the goals
            plan = (PLANNER or MaterialPlanner()).plan(b, moves)            # (Kaveh 2026-09-30: no hard-coded limits in the tactics)
            alloc = None; stop = None
            zs = ZoneSearch(spec=plan.spec); spent = 0
            if BUDGET_POLICY == "watch":                                    # the tactical policy's budget: king danger gets more, and deeper
                from .king_watch import budget_policy
                alloc = budget_policy(b, moves)
            elif BUDGET_POLICY == "learned":                                # phase 1 small; phase 2 shared by the learned chance that more
                from .search_policy import allocate, SMALL                  # search changes a candidate (the table kept between phases)
                zs.budget = SMALL
                zs.cap_total = TOTAL_CAP[0]
                alloc = allocate(b, moves, zs); spent = zs.total
                zs.budget = plan.spec.budget; zs.spent_before = spent
            elif BUDGET_POLICY == "decider":                                # the learned stop-or-continue call after every rung
                from .depth_decider import stopper                          # (depth_decider: trained on the depth ladder's labels)
                stop = stopper(b, tuple(sorted(plan.spec.depths)), DECIDER_THRESHOLD[0], DECIDER_PRICE[0])
            if BUDGET_POLICY in ("fixed", "decider"):
                zs.cap_total = TOTAL_CAP[0]
            got = zs.rank(b, moves, alloc=alloc, stop=stop)
            NODES[0] = spent + zs.total
            for k2, v2 in zs.kstats.items():
                KSTATS[k2] = KSTATS.get(k2, 0) + v2
            for m in moves:
                v, zl, _ = got[m]
                values[m.uci()] = v; trace[m.uci()] = ((v, None, [x.uci() for x in zl[1:]]), before)
            moves = []
            top = max(values.values()); tied = [u for u, v in values.items() if v == top]
            if plan.prefer is not None and len(tied) > 1:                  # the planner's preference, only among the moves the tactics tie
                sc = {u: plan.prefer(b, chess.Move.from_uci(u), got[chess.Move.from_uci(u)][1]) for u in tied}
                hi = max(sc.values())
                for u in tied:
                    if sc[u] < hi:
                        values[u] -= 1e-3                                   # (a tie-break: below the top, above anything worse)
        for m in moves:
            v, r, ln = two_ply_value(b, m, "instinct" if LEDGER == "two_ply_instinct" else "all")
            values[m.uci()] = v; trace[m.uci()] = ((v, None, [r] + ln if r else ln), before)
        top = max(values.values()); best = [u for u, v in values.items() if v == top]
        plan_scores = {}
        if PLANS != "none" and len(best) > 1:
            plan_scores = _layer().score_moves(b, [chess.Move.from_uci(u) for u in best])
            hi = max(plan_scores.values())
            if hi > 0:
                best = [u for u in best if plan_scores.get(u, 0.0) == hi]
        return dict(values=values, best=best, trace=trace, before=before, plans=plan_scores)
    for m in b.legal_moves:
        if JOINT:
            nb = b.copy(stack=False); nb.push(m)
            after = (0.0, None, []) if nb.is_checkmate() else joint_threats(nb, square_frame, select=SELECT, parent=book, changed=changed_squares(b, m))   # the pass position's book: only the frames the move touches
            v = 1e6 if nb.is_checkmate() else captured_value(b, m) - after[0] + before[0]   # a mate dominates: parrying their mate threat is worth ~+1000 too (04QtS: Nxe4 +1001 beat Qf2#)
            bef = before
        else:
            v, after, bef = ledger_move_value(b, m, square_frame, before=before)
        values[m.uci()] = v; trace[m.uci()] = (after, bef)
    top = max(values.values())
    best = [u for u, v in values.items() if v == top]
    plan_scores = {}
    if PLANS != "none" and len(best) > 1:                           # the plan layer ranks the moves the frames cannot tell apart
        plan_scores = _layer().score_moves(b, [chess.Move.from_uci(u) for u in best])
        hi = max(plan_scores.values())
        if hi > 0:
            best = [u for u in best if plan_scores.get(u, 0.0) == hi]
    return dict(values=values, best=best, trace=trace, before=before, plans=plan_scores)


def trace(fen: str, detail: int = 3) -> None:
    """A SAMPLE ANALYSIS, in the order the frames are analysed (Kaveh 2026-09-28: "how many lemmas do we use? how are
    the lemmas used? ... see what it analyzes in what order"): the store and the lemma library start empty, so what is
    computed and what is reused is this analysis's own. First the opponent's frames if the side to move passed (their
    threats), then each legal move's frames after it, then the counts and the decision."""
    from . import fates
    fates._JOINT.clear(); fates._RESOLVED.clear(); fates._LEMMAS.clear()
    fates.TRACE = []
    b = chess.Board(fen)
    t0 = time.perf_counter()
    c = choose(b)
    dt = time.perf_counter() - t0
    events = fates.TRACE; fates.TRACE = None
    side = "White" if b.turn else "Black"; other = "Black" if b.turn else "White"

    def pieces(f, sqs):
        bb = chess.Board(f)
        return " ".join(f"{bb.piece_at(q).symbol()}{chess.square_name(q)}" for q in sqs if bb.piece_at(q) is not None)

    def san(f, line):
        bb = chess.Board(f); out = []
        for u in line:
            m = chess.Move.from_uci(u)
            if m not in bb.legal_moves:
                break
            out.append(bb.san(m)); bb.push(m)
        return " ".join(out) or "nothing (stand)"

    def name(e):
        k = e["key"]
        if k[0] == "king":
            return f"the {'white' if k[1] else 'black'} king frame"
        if k[0] == "promo":
            return f"the promotion frame of the pawn on {chess.square_name(k[1])}"
        bb = chess.Board(e["fen"]); pc = bb.piece_at(k[1])
        return f"the frame of {pc.symbol() if pc else ''}{chess.square_name(k[1])}"

    def full(e):
        joined = f"; king frame joined: {', '.join('white' if k else 'black' for k in e['kings'])}" if e["kings"] else ""
        skipped = f"; king frame NOT joined (prior): {', '.join('white' if k else 'black' for k in e['skipped'])}" if e["skipped"] else ""
        return (f"    {name(e)}: {len(e['squares'])} squares, pieces {pieces(e['fen'], e['members'])}{joined}{skipped}\n"
                f"      -> {e['value']:+g} by {san(e['fen'], e['line'])}   [{e['source']}]")

    stand = passed(b)
    print(f"SAMPLE ANALYSIS {fen} ({side} to move); {dt:.1f}s")
    print(f"1. If {side} passed, {other}'s frames (the threats it must answer):")
    sfen = stand.fen() if stand is not None else None
    for e in events:
        if e.get("fen") == sfen and e["event"] == "frame":
            print(full(e))
    print(f"   their best: {c['before'][0]:+g}")
    from .frame_goals import plans_for, goes_through, final_answers, verify, _to_move, _line_san
    t1 = time.perf_counter()
    theirs = plans_for(b, not b.turn)                                  # what-ifs a quiet move or two deep (frame_goals)
    print(f"1b. {other}'s what-ifs one or two quiet moves deep (each judged by the frames with {side} passing), "
          f"{len(theirs)} found in {time.perf_counter() - t1:.1f}s:")
    for q in theirs[:8]:
        ok, _ = goes_through(b, q)
        ans = [] if ok else final_answers(b, q)
        defused = [b.san(m) for m in b.legal_moves if verify(_after(b, m), q.side, q.target, q.moves) is None]
        tp = b.piece_at(q.target)
        print(f"    {q.san(b)}: wins the {chess.piece_name(tp.piece_type)} on {chess.square_name(q.target)} (+{q.gain:g}, then "
              f"{' '.join(_line_san(_to_move(b, q.side), q.moves, q.line))}) -- {'; '.join(o.text for o in q.ops)}")
        print(f"      {'goes through' if ok else 'answered at the last step by ' + ', '.join(a.san for a in ans[:8])}; "
              f"{side} moves that defuse it now: {', '.join(defused[:12]) or 'none'}{' ...' if len(defused) > 12 else ''}")
    from .ablation import describe as what_matters
    print("1c. What matters, by ablation (each piece removed in turn; Kaveh: 'if the removal of this piece makes a big difference'):")
    for line in what_matters(b):
        print("    " + line)
    order = sorted(c["values"], key=lambda u: -c["values"][u])
    shown = set(order[:detail])
    print(f"2. {side}'s moves in the order they are generated: value = captured - {other}'s best frame after it + {other}'s best if we passed")
    for m in b.legal_moves:
        nb = b.copy(stack=False); nb.push(m)
        es = [e for e in events if e.get("fen") == nb.fen()]
        fr = [e for e in es if e["event"] == "frame"]
        tag = "  <- top" if m.uci() in c["best"] else ""
        if m.uci() in shown:
            print(f"   {b.san(m)}: value {c['values'][m.uci()]:+g}{tag}")
            for e in fr:
                print(full(e))
            if any(e["event"] == "book from memory" for e in es):
                print("    (the whole position was met before: its book reused)")
        else:
            srcs = {}
            for e in fr:
                s = e["source"].split(",")[0]; srcs[s] = srcs.get(s, 0) + 1
            print(f"   {b.san(m)}: value {c['values'][m.uci()]:+g}; {len(fr)} frames ({', '.join(f'{n} {s}' for s, n in srcs.items()) or 'book from memory'}){tag}")
    fr = [e for e in events if e["event"] == "frame"]
    by = {}
    for e in fr:
        by[e["source"]] = by.get(e["source"], 0) + 1
    print("3. Counts: " + f"{len(fr)} frames analysed -- " + "; ".join(f"{n} {s}" for s, n in sorted(by.items(), key=lambda kv: -kv[1])))
    print(f"   lemmas formed in this analysis: {len(fates._LEMMAS)}; king frames not joined by the prior: {sum(len(e['skipped']) for e in fr)}; "
          f"whole positions met again (book reused): {sum(1 for e in events if e['event'] == 'book from memory')}")
    print(f"4. Decision: {', '.join(b.san(chess.Move.from_uci(u)) for u in c['best'])} ({c['values'][c['best'][0]]:+g})")


def _after(b: chess.Board, m: chess.Move) -> chess.Board:
    nb = b.copy(stack=False); nb.push(m); return nb


def _choose_fen(fen: str) -> dict:
    from .fates import clear_caches
    clear_caches()
    return choose(chess.Board(fen))


def explain(b: chess.Board, u: str, c: dict) -> str:
    m = chess.Move.from_uci(u); after, bef = c["trace"][u]
    nm = lambda t: (f"{t[0]:+.0f} on {chess.square_name(t[1])} by {' '.join(t[2])}" if t[1] is not None else "nothing")
    return f"{b.san(m)} {c['values'][u]:+.1f}: their best frame after it {nm(after)}; if we passed {nm(bef)}"


def main(argv=None):
    global JOINT, PLANS, LEDGER
    import sys
    sys.stdout.reconfigure(line_buffering=True)                       # every OBS line reaches the log as it prints (a crash loses none)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--per-frame", action="store_true", help="resolve each frame alone (no pulled-in pieces), for comparison")
    ap.add_argument("--show", type=int, default=3); ap.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 2) - 1))
    ap.add_argument("--plans", default=PLANS, help="the tactical plan layer that breaks ties in the top set (plan_layers.LAYERS)")
    ap.add_argument("--ledger", default="zone_candidates", choices=["one_ply", "two_ply", "two_ply_candidates", "two_ply_instinct", "zone_candidates"])   # the zone verifier: better on the 300 game positions (2026-09-29); LEDGER stays one-ply for library callers (a building block)
    ap.add_argument("--no-king-prior", action="store_true", help="join every king frame the counter-check rule names (no learned prior)")
    ap.add_argument("--trades", action="store_true", help="the planner's material-count trade preference (tie-break among tactical ties)")
    ap.add_argument("--trace", default=None, metavar="FEN", help="a sample analysis of one position, in the order the frames are analysed")
    a = ap.parse_args(argv)
    JOINT = not a.per_frame
    PLANS = a.plans
    LEDGER = a.ledger
    if a.trades:                                                      # the planner's material-count trade preference (a tie-break)
        from .frame_planner import MaterialPlanner
        global PLANNER
        PLANNER = MaterialPlanner(trades=True)
    if a.no_king_prior:
        from . import fates
        fates.KING_JOIN_GATE = None
    if a.trace:
        trace(a.trace, a.show)
        return
    t0 = time.perf_counter()
    eng = Engine(threads=8, depth=a.depth)
    try:
        _run(a, eng, t0)
    finally:
        eng.close()                                                   # an open engine keeps the interpreter from exiting after a crash


def _run(a, eng, t0):
    rows = []
    boards = positions(a.n, a.seed)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers) as pool:                  # the frames' choice per position over a pool; Stockfish (cached) in the parent
        chosen = pool.map(_choose_fen, [b.fen() for b in boards], chunksize=2)
    for i, (b, c) in enumerate(zip(boards, chosen)):
        lines = eng.analyse(b, multipv=b.legal_moves.count(), depth=a.depth)
        if not lines:
            continue
        cp = {l.move: l.cp for l in lines}; best_cp = max(cp.values()); sf_best = max(cp, key=cp.get)
        pick = c["best"][0]                                         # the first tied move in move order: the frames cannot tell tied moves apart
        tied_ok = sum(1 for u in c["best"] if cp.get(u, -1e9) > best_cp - LOSS_CP)
        false_safe = sum(1 for u in c["best"] if cp.get(u, -1e9) <= best_cp - LOSS_CP)   # a top move Stockfish calls a loss
        rows.append(dict(i=i, fen=b.fen(), pick=pick, loss=best_cp - cp.get(pick, -10000), sf_best=sf_best, sf_in_top=sf_best in c["best"],
                         n_top=len(c["best"]), tied_ok=tied_ok, false_safe=false_safe, c=c, cp=cp, b=b))
    n = len(rows)
    mistakes = [r for r in rows if r["loss"] >= LOSS_CP]
    from . import fates as _f
    print(f"[frame_player] {n} held-out positions, ledger {LEDGER}, plan layer {PLANS}, king-join prior {'on' if _f._king_gate() else 'off'} ({time.perf_counter() - t0:.0f}s)")
    print(f"OBS Stockfish's best move among the frames' top moves: {sum(r['sf_in_top'] for r in rows)}/{n}; the frames' top set is a single move in "
          f"{sum(r['n_top'] == 1 for r in rows)}/{n}")
    print(f"OBS the pick (the first top move) loses 1.5 pawns or more against the best move: {len(mistakes)}/{n}; "
          f"of those, every top move does: {sum(1 for r in mistakes if r['tied_ok'] == 0)}")
    print(f"OBS false-safe top moves (a top move losing 1.5 pawns or more): {sum(r['false_safe'] for r in rows)} in {sum(r['false_safe'] > 0 for r in rows)}/{n} positions; "
          f"own misses (every top move does): {sum(r['false_safe'] == r['n_top'] for r in rows)}")
    print(f"OBS the pick is Stockfish's best move: {sum(r['pick'] == r['sf_best'] for r in rows)}/{n}; the pick's mean loss against the best move "
          f"{sum(min(r['loss'], 1000) for r in rows) / max(1, n):.0f} cp (capped at 1000); positions where the plan layer credited a tied move: "
          f"{sum(1 for r in rows if r['c'].get('plans') and max(r['c']['plans'].values()) > 0)}")
    better = worse = same = 0; gaps = []                              # the plan layer's credit, checked against Stockfish on the tied moves
    for r in rows:
        ps = r["c"].get("plans") or {}
        if not ps or max(ps.values()) <= 0:
            continue
        hi = max(ps.values()); cred = [u for u in ps if ps[u] == hi]; rest = [u for u in ps if ps[u] != hi]
        if not rest:
            continue
        c_cp = sum(r["cp"].get(u, -10000) for u in cred) / len(cred); o_cp = sum(r["cp"].get(u, -10000) for u in rest) / len(rest)
        gaps.append(c_cp - o_cp); better += c_cp > o_cp + 20; worse += c_cp < o_cp - 20; same += abs(c_cp - o_cp) <= 20
    wrong = []
    for r in rows:
        ps = r["c"].get("plans") or {}
        if ps and max(ps.values()) > 0:
            hi = max(ps.values()); cred = [u for u in ps if ps[u] == hi]; rest = [u for u in ps if ps[u] != hi]
            if rest and max(r["cp"].get(u, -10000) for u in cred) < max(r["cp"].get(u, -10000) for u in rest) - 20:
                wrong.append((r, cred, rest))
    for r, cred, rest in wrong[:a.show]:                              # the plan layer's wrong credits, for the dig
        b = r["b"]; nm = lambda us: ", ".join(f"{b.san(chess.Move.from_uci(u))} ({r['cp'].get(u, 0):+d})" for u in us[:6])
        print(f"\nWRONG CREDIT {b.fen()}  credited {nm(cred)} ({r['c']['plans'][cred[0]]:+g}); the best other tied moves {nm(sorted(rest, key=lambda u: -r['cp'].get(u, -10000)))}")
        for line in _layer().explain(b)[:8]:
            print("   ", line)
    if PLANS != "none":
        print(f"OBS the plan layer credited some tied moves over others in {len(gaps)}/{n} positions; Stockfish rates the credited moves above the "
              f"other tied moves by more than 0.2 in {better}, below in {worse}, within 0.2 in {same}; mean gap {sum(gaps) / max(1, len(gaps)):+.0f} cp")
    worst = sorted(mistakes, key=lambda r: -r["loss"])
    json.dump([dict(fen=r["fen"], pick=r["pick"], loss=r["loss"], sf_best=r["sf_best"], n_top=r["n_top"], tied_ok=r["tied_ok"], plans=r["c"].get("plans") or {},
                    cp={u: r["cp"].get(u) for u in (r["c"].get("plans") or {})}) for r in rows],
              open(paths.derived(f"frame_player_positions_{PLANS}.json"), "w"), indent=1)
    shown = 0
    for r in worst:
        if r["tied_ok"] or shown >= a.show:
            continue                                                # a miss the frames own: no tied move was fine
        shown += 1; b = r["b"]; c = r["c"]
        sf_line = eng.analyse(b, root_moves=[chess.Move.from_uci(r["sf_best"])], depth=a.depth)[0]
        my_line = eng.analyse(b, root_moves=[chess.Move.from_uci(r["pick"])], depth=a.depth)[0]
        print(f"\nMISS {b.fen()}  {'white' if b.turn else 'black'} to move")
        print(b.unicode(borders=False, empty_square='·'))
        print(f"  the frames chose {explain(b, r['pick'], c)}   (Stockfish {r['cp'][r['pick']]:+d})")
        print(f"  Stockfish's best {explain(b, r['sf_best'], c)}   (Stockfish {r['cp'][r['sf_best']]:+d})")
        print(f"  Stockfish's line after the frames' move: {my_line.san_line(b, 8)}")
        print(f"  Stockfish's line after its best move:   {sf_line.san_line(b, 8)}")


if __name__ == "__main__":
    main()
