"""THE REASONING LEDGER, from Python (Kaveh 2026-10-14; docs/ANALYST_PLAN.md section 39; the engine is rust/analyst_core/src/reasoning.rs).

  python -m catspace.incubator.analyst.discovery.ledger --fen "<fen>" [--line "Qc7 Rbd1 Rc6 h5 exd4"]

Prints the because-chain (the dependency path from the decision down to the facts), the counts by status, the questions asked against the
budget, the conjectures in the back pocket (CONDITIONAL findings and their premises), and with --line the roll forward: per ply how many findings
were retracted, carried over and re-asked, and what became of the pin on the e-pawn. `--mirror` mirrors the board first (the plan's wording; the
ledger reads either side, so it is not needed).
"""
from __future__ import annotations

import argparse
import json
import time

import chess

KAVEH = "5k1r/1p1n1p2/p1qb1p1p/4p3/2rP2PP/2PQ1PB1/PP2NK2/1R5R b - - 0 1"


def _a():
    import catspace.incubator.analyst.discovery  # noqa: F401  (installs the chess terms: job, pinned ...)
    import analyst_core as A
    return A


def load_frame(L, path: str | None = None) -> list:
    """Load the learned threat frame (watch.threat_frame -> data/derived/threat_frame.json) into a ledger."""
    import json
    from catspace.io import paths
    fr = json.load(open(path or paths.derived("threat_frame.json")))
    terms = [(h["term"], int(h["sign"]), h["family"] == "delta", h["gloss"]) for h in fr["terms"]]
    L.set_frame(terms)
    return terms


def load_defence_frame(L, path: str | None = None) -> list:
    """Load the learned DEFENCE frame (watch.defence_frame -> data/derived/defence_frame.json) into a ledger: S's moves ordered by it answer a threat."""
    import json
    from catspace.io import paths
    fr = json.load(open(path or paths.derived("defence_frame.json")))
    terms = [(h["term"], int(h["sign"]), h["family"] == "delta") for h in fr["terms"]]
    L.set_defence_frame(terms)
    return terms


def load_transitions(L, path: str | None = None) -> bool:
    """Load the transition table (data/derived/transitions.json, ANALYST_PLAN 33) into a ledger: the stake of a trade is read from it. False when absent."""
    from catspace.io import paths
    p = path or paths.derived("transitions.json")
    try:
        L.set_transitions(open(p).read())
        return True
    except OSError:
        return False


def opening_bonus(fen: str) -> dict:
    """{uci: score} of the opening tie-break principles (castle > develop > develop toward the centre > a centre pawn > a steady move; opening_rule.py), empty outside the opening."""
    from .opening_rule import opening_claims
    return {u: sc for u, (sc, _) in opening_claims(chess.Board(fen)).items()}


def decide(fen: str, questions: int = 120, max_candidates: int = 16) -> dict:
    """The decision read off the chain (ANALYST_PLAN 40.4): constraints, opportunities, candidates each checked by running the loop after it; the opening default when
    nothing is threatened and nothing is to be won."""
    L = new(fen, 5000)
    load_transitions(L)
    return L.decide_chain(opening_bonus(fen), questions, max_candidates)


def analyse(fen: str, budget: int = 200) -> dict:
    return _a().ledger(fen, budget)


def new(fen: str, budget: int = 200):
    return _a().Ledger(fen, budget)


def by_id(report: dict) -> dict:
    return {f["id"]: f for f in report["findings"]}


def chain(report: dict, root: int, depth: int = 0, seen: set | None = None, quiet_zero: bool = True) -> list:
    """The because-chain from a finding down: its claim, its status and interval, then what it cites (a finding cited twice is shown once; a
    job that is worth nothing is not a reason, only counted)."""
    fs = by_id(report)
    seen = seen if seen is not None else set()
    f = fs[root]
    pad = "  " * depth
    out = [f"{pad}[{f['status']}] {f['claim']}  value [{f['lo']}, {f['hi']}]  ({f['type']}#{f['id']})"]
    if root in seen:
        out[0] += "  (above)"
        return out
    seen.add(root)
    zero = 0
    for c in f["cites"]:
        g = fs.get(c)
        if g is None:
            continue
        if quiet_zero and g["type"] in ("JOB", "RECAP") and g["lo"] == 0:
            zero += 1
            continue
        out += chain(report, c, depth + 1, seen, quiet_zero)
    if zero:
        out.append(f"{pad}  ({zero} other guard jobs asked, worth nothing)")
    return out


def pins(report: dict, square: str | None = None) -> list:
    return [f for f in report["findings"] if f["type"] == "PIN" and (square is None or f["key"].split("|")[2].endswith(square))]


def show(report: dict, root: int | None = None) -> None:
    t = report["questions_asked"]
    print(f"OBS ledger: {t} questions asked of a budget of {report['budget']} ({len(report['skipped'])} skipped), {report['facts_evaluated']} facts evaluated, "
          f"{report['solver_calls']} solver call(s)")
    print(f"OBS ledger: live findings by status {report['counts']}")
    top = sorted(pins(report), key=lambda f: -f["lo"])
    if root is None and top:
        root = top[0]["id"]
    if root is not None:
        print("OBS the because-chain:")
        for line in chain(report, root):
            print("OBS   " + line)
    for c in report["confirmations"]:
        print(f"OBS solver on finding #{c['finding']}: " + ("CONFIRMS" if c["confirmed"] else "does NOT confirm") +
              f" -- their gain of >= {c['value']} stays forced after " + "; ".join(f"{m['move']} ({m['status']}: {m['line']})" for m in c["moves"]))
    print(f"OBS the agenda (by stake), first ten: {report['agenda'][:10]}")
    for f in report["back_pocket"]:
        print(f"OBS back pocket: {f['claim']}  [premise: {f['premise']}]  value {f['value']}")


def roll(fen: str, san_line: list, budget: int = 200, square: str = "e5", verbose: bool = True) -> list:
    """Roll the ledger through a line (SAN, from `fen`), return the per-ply records; prints the pin on `square` at each ply."""
    L = new(fen, budget)
    rep = L.run()
    b = chess.Board(fen)
    recs = [dict(ply=0, move=None, retracted=0, lookups=0, re_asked=rep["questions_asked"], pins=[dict(claim=f["claim"], lo=f["lo"], hi=f["hi"]) for f in pins(rep, square)])]
    if verbose:
        print(f"OBS ply 0: {rep['questions_asked']} questions asked; pin on {square}: " + "; ".join(f"{p['claim']} [{p['lo']}, {p['hi']}]" for p in recs[0]["pins"]))
    for san in san_line:
        m = b.parse_san(san)
        mover = chess.COLOR_NAMES[b.turn]
        b.push(m)
        r = L.push(m.uci())
        rep = L.report_dict()
        ps = [f for f in pins(rep, square)]
        r["pin_on_square"] = [dict(claim=f["claim"], lo=f["lo"], hi=f["hi"], status=f["status"]) for f in ps]
        recs.append(r)
        if verbose:
            before = sum(1 for f in rep["findings"] if f["ply"] < r["ply"])
            print(f"OBS ply {r['ply']} {mover} {san}: retracted {r['retracted']} (touched {','.join(r['touched'])}), carried over {before} findings "
                  f"(looked up {r['lookups']} times), re-asked {r['re_asked']}, premise listeners woken {r['premise_listeners_woken']}")
            for p, f in zip(r["pin_on_square"], ps):
                print(f"OBS    pin on {square}: [{p['status']}] {p['claim']}  [{p['lo']}, {p['hi']}]")
                for mv in json.loads(f["note"].split(" | ")[0] or "[]"):
                    print(f"OBS      {mv['move']}: gains {mv['gain']}, check {mv['check']}, costs {mv['cost']}")
            if not r["pin_on_square"]:
                print(f"OBS    no pin finding on {square} now")
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fen", default=KAVEH)
    ap.add_argument("--line", default=None, help='SAN moves from the position, e.g. "Qc7 Rbd1 Rc6 h5 exd4"')
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--square", default="e5", help="the piece whose pin is followed along the line")
    ap.add_argument("--trace", action="store_true", help="print the reasoning on this position step by step, in plain chess words")
    ap.add_argument("--threats", choices=("defined", "rules"), default="defined", help="what a threat is: the DEFINITION sized by the ledger (material lost with the best defence, or mate), or the older rules")
    ap.add_argument("--answers", choices=("frame", "hand"), default="frame", help="the answers to a threat: S's moves ordered by the learned defence frame, or the hand-written ones (comparison only)")
    ap.add_argument("--depth", type=int, default=4, help="plies of the line (trace)")
    ap.add_argument("--loop", choices=("event", "script"), default="event", help="trace: the event loop (ANALYST_PLAN 40, the default: the chain is printed first) or the older fixed script (comparison only)")
    ap.add_argument("--mirror", action="store_true", help="mirror the board first (the ledger reads either side; not needed)")
    a = ap.parse_args(argv)
    fen = chess.Board(a.fen).mirror().fen() if a.mirror else a.fen
    if a.trace:
        t0 = time.time()
        L = new(fen, max(a.budget, 1000))
        if a.answers == "frame":
            load_defence_frame(L)
        if a.loop == "event":
            load_transitions(L)
        r = L.trace(a.depth, a.threats, a.answers, a.loop == "event", opening_bonus(fen) if a.loop == "event" else None)
        for line in r["trace"]:
            print(line)
        print(f"OBS trace ({'learned' if r['learned'] else 'rules'} threats): {r['positions']} positions examined by the ledger, {r['looks']} static looks by the threat frame, {r['questions']} questions, {r['facts']} facts, {round(time.time() - t0, 2)} s; line "
              + " ".join(x.replace('|', ' [') + ']' for x in r["line"]))
        return
    t0 = time.time()
    rep = analyse(fen, a.budget)
    print(f"OBS ledger on {fen}: {round(time.time() - t0, 2)} s")
    show(rep)
    if a.line:
        t0 = time.time()
        roll(fen, a.line.split(), a.budget, a.square)
        print(f"OBS roll forward: {round(time.time() - t0, 2)} s")


def chain_main(argv=None):
    """python -m catspace.incubator.analyst.discovery.ledger chain "<FEN>" [--tag name] -- the event loop's chain on a position (ANALYST_PLAN 40), printed and
    saved to data/derived/chain_<tag>.txt."""
    from catspace.io import paths
    ap = argparse.ArgumentParser(prog="ledger chain")
    ap.add_argument("fen")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--questions", type=int, default=120, help="the decision's budget of questions")
    a = ap.parse_args(argv)
    t0 = time.time()
    L = new(a.fen, 5000)
    have = load_transitions(L)
    r = L.chain(a.questions)
    lines = [f"POSITION {a.fen}"] + ([] if have else ["(no transition table at data/derived/transitions.json: the stake of a trade is unknown)"]) + r["lines"]
    lines.append(f"OBS chain: {r['questions']} questions asked, {r['facts']} facts evaluated, {len(r['agenda']['events'])} events, {round(time.time() - t0, 2)} s")
    print("\n".join(lines))
    tag = a.tag or "".join(ch if ch.isalnum() else "_" for ch in a.fen.split()[0])[:24]
    out = paths.derived(f"chain_{tag}.txt")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"saved {out}")
    return r


# ---------------------------------------------------------------------------------------------------------------------------------------------
# A GAME WALKED THROUGH (Kaveh 2026-10-14: "why do we lose track of a thread after we've seen it? We should keep it going forward."): ONE ledger rolled forward with push
# through a real game. At each settled material change -- a position whose next move is not a capture or promotion, compared with the settled position before the capture
# chain -- A is the position before the loser's last move before the chain and B the position after it.
#   A  the threats against the loser if it had passed, the threads carried, the defence frame's choice there and whether it would have avoided the loss (sized by the
#      definition on a fresh ledger), and Stockfish (depth 14, the checker only) on the move played and on the frame's.
#   B  what the other side can win (the sizer against the side that just moved), the threads, whether the actual first capture was PREDICTED and at what size.
#   python -m catspace.incubator.analyst.discovery.ledger walk --pgn-index 2979 --events 7

LICHESS_PGN = "assets/corpora/raw/lichess_db_standard_rated_2019-01.pgn.zst"
PVAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def find_game(index: int, path: str = LICHESS_PGN):
    """The index-th game (0-based) of the file that passes the filter: Rated, not Bullet, both players >= 1600, decisive, >= 50 plies."""
    import io
    import re
    import subprocess
    import chess.pgn
    p = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    n, buf = 0, []

    def games():
        nonlocal buf
        for line in p.stdout:
            if line.startswith("[Event ") and buf:
                yield "".join(buf); buf = []
            buf.append(line)
        if buf:
            yield "".join(buf)
    try:
        for text in games():
            h = dict(re.findall(r'\[(\w+) "([^"]*)"\]', text))
            ev = h.get("Event", "")
            if "Rated" not in ev or "Bullet" in ev or h.get("Result") not in ("1-0", "0-1"):
                continue
            try:
                if int(h.get("WhiteElo", 0)) < 1600 or int(h.get("BlackElo", 0)) < 1600:
                    continue
            except ValueError:
                continue
            g = chess.pgn.read_game(io.StringIO(text))
            if sum(1 for _ in g.mainline_moves()) < 50:
                continue
            if n == index:
                return g, h.get("Site", "")
            n += 1
    finally:
        p.kill()
    raise SystemExit(f"no game #{index}")


def _bal(b: chess.Board) -> int:
    return sum(PVAL.get(p.piece_type, 0) * (1 if p.color == chess.WHITE else -1) for p in b.piece_map().values())


def material_events(game) -> tuple:
    """-> (the moves, the boards before each move, the events): a settled material change, the capture chain, the loser and the index of its last move before the chain."""
    b = game.board()
    moves, boards = [], []
    for m in game.mainline_moves():
        boards.append(b.copy(stack=False)); moves.append(m); b.push(m)
    boards.append(b.copy(stack=False))
    cap = [boards[i].is_capture(m) or m.promotion is not None for i, m in enumerate(moves)]
    events, i = [], 0
    while i < len(moves):
        if not cap[i]:
            i += 1; continue
        c0 = i
        while i < len(moves) and cap[i]:
            i += 1
        c1 = i - 1                                           # the chain is moves c0 .. c1; the settled positions are c0 - 1 and c1 + 1
        before, after = _bal(boards[c0]), _bal(boards[c1 + 1])
        if after == before:
            continue
        loser = chess.BLACK if after > before else chess.WHITE
        q = next((j for j in range(c0 - 1, -1, -1) if boards[j].turn == loser), None)
        win_first = next((j for j in range(c0, c1 + 1) if boards[j].turn != loser), None)
        if q is None or win_first is None:
            continue
        events.append(dict(c0=c0, c1=c1, loser=loser, q=q, win_first=win_first, delta=abs(after - before)))
    return moves, boards, events


def walk_main(argv=None):
    import json
    import chess.engine
    from catspace.io import paths
    ap = argparse.ArgumentParser(prog="ledger walk")
    ap.add_argument("--pgn-index", type=int, required=True)
    ap.add_argument("--events", type=int, default=None)
    ap.add_argument("--depth", type=int, default=14, help="Stockfish depth, the checker only")
    a = ap.parse_args(argv)
    t0 = time.time()
    game, site = find_game(a.pgn_index)
    sid = site.rstrip("/").split("/")[-1]
    moves, boards, events = material_events(game)
    events = events[:a.events] if a.events else events
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    eng.configure({"Threads": 4})
    from . import watch as W
    out = dict(site=site, index=a.pgn_index, plies=len(moves), white=game.headers.get("White"), black=game.headers.get("Black"), events=[])
    print(f"OBS WALK {site}: {len(moves)} plies, {len(events)} settled material changes (white {game.headers.get('White')} {game.headers.get('WhiteElo')}, black {game.headers.get('Black')} {game.headers.get('BlackElo')})")
    L = new(boards[0].fen(), 1000)
    load_defence_frame(L)
    L.run()
    by_q = {e["q"]: e for e in events}
    for i, m in enumerate(moves):
        e = by_q.get(i)
        if e:
            A_fen = boards[i].fen()
            who = chess.COLOR_NAMES[e["loser"]]
            actual = boards[i].san(m)
            repA = L.threat_report(True)
            ch = L.defence_choice()
            frame_uci = ch[0]["uci"] if ch else None
            frame_san = ch[0]["san"] if ch else None
            e["A"] = dict(fen=A_fen, played=actual, threats=[(t["san"], t["size"], t["detail"][:140], t["thread"]) for t in repA["threats"][:3]], frame=frame_san, frame_top5=[c["san"] for c in ch[:5]],
                          positions=repA["positions"], questions=repA["questions"])
            if frame_uci:
                if frame_uci == m.uci():
                    e["A"]["frame_size_after"] = "the move played"
                else:
                    fb = boards[i].copy(stack=False); fb.push_uci(frame_uci)
                    L2 = new(fb.fen(), 1000)
                    r2 = L2.threat_report(True, True)
                    e["A"]["frame_size_after"] = r2["threats"][0]["size"] if r2["threats"] else 0
            e["A"]["sf_played_loss"] = W.sf_move_loss(A_fen, m.uci(), eng, a.depth)
            e["A"]["sf_frame_loss"] = W.sf_move_loss(A_fen, frame_uci, eng, a.depth) if frame_uci and frame_uci != m.uci() else e["A"]["sf_played_loss"]
        L.push(m.uci())
        if e:
            repB = L.threat_report(True, True)
            actual_first = moves[e["win_first"]].uci()
            hit = next((t for t in repB["threats"] if t["uci"] == actual_first and t["size"] >= 1), None)
            tops = [(t["san"], t["size"], t["detail"][:140], t["thread"]) for t in repB["threats"][:3]]
            bB = boards[i + 1]
            sf = W.sf_capture_size(bB.fen(), actual_first, e["loser"], eng, a.depth) if boards[e["win_first"]].turn != e["loser"] and e["win_first"] == i + 1 else None
            # (when the winner's first capture is not the next move, the teacher is asked on the position where it is played)
            if sf is None:
                sf = W.sf_capture_size(boards[e["win_first"]].fen(), actual_first, e["loser"], eng, a.depth)
            e["B"] = dict(fen=bB.fen(), actual=boards[e["win_first"]].san(moves[e["win_first"]]), actual_loss=e["delta"], predicted=bool(hit), predicted_size=hit["size"] if hit else 0,
                          threats=tops, sf_size=sf, positions=repB["positions"], questions=repB["questions"])
            out["events"].append({k: v for k, v in e.items()})
    eng.quit()
    for n, e in enumerate(out["events"], 1):
        who = chess.COLOR_NAMES[e["loser"]]
        A, B = e["A"], e["B"]
        print(f"OBS EVENT {n}: {who} loses {e['delta']} (moves {e['c0'] // 2 + 1}{'...' if e['c0'] % 2 else '.'} onward); the loser's last move {A['played']} at ply {e['q'] + 1}")
        print(f"   A (before {A['played']}), {who} to move: threats if it passed: " + ("; ".join(f"{s} ({z}) [{d}]" + (f" THREAD {th}" if th else "") for s, z, d, th in A["threats"]) or "none")
              + f". The defence frame's choice: {A['frame']} (top five {A['frame_top5']}); after it the biggest threat against {who}: {A['frame_size_after']}. Stockfish d{a.depth}: the move played loses {A['sf_played_loss']}, the frame's {A['sf_frame_loss']}.")
        print(f"   B (after {A['played']}): what the other side can win: " + ("; ".join(f"{s} ({z}) [{d}]" + (f" THREAD {th}" if th else "") for s, z, d, th in B["threats"]) or "nothing")
              + f". Actual: {B['actual']} for {B['actual_loss']} (Stockfish sizes it {B['sf_size']}) -> PREDICTED: {'YES at size ' + str(B['predicted_size']) if B['predicted'] else 'NO'}")
    n_yes = sum(e["B"]["predicted"] for e in out["events"])
    print(f"OBS WALK done: {n_yes} of {len(out['events'])} events predicted at B; {round(time.time() - t0, 1)} s")
    out["predicted"] = n_yes
    json.dump(out, open(paths.derived(f"walk_{sid}.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "walk":
        walk_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "chain":
        chain_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "decide":                  # decide "<FEN>": the decision read off the chain (ANALYST_PLAN 40.4)
        print("\n".join(decide(sys.argv[2])["lines"]))
    else:
        main()
