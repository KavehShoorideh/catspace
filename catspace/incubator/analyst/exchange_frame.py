"""THE EXCHANGE AS A SEARCH INSIDE ITS FRAME (Kaveh 2026-10-08: "an alpha-beta search in the frame of the exchange ...
there are some orders of capture ... other exits that can be taken if I take the wrong route ... use lemmas to speed up
our search ... plans and sub goals to guide our search in the frame").

THE FRAME on square s: the participants of both sides, each a RAY -- its value, its square, and the pieces standing
behind it on its line to s (nearest first, either colour), which join their owner's capturers when it leaves (the x-ray
rule applied recursively: whoever stands behind, as long as it can attack along that line). THE MOVES at each turn:
capture on s with ANY available participant, or EXIT (stop and bank the material). THE SEARCH: alpha-beta over those
moves only -- never the board -- with the side to move's net material from here as the value; the plain count's line
(cheapest first) is searched first, so a frame where the plain count is right costs one line plus the cut-offs.

The value at a node is the classical negamax of the capture game: v = max(0 [exit], max over capturers a of
(on_s - v(child))). At the ROOT the first capture is forced (the value of capturing, possibly negative), and the best
capturer and every capture order reaching the best material are returned for the plan layer to choose among.

Availability is the plain one for now (every participant may capture; the king at 100 never captures into a defended
square because the exit cuts it); the rule lemmas (king last, the pin, ...) narrow it in later steps.

    python -m catspace.incubator.analyst.exchange_frame census --tag c300 --limit 3000
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)


@dataclass(frozen=True)
class Ray:
    """A participant: its value, square and colour, and the pieces behind it on its line to s (nearest first)."""
    value: int
    square: int
    white: bool
    behind: tuple = ()             # ((value, square, white), ...)


def _geometry(kind: int, q: int, s: int) -> bool:
    if not chess.BB_RAYS[q][s]:
        return False
    straight = chess.square_file(q) == chess.square_file(s) or chess.square_rank(q) == chess.square_rank(s)
    return kind == chess.QUEEN or (kind == chess.ROOK and straight) or (kind == chess.BISHOP and not straight)


def _behind(b: chess.Board, q: int, s: int) -> tuple:
    """The pieces behind q on the line from s through q, nearest first, while each can attack s along that line."""
    if not chess.BB_RAYS[q][s] or chess.square_distance(q, s) == 0:
        return ()
    df = (chess.square_file(q) > chess.square_file(s)) - (chess.square_file(q) < chess.square_file(s))
    dr = (chess.square_rank(q) > chess.square_rank(s)) - (chess.square_rank(q) < chess.square_rank(s))
    out = []
    f, r = chess.square_file(q) + df, chess.square_rank(q) + dr
    while 0 <= f < 8 and 0 <= r < 8:
        t = chess.square(f, r); p = b.piece_at(t)
        if p is not None:
            if p.piece_type in SLIDERS and _geometry(p.piece_type, t, s):
                out.append((VAL[p.piece_type], t, p.color == chess.WHITE))
            else:
                break
        f += df; r += dr
    return tuple(out)


def available(b: chess.Board, side: chess.Color, q: int, s: int) -> bool:
    """THE PIN, from king safety (a legal move never leaves one's own king attacked; chess-lean `legalMove_kingSafe`): a
    piece pinned to its king may capture on s only along its pin line. The lemma is read once per frame from the board
    (python-chess's pin mask), never by trying the moves."""
    return not b.is_pinned(side, q) or bool(b.pin(side, q) & chess.BB_SQUARES[s])


def frame_of(b: chess.Board, s: int, pins: bool = True) -> tuple:
    """-> (on_s, mover's rays, opponent's rays) for the side to move capturing on s; rays cheapest first; with `pins`,
    a pinned piece that cannot capture along its pin line is not a participant."""
    c = b.turn
    rays = lambda side: sorted((Ray(VAL[b.piece_at(q).piece_type], q, side == chess.WHITE, _behind(b, q, s)) for q in b.attackers(side, s)
                                if not pins or available(b, side, q, s)), key=lambda r: (r.value, r.square))
    return VAL[b.piece_at(s).piece_type], tuple(rays(c)), tuple(rays(not c))


def _play(ray: Ray, mine: tuple, theirs: tuple) -> tuple:
    """`ray` (of the mover) captures: it leaves `mine`; the first piece behind it joins its owner's rays, carrying the
    rest of the line behind it. -> (mine', theirs') from the mover's point of view, before the turn passes."""
    mine = tuple(r for r in mine if r.square != ray.square)
    if ray.behind:
        v, q, white = ray.behind[0]
        joined = Ray(v, q, white, ray.behind[1:])
        if white == ray.white:
            mine = tuple(sorted(mine + (joined,), key=lambda r: (r.value, r.square)))
        else:
            theirs = tuple(sorted(theirs + (joined,), key=lambda r: (r.value, r.square)))
    return mine, theirs


class ExchangeFrame:
    """The exchange as a Frame (frame_search.py): a state is (on_s, the mover's rays, the opponent's rays); a move is a
    ray of the mover capturing on s; its gain is on_s; the exit is worth 0 (stop and bank)."""

    def moves(self, st):
        return st[1]                                               # cheapest first: the plain count's line first

    def gain(self, st, ray):
        return st[0]

    def next(self, st, ray):
        m2, t2 = _play(ray, st[1], st[2])
        return (ray.value, t2, m2)

    def exit(self, st):
        return 0

    def key(self, st):
        return st


class LinkedExchangeFrame(ExchangeFrame):
    """THE TWO-SIDED INTERMEZZO LEMMA (mined 2026-10-08 from the replay: captures elsewhere carry 86% of the exchange
    frame's error; the one-sided subtraction was a bound, not a prediction): the exchange frame on s, plus, for each side,
    ONE move elsewhere worth its best static exchange on another square (a one-shot frame, independent of s), playable
    once at any of its turns. State: (on_s, mine, theirs, my elsewhere gain or 0 if spent, theirs). The exit stays."""

    def moves(self, st):
        out = list(st[1])
        if st[3] > 0:
            out.append("elsewhere")
        return out

    def gain(self, st, m):
        return st[3] if m == "elsewhere" else st[0]

    def next(self, st, m):
        on_s, mine, theirs, e_me, e_them = st
        if m == "elsewhere":
            return (on_s, theirs, mine, e_them, 0)              # s stands as it is; my one-shot spent
        m2, t2 = _play(m, mine, theirs)
        return (m.value, t2, m2, e_them, e_me)


def elsewhere_gain(b: chess.Board, side: chess.Color, skip: int, busy=frozenset()) -> int:
    """`side`'s best static exchange on any square but `skip` (0 when none pays), read on `b` with `side` to move, by
    pieces NOT busy in the exchange on `skip` (the piece standing there and the participants: independence -- bug
    2026-10-08: the capturer on s was counted as also capturing elsewhere). A busy piece is lifted off the board for the
    count (its lines open: an approximation, measured)."""
    from .exchange_law import see
    bb = b.copy(stack=False); bb.turn = side; bb.ep_square = None
    for q in busy | {skip}:
        p = bb.piece_at(q)
        if p is not None and p.color == side and p.piece_type != chess.KING:
            bb.remove_piece_at(q)
    best = 0
    for q, pc in bb.piece_map().items():
        if q != skip and pc.color != side and pc.piece_type != chess.KING and bb.attackers(side, q):
            best = max(best, see(bb, q)[0])
    return best


def linked_value(b: chess.Board, s: int) -> int:
    """The value of capturing on s for the side to move, with both sides' best capture elsewhere as one-shot moves: the
    opponent's read after our capture (the capturer left its post), ours before it."""
    from .frame_search import FrameSearch
    on_s, mine, theirs = frame_of(b, s)
    if not mine:
        return 0
    fs = FrameSearch(LinkedExchangeFrame(), [])
    best = None
    for ray in mine:
        m = next((x for x in b.legal_moves if x.from_square == ray.square and x.to_square == s), None)
        if m is None:
            continue
        b1 = b.copy(stack=False); b1.push(m)
        busy = frozenset(r.square for r in mine + theirs)
        e_them = elsewhere_gain(b1, b1.turn, s, busy)
        e_me = elsewhere_gain(b1, not b1.turn, s, busy)
        m2, t2 = _play(ray, mine, theirs)
        v = on_s - fs.value((ray.value, t2, m2, e_them, e_me))
        best = v if best is None else max(best, v)
    return best if best is not None else 0


def _exits_bound(st):
    """Both sides can always exit: the mover's value is at least 0 (its own stop) and at most on_s (after its capture
    the opponent can stop). Lean: valueF_nonneg, valueF_le_onS."""
    return (0, max(0, st[0]))


def _capture_bound(st, ray):
    """Capturing with a piece worth `a` is worth between on_s - a (the opponent recaptures, we stop) and on_s (the
    opponent stops) -- Kaveh's queen-takes-pawn: at least down nine, up one."""
    return (st[0] - ray.value if st[2] else st[0], st[0])


def exchange_lemmas():
    from .frame_search import Lemma
    return [Lemma("both sides can exit", "proved: Exchange.valueF_nonneg, Exchange.valueF_le_onS", state=_exits_bound),
            Lemma("capture bound", "proved: Exchange.valueF_nonneg, Exchange.valueF_le_onS", move=_capture_bound)]


def Search(lemmas: bool = True):
    from .frame_search import FrameSearch
    return FrameSearch(ExchangeFrame(), exchange_lemmas() if lemmas else [])


def brute(on_s: int, mine: tuple, theirs: tuple) -> int:
    """The frame's value with no pruning and no lemma at all (the reference)."""
    best = 0
    for ray in mine:
        m2, t2 = _play(ray, mine, theirs)
        best = max(best, on_s - brute(ray.value, t2, m2))
    return best


def orders(on_s: int, mine: tuple, theirs: tuple) -> list:
    """Every principal line as squares (with 'exit' where a side stops): the capture orders that reach the frame's
    value, for the plan layer to choose among (the pawn-vs-knight recapture)."""
    fs = Search(True)
    return [tuple(m if m == "exit" else chess.square_name(m.square) for m in line) for line in fs.lines((on_s, mine, theirs))]


def root_values(fs, on_s, mine, theirs) -> dict:
    """{square of the first capturer: value of capturing with it}."""
    return {ray.square: v for ray, v in fs.root((on_s, mine, theirs)).items()}


def _row(fen: str):
    from .exchange_law import see, see_x
    b = chess.Board(fen); c = b.turn; out = []
    for s, pc in b.piece_map().items():
        if pc.color == c or pc.piece_type == chess.KING or not b.attackers(c, s):
            continue
        on_s, mine, theirs = frame_of(b, s)
        plain, lem = Search(False), Search(True)
        rp, rl = root_values(plain, on_s, mine, theirs), root_values(lem, on_s, mine, theirs)
        bf = {ray.square: on_s - brute(ray.value, *reversed(_play(ray, mine, theirs))) for ray in mine}
        # truth maintenance: retract both lemmas (only the records resting on them re-derive), then reinstate (nothing)
        n0 = lem.nodes; lem_fresh = lem.fresh; affected = set()
        for name in ("both sides can exit", "capture bound"):
            affected |= set(lem.retract(name))
        rr = root_values(lem, on_s, mine, theirs); n_retract = lem.nodes - n0
        lem.reinstate("both sides can exit"); lem.reinstate("capture bound")
        n1 = lem.nodes; rb = root_values(lem, on_s, mine, theirs); n_reinstate = lem.nodes - n1
        # budget and resume: one node, then the rest; the total work against one full run
        bud = Search(True); bud.budget = 1
        r1 = root_values(bud, on_s, mine, theirs); unfinished = not bud.finished()
        bud.resume(None); r2 = root_values(bud, on_s, mine, theirs)
        out.append(dict(n=len(mine) + len(theirs) + sum(len(x.behind) for x in mine + theirs), nodes=plain.nodes, lnodes=n0,
                        cuts=dict(lem.cuts), agree=rp == bf and rl == bf, best=max(rl.values()), cheapest=rl[mine[0].square], see_x=see_x(b, s)[0],
                        retract_ok=rr == bf and rb == bf, n_retract=n_retract, n_reinstate=n_reinstate, affected=len(affected),
                        resume_ok=r2 == bf and bud.finished(), unfinished=unfinished, resumed_nodes=bud.nodes, resumed_fresh=bud.fresh, full_fresh=lem_fresh))
    return out


def census(tag: str, limit: int = 3000) -> None:
    """Every attacked enemy piece of the census: the frame search's root values equal the unpruned search's (must be
    always); its cheapest-first line equals the x-ray static count (must be always); how often a capture order other
    than cheapest-first is better; nodes visited by frame size."""
    import os
    from multiprocessing import Pool

    from .tactic_census_measure import load
    rows = load(tag)[:limit]
    frames = agree = see_x_eq = better = 0; nodes = Counter(); lnodes = Counter(); count = Counter(); cuts = Counter()
    tm = Counter()
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for recs in pool.imap_unordered(_row, [r["fen"] for r in rows], chunksize=8):
            for rec in recs:
                frames += 1; agree += rec["agree"]; see_x_eq += rec["cheapest"] == rec["see_x"]; better += rec["best"] > rec["cheapest"]
                nodes[rec["n"]] += rec["nodes"]; lnodes[rec["n"]] += rec["lnodes"]; count[rec["n"]] += 1; cuts.update(rec["cuts"])
                tm["retract_ok"] += rec["retract_ok"]; tm["n_retract"] += rec["n_retract"]; tm["n_reinstate"] += rec["n_reinstate"]
                tm["affected"] += rec["affected"] > 0; tm["resume_ok"] += rec["resume_ok"]; tm["unfinished"] += rec["unfinished"]
                tm["resumed_nodes"] += rec["resumed_nodes"]; tm["resumed_fresh"] += rec["resumed_fresh"]; tm["full_fresh"] += rec["full_fresh"]
    print(f"OBS exchange frame search on {len(rows)} positions of {tag}, {frames} frames: root values (alpha-beta, alpha-beta + lemmas) == unpruned "
          f"{agree}/{frames}; cheapest-first line == x-ray static count {see_x_eq}/{frames}; another first capturer better than the cheapest {better}")
    print(f"OBS   nodes, alpha-beta -> with lemmas: total {sum(nodes.values())} -> {sum(lnodes.values())}; cuts by lemma: " + ", ".join(f"{k} {v}" for k, v in cuts.most_common()))
    print(f"OBS   retract both lemmas after the search: values still exact {tm['retract_ok']}/{frames}; frames with decisions resting on them "
          f"{tm['affected']}; nodes re-derived {tm['n_retract']}; reinstated: nodes re-derived {tm['n_reinstate']} (must be 0)")
    print(f"OBS   budget of 1 node, then resumed: unfinished after the budget {tm['unfinished']}/{frames}; exact after resuming {tm['resume_ok']}/{frames}; "
          f"nodes in total {tm['resumed_nodes']} vs one full run {sum(lnodes.values())}; of which states expanded for the first time {tm['resumed_fresh']} vs {tm['full_fresh']} (the rest: re-entries of unfinished parents, answered from records)")
    print("OBS   mean nodes by frame size (participants + pieces behind): " + ", ".join(f"{k}: {nodes[k] / count[k]:.1f} -> {lnodes[k] / count[k]:.1f} (n {count[k]})" for k in sorted(count)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["census"])
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--limit", type=int, default=3000)
    a = ap.parse_args(argv)
    census(a.tag, a.limit)


if __name__ == "__main__":
    main()
