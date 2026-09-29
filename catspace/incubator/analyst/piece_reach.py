"""CAN THE PIECE MAKE IT THERE? (Kaveh 2026-09-29: "you don't need an interceptor ... consider the piece, and then see
if in the next few moves, can the piece make it there ... in a frame"; "the rook should be in the frame ... the rook
can't make it, the bishop can't make it, so none of these pieces can make it ... until something changes in one of
these frames, we don't need to recalculate anything").

PIECE REACH -- one piece's own frame: every square it can stand on within k of its moves, and every square it can
attack by then, with the move count (a breadth-first walk over the piece's own moves, the other pieces standing
still; pseudo-legal: pins and checks are for the search that verifies). The walk records every square it READ --
the rays it scanned up to and including the blocker, the knight's and king's targets, the pawn's push and capture
squares -- and the answer is kept under the contents of exactly those squares: it is served again until one of them
changes (a piece arrives on the rook's file, the blocker moves away), whatever else happens on the board.

THE GOAL FRAME -- a goal (a piece reaching a square; a pawn running to its last rank) with the pieces that can make it
there in time. The goal piece's path and its arrival times; each opposing piece's reach says whether it can stand on
a path square, or attack one, before the goal piece arrives there. The pieces that make it in time are the frame's
MEMBERS, each with its route and what it leaves behind (its JOB: what its side gives up if it is gone -- our best
frame's rise plus its own side's best frame's fall, the ablation read). The pieces that cannot make it stay out, and their reach footprints are the frame's footprint: nothing is
re-read until one of those squares changes. No opposing member: the goal is UNOPPOSED ("it runs"). Members: the
goal is contested, and the frame -- its squares and members -- is what the tactics resolve.

Which goals to ask about is the layer above's business (Kaveh: the subgoals are to be discovered, not coded); the
pawn run is the first user.

    python -m catspace.incubator.analyst.piece_reach --fen "8/p7/2P5/1P1rk3/R7/6p1/8/6K1 w - - 3 43" --goal c6
    python -m catspace.incubator.analyst.piece_reach exact         # 'nobody can make it', checked by searching every defence
    python -m catspace.incubator.analyst.piece_reach tablebase     # the same positions' results in the Syzygy tables
    python -m catspace.incubator.analyst.piece_reach reuse         # how much is served unchanged along real games
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from dataclasses import dataclass, field

import chess

VERIFY = os.environ.get("ANALYST_VERIFY_FRAMES") == "1"          # recompute every served reach and assert equality

_ROOK_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))
_BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_KNIGHT = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_KING = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


@dataclass
class Reach:
    square: int
    piece: str                          # its symbol
    k: int
    stand: dict                         # square -> fewest moves to stand there (0: where it is)
    attack: dict                        # square -> fewest moves after which it attacks that square
    parent: dict                        # square -> the square it came from (its route)
    footprint: frozenset                # the squares whose contents the walk read

    def route(self, sq: int) -> list:
        out = []
        while sq != self.square:
            out.append(sq); sq = self.parent[sq]
        return out[::-1]

    def attack_route(self, target: int) -> list:
        """The route to a square from which it attacks target, fewest moves."""
        best = min((s for s in self.stand if target in self._attacks_from.get(s, ())), key=lambda s: self.stand[s], default=None)
        return [] if best is None else self.route(best)

    _attacks_from: dict = field(default_factory=dict, repr=False)


def _steps(pc: chess.Piece, sq: int, occupied, reads: set) -> tuple:
    """-> (squares it can move to -- empty ones, its route; squares it attacks), reading occupancy through `occupied`."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    moves, att = [], []
    t = pc.piece_type
    if t in (chess.ROOK, chess.BISHOP, chess.QUEEN):
        dirs = (_ROOK_DIRS if t == chess.ROOK else _BISHOP_DIRS if t == chess.BISHOP else _ROOK_DIRS + _BISHOP_DIRS)
        for df, dr in dirs:
            ff, rr = f + df, r + dr
            while 0 <= ff < 8 and 0 <= rr < 8:
                s = chess.square(ff, rr); reads.add(s); att.append(s)
                if occupied(s):
                    break
                moves.append(s); ff += df; rr += dr
    elif t in (chess.KNIGHT, chess.KING):
        for df, dr in (_KNIGHT if t == chess.KNIGHT else _KING):
            ff, rr = f + df, r + dr
            if 0 <= ff < 8 and 0 <= rr < 8:
                s = chess.square(ff, rr); reads.add(s); att.append(s)
                if not occupied(s):
                    moves.append(s)
    else:                                                          # a pawn: pushes onto empty squares, attacks diagonally
        d = 1 if pc.color == chess.WHITE else -1
        for df in (-1, 1):
            if 0 <= f + df < 8 and 0 <= r + d < 8:
                att.append(chess.square(f + df, r + d))
        if 0 <= r + d < 8:
            s = chess.square(f, r + d); reads.add(s)
            if not occupied(s):
                moves.append(s)
                if r == (1 if pc.color == chess.WHITE else 6):
                    s2 = chess.square(f, r + 2 * d); reads.add(s2)
                    if not occupied(s2):
                        moves.append(s2)
    return moves, att


_STORE: dict = {}                                                  # (square, piece, k) -> [(footprint, contents, Reach)]
STATS = {"computed": 0, "served": 0}


def _contents(b: chess.Board, squares, ignore=frozenset()) -> tuple:
    """What the squares hold; an ignored square (the goal piece, which moves on) reads as empty '-'."""
    return tuple((s, "-" if s in ignore else (b.piece_at(s).symbol() if b.piece_at(s) else None)) for s in sorted(squares))


def piece_reach(b: chess.Board, sq: int, k: int = 4, ignore: frozenset = frozenset()) -> Reach:
    """The piece on sq: where it can stand and what it can attack within k of its own moves, the others still.
    `ignore`: squares read as empty (the goal piece, which will have moved on). Served from the store while the
    squares it read hold what they held -- whichever goal asked, if the walk never read that goal's square."""
    pc = b.piece_at(sq)
    ident = (sq, pc.symbol(), k)
    for foot, contents, r in _STORE.get(ident, ()):
        if _contents(b, foot, ignore) == contents:
            if VERIFY:
                fresh = _compute(b, sq, k, ignore)
                assert (fresh.stand, fresh.attack) == (r.stand, r.attack), f"piece_reach footprint leak: {b.fen()} {chess.square_name(sq)}"
            STATS["served"] += 1
            return r
    r = _compute(b, sq, k, ignore)
    STATS["computed"] += 1
    lst = _STORE.setdefault(ident, [])
    lst.append((r.footprint, _contents(b, r.footprint, ignore), r))
    if len(lst) > 8:
        del lst[0]
    if len(_STORE) > 200000:
        _STORE.clear()
    return r


def clear_store() -> None:
    _STORE.clear(); STATS.update(computed=0, served=0)


def _compute(b: chess.Board, sq: int, k: int, ignore: frozenset) -> Reach:
    pc = b.piece_at(sq)
    occ = b.occupied & ~chess.BB_SQUARES[sq]
    for s in ignore:
        occ &= ~chess.BB_SQUARES[s]
    occupied = lambda s: bool(occ & chess.BB_SQUARES[s])
    reads = {sq}
    stand = {sq: 0}; parent = {}; attack = {}; att_from = {}
    q = deque([sq])
    while q:
        s = q.popleft(); d = stand[s]
        moves, att = _steps(pc, s, occupied, reads)
        att_from[s] = tuple(att)
        for a in att:
            if a not in attack or attack[a] > d:
                attack[a] = d
        if d >= k:
            continue
        for m in moves:
            if m not in stand:
                stand[m] = d + 1; parent[m] = s; q.append(m)
    r = Reach(sq, pc.symbol(), k, stand, attack, parent, frozenset(reads))
    r._attacks_from = att_from
    return r


# ---------------------------------------------------------------- the goal frame
@dataclass
class Member:
    square: int
    piece: str
    how: str                            # "stands on" | "attacks"
    at: int                             # the path square
    moves: int                          # its moves to get there
    allowed: int                        # the moves it has before the goal piece arrives there
    route: list
    job: float | None = None            # what its side gives up if this piece is gone: our best frame's rise plus theirs's fall; None: not read


@dataclass
class GoalFrame:
    goal: int                           # the goal piece's square
    target: int
    path: list                          # [(square, arrival in the goal side's moves)]
    members: list                       # opposing pieces that make it in time
    helpers: list                       # our pieces that make it in time (support)
    outside: list                       # opposing pieces that cannot make it: (square, piece)
    footprint: frozenset
    blocked_by_own: int | None = None   # one of our own pieces stands on the path

    @property
    def unopposed(self) -> bool:
        return not self.members and self.blocked_by_own is None

    @property
    def squares(self) -> set:
        """The frame's squares: the path, the members' and helpers' squares and routes."""
        out = {s for s, _ in self.path} | {self.goal}
        for m in self.members + self.helpers:
            out |= {m.square, m.at} | set(m.route)
        return out


def _path(b: chess.Board, sq: int, target: int | None) -> list:
    """The goal piece's path with its arrival times (its own moves). A pawn: its pushes to the last rank (two from its
    start rank in one move). Another piece: its fewest-move route to the target."""
    pc = b.piece_at(sq)
    if pc.piece_type == chess.PAWN:
        d = 1 if pc.color == chess.WHITE else -1
        f, r = chess.square_file(sq), chess.square_rank(sq)
        last = 7 if pc.color == chess.WHITE else 0
        out = [(sq, 0)]; t = 0; rr = r
        while rr != last:
            rr += d
            if not (r == (1 if pc.color == chess.WHITE else 6) and rr == r + 2 * d):
                t += 1
            out.append((chess.square(f, rr), t))
        return out
    rc = piece_reach(b, sq, k=6)
    if target not in rc.stand:
        return [(sq, 0)]
    return [(sq, 0)] + [(s, rc.stand[s]) for s in rc.route(target)]


def goal_frame(b: chess.Board, goal: int, target: int | None = None, jobs: bool = True) -> GoalFrame:
    """The goal (the piece on `goal` reaching `target`; a pawn: its last rank) with the pieces that can make it to its
    path in time. A square the goal piece reaches on its t-th move can be stood on or attacked in time by an opposing
    piece that needs at most t - 1 moves when the goal side is to move (t when the opponent is), so it is there, or
    can take, when the goal piece arrives."""
    pc = b.piece_at(goal)
    side = pc.color
    path = _path(b, goal, target)
    target = path[-1][0] if target is None else target
    first = 0 if b.turn == side else 1                               # the opponent's extra move when it is to move
    horizon = max(t for _, t in path) + first
    ignore = frozenset({goal})
    members, helpers, outside = [], [], []
    foot = {s for s, _ in path}
    own_block = next((s for s, _ in path[1:] if b.piece_at(s) is not None and b.piece_at(s).color == side), None)
    our_king = b.king(side)
    last_allowed = max(t for _, t in path) - 1 + first               # the moves they have before the goal is reached
    discoverers = {}                                                  # an opposing piece alone on an opposing slider's line to our king
    if our_king is not None and last_allowed >= 1:
        for sl in b.pieces(chess.BISHOP, not side) | b.pieces(chess.ROOK, not side) | b.pieces(chess.QUEEN, not side):
            if not chess.BB_SQUARES[our_king] & chess.BB_RAYS[sl][our_king] or sl == our_king:
                continue
            between = [x for x in chess.SquareSet(chess.between(sl, our_king)) if b.piece_at(x) is not None and x != goal]
            if len(between) == 1 and b.piece_at(between[0]).color != side:
                line_ok = (chess.square_file(sl) == chess.square_file(our_king) or chess.square_rank(sl) == chess.square_rank(our_king))
                kind = b.piece_at(sl).piece_type
                if kind == chess.QUEEN or (kind == chess.ROOK) == line_ok:
                    discoverers[between[0]] = f"{b.piece_at(sl).symbol()}{chess.square_name(sl)}"
                    foot |= {sl, our_king} | set(chess.SquareSet(chess.between(sl, our_king)))
    for s, p in b.piece_map().items():
        if s == goal:
            continue
        rc = piece_reach(b, s, k=max(horizon, 1), ignore=ignore)
        foot |= rc.footprint
        best = None
        for at, t in path:
            allowed = t - 1 + first if p.color != side else t
            if allowed < 0:
                continue
            for how, dist in (("stands on", rc.stand.get(at)), ("attacks", rc.attack.get(at))):
                if dist is not None and dist <= allowed and (at != goal or how == "attacks"):
                    if best is None or (dist, t) < (best.moves, best.allowed):
                        route = rc.route(at) if how == "stands on" else rc.attack_route(at)
                        best = Member(s, p.symbol(), how, at, dist, allowed, route)
        if p.color != side and best is None and our_king is not None and p.piece_type != chess.KING:
            d = rc.attack.get(our_king)                               # a CHECK in time buys a move (the king frame joins, as
            if d is not None and d <= last_allowed:                   # in joint frames: 8/P7/K7/8/3r4/.. Rd6+ or Ra4+ then Rxa8)
                best = Member(s, p.symbol(), "checks the king from", our_king, d, last_allowed, rc.attack_route(our_king))
        if p.color != side and best is None and s in discoverers:    # it steps off a line and a slider behind it checks
            best = Member(s, p.symbol(), f"uncovers a check by {discoverers[s]} on", our_king, 1, last_allowed, [])
        if p.color != side:
            if best is not None:
                members.append(best)
            else:
                outside.append((s, p.symbol()))
        elif best is not None and p.piece_type != chess.PAWN:
            helpers.append(best)
    if jobs and members:                                            # a member's JOB: what its side gives up if it is gone -- our best
        from .ablation import best_frame                            # frame rises, theirs falls (an upper bound for moving it away)
        ours, theirs = best_frame(b, side), best_frame(b, not side)
        for m in members:
            if b.piece_at(m.square).piece_type == chess.KING:
                continue
            e = b.copy(stack=False); e.remove_piece_at(m.square)
            o2, t2 = best_frame(e, side), best_frame(e, not side)
            if None not in (ours, theirs, o2, t2):
                m.job = (o2[0] - ours[0]) + (theirs[0] - t2[0])
    members.sort(key=lambda m: (m.moves, m.at))
    return GoalFrame(goal, target, path, members, helpers, outside, frozenset(foot), own_block)


def explain(b: chess.Board, g: GoalFrame) -> list:
    """The goal frame in words."""
    n = lambda s: f"{b.piece_at(s).symbol() if b.piece_at(s) else ''}{chess.square_name(s)}"
    arrive = max(t for _, t in g.path)
    lines = [f"goal: {n(g.goal)} reaches {chess.square_name(g.target)} in {arrive} move{'s' if arrive != 1 else ''} "
             f"(path {' '.join(chess.square_name(s) for s, _ in g.path[1:])}); {'we' if b.turn == b.piece_at(g.goal).color else 'they'} move first"]
    if g.blocked_by_own is not None:
        lines.append(f"  our own {n(g.blocked_by_own)} stands on the path")
    if not g.members:
        lines.append("  UNOPPOSED: no opposing piece can make it in time -- " + (", ".join(f"{p}{chess.square_name(s)}" for s, p in g.outside) or "there are none"))
    for m in g.members:
        via = " ".join(chess.square_name(s) for s in m.route) or "from where it stands"
        job = "" if m.job is None else (f"; it leaves a job worth {m.job:+g} to us" if m.job > 0 else "; it has no job elsewhere")
        lines.append(f"  {m.piece}{chess.square_name(m.square)} {m.how} {chess.square_name(m.at)} in {m.moves} move{'s' if m.moves != 1 else ''} "
                     f"(it has {m.allowed}; via {via}){job}")
    if g.members and g.outside:
        lines.append("  cannot make it: " + ", ".join(f"{p}{chess.square_name(s)}" for s, p in g.outside))
    for h in g.helpers:
        lines.append(f"  support: {h.piece}{chess.square_name(h.square)} {h.how} {chess.square_name(h.at)} in {h.moves}")
    return lines


# ---------------------------------------------------------------- gauges
def gauge_tablebase(n: int = 4000, seed: int = 0, log=print) -> None:
    """Ground truth for the pawn run: random positions of a white king and pawn against a black king and at most one
    piece, probed in the Syzygy tables. When the goal frame says no opposing piece can make it (UNOPPOSED), the pawn
    should promote and the position be won -- the soundness of 'nobody can make it'. Contested positions are the
    frame's business to resolve; their win rate is printed as a baseline."""
    import random
    import chess.syzygy
    from catspace.io import paths
    rng = random.Random(seed)
    tb = chess.syzygy.open_tablebase(str(paths.assets_dir() / "tablebases" / "syzygy"))
    rows = {}; bad = []
    try:
        while sum(len(v) for v in rows.values()) < n:
            extra = rng.choice([None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
            b = chess.Board(None)
            sqs = rng.sample(range(64), 4)
            ps = chess.square(rng.randrange(8), rng.randrange(1, 7))
            if ps in sqs[:3]:
                continue
            b.set_piece_at(sqs[0], chess.Piece(chess.KING, chess.WHITE)); b.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
            b.set_piece_at(ps, chess.Piece(chess.PAWN, chess.WHITE))
            if extra:
                if sqs[2] == ps:
                    continue
                b.set_piece_at(sqs[2], chess.Piece(extra, chess.BLACK))
            b.turn = rng.choice([chess.WHITE, chess.BLACK])
            if not b.is_valid() or b.is_game_over():
                continue
            try:
                wdl = tb.probe_wdl(b)
            except (KeyError, chess.syzygy.MissingTableError):
                continue
            white = wdl if b.turn == chess.WHITE else -wdl
            g = goal_frame(b, ps, jobs=False)
            kind = "unopposed" if g.unopposed else ("our own piece on the path" if g.blocked_by_own is not None else "contested")
            key = (chess.piece_name(extra) if extra else "king only", kind)
            rows.setdefault(key, []).append(white > 0)
            if kind == "unopposed" and white <= 0 and len(bad) < 5:
                bad.append(b.fen())
    finally:
        tb.close()
    for kind in ("unopposed", "contested", "our own piece on the path"):
        xs = [w for (m, k2), v in rows.items() if k2 == kind for w in v]
        if xs:
            log(f"OBS {kind}: {len(xs)} positions, White wins {sum(xs)} ({sum(xs) / len(xs):.3f})")
    for (m, kind), v in sorted(rows.items()):
        log(f"   {m}, {kind}: {len(v)} positions, White wins {sum(v)}")
    for f in bad:
        log(f"   unopposed but not won: {f}")


def _gets_through(b: chess.Board, sq: int, side: chess.Color, left: int, memo: dict) -> bool:
    """Exact: does the pawn on sq reach its last rank within `left` of its side's moves? Its side only pushes it, or
    answers a check; the other side tries every legal move (a capture of the pawn, a block, a check that costs a move)."""
    key = (b._transposition_key(), sq, left)
    if key in memo:
        return memo[key]
    if b.turn == side:
        if left == 0:
            r = False
        else:
            moves = list(b.legal_moves) if b.is_check() else [m for m in b.legal_moves if m.from_square == sq and not b.is_capture(m)]
            r = False
            for m in moves:
                if m.promotion not in (None, chess.QUEEN):
                    continue
                if m.from_square == sq and m.promotion:
                    r = True; break
                b.push(m)
                try:
                    r = _gets_through(b, m.to_square if m.from_square == sq else sq, side, left - 1, memo)
                finally:
                    b.pop()
                if r:
                    break
    else:
        moves = list(b.legal_moves)
        r = bool(moves) or b.is_check()                               # stalemate stops it; mate does not
        for m in moves:
            if m.to_square == sq:
                r = False; break
            b.push(m)
            try:
                ok = _gets_through(b, sq, side, left, memo)
            finally:
                b.pop()
            if not ok:
                r = False; break
    memo[key] = r
    return r


def gauge_exact(n: int = 3000, seed: int = 0, log=print) -> None:
    """The frame's claim, checked exactly: a white pawn within three moves of its last rank, a black king and at most
    one black piece, a white king. UNOPPOSED claims the pawn reaches the last rank in its own moves whatever Black does
    -- `_gets_through` decides it by searching every Black defence. An unopposed position the pawn does not get
    through is a soundness failure; a contested one it gets through is the frame being cautious."""
    import random
    rng = random.Random(seed)
    tally = {}; bad = []
    while sum(tally.values()) < n:
        extra = rng.choice([None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        b = chess.Board(None)
        sqs = rng.sample(range(64), 3)
        ps = chess.square(rng.randrange(8), rng.randrange(4, 7))
        if ps in sqs:
            continue
        b.set_piece_at(sqs[0], chess.Piece(chess.KING, chess.WHITE)); b.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
        b.set_piece_at(ps, chess.Piece(chess.PAWN, chess.WHITE))
        if extra:
            b.set_piece_at(sqs[2], chess.Piece(extra, chess.BLACK))
        b.turn = rng.choice([chess.WHITE, chess.BLACK])
        if not b.is_valid() or b.is_game_over():
            continue
        g = goal_frame(b, ps, jobs=False)
        if g.blocked_by_own is not None:
            continue
        dist = max(t for _, t in g.path)
        truth = _gets_through(b, ps, chess.WHITE, dist, {})
        key = ("unopposed" if g.unopposed else "contested", truth)
        tally[key] = tally.get(key, 0) + 1
        if g.unopposed and not truth and len(bad) < 6:
            bad.append(b.fen())
    u_ok, u_bad = tally.get(("unopposed", True), 0), tally.get(("unopposed", False), 0)
    c_ok, c_no = tally.get(("contested", True), 0), tally.get(("contested", False), 0)
    log(f"OBS unopposed {u_ok + u_bad}: the pawn gets through in {u_ok}, stopped in {u_bad} (soundness failures)")
    log(f"OBS contested {c_ok + c_no}: stopped in {c_no}, gets through anyway in {c_ok} (the frame cautious)")
    for f in bad:
        log(f"   unopposed but stopped: {f}")


def gauge_reuse(games: int = 40, log=print) -> None:
    """Kaveh: 'until something changes in one of these frames, we don't need to recalculate anything'. Replay games;
    at every position build the goal frame of every pawn (both sides); count the piece reaches computed against those
    served unchanged from the store."""
    import io
    import chess.pgn
    import zstandard
    from catspace.io import paths
    clear_store()
    fh = open(paths.data_dir() / "external" / "lichess" / "lichess_2015-06.pgn.zst", "rb")
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
    positions = frames = 0
    first = [0, 0]
    try:
        for _ in range(games):
            g = chess.pgn.read_game(text)
            if g is None:
                break
            b = g.board()
            for mv in g.mainline_moves():
                b.push(mv)
                if b.is_game_over():
                    break
                positions += 1
                c0, s0 = STATS["computed"], STATS["served"]
                for s in b.piece_map():                               # move to move: each piece's own reach, asked once
                    piece_reach(b, s, k=4)
                first[0] += STATS["computed"] - c0; first[1] += STATS["served"] - s0
                for s in b.pieces(chess.PAWN, chess.WHITE) | b.pieces(chess.PAWN, chess.BLACK):
                    goal_frame(b, s, jobs=False); frames += 1
    finally:
        fh.close()
    tot = STATS["computed"] + STATS["served"]
    log(f"OBS {games} games, {positions} positions: each piece's reach (k=4) asked once a position -- computed {first[0]}, "
        f"served unchanged from an earlier position {first[1]} ({first[1] / max(sum(first), 1):.3f})")
    log(f"OBS with the {frames} pawn goal frames on top: piece reaches computed {STATS['computed']}, served unchanged "
        f"{STATS['served']} ({STATS['served'] / max(tot, 1):.3f} of {tot})")


def main(argv=None):
    import sys
    if argv is None and len(sys.argv) > 1 and sys.argv[1] in ("tablebase", "reuse", "exact"):
        return {"tablebase": gauge_tablebase, "reuse": gauge_reuse, "exact": gauge_exact}[sys.argv[1]]()
    ap = argparse.ArgumentParser(); ap.add_argument("--fen", required=True); ap.add_argument("--goal", required=True, help="the goal piece's square")
    ap.add_argument("--target", default=None, help="its target square (a pawn: its last rank)")
    a = ap.parse_args(argv)
    b = chess.Board(a.fen)
    g = goal_frame(b, chess.parse_square(a.goal), chess.parse_square(a.target) if a.target else None)
    for line in explain(b, g):
        print(line)


if __name__ == "__main__":
    main()
