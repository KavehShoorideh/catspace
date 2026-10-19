"""52.17 (1): THE REGION OF A PIECE and the goals that die by reasoning (Lean: Ideas.Region; words: region, zone, wall, room in GLOSSARY.md).

REGION. For a piece on `s` the connected component of `s` in the graph whose vertices are the squares where the piece may STAND and whose edges are its single
moves over empty squares. Standing: a king -- not held by a piece of theirs (their KING is not a wall: it moves, and the kill must hold whatever it does); any other
piece -- empty or holding a piece of theirs, and not lost there by the exchange word (`G.see_for`). A square holding a piece of theirs is a member and a dead end (the
piece would take it: a different position). The region has a distance (moves from `s`) and a READ SET: the pieces that hold or block its boundary.

ZONE. The squares a goal needs the piece on. KILL by region: the region meets no zone square -- sound as a kill because they need only keep the wall standing (one
defence suffices). The claim direction (we do reach the zone) is a race and stays with the Passer2 form: NEVER claim reach from the region alone.

ROOM. Their king's retreats (Lean `evasion_step`): a step of the lone king that keeps the distance from ours no smaller than it was after any one king move of
ours -- the measure they hold while the room is not taken away (`held_measure`).
"""
from __future__ import annotations

import chess

from catspace.incubator.analyst.discovery import graph as G

KIND_NAME = {chess.KING: "K", chess.QUEEN: "Q", chess.ROOK: "R", chess.BISHOP: "B", chess.KNIGHT: "N", chess.PAWN: "P"}


def cheb(a: int, b: int) -> int:
    """Chebyshev (king-move) distance between two squares: Lean `chebLe`."""
    return chess.square_distance(a, b)


class Region:
    """The region of the piece on `start`: `dist` (square -> moves from start), `reads` (the squares of the pieces that hold or block the boundary), `walls` (the
    boundary squares the piece may not stand on, with who holds them)."""

    def __init__(self, board: chess.Board, start: int):
        pc = board.piece_at(start)
        assert pc is not None and pc.piece_type != chess.PAWN
        self.board, self.start, self.piece = board, start, pc
        self.dist: dict = {start: 0}
        self.walls: dict = {}                                     # boundary square -> the squares of the pieces that hold or block it
        self.reads: set = {start}
        self._build()

    # ---- standing
    def _lifted(self) -> chess.Board:
        c = self.board.copy(stack=False)
        c.remove_piece_at(self.start)
        if self.piece.piece_type == chess.KING:
            ek = c.king(not self.piece.color)                   # their king is no wall for a king: it moves (it holds neither its square nor its neighbours for good)
            if ek is not None:
                c.remove_piece_at(ek)
        return c

    def holders(self, lifted: chess.Board, s: int) -> list:
        """Who keeps the piece from standing on `s`: -> [] when it may stand there."""
        me, b = self.piece.color, self.board
        occ = lifted.piece_at(s)
        if occ is not None and occ.color == me:
            return [s]                                            # our own piece blocks
        if self.piece.piece_type == chess.KING:
            held = [a for a in lifted.attackers(not me, s) if lifted.piece_at(a).piece_type != chess.KING]    # (their king is no wall)
            return sorted(held)
        c = G.with_piece(lifted, self.piece, s)
        if occ is not None and occ.piece_type == chess.KING:
            return [s]                                            # the king is never taken
        if G.see_for(c, s, not me) > 0:
            return sorted(c.attackers(not me, s))
        return []

    def _targets(self, lifted: chess.Board, s: int) -> list:
        """The squares one move of the piece reaches from `s` (the piece placed on `s`, lifted from its start): its attacks, so a slider ends at the first piece in
        the way (a blocker of ours is a wall, `holders`; one of theirs is a member and an end)."""
        return list(G.with_piece(lifted, self.piece, s).attacks(s))

    def _build(self) -> None:
        lifted = self._lifted()
        frontier = [self.start]
        while frontier:
            nxt = []
            for s in frontier:
                for t in self._targets(lifted, s):
                    if t in self.dist:
                        continue
                    h = self.holders(lifted, t)
                    if h:
                        self.walls.setdefault(t, set()).update(h)
                        self.reads.update(h)
                        continue
                    self.dist[t] = self.dist[s] + 1
                    occ = lifted.piece_at(t)
                    if occ is None:
                        nxt.append(t)                             # a square holding a piece of theirs is a member and an end
                    else:
                        self.reads.add(t)
            frontier = nxt

    def squares(self) -> set:
        return set(self.dist)

    def meets(self, zone) -> list:
        return sorted(s for s in zone if s in self.dist)


def region(board: chess.Board, sq: int) -> Region:
    """`region(board, sq)`: mirrors the Lean `Region kingStep stand` for a king (and the same closure under each piece's own step for the others)."""
    return Region(board, sq)


def show(sqs) -> str:
    return "{" + ", ".join(G.name(s) for s in sorted(sqs)) + "}"


# ---------------------------------------------------------------------------------------------------------------- zones

def king_cover_zone(board: chess.Board, me: bool, flights: list) -> set:
    """The squares from which OUR king covers at least one of the flight squares of their king (distance 2 from their king, adjacent to a flight): the king's zone for
    a mate that is short of a cover. Empty flights: no zone."""
    ksq = board.king(not me)
    out = set()
    for s in chess.SQUARES:
        if ksq is not None and cheb(s, ksq) < 2:
            continue                                              # kings are never adjacent
        if any(cheb(s, f) == 1 for f in flights):
            out.add(s)
    return out


def pawn_cover_future(board: chess.Board, me: bool) -> set:
    """Every square a pawn of ours attacks now or from any square of its file ahead (a push is a tempo, but it is a way to cover a square)."""
    out = set()
    d = 1 if me == chess.WHITE else -1
    for p in chess.SquareSet(board.pawns & board.occupied_co[me]):
        f, r = chess.square_file(p), chess.square_rank(p)
        while 0 <= r + d <= 7:
            for df in (-1, 1):
                if 0 <= f + df <= 7:
                    out.add(chess.square(f + df, r + d))
            r += d
    return out


# ---------------------------------------------------------------------------------------------------------------- the two kills of a mate

def other_cover_pieces(board: chess.Board, me: bool, checker: int) -> list:
    """Our pieces besides the king, the pawns and the checker of the root: the ones that could still cover a flight. Empty = the mate has only the king and the pawns
    to cover with."""
    return [s for s in chess.SQUARES if board.piece_at(s) is not None and board.piece_at(s).color == me
            and board.piece_at(s).piece_type not in (chess.KING, chess.PAWN) and s != checker]


def region_kill(board: chess.Board, me: bool, checker_from: int, flights: list, held_after: "chess.Board | None" = None) -> "dict | None":
    """KILL BY THE WALL (Lean `king_region_wall_file` and its rank form; `rook_wall_file`): the mate has flights to cover, our king and our pawns are the only
    covers left (the root's piece stands on its square; no other piece of ours), and the region of our king meets no square from which it covers a flight, nor does any
    pawn of ours (now or after pushes). -> {why, reads} or None."""
    ksq = board.king(me)
    if ksq is None or not flights:
        return None
    if other_cover_pieces(board, me, checker_from):
        return None
    if set(flights) & pawn_cover_future(board, me):
        return None
    zone = king_cover_zone(board, me, flights)
    reg = region(G.without(board, checker_from) if board.piece_at(checker_from) is not None else board, ksq)
    if reg.meets(zone):
        return None
    holders = sorted({h for w in reg.walls.values() for h in w if board.piece_at(h) is not None and board.piece_at(h).color != me})
    return dict(why=f"our K's region {show(reg.squares())} meets no square from which it covers a flight ({show(flights)}; zone {show(zone)}): the wall is "
                    + ", ".join(f"their {KIND_NAME[board.piece_at(h).piece_type].lower()} on {G.name(h)}" for h in holders),
                reads=frozenset(reg.reads) | frozenset(flights))


def open_box(sq: int) -> bool:
    """The king's whole 3x3 box is on the board (Lean `no_rook_mate_in_open`'s premise: a king off the edge)."""
    return 1 <= chess.square_file(sq) <= 6 and 1 <= chess.square_rank(sq) <= 6


def mate_within(board: chess.Board, k: int) -> bool:
    """The rules: the side to move can force a checkmate within `k` of its moves (k = 1, 2; every reply of theirs, all legal moves read)."""
    for m in list(board.legal_moves):
        board.push(m)
        try:
            if board.is_checkmate():
                return True
            if k >= 2 and not board.is_stalemate():
                reps = list(board.legal_moves)
                if reps and all(_after_reply_mates(board, r) for r in reps):
                    return True
        finally:
            board.pop()
    return False


def _after_reply_mates(board: chess.Board, r: chess.Move) -> bool:
    board.push(r)
    try:
        return mate_within(board, 1)
    finally:
        board.pop()


def held_measure(board: chess.Board, me: bool, firsts: list, pushes: int) -> "dict | None":
    """KILL BY THE MEASURE THEY HOLD (Lean `evasion_step`, `no_rook_mate_in_open`; "a king never gains on a king with room"; Kaveh 2026-10-19 on 0LRTF).
    The mate attempt is a king and ONE rook (pawns aside), and every first move `m` of it (the check or the mate threat) has a reply that holds the measure:
    (a) a step of their king onto a square with its whole box on the board, no nearer to our king than it was, after which we have no mate within the pushes their passer
    needs (pushes <= 2 is read by the rules); or (b) the capture of the rook. The mate would have to shrink their room first (the rook's work) while their passer promotes
    in `pushes`: the tempo count is unbounded against it. -> {why, reads} or None. PLAUSIBLE, NOT PROVEN as a whole: the Lean lemmas prove the pieces (no K+R mate in the
    open, the distance held by the step away); that the room cannot be shrunk in time is read from the rules for `pushes` moves only."""
    if pushes < 1 or pushes > 2:
        return None
    ksq, wsq = board.king(not me), board.king(me)
    heavy = [s for s in chess.SQUARES if board.piece_at(s) is not None and board.piece_at(s).color == me and board.piece_at(s).piece_type not in (chess.KING, chess.PAWN)]
    if ksq is None or wsq is None or len(heavy) != 1 or board.piece_at(heavy[0]).piece_type != chess.ROOK or not firsts:
        return None
    rook = heavy[0]
    d0 = cheb(wsq, ksq)
    reg = region(board, wsq)
    our_steps = [wsq] + [t for t, k in reg.dist.items() if k == 1]      # the moves our king has: its region's first ring (the walls leave the others out)
    lines, reads = [], {ksq, wsq, rook} | reg.reads
    for m in firsts:
        c = board.copy(stack=False)
        san = c.san(m)
        c.push(m)
        ok = None
        for r in list(c.legal_moves):
            held = None
            if r.from_square == ksq and open_box(r.to_square):
                if all(cheb(w2, r.to_square) >= d0 for w2 in our_steps):
                    held = f"{c.san(r)} (the box of {G.name(r.to_square)} is on the board, distance {d0} held)"
            elif r.to_square == m.to_square and c.piece_at(r.to_square) is not None and c.piece_at(r.to_square).piece_type == chess.ROOK:
                held = f"{c.san(r)} (the rook is taken)"
            if held is None:
                continue
            d = c.copy(stack=False)
            d.push(r)
            if d.is_game_over() or mate_within(d, pushes):
                continue
            ok = held
            reads.add(r.to_square)
            break
        if ok is None:
            return None
        lines.append(f"{san} {ok}")
    return dict(why="their K keeps its room: " + "; ".join(lines) + f"; no mate within {pushes} follows (the rules), a rook and a king never mate a king in the open (Ideas.Region.no_rook_mate_in_open), "
                "the distance is held by the step away (Ideas.Region.evasion_step); shrinking the room is the rook's work, and their passer promotes in " + str(pushes), reads=frozenset(reads))


def bound_pieces(board: chess.Board, me: bool) -> list:
    """The reverse lemma (Kaveh 2026-10-19): our piece BOUND to a duty -- it guards a piece of ours and, without its guard, the exchange on that piece wins for them.
    -> [(guard square, guarded square)]; the guarded piece is never the guard itself."""
    out = []
    for g in chess.SquareSet(board.occupied_co[me]):
        for y in board.attacks(g):
            py = board.piece_at(y)
            if py is None or py.color != me or py.piece_type == chess.KING:
                continue
            att = len(board.attackers(not me, y))
            guards = [x for x in board.attackers(me, y) if x != g]
            if att and len(board.attackers(me, y)) >= att and len(guards) < att:
                out.append((g, y))                                    # (counted as the state lines count: attacked n times, guarded n times, one less once g is gone)
    return sorted(out)
