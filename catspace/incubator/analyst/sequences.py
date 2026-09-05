"""sequences.py -- SEQUENCE MOTIFS (L4.5): local tactics that are not static facts but two-to-three-ply
patterns over edge events, defined in the alphabet motifs.py derived from the puzzle data:
    removal_of_guard  S captures the piece D that defended Y; then S takes Y (or Y is left hanging)
    deflection        S forces D to move off the defence of Y (check, capture or offer); then S takes Y
    attraction        S offers a piece on q; R takes it with P; then S attacks P on q (check / fork / capture)
    interference      S interposes on the line D->Y so D no longer defends Y; then Y falls
    clearance         S vacates a square or line with one piece; two plies later another piece uses it
Each returns the pieces involved (squares at the START position), so the verbalizer can name them."""
from __future__ import annotations

from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class SeqRelation:
    name: str
    squares: tuple[int, ...]     # role order, squares as of the start position (or the square used)
    pieces: tuple[str, ...]
    plies: int

    def names(self):
        return tuple(chess.square_name(s) for s in self.squares)


def _defends(board: chess.Board) -> set[tuple[int, int]]:
    out = set()
    for s, p in board.piece_map().items():
        for t in board.attacks(s):
            q = board.piece_at(t)
            if q is not None and q.color == p.color:
                out.add((s, t))
    return out


def _sym(board, sq):
    p = board.piece_at(sq)
    return p.symbol() if p else "-"


def find_sequences(board: chess.Board, moves, max_plies: int = 3) -> list[SeqRelation]:
    """Motifs in the first <= 3 plies of a line, from the side to move (S) at `board`."""
    mv = [m if isinstance(m, chess.Move) else chess.Move.from_uci(m) for m in moves[:max_plies]]
    if len(mv) < 2:
        return []
    S = board.turn
    b0 = board.copy(stack=False)
    d0 = _defends(b0)
    out = []
    m0 = mv[0]
    cap0 = b0.piece_at(m0.to_square)
    b1 = b0.copy(); b1.push(m0)
    d1 = _defends(b1)
    m1 = mv[1] if len(mv) > 1 and mv[1] in b1.legal_moves else None
    b2 = b1.copy()
    if m1: b2.push(m1)
    d2 = _defends(b2)
    m2 = mv[2] if len(mv) > 2 and m1 and mv[2] in b2.legal_moves else None
    b3 = b2.copy()
    if m2: b3.push(m2)
    enemy = not S

    # ---- removal of the guard: S takes D on ply 0; D defended Y; S takes Y on ply 2 (or Y hangs after ply 1) ----
    if cap0 and cap0.color == enemy:
        guarded = [y for (d, y) in d0 if d == m0.to_square and b0.piece_at(y) and b0.piece_at(y).color == enemy]
        for y in guarded:
            took = m2 is not None and m2.to_square == y and b2.piece_at(y) and b2.piece_at(y).color == enemy
            hangs = (m1 is not None and b2.piece_at(y) and b2.piece_at(y).color == enemy and b2.attackers(S, y) and not b2.attackers(enemy, y))
            if took or hangs:
                out.append(SeqRelation("removal_of_guard", (m0.to_square, y), (_sym(b0, m0.to_square), _sym(b0, y)), 3 if took else 2))
    # ---- deflection: ply 1 moves an enemy piece D off the defence of Y; ply 2 S takes Y ----
    if m1 is not None and m2 is not None:
        d_from = m1.from_square
        lost = [(d, y) for (d, y) in d1 if d == d_from and (d, y) not in d2 and b2.piece_at(y) and b2.piece_at(y).color == enemy]
        lost = [(d, y) for (d, y) in lost if not any(dd == m1.to_square and yy == y for (dd, yy) in d2)]   # not defending it from the new square either
        for d, y in lost:
            if m2.to_square == y and b2.piece_at(y) and b2.piece_at(y).color == enemy:
                forcing = "check" if b1.is_check() else "capture" if cap0 else "offer"
                out.append(SeqRelation("deflection", (m0.to_square, d, y), (_sym(b1, m0.to_square) if b1.piece_at(m0.to_square) else _sym(b0, m0.from_square), _sym(b0, d), _sym(b0, y)), 3))
    # ---- attraction: ply 0 offers on q; ply 1 enemy P takes on q; ply 2 S attacks q (check / capture / new attack) ----
    if m1 is not None and m2 is not None and m1.to_square == m0.to_square and cap_by_enemy(b1, m1):
        q = m0.to_square
        pq = b2.piece_at(q)                                  # the enemy piece pulled onto q
        gives = b3.is_check() and pq is not None and pq.piece_type == chess.KING
        took = m2.to_square == q
        hangs = pq is not None and b3.piece_at(q) is not None and b3.attackers(S, q) and not b3.attackers(enemy, q) and not b2.attackers(S, q)
        if took or gives or hangs:
            out.append(SeqRelation("attraction", (m0.from_square, q, m1.from_square), (_sym(b0, m0.from_square), _sym(b0, m0.to_square), _sym(b1, m1.from_square)), 3))
    # ---- interference: ply 0 lands BETWEEN enemy D and Y (D defended Y through that square); then Y falls ----
    for (d, y) in d0:
        if b0.piece_at(d).color != enemy:
            continue
        if (d, y) not in d1 and m0.to_square in chess.SquareSet.between(d, y) and b1.piece_at(y) and b1.piece_at(y).color == enemy:
            took = m2 is not None and m2.to_square == y
            hangs = b1.attackers(S, y) and not b1.attackers(enemy, y)
            if took or hangs:
                out.append(SeqRelation("interference", (m0.to_square, d, y), (_sym(b1, m0.to_square), _sym(b0, d), _sym(b0, y)), 3 if took else 1))
    # ---- clearance: ply 0 vacates a square/line; ply 2 ANOTHER own piece uses that square or a ray through it ----
    if m2 is not None and m2.from_square != m0.to_square:
        vac = m0.from_square
        uses_square = m2.to_square == vac
        uses_line = vac in chess.SquareSet.between(m2.from_square, m2.to_square) if chess.SquareSet.between(m2.from_square, m2.to_square) else False
        if uses_square or uses_line:
            out.append(SeqRelation("clearance", (vac, m2.from_square, m2.to_square), (_sym(b0, vac), _sym(b2, m2.from_square), "-"), 3))
    return out


def cap_by_enemy(b1: chess.Board, m1: chess.Move) -> bool:
    return b1.is_capture(m1)
