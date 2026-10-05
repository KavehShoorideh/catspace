"""FRAMES of a position (Kaveh 2026-09-28: frames are squares): every square holding a piece, other than a king, that
the other side attacks; each frame's ATTRIBUTES, the vocabulary rules are written in; and the OBSERVATION: what the
game did on each square from a position until the exchanges settled.

Perspective: a position is seen just after a move; the MOVER made it, the other side is to move. Material and
outcomes are counted for the mover."""
from __future__ import annotations

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
LETTER = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}
SETTLE_CAP = 12                                                              # plies at most before the window closes


def material(b: chess.Board, side: bool) -> int:
    return sum(VAL[p.piece_type] * (1 if p.color == side else -1) for p in b.piece_map().values())


def legal_attackers(b: chess.Board, color: bool, sq: int) -> list[int]:
    """Pieces of `color` that attack sq and could take there: a pinned piece only along its pin."""
    return [a for a in b.attackers(color, sq) if not b.is_pinned(color, a) or sq in b.pin(color, a)]


def live_frames(b: chess.Board) -> list[int]:
    """Squares holding a non-king piece that the other side attacks (legally)."""
    out = []
    for sq, p in b.piece_map().items():
        if p.piece_type != chess.KING and legal_attackers(b, not p.color, sq):
            out.append(sq)
    return sorted(out)


def see(b: chess.Board, sq: int, side: bool) -> int:
    """Static exchange on sq for `side` capturing first, both sides taking with their cheapest legal attacker and each
    free to stop: what `side` gains (0 when it cannot take). The textbook swap-off, played on a board copy so x-rays
    and pins update as pieces leave."""
    target = b.piece_at(sq)
    if target is None or target.color == side:
        return 0
    bb = b.copy(stack=False)
    gains = []
    cur_side, victim = side, VAL[target.piece_type]
    while True:
        att = legal_attackers(bb, cur_side, sq)
        att = [a for a in att if bb.piece_at(a).piece_type != chess.KING or not bb.attackers(not cur_side, sq)]
        if not att:
            break
        a = min(att, key=lambda s: VAL[bb.piece_at(s).piece_type] if bb.piece_at(s).piece_type != chess.KING else 100)
        gains.append(victim)
        victim = VAL[bb.piece_at(a).piece_type] if bb.piece_at(a).piece_type != chess.KING else 100
        bb.remove_piece_at(sq)
        piece = bb.remove_piece_at(a)
        bb.set_piece_at(sq, piece)
        cur_side = not cur_side
    # gains[i] is what the i-th capture takes; each side may stop: back-propagate the swap list
    val = 0
    for g in reversed(gains):
        val = max(0, g - val)
    return val if gains else 0


def attributes(b: chess.Board, sq: int, mover: bool, last: chess.Move | None) -> dict:
    """A frame's attributes: the words rules are written in. Each is a plain function of the board (a relation Lean
    can state)."""
    p = b.piece_at(sq)
    att = legal_attackers(b, not p.color, sq)
    dfd = legal_attackers(b, p.color, sq)
    aval = [VAL[b.piece_at(a).piece_type] if b.piece_at(a).piece_type != chess.KING else 100 for a in att]
    dval = [VAL[b.piece_at(d).piece_type] if b.piece_at(d).piece_type != chess.KING else 100 for d in dfd]
    to_move = b.turn
    attacker_side = not p.color
    return dict(
        occ=LETTER[p.piece_type], occ_value=VAL[p.piece_type],
        occ_mover=(p.color == mover),                                        # the occupant belongs to the side that just moved
        attacker_to_move=(attacker_side == to_move),                         # the attackers can take now
        n_att=len(att), n_def=len(dfd),
        min_att=min(aval) if aval else 0, min_def=min(dval) if dval else 0,
        att_lower=bool(aval) and min(aval) < VAL[p.piece_type],              # attacked by a cheaper piece
        hanging=bool(att) and not dfd,
        outnumbered=len(att) > len(dfd),
        see_att=see(b, sq, attacker_side),                                   # what the attackers win by capturing first
        see_mover=(-1 if p.color == mover else 1) * see(b, sq, attacker_side),   # that exchange, in the mover's material
        just_moved=last is not None and last.to_square == sq,
        recapture=last is not None and last.to_square == sq and _was_capture(b, last),   # the last move took on this square
        pinned=b.is_pinned(p.color, sq),
        in_check=b.is_check(),
    )


def _was_capture(b: chess.Board, last: chess.Move) -> bool:
    """Whether `last`, already played to reach b, captured: read from the board before it."""
    if not b.move_stack or b.move_stack[-1] != last:
        return False
    prev = b.copy()
    prev.pop()
    return prev.is_capture(last)


def quiet(b: chess.Board, m: chess.Move) -> bool:
    return not (b.is_capture(m) or b.gives_check(m) or m.promotion)


def observe(boards: list[chess.Board], moves: list[chess.Move], t: int) -> dict:
    """What the game did from position t (the board after moves[t-1]) until the exchanges settled: the window closes at
    the first position s >= t whose next two moves are both quiet (no capture, check or promotion), at most
    SETTLE_CAP plies on, or at the end of the game. -> dict(end=s, by_square={square: mover's material}, delta=total).
    A capture is booked on the square of the piece taken (en passant: the pawn's square); a promotion on its square."""
    mover = not boards[t].turn
    by_sq: dict[int, int] = {}
    s = t
    n = len(moves)
    while s < n and s < t + SETTLE_CAP:
        q1 = quiet(boards[s], moves[s])
        q2 = s + 1 >= n or quiet(boards[s + 1], moves[s + 1])
        if q1 and q2:
            break
        b, m = boards[s], moves[s]
        sign = 1 if b.turn == mover else -1
        if b.is_capture(m):
            sq = m.to_square
            if b.is_en_passant(m):
                sq = chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square))
            by_sq[sq] = by_sq.get(sq, 0) + sign * VAL[b.piece_at(sq).piece_type]
        if m.promotion:
            by_sq[m.to_square] = by_sq.get(m.to_square, 0) + sign * (VAL[m.promotion] - 1)
        s += 1
    return dict(end=s, by_square=by_sq, delta=sum(by_sq.values()))
