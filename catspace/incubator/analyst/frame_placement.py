"""WHERE EACH PIECE IS WORTH MOST (Kaveh 2026-09-28: "recomputation allows you to basically consider multiple different
versions of a frame ... if the knight were to join this frame, this is what it could achieve; if it were to join that
other frame, that's what it could achieve; so let me push the knight to that other frame ... even though the frames
are not exactly the same, it gives you an idea of which piece is more valuable where").

A piece's ROLES are its marginal worth in the frames it bears on: for an enemy piece it attacks, what the frame wins
for its side with it, less without it; for an own piece it defends, the enemy's threat on that frame without it, less
with it. Its WORTH is the sum. PLACEMENTS relocate the piece -- hypothetically, the rest of the board unchanged -- to
every square it reaches in one or two quiet moves, and read its worth there: the map of where it is worth most. A move
scores the moved piece's worth where it lands (or, at half weight, one quiet move further) less its worth where it
stood. Every number is a frame resolution through the persistent store (`fates.resolve_frame`), so the many versions
of a frame cost little.

    python -m catspace.incubator.analyst.frame_placement --fen "<fen>"
"""
from __future__ import annotations

import argparse
import time

import chess

from .frame_goals import frame_on

HALF = 0.5                                                        # a placement one quiet move further counts half


def _turned(b: chess.Board, side: chess.Color) -> chess.Board | None:
    """The same placement with `side` to move (the hypothetical frames are read from its side); None when the other
    side's king would stand in check (not a position)."""
    nb = b.copy(stack=False); nb.turn = side; nb.ep_square = None
    k = nb.king(not side)
    if k is not None and nb.is_attacked_by(side, k):
        return None
    return nb


def _without(b: chess.Board, sq: int) -> chess.Board:
    nb = b.copy(stack=False); nb.remove_piece_at(sq); return nb


def _moved(b: chess.Board, frm: int, to: int) -> chess.Board:
    nb = b.copy(stack=False); pc = nb.remove_piece_at(frm); nb.set_piece_at(to, pc); return nb


def frame_value(b: chess.Board, t: int, side: chess.Color) -> float:
    """What the frame of the piece on t wins for `side` with `side` to move (0 when t holds none of the enemy's)."""
    pc = b.piece_at(t)
    if pc is None or pc.color == side or pc.piece_type == chess.KING:
        return 0.0
    bb = _turned(b, side)
    return 0.0 if bb is None else frame_on(bb, t)[0]


def roles(b: chess.Board, sq: int) -> dict:
    """{target square: the piece on sq's marginal worth in that frame} over the frames it bears on."""
    pc = b.piece_at(sq)
    if pc is None or pc.piece_type == chess.KING:
        return {}
    side = pc.color; wo = _without(b, sq); out = {}
    for t in b.attacks(sq):
        tp = b.piece_at(t)
        if tp is None or tp.piece_type == chess.KING:
            continue
        if tp.color != side:
            g = frame_value(b, t, side) - frame_value(wo, t, side)
        else:
            g = frame_value(wo, t, not side) - frame_value(b, t, not side)
        if g:
            out[t] = g
    return out


def worth(b: chess.Board, sq: int) -> float:
    return sum(roles(b, sq).values())


def _hops(b: chess.Board, sq: int) -> list:
    """The squares the piece on sq reaches in one quiet legal move (its side to move, the board otherwise unchanged)."""
    pc = b.piece_at(sq)
    bb = b.copy(stack=False); bb.turn = pc.color; bb.ep_square = None
    try:
        return [m.to_square for m in bb.legal_moves if m.from_square == sq and not bb.is_capture(m) and m.promotion in (None, chess.QUEEN)]
    except Exception:
        return []


def placements(b: chess.Board, sq: int, hops: int = 2) -> list:
    """[(square, hops, worth there, roles there)] for every square the piece reaches in `hops` quiet moves, best first."""
    seen = {sq: 0}; frontier = [sq]; cur = b
    boards = {sq: b}
    for h in range(1, hops + 1):
        nxt = []
        for s in frontier:
            for q in _hops(boards[s], s):
                if q in seen:
                    continue
                seen[q] = h; boards[q] = _moved(boards[s], s, q); nxt.append(q)
        frontier = nxt
    out = []
    for q, h in seen.items():
        if q == sq:
            continue
        r = roles(boards[q], q)
        out.append((q, h, sum(r.values()), r))
    out.sort(key=lambda x: (-x[2], x[1]))
    return out


def move_score(b: chess.Board, m: chess.Move) -> float:
    """The moved piece's worth where it lands (or HALF of its best one quiet move further) less its worth where it stood.
    Captures and king moves score 0 here: the frames judge them."""
    pc = b.piece_at(m.from_square)
    if pc is None or pc.piece_type == chess.KING or b.is_capture(m) or m.promotion:
        return 0.0
    now = worth(b, m.from_square)
    at = _moved(b, m.from_square, m.to_square)
    here = worth(at, m.to_square)
    further = max((HALF * w for q, h, w, r in placements(at, m.to_square, 1)), default=0.0)
    return max(here, further) - now


def score_moves(b: chess.Board, moves) -> dict:
    return {m.uci(): float(move_score(b, m)) for m in moves}


def describe(b: chess.Board, side: chess.Color | None = None, top: int = 3) -> list:
    """The map: every piece of `side`, its roles where it stands and its best placements within two quiet moves."""
    side = b.turn if side is None else side
    lines = []
    for sq, pc in sorted(b.piece_map().items(), key=lambda kv: -kv[1].piece_type):
        if pc.color != side or pc.piece_type == chess.KING:
            continue
        r = roles(b, sq)
        now = ", ".join(f"{chess.square_name(t)} {g:+g}" for t, g in r.items()) or "nothing"
        best = [(q, h, w) for q, h, w, _ in placements(b, sq, 2) if w > sum(r.values())][:top]
        more = "; ".join(f"{chess.square_name(q)} in {h}: {w:+g}" for q, h, w in best) or "nowhere better"
        lines.append(f"the {chess.piece_name(pc.piece_type)} on {chess.square_name(sq)}: worth {sum(r.values()):+g} where it stands ({now}); better at {more}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--fen", required=True)
    a = ap.parse_args(argv)
    b = chess.Board(a.fen); t0 = time.perf_counter()
    for c in (b.turn, not b.turn):
        print("White:" if c else "Black:")
        for l in describe(b, c):
            print("  " + l)
    sc = score_moves(b, list(b.legal_moves))
    top = sorted(sc.items(), key=lambda kv: -kv[1])[:6]
    print("moves: " + ", ".join(f"{b.san(chess.Move.from_uci(u))} {v:+g}" for u, v in top))
    print(f"[frame_placement] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
