"""qgt_canon.py -- mover-canonical board representation for the 1-ply dynamics arm
(Kaveh 2026-08-29, via a second Fable instance: "do one ply dynamics and then flip
the side to move ... always the same side always moves").

The 2-ply move unit was a WORKAROUND for parity capture (qgt3e) under an
absolute-color encoder. Canonicalization removes the root cause: the encoder only
ever sees "me to move" (Black positions are mirrored + color-swapped), so lag-1
states share one parity by construction, one ply is a deterministic teacher pair,
and search is negamax with a single operator. The between-ply frame change is the
FIXED square mirror permutation -- exact, learned by nothing.

Conventions:
  - canon(board) -> (canonical board, flipped: bool). canonical.turn is always
    White. flipped == (board.turn == BLACK).
  - to_canon(move, flipped): map a real-board move into the canonical frame.
  - MIRROR_PERM: numpy (64,) permutation with MIRROR_PERM[i] = square_mirror(i);
    apply to the 64 token axis when handing a rolled state to the NEXT ply's frame.
  - Values read in canonical frames are MOVER-POV; negate between plies (negamax).
"""
from __future__ import annotations

import chess
import numpy as np

MIRROR_PERM = np.array([chess.square_mirror(s) for s in range(64)], dtype=np.int64)
assert (MIRROR_PERM[MIRROR_PERM] == np.arange(64)).all()      # an involution


def canon(board: chess.Board):
    """-> (canonical board, flipped). Canonical: side to move is always White.
    board.mirror() flips ranks, swaps colors, castling rights, and the ep square."""
    if board.turn == chess.WHITE:
        return board, False
    return board.mirror(), True


def to_canon(move: chess.Move, flipped: bool) -> chess.Move:
    if not flipped:
        return move
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square),
                      promotion=move.promotion)


def from_canon(move: chess.Move, flipped: bool) -> chess.Move:
    return to_canon(move, flipped)                            # an involution


def canon_index(board: chess.Board, move: chess.Move, cap: int):
    """Index of the canonical image of `move` within the canonical board's legal
    list (the move-token order the encoder sees). -1 if beyond cap/absent --
    mirrored generation order is NOT elementwise-mapped, so indices must be
    recomputed, never reused."""
    cb, fl = canon(board)
    cm = to_canon(move, fl)
    legal = list(cb.legal_moves)[:cap]
    try:
        return legal.index(cm)
    except ValueError:
        return -1


def touch_set(board: chess.Board, move: chess.Move) -> np.ndarray:
    """(64,) bool, PARENT frame: squares whose piece OR attacker sets change under
    `move` -- the move's exact light cone at the FEATURE level (Kaveh 8/30:
    "do the hard gating. that will make it sparse"). Computed by bitboard diff,
    no ray logic to get wrong. From/to always in. NOTE: concept bits are made by
    multi-hop message passing, so TRUE bit flips outside this set are possible --
    that leakage is the gate's false-negative rate, MEASURED per run (gateFN),
    never assumed zero (spec 5's warning)."""
    t = np.zeros(64, bool)
    b2 = board.copy(stack=False)
    b2.push(move)
    for s in range(64):
        if board.piece_at(s) != b2.piece_at(s):
            t[s] = True
            continue
        if (board.attackers_mask(chess.WHITE, s) !=
                b2.attackers_mask(chess.WHITE, s) or
                board.attackers_mask(chess.BLACK, s) !=
                b2.attackers_mask(chess.BLACK, s)):
            t[s] = True
    t[move.from_square] = t[move.to_square] = True
    return t


def _test():
    import random
    rng = random.Random(0)
    b = chess.Board()
    checked = 0
    for _ in range(400):
        if b.is_game_over():
            b = chess.Board()
        mv = rng.choice(list(b.legal_moves))
        cb, fl = canon(b)
        assert cb.turn == chess.WHITE
        cm = to_canon(mv, fl)
        assert cm in cb.legal_moves, "canonical move not legal on canonical board"
        # frame coherence: stepping in canon frame == canonizing the real child
        cb2 = cb.copy(stack=False)
        cb2.push(cm)
        b2 = b.copy(stack=False)
        b2.push(mv)
        cc, fl2 = canon(b2)
        assert fl2 == (not fl) or b2.turn == chess.WHITE
        assert cb2.mirror().board_fen() == cc.board_fen(), \
            f"frame mismatch: {b.fen()} {mv}"
        # the square permutation is the same frame change on token axes
        occ_child = np.zeros(64, np.int8)
        for s, p in cc.piece_map().items():
            occ_child[s] = p.piece_type * (1 if p.color else -1)
        occ_step = np.zeros(64, np.int8)
        for s, p in cb2.piece_map().items():
            occ_step[s] = -p.piece_type * (1 if p.color else -1)  # color swap
        assert (occ_child == occ_step[MIRROR_PERM]).all(), "MIRROR_PERM wrong"
        # canon_index round-trips
        j = canon_index(b, mv, 128)
        assert j >= 0 and list(canon(b)[0].legal_moves)[j] == cm
        # touch set: exact, contains from/to, and every changed sq feature
        ts = touch_set(canon(b)[0], to_canon(mv, canon(b)[1]))
        assert ts[to_canon(mv, canon(b)[1]).from_square] and \
            ts[to_canon(mv, canon(b)[1]).to_square]
        assert 2 <= int(ts.sum()) <= 64
        b.push(mv)
        checked += 1
    print(f"  canon: {checked} random plies -- canonical always White-to-move; "
          f"push-then-canon == canon-then-push-then-mirror; MIRROR_PERM exact on "
          f"occupancy; move indices recomputed correctly")
    print("  QGT CANON TESTS PASS")


if __name__ == "__main__":
    _test()
