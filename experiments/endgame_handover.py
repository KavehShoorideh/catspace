#!/usr/bin/env python
"""experiments/endgame_handover.py -- the HANDOVER primitive (Kaveh's pivot): the tablebase IS
the endgame. At <=7 pieces the outcome is ASSUMED via a tablebase lookup, so the full-board
field's committor is GROUNDED here (exact WDL) and its embedding TERMINATES at the goal region =
<=7-piece tablebase-WON configs. Mirrors Stockfish/Lc0: WDL for the value, DTZ for the move.
Thin wrapper over catspace.tb (white_pov_value = WDL, tb_best_move = DTZ-optimal).

Run `python experiments/endgame_handover.py` for the self-test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.tb import tb_best_move
from experiments.value_fixed_point import white_pov_value

TB_MAX_PIECES = 7                                            # Syzygy ceiling (no 8-piece)


def piece_count(board: chess.Board) -> int:
    return chess.popcount(board.occupied)


def in_tablebase(board: chess.Board) -> bool:
    return piece_count(board) <= TB_MAX_PIECES


def endgame_lookup(board: chess.Board, tb):
    """HANDOVER: if <=7 pieces, return the ASSUMED outcome + conversion move from the tablebase.
    Returns dict {in_tb, score, move} where score = White-POV expected score (1 win / .5 draw /
    0 loss), move = tablebase-optimal (DTZ) move; or {in_tb: False} above the boundary."""
    if not in_tablebase(board):
        return {"in_tb": False, "score": None, "move": None}
    score = white_pov_value(board, tb)                      # 1.0 / 0.5 / 0.0 (White POV, WDL)
    move = None if board.is_game_over() else tb_best_move(board, tb, set())
    return {"in_tb": True, "score": score, "move": move}


def is_goal(board: chess.Board, tb, side: bool = chess.WHITE) -> bool:
    """Goal-region membership: a <=7-piece tablebase-WON config for `side` -- the terminal the
    full-board committor navigates toward (there the outcome is assumed)."""
    r = endgame_lookup(board, tb)
    if not r["in_tb"]:
        return False
    return r["score"] == (1.0 if side == chess.WHITE else 0.0)


def committor_anchor(board: chess.Board, tb):
    """Exact committor (P(win) / expected score, White POV) when <=7 pieces, else None. This is
    the ground-truth boundary condition the full-board committor is trained toward."""
    r = endgame_lookup(board, tb)
    return r["score"] if r["in_tb"] else None


def _tests():
    import numpy as np
    from catspace.tb import TB, DEFAULT_SYZYGY
    from experiments.gen_dtm_data import random_class_start
    tb = TB(str(DEFAULT_SYZYGY), cache_db=None); rng = np.random.default_rng(0); ok = True

    def sample(cls, want):                                   # a valid position of `cls` with WDL==want
        for _ in range(500):
            b = random_class_start(rng, cls)
            if b is None or b.is_game_over():
                continue
            r = endgame_lookup(b, tb)
            if r["in_tb"] and abs(r["score"] - want) < 1e-9:
                return b, r
        return None, None

    for cls, want, desc in [("KQvK", 1.0, "KQvK White-won -> score 1.0"),
                            ("KRvK", 1.0, "KRvK White-won -> score 1.0"),
                            ("KRvKR", 0.5, "KRvKR -> draw 0.5"),
                            ("KvKQ", 0.0, "White down a queen -> loss 0.0")]:
        b, r = sample(cls, want)
        good = r is not None and r["in_tb"] and abs(r["score"] - want) < 1e-9
        ok &= good
        mv = r["move"].uci() if (r and r["move"]) else None
        print(f"  {'OK ' if good else 'FAIL'} {desc}: score={r['score'] if r else None} move={mv}")
        if good and want == 1.0:
            assert is_goal(b, tb) is True and committor_anchor(b, tb) == 1.0

    assert endgame_lookup(chess.Board(), tb)["in_tb"] is False, ">7 pieces must be outside the tablebase"
    print(f"  OK  goal/anchor + boundary (startpos not in TB) | TB_MAX_PIECES={TB_MAX_PIECES}")
    tb.close()
    print("ALL HANDOVER TESTS PASSED" if ok else "HANDOVER TESTS FAILED")


if __name__ == "__main__":
    _tests()
