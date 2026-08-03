"""Material signatures (Endgame component; extracted from
catspace/approaches/bootstrap_mate/experiments/bootstrap_mate_engine.py 2026-07-30)."""
from __future__ import annotations

import chess


def mat_sig(b: chess.Board) -> str:
    """material signature, e.g. 'KRRvbkn' (white upper, black lower, sorted) --
    the planner's goal-region vocabulary (rules-structure, no concepts)."""
    w = sorted(p.symbol() for p in b.piece_map().values() if p.color)
    bl = sorted(p.symbol().lower() for p in b.piece_map().values() if not p.color)
    return "".join(w) + "v" + "".join(bl)
