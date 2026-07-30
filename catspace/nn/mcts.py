"""Alias -> catspace.navigator.mcts (component refactor 2026-07-30)."""
from catspace.navigator.mcts import *           # noqa: F401,F403
from catspace.navigator.mcts import (MCTS, FBMCTSPolicy, MATE_V, MATED_V,  # noqa: F401
                                     DRAW_V, PLY_DISCOUNT, _Node, game_truth,
                                     is_tactical_move)
