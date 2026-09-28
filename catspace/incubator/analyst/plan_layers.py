"""THE TACTICAL PLAN LAYER, MODULAR (Kaveh 2026-09-28: "make it modular so that multiple different systems/algorithms
can be tested as the 'tactical plan' layer"). The frames (`fates`) settle what the board holds now; a plan layer adds
what a few quiet moves can make of it. Every layer answers the same two questions, so they can be swapped and scored
side by side (`frame_player --plans NAME`):

    score_moves(board, moves) -> {uci: credit}   the side to move's credit for each candidate move (0 = no plan)
    explain(board)            -> [str]           the layer's reasoning, in words

Layers, by name (the registry is the only place that imports them):
    none              no plans: the baseline every layer is scored beside
    frame_goals       the graph of reasoning over frames (`frame_goals`): goals = enemy frames to break, operators
                      add an attacker / take a counted defender off, AND over the opponent's answers, dilemmas
    logical_planner   the September planner (`plans.plans_for`): backward chaining over win/defend goals with reach and
                      the exchange ledger, the opponent's counter-goals as a race
    lambda_frames     order-n threats over frames (`lambda_frames`, Thomsen's lambda-search with Cazenave's zones)
    frame_placement   where each piece is worth most (`frame_placement`): a piece's marginal worth in the frames it
                      bears on, read in hypothetical placements ("what if the knight were here")
"""
from __future__ import annotations

from typing import Protocol

import chess


class PlanLayer(Protocol):
    name: str

    def score_moves(self, board: chess.Board, moves) -> dict: ...

    def explain(self, board: chess.Board) -> list: ...


class NoPlans:
    name = "none"

    def score_moves(self, board, moves):
        return {m.uci(): 0.0 for m in moves}

    def explain(self, board):
        return []


class FrameGoals:
    name = "frame_goals"

    def score_moves(self, board, moves):
        from .frame_goals import score_moves
        return score_moves(board, moves)

    def explain(self, board):
        from .frame_goals import describe, reasoning
        return describe(board, reasoning(board))


class LogicalPlanner:
    """The September planner: a move that starts a winning plan whose race is ours scores the target's value less the
    plan's opportunity cost; a plan the opponent meets in time scores 0."""
    name = "logical_planner"

    def score_moves(self, board, moves):
        from .facts import value
        from .plans import plans_for
        out = {m.uci(): 0.0 for m in moves}
        for pl in plans_for(board, board.turn):
            if pl.goal.kind != "win" or pl.race != "wins" or not pl.steps or pl.steps[0].move == chess.Move.null():
                continue
            u = pl.steps[0].move.uci()
            if u in out:
                out[u] = max(out[u], value(board, pl.goal.square) - pl.cost)
        return out

    def explain(self, board):
        from .plans import plan_text, plans_for
        return [plan_text(board, pl) for c in (board.turn, not board.turn) for pl in plans_for(board, c)]


class LambdaFrames:
    """Order-1 threats by default (order 2 costs minutes a position in Python)."""
    name = "lambda_frames"

    def __init__(self, order: int = 1, budget: int = 400):
        self.order = order; self.budget = budget

    def score_moves(self, board, moves):
        from .lambda_frames import threat_after
        return {m.uci(): threat_after(board, m, self.order, self.budget) for m in moves}

    def explain(self, board):
        from .lambda_frames import Search, line_san
        s = Search(self.budget * 10)
        v, ln, _ = s.attack(board, self.order, 2 * self.order + 1)
        return [f"order <= {self.order}: {v:+.0f} by {line_san(board, ln)}"] if v > 0 else [f"order <= {self.order}: nothing forced"]


class FramePlacement:
    """Kaveh 2026-09-28, 'what if the knight joined this frame': a move scores the moved piece's worth in the frames it
    bears on where it lands (or half of its best one quiet move further) less its worth where it stood."""
    name = "frame_placement"

    def score_moves(self, board, moves):
        from .frame_placement import score_moves
        return score_moves(board, moves)

    def explain(self, board):
        from .frame_placement import describe
        return describe(board)


LAYERS = {"none": NoPlans, "frame_goals": FrameGoals, "logical_planner": LogicalPlanner, "lambda_frames": LambdaFrames,
          "frame_placement": FramePlacement}


def get(name: str) -> PlanLayer:
    if name not in LAYERS:
        raise KeyError(f"no plan layer {name!r}; known: {', '.join(LAYERS)}")
    return LAYERS[name]()
