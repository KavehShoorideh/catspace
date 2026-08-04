"""catspace/engine/engine.py -- LayeredEngine: the composition. Every layer is injected;
swap any of them independently (Kaveh 2026-07-23). Phase logic follows DECISIONS sec 3:
PLAN (subgoal-guided search) until near-goal or stalled, then EXECUTE (finisher search).

    engine = LayeredEngine(
        value=DTMCNNValue(...),                    # or ConstantValue() / FieldGoalDistanceValue(...)
        subgoals=my_selector,                      # SubgoalSelector or None (skip planning)
        prior=MixturePrior(sub, alpha=0.7),        # or UniformPrior() / None
        plan_search=MCTSSearch(nodes=400),
        execute_search=MCTSSearch(nodes=400),      # finisher: pure by default (value=None)
    )
    result = engine.move(board)
"""
from __future__ import annotations

import chess

from catspace.interfaces import Region, SearchOutcome
from catspace.research.components.search.approaches.puct_mcts.src.layer import MCTSSearch


class LayeredEngine:
    def __init__(self, value=None, subgoals=None, prior=None,
                 plan_search: MCTSSearch | None = None,
                 execute_search: MCTSSearch | None = None,
                 handoff_pieces: int = 5, stall_patience: int = 3,
                 execute_value=None):
        self.value = value                     # GLOBAL leaf value for the plan phase
        self.subgoals = subgoals               # SubgoalSelector | None (the RL seam)
        self.prior = prior                     # MovePrior | None
        self.plan_search = plan_search or MCTSSearch(nodes=400)
        # EXECUTE defaults to PURE search (no value): the validated finisher --
        # the field value HURTS near mate (DECISIONS sec 3).
        self.execute_search = execute_search or MCTSSearch(nodes=400)
        self.execute_value = execute_value     # usually None (pure)
        self.handoff_pieces = handoff_pieces
        self.stall_patience = stall_patience
        self.reset()

    def reset(self):
        self._value_hist: list[float] = []
        self._stall = 0
        self.current_subgoal: Region | None = None

    # ------------------------------------------------------------------ phase
    def _update_stall(self, v: float):
        self._value_hist.append(v)
        h = self._value_hist
        if len(h) > self.stall_patience and v <= max(h[-self.stall_patience - 1:-1]) + 1e-3:
            self._stall += 1
        else:
            self._stall = 0

    def phase(self, board: chess.Board) -> str:
        near = len(board.piece_map()) <= self.handoff_pieces
        if self.current_subgoal is not None and self.current_subgoal.contains(board):
            return "execute"                    # reached the region -> finish locally
        return "execute" if (near or self._stall >= self.stall_patience) else "plan"

    # ------------------------------------------------------------------- move
    def move(self, board: chess.Board) -> dict:
        if self.value is not None:
            v = float(self.value.values([board])[0])
            self._update_stall(v)
        else:
            v = None
        ph = self.phase(board)
        if ph == "plan":
            if self.subgoals is not None and (self.current_subgoal is None
                                              or self.current_subgoal.contains(board)):
                self.current_subgoal = self.subgoals.select(board)   # RL SEAM
            out: SearchOutcome = self.plan_search.best_move(board, value=self.value, prior=self.prior)
        else:
            out = self.execute_search.best_move(board, value=self.execute_value, prior=None)
        return dict(uci=out.move.uci(), phase=ph, pv=[m.uci() for m in out.pv],
                    evals_used=out.evals_used, value=v,
                    subgoal=(self.current_subgoal.meta.get("name") if self.current_subgoal else None))
