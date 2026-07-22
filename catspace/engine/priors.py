"""catspace/engine/priors.py -- MovePrior implementations. Subgoals enter the search HERE
(DECISIONS sec 8: subgoal biases the PRIOR; the VALUE stays global). MixturePrior's alpha
is the focus dial: 1 = blind-focused human, 0 = priorless global search."""
from __future__ import annotations

import chess


class UniformPrior:
    """The validated default for local search (uniform beats the collapsed-field prior)."""

    def priors(self, board: chess.Board) -> dict:
        lm = list(board.legal_moves)
        return {m: 1.0 / len(lm) for m in lm} if lm else {}


class MixturePrior:
    """pi = alpha * pi_subgoal + (1 - alpha) * pi_global. Both components are MovePriors;
    the subgoal component concentrates on subgoal-advancing moves, the global one keeps
    sacrifice/tactic lines discoverable (the anti-blindness term)."""

    def __init__(self, subgoal_prior, global_prior=None, alpha: float = 0.7):
        self.subgoal_prior = subgoal_prior
        self.global_prior = global_prior or UniformPrior()
        self.alpha = float(alpha)

    def priors(self, board: chess.Board) -> dict:
        ps = self.subgoal_prior.priors(board)
        pg = self.global_prior.priors(board)
        keys = set(ps) | set(pg)
        out = {m: self.alpha * ps.get(m, 0.0) + (1.0 - self.alpha) * pg.get(m, 0.0) for m in keys}
        z = sum(out.values()) or 1.0
        return {m: p / z for m, p in out.items()}
