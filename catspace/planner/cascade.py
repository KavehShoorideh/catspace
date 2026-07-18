"""
planner/cascade.py — the hand-coded decide loop (PLANNER_PROBE_DESIGN.md
phase B, reorganized around the energy objective E_mu[score] - c*compute,
Kaveh 2026-07-18).

The planner's probing problem is best-ACTION identification, not value
estimation: it needs to know WHICH candidate is best (to within eps), not how
good each one is. So the loop is LUCB-style: coarse-probe every candidate
once, then spend increments only on the pair that still blocks the decision
(the leader and the highest-upper-bound challenger), and stop the moment one
candidate's certified lower bound clears every rival's upper bound — after
that, further search is wasted energy by definition.

Honesty scope: certified [lo, hi] intervals come from probe()'s depth-1
game-truth bounds, so early in a game they are vacuous and the cascade
CANNOT certify-stop — it then decides by point estimate when the budget is
exhausted (reported as such in Decision.certified). Resign and draw-offer
fire ONLY on certified bounds: resign when even the best candidate provably
cannot beat a loss, offer a draw when the best provably cannot beat a draw.
Point estimates never trigger game-ending actions (network confidence may
propose, never close). Tier 0 (plan-memory hit) and tier 1 (label store)
plug in ABOVE this loop once those components exist in play.

White-POV values throughout ([-1, 1]); score = value + 1 lands on Kaveh's
2/1/0 scale. The mover is assumed to be the side to move at `board`.

CAVEAT (found writing the tests): each candidate's MCTS self-calibrates its
value squash from its own root children, so the POINT-ESTIMATE fallback
compares numbers in per-candidate units — sound only when candidates share
a readout/field (the toy MVP case), and never sound enough for game-ending
actions (which is why resign/draw-offer key on certified bounds only).
Certified [lo, hi] is game truth and always comparable. The label store
(probe design §3) is the planned common currency for uncertified values.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from catspace.nn.mcts import DRAW_V, MATED_V, MATE_V, MCTS
from catspace.planner.probe import ProbeResult, deepen, probe


@dataclass
class Candidate:
    """A region/readout to probe: the MCTS instance carries the reach field,
    the goal (committor surface / goal embedding), and the opponent model."""
    name: str
    mcts: MCTS


@dataclass
class Decision:
    action: str                    # "move" | "offer_draw" | "resign"
    move: chess.Move | None
    chosen: str                    # winning candidate's name
    certified: bool                # True: interval dominance; False: point estimate
    evals_spent: int               # total fresh NN evals across all probes
    rounds: int
    probes: dict = field(default_factory=dict)   # name -> ProbeResult


def _mover_lo(r: ProbeResult, white: bool) -> float:
    return r.lo if white else -r.hi


def _mover_hi(r: ProbeResult, white: bool) -> float:
    return r.hi if white else -r.lo


def _mover_value(r: ProbeResult, white: bool) -> float:
    return r.value if white else -r.value


class DecisionCascade:
    """Coarse-probe all candidates, LUCB-deepen only decision-blocking ones,
    stop on certified dominance or budget exhaustion.

    eps: indifference margin — candidates within eps are interchangeable, so
    dominance only needs lo >= rival_hi - eps (spending evals to split finer
    than eps buys no score).
    """

    def __init__(self, candidates: list[Candidate], coarse_budget: int = 64,
                 deepen_budget: int = 128, max_rounds: int = 4,
                 eps: float = 0.05):
        assert candidates
        self.candidates = candidates
        self.coarse_budget = coarse_budget
        self.deepen_budget = deepen_budget
        self.max_rounds = max_rounds
        self.eps = eps

    def decide(self, board: chess.Board) -> Decision:
        white = board.turn == chess.WHITE
        results: dict[str, ProbeResult] = {}
        spent = 0
        for c in self.candidates:                       # coarse pass, every candidate
            r = probe(c.mcts, board, budget=self.coarse_budget)
            results[c.name] = r
            spent += r.evals_spent

        rounds = 0
        while rounds < self.max_rounds:
            leader, certified = self._leader(results, white)
            if certified:
                break
            # LUCB: deepen the leader and the strongest challenger by upper
            # bound — the only two probes whose refinement can change argmax.
            challenger = max((n for n in results if n != leader),
                             key=lambda n: _mover_hi(results[n], white),
                             default=None)
            if challenger is None:                      # single candidate
                break
            rounds += 1
            for name in (leader, challenger):
                c = next(c for c in self.candidates if c.name == name)
                r = deepen(c.mcts, board, results[name], self.deepen_budget)
                results[name] = r
                spent += r.evals_spent

        leader, certified = self._leader(results, white)
        best = results[leader]
        action = "move"
        if _mover_hi(best, white) <= MATED_V + self.eps:
            action = "resign"                           # provably cannot avoid loss
        elif _mover_hi(best, white) <= DRAW_V + 1e-9:
            action = "offer_draw"                       # provably cannot beat a draw
        return Decision(action=action, move=best.best_move, chosen=leader,
                        certified=certified, evals_spent=spent,
                        rounds=rounds, probes=results)

    def _leader(self, results: dict, white: bool) -> tuple[str, bool]:
        """(leader name, certified?) — certified when the leader's lower bound
        clears every rival's upper bound within eps."""
        leader = max(results, key=lambda n: (_mover_lo(results[n], white),
                                             _mover_value(results[n], white)))
        lo = _mover_lo(results[leader], white)
        certified = all(lo >= _mover_hi(results[n], white) - self.eps
                        for n in results if n != leader)
        if not certified:
            # fall back to point-estimate leader for reporting/deepening
            leader = max(results, key=lambda n: _mover_value(results[n], white))
        return leader, certified
