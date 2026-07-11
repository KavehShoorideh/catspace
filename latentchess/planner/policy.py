"""planner/policy.py — the Policy protocol and its implementations."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from latentchess.chain import TransitionChain
from latentchess.cone.embedding import QuasimetricEmbedding
from latentchess.scoring import TerminalScores, dtm_filled, fill_terminal_state_scores
from latentchess.planner.readout import ReplyAgg, backup, greedy_policy, move_values


class Policy(Protocol):
    def move_id(self, chain: TransitionChain, s: int, rng: np.random.Generator) -> int:
        """Return a GLOBAL move id (an index into chain.move_kind/move_names)."""
        ...


class TablePolicy:
    """Wraps a precomputed per-live-state LOCAL move-index array -- the `pol`
    convention used throughout the original trainers."""

    def __init__(self, local_moves: np.ndarray):
        self.local_moves = local_moves

    def move_id(self, chain: TransitionChain, s: int, rng: np.random.Generator) -> int:
        return int(chain.move_ptr[s]) + int(self.local_moves[s])


class RandomPolicy:
    def move_id(self, chain: TransitionChain, s: int, rng: np.random.Generator) -> int:
        a, b = int(chain.move_ptr[s]), int(chain.move_ptr[s + 1])
        return a + int(rng.integers(0, b - a))


class EpsGreedy:
    """`base` policy w.p. 1-eps, else a uniform-random move -- the eps_w
    curriculum used throughout the PI trainers."""

    def __init__(self, base: Policy, eps: float):
        self.base = base
        self.eps = eps
        self._random = RandomPolicy()

    def move_id(self, chain: TransitionChain, s: int, rng: np.random.Generator) -> int:
        if self.eps > 0.0 and rng.random() <= self.eps:
            return self._random.move_id(chain, s, rng)
        return self.base.move_id(chain, s, rng)


class DTMOraclePolicy:
    """The exact DTM-minimizing ceiling policy: white minimizes black's best
    (dtm-maximizing) reply, immediate mate always preferred. Reuses the
    MIN-aggregation readout on negated DTM -- MIN(-dtm) = -MAX(dtm), i.e.
    exactly "minimize the worst-case distance to mate"."""

    def __init__(self, chain: TransitionChain, dtm: np.ndarray):
        neg_dtm_full = -dtm_filled(dtm, chain.n)
        neg_dtm_full = fill_terminal_state_scores(neg_dtm_full, chain, TerminalScores.big())
        self._table = greedy_policy(neg_dtm_full, chain, ReplyAgg.MIN, TerminalScores.big())

    def move_id(self, chain: TransitionChain, s: int, rng: np.random.Generator) -> int:
        return int(chain.move_ptr[s]) + int(self._table[s])


def _stratum_of(chain: TransitionChain, s: int) -> str | None:
    for name, rng_ in chain.strata.items():
        if s in rng_:
            return name
    return None


class PlanningPolicy:
    """Consults PlanMemory each move: advances the wake clock (discrete events
    from the last move played, under every registered MoveIdentity, plus
    stratum-crossing), lets the selector pick a plan, greedily maximizes that
    plan's goal reach, then records progress/replan/achieved back into memory.

    Deterministic given (chain, emb, memory, selector) -- consumes no rng, so
    swapping in a PlanningPolicy for a plain greedy_policy table with a single
    MATE goal and tau=-inf must choose IDENTICAL moves (see
    test_planning_policy_parity)."""

    def __init__(self, chain: TransitionChain, emb: QuasimetricEmbedding,
                 memory, ts: TerminalScores, agg: ReplyAgg = ReplyAgg.MIN,
                 depth: int = 1, selector=None, identities: list | None = None):
        self.chain = chain
        self.emb = emb
        self.memory = memory
        self.ts = ts
        self.agg = agg
        self.depth = depth
        self.selector = selector
        self.identities = identities or []
        self._V: dict = {}
        self._prev_state: int | None = None
        self._prev_mid: int | None = None

    def _values(self, goal) -> np.ndarray:
        if goal.name not in self._V:
            from latentchess.cone.embedding import reach as reach_fn
            scores = fill_terminal_state_scores(reach_fn(self.emb, goal, None), self.chain, self.ts)
            if self.depth > 1:
                scores = backup(scores, self.chain, self.agg, self.ts, self.depth - 1)
            self._V[goal.name] = move_values(scores, self.chain, self.agg, self.ts)
        return self._V[goal.name]

    def move_id(self, chain: TransitionChain, s: int, rng: np.random.Generator) -> int:
        from latentchess.planner.selector import GreedyReach

        events = []
        if self._prev_mid is not None:
            for identity in self.identities:
                events.append(identity.key(chain, self._prev_state, self._prev_mid))
            prev_stratum = _stratum_of(chain, self._prev_state)
            cur_stratum = _stratum_of(chain, s)
            if prev_stratum != cur_stratum:
                events.append(("stratum", cur_stratum))

        F_now = self.emb.F_of(np.array([s]))[0] if hasattr(self.emb, "F_of") else None
        self.memory.on_ply(s, events, F_now)

        selector = self.selector or GreedyReach()
        plan = selector.select(s, self.memory)
        goal = plan.goal if plan is not None else self.memory.goals[0]

        V = self._values(goal)
        lo, hi = int(chain.move_ptr[s]), int(chain.move_ptr[s + 1])
        mid = lo + int(np.argmax(V[lo:hi]))

        if plan is not None:
            self.memory.update(plan, s, mid)

        self._prev_state, self._prev_mid = s, mid
        return mid
