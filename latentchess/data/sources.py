"""
data/sources.py — the PairSource protocol: bounded-memory batches of (anchor,
goal) training pairs for InfoNCE-style FB training, whatever the origin
(toy chain rollouts now; Lichess game shards in data/lichess.py).

ChainRolloutSource replicates the original neural.py build_pairs/
sample_episodes geometric-horizon pairing exactly (k = 1 + Geometric(1-gamma),
holdout excluded in both the anchor and goal role) on top of the unified
TransitionChain/Policy/Opponent stack, so a full re-run reproduces the
original generalization result (see experiments/generalization.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol

import numpy as np

from latentchess.chain import TransitionChain
from latentchess.game import play_game
from latentchess.opponents import Opponent
from latentchess.planner.policy import Policy


@dataclass
class PairBatch:
    anchors: np.ndarray
    goals: np.ndarray
    meta: dict = field(default_factory=dict)


class PairSource(Protocol):
    def batches(self, batch_size: int, seed: int) -> Iterator[PairBatch]:
        ...


class ChainRolloutSource:
    """Toy-domain PairSource: rolls out (white, black) games on a
    TransitionChain, then samples geometric-horizon (anchor, goal) index
    pairs from each episode -- the InfoNCE training signal for NeuralFB."""

    def __init__(self, chain: TransitionChain, white: Policy, black: Opponent,
                 gamma: float, n_games: int, max_plies: int = 200,
                 holdout_mask: np.ndarray | None = None,
                 encoder: Callable[[np.ndarray], np.ndarray] | None = None):
        self.chain = chain
        self.white = white
        self.black = black
        self.gamma = gamma
        self.n_games = n_games
        self.max_plies = max_plies
        self.holdout_mask = holdout_mask
        self.encoder = encoder

    def _episodes(self, rng: np.random.Generator):
        chain = self.chain
        starts = rng.integers(0, chain.n_live, size=self.n_games)
        for s0 in starts:
            rec = play_game(chain, self.white, self.black, int(s0), cap=self.max_plies, rng=rng)
            ep = list(rec.states)
            if rec.result == "mate":
                ep.append(chain.terminals.mate)
            elif rec.result == "draw":
                ep.append(chain.terminals.draw)
            elif rec.result == "bwin":
                ep.append(chain.terminals.bwin)
            yield ep

    def _held_out(self, s: int) -> bool:
        return (self.holdout_mask is not None and s < self.chain.n_live
                and bool(self.holdout_mask[s]))

    def _pairs(self, rng: np.random.Generator):
        n_live = self.chain.n_live
        for ep in self._episodes(rng):
            L = len(ep)
            for i in range(L - 1):
                s = ep[i]
                if s >= n_live or self._held_out(s):
                    continue
                k = 1 + int(rng.geometric(1.0 - self.gamma))
                j = min(i + k, L - 1)
                g = ep[j]
                if self._held_out(g):
                    continue
                yield s, g

    def batches(self, batch_size: int, seed: int) -> Iterator[PairBatch]:
        rng = np.random.default_rng(seed)
        buf_s: list[int] = []
        buf_g: list[int] = []
        for s, g in self._pairs(rng):
            buf_s.append(s)
            buf_g.append(g)
            if len(buf_s) == batch_size:
                yield self._make_batch(buf_s, buf_g)
                buf_s, buf_g = [], []
        # trailing partial batch is dropped by design (documented above)

    def all_pairs(self, seed: int) -> tuple:
        """Collect every (anchor, goal) index pair from one full `n_games`
        rollout -- for algorithms that resample a fixed pool many times (e.g.
        InfoNCE training) rather than streaming fresh minibatches."""
        rng = np.random.default_rng(seed)
        buf_s, buf_g = [], []
        for s, g in self._pairs(rng):
            buf_s.append(s)
            buf_g.append(g)
        return np.array(buf_s, dtype=np.int32), np.array(buf_g, dtype=np.int32)

    def _make_batch(self, buf_s: list, buf_g: list) -> PairBatch:
        anchors = np.array(buf_s, dtype=np.int32)
        goals = np.array(buf_g, dtype=np.int32)
        if self.encoder is not None:
            anchors = self.encoder(anchors)
            goals = self.encoder(goals)
        return PairBatch(anchors=anchors, goals=goals)
