"""catspace/interfaces.py -- the layer Protocols. Any implementation satisfying a
Protocol can be injected into LayeredEngine; this is what makes layers swappable."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import chess
import numpy as np


@dataclass
class Region:
    """A goal-as-REGION (DECISIONS sec 7-9; measured 2026-07-23: exact positions are 87%
    adversary-denied, regions at this granularity are 99% forceable): material signature
    + king zones around an exemplar, optionally carrying its embedding bank."""
    material: str                      # sorted piece symbols, e.g. "KRRk"
    black_king: int                    # exemplar square
    white_king: int
    bk_tol: int = 1                    # king-distance tolerances defining the zone
    wk_tol: int = 2
    bank: np.ndarray | None = None     # optional (n,d) B-embeddings of exemplars
    meta: dict = field(default_factory=dict)

    def contains(self, b: chess.Board) -> bool:
        if "".join(sorted(p.symbol() for p in b.piece_map().values())) != self.material:
            return False
        bk, wk = b.king(chess.BLACK), b.king(chess.WHITE)
        return (bk is not None and wk is not None
                and chess.square_distance(bk, self.black_king) <= self.bk_tol
                and chess.square_distance(wk, self.white_king) <= self.wk_tol)


@dataclass
class SearchOutcome:
    move: chess.Move
    pv: list                            # principal variation (moves)
    evals_used: int                     # actual search spent (strength-per-node accounting)
    stats: dict = field(default_factory=dict)


@runtime_checkable
class ValueModel(Protocol):
    """Leaf evaluation: boards -> white-POV values in [-1, 1]. GLOBAL objective only --
    subgoals must never enter the value (DECISIONS sec 8)."""
    def values(self, boards: list) -> np.ndarray: ...


@runtime_checkable
class MovePrior(Protocol):
    """Move prior for the search: board -> {move: prob}. Subgoals enter HERE (the
    alpha-mixture), never the value."""
    def priors(self, board: chess.Board) -> dict: ...


@runtime_checkable
class SubgoalSelector(Protocol):
    """The RL-swappable seam: propose the current Region subgoal (or None to skip
    straight to global search)."""
    def select(self, board: chess.Board) -> Region | None: ...
