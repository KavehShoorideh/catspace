"""
chain.py — the one transition-chain representation for all domains (KRk, KRkn, KRRk).

A TransitionChain is a flattened CSR-style encoding of "live" (to-move) states,
their moves, and each move's outcome distribution (uniform over black replies):

  move_ptr[s] .. move_ptr[s+1]   -> global move ids for state s
  move_kind[mid]                 -> KIND_ONGOING | KIND_MATE | KIND_STALEMATE | KIND_WHITE_TERMINAL
  out_ptr[mid] .. out_ptr[mid+1] -> slice into out_flat: outcome state indices for move mid

Outcome/state indices index into a single flat space: live states [0, n_live),
then absorbing terminals (MATE, DRAW[, BWIN]).

This replaces the two incompatible chain idioms that used to coexist
(learn.Chain's python list-of-lists vs KRKNChain/UnionChain's bespoke CSR
arrays) with one type every readout/rollout/trainer function operates on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

KIND_ONGOING = 0
KIND_MATE = 1
KIND_STALEMATE = 2
KIND_WHITE_TERMINAL = 3  # white has no moves (black wins) -- KRkn only


@dataclass(frozen=True)
class Terminals:
    mate: int
    draw: int
    bwin: int | None = None

    @property
    def indices(self) -> tuple[int, ...]:
        idx = (self.mate, self.draw)
        return idx + ((self.bwin,) if self.bwin is not None else ())


@dataclass
class TransitionChain:
    n: int                        # total states incl. absorbing
    n_live: int                   # number of to-move ("W") states with real moves
    move_ptr: np.ndarray          # int64 (n_live+1,)
    move_kind: np.ndarray         # int8  (n_moves,)
    out_ptr: np.ndarray           # int64 (n_moves+1,)
    out_flat: np.ndarray          # int32
    terminals: Terminals
    move_names: list
    strata: dict = field(default_factory=dict)   # name -> range over live-state indices

    def __post_init__(self):
        self.mp0 = self.move_ptr[:-1]
        self.mp1 = self.move_ptr[1:]
        self.op0 = self.out_ptr[:-1]
        self.move_counts = np.diff(self.move_ptr)
        self.out_counts = np.diff(self.out_ptr)
        self.n_moves = len(self.move_kind)
        self.pos_idx = np.arange(self.n_moves)

    def moves_of(self, s: int) -> range:
        return range(int(self.move_ptr[s]), int(self.move_ptr[s + 1]))

    def outs_of(self, mid: int) -> np.ndarray:
        return self.out_flat[self.out_ptr[mid]:self.out_ptr[mid + 1]]


def build_csr(per_state, n: int, n_live: int, terminals: Terminals, strata=None) -> TransitionChain:
    """per_state: iterable (over live states 0..n_live-1, in order) of
    lists of (kind, outcome_indices, name) tuples for that state's moves."""
    mp, mk, op, of, names = [0], [], [0], [], []
    for moves in per_state:
        for kind, outs, name in moves:
            mk.append(kind)
            of.extend(int(o) for o in outs)
            op.append(len(of))
            names.append(name)
        mp.append(len(mk))
    return TransitionChain(
        n=n, n_live=n_live,
        move_ptr=np.array(mp, dtype=np.int64),
        move_kind=np.array(mk, dtype=np.int8),
        out_ptr=np.array(op, dtype=np.int64),
        out_flat=np.array(of, dtype=np.int32),
        terminals=terminals,
        move_names=names,
        strata=strata or {},
    )


def exact_P(chain: TransitionChain) -> sp.csr_matrix:
    """Exact transition matrix under white=uniform-random, black=uniform-random reply."""
    rows, cols, vals = [], [], []
    for s in range(chain.n_live):
        a, b = int(chain.move_ptr[s]), int(chain.move_ptr[s + 1])
        k = b - a
        if k == 0:
            continue
        for mid in range(a, b):
            outs = chain.outs_of(mid)
            p_move = 1.0 / k
            p_out = p_move / len(outs)
            for o in outs:
                rows.append(s); cols.append(int(o)); vals.append(p_out)
    for a_idx in chain.terminals.indices:
        rows.append(a_idx); cols.append(a_idx); vals.append(1.0)
    P = sp.coo_matrix((vals, (rows, cols)), shape=(chain.n, chain.n)).tocsr()
    P.sum_duplicates()
    return P


def empirical_P(rows, cols, n: int, terminals: Terminals):
    """THE single empirical-transition-matrix builder: row-stochastic, absorbing
    rows forced to identity, unvisited rows self-loop (flagged by the visited mask).
    Collapses ~4 near-identical copies previously scattered across trainer scripts."""
    rows = np.asarray(rows); cols = np.asarray(cols)
    counts = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    rowsum = np.asarray(counts.sum(axis=1)).ravel()
    visited = rowsum > 0
    rowsum_safe = np.where(visited, rowsum, 1.0)
    P = (sp.diags(1.0 / rowsum_safe) @ counts).tolil()
    for i in np.where(~visited)[0]:
        P[i, i] = 1.0
    for a in terminals.indices:
        P[a, :] = 0
        P[a, a] = 1.0
    P = P.tocsr()
    P.eliminate_zeros()
    return P, visited
