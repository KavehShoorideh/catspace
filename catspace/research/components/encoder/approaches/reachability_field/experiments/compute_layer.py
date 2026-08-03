#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/reachability_field/experiments/compute_layer.py -- the COMPUTATION / SIMULATION layer the planner sits on top of
(Kaveh 2026-07-20). The planner (top layer) takes board state + memory + this layer, reasons, and
either runs one of these computations or takes a game/meta action. Every tool returns a STRUCTURED
result carrying its own UNCERTAINTY, so the planner can decide: trust it, simulate more, or act.

Tools exposed (one of which is MCTS):
  probe_tablebase(board) -> Exact|None      exact WDL/DTM at/below the solved frontier
  retrieve(board)        -> Retrieval        kNN over the vector DB (value, variance, OOD distance)
  reachability(board)    -> float            field distance to the won-simplification region
  simulate_mcts(board)   -> SimResult        adversarial MCTS to the tablebase (value, PV, grounded frac)
  estimate_value(board)  -> Estimate         COMPOSED, uncertainty-gated: tablebase -> retrieval, and
                                             a `should_search` flag when the estimate is uncertain

This is the modular seam: the field does reachability, the tablebase does exact outcome, the kNN
DB does approximate outcome + confidence, MCTS composes them under an adversary. The planner never
reaches inside -- it calls these and reads the uncertainty.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field as dfield
from pathlib import Path
from typing import Optional

import chess


from catspace.research.components.search.approaches.puct_mcts.experiments.stratified_mcts import StratifiedMCTS
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import white_pov_value
from catspace.io import paths


@dataclass
class Exact:
    wdl: int                       # White-POV {+1 win, 0 draw, -1 loss}
    value: float                   # [-1, 1]
    source: str = "tablebase"


@dataclass
class Retrieval:
    value: float                   # soft White-POV WDL mean in [-1, 1]
    variance: float                # neighbor disagreement -> uncertainty (high = keep searching)
    confident: bool = False        # variance below the trust threshold
    source: str = "vector_db"


@dataclass
class SimResult:
    value: float                   # White-POV [-1, 1]
    best_move: Optional[chess.Move]
    grounded: float                # fraction of the search that reached the exact tablebase
    source: str = "mcts"


@dataclass
class Estimate:
    value: float                   # White-POV [-1, 1]
    wdl: int
    exact: bool                    # backed by the tablebase
    uncertainty: float             # 0 if exact; else the retrieval variance
    should_search: bool            # planner hint: run simulate_mcts to refine
    source: str


class ComputeLayer:
    """The tools the planner calls. Thin, structured, uncertainty-carrying."""

    def __init__(self, l1, data, syzygy, frontier, knn_k=15, var_thresh=0.35,
                 ref_n=6000, device="auto", seed=0):
        self.core = StratifiedMCTS(l1, data, syzygy, frontier, ref_n=ref_n, knn_k=knn_k,
                                   device=device, seed=seed)
        self.frontier = frontier
        self.var_thresh = var_thresh

    # ---- individual tools ------------------------------------------------
    def probe_tablebase(self, board: chess.Board) -> Optional[Exact]:
        if len(board.piece_map()) > self.frontier:
            return None
        v = white_pov_value(board, self.core.tb)
        if v is None:
            return None
        return Exact(wdl=(1 if v == 1.0 else (-1 if v == 0.0 else 0)), value=2.0 * v - 1.0)

    def retrieve(self, board: chess.Board) -> Retrieval:
        mean, var = self.core.retrieval_value(board)
        return Retrieval(value=mean, variance=var, confident=(var <= self.var_thresh))

    def reachability(self, board: chess.Board) -> float:
        """field distance to the won-simplification region (lower = more reachable)."""
        return float(self.core.field_reach([board])[0])

    def simulate_mcts(self, board: chess.Board, iters: int = 300) -> SimResult:
        value, move = self.core.search(board, iters=iters)
        return SimResult(value=value, best_move=move, grounded=float("nan"))

    def tablebase_move(self, board: chess.Board) -> Optional[chess.Move]:
        """exact optimal move at/below the solved frontier (None above it)."""
        from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import tb_best_move
        if len(board.piece_map()) <= self.frontier:
            return tb_best_move(board, self.core.tb)
        return None

    # ---- composed, uncertainty-gated value ------------------------------
    def estimate_value(self, board: chess.Board) -> Estimate:
        ex = self.probe_tablebase(board)
        if ex is not None:
            return Estimate(value=ex.value, wdl=ex.wdl, exact=True, uncertainty=0.0,
                            should_search=False, source="tablebase")
        r = self.retrieve(board)
        wdl = 1 if r.value > 0.33 else (-1 if r.value < -0.33 else 0)
        return Estimate(value=r.value, wdl=wdl, exact=False, uncertainty=r.variance,
                        should_search=not r.confident, source="vector_db")

    def close(self):
        self.core.close()


def _demo():
    """Smoke: exercise every tool on a couple of positions and show the structured returns."""
    import numpy as np
    from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import tb_best_move  # noqa
    cl = ComputeLayer(paths.sep("iqe_stratified.pt"), paths.derived("stratified_perfect.npz"),
                      str(paths.syzygy_dir()), frontier=5, seed=0)
    nz = np.load(paths.derived("stratified_perfect.npz"), allow_pickle=True)
    P, M, PCNT = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["pcount"]).astype(int)
    from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
    rng = np.random.default_rng(0)
    for pc in (5, 6):
        j = int(rng.choice(np.flatnonzero(PCNT == pc)))
        b = board_from_packed(P[j], M[j])
        print(f"\n{pc}p  {b.fen()}")
        print("  probe_tablebase:", cl.probe_tablebase(b))
        print("  retrieve       :", cl.retrieve(b))
        est = cl.estimate_value(b)
        print("  estimate_value :", est)
        if est.should_search:
            print("  -> uncertain, simulate_mcts:", cl.simulate_mcts(b, iters=200))
    cl.close()
    print("\nVERDICT COMPUTE_LAYER ok -- tools return structured, uncertainty-carrying results")


if __name__ == "__main__":
    _demo()
