#!/usr/bin/env python
"""experiments/catspace_engine.py -- the PLANNER (top layer): a thin CODED POLICY over the
ComputeLayer (Kaveh 2026-07-20: a learned/coded tool-selection policy now, maybe an LLM later).

The planner holds NO tooling of its own -- field, kNN vector DB, tablebase, and MCTS all live in
the ComputeLayer behind clean, uncertainty-carrying interfaces. The planner only READS those and
DECIDES: play the exact move when the tablebase has it, SIMULATE (MCTS) when the estimate is
uncertain, else act greedily on the composed value. (Reasoning-memory and the richer action set --
resign / offer draw / check opponent lag -- plug in here later.)

This refactor collapses the old duplicated tooling into the ComputeLayer; the one-off validation
scripts (search_outcome, search_retrieval_combined, adversarial_distance_validation) still carry
their own copies and are slated for a future pass.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from experiments.compute_layer import ComputeLayer


class Planner:
    """Coded tool-selection policy over the ComputeLayer. move() is one decision step."""

    def __init__(self, cl: ComputeLayer, sim_iters: int = 250):
        self.cl = cl
        self.sim_iters = sim_iters

    def move(self, board: chess.Board, use_search: bool = True):
        # 1) exact play where the tablebase has it
        mv = self.cl.tablebase_move(board)
        if mv is not None:
            return mv
        # 2) above the frontier: is the position's value uncertain?
        est = self.cl.estimate_value(board)
        if use_search and est.should_search:
            sim = self.cl.simulate_mcts(board, iters=self.sim_iters)   # SIMULATE (a tool call)
            if sim.best_move is not None:
                return sim.best_move
        # 3) confident (or search off): greedy on the composed value over children
        best_m, best_v = None, -1e9
        for m in board.legal_moves:
            c = board.copy(stack=False); c.push(m)
            v = self.cl.estimate_value(c).value                        # White-POV in [-1,1]
            vv = v if board.turn == chess.WHITE else -v                # mover maximizes its own outcome
            if vv > best_v:
                best_v, best_m = vv, m
        return best_m


def play(planner, start, use_search, ply_cap, tb):
    from experiments.value_fixed_point import tb_best_move
    b = start.copy(stack=False)
    for _ in range(ply_cap):
        if b.is_game_over(claim_draw=True):
            return b.is_checkmate() and b.turn == chess.BLACK
        m = planner.move(b, use_search=use_search) if b.turn == chess.WHITE else tb_best_move(b, tb)
        if m is None:
            return False
        b.push(m)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1", default="data/derived/sep/iqe_stratified.pt")
    ap.add_argument("--data", default="data/derived/stratified_perfect.npz")
    ap.add_argument("--syzygy", default="data/syzygy")
    ap.add_argument("--frontier", type=int, default=5)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sim-iters", type=int, default=200)
    ap.add_argument("--ply-cap", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    cl = ComputeLayer(args.l1, args.data, args.syzygy, args.frontier, seed=args.seed)
    planner = Planner(cl, sim_iters=args.sim_iters)

    nz = np.load(args.data, allow_pickle=True)
    P, M, SDTM, PCNT = (np.asarray(nz["packed"]), np.asarray(nz["meta"]),
                        np.asarray(nz["sdtm"]), np.asarray(nz["pcount"]).astype(int))
    rng = np.random.default_rng(args.seed)
    cand = np.flatnonzero((SDTM > 0) & (PCNT > args.frontier) & (PCNT <= 6))
    starts = []
    for j in rng.permutation(cand):
        b = board_from_packed(P[j], M[j])
        if b.turn == chess.WHITE and not b.is_game_over():
            starts.append(b)
        if len(starts) >= args.n:
            break
    print(f"[stage] planner over ComputeLayer: {len(starts)} won starts, frontier<= {args.frontier}p, "
          f"vs optimal defense (structure demo -- value quality is the round-0 field's)", flush=True)
    tb = cl.core.tb
    ws = sum(play(planner, b, True, args.ply_cap, tb) for b in starts)
    print(f"  planner (uncertainty-gated simulate): {ws}/{len(starts)} = {ws/len(starts):.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    ng = sum(play(planner, b, False, args.ply_cap, tb) for b in starts)
    print(f"  planner (no search, greedy value):    {ng}/{len(starts)} = {ng/len(starts):.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    cl.close()
    print(f"VERDICT PLANNER frontier={args.frontier} n={len(starts)} gated={ws/len(starts):.3f} "
          f"greedy={ng/len(starts):.3f} (modular: planner -> ComputeLayer -> tools)")


if __name__ == "__main__":
    main()
