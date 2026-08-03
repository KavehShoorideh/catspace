#!/usr/bin/env python
"""experiments/gradient_planner.py -- human-style planning as GRADIENT DESCENT toward a concept
region (Kaveh 2026-07-20: "each player just moves toward higher WDL / toward the good region").
NOT flat MCTS. The plan is: steer toward the winning-simplification / White-mate region on the
ADVERSARIAL (occupancy) field, value-gated so we never blunder the win; the tablebase is the
forced base case that finishes.

move(board):
  * <= frontier            -> tb_best_move (the concept 'won simpler endgame' is FORCED here; execute)
  * above the frontier     -> among non-blundering children (value gate), pick the one that
                              descends d(F(child) -> B(White-mate region)) the most  (navigate)

The whole thing is a gradient-follower; its ceiling is the FIELD's gradient. We compare fields
(occupancy vs. others) and measure mate rate vs optimal defense.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from experiments.value_fixed_point import TB, tb_best_move, white_pov_value

BOARD_ONLY = (18, 19)


class GradientPlanner:
    def __init__(self, l1, data, syzygy, frontier, goal_n=800, device="auto", seed=0):
        self.dev = pick_device(device)
        self.fb, _ = load_ckpt(Path(l1), self.dev); self.fb.eval()
        self.om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
        self.tb = TB(syzygy); self.frontier = frontier
        nz = np.load(data, allow_pickle=True)
        P, M, SD, PC = (np.asarray(nz["packed"]), np.asarray(nz["meta"]),
                        np.asarray(nz["sdtm"]), np.asarray(nz["pcount"]).astype(int))
        rng = np.random.default_rng(seed)
        # concept region = WHITE-MATE: won, near-mate (low DTM) positions (White about to mate)
        wm = np.flatnonzero((SD > 0) & (SD <= 6) & (PC <= 6))
        wm = wm[rng.permutation(len(wm))[:goal_n]]
        self.B_goal = self._embB(P[wm], M[wm])

    def _emb(self, pk, mt, side):
        pl = feature_planes(pk, mt); pl[:, BOARD_ONLY] = 0.0
        t = torch.from_numpy(pl).to(self.dev)
        with torch.no_grad():
            if side == "F":
                return self.fb.embed_F(t, torch.from_numpy(np.tile(self.om, (len(pk), 1))).to(self.dev))
            return self.fb.embed_B(t)

    def _embB(self, pk, mt):
        return self._emb(pk, mt, "B")

    def dist_to_goal(self, boards):
        pk = np.stack([encode_packed(b) for b in boards]); mt = np.stack([encode_meta(b) for b in boards])
        with torch.no_grad():
            return self.fb.distance_matrix(self._emb(pk, mt, "F"), self.B_goal).min(1).values.cpu().numpy()

    def move(self, board):
        if len(board.piece_map()) <= self.frontier:
            return tb_best_move(board, self.tb)
        kids = [(m, (lambda c: (c.push(m), c)[1])(board.copy(stack=False))) for m in board.legal_moves]
        # value gate (White-POV): tablebase where the child is <= frontier, else the field distance
        wpov = []
        for m, c in kids:
            if c.is_checkmate():
                wpov.append(1.0 if c.turn == chess.BLACK else 0.0)
            elif len(c.piece_map()) <= self.frontier:
                v = white_pov_value(c, self.tb); wpov.append(0.5 if v is None else v)
            else:
                wpov.append(None)
        # descend the field distance to the White-mate region among non-blundering children
        d = self.dist_to_goal([c for _, c in kids])
        val = np.array([(-dd if w is None else (2 * w - 1) * 100) for (w, dd) in zip(wpov, d)])  # exact children win big
        if board.turn == chess.WHITE:
            return kids[int(np.argmax(val))][0]                    # White: maximize (closer to White mate / winning)
        return kids[int(np.argmin(val))][0]                        # Black (if ever): minimize

    def close(self):
        self.tb.close()


def play(planner, start, ply_cap):
    b = start.copy(stack=False)
    for _ in range(ply_cap):
        if b.is_game_over(claim_draw=True):
            return b.is_checkmate() and b.turn == chess.BLACK
        m = planner.move(b) if b.turn == chess.WHITE else tb_best_move(b, planner.tb)
        if m is None:
            return False
        b.push(m)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="data/derived/sep/iqe_occupancy.pt")
    ap.add_argument("--data", default="data/derived/stratified_perfect.npz")
    ap.add_argument("--syzygy", default="data/syzygy")
    ap.add_argument("--frontier", type=int, default=5)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--ply-cap", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    pl = GradientPlanner(args.field, args.data, args.syzygy, args.frontier, seed=args.seed)
    nz = np.load(args.data, allow_pickle=True)
    P, M, SD, PC = (np.asarray(nz["packed"]), np.asarray(nz["meta"]),
                    np.asarray(nz["sdtm"]), np.asarray(nz["pcount"]).astype(int))
    rng = np.random.default_rng(args.seed)
    cand = np.flatnonzero((SD > 0) & (PC > args.frontier) & (PC <= 6))
    starts = []
    for j in rng.permutation(cand):
        b = board_from_packed(P[j], M[j])
        if b.turn == chess.WHITE and not b.is_game_over():
            starts.append(b)
        if len(starts) >= args.n:
            break
    wins = sum(play(pl, b, args.ply_cap) for b in starts)
    pl.close()
    print(f"VERDICT GRADIENT_PLANNER field={Path(args.field).stem} frontier={args.frontier} "
          f"n={len(starts)} mate_rate={wins/len(starts):.3f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
