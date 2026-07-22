#!/usr/bin/env python
"""experiments/planner_longshort.py -- Kaveh's architecture (2026-07-20): L2 (a lichess-trained
field) does LONG-DISTANCE goal planning; a SHORT adversarial search does the local execution.

Field for the strategy over distance, search for the tactics up close -- NOT ProQ (which is a pure
field-follower with no short-search executor and no exact grounding, and plateaus for exactly that
reason). Here:

  move(board):
    shallow adversarial negamax (depth `short_depth`) whose LEAF VALUE is the L2 field's
    distance-to-the-goal-region (long-range navigation), with the TABLEBASE as the exact leaf at/
    below the frontier and true terminals exact. White maximizes (toward the goal / White mate),
    Black minimizes. The short search resolves the local tactics the coarse field can't; the field
    supplies the long-range gradient the short search can't see.

Field-agnostic: point --field at the lichess L2 field (or any). Goal region = winning-simplification
/ White-mate. Reports mate rate vs optimal defense.
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

from catspace.data.encode import board_from_packed, encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device
from experiments.value_fixed_point import TB, tb_best_move, white_pov_value

BOARD_ONLY = (18, 19)
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


class LongShortPlanner:
    def __init__(self, field, data, syzygy, frontier, short_depth=3, qdepth=0, goal_n=800, device="auto", seed=0):
        self.dev = pick_device(device)
        self.fb, _ = load_ckpt(Path(field), self.dev); self.fb.eval()
        self.om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
        self.tb = TB(syzygy); self.frontier = frontier; self.short_depth = short_depth; self.qdepth = qdepth
        self.quasi = getattr(self.fb, "quasimetric", False)
        nz = np.load(data, allow_pickle=True)
        P, M, SD, PC = (np.asarray(nz["packed"]), np.asarray(nz["meta"]),
                        np.asarray(nz["sdtm"]), np.asarray(nz["pcount"]).astype(int))
        rng = np.random.default_rng(seed)
        wm = np.flatnonzero((SD > 0) & (SD <= 6) & (PC <= 6))       # White-mate / winning-simplification region
        wm = wm[rng.permutation(len(wm))[:goal_n]]
        self.B_goal = self._embB(P[wm], M[wm])
        self._dcache = {}

    def _planes(self, pk, mt):
        pl = feature_planes(pk, mt); pl[:, BOARD_ONLY] = 0.0
        return torch.from_numpy(pl).to(self.dev)

    def _embF(self, pk, mt):
        with torch.no_grad():
            return self.fb.embed_F(self._planes(pk, mt), torch.from_numpy(np.tile(self.om, (len(pk), 1))).to(self.dev))

    def _embB(self, pk, mt):
        with torch.no_grad():
            return self.fb.embed_B(self._planes(pk, mt))

    def field_value(self, board):
        """long-range leaf value in [-1,1], White-POV: -distance to the White-mate region, squashed.
        (If the field is non-quasimetric, fall back to cosine score.)"""
        key = board._transposition_key()
        v = self._dcache.get(key)
        if v is None:
            F = self._embF(encode_packed(board)[None], encode_meta(board)[None])
            with torch.no_grad():
                if self.quasi:
                    d = float(self.fb.distance_matrix(F, self.B_goal).min(1).values[0])
                    v = float(np.tanh((20.0 - d) / 20.0))          # closer to goal -> higher
                else:
                    s = float((F @ self.B_goal.T).max())            # cosine similarity to goal region
                    v = float(np.tanh(s))
            self._dcache[key] = v
        return v

    def _exact(self, board):
        if board.is_checkmate():
            return 1.0 if board.turn == chess.BLACK else -1.0
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0
        if len(board.piece_map()) <= self.frontier:
            v = white_pov_value(board, self.tb)
            if v is not None:
                return 2.0 * v - 1.0
        return None

    def quiesce(self, board, alpha, beta, qdepth):
        """Quiescence: past the depth cap, keep searching FORCING moves (captures + checks) until
        the position is quiet, then take the field value. This lets forcing conversion lines reach
        the EXACT tablebase (via _exact) instead of being cut off mid-combination -- the mechanism
        that actually converts a won 6p position by winning a piece."""
        ex = self._exact(board)
        if ex is not None:
            return ex
        white = board.turn == chess.WHITE
        in_check = board.is_check()
        stand = self.field_value(board)                       # stand-pat (a bound; skipped if in check)
        if qdepth == 0:
            return stand
        if in_check:
            moves = list(board.legal_moves)                   # must answer check: search all evasions
        else:
            moves = [m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)]
            if not moves:
                return stand                                  # quiet -> field value
        best = -1e9 if (white and in_check) else (1e9 if (not white and in_check) else stand)
        for m in sorted(moves, key=lambda m: not board.is_capture(m)):
            c = board.copy(stack=False); c.push(m)
            v = self.quiesce(c, alpha, beta, qdepth - 1)
            if white:
                best = max(best, v); alpha = max(alpha, v)
            else:
                best = min(best, v); beta = min(beta, v)
            if alpha >= beta:
                break
        return best

    def minimax(self, board, depth, alpha, beta):
        """White-POV value; exact at tablebase/terminal, else L2 field value at the depth cap.
        White maximizes (toward the goal), Black minimizes."""
        ex = self._exact(board)
        if ex is not None:
            return ex
        if depth == 0:
            return self.quiesce(board, alpha, beta, self.qdepth) if self.qdepth else self.field_value(board)
        white = board.turn == chess.WHITE
        best = -1e9 if white else 1e9
        for m in sorted(board.legal_moves, key=lambda m: not board.is_capture(m)):
            c = board.copy(stack=False); c.push(m)
            v = self.minimax(c, depth - 1, alpha, beta)
            if white:
                best = max(best, v); alpha = max(alpha, v)
            else:
                best = min(best, v); beta = min(beta, v)
            if alpha >= beta:
                break
        return best

    def move(self, board):
        if len(board.piece_map()) <= self.frontier:
            return tb_best_move(board, self.tb)                    # forced base case
        best_m, best_v = None, (-1e9 if board.turn == chess.WHITE else 1e9)
        for m in board.legal_moves:
            c = board.copy(stack=False); c.push(m)
            v = self.minimax(c, self.short_depth - 1, -1e9, 1e9)   # short search from each child
            if board.turn == chess.WHITE and v > best_v:
                best_v, best_m = v, m
            elif board.turn == chess.BLACK and v < best_v:
                best_v, best_m = v, m
        return best_m

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
    ap.add_argument("--field", default="data/derived/sep/iqe_nucleus_gn.pt")
    ap.add_argument("--data", default="data/derived/stratified_perfect.npz")
    ap.add_argument("--syzygy", default="data/syzygy")
    ap.add_argument("--frontier", type=int, default=5)
    ap.add_argument("--short-depth", type=int, default=3)
    ap.add_argument("--qdepth", type=int, default=0, help="quiescence: extra plies of captures+checks past the cap")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--ply-cap", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    pl = LongShortPlanner(args.field, args.data, args.syzygy, args.frontier, args.short_depth,
                          qdepth=args.qdepth, seed=args.seed)
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
    print(f"VERDICT LONGSHORT field={Path(args.field).stem} depth={args.short_depth} qdepth={args.qdepth} "
          f"frontier={args.frontier} n={len(starts)} mate_rate={wins/len(starts):.3f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
