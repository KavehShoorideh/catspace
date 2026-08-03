#!/usr/bin/env python
"""experiments/gen_traj_lc0.py -- REAL-history trajectory data, lc0 (lczerolens) 112-plane
encoding, PRECOMPUTED. Emits the actual LINE trajectory positions (game_id + ply) so the
MULTI-GOAL quasimetric has same-line pairs d(s_i,s_j)=j-i (triangulation -> rank + composability,
+ fine ordering at short gaps). Rolls out won endgame games (tb-optimal both sides + epsilon
exploration for variety, so the line also visits off-optimal / clock-draw positions). Per line
position: 112-plane real-history tensor (uint8) + clock-aware DTZ + ending + game_id + ply.
Single-space field (shared phi) consumes these; same rollout machinery extends to full-game later.
"""
from __future__ import annotations

import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB, DEFAULT_SYZYGY, tb_best_move
from experiments.gen_dtm_data import random_class_start
from experiments.value_fixed_point import white_pov_value

# ALL-OUTCOME classes so the committor learns the W/D/L basins (not win-only).
CLASSES = (["KQvK", "KRvK", "KRRvK", "KBNvK", "KBBvK", "KQvKR", "KRvKN"]        # WIN (White ahead)
           + ["KvKQ", "KvKR", "KvKRR", "KvKBN", "KvKBB", "KRvKQ", "KNvKR"]      # LOSS (Black ahead)
           + ["KNvK", "KBvK", "KNNvK", "KRvKR", "KQvKQ", "KBvKN", "KPvKP"])     # DRAW


def _label(board, tb, syz):
    """clock-aware (dtz, ending) for CURRENT position. dtz: >=1 won |DTZ|, 0 mate, -1 INF.
    Ending: 0 WIN_MATE, 1 DRAW_FIFTY, 2 STALE, 3 INSUF, 4 DRAW_REP, 5 LOSS_MATE."""
    ch = board.halfmove_clock
    if board.is_checkmate():
        return 0, (0 if board.turn == chess.BLACK else 5)    # black mated=WIN; white mated=LOSS
    if board.is_stalemate():
        return -1, 2
    if board.is_insufficient_material():
        return -1, 3
    if board.is_game_over(claim_draw=True):
        return -1, 4
    v = white_pov_value(board, tb)
    if v == 1.0:                                             # WIN basin: distance to White-mate
        try:
            d = abs(syz.probe_dtz(board)); return (d, 0) if d <= (100 - ch) else (-1, 1)
        except Exception:
            return -1, 1
    return (-1, 5) if v == 0.0 else (-1, 4)                  # LOSS basin -> LOSS_MATE; else DRAW


def worker(task):
    classes, n_games, seed, syzygy, eps, cap = task
    rng = np.random.default_rng(seed)
    tb = TB(str(syzygy), cache_db=None); syz = tb.tb
    from lczerolens import LczeroBoard
    planes, dtzs, ends, gids, plies = [], [], [], [], []
    games = 0; game_id = seed * 1_000_000
    while games < n_games:
        cls = classes[rng.integers(0, len(classes))]
        b0 = random_class_start(rng, cls)
        if b0 is None or b0.turn != chess.WHITE or b0.is_game_over():   # any outcome (W/D/L basins)
            continue
        games += 1; game_id += 1
        board = LczeroBoard(b0.fen())
        for ply in range(cap):
            if board.is_game_over(claim_draw=True):
                break
            key, end = _label(board, tb, syz)                # LINE position (real history via move stack)
            planes.append(board.to_input_tensor().to(dtype=torch.uint8).numpy())
            dtzs.append(key); ends.append(end); gids.append(game_id); plies.append(ply)
            moves = list(board.legal_moves)
            if board.turn == chess.WHITE:                    # White: optimal, or epsilon-random for variety
                mv = moves[rng.integers(0, len(moves))] if rng.random() < eps else tb_best_move(board, tb, set())
            else:
                mv = tb_best_move(board, tb, set())
            if mv is None:
                break
            board.push(mv)
    tb.close()
    if not planes:
        z = np.zeros
        return (z((0, 112, 8, 8), np.uint8), z(0, np.int32), z(0, np.int8), z(0, np.int64), z(0, np.int32))
    return (np.stack(planes), np.asarray(dtzs, np.int32), np.asarray(ends, np.int8),
            np.asarray(gids, np.int64), np.asarray(plies, np.int32))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=2000); ap.add_argument("--eps", type=float, default=0.0)  # 0 = both sides tb-optimal (clean basins)
    ap.add_argument("--cap", type=int, default=80); ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="data/derived/traj_lc0_v2.npz"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    tasks = [(CLASSES, max(1, args.games // W), args.seed + i, str(DEFAULT_SYZYGY), args.eps, args.cap) for i in range(W)]
    print(f"[gen-traj-lc0] {W} workers x {args.games // W} games | lc0 112-plane real-history LINE positions", flush=True)
    P, D, E, G, PL = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for i, r in enumerate(ex.map(worker, tasks)):
            P.append(r[0]); D.append(r[1]); E.append(r[2]); G.append(r[3]); PL.append(r[4])
            print(f"  w{i+1}/{W}: {len(r[1])} rows [{time.time()-t0:.0f}s]", flush=True)
    planes = np.concatenate(P); dtz = np.concatenate(D); end = np.concatenate(E)
    gid = np.concatenate(G); ply = np.concatenate(PL)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, planes=planes, dtz=dtz, ending=end, game=gid, ply=ply)
    print(f"\n=== {args.out}: {len(dtz)} rows ({planes.nbytes/1e6:.0f}MB) [{time.time()-t0:.0f}s] ===")
    print(f"  won {int((dtz>=1).sum())} | mate {int((dtz==0).sum())} | INF {int((dtz<0).sum())} | games {len(np.unique(gid))}")
    print("DONE gen_traj_lc0", flush=True)


if __name__ == "__main__":
    main()
