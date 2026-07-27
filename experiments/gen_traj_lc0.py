#!/usr/bin/env python
"""experiments/gen_traj_lc0.py -- REAL-history trajectory data with lc0 (lczerolens) 112-plane
encoding, PRECOMPUTED (Kaveh: real history, borrow lc0's encoder, no zero-fill). Rolls out won
endgame games (tablebase-optimal both sides + epsilon exploration for variety); at each White-to-
move parent along the REAL line, encodes every child with LczeroBoard.to_input_tensor() -- so the
112 planes carry the child's REAL last-8-position history. Stores uint8 planes + clock-aware DTZ
+ ending + group (siblings share the parent for the 1-ply rank loss). Precompute avoids slow
move-stack replay at train time. Same rollout machinery extends to full-game/lichess later.
"""
from __future__ import annotations

import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.tb import TB, DEFAULT_SYZYGY, tb_best_move
from experiments.gen_dtm_data import random_class_start
from experiments.value_fixed_point import white_pov_value

CLASSES = ["KQvK", "KRvK", "KRRvK", "KBNvK", "KBBvK", "KQvKR", "KRvKN"]


def worker(task):
    classes, n_games, seed, syzygy, eps, cap = task
    rng = np.random.default_rng(seed)
    tb = TB(str(syzygy), cache_db=None); syz = tb.tb
    from lczerolens import LczeroBoard
    planes, dtzs, ends, grps = [], [], [], []
    gid = seed * 1_000_000; games = 0
    while games < n_games:
        cls = classes[rng.integers(0, len(classes))]
        b0 = random_class_start(rng, cls)
        if b0 is None or b0.turn != chess.WHITE or b0.is_game_over() or white_pov_value(b0, tb) != 1.0:
            continue
        games += 1
        board = LczeroBoard(b0.fen())                       # carries a real move stack as we push
        for _ in range(cap):
            if board.is_game_over(claim_draw=True):
                break
            if board.turn == chess.WHITE:
                gid += 1
                moves = list(board.legal_moves)
                for m in moves:                             # emit each child w/ REAL history
                    board.push(m)
                    ch = board.halfmove_clock
                    if board.is_checkmate():
                        key, end = 0, (0 if board.turn == chess.BLACK else 5)
                    elif board.is_stalemate():
                        key, end = -1, 2
                    elif board.is_insufficient_material():
                        key, end = -1, 3
                    elif board.is_game_over(claim_draw=True) or white_pov_value(board, tb) != 1.0:
                        key, end = -1, 4
                    else:
                        try:
                            d = abs(syz.probe_dtz(board))
                            key, end = (d, 0) if d <= (100 - ch) else (-1, 1)
                        except Exception:
                            key, end = -1, 1
                    planes.append(board.to_input_tensor().to(dtype=__import__("torch").uint8).numpy())
                    dtzs.append(key); ends.append(end); grps.append(gid)
                    board.pop()
                # advance the real line: optimal, or epsilon-random for variety
                mv = (moves[rng.integers(0, len(moves))] if rng.random() < eps
                      else tb_best_move(board, tb, set()))
                if mv is None: break
                board.push(mv)
            else:
                mv = tb_best_move(board, tb, set())
                if mv is None: break
                board.push(mv)
    tb.close()
    if not planes:
        return (np.zeros((0, 112, 8, 8), np.uint8), np.zeros(0, np.int32), np.zeros(0, np.int8), np.zeros(0, np.int64))
    return (np.stack(planes), np.asarray(dtzs, np.int32), np.asarray(ends, np.int8), np.asarray(grps, np.int64))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=1200); ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--cap", type=int, default=60); ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="data/derived/traj_lc0_v1.npz"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    tasks = [(CLASSES, max(1, args.games // W), args.seed + i, str(DEFAULT_SYZYGY), args.eps, args.cap) for i in range(W)]
    print(f"[gen-traj-lc0] {W} workers x {args.games // W} games | lczerolens 112-plane, real history", flush=True)
    P, D, E, G = [], [], [], []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for i, (p, d, e, g) in enumerate(ex.map(worker, tasks)):
            P.append(p); D.append(d); E.append(e); G.append(g)
            print(f"  w{i+1}/{W}: {len(d)} rows [{time.time()-t0:.0f}s]", flush=True)
    planes = np.concatenate(P); dtz = np.concatenate(D); end = np.concatenate(E); grp = np.concatenate(G)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, planes=planes, dtz=dtz, ending=end, group=grp)
    print(f"\n=== {args.out}: {len(dtz)} rows ({planes.nbytes/1e6:.0f}MB planes) [{time.time()-t0:.0f}s] ===")
    print(f"  won {int((dtz>=1).sum())} | mate {int((dtz==0).sum())} | INF {int((dtz<0).sum())} | groups {len(np.unique(grp))}")
    print("DONE gen_traj_lc0", flush=True)


if __name__ == "__main__":
    main()
