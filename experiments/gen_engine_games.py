#!/usr/bin/env python
"""experiments/gen_engine_games.py -- generate STRONG-ENGINE-vs-STRONG-ENGINE games (Stockfish vs
Stockfish, ~2700+ Elo = "no human to make a mistake"), in the SAME lc0-plane npz format as Stage C,
so the metastability analyses (committor_by_material, UMAP, basin purity) run on engine vs human and
compare directly. Kaveh's control: under (near-)perfect play the 3 W/D/L basins should be SHARP.

Diversified openings (a few random plies) so we get W/D/L variety, then SF vs SF to the end.
Per sampled position: lc0 112-plane real-history tensor, white-POV result, ply, game_id, pieces.
"""
from __future__ import annotations

import argparse, os, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(task):
    n_games, seed, depth, cap, stride, tail, open_lo, open_hi = task
    import chess, chess.engine
    import torch
    from lczerolens import LczeroBoard
    sf = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")
    rng = np.random.default_rng(seed)
    planes, results, plies, gids = [], [], [], []
    gid0 = seed * 100000
    RMAP = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
    for g in range(n_games):
        b = LczeroBoard()
        for _ in range(int(rng.integers(open_lo, open_hi))):        # random opening plies -> imbalance/variety
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(ms[rng.integers(0, len(ms))])
        line_start = len(b.move_stack)
        while not b.is_game_over(claim_draw=True) and len(b.move_stack) < cap:
            try:
                r = sf.play(b, chess.engine.Limit(depth=depth))
            except Exception:
                break
            if r.move is None:
                break
            b.push(r.move)
        res = RMAP.get(b.result(claim_draw=True), 0)
        # re-play to record planes at sampled plies (real history)
        n = len(b.move_stack); tail_start = n - tail
        rb = LczeroBoard()
        moves = list(b.move_stack)
        for ply, mv in enumerate(moves):
            rb.push(mv)
            on_stride = ply >= line_start and (ply - line_start) % stride == 0
            is_tail = ply >= tail_start
            if not (on_stride or is_tail):
                continue
            planes.append(rb.to_input_tensor().to(dtype=torch.uint8).numpy())
            results.append(res); plies.append(ply); gids.append(gid0 + g)
    sf.quit()
    if not planes:
        z = np.zeros
        return (z((0, 112, 8, 8), np.uint8), z(0, np.int8), z(0, np.int32), z(0, np.int64))
    return (np.stack(planes), np.asarray(results, np.int8), np.asarray(plies, np.int32), np.asarray(gids, np.int64))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--cap", type=int, default=160)
    ap.add_argument("--stride", type=int, default=6); ap.add_argument("--tail", type=int, default=4)
    ap.add_argument("--open-lo", type=int, default=3); ap.add_argument("--open-hi", type=int, default=9)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="data/derived/engine_sfsf.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    per = max(1, args.games // W)
    tasks = [(per, args.seed + i, args.depth, args.cap, args.stride, args.tail, args.open_lo, args.open_hi)
             for i in range(W)]
    print(f"[gen-engine] SF(depth {args.depth}) vs SF | {W} workers x {per} games | out {args.out}", flush=True)
    P, R, PL, G = [], [], [], []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for i, r in enumerate(ex.map(worker, tasks)):
            P.append(r[0]); R.append(r[1]); PL.append(r[2]); G.append(r[3])
            print(f"  w{i+1}/{W}: {len(r[1])} positions [{time.time()-t0:.0f}s]", flush=True)
    planes = np.concatenate(P); result = np.concatenate(R); ply = np.concatenate(PL); game = np.concatenate(G)
    # ending from result (white-POV), dtz placeholder (endgame grounding not needed for this control)
    ending = np.where(result == 1, 0, np.where(result == -1, 5, 4)).astype(np.int8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, planes=planes, result=result, ending=ending,
                        dtz=np.full(len(result), -1, np.int32), game=game, ply=ply)
    w = int((result == 1).sum()); l = int((result == -1).sum()); d = len(result) - w - l
    print(f"\n=== {args.out}: {len(result)} positions games {len(np.unique(game))} "
          f"| W {w/len(result):.0%} D {d/len(result):.0%} L {l/len(result):.0%} [{time.time()-t0:.0f}s] ===")
    print("DONE gen_engine_games", flush=True)


if __name__ == "__main__":
    main()
