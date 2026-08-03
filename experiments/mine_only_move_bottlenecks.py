#!/usr/bin/env python
"""experiments/mine_only_move_bottlenecks.py -- find "only move" positions: where
exactly one legal move holds the current outcome basin (Win/Draw/Loss, mover POV)
and every other legal move drops to a strictly worse basin. Kaveh's ask (2026-08-01/
02): "find the basins, find where the transitions between them are effectively
narrow" -- using the SF-vs-SF opening-pool games (data/derived/
opening_pool_sfsf_moves.tsv, full move lists, 100k games) as the position source.

Ground truth per position: Stockfish MultiPV (multipv = number of legal moves,
depth 12, matching the repo's standard depth convention), which evaluates every
legal move's best-play continuation in ONE search rather than N separate searches.
Each move's WDL (mover POV) is classified into a basin bucket via argmax(win, draw,
loss). "Safe" move = bucket matches the best (PV1) move's bucket -- i.e. doesn't
concede a worse guaranteed outcome. n_safe >= 1 always (PV1 matches itself).
"only-move bottleneck" = n_safe == 1: every alternative to the top move drops a
basin. n_safe counts the size of the safe set generally, not just the binary case.

Cost is the binding constraint, not correctness: MultiPV over every legal move at
every position in the full 1.19M-position corpus is NOT affordable (opening
position alone: 20 moves, depth 12, 0.70s -- middlegame branching is higher).
This script samples a bounded number of (game, ply) positions rather than running
on everything; report the real timing from a small run before scaling up.

Usage:
  experiments/mine_only_move_bottlenecks.py --n 200 --workers 8
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sample_positions(moves_tsv, n, seed, min_ply=10, max_ply_from_end=4):
    """-> list of (game_id, ply, uci_move_list) -- one random mid-game ply per
    sampled game, avoiding the first few opening moves and the last few plies
    (repetitive/near-terminal, less informative for basin transitions)."""
    rng = np.random.default_rng(seed)
    games = []
    with open(moves_tsv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gid, result, moves = parts[0], parts[1], parts[2].split()
            if len(moves) > min_ply + max_ply_from_end:
                games.append((int(gid), moves))
    rng.shuffle(games)
    games = games[:n]
    out = []
    for gid, moves in games:
        lo, hi = min_ply, len(moves) - max_ply_from_end
        ply = int(rng.integers(lo, hi))
        out.append((gid, ply, moves[:ply]))
    return out


def worker(task):
    positions, depth, wid = task
    import chess, chess.engine
    eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")
    try:
        eng.configure({"UCI_ShowWDL": True})
    except Exception:
        pass
    out = []
    for i, (gid, ply, prefix_moves) in enumerate(positions):
        try:
            b = chess.Board()
            for mv in prefix_moves:
                b.push(chess.Move.from_uci(mv))
            legal = list(b.legal_moves)
            if len(legal) < 2 or b.is_game_over():
                continue
            info = eng.analyse(b, chess.engine.Limit(depth=depth), multipv=len(legal))
            buckets = []
            for pv in info:
                wdl = pv.get("wdl")
                if wdl is None:
                    continue
                w = wdl.pov(b.turn)
                probs = (w.wins, w.draws, w.losses)
                bucket = int(np.argmax(probs))          # 0=win 1=draw 2=loss, mover POV
                mv0 = pv["pv"][0].uci() if pv.get("pv") else None
                buckets.append((mv0, bucket, probs))
            if not buckets:
                continue
            best_bucket = buckets[0][1]
            n_safe = sum(1 for _, bk, _ in buckets if bk == best_bucket)
            safe_moves = [mv for mv, bk, _ in buckets if bk == best_bucket]
            out.append(dict(game=gid, ply=ply, fen=b.fen(), n_legal=len(legal),
                             n_safe=n_safe, best_bucket=best_bucket,
                             best_move=buckets[0][0], safe_moves=safe_moves))
        except Exception as e:
            out.append(dict(error=str(e), game=gid, ply=ply))
        if (i + 1) % 20 == 0:
            print(f"  [worker {wid}] {i + 1}/{len(positions)}", flush=True)
    eng.quit()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", default="data/derived/opening_pool_sfsf_moves.tsv")
    ap.add_argument("--n", type=int, default=200, help="number of positions to sample")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/only_move_bottlenecks.npz")
    args = ap.parse_args()
    t0 = time.time()

    positions = sample_positions(args.moves, args.n, args.seed)
    print(f"[bottlenecks] sampled {len(positions)} positions from {args.moves}", flush=True)

    W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    chunks = [positions[i::W] for i in range(W)]
    tasks = [(c, args.depth, i) for i, c in enumerate(chunks) if c]
    print(f"[bottlenecks] {len(tasks)} workers, depth {args.depth}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for r in ex.map(worker, tasks):
            results.extend(r)

    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    elapsed = time.time() - t0
    print(f"[bottlenecks] {len(ok)} ok, {len(errs)} errors, [{elapsed:.0f}s, "
          f"{elapsed / max(1, len(ok)):.2f}s/position]", flush=True)

    if ok:
        n_safe = np.array([r["n_safe"] for r in ok])
        n_legal = np.array([r["n_legal"] for r in ok])
        only_move = n_safe == 1
        print(f"VERDICT only-move-bottlenecks: n={len(ok)} | only-move (n_safe==1): "
              f"{only_move.sum()} ({only_move.mean():.1%}) | median n_safe {np.median(n_safe):.0f} "
              f"| median n_legal {np.median(n_legal):.0f} | median safe-fraction "
              f"{np.median(n_safe / n_legal):.3f}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out,
                  game=[r["game"] for r in ok], ply=[r["ply"] for r in ok],
                  fen=[r["fen"] for r in ok], n_legal=n_legal, n_safe=n_safe,
                  best_bucket=[r["best_bucket"] for r in ok],
                  best_move=[r["best_move"] for r in ok],
                  safe_moves=np.array([",".join(r["safe_moves"]) for r in ok], dtype=object))
        print(f"wrote {args.out}")
    print(f"DONE mine_only_move_bottlenecks [{elapsed:.0f}s]")


if __name__ == "__main__":
    main()
