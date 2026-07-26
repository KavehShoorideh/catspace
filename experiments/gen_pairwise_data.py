#!/usr/bin/env python
"""experiments/gen_pairwise_data.py -- STRONG-OPPONENT pairwise distance data for the
multi-goal quasimetric field (Kaveh 2026-07-26). MVP opponent = tablebase-optimal
(adversarial, both-sides-optimal): the cleanest 'very strong opponent', for which the
reach-distance is a genuine shortest-path/minimax distance (quasimetric-safe).

For each sampled WON endgame position we roll out the tablebase-optimal line to mate
(catspace.tb.rollout_line) and hindsight-relabel pairs (s=line[i], g=line[j], delta=j-i)
for i<j on the SAME optimal line -- reachable, exact, strong-opponent labels. We sample
goals at MIXED temporal ranges per source state (near + far), which is what fixes the
'magnitude ok, ordering stuck' failure of the scalar field: an anchor is only triangulated
if it sees landmarks at varied distances. Terminal landmarks (the mate at line[-1]) are
tagged (is_mate=1) so the trainer can pin them and so distance-to-mate-region = min over
mate landmarks is checkable at inference.

Saves s_packed/g_packed/s_meta/g_meta/delta/is_mate/material + a COVERAGE report (range
histogram, goal spread over piece-count) so we can see the landmark distribution is spread,
not clustered, before spending anything on training.

Usage: gen_pairwise_data.py --per 8000 --pairs-per-state 6 --out data/derived/pairwise_tb_v1.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import encode_meta, encode_packed
from catspace.tb import TB, rollout_line
from experiments.gen_dtm_data import random_class_start
from experiments.selfplay_generate import random_endgame_start
from experiments.value_fixed_point import white_pov_value


def gen_chunk(task):
    material, n, seed, syzygy_dir, pairs_per_state, near_frac, cap = task
    tb = TB(syzygy_dir, cache_db=None)                 # parallel workers: no shared sqlite (direct probe)
    rng = np.random.default_rng(seed)
    S_pk, G_pk, S_mt, G_mt, dl, mate, mat = [], [], [], [], [], [], []
    got = tries = 0
    while got < n and tries < n * 300:
        tries += 1
        b = (random_class_start(rng, material) if "v" in material
             else random_endgame_start(rng, material))
        if b is None or b.turn != chess.WHITE:
            continue
        if white_pov_value(b, tb) != 1.0:                 # WON only (distance defined)
            continue
        line = rollout_line(b, tb, cap=cap)
        if line is None or len(line) < 2:
            continue
        L = len(line)
        # precompute packed/meta once per line position
        pk = [encode_packed(p) for p in line]
        mt = [encode_meta(p) for p in line]
        # sample pairs (i<j) at MIXED ranges from this line
        for _ in range(pairs_per_state):
            i = int(rng.integers(0, L - 1))
            if rng.random() < near_frac:                  # NEAR goal: small offset
                j = min(L - 1, i + 1 + int(rng.integers(0, max(1, (L - 1 - i) // 3 + 1))))
            else:                                         # FAR goal: anywhere ahead (incl. mate)
                j = int(rng.integers(i + 1, L))
            S_pk.append(pk[i]); S_mt.append(mt[i])
            G_pk.append(pk[j]); G_mt.append(mt[j])
            dl.append(j - i); mate.append(1 if j == L - 1 else 0)
            mat.append(_piece_count(line[i]))
        got += 1
    tb.close()
    if not S_pk:
        return (np.zeros((0, 12), np.uint64),) * 2 + (np.zeros((0, 8), np.uint8),) * 2 + \
               (np.zeros(0, np.int16),) * 3
    return (np.stack(S_pk), np.stack(G_pk), np.stack(S_mt), np.stack(G_mt),
            np.asarray(dl, np.int16), np.asarray(mate, np.int16), np.asarray(mat, np.int16))


def _piece_count(board):
    return chess.popcount(board.occupied)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per", type=int, default=8000, help="won source positions per class")
    ap.add_argument("--pairs-per-state", type=int, default=6)
    ap.add_argument("--near-frac", type=float, default=0.5, help="fraction of NEAR-range goals")
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--classes", nargs="*", default=["KQvK", "KRvK", "KRRvK", "KBBvK", "KBNvK"])
    ap.add_argument("--syzygy", default=None)
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count-1")
    ap.add_argument("--out", default="data/derived/pairwise_tb_v1.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    from catspace.tb import DEFAULT_SYZYGY
    syz = args.syzygy or str(DEFAULT_SYZYGY)

    # parallel: split each class into W sub-chunks, run across a process pool (each worker
    # has its own cache-free TB handle -- tablebase rollouts are embarrassingly parallel).
    import os
    from concurrent.futures import ProcessPoolExecutor
    W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    per_w = max(1, args.per // W)
    tasks, sid = [], 0
    for ci, cls in enumerate(args.classes):
        for w in range(W):
            tasks.append((cls, per_w, args.seed + 1000 * ci + w, syz,
                          args.pairs_per_state, args.near_frac, args.cap))
            sid += 1
    print(f"  {len(tasks)} chunks across {W} workers ({len(args.classes)} classes x {W})", flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for i, r in enumerate(ex.map(gen_chunk, tasks)):
            parts.append(r)
            print(f"  chunk {i+1}/{len(tasks)}: {len(r[4])} pairs [{time.time()-t0:.0f}s]", flush=True)
    S_pk = np.concatenate([p[0] for p in parts]); G_pk = np.concatenate([p[1] for p in parts])
    S_mt = np.concatenate([p[2] for p in parts]); G_mt = np.concatenate([p[3] for p in parts])
    dl = np.concatenate([p[4] for p in parts]); mate = np.concatenate([p[5] for p in parts])
    mat = np.concatenate([p[6] for p in parts])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, s_packed=S_pk, g_packed=G_pk, s_meta=S_mt, g_meta=G_mt,
                        delta=dl, is_mate=mate, material=mat)

    # --- COVERAGE report: are landmarks spread, not clustered? ---
    print(f"\n=== COVERAGE {args.out} ({len(dl)} pairs, {time.time()-t0:.0f}s) ===")
    print(f"  delta (reach-distance) plies: min {dl.min()} med {int(np.median(dl))} "
          f"max {dl.max()} mean {dl.mean():.1f}")
    qs = np.percentile(dl, [10, 25, 50, 75, 90]).astype(int)
    print(f"  delta percentiles [10/25/50/75/90]: {qs.tolist()}")
    bins = [1, 2, 4, 8, 16, 32, 64, 999]
    h = np.histogram(dl, bins=bins)[0]
    print(f"  delta histogram {list(zip(bins[:-1], h.tolist()))}")
    print(f"  is_mate goals: {int(mate.sum())} ({100*mate.mean():.1f}% of pairs reach the mate landmark)")
    pc = Counter(mat.tolist())
    print(f"  source piece-count spread: {dict(sorted(pc.items()))}")
    print(f"  goal piece-count spread:  {dict(sorted(Counter(_gpc(G_pk)).items()))}")
    print("DONE gen_pairwise_data", flush=True)


def _gpc(g_packed):
    from catspace.data.encode import decode_planes
    pl = decode_planes(g_packed).reshape(len(g_packed), 12, 64)
    return pl.sum((1, 2)).astype(int).tolist()          # 12 channels already include both kings


if __name__ == "__main__":
    main()
