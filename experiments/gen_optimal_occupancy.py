#!/usr/bin/env python
"""experiments/gen_optimal_occupancy.py -- the OPTIMAL-PLAY OCCUPANCY for <=6 (Kaveh 2026-07-20):
embed what STRONG (tablebase-optimal) play actually does, so the field is adversarial + carries
the DTM gradient the cooperative reachability field lacked.

Crucially trains on ALL THREE OUTCOMES. A field of only winning lines can't AVOID losing (it never
represents the loss region -> hangs pieces / blunders won games). So we roll optimal play from
starts across win/draw/loss and collect DENSE trajectory pairs (s_i -> s_j, gap) from every line --
winning lines teach 'reach mate', losing lines teach 'this is a bad future to steer away from',
drawn lines teach the boundary between them.

Output pairs: a_packed/a_meta -> b_packed/b_meta with gap (optimal-play plies) and the White-POV
outcome {+1,0,-1} of the line. The occupancy trainer pins d(F(a)->B(b)) ~ gap.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import encode_meta, encode_packed
from experiments import selfplay_generate as sg
from experiments.gen_stratified_perfect import ALL_MENUS, STRATA_MENUS, optimal_line
from experiments.selfplay_generate import random_endgame_start
from experiments.value_fixed_point import TB


def gen_chunk(task):
    name, n, seed, syzygy_dir, pairs_per, seg_max = task
    sg._ENDGAME_MENUS.update(ALL_MENUS)
    tb = TB(syzygy_dir)
    rng = np.random.default_rng(seed)
    ap, am, bp, bm, gap, outc = [], [], [], [], [], []
    got = tries = 0
    while got < n and tries < n * 300:
        tries += 1
        b = random_endgame_start(rng, name)
        if b is None or b.is_game_over(claim_draw=True):
            continue
        line, winner = optimal_line(b, tb)                     # optimal-vs-optimal to terminal (any outcome)
        if len(line) < 2:
            continue
        L = len(line)
        for _ in range(pairs_per):
            i = int(rng.integers(0, L - 1))
            j = int(min(L - 1, i + 1 + rng.integers(0, seg_max)))   # short-to-medium optimal-play hops
            ap.append(encode_packed(line[i])); am.append(encode_meta(line[i]))
            bp.append(encode_packed(line[j])); bm.append(encode_meta(line[j]))
            gap.append(float(j - i)); outc.append(int(winner))
        got += 1
    tb.close()

    def stk(x, dt): return (np.stack(x).astype(dt) if x else np.zeros((0,), dt))
    return dict(name=name, got=got,
                ap=stk(ap, np.uint64), am=stk(am, np.uint8), bp=stk(bp, np.uint64), bm=stk(bm, np.uint8),
                gap=np.array(gap, np.float32), outc=np.array(outc, np.int8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=5000, help="optimal lines per material")
    ap.add_argument("--pairs-per", type=int, default=8, help="dense trajectory pairs per line")
    ap.add_argument("--seg-max", type=int, default=18, help="max ply-gap of a sampled hop")
    ap.add_argument("--out", default="data/derived/optimal_occupancy.npz")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    names = list(STRATA_MENUS)                                 # <=6 materials (win/draw/loss all occur)
    Wk = max(1, args.workers); t0 = time.time()
    tasks = []
    for name in names:
        base, rem = divmod(args.per, Wk)
        for w in range(Wk):
            nn = base + (1 if w < rem else 0)
            if nn:
                tasks.append((name, nn, args.seed + 1000 * names.index(name) + w, args.syzygy_dir,
                              args.pairs_per, args.seg_max))
    print(f"[stage] optimal-play occupancy: {len(names)} materials, {Wk} workers, {len(tasks)} chunks", flush=True)
    keys = ["ap", "am", "bp", "bm", "gap", "outc"]; agg = {k: [] for k in keys}; done = 0
    with ProcessPoolExecutor(max_workers=Wk) as ex:
        futs = {ex.submit(gen_chunk, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result(); done += r["got"]
            for k in keys:
                if len(r[k]):
                    agg[k].append(r[k])
            print(f"  {r['name']:8s} lines+{r['got']:5d} total {done} ({time.time()-t0:.0f}s)", flush=True)
    for k in keys:
        agg[k] = np.concatenate(agg[k], axis=0) if agg[k] else np.zeros((0,), np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, a_packed=agg["ap"], a_meta=agg["am"], b_packed=agg["bp"],
                        b_meta=agg["bm"], gap=agg["gap"], outcome=agg["outc"])
    o = agg["outc"]
    print(f"[stage] {len(agg['ap'])} occupancy pairs: W={int((o==1).sum())} D={int((o==0).sum())} "
          f"L={int((o==-1).sum())}  ({time.time()-t0:.0f}s)")
    print(f"VERDICT OPT_OCC pairs={len(agg['ap'])} W/D/L={int((o==1).sum())}/{int((o==0).sum())}/"
          f"{int((o==-1).sum())} gap[med={np.median(agg['gap']):.0f} max={agg['gap'].max():.0f}] -> {args.out}")


if __name__ == "__main__":
    main()
