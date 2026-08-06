#!/usr/bin/env python
"""export_training_umap.py -- watch the field ORGANISE: one shared UMAP across training checkpoints.

Kaveh: "let's make it so we can load the checkpoints into the viz and see as they go. each
checkpoint, reinsert into viz."

THE TRAP, and why this is not just a loop over checkpoints. Each checkpoint is a DIFFERENT MODEL,
so its embedding lives in a different space -- z_B at step 500 and z_B at step 20000 are not
comparable coordinates. Fitting a UMAP per checkpoint would produce clouds that jump between frames
for reasons that are purely algorithmic, and the resulting animation would look like dramatic
reorganisation even if nothing had changed. Projecting later checkpoints into a UMAP fitted on an
early one is no better: transform() can only map onto already-fitted structure, so genuinely new
geometry gets squashed onto whatever the early model happened to have.

WHAT THIS DOES INSTEAD: embed the SAME fixed set of positions under every checkpoint, concatenate
all of them, and fit ONE UMAP over the union. Every frame then shares one coordinate system, so a
cloud that moves between steps really moved -- the same co-fitting logic already used for the game
traces and the pole positions. The cost is that the projection is a compromise across all
checkpoints rather than optimal for any one of them, which is the correct trade for an animation.

Poles are embedded per checkpoint too, so you can watch the learned ending poles separate (or fail
to -- take 2's never moved at all, which is exactly the kind of thing this makes visible).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.experiments.plot_strata_figures import (
    embed)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default=paths.experiment("reach_vit_poles_take3"))
    ap.add_argument("--every", type=int, default=2000, help="use checkpoints at multiples of this")
    ap.add_argument("--n-pos", type=int, default=3500, help="positions per checkpoint frame")
    ap.add_argument("--n-term", type=int, default=1200, help="terminal positions included")
    ap.add_argument("--max-ply", type=int, default=210)
    ap.add_argument("--dims", type=int, default=3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=paths.experiment("reach_vit_training_umap.json"))
    args = ap.parse_args()

    t0 = time.time()
    cks = sorted(glob.glob(f"{args.prefix}_step*.pt"),
                 key=lambda f: int(re.search(r"step(\d+)", f).group(1)))
    cks = [f for f in cks if int(re.search(r"step(\d+)", f).group(1)) % args.every == 0]
    if not cks:
        raise SystemExit(f"no checkpoints at multiples of {args.every} under {args.prefix}")
    steps_avail = [int(re.search(r"step(\d+)", f).group(1)) for f in cks]
    print(f"[train-umap] {len(cks)} checkpoints: {steps_avail}", flush=True)

    net0, pay0 = load_net(cks[0], args.device)
    c = pay0["cfg"]
    tr = T.build(n_human=c["games"] // 2, n_sf=c["games"] // 2, seed=c["traj_seed"],
                 max_plies=c["max_plies"], verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    keep = np.flatnonzero(split == 2)
    game, ply, pc = tr.game_of_row(), tr.ply_of_row(), tr.piece_count()
    arrived = np.full(tr.n_positions, -1, np.int8)
    t_rows, t_term = tr.terminal_rows()
    arrived[t_rows] = T.TERM_OUTCOME[t_term]
    rng = np.random.default_rng(0)

    # ONE FIXED SET OF POSITIONS for every frame. Resampling per checkpoint would confound "the
    # field moved" with "different positions were drawn".
    pool = np.flatnonzero(np.isin(game, keep) & (ply <= args.max_ply))
    rows = rng.choice(pool, min(args.n_pos, len(pool)), replace=False)
    tk = t_rows[np.isin(game[t_rows], keep) & (ply[t_rows] <= args.max_ply)]
    if len(tk) > args.n_term:
        tk = rng.choice(tk, args.n_term, replace=False)
    rows = np.unique(np.concatenate([rows, tk]))
    print(f"[train-umap] {len(rows):,} fixed positions x {len(cks)} frames", flush=True)

    frames, blocks, pole_blocks, pole_names = [], [], [], []
    for f in cks:
        net, pay = load_net(f, args.device)
        Z = embed(net, tr, rows, args.device).numpy()
        blocks.append(Z)
        if getattr(net, "poles", None) is not None:
            P = net.poles.poles.detach().float().cpu().numpy()
            pole_blocks.append(P)
            pole_names = c.get("pole_names") or [f"P{k}" for k in range(len(P))]
        frames.append(int(pay["step"]))
        print(f"[train-umap]   step {pay['step']:>6} embedded [{time.time()-t0:.0f}s]", flush=True)

    allZ = np.concatenate(blocks + (pole_blocks if pole_blocks else []))
    import umap
    red = umap.UMAP(n_neighbors=25, min_dist=0.08, n_components=args.dims,
                    metric="euclidean", random_state=0, verbose=False)
    XY = red.fit_transform(allZ)
    lo, hi = XY.min(0), XY.max(0)
    XY = (XY - lo) / np.maximum(hi - lo, 1e-9)
    print(f"[train-umap] ONE shared fit over {len(allZ):,} points [{time.time()-t0:.0f}s]", flush=True)

    n, out_frames, off = len(rows), [], 0
    for k, st in enumerate(frames):
        seg = XY[off:off + n]; off += n
        out_frames.append({"step": st,
                           "x": [round(float(v), 4) for v in seg[:, 0]],
                           "y": [round(float(v), 4) for v in seg[:, 1]],
                           "z": ([round(float(v), 4) for v in seg[:, 2]] if args.dims > 2 else None)})
    pole_frames = []
    for k in range(len(pole_blocks)):
        m = len(pole_blocks[k]); seg = XY[off:off + m]; off += m
        pole_frames.append([{"name": pole_names[i], "x": round(float(seg[i, 0]), 4),
                             "y": round(float(seg[i, 1]), 4),
                             "z": (round(float(seg[i, 2]), 4) if args.dims > 2 else None)}
                            for i in range(m)])

    data = {"dims": args.dims, "n": int(n), "steps": frames, "frames": out_frames,
            "pole_frames": pole_frames or None,
            "ply": [int(v) for v in ply[rows]], "pc": [int(v) for v in pc[rows]],
            "arr": [int(v) for v in arrived[rows]],
            "src": [int(v) for v in np.repeat(tr.source, tr.length)[rows]]}
    with open(args.out, "w") as fh:
        json.dump(data, fh)
    print(f"[train-umap] -> {args.out} ({os.path.getsize(args.out)/2**20:.1f} MB) "
          f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
