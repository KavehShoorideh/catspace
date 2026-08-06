#!/usr/bin/env python
"""export_training_umap.py -- watch the field ORGANISE: one shared UMAP across training
checkpoints, with the FULL ply-viewer schema (phase, castling, outcome, traces + PGN, poles).

Kaveh: "let's make it so we can load the checkpoints into the viz and see as they go. each
checkpoint, reinsert into viz" -- and, on the first standalone version: "the viz changed a bit; i
liked it before ... merge into the old viewer". So this export feeds the SAME viewer template as
the ply cross-sections, extended with a training-step slider; everything the old page could colour
and follow is exported here too, once per checkpoint frame where it varies (coordinates, poles) and
once total where it does not (ply, piece count, castling, phase, outcome, SAN).

THE TRAP, unchanged from the first version: each checkpoint is a DIFFERENT MODEL, so its embedding
lives in a different space. A UMAP per checkpoint would make clouds jump between frames for purely
algorithmic reasons; transform() into an early fit can only map onto already-fitted structure. So
the SAME fixed set of positions (and traces, and poles) is embedded under EVERY checkpoint and one
UMAP is fitted over the union: one coordinate system, so motion between frames is real motion.
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
    ap.add_argument("--prefix", default=paths.experiment("reach_lj_nowall"))
    ap.add_argument("--every", type=int, default=2500, help="use checkpoints at multiples of this")
    ap.add_argument("--max-frames", type=int, default=10, help="cap on frames (size budget)")
    ap.add_argument("--per-ply", type=int, default=100, help="positions sampled per ply")
    ap.add_argument("--max-ply", type=int, default=210)
    ap.add_argument("--n-term", type=int, default=2500, help="terminal positions included")
    ap.add_argument("--n-trace", type=int, default=30, help="games traced per population")
    ap.add_argument("--dims", type=int, default=3)
    ap.add_argument("--neighbors", type=int, default=25)
    ap.add_argument("--min-dist", type=float, default=0.08)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=paths.experiment("reach_training_umap.json"))
    args = ap.parse_args()

    t0 = time.time()
    cks = sorted(glob.glob(f"{args.prefix}_step*.pt"),
                 key=lambda f: int(re.search(r"step(\d+)", f).group(1)))
    cks = [f for f in cks if int(re.search(r"step(\d+)", f).group(1)) % args.every == 0]
    if len(cks) > args.max_frames:                 # keep the LAST frames -- the interesting end
        cks = cks[-args.max_frames:]
    if not cks:
        raise SystemExit(f"no checkpoints at multiples of {args.every} under {args.prefix}")
    steps_avail = [int(re.search(r"step(\d+)", f).group(1)) for f in cks]
    print(f"[train-umap] {len(cks)} checkpoint frames: {steps_avail}", flush=True)

    net0, pay0 = load_net(cks[0], args.device)
    c = pay0["cfg"]
    tr = T.build(n_human=c["games"] // 2, n_sf=c["games"] // 2, seed=c["traj_seed"],
                 max_plies=c["max_plies"], verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    keep_games = np.flatnonzero(split == 2)
    game, ply, pc = tr.game_of_row(), tr.ply_of_row(), tr.piece_count()
    outcome = tr.outcome_of_row()
    src = np.repeat(tr.source, tr.length)
    in_test = np.isin(game, keep_games)
    rng = np.random.default_rng(0)

    # ---- ONE FIXED SAMPLE for every frame (resampling per checkpoint would confound "the field
    # moved" with "different positions were drawn"), balanced by ply, deduplicated by position
    # identity -- ply 0 has ONE unique position and ply 1 twenty, and duplicate draws rendered as
    # a misleading scatter of specks in the first ply export.
    rows = []
    for p in range(0, args.max_ply + 1):
        cand = np.flatnonzero(in_test & (ply == p))
        if len(cand) == 0:
            continue
        h = T.position_hash(tr.tok[cand], tr.glob[cand])
        cand = cand[np.unique(h, return_index=True)[1]]
        take = cand if len(cand) <= args.per_ply else rng.choice(cand, args.per_ply, replace=False)
        rows.append(take)
    rows = np.concatenate(rows)
    t_rows, t_term = tr.terminal_rows()
    t_keep = np.isin(game[t_rows], keep_games) & (ply[t_rows] <= args.max_ply)
    t_rows, t_term = t_rows[t_keep], t_term[t_keep]
    if len(t_rows) > args.n_term:
        pick = rng.choice(len(t_rows), args.n_term, replace=False)
        t_rows, t_term = t_rows[pick], t_term[pick]
    rows = np.unique(np.concatenate([rows, t_rows]))
    arrived = np.full(tr.n_positions, -1, np.int8)
    arrived[t_rows] = T.TERM_OUTCOME[t_term]

    # ---- traces + PGN, same fixed games in every frame ----------------------------------------
    import chess
    uci_by_id = {}
    for g in T.load_human_games(c["games"] // 2, 0, None, 400):
        uci_by_id[(T.HUMAN, g[0])] = g[2]
    for g in T.load_sf_games(c["games"] // 2, 0, None, 400):
        uci_by_id[(T.SF, g[0])] = g[2]

    def san_of(gi):
        ucis = uci_by_id.get((int(tr.source[gi]), int(tr.game_id[gi])))
        if not ucis:
            return []
        b, out = chess.Board(), []
        for u in ucis:
            try:
                mv = chess.Move.from_uci(u)
                out.append(b.san(mv)); b.push(mv)
            except Exception:
                break
        return out

    trace_rows, trace_meta = [], []
    for pop in (T.HUMAN, T.SF):
        gs = np.flatnonzero((tr.source == pop) & np.isin(np.arange(len(tr)), keep_games))
        if not len(gs):
            continue
        for gi in rng.choice(gs, min(args.n_trace, len(gs)), replace=False):
            st, ln = int(tr.start[gi]), int(tr.length[gi])
            if ln < 4:
                continue
            gr = np.arange(st, st + min(ln, args.max_ply + 1))
            trace_rows.append(gr)
            trace_meta.append((int(pop), int(ply[gr[0]]), int(tr.term[gi]), len(gr), san_of(gi)))
    all_trace = np.concatenate(trace_rows) if trace_rows else np.zeros(0, np.int64)
    fit_rows = np.concatenate([rows, all_trace])
    print(f"[train-umap] {len(rows):,} cross-section + {len(all_trace):,} trace positions "
          f"x {len(cks)} frames", flush=True)

    # ---- embed the SAME rows under EVERY checkpoint -------------------------------------------
    frames_meta, blocks, pole_blocks, pole_names = [], [], [], []
    for f in cks:
        net, pay = load_net(f, args.device)
        Z = embed(net, tr, fit_rows, args.device).numpy()
        if getattr(net, "poles", None) is not None:
            P = net.poles.poles.detach().float().cpu().numpy()
            pole_blocks.append(P)
            pole_names = c.get("pole_names") or [f"P{k}" for k in range(len(P))]
        blocks.append(Z)
        frames_meta.append(int(pay["step"]))
        print(f"[train-umap]   step {pay['step']:>6} embedded [{time.time()-t0:.0f}s]", flush=True)

    allZ = np.concatenate(blocks + (pole_blocks if pole_blocks else []))
    import umap
    red = umap.UMAP(n_neighbors=args.neighbors, min_dist=args.min_dist, n_components=args.dims,
                    metric="euclidean", random_state=0, verbose=False)
    XY = red.fit_transform(allZ)
    lo, hi = XY.min(0), XY.max(0)
    XY = (XY - lo) / np.maximum(hi - lo, 1e-9)
    print(f"[train-umap] ONE shared fit over {len(allZ):,} points [{time.time()-t0:.0f}s]", flush=True)

    # ---- static attributes (identical across frames) ------------------------------------------
    W = {2: 1, 3: 1, 4: 2, 5: 4, 8: 1, 9: 1, 10: 2, 11: 4}
    phase_val = np.zeros(tr.n_positions, np.int16)
    for pid, wt in W.items():
        phase_val += wt * (tr.tok == pid).sum(1).astype(np.int16)
    phase = np.where(phase_val <= 10, 2,
                     np.where((ply <= 24) & (phase_val >= 20), 0, 1)).astype(np.int8)
    castle = ((tr.glob[:, 1] > 0) * 8 + (tr.glob[:, 2] > 0) * 4
              + (tr.glob[:, 3] > 0) * 2 + (tr.glob[:, 4] > 0)).astype(np.int16)

    # ---- unpack per-frame coordinate blocks ----------------------------------------------------
    n_fit, n_cross = len(fit_rows), len(rows)
    r3 = lambda v: round(float(v), 3)
    out_frames, off = [], 0
    n_poles = len(pole_blocks[0]) if pole_blocks else 0
    trace_frames = []
    for k, st in enumerate(frames_meta):
        seg = XY[off:off + n_fit]; off += n_fit
        cross, tseg = seg[:n_cross], seg[n_cross:]
        fr = {"step": st,
              "x": [r3(v) for v in cross[:, 0]], "y": [r3(v) for v in cross[:, 1]],
              "z": [r3(v) for v in cross[:, 2]] if args.dims > 2 else None}
        out_frames.append(fr)
        # traces for this frame, sliced per game
        tt, o2 = [], 0
        for (_pop, _p0, _end, n, _san) in trace_meta:
            s2 = tseg[o2:o2 + n]; o2 += n
            tt.append({"x": [r3(v) for v in s2[:, 0]], "y": [r3(v) for v in s2[:, 1]],
                       "z": [r3(v) for v in s2[:, 2]] if args.dims > 2 else None})
        trace_frames.append(tt)
    pole_frames = []
    for k in range(len(pole_blocks)):
        seg = XY[off:off + n_poles]; off += n_poles
        pole_frames.append([{"name": pole_names[i], "x": r3(seg[i, 0]), "y": r3(seg[i, 1]),
                             "z": r3(seg[i, 2]) if args.dims > 2 else None}
                            for i in range(n_poles)])

    traces = [{"pop": pop, "p0": p0, "end": end, "san": san}
              for (pop, p0, end, _n, san) in trace_meta]
    data = {
        "kind": "training",                      # tells the viewer to show the step slider
        "dims": int(args.dims), "n": int(n_cross), "steps": frames_meta,
        "frames": out_frames, "trace_frames": trace_frames, "pole_frames": pole_frames or None,
        "ply": [int(v) for v in ply[rows]], "pc": [int(v) for v in pc[rows]],
        "src": [int(v) for v in src[rows]], "out": [int(v) for v in outcome[rows]],
        "arr": [int(v) for v in arrived[rows]], "cas": [int(v) for v in castle[rows]],
        "ph": [int(v) for v in phase[rows]], "phv": [int(v) for v in phase_val[rows]],
        "traces": traces,
    }
    with open(args.out, "w") as fh:
        json.dump(data, fh)
    print(f"[train-umap] -> {args.out} ({os.path.getsize(args.out)/2**20:.1f} MB) "
          f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
