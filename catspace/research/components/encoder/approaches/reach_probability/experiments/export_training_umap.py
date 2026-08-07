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
import pathlib
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
    ap.add_argument("--max-frames", type=int, default=8, help="cap on frames (size budget)")
    ap.add_argument("--per-ply", type=int, default=80, help="positions sampled per ply")
    ap.add_argument("--n-cohort", type=int, default=300,
                    help="games followed END-TO-END as a true cohort (all their plies; late-ply "
                         "sparsity in this set is REAL attrition, not a sampling gap)")
    ap.add_argument("--max-ply", type=int, default=210)
    ap.add_argument("--n-term", type=int, default=2000, help="terminal positions included")
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
    # ---- TRUE COHORT (Kaveh: "aren't we following a fixed set of games?" -> a toggle that
    # switches the cloud between the per-ply-balanced sample and a fixed cohort followed
    # end-to-end). Cohort games are excluded from the balanced pool so the two sets are disjoint;
    # in cohort mode the arrived-density IS meaningful and late-ply sparsity is real attrition.
    t_rows_full, t_term_full = tr.terminal_rows()
    arrived = np.full(tr.n_positions, -1, np.int8)
    arrived[t_rows_full] = T.TERM_OUTCOME[t_term_full]    # ALL terminals, so cohort endings colour
    coh_games = []
    for pop in (T.HUMAN, T.SF):
        gs = np.flatnonzero((tr.source == pop) & np.isin(np.arange(len(tr)), keep_games)
                            & (tr.length >= 4))
        coh_games.append(rng.choice(gs, min(args.n_cohort // 2, len(gs)), replace=False))
    coh_games = np.concatenate(coh_games)
    coh_rows = np.concatenate([np.arange(int(tr.start[g]),
                                         int(tr.start[g]) + min(int(tr.length[g]), args.max_ply + 1))
                               for g in coh_games])
    in_cohort_game = np.isin(game, coh_games)

    rows = []
    for p in range(0, args.max_ply + 1):
        cand = np.flatnonzero(in_test & (ply == p) & ~in_cohort_game)
        if len(cand) == 0:
            continue
        h = T.position_hash(tr.tok[cand], tr.glob[cand])
        cand = cand[np.unique(h, return_index=True)[1]]
        take = cand if len(cand) <= args.per_ply else rng.choice(cand, args.per_ply, replace=False)
        rows.append(take)
    rows = np.concatenate(rows)
    t_keep = (np.isin(game[t_rows_full], keep_games) & (ply[t_rows_full] <= args.max_ply)
              & ~in_cohort_game[t_rows_full])
    t_rows = t_rows_full[t_keep]
    if len(t_rows) > args.n_term:
        t_rows = rng.choice(t_rows, args.n_term, replace=False)
    bal_rows = np.unique(np.concatenate([rows, t_rows]))
    # cohort rows may not overlap bal_rows (cohort games were excluded above), so plain concat
    rows = np.concatenate([bal_rows, coh_rows])
    coh_flag = np.zeros(len(rows), np.int8); coh_flag[len(bal_rows):] = 1
    print(f"[train-umap] balanced {len(bal_rows):,} + cohort {len(coh_rows):,} rows "
          f"({len(coh_games)} games end-to-end)", flush=True)

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
    # EMBEDDING CACHE (Kaveh: "checkpoints for the exports might be needed!"). The harness has
    # twice killed this export mid-flight, losing every per-frame embedding (~1-2 min each). Each
    # frame's embeddings now land on disk keyed by (step, n_rows, row-checksum); a rerun loads
    # instead of recomputing, so a killed export resumes at the frame it died on.
    cache_dir = pathlib.Path(str(args.out) + ".embcache")
    cache_dir.mkdir(exist_ok=True)
    row_sig = int(np.bitwise_xor.reduce(fit_rows.astype(np.uint64)) % 10**12)
    # START-GAP DIAGNOSTIC (Kaveh: "the start point embedding ... maps to a different point than
    # START on the map"). The start anchor trains d_IQE(START pole -> ply-0 position) toward
    # log1p(0) = 0 -- and an IQE zero is DOMINATION, not identity: the pole sits coordinatewise
    # behind the start position, not on it. The UMAP is fitted on raw vectors, so pole and ply-0
    # can render apart even at convergence. The honest check is the trained quantity itself,
    # d_IQE in both directions, reported per frame.
    import torch as _t
    ply0 = np.flatnonzero(ply[rows] == 0)
    start_gap, term_gap = [], []
    frames_meta, blocks, pole_blocks, pole_names = [], [], [], []
    for f in cks:
        step_no = int(re.search(r"step(\d+)", f).group(1))
        cpath = cache_dir / f"step{step_no}_n{len(fit_rows)}_sig{row_sig}.npz"
        if cpath.exists():
            _c = np.load(cpath, allow_pickle=False)
            blocks.append(_c["Z"].astype(np.float32))   # fp16 on disk; fp32 in math (norms of ~750-scale coords overflow fp16 intermediates)
            if "P" in _c:
                pole_blocks.append(_c["P"])
                pole_names = c.get("pole_names") or [f"P{k}" for k in range(len(_c["P"]))]
                pd = np.linalg.norm(_c["P"][:, None] - _c["P"][None], axis=-1)
            frames_meta.append(step_no)
            if len(_c.get("sg", [])):
                start_gap.append([float(v) for v in _c["sg"]])
            if "tg" in _c and _c["tg"].size:
                term_gap.append(float(_c["tg"]))
            print(f"[train-umap]   step {step_no:>6} loaded from cache", flush=True)
            continue
        net, pay = load_net(f, args.device)
        Z = embed(net, tr, fit_rows, args.device).numpy()
        # TERMINAL->OWN-POLE convergence (Kaveh: "the pole locations are still far from
        # endings"): median d_IQE from arrived terminals to their outcome pole, trained toward
        # log1p(1) i.e. raw 1. Falling toward ~1 across frames = the anchors are converging;
        # large and flat = they are not.
        if getattr(net, "poles", None) is not None and len(ply0):
            _iqe = net.qhead.iqe if getattr(net, "dual", False) else net.iqe
            _pn = c.get("pole_names") or []
            if "START" in _pn:
                _P = net.poles.poles.detach()[_pn.index("START")][None].float()
                _z0 = _t.from_numpy(Z[ply0[0]][None]).float()
                with _t.no_grad():
                    start_gap.append([round(float(_iqe(_P.to(args.device), _z0.to(args.device))), 3),
                                      round(float(_iqe(_z0.to(args.device), _P.to(args.device))), 3)])
            _tm = np.flatnonzero(arrived[rows] >= 0)
            if len(_tm):
                _tm = _tm[rng.choice(len(_tm), min(400, len(_tm)), replace=False)]
                _pi = [_pn.index(n) for n in ("WIN", "DRAW", "LOSS")]
                _Pall = net.poles.poles.detach().float()[[_pi[o] for o in arrived[rows][_tm]]]
                _zt = _t.from_numpy(Z[_tm]).float()
                with _t.no_grad():
                    term_gap.append(round(float(_iqe(_zt.to(args.device),
                                                     _Pall.to(args.device)).median()), 3))
        if getattr(net, "poles", None) is not None:
            P = net.poles.poles.detach().float().cpu().numpy()
            pole_blocks.append(P)
            pole_names = c.get("pole_names") or [f"P{k}" for k in range(len(P))]
        blocks.append(Z)
        frames_meta.append(int(pay["step"]))
        np.savez_compressed(cpath, Z=Z.astype(np.float16),
                            **({"P": pole_blocks[-1]} if pole_blocks else {}),
                            sg=np.array(start_gap[-1] if start_gap else [], np.float32),
                            tg=np.array(term_gap[-1] if term_gap else np.nan, np.float32))
        print(f"[train-umap]   step {pay['step']:>6} embedded + cached [{time.time()-t0:.0f}s]", flush=True)

    # POLE SEPARATION DIAGNOSTIC (Kaveh: "poles are placed too close to each other compared to
    # the scale of the game"). The UMAP location of the poles is honest but distorted, like any
    # projection; this measures the real thing in the embedding space (same Euclidean proxy the
    # UMAP itself is fitted on): median pole-pole distance over median point-point distance,
    # per frame. If this ratio is small the crowding is REAL geometry, not a projection artefact.
    pole_sep = []
    for k in range(len(pole_blocks)):
        P, Zk = pole_blocks[k], blocks[k]
        pd = np.linalg.norm(P[:, None] - P[None], axis=-1)
        pd = np.median(pd[np.triu_indices(len(P), 1)])
        sub = Zk[rng.choice(len(Zk), min(2000, len(Zk)), replace=False)]
        zd = np.median(np.linalg.norm(sub[:, None][:200] - sub[None], axis=-1))
        pole_sep.append(round(float(pd / max(zd, 1e-9)), 3))
    if pole_sep:
        print(f"[train-umap] pole/point separation ratio per frame: {pole_sep}", flush=True)
    if start_gap:
        print(f"[train-umap] d_IQE(START->ply0), d_IQE(ply0->START) per frame: {start_gap}", flush=True)
    if term_gap:
        print(f"[train-umap] median d_IQE(terminal->own pole) per frame (target ~1): {term_gap}", flush=True)

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
    # 4 decimals: the viewer zooms to 2000x, where 3-decimal coords quantise to a grid
    r3 = lambda v: round(float(v), 4)
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
        "coh": [int(v) for v in coh_flag],
        # ending type of the row's GAME, back-projected to every ply (like `out`): index into the
        # TERMINALS taxonomy, -2 = time forfeit (censored), other negatives = not recorded
        "endt": [int(v) for v in tr.term[game[rows]]],
        "traces": traces, "pole_sep": pole_sep or None, "start_gap": start_gap or None,
        "term_gap": term_gap or None,
    }
    with open(args.out, "w") as fh:
        json.dump(data, fh)
    print(f"[train-umap] -> {args.out} ({os.path.getsize(args.out)/2**20:.1f} MB) "
          f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
