#!/usr/bin/env python
"""experiments/diag_transfer.py -- did completing human trajectories into the endgame make the field's
LONG-RANGE quasimetric discriminative there? (Kaveh 2026-07-21 transfer thread.)

The blocker (JOURNAL 17:05): on a middlegame-only field, from a KRRvKBP start the quasimetric distance to
every endgame basin is identical to within 0.03 (reach std), so unsupervised subgoal SELECTION is degenerate
(1 distinct winning basin across all starts). This compares that basin-discrimination signal across fields --
e.g. control (lichess only) vs treatment (lichess + endgame-completed continuations). If the treatment's
reach std >> control's and it picks many distinct basins, transfer worked: the endgame metric is no longer
flat, and field-subgoal navigation is unblocked.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import encode_meta, encode_packed
from experiments.conversion_field_subgoal import FieldSubgoalPlanner


def diagnose(field, data, syzygy, dtm_npz, fixed_set, bank, n_basins, n_starts, seed):
    dev = "cpu"
    sg = FieldSubgoalPlanner(field, data, syzygy, 5, 3, 0, device=dev, seed=seed)
    dz = np.load(dtm_npz)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dz["packed"]))[:bank]
    sg.setup_subgoals(dz["packed"][idx], dz["meta"][idx], 1.0, 4, n_basins)
    bd = sg.basin_dmate.cpu().numpy()
    fens = json.loads(Path(fixed_set).read_text())["fens"][:n_starts]
    winners, spreads = [], []
    with torch.no_grad():
        for f in fens:
            b = chess.Board(f)
            fF = sg._embF(encode_packed(b)[None], encode_meta(b)[None])
            reach = torch.stack([sg.fb.distance_matrix(fF, mem)[0].min() for mem, _ in sg.basins]).cpu().numpy()
            comp = reach + 1.0 * bd
            winners.append(int(comp.argmin())); spreads.append(float(reach.std()))
    return dict(basins=len(sg.basins), reach_std=float(np.median(spreads)), dmate_std=float(bd.std()),
                distinct=len(set(winners)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields", nargs="+", required=True, help="ckpts to compare (label=path or just path)")
    ap.add_argument("--data", default="data/derived/stratified_perfect.npz")
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_test_n200.json")
    ap.add_argument("--syzygy", default="data/syzygy")
    ap.add_argument("--bank", type=int, default=1200)
    ap.add_argument("--n-basins", type=int, default=40)
    ap.add_argument("--n-starts", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    print("VERDICT DIAG_TRANSFER (endgame basin discrimination; reach_std ~0 => flat => selection degenerate)")
    print(f"  {'field':22s} | {'basins':>6s} | {'reach_std':>9s} {'dmate_std':>9s} | {'distinct/'+str(args.n_starts):>10s}")
    for spec in args.fields:
        label, path = (spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec))
        r = diagnose(path, args.data, args.syzygy, args.dtm_npz, args.fixed_set,
                     args.bank, args.n_basins, args.n_starts, args.seed)
        flag = "DISCRIMINATES" if r["reach_std"] > 0.2 * r["dmate_std"] and r["distinct"] > 1 else "flat/degenerate"
        print(f"  {label:22s} | {r['basins']:>6d} | {r['reach_std']:>9.3f} {r['dmate_std']:>9.3f} | "
              f"{r['distinct']:>10d}  {flag}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
