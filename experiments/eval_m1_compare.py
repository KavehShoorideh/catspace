#!/usr/bin/env python
"""experiments/eval_m1_compare.py -- the M1 FAIR-COMPARISON table: ClockField v3 (incumbent, own
trained trunk, planes input) vs the frozen-Leela-trunk IQE heads (features input), all on the
IDENTICAL held-out val split of the SAME dataset (field_std_v1, ALL PHASES -- openings included).
v3's historical 0.94 pair-order was measured on the opening-free v1 data; the gate requires the
same protocol, so v3 is re-measured HERE. Metrics: pair-order Spearman, d_mate-vs-DTZ Spearman
(tb-won val rows), eff_rank. Prints ONE verdict table (the DoD artifact for the trunk choice/kill).
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_iqe_head import IQEHead, build_pairs
from experiments.train_clock_field import ClockField
from experiments.arch_bakeoff import eff_rank
from catspace.research.tools.training_infra.train.scaffold import resolve_device
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/field_std_v1.npz")
    ap.add_argument("--v3", default="artifacts/experiments/field_fullgame_v3_final.pt")
    ap.add_argument("--heads", nargs="+", default=[
        "artifacts/experiments/iqe_head_maia1500_latest.pt",
        "artifacts/experiments/iqe_head_maia1900_latest.pt"])
    ap.add_argument("--feats", nargs="+", default=[
        "data/derived/trunk_feats/maia-1500__field_std_v1.npy",
        "data/derived/trunk_feats/maia-1900__field_std_v1.npy"])
    ap.add_argument("--val-frac", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); rng = np.random.default_rng(args.seed)

    z = np.load(args.data)
    dtz = z["dtz"].astype(np.int32); game = z["game"]; ply = z["ply"]
    games = np.unique(game)
    val_games = set(rng.choice(games, size=max(1, int(len(games) * args.val_frac)), replace=False).tolist())
    V_s, V_g, V_d = build_pairs(game, ply, val_games, np.random.default_rng(args.seed + 1))
    is_val = np.array([int(g) in val_games for g in game])
    tb_val = np.flatnonzero((dtz >= 1) & is_val); va_idx = np.flatnonzero(is_val)
    te = rng.integers(0, len(V_s), min(4000, len(V_s)))
    er_rows = va_idx[rng.integers(0, len(va_idx), 3000)]
    print(f"[m1-compare] identical protocol: {len(te)} val pairs | {len(tb_val)} tb-won val | all-phase data", flush=True)

    rows_needed = np.unique(np.concatenate([V_s[te], V_g[te], tb_val, er_rows]))
    print(f"  materializing {len(rows_needed):,} plane rows for v3...", flush=True)
    planes_all = z["planes"]                                 # one 8GB transient load (eval-only)
    sub_planes = planes_all[rows_needed]; del planes_all
    rowpos = {r: i for i, r in enumerate(rows_needed)}

    def v3_metrics():
        p = torch.load(args.v3, map_location=dev, weights_only=False); cfg = p["cfg"]
        net = ClockField(cfg["d"], ch=cfg["ch"], blocks=cfg["blocks"], in_planes=112).to(dev)
        net.load_state_dict(p["state_dict"]); net.eval()
        def fp(idx):
            return torch.from_numpy(sub_planes[[rowpos[r] for r in idx]].astype(np.float32)).to(dev)
        with torch.no_grad():
            dp = np.concatenate([net.d_pair(fp(V_s[te][i:i+1024]), fp(V_g[te][i:i+1024])).cpu().numpy()
                                 for i in range(0, len(te), 1024)])
            po = float(spearmanr(dp, np.expm1(V_d[te])).correlation)
            dm = np.concatenate([net.d_mate(fp(tb_val[i:i+1024])).cpu().numpy() for i in range(0, len(tb_val), 1024)])
            mr = float(spearmanr(dm, dtz[tb_val]).correlation)
            er = float(eff_rank(np.concatenate([net.phi(fp(er_rows[i:i+1024])).cpu().numpy()
                                                for i in range(0, len(er_rows), 1024)])))
        return po, mr, er

    def head_metrics(ckpt, featpath):
        feats = np.load(featpath, mmap_mode="r")
        p = torch.load(ckpt, map_location=dev, weights_only=False); cfg = p["cfg"]
        net = IQEHead(in_ch=cfg["in_ch"], d=cfg["d"], components=cfg["components"],
                      adapter_ch=cfg["adapter_ch"]).to(dev)
        net.load_state_dict(p["state_dict"]); net.eval()
        def fx(idx):
            return torch.from_numpy(np.asarray(feats[idx], dtype=np.float32)).to(dev)
        with torch.no_grad():
            dp = net.d_pair_emb(net.phi(fx(V_s[te])), net.phi(fx(V_g[te]))).cpu().numpy()
            po = float(spearmanr(dp, np.expm1(V_d[te])).correlation)
            dm = net.d_mate_emb(net.phi(fx(tb_val))).cpu().numpy()
            mr = float(spearmanr(dm, dtz[tb_val]).correlation)
            er = float(eff_rank(net.phi(fx(er_rows)).cpu().numpy()))
        return po, mr, er

    print("\n===== M1 FAIR COMPARISON (identical all-phase val protocol) =====")
    print(f"{'model':<28} {'pair-order':>10} {'d_mate rho':>10} {'eff_rank':>9}")
    po, mr, er = v3_metrics()
    print(f"{'ClockField v3 (incumbent)':<28} {po:>+10.3f} {mr:>+10.3f} {er:>9.1f}")
    for ckpt, fp_ in zip(args.heads, args.feats):
        tag = Path(ckpt).stem.replace("_latest", "")
        po, mr, er = head_metrics(ckpt, fp_)
        print(f"{tag:<28} {po:>+10.3f} {mr:>+10.3f} {er:>9.1f}")
    print(f"VERDICT M1-COMPARE done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
