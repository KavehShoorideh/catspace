#!/usr/bin/env python
"""
experiments/viz/build_eval_dashboard.py — populate
catspace/viz/templates/eval_dashboard.html: the rigor plots for the --repr
{F,B,FB} eval-head ablation plus the zero-label baseline (see
nn/eval_head.py, experiments/train_eval_heads.py): ROC + AUC, reliability,
per-ply AUC, per-Elo AUC, and normative agreement with the Stockfish winprob.

Acceptance check: at ckpt step 30000 these AUCs should reproduce the
journaled numbers within about +-0.01 (F 0.625, B 0.596, FB 0.636, baseline
0.598) -- if they don't, check device placement and z-normalization first.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from catspace.data.shards import sample_shard_rows
from catspace.io.paths import derived_dir, generated_dir, newest_shard_dir
from catspace.nn.eval_head import load_heads
from catspace.nn.features import elo_bin, winprob_cp
from catspace.nn.fb import load_ckpt, pick_device
from catspace.util import auc
from catspace.viz.build_html import build_html
from catspace.viz.realboard import embed_positions

COLS = ("packed", "meta", "ply", "clock", "eval_cp", "result", "white_elo", "black_elo", "game_id")
MODELS = ["F", "B", "FB", "baseline"]
PLY_BUCKETS = [(0, 20), (20, 40), (40, 70), (70, 10_000)]


def load_rows(shard_dir: Path, picks: list) -> dict:
    by_file: dict = {}
    for name, row in picks:
        by_file.setdefault(name, []).append(row)
    out: dict = {k: [] for k in COLS}
    for name, rows in sorted(by_file.items()):
        npz = np.load(shard_dir / name)
        idx = np.array(sorted(rows))
        for k in COLS:
            out[k].append(npz[k][idx])
    return {k: np.concatenate(v) for k, v in out.items()}


def roc_curve(pos: np.ndarray, neg: np.ndarray, n: int = 101):
    thresholds = np.linspace(1.0, 0.0, n)
    tpr = [float((pos >= t).mean()) if len(pos) else 0.0 for t in thresholds]
    fpr = [float((neg >= t).mean()) if len(neg) else 0.0 for t in thresholds]
    return fpr, tpr


def reliability(score: np.ndarray, outcome: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    mean_pred, mean_emp, counts = [], [], []
    for i in range(n_bins):
        m = (score >= edges[i]) & (score < edges[i + 1] if i < n_bins - 1 else score <= edges[i + 1])
        if m.sum() == 0:
            mean_pred.append(None); mean_emp.append(None); counts.append(0)
            continue
        mean_pred.append(round(float(score[m].mean()), 4))
        mean_emp.append(round(float(outcome[m].mean()), 4))
        counts.append(int(m.sum()))
    return dict(edges=[round(float(e), 3) for e in edges], mean_pred=mean_pred, mean_emp=mean_emp, counts=counts)


def rescale01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / ((hi - lo) or 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    ckpt_path = Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt"
    device = pick_device(args.device)

    t0 = time.time()
    fb, payload = load_ckpt(ckpt_path, device)
    fb.eval()
    step = payload.get("step", "?")
    zdiff = payload["zgoals"]["MATE_DIFF"].numpy().astype(np.float32)
    z = zdiff / np.linalg.norm(zdiff)
    heads = {}
    for repr_, fname in (("F", "eval_heads.pt"), ("B", "eval_heads_B.pt"), ("FB", "eval_heads_FB.pt")):
        desc, norm, meta = load_heads(derived_dir() / fname, device)
        desc.eval(); norm.eval()
        assert meta.get("repr") == repr_, f"{fname} meta repr={meta.get('repr')!r}, expected {repr_!r}"
        heads[repr_] = (desc, norm)
    print(f"load: {time.time() - t0:.1f}s  ckpt step={step}  device={device}")

    t0 = time.time()
    picks = sample_shard_rows(shard_dir, args.n, seed=args.seed, holdout_only=True)
    data = load_rows(shard_dir, picks)
    n = len(data["ply"])
    print(f"sampled {n} holdout rows: {time.time() - t0:.1f}s")

    t0 = time.time()
    F, B = embed_positions(fb, data["packed"], data["meta"], data["white_elo"], data["black_elo"],
                           data["clock"], device)
    F_t, B_t = torch.from_numpy(F).to(device), torch.from_numpy(B).to(device)
    FB_t = torch.cat([F_t, B_t], dim=1)
    reps = {"F": F_t, "B": B_t, "FB": FB_t}

    e_desc, e_norm = {}, {}
    with torch.no_grad():
        for m in ("F", "B", "FB"):
            desc, norm = heads[m]
            e_desc[m] = desc.expected_score(reps[m]).cpu().numpy()
            e_norm[m] = norm.expected_score(reps[m]).cpu().numpy()
    baseline_raw = F @ z
    e_desc["baseline"] = rescale01(baseline_raw)
    e_norm["baseline"] = e_desc["baseline"]     # zero-label has one readout, reused for both panels
    print(f"embed+score {n} rows: {time.time() - t0:.1f}s")

    result = data["result"]
    decisive = result != 0
    welo_bin = elo_bin(data["white_elo"])
    wp = winprob_cp(data["eval_cp"])
    fin = np.isfinite(wp)

    # -------------------------------------------------------------- ROC + AUC
    roc, aucs = {}, {}
    for m in MODELS:
        pos, neg = e_desc[m][decisive & (result == 1)], e_desc[m][decisive & (result == -1)]
        aucs[m] = float(auc(pos, neg))
        fpr, tpr = roc_curve(pos, neg)
        roc[m] = dict(fpr=[round(v, 4) for v in fpr], tpr=[round(v, 4) for v in tpr])
    print("AUC:", {m: round(aucs[m], 3) for m in MODELS})

    # ---------------------------------------------------------- reliability
    rel = {m: reliability(e_desc[m][decisive], (result[decisive] == 1).astype(np.float64)) for m in MODELS}

    # ------------------------------------------------------- per-ply / per-Elo AUC
    ply_auc = {m: [] for m in MODELS}
    for lo, hi in PLY_BUCKETS:
        bucket = decisive & (data["ply"] >= lo) & (data["ply"] < hi)
        for m in MODELS:
            pos, neg = e_desc[m][bucket & (result == 1)], e_desc[m][bucket & (result == -1)]
            ply_auc[m].append(round(float(auc(pos, neg)), 4) if len(pos) and len(neg) else None)

    elo_auc = {m: [] for m in MODELS}
    elo_bins_present = sorted(set(int(b) for b in welo_bin))
    for b in elo_bins_present:
        bucket = decisive & (welo_bin == b)
        for m in MODELS:
            pos, neg = e_desc[m][bucket & (result == 1)], e_desc[m][bucket & (result == -1)]
            elo_auc[m].append(round(float(auc(pos, neg)), 4) if len(pos) and len(neg) else None)

    # --------------------------------------------------------- normative agreement
    spear = {}
    norm_scatter = {}
    rng = np.random.default_rng(args.seed)
    sub = rng.choice(np.flatnonzero(fin), size=min(800, int(fin.sum())), replace=False)
    for m in MODELS:
        spear[m] = float(spearmanr(e_norm[m][fin], wp[fin]).statistic)
        norm_scatter[m] = dict(e_norm=[round(float(v), 3) for v in e_norm[m][sub]],
                               sf=[round(float(v), 3) for v in wp[sub]])
    print("Spearman(e_norm, sf):", {m: round(spear[m], 3) for m in MODELS})

    data_out = dict(
        meta=dict(title=f"catspace — eval calibration & ablation dashboard  ·  ckpt step {step}  ·  n={n}"),
        models=MODELS,
        auc=aucs, roc=roc, reliability=rel,
        ply_buckets=[f"{lo}-{hi if hi < 10_000 else '+'}" for lo, hi in PLY_BUCKETS], ply_auc=ply_auc,
        elo_bins=elo_bins_present, elo_auc=elo_auc,
        spearman=spear, norm_scatter=norm_scatter,
    )

    out = Path(args.out) if args.out else generated_dir() / "eval-dashboard.html"
    template = Path(__file__).resolve().parents[2] / "catspace" / "viz" / "templates" / "eval_dashboard.html"
    build_html(template, data_out, out)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
