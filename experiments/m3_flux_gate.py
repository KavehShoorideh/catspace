#!/usr/bin/env python
"""experiments/m3_flux_gate.py -- M3 DoD gate: does PREDICTED crossing-flux find where opponents
actually cross? On the SF-labeled M2a data (transition_data_labeled.npz: real positions with
`mover_loss` = the realized SF committor swing = an ACTUAL crossing), we train the fast flux predictor
T(phi, rating-context) on train games and, on HELD-OUT games, check:
  (a) top-decile predicted-flux positions have >= 2x the base crossing rate (mover_loss >= thr),
  (b) for >= 2 rating bands,
  (c) the bands' flux MAPS measurably differ (rating-conditioning is real, not a shared map).
T is the fast atlas predictor; the SF `mover_loss` is the referee-defined ground truth it's judged against.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_transition_estimator import context_feats, T
from catspace.train.scaffold import resolve_device
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/transition_data_labeled.npz")
    ap.add_argument("--thr", type=float, default=0.2, help="mover_loss >= thr counts as a crossing")
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--val-frac", type=float, default=0.3); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); rng = np.random.default_rng(args.seed)

    z = dict(np.load(args.data, allow_pickle=True))
    ok = ~np.isnan(z["mover_loss"])
    for k in list(z):
        if hasattr(z[k], "__len__") and len(z[k]) == len(ok):
            z[k] = z[k][ok]
    phi = z["phi"].astype(np.float32); y = z["mover_loss"].astype(np.float32)
    game = z["game"]; elo = z["elo_mover"].astype(np.int32)
    ctx = context_feats(z, use_rating=True, use_clock=False)
    games = np.unique(game)
    val_games = set(rng.choice(games, int(len(games) * args.val_frac), replace=False).tolist())
    vm = np.array([int(g) in val_games for g in game])
    print(f"[m3-gate] {len(y):,} positions | base crossing rate {np.mean(y >= args.thr):.1%} | "
          f"val games {len(val_games)} [{time.time()-t0:.0f}s]", flush=True)

    P = torch.from_numpy(phi).to(dev); C = torch.from_numpy(ctx).to(dev); Y = torch.from_numpy(y).to(dev)
    tr = np.flatnonzero(~vm); net = T(phi.shape[1], ctx.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    for s in range(args.steps):
        b = tr[rng.integers(0, len(tr), args.batch)]
        loss = ((net(P[b], C[b]) - Y[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    va = np.flatnonzero(vm)
    with torch.no_grad():
        pred = net(P[va], C[va]).cpu().numpy()
    yv = y[va]; ev = elo[va]

    def decile_lift(pred_, actual_):
        base = actual_.mean()
        n = max(1, int(0.1 * len(pred_)))
        top = np.argsort(pred_)[::-1][:n]
        return base, actual_[top].mean(), (actual_[top].mean() / base if base > 0 else np.nan)

    print("\n===== M3 FLUX GATE (held-out games; crossing = mover_loss >= %.2f) =====" % args.thr)
    bands = [("<1500", ev < 1500), (">=1500", ev >= 1500)]
    passes = []
    for name, m in bands:
        cross = (yv[m] >= args.thr).astype(float)
        base, top, lift = decile_lift(pred[m], cross)
        rho = spearmanr(pred[m], yv[m]).correlation
        ok_band = lift >= 2.0
        passes.append(ok_band)
        print(f"  band {name:>7} (n={m.sum():>6,}): base {base:.1%} | top-decile flux {top:.1%} | "
              f"lift {lift:.2f}x  rho {rho:+.3f}  {'PASS' if ok_band else 'fail'}")

    # bands differ: predict flux for the SAME held-out positions under each band's rating context,
    # compare rankings. Low Spearman => the maps genuinely differ by opponent rating.
    zv = {k: z[k][va] for k in z if hasattr(z[k], "__len__") and len(z[k]) == len(vm)}
    maps = {}
    for name, band_elo in (("<1500", 1300), (">=1500", 1800)):
        zb = dict(zv); zb["elo_mover"] = np.full(len(va), band_elo, np.int16)
        cb = context_feats(zb, use_rating=True, use_clock=False)
        with torch.no_grad():
            maps[name] = net(P[va], torch.from_numpy(cb).to(dev)).cpu().numpy()
    rho_maps = spearmanr(maps["<1500"], maps[">=1500"]).correlation
    top10_a = set(np.argsort(maps["<1500"])[::-1][:int(0.1 * len(va))].tolist())
    top10_b = set(np.argsort(maps[">=1500"])[::-1][:int(0.1 * len(va))].tolist())
    jacc = len(top10_a & top10_b) / len(top10_a | top10_b)
    differ = rho_maps < 0.98
    print(f"  BANDS DIFFER: Spearman(flux@1300, flux@1800) = {rho_maps:+.3f} | top-decile Jaccard {jacc:.2f}  "
          f"{'PASS' if differ else 'fail'}")

    gate = all(passes) and differ
    print(f"\nVERDICT M3-flux: {'PASS' if gate else 'PARTIAL/FAIL'} "
          f"(gate = top-decile >=2x in >=2 bands AND bands differ) [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
