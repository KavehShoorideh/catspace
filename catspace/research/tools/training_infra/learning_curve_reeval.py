#!/usr/bin/env python
"""experiments/learning_curve_reeval.py -- re-measure the learning curve from the SAVED sweep
checkpoints using the FULL holdout instead of a 4,000-row sample.

Why this exists. The first learning curve was evaluated on a 4,000-row subsample per arm, and the
resulting train/holdout gap swung 0.007 -> 0.027 -> 0.008 -> 0.007 -> 0.024 NON-monotonically
across sizes. A quantity that jumps around by 4x with no relation to the x-axis is measurement
noise of the same magnitude as the effect being measured, which makes the curve unreadable in
either direction -- it cannot support "bias-limited" any more than it can support the opposite.

Nothing needs retraining to fix that: every sweep point saved a checkpoint, so the models are
fixed and only the ESTIMATOR was noisy. This re-evaluates each on:
  * the ENTIRE holdout (~60,000 rows, ~15x the original sample), and
  * an equally large train-side sample,
which cuts the standard error by ~4x and makes the two arms comparable at the same precision.

It also reports a bootstrap 95% CI on every number, so "is this gap real" is answered on the
figure rather than by eye. The split is reconstructed with the SAME seed and the same by-game
logic as training, so the holdout here is exactly the holdout the models never saw.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from catspace.research.tools.training_infra.losses import basin_ce, basin_logp
from catspace.research.tools.embeddings.basin_simplex_chart import load_head

FRAC_RE = re.compile(r"lc_frac(\d+)_step")


def boot_ci(v, n_boot=400, seed=0):
    """Bootstrap 95% CI of the mean. v is a per-row loss/correctness vector."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n_boot, min(len(v), 20000)))
    means = v[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@torch.no_grad()
def evaluate(net, mm, rows, y, device, batch=8192):
    """-> (per-row CE, per-row correct) on the given rows."""
    ce = np.empty(len(rows), np.float32)
    ok = np.empty(len(rows), np.float32)
    T = net.temperature
    for i in range(0, len(rows), batch):
        sl = slice(i, i + batch)
        x = torch.from_numpy(np.asarray(mm[rows[sl]], dtype=np.float32)).to(device)
        d = net.d_poles(net.phi(x))
        lp = basin_logp(d, T)
        yy = torch.from_numpy(y[sl].astype(np.int64)).to(device)
        ce[sl] = (-lp.gather(1, yy.unsqueeze(1)).squeeze(1)).cpu().numpy()
        ok[sl] = (lp.argmax(1) == yy).float().cpu().numpy()
    return ce, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combined", default="data/derived/field_combined_sub600k.npz")
    ap.add_argument("--ckpt-glob", default="artifacts/experiments/lc/lc_frac*_step6000.pt")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1, help="MUST match the sweep's --seed")
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_learning_curve")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    t0 = time.time()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.combined, allow_pickle=True)
    meta = eval(str(z["_meta"][0]))
    mm = np.load(meta["feats"][0], mmap_mode="r")
    game, y = z["game"], z["y"]

    # Reconstruct the EXACT split the sweep used: same seed, same by-game draw, same order.
    rng = np.random.default_rng(args.seed)
    games = np.unique(game)
    val_games = set(rng.choice(games, size=max(1, int(len(games) * args.val_frac)),
                              replace=False).tolist())
    is_val = np.isin(game, list(val_games))
    val_rows = np.flatnonzero(is_val)
    tr_pool = np.flatnonzero(~is_val)
    tr_rows = np.sort(np.random.default_rng(args.seed + 7).choice(
        tr_pool, min(len(val_rows), len(tr_pool)), replace=False))   # equal-size train arm
    print(f"holdout {len(val_rows):,} rows ({len(val_games):,} games) | train arm "
          f"{len(tr_rows):,} rows -- both ~15x the original 4,000-row sample")

    ck = sorted(Path().glob(args.ckpt_glob), key=lambda p: int(FRAC_RE.search(p.name).group(1)))
    n_games_total = len(games)
    rows_out = []
    for c in ck:
        frac = int(FRAC_RE.search(c.name).group(1)) / 1000.0
        net = load_head(str(c), args.device)
        ce_t, ok_t = evaluate(net, mm, tr_rows, y[tr_rows], args.device)
        ce_v, ok_v = evaluate(net, mm, val_rows, y[val_rows], args.device)
        r = dict(frac=frac, train_games=int(n_games_total * (1 - args.val_frac) * frac),
                 tr_ce=float(ce_t.mean()), va_ce=float(ce_v.mean()),
                 tr_acc=float(ok_t.mean()), va_acc=float(ok_v.mean()),
                 tr_ce_ci=boot_ci(ce_t), va_ce_ci=boot_ci(ce_v),
                 tr_acc_ci=boot_ci(ok_t), va_acc_ci=boot_ci(ok_v))
        # Is the gap real? Bootstrap the DIFFERENCE, not the two means separately.
        rgen = np.random.default_rng(3)
        d = [ce_v[rgen.integers(0, len(ce_v), 20000)].mean()
             - ce_t[rgen.integers(0, len(ce_t), 20000)].mean() for _ in range(400)]
        r["gap"] = r["va_ce"] - r["tr_ce"]
        r["gap_ci"] = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
        rows_out.append(r)
        print(f"  frac {frac:.2f} ({r['train_games']:>7,} games): train CE {r['tr_ce']:.4f} | "
              f"holdout CE {r['va_ce']:.4f} | gap {r['gap']:+.4f} "
              f"[{r['gap_ci'][0]:+.4f},{r['gap_ci'][1]:+.4f}] | holdout acc {r['va_acc']:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    json.dump(rows_out, open(f"{args.out_prefix}.json", "w"), indent=1)
    n = [r["train_games"] for r in rows_out]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, (a, b, lab) in zip(axes, [("tr_ce", "va_ce", "basin cross-entropy"),
                                      ("tr_acc", "va_acc", "basin accuracy")]):
        for key, col, mk, nm in [(a, "#2a78d6", "o", "train"), (b, "#e34948", "s", "holdout")]:
            v = np.array([r[key] for r in rows_out])
            lo = np.array([r[key + "_ci"][0] for r in rows_out])
            hi = np.array([r[key + "_ci"][1] for r in rows_out])
            ax.plot(n, v, mk + "-", color=col, lw=2, label=nm)
            ax.fill_between(n, lo, hi, color=col, alpha=0.18, lw=0)
        ax.set_xscale("log"); ax.set_xlabel("training games (log)"); ax.set_ylabel(lab)
        ax.legend(frameon=False, fontsize=9); ax.set_title(lab, color="#1c1b19")
    fig.suptitle("Learning curve, full holdout (~60k rows/arm) with bootstrap 95% CI\n"
                 "bands overlapping = the difference is not resolved at this sample size")
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}.png", dpi=140)

    print("\nLEARNING CURVE (full holdout, 95% CI)")
    print(f"  {'games':>9s} {'train CE':>9s} {'hold CE':>9s} {'gap':>9s} {'gap 95% CI':>22s} {'hold acc':>9s}")
    for r in rows_out:
        print(f"  {r['train_games']:>9,} {r['tr_ce']:>9.4f} {r['va_ce']:>9.4f} {r['gap']:>+9.4f} "
              f"  [{r['gap_ci'][0]:+.4f}, {r['gap_ci'][1]:+.4f}] {r['va_acc']:>9.4f}")
    print(f"wrote {args.out_prefix}.png / .json [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
