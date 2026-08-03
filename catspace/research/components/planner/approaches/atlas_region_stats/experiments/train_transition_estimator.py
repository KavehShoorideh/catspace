#!/usr/bin/env python
"""catspace/research/components/planner/approaches/atlas_region_stats/experiments/train_transition_estimator.py -- M2a: the TRANSITION ESTIMATOR T. An MLP head over the
FROZEN M1 embedding phi(s) + game CONTEXT [clocks, Elos, ply, time-control] -> predicted mover-POV
crossing risk (regressed on the SF-labeled realized self-blunder loss). Locked decision 3: context
enters HERE, not the encoder.

M2a gates (all CI'd via catspace/stats.py, cluster-bootstrapped by game; paired deltas for ablations):
  - T predicts realized crossings (Spearman pred vs SF mover_loss, held-out games)
  - CONTEXT adds signal: paired delta-Spearman for phi-only -> +rating -> +clock (the novel claim)
  - clock sign correct (lower clock -> higher risk); rating sign correct (higher Elo -> lower risk)
Scaffold-tracked (MLflow).
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from catspace.research.tools.training_infra.train.scaffold import standard_train, TrainConfig, resolve_device
from catspace.research.tools.stats_eval.stats import spearman_ci, paired_delta_ci, fmt_ci
from catspace.io import paths


def context_feats(z, use_rating=True, use_clock=True):
    """context vector. ALWAYS-ON position block = sharpness (committor_before near 0.5 = contested =
    blunder-affording); rating/clock blocks carry their SHARPNESS INTERACTIONS (time/skill matter
    mainly in sharp positions -- the 3-way interaction the controlled analysis exposed). Ablations
    zero the rating/clock blocks."""
    base = np.maximum(z["base_s"].astype(np.float32), 1.0)
    cm = z["clk_mover"].astype(np.float32); co = z["clk_opp"].astype(np.float32)
    ply = z["ply"].astype(np.float32)
    em = z["elo_mover"].astype(np.float32); eo = z["elo_opp"].astype(np.float32)
    cb = z["committor_before"].astype(np.float32)
    sharp = 1.0 - np.abs(2 * cb - 1)                    # 1 at c=0.5, 0 at c in {0,1}
    is_blitz = (base <= 240).astype(np.float32)
    pos = np.stack([sharp, cb, ply / 100.0], 1)         # POSITION block (always on)
    cmn = np.clip(cm / base, 0, 3); cslow = np.clip(1 - cm / base, 0, 1)  # fraction time LEFT / USED
    clock = np.stack([np.log1p(cm), cmn, np.clip(co / base, 0, 3), np.log1p(base),
                      cslow * sharp * is_blitz,          # time-pressure x sharp x blitz (the real signal)
                      cmn * sharp], 1)
    rating = np.stack([em / 1000.0, (em - eo) / 1000.0,
                       (em / 1000.0) * sharp], 1)        # skill x sharp
    if not use_clock:
        clock = np.zeros_like(clock)
    if not use_rating:
        rating = np.zeros_like(rating)
    return np.concatenate([pos, rating, clock], 1).astype(np.float32)


from catspace.research.components.planner.approaches.atlas_region_stats.src.transition import T  # component home (refactor 2026-07-30)


def run_variant(phi, ctx, y, games, dev, steps, batch, val_mask, seed, name):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    tr = np.flatnonzero(~val_mask); va = np.flatnonzero(val_mask)
    net = T(phi.shape[1], ctx.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    P = torch.from_numpy(phi).float().to(dev); C = torch.from_numpy(ctx).float().to(dev)
    Y = torch.from_numpy(y).float().to(dev)

    def step(_net, s):
        b = tr[rng.integers(0, len(tr), batch)]
        pred = net(P[b], C[b]); loss = ((pred - Y[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        return {"loss": float(loss)}

    cfg = TrainConfig(out=paths.experiment(f"T_{name}"), steps=steps, ckpt_every=steps,
                      eval_every=steps, experiment="catspace_m2a_T", run_name=name)
    standard_train(step, net, cfg, args=None)
    net.eval()
    with torch.no_grad():
        pv = net(P[va], C[va]).cpu().numpy()
    return pv, va


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=paths.derived("transition_data_labeled.npz"))
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--val-frac", type=float, default=0.15); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); rng = np.random.default_rng(args.seed)
    z = dict(np.load(args.data, allow_pickle=True))
    ok = ~np.isnan(z["mover_loss"])
    for k in z:
        if hasattr(z[k], "__len__") and len(z[k]) == len(ok):
            z[k] = z[k][ok]
    phi = z["phi"].astype(np.float32); y = z["mover_loss"].astype(np.float32)
    game = z["game"]
    games = np.unique(game)
    val_games = set(rng.choice(games, size=max(1, int(len(games) * args.val_frac)), replace=False).tolist())
    val_mask = np.array([int(g) in val_games for g in game])
    print(f"[T] {len(y):,} labeled | val games {len(val_games)} | crossing-rate(>=0.2) {np.mean(y>=0.2):.1%}", flush=True)

    ctx_full = context_feats(z, True, True)
    ctx_rat = context_feats(z, True, False)
    ctx_none = context_feats(z, False, False)
    variants = {}
    for name, ctx in [("phi_only", ctx_none), ("phi_rating", ctx_rat), ("phi_rating_clock", ctx_full)]:
        pv, va = run_variant(phi, ctx, y, game, dev, args.steps, args.batch, val_mask, args.seed, name)
        variants[name] = pv
    yv = y[va]; gv = game[va]

    print("\n===== M2a TRANSITION ESTIMATOR gates (held-out games; cluster-bootstrap CIs) =====")
    for name, pv in variants.items():
        rho, lo, hi = spearman_ci(pv, yv, clusters=gv, n_boot=1000, seed=1)
        print(f"  T[{name:<18}] Spearman(pred, realized crossing) = {fmt_ci(rho, lo, hi)}")
    # paired ablation deltas (the novel-signal claims)
    d1, l1, h1, p1 = paired_delta_ci(variants["phi_rating"], variants["phi_only"], yv, clusters=gv, n_boot=1000, seed=2)
    d2, l2, h2, p2 = paired_delta_ci(variants["phi_rating_clock"], variants["phi_rating"], yv, clusters=gv, n_boot=1000, seed=3)
    print(f"  RATING adds: delta-rho {fmt_ci(d1, l1, h1)}  P(helps)={p1:.2f}")
    print(f"  CLOCK  adds: delta-rho {fmt_ci(d2, l2, h2)}  P(helps)={p2:.2f}")
    # sign checks: correlation of realized crossing with clock / rating (raw, held-out)
    from scipy.stats import spearmanr
    cm = z["clk_mover"][val_mask]; em = z["elo_mover"][val_mask]
    print(f"  SIGN clock: Spearman(clk_mover, realized_loss) = {spearmanr(cm, yv).correlation:+.3f} (expect <0: less time -> more crossings)")
    print(f"  SIGN rating: Spearman(elo_mover, realized_loss) = {spearmanr(em, yv).correlation:+.3f} (expect <0: higher Elo -> fewer)")
    print(f"VERDICT M2a-T done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
