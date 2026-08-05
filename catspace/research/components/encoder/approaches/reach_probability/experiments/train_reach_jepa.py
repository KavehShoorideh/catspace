#!/usr/bin/env python
"""train_reach_jepa.py -- train the POSITIVES-ONLY reachability JEPA (reach_probability).

Objective, in full:

    z_a      = enc(feats_a)                      # online branch, gradients flow
    z_b      = enc_ema(feats_b).detach()         # EMA target branch, no gradient path
    mu, ls   = predict(z_a)                      # the predicted REACHABLE REGION from a
    loss     = reach_region_nll(mu, ls, z_b)     # only observed reachable pairs enter here
             + w_var * vicreg_variance(z_online) # anti-collapse (scale)
             + w_cov * vicreg_covariance(z_online) # anti-collapse (rank)
             + w_l1  * ||predictor.head_in||_1   # Kaveh's sparsity, placed to be interpretable

There is no negative term anywhere, by construction (Kaveh 2026-08-05: "I don't want negatives").
Consequently a constant encoder is a global optimum of the align term alone, so the two VICReg terms
and the EMA target are not regularisers-of-taste -- they are the only things standing between this
run and a silent collapse that would report a beautiful loss. eff_rank is gated every eval for the
same reason: collapse must surface as a logged number, not as suspiciously good NLL.

RAM, NOT MEMMAP (a recorded scar). The trunk-feature file is 12.2 GB against ~12.9 GB free on this
box, and field training here has previously gone 98% disk-bound on exactly this access pattern --
random row gathers of 32 KB each. So --rows loads a contiguous PREFIX of the feature memmap into RAM
once and trains from there. Rows are grouped by game in the source npz, so a prefix is a set of whole
games, and pairs are filtered to that set; splits are assigned per game upstream, so no game
straddles a split and filtering cannot leak.

WHAT IS BEING MEASURED. The headline question is not this model's accuracy -- it is whether the
irreversible stratification of chess appears WITHOUT anything chess-specific being programmed
(Kaveh 2026-08-05: "key point is whether we can get strata without programming anything chess
specific"). Nothing here is told about piece count, captures, or legality. interpret_reach.py asks
afterwards what the learned structure turned out to be.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.src.reach_jepa import ReachJEPA
from catspace.research.components.encoder.approaches.reachability_field.experiments.arch_bakeoff import eff_rank
from catspace.research.tools.training_infra.losses import (
    reach_region_nll, vicreg_variance, vicreg_covariance)
from catspace.research.tools.training_infra.train.scaffold import (
    TrainConfig, resolve_device, standard_train)


def load_subset(feats_path, n_rows, pairs, verbose=True):
    """Load a contiguous prefix of the fp16 feature memmap into RAM; keep only pairs inside it."""
    mm = np.load(feats_path, mmap_mode="r")
    n = min(int(n_rows), mm.shape[0])
    t0 = time.time()
    feats = np.ascontiguousarray(mm[:n])                  # one sequential read, then RAM-resident
    if verbose:
        print(f"[feats] {n:,} rows {feats.shape[1:]} {feats.dtype} "
              f"= {feats.nbytes/2**30:.2f} GB in RAM [{time.time()-t0:.0f}s]", flush=True)
    keep = (pairs["i"] < n) & (pairs["j"] < n)
    return feats, keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=paths.derived("reach_pairs_v1.npz"))
    ap.add_argument("--feats", default=paths.derived("trunk_feats/t1-256x10__field_std_v2.npy"))
    ap.add_argument("--rows", type=int, default=131072, help="feature rows held in RAM")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--adapter-ch", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--w-var", type=float, default=1.0, help="VICReg variance (anti-collapse)")
    ap.add_argument("--w-cov", type=float, default=0.04, help="VICReg covariance (anti-rank-collapse)")
    ap.add_argument("--w-l1", type=float, default=0.0,
                    help="L1 added to the loss (subgradient route; monitoring only -- measured NOT "
                         "to sparsify under Adam, which is why --l1-prox exists)")
    ap.add_argument("--l1-prox", type=float, default=2.0,
                    help="proximal soft-threshold strength on the predictor input layer (ISTA). "
                         "This is what actually drives coordinates to exact zero")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out slice of TRAIN games; the "
                    "calibration split is left untouched so the conformal guarantee stays honest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=paths.experiment("reach_jepa_v1"))
    args = ap.parse_args()

    t0 = time.time()
    dev = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    z = np.load(args.pairs, allow_pickle=True)
    pairs = {k: z[k] for k in ("i", "j", "gap", "game", "split")}
    feats, keep = load_subset(args.feats, args.rows, pairs)
    for k in pairs:
        pairs[k] = pairs[k][keep]

    # Monitoring set = a slice of TRAIN games. The calibration split (1) is deliberately never read
    # here: selecting anything on it would quietly weaken the conformal coverage it later certifies.
    tr_games = np.unique(pairs["game"][pairs["split"] == 0])
    rng.shuffle(tr_games)
    val_games = set(tr_games[:max(1, int(len(tr_games) * args.val_frac))].tolist())
    is_val = np.fromiter((g in val_games for g in pairs["game"]), bool, len(pairs["game"]))
    fit_idx = np.flatnonzero((pairs["split"] == 0) & ~is_val)
    val_idx = np.flatnonzero((pairs["split"] == 0) & is_val)
    print(f"[pairs] fit {len(fit_idx):,} | val {len(val_idx):,} "
          f"(cal/test held back: {(pairs['split']!=0).sum():,})", flush=True)

    net = ReachJEPA(in_ch=feats.shape[1], d=args.d, adapter_ch=args.adapter_ch,
                    hidden=args.hidden, ema_decay=args.ema).to(dev)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)

    def gather(idx):
        """pair indices -> (feats_a, feats_b) on device, float32."""
        fa = torch.from_numpy(feats[pairs["i"][idx]]).to(dev, torch.float32)
        fb = torch.from_numpy(feats[pairs["j"][idx]]).to(dev, torch.float32)
        return fa, fb

    def step_fn(model, step):
        idx = fit_idx[rng.integers(0, len(fit_idx), args.batch)]
        fa, fb = gather(idx)
        z_a = model.encode(fa)
        z_b_online = model.encode(fb)                     # for VICReg only
        with torch.no_grad():
            z_b = model.encode_target(fb)                 # the prediction TARGET
        mu, ls = model.predict(z_a)
        l_nll = reach_region_nll(mu, ls, z_b)
        z_all = torch.cat([z_a, z_b_online], 0)
        l_var = vicreg_variance(z_all)
        l_cov = vicreg_covariance(z_all)
        l_l1 = model.l1_penalty()
        loss = l_nll + args.w_var * l_var + args.w_cov * l_cov + args.w_l1 * l_l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if args.l1_prox > 0:
            model.prox_l1(args.lr * args.l1_prox)         # ISTA step: the thing that actually zeroes
        model.update_target()
        return {"loss": float(loss.detach()), "nll": float(l_nll.detach()),
                "var": float(l_var.detach()), "cov": float(l_cov.detach()),
                "l1": float(l_l1.detach())}

    def gates_fn(model):
        model.eval()
        g = {}
        sub = val_idx[rng.integers(0, len(val_idx), min(8192, len(val_idx)))]
        fa, fb = gather(sub)
        z_a, z_b = model.encode(fa), model.encode_target(fb)
        mu, ls = model.predict(z_a)
        var = torch.exp(2.0 * ls)
        per = (0.5 * ((z_b - mu) ** 2) / var + ls).sum(-1)         # per-pair NLL
        g["val_nll"] = float(per.mean())
        # Stratified at the lc0 history boundary: for gap <= 8 position a is inside b's own input
        # tensor, so that band can be right for reasons that are not reachability. The strata claim
        # is read on the gap > 8 band.
        gp = pairs["gap"][sub]
        for name, m in (("in_hist", gp <= 8), ("out_hist", gp > 8)):
            if m.sum() > 32:
                g[f"val_nll_{name}"] = float(per[torch.from_numpy(m).to(dev)].mean())
        # Collapse gate: eff_rank of the ONLINE embedding (entropy-of-singular-values form, the same
        # one train_iqe_head logs -- deliberately not probe_rank.py's participation ratio).
        g["eff_rank"] = eff_rank(z_a.detach().float().cpu().numpy())
        g["z_std"] = float(z_a.std())
        g["sigma_med"] = float(torch.exp(ls).median())
        # Sparsity as an EXACT count of surviving input coordinates, which only means something
        # because prox_l1 produces true zeros -- a relative threshold would read uniform shrinkage
        # as sparsity and report a number that is purely a convention.
        g["l1_support"] = int(model.input_support().sum())
        model.train()
        return g

    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      eval_every=args.eval_every, experiment="reach_probability",
                      run_name=f"reach_jepa_d{args.d}_l1{args.w_l1}", device=str(dev),
                      extra={"cfg": {"in_ch": int(feats.shape[1]), "d": args.d,
                                     "adapter_ch": args.adapter_ch, "hidden": args.hidden,
                                     "trunk": "t1-256x10", "pairs": args.pairs}})
    last = standard_train(step_fn, net, cfg, args=args, gates_fn=gates_fn)

    print(f"\nVERDICT REACH-JEPA steps={args.steps} "
          f"val_nll={last.get('val_nll', float('nan')):.4f} "
          f"in_hist={last.get('val_nll_in_hist', float('nan')):.4f} "
          f"out_hist={last.get('val_nll_out_hist', float('nan')):.4f} "
          f"eff_rank={last.get('eff_rank', float('nan')):.1f}/{args.d} "
          f"z_std={last.get('z_std', float('nan')):.3f} "
          f"out={args.out}_latest.pt [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
