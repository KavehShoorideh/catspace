#!/usr/bin/env python
"""catspace/research/components/planner/approaches/atlas_region_stats/experiments/m3_train_flux_T.py -- train the T flux scorer for the M3 subgoal API and package
its REGION inputs. T(phi, ctx) -> expected mover committor-loss (the M2a-validated fast predictor,
rating context on, clock off -- matches the m3_flux_gate protocol). Trained on EVEN-hash games
only (odd = the gates' held-out). Packaged per region: phi_centroid + region cb/ply means (the
position block at region grain), so the API computes flux(g | any Elo) continuously:
    net_flux(g) = T(g, mover_elo=opponent) - T(g, mover_elo=self)     (their risk minus ours)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from catspace.research.components.planner.approaches.atlas_region_stats.experiments.train_transition_estimator import T                    # noqa: E402
from catspace.research.tools.training_infra.train.scaffold import resolve_device, save_torch_ckpt     # noqa: E402
from catspace.io import paths

D_CTX = 3 + 2   # position block (sharp, cb, ply/100) + rating block (em, eo) -- normalized below


def rating_block(elo_mover, elo_opp):
    return np.stack([(np.asarray(elo_mover, np.float32) - 1500) / 400,
                     (np.asarray(elo_opp, np.float32) - 1500) / 400], 1)


def ctx_of(cb, ply, em, eo):
    sharp = 1.0 - np.abs(2 * np.asarray(cb, np.float32) - 1)
    pos = np.stack([sharp, np.asarray(cb, np.float32),
                    np.asarray(ply, np.float32) / 100.0], 1)
    return np.concatenate([pos, rating_block(em, eo)], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default=paths.derived("transition_data_labeled.npz"))
    ap.add_argument("--table", default=paths.reach("region_table_v1k.npz"))
    ap.add_argument("--out", default=paths.experiment("m3_flux_T"))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = resolve_device("auto"); rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    d = dict(np.load(args.labeled, allow_pickle=True))
    ok = ~np.isnan(d["mover_loss"])
    train = ok & (d["game"].astype(np.int64) % 2 == 0)                  # even games only
    phi = d["phi"][train].astype(np.float32)
    y = d["mover_loss"][train].astype(np.float32)
    ctx = ctx_of(d["committor_before"][train], d["ply"][train],
                 d["elo_mover"][train], d["elo_opp"][train])

    net = T(phi.shape[1], ctx.shape[1]).to(dev)
    P = torch.from_numpy(phi).to(dev); C = torch.from_numpy(ctx).to(dev)
    Y = torch.from_numpy(y).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    for s in range(args.steps):
        b = rng.integers(0, len(y), args.batch)
        loss = ((net(P[b], C[b]) - Y[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    print(f"trained T on {len(y):,} even-game rows, final mse {loss.item():.4f}")

    # region-grain position block from the labeled train half (assigned to the table's regions)
    t = np.load(args.table, allow_pickle=True)
    bank = t["regions"].astype(np.float32); G = len(bank)
    d2 = (phi * phi).sum(1)[:, None] + (bank * bank).sum(1)[None, :] - 2.0 * phi @ bank.T
    region = d2.argmin(1)
    cnt = np.maximum(np.bincount(region, minlength=G), 1)
    cb_mean = np.bincount(region, weights=d["committor_before"][train], minlength=G) / cnt
    ply_mean = np.bincount(region, weights=d["ply"][train], minlength=G) / cnt
    save_torch_ckpt(net, args.out, args.steps, args=args,
                    extra={"cb_mean": cb_mean.astype(np.float32),
                           "ply_mean": ply_mean.astype(np.float32),
                           "d_phi": phi.shape[1], "d_ctx": ctx.shape[1],
                           "table": args.table})
    print(f"packaged region features for {G} regions -> {args.out}_latest.pt")


if __name__ == "__main__":
    main()
