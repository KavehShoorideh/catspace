#!/usr/bin/env python
"""experiments/train_iqe_head.py -- M1: the IQE reachability head over FROZEN Leela-trunk features.
Geometry-first (MILESTONES locked decision 1): NO committor/WDL head -- the trainable object is a
thin adapter + IQE quasimetric over precomputed trunk features (precompute_trunk_features.py,
fp16 memmap; the trunk itself is never touched).

Losses (all tested, experiments/losses.py):
  multi-goal  quasimetric_regression( d(phi_i -> phi_j), log1p(ply_gap) )   same-game pairs
  mate        quasimetric_regression( d(phi -> MATE),   log1p(DTZ) )        tablebase-won anchors
  hinge       wdl_hinge( d_mate, won, log(margin) )                          distance margin (geometry)
  repulsion   relu( margin - log1p(d(phi_s -> phi_perm)) )                  anti-collapse

Gates logged on HELD-OUT val games (same protocol as ClockField v3 for the kill decision):
pair-order Spearman | d_mate-vs-DTZ Spearman | eff_rank(phi_head). Off-distribution d_mate +
opening-sanity run post-train (eval_iqe_field.py). Scaffold-tracked (MLflow + ladders + provenance).
"""
from __future__ import annotations

import argparse, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.nn.iqe import IQE
from experiments.losses import quasimetric_regression, wdl_hinge
from experiments.arch_bakeoff import eff_rank
from catspace.train.scaffold import standard_train, TrainConfig, resolve_device


class IQEHead(nn.Module):
    """thin adapter over frozen trunk features (C,8,8) -> phi (d) + IQE quasimetric + mate anchor."""

    def __init__(self, in_ch: int = 64, d: int = 64, components: int = 16, adapter_ch: int = 32):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_ch, adapter_ch, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(adapter_ch * 64, d))
        self.iqe = IQE(d, components=components)
        self.mate = nn.Parameter(torch.randn(d) * 0.01)

    def phi(self, feats):                                    # (B,C,8,8) -> (B,d)
        return self.adapter(feats)

    def d_pair_emb(self, es, eg):
        return self.iqe(es, eg)

    def d_mate_emb(self, es):
        return self.iqe(es, self.mate.unsqueeze(0).expand(len(es), -1))


def build_pairs(game, ply, games_set, rng, per_game=10):
    rows_by_game = defaultdict(list)
    for i in range(len(game)):
        g = int(game[i])
        if g in games_set:
            rows_by_game[g].append(i)
    S, G, D = [], [], []
    for rows in rows_by_game.values():
        rows = sorted(rows, key=lambda i: ply[i])
        if len(rows) < 2:
            continue
        for _ in range(min(per_game, len(rows))):
            a, b = sorted(rng.integers(0, len(rows), 2))
            if a == b:
                continue
            si, gj = rows[a], rows[b]; delta = int(ply[gj] - ply[si])
            if delta <= 0:
                continue
            S.append(si); G.append(gj); D.append(np.log1p(delta))
    return np.array(S), np.array(G), np.array(D, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats", default="data/derived/trunk_feats/maia-1500__field_std_v1.npy")
    ap.add_argument("--data", default="data/derived/field_std_v1.npz")
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--components", type=int, default=16)
    ap.add_argument("--adapter-ch", type=int, default=32)
    ap.add_argument("--w-multi", type=float, default=1.0); ap.add_argument("--w-mate", type=float, default=1.0)
    ap.add_argument("--w-hinge", type=float, default=0.5); ap.add_argument("--w-repel", type=float, default=1.0)
    ap.add_argument("--repel-margin", type=float, default=4.0); ap.add_argument("--margin", type=float, default=400.0)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--rows", default="", help="rows .npy: train on this game-subset of the data")
    ap.add_argument("--out", default=""); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tag = Path(args.feats).stem.split("__")[0]
    out = args.out or f"artifacts/experiments/iqe_head_{tag}"

    feats = np.load(args.feats, mmap_mode="r")               # (N,C,8,8) fp16, NEVER fully in RAM
    z = np.load(args.data)
    dtz = z["dtz"].astype(np.int32); game = z["game"]; ply = z["ply"]
    if args.rows:
        rows = np.load(args.rows)
        dtz, game, ply = dtz[rows], game[rows], ply[rows]
        fmap = rows if len(feats) != len(rows) else None     # full-size memmap -> map; subset-sized -> direct
    else:
        fmap = None
    N, C = len(dtz), feats.shape[1]
    games = np.unique(game)
    val_games = set(rng.choice(games, size=max(1, int(len(games) * args.val_frac)), replace=False).tolist())
    train_games = set(int(g) for g in games) - val_games
    MG_s, MG_g, MG_d = build_pairs(game, ply, train_games, rng)
    V_s, V_g, V_d = build_pairs(game, ply, val_games, np.random.default_rng(args.seed + 1))
    is_val = np.array([int(g) in val_games for g in game])
    tb_train = np.flatnonzero((dtz >= 1) & ~is_val); tb_val = np.flatnonzero((dtz >= 1) & is_val)
    not_won_train = np.flatnonzero((dtz < 0) & ~is_val)
    va_idx = np.flatnonzero(is_val)
    print(f"[iqe-head:{tag}] N={N:,} C={C} | train pairs {len(MG_s):,} val pairs {len(V_s):,} | "
          f"tb-won train {len(tb_train):,} val {len(tb_val):,} | device {dev}", flush=True)

    net = IQEHead(in_ch=C, d=args.d, components=args.components, adapter_ch=args.adapter_ch).to(dev)
    print(f"  head params: {sum(p.numel() for p in net.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    logM = float(np.log1p(args.margin))
    tgt_mate = torch.from_numpy(np.log1p(np.clip(dtz, 0, None)).astype(np.float32)).to(dev)

    def fx(idx):                                             # memmap rows -> fp32 on device
        ridx = fmap[idx] if fmap is not None else idx
        return torch.from_numpy(np.asarray(feats[ridx], dtype=np.float32)).to(dev)

    def step(_net, s):
        pi = rng.integers(0, len(MG_s), args.batch)
        es = net.phi(fx(MG_s[pi])); eg = net.phi(fx(MG_g[pi]))
        L_multi = quasimetric_regression(net.d_pair_emb(es, eg), torch.from_numpy(MG_d[pi]).to(dev))
        perm = torch.randperm(len(pi), device=dev)
        L_repel = F.relu(args.repel_margin - torch.log1p(net.d_pair_emb(es, eg[perm]).clamp(min=0))).mean()
        hb = args.batch // 4
        bw = tb_train[rng.integers(0, len(tb_train), hb)]
        bn = not_won_train[rng.integers(0, len(not_won_train), hb)]
        e_all = net.phi(fx(np.concatenate([bw, bn])))
        dm = net.d_mate_emb(e_all)
        won = torch.zeros(2 * hb, device=dev); won[:hb] = 1.0
        L_mate = quasimetric_regression(dm[:hb], tgt_mate[bw])
        L_hinge = wdl_hinge(dm, won, logM)
        loss = args.w_multi * L_multi + args.w_repel * L_repel + args.w_mate * L_mate + args.w_hinge * L_hinge
        opt.zero_grad(); loss.backward(); opt.step()
        return {k: float(v.detach()) for k, v in
                {"loss": loss, "multi": L_multi, "repel": L_repel, "mate": L_mate, "hinge": L_hinge}.items()}

    from scipy.stats import spearmanr

    def gates(_net):
        with torch.no_grad():
            te = rng.integers(0, len(V_s), min(4000, len(V_s)))
            dp = net.d_pair_emb(net.phi(fx(V_s[te])), net.phi(fx(V_g[te]))).cpu().numpy()
            pair_order = float(spearmanr(dp, np.expm1(V_d[te])).correlation)
            er = float(eff_rank(net.phi(fx(va_idx[rng.integers(0, len(va_idx), 3000)])).cpu().numpy()))
            if len(tb_val) >= 50:
                dm = net.d_mate_emb(net.phi(fx(tb_val))).cpu().numpy()
                mate_rho = float(spearmanr(dm, dtz[tb_val]).correlation)
            else:
                mate_rho = float("nan")
        return {"pair_order": pair_order, "eff_rank": er, "mate_rho": mate_rho}

    cfg = TrainConfig(out=out, steps=args.steps, ckpt_every=args.ckpt_every, eval_every=args.eval_every,
                      experiment="catspace_m1_iqe_head", run_name=Path(out).name,
                      extra={"cfg": {"in_ch": C, "d": args.d, "components": args.components,
                                     "adapter_ch": args.adapter_ch, "trunk": tag}})
    last = standard_train(step, net, cfg, args=args, gates_fn=gates)
    print(f"VERDICT M1-IQE-HEAD {tag}: pair-order {last.get('pair_order', float('nan')):+.3f} "
          f"(gate >=0.94) | d_mate rho {last.get('mate_rho', float('nan')):+.3f} (gate >=0.81) | "
          f"eff_rank {last.get('eff_rank', float('nan')):.1f} | [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
