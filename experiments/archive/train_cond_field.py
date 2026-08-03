#!/usr/bin/env python
"""experiments/train_cond_field.py -- the CONTEXT-CONDITIONED reachability head (MILESTONES decision 3
AMENDMENT, Kaveh 2026-07-27): d and T MERGE. A zero-init FiLM residual over the FROZEN base field phi,
conditioned on context = [clocks, Elos, (z, z_unc), ply], + THREE OUTCOME-BASIN PROTOTYPES (Win/Draw/
Loss -- M0's three basins). The transition FALLS OUT of the geometry: softmax_b(-d(phi_c, proto_b))
matches the 3-way WDL, so each basin is a reachable region; flux = d_loss - d_win; contested = entropy
over the three. ONE head -> MANY reachability maps (per opponent/clock/rating); zero-init FiLM =>
context=base reproduces the base field.

SHORT RUN: validate the MECHANISM on the M2a SF-labeled data (frozen phi, clocks, Elos, committor,
mover_loss, game, ply) + the trunk 3-way WDL (precompute_trunk_wdl.py) with clock+Elo context. Gates:
(a) pair-order reachability holds; (b) basins fall out -- field P(basin) matches WDL AND flux ranks
committor; (c) transition -- basin-entropy ranks actual SF crossings (mover_loss); (d) map MOVES with
context. z is added for the full run once the z-labeled dataset exists.
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
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE
from experiments.losses import quasimetric_regression
from catspace.research.tools.training_infra.train.scaffold import resolve_device


def context_vec(z, idx):
    base = np.maximum(z["base_s"][idx].astype(np.float32), 1.0)
    cm = z["clk_mover"][idx].astype(np.float32); co = z["clk_opp"][idx].astype(np.float32)
    em = z["elo_mover"][idx].astype(np.float32); eo = z["elo_opp"][idx].astype(np.float32)
    ply = z["ply"][idx].astype(np.float32)
    return np.stack([np.log1p(cm), np.clip(cm / base, 0, 3), np.log1p(co), np.log1p(base),
                     (em - 1500) / 400, (eo - 1500) / 400, (em - eo) / 400, ply / 100.0], 1).astype(np.float32)


class ConditionedField(nn.Module):
    def __init__(self, d_phi=64, d_ctx=8, components=16, hidden=64):
        super().__init__()
        self.film = nn.Sequential(nn.Linear(d_ctx, hidden), nn.ReLU(), nn.Linear(hidden, 2 * d_phi))
        nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)   # gamma=1, beta=0 at init
        self.iqe = IQE(d_phi, components=components)
        self.proto = nn.Parameter(torch.randn(3, d_phi) * 0.1)                     # Win / Draw / Loss
        self.log_tau = nn.Parameter(torch.zeros(()))                              # basin softmax temperature

    def phi_c(self, phi, ctx):
        dg, b = self.film(ctx).chunk(2, -1)
        return phi * (1.0 + dg) + b

    def d_pair(self, phi_s, phi_g, ctx):
        return self.iqe(self.phi_c(phi_s, ctx), self.phi_c(phi_g, ctx))

    def basin(self, phi, ctx):
        """returns (logits (B,3), dist (B,3)) -- distance to each outcome-basin prototype."""
        pc = self.phi_c(phi, ctx)
        d = torch.stack([self.iqe(pc, self.proto[b].unsqueeze(0).expand(len(pc), -1)) for b in range(3)], 1)
        return -d / self.log_tau.exp().clamp(min=0.1), d


def build_pairs(game, ply, games_set, rng, per_game=10):
    rows_by_game = defaultdict(list)
    for i in range(len(game)):
        if int(game[i]) in games_set:
            rows_by_game[int(game[i])].append(i)
    S, G, D = [], [], []
    for rows in rows_by_game.values():
        rows = sorted(rows, key=lambda i: ply[i])
        if len(rows) < 2:
            continue
        for _ in range(min(per_game, len(rows))):
            a, b = sorted(rng.integers(0, len(rows), 2))
            if a == b or int(ply[rows[b]] - ply[rows[a]]) <= 0:
                continue
            S.append(rows[a]); G.append(rows[b]); D.append(np.log1p(int(ply[rows[b]] - ply[rows[a]])))
    return np.array(S), np.array(G), np.array(D, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/transition_data_labeled.npz")
    ap.add_argument("--wdl", default="data/derived/m2a_trunk_wdl.npy")
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--w-basin", type=float, default=1.0); ap.add_argument("--w-repel", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    z = dict(np.load(args.data, allow_pickle=True))
    wdl_all = np.load(args.wdl).astype(np.float32)                                 # (N,3) win/draw/loss
    ok = ~np.isnan(z["mover_loss"]) & ~np.isnan(z["committor_before"])
    for k in list(z):
        if hasattr(z[k], "__len__") and len(z[k]) == len(ok):
            z[k] = z[k][ok]
    wdl = wdl_all[ok]; wdl = wdl / wdl.sum(1, keepdims=True).clip(1e-6)
    phi = z["phi"].astype(np.float32); cb = z["committor_before"].astype(np.float32)
    ml = z["mover_loss"].astype(np.float32); game = z["game"]; ply = z["ply"]
    ctx = context_vec(z, np.arange(len(phi)))
    games = np.unique(game)
    val_games = set(rng.choice(games, max(1, int(len(games) * args.val_frac)), replace=False).tolist())
    tr_games = set(int(g) for g in games) - val_games
    Ss, Gs, Ds = build_pairs(game, ply, tr_games, rng)
    Sv, Gv, Dv = build_pairs(game, ply, val_games, np.random.default_rng(args.seed + 1))
    vm = np.array([int(g) in val_games for g in game]); tr = np.flatnonzero(~vm); va = np.flatnonzero(vm)
    bc = np.bincount(wdl.argmax(1), minlength=3)
    print(f"[cond-field] {len(phi):,} pos | train pairs {len(Ss):,} val {len(Sv):,} | basin mix "
          f"W/D/L {bc[0]}/{bc[1]}/{bc[2]} | device {dev} [{time.time()-t0:.0f}s]", flush=True)

    P = torch.from_numpy(phi).to(dev); Cx = torch.from_numpy(ctx).to(dev); W = torch.from_numpy(wdl).to(dev)
    net = ConditionedField(d_phi=phi.shape[1], d_ctx=ctx.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def step():
        pi = rng.integers(0, len(Ss), args.batch); cs = Cx[Ss[pi]]
        L_pair = quasimetric_regression(net.d_pair(P[Ss[pi]], P[Gs[pi]], cs), torch.from_numpy(Ds[pi]).to(dev))
        perm = torch.randperm(len(pi), device=dev)
        L_repel = F.relu(4.0 - torch.log1p(net.d_pair(P[Ss[pi]], P[Gs[pi]][perm], cs).clamp(min=0))).mean()
        ai = tr[rng.integers(0, len(tr), args.batch)]
        logits, _ = net.basin(P[ai], Cx[ai])
        L_basin = -(W[ai] * F.log_softmax(logits, -1)).sum(-1).mean()              # cross-entropy to WDL
        loss = L_pair + args.w_basin * L_basin + args.w_repel * L_repel
        opt.zero_grad(); loss.backward(); opt.step()
        return float(loss), float(L_pair), float(L_basin)

    for s in range(args.steps):
        lo, lp, lb = step()
        if (s + 1) % 1000 == 0:
            print(f"  step {s+1}: loss {lo:.3f} pair {lp:.3f} basin {lb:.3f} [{time.time()-t0:.0f}s]", flush=True)

    from scipy.stats import spearmanr
    net.eval()
    with torch.no_grad():
        te = rng.integers(0, len(Sv), min(4000, len(Sv)))
        dp = net.d_pair(P[Sv[te]], P[Gv[te]], Cx[Sv[te]]).cpu().numpy()
        pair_order = spearmanr(dp, np.expm1(Dv[te])).correlation
        logits, dist = net.basin(P[va], Cx[va]); pb = F.softmax(logits, -1).cpu().numpy()
        wv = wdl[va]
        basin_match = spearmanr(pb[:, 0], wv[:, 0]).correlation                    # field P(win) vs WDL P(win)
        flux = (dist[:, 2] - dist[:, 0]).cpu().numpy()                             # d_loss - d_win
        flux_rho = spearmanr(flux, cb[va]).correlation                            # ranks committor
        ent = -(pb * np.log(pb + 1e-9)).sum(1)                                     # basin entropy = contested
        cross_rho = spearmanr(ent, ml[va]).correlation                           # contested ranks SF crossings
        cw = ctx[va].copy(); cw[:, 4] = (1100 - 1500) / 400; cs = ctx[va].copy(); cs[:, 4] = (1900 - 1500) / 400
        fw = (net.basin(P[va], torch.from_numpy(cw).to(dev))[1][:, 2]
              - net.basin(P[va], torch.from_numpy(cw).to(dev))[1][:, 0]).cpu().numpy()
        fs = (net.basin(P[va], torch.from_numpy(cs).to(dev))[1][:, 2]
              - net.basin(P[va], torch.from_numpy(cs).to(dev))[1][:, 0]).cpu().numpy()
        moves_rho = spearmanr(fw, fs).correlation

    print(f"\n===== COND-FIELD short-run gates (held-out games; 3 basins W/D/L) =====")
    print(f"  (a) pair-order reachability            = {pair_order:+.3f}  (geometry intact)")
    print(f"  (b1) field P(win) vs trunk WDL P(win)  = {basin_match:+.3f}  (basins learned)")
    print(f"  (b2) flux(d_loss-d_win) vs committor   = {flux_rho:+.3f}  (basins fall out)")
    print(f"  (c) basin-entropy vs SF crossing       = {cross_rho:+.3f}  (contested => crossings)")
    print(f"  (d) map moves w/ ctx Spearman(1100,1900)= {moves_rho:+.3f}  (<1: context matters)")
    print(f"VERDICT cond-field short-run done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
