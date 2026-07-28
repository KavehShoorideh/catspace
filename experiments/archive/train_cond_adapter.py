#!/usr/bin/env python
"""experiments/train_cond_adapter.py -- the z-CONDITIONED reachability ADAPTER (Kaveh 2026-07-27, decision
3 amendment). ONLY reachability: a zero-init FiLM adapter over the FROZEN base field phi, conditioned on
context = [z, z_unc, Elo, ply] (z = the masked style embedding; NO raw player IDs), trained with the SAME
quasimetric objective as the original IQE head (multi-goal same-game pairs + repulsion; the DTZ mate
anchor folds in with the full data). Result: context-conditioned reachability -- many maps, one per z.

Gates: (a) pair-order reachability holds under conditioning; (b) Z-LIFT -- conditioning on the CORRECT
player's z gives better reachability of that player's trajectories than z=0 (base) or a WRONG player's z;
(c) the map MOVES with z (two players -> different reachability).
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
from experiments.losses import quasimetric_regression
from catspace.train.scaffold import resolve_device


class CondAdapter(nn.Module):
    def __init__(self, d_phi=64, d_ctx=19, components=16, hidden=128):
        super().__init__()
        self.film = nn.Sequential(nn.Linear(d_ctx, hidden), nn.ReLU(), nn.Linear(hidden, 2 * d_phi))
        nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)   # identity at init
        self.iqe = IQE(d_phi, components=components)

    def phi_c(self, phi, ctx):
        dg, b = self.film(ctx).chunk(2, -1)
        return phi * (1.0 + dg) + b

    def d_pair(self, phi_s, phi_g, ctx):
        return self.iqe(self.phi_c(phi_s, ctx), self.phi_c(phi_g, ctx))


def build_pairs(game, ply, rows, rng, per_game=8):
    by_game = defaultdict(list)
    for i in rows:
        by_game[int(game[i])].append(i)
    S, G, D = [], [], []
    for rs in by_game.values():
        rs = sorted(rs, key=lambda i: ply[i])
        if len(rs) < 2:
            continue
        for _ in range(min(per_game, len(rs))):
            a, b = sorted(rng.integers(0, len(rs), 2))
            if a == b or int(ply[rs[b]] - ply[rs[a]]) <= 0:
                continue
            S.append(rs[a]); G.append(rs[b]); D.append(np.log1p(int(ply[rs[b]] - ply[rs[a]])))
    return np.array(S), np.array(G), np.array(D, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/cond_reach_data.npz")
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    d = np.load(args.data)
    phi = d["phi"].astype(np.float32); z = d["z"].astype(np.float32); zunc = d["z_unc"].astype(np.float32)
    elo = d["elo"].astype(np.float32); ply = d["ply"].astype(np.float32); game = d["game"]
    ctx = np.concatenate([z, zunc[:, None], ((elo - 1500) / 400)[:, None], (ply / 100.0)[:, None]], 1).astype(np.float32)
    N = len(phi); games = np.unique(game)
    vg = set(rng.choice(games, max(1, int(len(games) * args.val_frac)), replace=False).tolist())
    vm = np.array([int(g) in vg for g in game])
    tr_rows = np.flatnonzero(~vm); va_rows = np.flatnonzero(vm)
    Ss, Gs, Ds = build_pairs(game, ply, tr_rows, rng)
    Sv, Gv, Dv = build_pairs(game, ply, va_rows, np.random.default_rng(args.seed + 1))
    print(f"[cond-adapter] {N:,} pos | ctx {ctx.shape[1]}d (z16+unc+elo+ply) | train pairs {len(Ss):,} "
          f"val {len(Sv):,} | device {dev} [{time.time()-t0:.0f}s]", flush=True)

    P = torch.from_numpy(phi).to(dev); Cx = torch.from_numpy(ctx).to(dev)
    net = CondAdapter(d_phi=phi.shape[1], d_ctx=ctx.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def step():
        pi = rng.integers(0, len(Ss), args.batch); cs = Cx[Ss[pi]]
        L_multi = quasimetric_regression(net.d_pair(P[Ss[pi]], P[Gs[pi]], cs), torch.from_numpy(Ds[pi]).to(dev))
        perm = torch.randperm(len(pi), device=dev)
        L_repel = F.relu(4.0 - torch.log1p(net.d_pair(P[Ss[pi]], P[Gs[pi]][perm], cs).clamp(min=0))).mean()
        loss = L_multi + L_repel
        opt.zero_grad(); loss.backward(); opt.step()
        return float(loss), float(L_multi)

    for s in range(args.steps):
        lo, lm = step()
        if (s + 1) % 1500 == 0:
            print(f"  step {s+1}: loss {lo:.3f} multi {lm:.3f} [{time.time()-t0:.0f}s]", flush=True)

    from scipy.stats import spearmanr
    net.eval()
    te = rng.integers(0, len(Sv), min(6000, len(Sv)))
    tgt = np.expm1(Dv[te])
    cs = Cx[Sv[te]]
    base_ctx = cs.clone(); base_ctx[:, :16] = 0.0; base_ctx[:, 16] = 0.0        # z=0, z_unc=0 (base map)
    wrong = Cx[Sv[te][rng.permutation(len(te))]].clone()                       # another player's z
    wrong[:, 16:] = cs[:, 16:]                                                 # keep elo/ply, swap only z
    with torch.no_grad():
        d_cor = net.d_pair(P[Sv[te]], P[Gv[te]], cs).cpu().numpy()
        d_base = net.d_pair(P[Sv[te]], P[Gv[te]], base_ctx).cpu().numpy()
        d_wrong = net.d_pair(P[Sv[te]], P[Gv[te]], wrong).cpu().numpy()
    ro_cor = spearmanr(d_cor, tgt).correlation
    ro_base = spearmanr(d_base, tgt).correlation
    ro_wrong = spearmanr(d_wrong, tgt).correlation
    moves = spearmanr(d_cor, d_wrong).correlation

    print(f"\n===== COND-ADAPTER gates (held-out games; reachability only) =====")
    print(f"  (a) pair-order (correct z)   = {ro_cor:+.4f}")
    print(f"  (b) Z-LIFT: correct {ro_cor:+.4f} vs base(z=0) {ro_base:+.4f} vs wrong-z {ro_wrong:+.4f}  "
          f"(dcorrect-base {ro_cor-ro_base:+.4f}, correct-wrong {ro_cor-ro_wrong:+.4f})")
    print(f"  (c) map moves with z: Spearman(d|correct, d|wrong) = {moves:+.4f}  (<1: z changes the map)")
    print(f"VERDICT cond-adapter done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
