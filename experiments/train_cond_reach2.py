#!/usr/bin/env python
"""experiments/train_cond_reach2.py -- z-conditioned reachability with the PROBABILITY-ADJUSTED target
(Kaveh 2026-07-27): d(s->g) <- log1p( n_moves / P(path s->g | z) ), P(path)=prod P_player(move_i|z) from
the estimator. Forced move (P=1) -> gap exactly 1; unlikely path -> larger; never-taken -> repulsion=inf.
The target is now z-DEPENDENT (via the per-move surprisal), so the FiLM adapter finally has z-signal.

Re-audit points baked in: P clamped away from 0 (surprisal finite); path surprisal summed over the
player's OWN moves along s->g (cumulative, direction s->g); target = logaddexp(0, log(n)+sum_surprisal)
so certain single move -> log1p(1)=ln2 -> gap 1 (never 0). z-lift: correct-z vs base(z=0) vs wrong-z.
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
from catspace.style.model import StyleResidual, VOCAB
from catspace.style.dataio import load_cache
from experiments.losses import quasimetric_regression, reachability_target
from catspace.train.scaffold import resolve_device


class CondAdapter(nn.Module):
    def __init__(self, d_phi=64, d_ctx=19, components=16, hidden=128):
        super().__init__()
        self.film = nn.Sequential(nn.Linear(d_ctx, hidden), nn.ReLU(), nn.Linear(hidden, 2 * d_phi))
        nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
        self.iqe = IQE(d_phi, components=components)

    def phi_c(self, phi, ctx):
        dg, b = self.film(ctx).chunk(2, -1)
        return phi * (1.0 + dg) + b

    def d_pair(self, phi_s, phi_g, ctx):
        return self.iqe(self.phi_c(phi_s, ctx), self.phi_c(phi_g, ctx))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/derived/m2b/cache_dense")
    ap.add_argument("--model", default="artifacts/experiments/m2b_style_3k.pt")
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--per-group", type=int, default=12); ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--adjacent", action="store_true", help="1-step edges only (z-dependent local cost; "
                    "IQE triangle inequality composes multi-step reachability)")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    z = load_cache(args.cache); K = z["cand_idx"].shape[1]
    split = z["split"]; pidx = z["pidx"].astype(np.int64); m = (split == "train") & (pidx >= 0)
    idx = np.flatnonzero(m)
    phi = z["phi"][idx].astype(np.float32); ci = z["cand_idx"][idx].astype(np.int64)
    clp = z["cand_logp"][idx].astype(np.float32); ps = z["played_slot"][idx].astype(np.int64)
    pi = pidx[idx]; elo = z["elo_self"][idx].astype(np.float32); game = z["game_id"][idx]; ply = z["ply"][idx]

    ck = torch.load(args.model, map_location=dev, weights_only=False)
    model = StyleResidual(n_individual=ck["n_individual"], d_z=ck["d_z"], lam_prior=ck["lam"],
                          learn_mu=ck.get("learn_mu", False)).to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
    z_train = model.delta.weight.detach()
    cnt = np.bincount(pi, minlength=z_train.shape[0]).astype(np.float32)
    zunc_p = 1.0 / np.sqrt(np.maximum(cnt, 1.0)); zunc_p /= (zunc_p.max() + 1e-9)

    # --- per-position conditioned surprisal = -log P_player(played move | z) ---
    P = torch.from_numpy(phi).to(dev); CI = torch.from_numpy(ci).to(dev); CLP = torch.from_numpy(clp).to(dev)
    PS = torch.from_numpy(ps).to(dev); mask = CI != VOCAB
    rank = (torch.arange(K).float() / (K - 1)).unsqueeze(0).expand(len(idx), -1).to(dev)
    ZP = z_train[torch.from_numpy(pi).to(dev)]                                    # (N, dz) per-position z
    surpr = np.empty(len(idx), np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), 8192):
            e = slice(s, s + 8192)
            U = model.U_of(P[e], CI[e], CLP[e], rank[e])                         # (b,K,dz)
            style = (U * ZP[e].unsqueeze(1)).sum(-1)
            logit = (CLP[e] + style).masked_fill(~mask[e], -1e9)
            lp = F.log_softmax(logit, -1).gather(1, PS[e].view(-1, 1)).squeeze(1)
            surpr[e] = (-lp).clamp(min=0).cpu().numpy()                          # >=0; P clamped by softmax
    print(f"[cond-reach2] {len(idx):,} pos | surprisal mean {surpr.mean():.3f} med {np.median(surpr):.3f} "
          f"| device {dev} [{time.time()-t0:.0f}s]", flush=True)

    # --- pairs within (player, game), path surprisal via cumulative sum; new target ---
    groups = defaultdict(list)
    for k in range(len(idx)):
        groups[(int(pi[k]), int(game[k]))].append(k)
    games = np.array(sorted(set(int(g) for g in game)))
    vg = set(rng.choice(games, max(1, int(len(games) * args.val_frac)), replace=False).tolist())

    def make_pairs(want_val):
        S, G, T = [], [], []
        for (p, g), rows in groups.items():
            if (g in vg) != want_val or len(rows) < 2:
                continue
            rows = sorted(rows, key=lambda k: ply[k])
            cs = np.concatenate([[0.0], np.cumsum(surpr[rows])])                 # cs[k] = sum surpr rows[0:k]
            if args.adjacent:                                                    # 1-step edges only
                for a in range(len(rows) - 1):
                    if int(ply[rows[a + 1]] - ply[rows[a]]) != 2:
                        continue
                    S.append(rows[a]); G.append(rows[a + 1])
                    T.append(float(reachability_target(1, surpr[rows[a]])))      # log1p(1/P(move|z))
                continue
            for _ in range(min(args.per_group, len(rows))):
                a, b = sorted(rng.integers(0, len(rows), 2))
                if a == b:
                    continue
                if int(ply[rows[b]] - ply[rows[a]]) != 2 * (b - a):            # gap in sampled path -> skip
                    continue
                n_moves = b - a; sp = cs[b] - cs[a]                             # player's moves a..b-1
                S.append(rows[a]); G.append(rows[b])
                T.append(float(reachability_target(n_moves, sp)))               # tested: log1p(n_moves/P(path))
        return np.array(S), np.array(G), np.array(T, np.float32)

    Ss, Gs, Ds = make_pairs(False); Sv, Gv, Dv = make_pairs(True)
    ctx = np.concatenate([z_train.cpu().numpy()[pi], zunc_p[pi][:, None],
                          ((elo - 1500) / 400)[:, None], (ply / 100.0)[:, None]], 1).astype(np.float32)
    Cx = torch.from_numpy(ctx).to(dev)
    print(f"[cond-reach2] train pairs {len(Ss):,} val {len(Sv):,} | ctx {ctx.shape[1]}d | "
          f"target mean {Ds.mean():.2f} (base log1p(n) would be lower) [{time.time()-t0:.0f}s]", flush=True)

    net = CondAdapter(d_phi=phi.shape[1], d_ctx=ctx.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    for st in range(args.steps):
        b = rng.integers(0, len(Ss), args.batch); cs = Cx[Ss[b]]
        L = quasimetric_regression(net.d_pair(P[Ss[b]], P[Gs[b]], cs), torch.from_numpy(Ds[b]).to(dev))
        perm = torch.randperm(len(b), device=dev)
        Lr = F.relu(4.0 - torch.log1p(net.d_pair(P[Ss[b]], P[Gs[b]][perm], cs).clamp(min=0))).mean()
        (L + Lr).backward(); opt.step(); opt.zero_grad()
        if (st + 1) % 2000 == 0:
            print(f"  step {st+1}: multi {float(L):.3f} [{time.time()-t0:.0f}s]", flush=True)

    from scipy.stats import spearmanr
    net.eval(); te = rng.integers(0, len(Sv), min(8000, len(Sv))); tgt = Dv[te]
    cs = Cx[Sv[te]]; base = cs.clone(); base[:, :16] = 0; base[:, 16] = 0
    wrong = Cx[Sv[te][rng.permutation(len(te))]].clone(); wrong[:, 16:] = cs[:, 16:]
    with torch.no_grad():
        dc = net.d_pair(P[Sv[te]], P[Gv[te]], cs).cpu().numpy()
        db = net.d_pair(P[Sv[te]], P[Gv[te]], base).cpu().numpy()
        dw = net.d_pair(P[Sv[te]], P[Gv[te]], wrong).cpu().numpy()
    rc, rb, rw = (spearmanr(x, tgt).correlation for x in (dc, db, dw))
    print(f"\n===== COND-REACH2 gates (prob-adjusted target; held-out games) =====")
    print(f"  pair-order (correct z) = {rc:+.4f}")
    print(f"  Z-LIFT: correct {rc:+.4f} vs base(z=0) {rb:+.4f} vs wrong-z {rw:+.4f}  "
          f"(correct-base {rc-rb:+.4f}, correct-wrong {rc-rw:+.4f})")
    print(f"  map moves w/ z: Spearman(d|correct, d|wrong) = {spearmanr(dc, dw).correlation:+.4f}")
    print(f"VERDICT cond-reach2 done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
