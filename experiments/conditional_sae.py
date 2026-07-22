#!/usr/bin/env python
"""experiments/conditional_sae.py -- the FULL covariate-conditioned sparse autoencoder (Kaposi
2026-07-21). Concepts are conditional: a structural feature matters in one context and not another
(passed pawn: +0.06 opening -> +0.20 endgame). A GLOBAL SAE averages those away. So condition the
discovery on a context vector c and let each atom carry its DOMAIN OF APPLICABILITY.

Architecture -- FiLM-gated SAE:
    code_k = ReLU(W_e F)_k  *  sigmoid(film(c))_k          # atom k fires only where its domain-gate is open
    F_hat  = W_d code + b                                   # sparse reconstruction (L1 on code)
The gate is a small net of c, so each atom = (structure direction W_d[:,k], native heatmap) x
(DOMAIN = how its gate depends on c). c = phase(piece_count) + distance-to-end + openness(open files)
+ advantage(eval_cp) + distance to each B-cluster ANCHOR (the geometric "distance to cluster in B"
conditioning). Named features are a post-hoc mirror only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device
from experiments.concept_features import features as named_features   # MIRROR ONLY
from experiments.conditional_concepts import openness


class CondSAE(nn.Module):
    def __init__(self, d, m, c_dim):
        super().__init__()
        self.enc = nn.Linear(d, m)
        self.dec = nn.Linear(m, d)
        self.b_pre = nn.Parameter(torch.zeros(d))
        self.film = nn.Sequential(nn.Linear(c_dim, 64), nn.ReLU(), nn.Linear(64, m))
        self.film[-1].bias.data.fill_(2.0)                 # gates start ~open (sigmoid(2)=0.88)

    def forward(self, x, c):
        gate = torch.sigmoid(self.film(c))                 # (b, m) domain gate
        code = torch.relu(self.enc(x - self.b_pre)) * gate
        return self.dec(code) + self.b_pre, code, gate

    @torch.no_grad()
    def normalize_atoms(self):
        w = self.dec.weight
        self.dec.weight.copy_(w / (w.norm(dim=0, keepdim=True) + 1e-8))


def heatmap(Pk, Mk, order, top=40):
    sym = {}
    for i in order[:top]:
        for sq, p in board_from_packed(Pk[i], Mk[i]).piece_map().items():
            sym[(p.symbol(), sq)] = sym.get((p.symbol(), sq), 0) + 1
    return " ".join(f"{s}{chess.square_name(sq)}:{100*c//top}%"
                    for (s, sq), c in sorted(sym.items(), key=lambda kv: -kv[1])[:6])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="data/derived/sep/lichess_gn_iqeqrl_full.pt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--dict", type=int, default=128)
    ap.add_argument("--anchors", type=int, default=6, help="B-cluster anchors for distance-conditioning")
    ap.add_argument("--l1", type=float, default=8e-3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.shard)
    P, M, ply = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["ply"]).astype(int)
    gid = np.asarray(nz["game_id"]); ev = np.asarray(nz["eval_cp"]).astype(np.float32)
    nrows = len(P)
    change = np.flatnonzero(np.diff(gid)) + 1
    last_of = np.repeat(np.concatenate([change, [nrows]]) - 1,
                        np.diff(np.concatenate([[0], change, [nrows]])))
    okr = np.flatnonzero((ply >= args.min_ply) & np.isfinite(ev) & (np.abs(ev) < 2000))
    idx = okr[rng.permutation(len(okr))[:args.n]]
    Pk, Mk = P[idx], M[idx]
    boards = [board_from_packed(Pk[i], Mk[i]) for i in range(len(Pk))]

    with torch.no_grad():
        t = torch.from_numpy(feature_planes(Pk, Mk)).to(dev)
        F = fb.embed_F(t, torch.from_numpy(np.tile(om, (len(Pk), 1))).to(dev))
        B = fb.embed_B(t)
        km = KMeans(n_clusters=args.anchors, n_init=5, random_state=args.seed).fit(B.cpu().numpy())
        Bc = torch.from_numpy(km.cluster_centers_).float().to(dev)
        d_anchor = fb.distance_matrix(F, Bc).cpu().numpy()          # (n, anchors) -- distance to B-clusters
        F = F.cpu().numpy()

    # ---- context vector c ----
    pc = np.array([len(b.piece_map()) for b in boards], float)
    opn = np.array([openness(b) for b in boards], float)
    d2e = (ply[last_of[idx]] - ply[idx]).astype(float)              # plies to game end (~distance to terminal)
    adv = np.clip(ev[idx], -1000, 1000)
    cov_names = ["phase(pc)", "dist_to_end", "openness", "advantage"] + [f"d_anchor{k}" for k in range(args.anchors)]
    C = np.column_stack([pc, d2e, opn, adv, d_anchor])
    # ORTHOGONALIZE: partial phase (col 0) out of every other covariate -> each carries only its
    # residual-of-phase signal ("open FOR its phase", "winning FOR its phase", "this far from cluster
    # FOR its phase"). Decollinearizes the domains so conditional concepts aren't just phase-variants.
    zc = C[:, 0] - C[:, 0].mean()
    for k in range(1, C.shape[1]):
        ck = C[:, k] - C[:, k].mean()
        C[:, k] = ck - (ck @ zc) / (zc @ zc + 1e-9) * zc
    C = (C - C.mean(0)) / (C.std(0) + 1e-8)
    print("[cov] phase partialled out of all other covariates (residualized)", flush=True)
    Xn = (F - F.mean(0)) / (F.std(0) + 1e-8)
    X = torch.from_numpy(Xn).float().to(dev); Ct = torch.from_numpy(C).float().to(dev)
    print(f"[stage] {len(Pk)} positions, c_dim={C.shape[1]} ({', '.join(cov_names)}) ({time.time()-t0:.0f}s)", flush=True)

    sae = CondSAE(X.shape[1], args.dict, C.shape[1]).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for step in range(args.steps):
        bi = torch.from_numpy(rng.integers(0, len(X), size=1024)).to(dev)
        recon, code, _ = sae(X[bi], Ct[bi])
        loss = (recon - X[bi]).pow(2).mean() + args.l1 * code.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sae.normalize_atoms()
        if step % 1000 == 0 or step == args.steps - 1:
            with torch.no_grad():
                r, c_, _ = sae(X, Ct); ev_ = 1 - (r - X).pow(2).mean() / X.var()
                l0 = (c_ > 1e-4).float().sum(1).mean()
            print(f"  step {step:4d}  var-expl {float(ev_):.3f}  active/pos {float(l0):.1f} ({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        _, code, gate = sae(X, Ct)
    code = code.cpu().numpy(); gate = gate.cpu().numpy()
    alive = np.flatnonzero((code > 1e-4).mean(0) > 0.003)
    feats = [named_features(b) for b in boards]
    fnames = [n for n in feats[0] if not n.endswith("_ctrl")]
    Fmat = np.array([[float(f[n][0]) for n in fnames] for f in feats])

    print(f"VERDICT CONDITIONAL_SAE field={Path(args.field).stem} dict={args.dict} alive={len(alive)} anchors={args.anchors}")
    print("  named concepts re-discovered (best atom |corr|):")
    for j, nm in enumerate(fnames):
        cors = [abs(np.corrcoef(code[:, a], Fmat[:, j])[0, 1]) for a in alive]
        print(f"      {nm:18s} atom {alive[int(np.argmax(cors))]:3d}  |corr| {max(cors):.2f}")
    # DOMAIN of each atom: which covariate its gate tracks (concept + where it applies).
    # Show, for EACH covariate, the concept most conditioned on it (the conditional concept for that context).
    dom = np.array([[np.corrcoef(gate[:, a], C[:, k])[0, 1] for k in range(C.shape[1])] for a in alive])
    print("  conditional concept per covariate (atom whose DOMAIN-gate most tracks that covariate):")
    for k, cn in enumerate(cov_names):
        ai = int(np.abs(dom[:, k]).argmax()); a = alive[ai]
        cors = [abs(np.corrcoef(code[:, a], Fmat[:, j])[0, 1]) for j in range(len(fnames))]
        jm = int(np.argmax(cors)); named = f"{fnames[jm].replace('_w','')}({cors[jm]:.2f})" if cors[jm] > 0.30 else "novel"
        order = np.argsort(-code[:, a])
        print(f"      {cn:12s}{'+' if dom[ai,k]>0 else '-'} r={dom[ai,k]:+.2f} -> atom {a:3d} [{named:16s}] {heatmap(Pk, Mk, order)}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
