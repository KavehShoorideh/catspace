#!/usr/bin/env python
"""catspace/research/components/planner/approaches/atlas_region_stats/experiments/msm_basins.py -- MARKOV STATE MODEL on the trained field: let the metastable basins
fall out of the HUMAN transition DYNAMICS (Kaveh), not from argmax of WDL. Transition Path Theory /
MSM (molecular-kinetics framework; deeptime won't build on py3.14 so the core is implemented here).

Pipeline:
  1. embed human game positions with the field's phi; keep game trajectories (game_id, ply order).
  2. DISCRETIZE phi -> n_micro microstates (k-means).
  3. estimate the REVERSIBLE transition operator T at lag = 1 sampled step (symmetrized counts) from
     the real trajectories.
  4. SPECTRUM of T: eigenvalues near 1 = slow/metastable modes; implied timescales; the spectral GAP
     tells you HOW MANY basins there really are (not assumed to be 3).
  5. PCCA-style metastable clustering (k-means on the top eigenvectors) -> macrostates = the BASINS,
     defined by the dynamics.
  6. characterize each basin (committor / W-D-L / piece count / ply) and the coarse basin->basin
     transition matrix + stationary weights. Figure: UMAP colored by basin + spectrum + matrix.
"""
from __future__ import annotations

import argparse, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from catspace.research.components.encoder.approaches.reachability_field.experiments.train_clock_field import ClockField
from catspace.research.tools.training_infra.train.scaffold import resolve_device
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=paths.experiment("field_fullgame_v3_final.pt"))
    ap.add_argument("--data", default=paths.derived("field_fullgame_v1.npz"))
    ap.add_argument("--games", type=int, default=40000)
    ap.add_argument("--n-micro", type=int, default=150)
    ap.add_argument("--n-macro", type=int, default=0, help="0 = pick from the spectral gap")
    ap.add_argument("--out", default=paths.experiment("msm_basins.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); rng = np.random.default_rng(args.seed)

    p = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = p["cfg"]; net = ClockField(cfg["d"], ch=cfg["ch"], blocks=cfg["blocks"], in_planes=112).to(dev)
    net.load_state_dict(p["state_dict"]); net.eval()
    z = np.load(args.data); planes = z["planes"]; game = z["game"]; ply = z["ply"]

    by_game = defaultdict(list)
    for i in range(len(game)):
        by_game[int(game[i])].append(i)
    games = [g for g in by_game if len(by_game[g]) >= 3]
    sel = rng.choice(games, size=min(args.games, len(games)), replace=False)
    idx = np.array(sorted(i for g in sel for i in by_game[g]))
    print(f"[msm] {len(sel)} games, {len(idx)} positions | n_micro {args.n_micro}", flush=True)

    import torch.nn.functional as F
    phi_l, comm_l, wdl_l = [], [], []
    for i in range(0, len(idx), 4096):
        x = torch.from_numpy(planes[idx[i:i+4096]].astype(np.float32)).to(dev)
        with torch.no_grad():
            phi_l.append(net.phi(x).cpu().numpy())
            pe = F.softmax(net.d_mate_and_end(x)[1], 1).cpu().numpy()
        comm_l.append(pe[:, 0]); wdl_l.append(np.stack([pe[:, 0], pe[:, 1:5].sum(1), pe[:, 5]], 1))
    phi = np.concatenate(phi_l); comm = np.concatenate(comm_l); wdl = np.concatenate(wdl_l)
    pieces = planes[idx][:, 0:12].reshape(len(idx), 12, -1).sum(axis=(1, 2)).astype(int)
    pos = {v: k for k, v in enumerate(idx)}

    # 2. microstates
    from sklearn.cluster import KMeans, MiniBatchKMeans
    print("  clustering microstates...", flush=True)
    km = MiniBatchKMeans(n_clusters=args.n_micro, random_state=args.seed, n_init=3, batch_size=4096).fit(phi)
    micro = km.labels_

    # 3. reversible transition operator at lag 1 (symmetrized counts along trajectories)
    C = np.zeros((args.n_micro, args.n_micro))
    for g in sel:
        rows = sorted(by_game[g], key=lambda i: ply[i])
        seq = [micro[pos[r]] for r in rows]
        for a, b in zip(seq[:-1], seq[1:]):
            C[a, b] += 1
    Csym = C + C.T + 1e-6
    piw = Csym.sum(1)
    T = Csym / piw[:, None]                                    # reversible T, stationary pi ~ piw

    # 4. spectrum
    from scipy.linalg import eig
    w, V = eig(T)                                             # right eigenvectors
    order = np.argsort(-w.real); w = w.real[order]; V = V.real[:, order]
    tau = 1.0
    its = np.where(w[1:] > 1e-6, -tau / np.log(np.clip(w[1:], 1e-9, 0.999999)), np.nan)
    print("  top eigenvalues:", np.round(w[:8], 4))
    print("  implied timescales (sampled steps):", np.round(its[:7], 2))
    gaps = w[:7] - w[1:8]
    n_macro = args.n_macro or int(np.argmax(gaps[1:6]) + 2)    # gap after the stationary mode
    print(f"  spectral gap suggests n_macro = {n_macro}", flush=True)

    # 5. PCCA-style metastable clustering: cluster microstates on the top n_macro eigenvectors
    feat = V[:, :n_macro] * piw[:, None] ** 0.0               # right eigenvectors
    macro_of_micro = KMeans(n_clusters=n_macro, random_state=args.seed, n_init=5).fit_predict(feat)
    macro = macro_of_micro[micro]                            # per-position macrostate (basin)

    # 6. characterize + coarse transition matrix
    print("\nBASINS (metastable macrostates from the dynamics):")
    order_by_comm = np.argsort([comm[macro == m].mean() if (macro == m).any() else 0 for m in range(n_macro)])
    relabel = {old: new for new, old in enumerate(order_by_comm[::-1])}   # 0 = most winning
    macro = np.array([relabel[m] for m in macro]); macro_of_micro = np.array([relabel[m] for m in macro_of_micro])
    for m in range(n_macro):
        mm = macro == m
        print(f"  basin {m}: n={mm.sum():6d} | committor {comm[mm].mean():.2f} | "
              f"W/D/L {wdl[mm,0].mean():.2f}/{wdl[mm,1].mean():.2f}/{wdl[mm,2].mean():.2f} | "
              f"pieces {np.median(pieces[mm]):.0f} | ply {int(np.median(ply[idx][mm]))}")
    # coarse basin transition matrix (aggregate micro T by macro, weighted by pi)
    Tm = np.zeros((n_macro, n_macro))
    for i in range(args.n_micro):
        for j in range(args.n_micro):
            Tm[macro_of_micro[i], macro_of_micro[j]] += piw[i] * T[i, j]
    Tm /= Tm.sum(1, keepdims=True)
    pim = np.array([piw[macro_of_micro == m].sum() for m in range(n_macro)]); pim /= pim.sum()
    print("\nBASIN transition matrix T (rows=from, cols=to), lag=1 sampled step (~6 plies):")
    print("       " + "  ".join(f"->b{j}" for j in range(n_macro)))
    for i in range(n_macro):
        print(f"  b{i} ({pim[i]:.0%})  " + "  ".join(f"{Tm[i,j]:.3f}" for j in range(n_macro)))

    # figure
    import umap, matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("  UMAP...", flush=True)
    sub = rng.choice(len(idx), size=min(9000, len(idx)), replace=False)
    xy = umap.UMAP(n_neighbors=25, min_dist=0.15, random_state=args.seed).fit_transform(phi[sub])
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 2, 2])
    ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1]); ax2 = fig.add_subplot(gs[2])
    cmap = plt.get_cmap("tab10")
    for m in range(n_macro):
        mm = macro[sub] == m
        ax0.scatter(xy[mm, 0], xy[mm, 1], s=6, alpha=0.5, color=cmap(m), label=f"basin {m}")
    ax0.legend(markerscale=2, fontsize=8); ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_title("Field UMAP colored by DYNAMICS-DEFINED basin (PCCA-style)")
    ax1.plot(range(1, 9), w[:8], "o-"); ax1.axvline(n_macro + 0.5, color="r", ls="--", label=f"n_macro={n_macro}")
    ax1.set_xlabel("eigenvalue index"); ax1.set_ylabel("eigenvalue (1=stationary)"); ax1.legend()
    ax1.set_title("Transition-operator spectrum\n(gap after slow modes = # basins)")
    im = ax2.imshow(Tm, cmap="magma", vmin=0, vmax=1)
    ax2.set_xticks(range(n_macro)); ax2.set_yticks(range(n_macro))
    ax2.set_xlabel("to basin"); ax2.set_ylabel("from basin"); ax2.set_title("Basin transition matrix (lag~6 plies)")
    for i in range(n_macro):
        for j in range(n_macro):
            ax2.text(j, i, f"{Tm[i,j]:.2f}", ha="center", va="center",
                     color="white" if Tm[i, j] < 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nVERDICT msm-basins: n_macro {n_macro} -> {args.out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
