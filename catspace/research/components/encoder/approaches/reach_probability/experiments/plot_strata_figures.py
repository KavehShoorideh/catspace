#!/usr/bin/env python
"""plot_strata_figures.py -- can you SEE the strata? Four views of the same trained field.

Kaveh asked for a visual: "clouds of disky shaped transitions, connected to each other through
sharp bottleneck vias, not to go back". This script builds the honest versions of that, including
the one that is most likely to DISAPPOINT -- because the measurement we actually have says the
structure is DIRECTIONAL, not positional, and a figure should not imply otherwise.

  A  MATERIAL TRANSITION MATRIX. Mean d(material i -> material j) over cross-game pairs. This is
     the vias idea made measurable: if the ratchet is real the matrix is ASYMMETRIC across its
     diagonal -- cheap downward (material falls), expensive upward (material would have to rise).
     It needs no visual clustering to work, which is why it is the primary geometric figure.

  B  FORWARD vs REVERSE SCATTER. Per observed pair, d(a->b) against d(b->a), coloured by whether
     material fell across the pair. Symmetric pairs land on the diagonal; the claim is that
     capture-crossing pairs sit ABOVE it and quiet ones sit closer to it. This is the measured
     effect itself rather than a summary of it.

  C  ASYMMETRY vs MATERIAL DROP. Median d(rev)/d(fwd) as a function of how many pieces were lost
     across the pair, ply-gap matched. If the ratchet is graded rather than binary, this rises with
     the size of the material drop.

  D  EMBEDDING PROJECTION coloured by piece count -- THE DISKS-AND-VIAS PICTURE, and the one I
     expect to underwhelm. The paired ratchet stayed at 0.500 for the whole run, which says
     cross-position material comparison is NOT encoded, so positions have little reason to lay out
     in material-ordered clusters. It is included because checking is cheaper than assuming, and
     because a negative here is informative: it localises the finding to trajectory-local geometry.

    .venv/bin/python .../plot_strata_figures.py --ckpt artifacts/experiments/reach_vit_v1_latest.pt
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402

from catspace.io import paths                                              # noqa: E402
from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (  # noqa: E402
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (  # noqa: E402
    load_net, match_on_gap)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T  # noqa: E402


@torch.no_grad()
def embed(net, tr, rows, device, batch=4096):
    out = []
    for s in range(0, len(rows), batch):
        r = rows[s:s + batch]
        out.append(net.encode_q(
            torch.from_numpy(tr.tok[r].astype(np.int64)).to(device),
            torch.from_numpy(tr.glob[r].astype(np.float32)).to(device)).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=paths.experiment("reach_vit_v1_latest.pt"))
    ap.add_argument("--n-pos", type=int, default=24000, help="positions embedded for A and D")
    ap.add_argument("--n-pair", type=int, default=30000, help="observed pairs for B and C")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=paths.figure("reach_vit_v1_strata.png"))
    args = ap.parse_args()

    t0 = time.time()
    net, payload = load_net(args.ckpt, args.device)
    c = payload["cfg"]
    tr = T.build(n_human=c["games"] // 2, n_sf=c["games"] // 2, seed=c["traj_seed"],
                 max_plies=c["max_plies"], verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    test = np.flatnonzero(split == 2)
    game, ply, pc, cov = tr.game_of_row(), tr.ply_of_row(), tr.piece_count(), tr.coverage()
    rows_all = np.flatnonzero(np.isin(game, test))
    rng = np.random.default_rng(0)
    print(f"[fig] {len(rows_all):,} test positions | ckpt step {payload.get('step')}", flush=True)

    iqe = net.qhead.iqe if getattr(net, "dual", False) else net.iqe
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(f"Do the strata show? reach_vit_v1 @ step {payload.get('step')}", fontsize=14)

    # ---- A: material transition matrix ---------------------------------------------------------
    sel = rows_all[rng.integers(0, len(rows_all), args.n_pos)]
    Z = embed(net, tr, sel, args.device)
    pcs = pc[sel]
    lo, hi = 8, 32
    buckets = [np.flatnonzero((pcs >= m) & (pcs < m + 2)) for m in range(lo, hi, 2)]
    labels = [f"{m}-{m+1}" for m in range(lo, hi, 2)]
    n_b = len(buckets)
    M = np.full((n_b, n_b), np.nan)
    for i in range(n_b):
        for j in range(n_b):
            if len(buckets[i]) < 30 or len(buckets[j]) < 30:
                continue
            a = buckets[i][rng.integers(0, len(buckets[i]), 120)]
            b = buckets[j][rng.integers(0, len(buckets[j]), 120)]
            with torch.no_grad():
                M[i, j] = float(iqe.pairwise(Z[a].to(args.device),
                                             Z[b].to(args.device)).mean())
    im = ax[0][0].imshow(M, cmap="magma", origin="lower")
    ax[0][0].set_xticks(range(n_b), labels, rotation=90, fontsize=7)
    ax[0][0].set_yticks(range(n_b), labels, fontsize=7)
    ax[0][0].set_xlabel("TO  (pieces)"); ax[0][0].set_ylabel("FROM  (pieces)")
    ax[0][0].set_title("A. mean d(from -> to) by material\n(ratchet => asymmetric across diagonal)")
    plt.colorbar(im, ax=ax[0][0], fraction=0.046)
    asym = np.nanmean(np.triu(M, 1)) - np.nanmean(np.tril(M, -1))
    print(f"[A] mean d(material UP) - d(material DOWN) = {asym:+.3f}"
          f"   (positive => climbing material costs more)")

    # ---- observed pairs, for B and C ------------------------------------------------------------
    i0 = rows_all[rng.integers(0, len(rows_all), args.n_pair)]
    g = game[i0]
    end = tr.start[g] + tr.length[g] - 1
    j0 = i0 + 1 + (rng.random(args.n_pair) * np.minimum(40, end - i0)).astype(np.int64)
    ok = (j0 <= end) & (j0 > cov[i0])                      # unobserved reversal only
    i0, j0 = i0[ok], j0[ok]
    drop = (pc[i0].astype(int) - pc[j0].astype(int))
    Za, Zb = embed(net, tr, i0, args.device), embed(net, tr, j0, args.device)
    with torch.no_grad():
        d_f = iqe(Za.to(args.device), Zb.to(args.device)).float().cpu().numpy()
        d_r = iqe(Zb.to(args.device), Za.to(args.device)).float().cpu().numpy()

    # ---- B: forward vs reverse -----------------------------------------------------------------
    cap, qui = drop >= 1, drop == 0
    s = rng.permutation(len(d_f))[:6000]
    for m, col, lab in ((qui[s], "#2e5f9e", "quiet (no capture)"),
                        (cap[s], "#c0392b", "capture-crossing")):
        ax[0][1].scatter(d_f[s][m], d_r[s][m], s=3, alpha=0.25, c=col, label=lab, edgecolors="none")
    lim = float(np.nanpercentile(np.r_[d_f, d_r], 99))
    ax[0][1].plot([0, lim], [0, lim], "k--", lw=1, label="symmetric (d fwd = d rev)")
    ax[0][1].set_xlim(0, lim); ax[0][1].set_ylim(0, lim)
    ax[0][1].set_xlabel("d(a -> b)  forward, along the game")
    ax[0][1].set_ylabel("d(b -> a)  the reversal")
    ax[0][1].set_title("B. above the diagonal = harder to undo than to do")
    ax[0][1].legend(fontsize=8, markerscale=3)

    # ---- C: asymmetry vs size of the material drop, ply-gap matched -----------------------------
    gap = (ply[j0] - ply[i0]).astype(np.int64)
    ratio = d_r / np.maximum(d_f, 1e-6)
    ref = gap[qui]
    xs, ys, los, his = [], [], [], []
    for k in range(0, 7):
        m = np.flatnonzero(drop == k)
        if len(m) < 200:
            continue
        mm, _ = match_on_gap(gap[m], ref, rng, len(m))      # matched to the QUIET gap distribution
        v = ratio[m[mm]] if len(mm) else ratio[m]
        boot = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(400)]
        xs.append(k); ys.append(float(np.median(v)))
        los.append(float(np.percentile(boot, 2.5))); his.append(float(np.percentile(boot, 97.5)))
    ax[1][0].errorbar(xs, ys, yerr=[np.array(ys) - los, np.array(his) - np.array(ys)],
                      fmt="o-", color="#1a7f5a", capsize=3)
    ax[1][0].axhline(1.0, color="k", lw=0.8, label="symmetric")
    ax[1][0].set_xlabel("pieces lost across the pair"); ax[1][0].set_ylabel("median d(rev) / d(fwd)")
    ax[1][0].set_title("C. is the ratchet GRADED in how much material fell?\n(ply-gap matched)")
    ax[1][0].legend(fontsize=8)
    print("[C] drop -> ratio: " + "  ".join(f"{k}:{v:.2f}" for k, v in zip(xs, ys)))

    # ---- D: the disks-and-vias picture (expected to underwhelm; included honestly) --------------
    sub = rng.permutation(len(Z))[:6000]
    Zc = (Z[sub] - Z[sub].mean(0)).numpy()
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Zc @ Vt[:2].T
    sc = ax[1][1].scatter(P[:, 0], P[:, 1], c=pcs[sub], s=4, cmap="viridis", alpha=0.6,
                          edgecolors="none")
    plt.colorbar(sc, ax=ax[1][1], fraction=0.046, label="piece count")
    var = (S ** 2 / (S ** 2).sum())[:2].sum()
    ax[1][1].set_title(f"D. embedding PC1-PC2 by material ({var:.0%} var)\n"
                       f"the paired ratchet stayed at 0.500, so do NOT expect clean strata here")
    ax[1][1].set_xlabel("PC1"); ax[1][1].set_ylabel("PC2")

    fig.text(0.5, 0.005, "Single seed. The measured effect is DIRECTIONAL (panels A-C); panel D is "
             "shown because checking is cheaper than assuming.", ha="center", fontsize=9,
             color="#b03a2e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(args.out, dpi=140)
    print(f"[fig] -> {args.out} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
