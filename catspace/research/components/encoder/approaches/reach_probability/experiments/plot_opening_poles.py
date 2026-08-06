#!/usr/bin/env python
"""plot_opening_poles.py -- the early game in embedding space, and the pole frame.

Kaveh: "i wanna see the poles", "i wanna see all the embeddings in our position in ply 1", "and
ply 5". Six panels, deliberately more than needed so there is something to choose from.

WHAT IS AND IS NOT REAL HERE, stated up front because one panel is weaker than the others:
the completed 20k-step run is the STRATA run and is deliberately POLE-FREE -- poles would have fed
ending labels into a claim about nothing chess-specific being programmed. So the only pole-bearing
checkpoint is a 300-step stage-2 smoke. The FIXED SIMPLEX itself is analytic and exact (it is a
buffer, not learned), so panel E is real geometry; panel F is a barely-trained readout and is
labelled as such rather than presented as a result.

Ply 1 has only ~20 distinct positions -- the legal first moves -- so it can be labelled move by
move, which makes it the most directly interpretable view of the space we have.
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
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (  # noqa: E402
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.experiments.plot_strata_figures import (  # noqa: E402
    embed)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T  # noqa: E402
from catspace.research.components.encoder.approaches.reach_probability.src.reach_vit_jepa import (  # noqa: E402
    simplex_poles)


def project(Z, basis=None):
    Zc = Z - Z.mean(0, keepdims=True)
    if basis is None:
        _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        return Zc @ Vt[:2].T, Vt, (S ** 2 / (S ** 2).sum())[:2].sum()
    return Zc @ basis[:2].T, basis, float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=paths.experiment("reach_vit_v1_latest.pt"))
    ap.add_argument("--pole-ckpt", default=None, help="a POLE-bearing checkpoint (stage-2 smoke)")
    ap.add_argument("--n-late", type=int, default=8000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=paths.figure("reach_vit_v1_opening_poles.png"))
    args = ap.parse_args()

    t0 = time.time()
    net, payload = load_net(args.ckpt, args.device)
    c = payload["cfg"]
    tr = T.build(n_human=c["games"] // 2, n_sf=c["games"] // 2, seed=c["traj_seed"],
                 max_plies=c["max_plies"], verbose=False)
    ply, pc = tr.ply_of_row(), tr.piece_count()
    src = np.repeat(tr.source, tr.length)
    iqe = net.qhead.iqe if getattr(net, "dual", False) else net.iqe
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(f"Opening structure and the pole frame -- reach_vit_v1 @ step {payload.get('step')}",
                 fontsize=14)

    # ---- shared projection basis from a broad sample, so panels are COMPARABLE -----------------
    broad = np.flatnonzero(ply <= 40)
    broad = broad[rng.integers(0, len(broad), args.n_late)]
    Zb = embed(net, tr, broad, args.device).numpy()
    _, basis, var = project(Zb)
    print(f"[proj] shared basis from {len(broad):,} positions, PC1-2 = {var:.0%} of variance")

    # ---- A: ply 1 -- the ~20 legal first moves, labelled --------------------------------------
    import chess
    r1 = np.flatnonzero(ply == 1)
    u1, idx1 = np.unique(tr.tok[r1], axis=0, return_index=True)
    rows1 = r1[idx1]
    Z1 = embed(net, tr, rows1, args.device).numpy()
    P1, _, _ = project(Z1, basis)
    names = []
    for r in rows1:
        b = chess.Board()
        tgt = tr.tok[r]
        mv = next((m for m in b.legal_moves
                   if (lambda bb: (bb.push(m), np.array_equal(
                       __import__("catspace.research.components.encoder.approaches.jepa_tokenizer."
                                  "src.jepa", fromlist=["tokenize"]).tokenize(bb)[0], tgt))[1])(
                       chess.Board())), None)
        names.append(b.san(mv) if mv else "?")
    ax[0][0].scatter(P1[:, 0], P1[:, 1], s=60, c="#2e5f9e")
    for k, nm in enumerate(names):
        ax[0][0].annotate(nm, (P1[k, 0], P1[k, 1]), fontsize=8, xytext=(3, 3),
                          textcoords="offset points")
    ax[0][0].set_title(f"A. PLY 1 -- all {len(rows1)} distinct first moves")
    ax[0][0].set_xlabel("PC1"); ax[0][0].set_ylabel("PC2")

    # ---- B: ply 5 ------------------------------------------------------------------------------
    r5 = np.flatnonzero(ply == 5)
    u5, idx5 = np.unique(tr.tok[r5], axis=0, return_index=True)
    rows5 = r5[idx5][:4000]
    Z5 = embed(net, tr, rows5, args.device).numpy()
    P5, _, _ = project(Z5, basis)
    ax[0][1].scatter(P5[:, 0], P5[:, 1], s=6, alpha=0.5, c="#7d3c98", edgecolors="none")
    ax[0][1].scatter(P1[:, 0], P1[:, 1], s=50, c="#2e5f9e", label="ply 1", zorder=3)
    ax[0][1].set_title(f"B. PLY 5 -- {len(rows5):,} distinct positions (ply 1 overlaid)")
    ax[0][1].legend(fontsize=8); ax[0][1].set_xlabel("PC1")

    # ---- C: the opening FAN -- how the space opens up with ply --------------------------------
    for p, col in ((1, "#2e5f9e"), (5, "#7d3c98"), (15, "#1a7f5a"), (40, "#c0392b")):
        rp = np.flatnonzero(ply == p)
        if not len(rp):
            continue
        rp = rp[rng.integers(0, len(rp), min(2500, len(rp)))]
        Pp, _, _ = project(embed(net, tr, rp, args.device).numpy(), basis)
        ax[0][2].scatter(Pp[:, 0], Pp[:, 1], s=4, alpha=0.35, c=col, label=f"ply {p}",
                         edgecolors="none")
    ax[0][2].set_title("C. the opening FAN: ply 1 / 5 / 15 / 40")
    ax[0][2].legend(fontsize=8, markerscale=3); ax[0][2].set_xlabel("PC1")

    # ---- D: same space, coloured by MATERIAL (the strata question, positionally) ---------------
    sc = ax[1][0].scatter(project(Zb, basis)[0][:, 0], project(Zb, basis)[0][:, 1],
                          c=pc[broad], s=4, cmap="viridis", alpha=0.6, edgecolors="none")
    plt.colorbar(sc, ax=ax[1][0], fraction=0.046, label="piece count")
    ax[1][0].set_title("D. same space by MATERIAL\n(paired ratchet was 0.500 -- expect little)")
    ax[1][0].set_xlabel("PC1"); ax[1][0].set_ylabel("PC2")

    # ---- E: the FIXED SIMPLEX -- exact, analytic, not learned ----------------------------------
    d_head = c["d"]; comps = c["components"]
    P = simplex_poles(d_head, comps, 3, 3.0)
    dm = np.array([[float(iqe(P[i:i + 1].to(args.device), P[j:j + 1].to(args.device))[0])
                    for j in range(3)] for i in range(3)])
    im = ax[1][1].imshow(dm, cmap="magma")
    for i in range(3):
        for j in range(3):
            ax[1][1].text(j, i, f"{dm[i,j]:.2f}", ha="center", va="center",
                          color="w" if dm[i, j] < dm.max() * 0.6 else "k")
    ax[1][1].set_xticks(range(3), ["WIN", "DRAW", "LOSS"])
    ax[1][1].set_yticks(range(3), ["WIN", "DRAW", "LOSS"])
    ax[1][1].set_title("E. the FIXED pole frame d(row -> col)\nexact simplex, a buffer not learned")

    # ---- F: committor readout from a POLE-BEARING checkpoint (weak; labelled) ------------------
    if args.pole_ckpt:
        pnet, ppay = load_net(args.pole_ckpt, args.device)
        piqe = pnet.qhead.iqe if getattr(pnet, "dual", False) else pnet.iqe
        Zp = embed(pnet, tr, broad, args.device).to(args.device)
        PP = pnet.poles.poles[:3]
        dP = torch.stack([piqe(Zp, PP[k].expand(len(Zp), -1)) for k in range(3)], 1)
        p_wdl = torch.softmax(-torch.log1p(dP), 1).float().cpu().numpy()
        sc2 = ax[1][2].scatter(project(Zb, basis)[0][:, 0], project(Zb, basis)[0][:, 1],
                               c=p_wdl[:, 0] - p_wdl[:, 2], s=4, cmap="coolwarm", alpha=0.7,
                               vmin=-0.3, vmax=0.3, edgecolors="none")
        plt.colorbar(sc2, ax=ax[1][2], fraction=0.046, label="P(win) - P(loss)")
        ax[1][2].set_title(f"F. committor from a POLE checkpoint (step {ppay.get('step')})\n"
                           f"300-step smoke -- NOT a result, shown for shape only")
    else:
        ax[1][2].text(0.5, 0.5, "no --pole-ckpt given\n(the 20k run is deliberately pole-free)",
                      ha="center", va="center", fontsize=11)
        ax[1][2].set_title("F. committor readout -- unavailable")
    ax[1][2].set_xlabel("PC1")

    fig.text(0.5, 0.005, "Single seed. Panels A-D come from the completed 20k pole-free strata run; "
             "E is exact analytic geometry; F is a 300-step smoke and is not evidence.",
             ha="center", fontsize=9, color="#b03a2e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(args.out, dpi=140)
    print(f"[fig] -> {args.out} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
