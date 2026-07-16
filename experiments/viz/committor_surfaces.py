#!/usr/bin/env python
"""
experiments/viz/committor_surfaces.py — visualize the committor field and its
boundary surfaces on the toy (Kaveh 2026-07-15: goals are SURFACES).

Topology discipline (Kaveh): a 2-D projection cannot guarantee that surfaces
disjoint in 64-d stay disjoint on paper, so the load-bearing view is
FIELD-NATIVE coordinates -- committor space itself, where the surfaces are
coordinate loci by construction and any apparent contact is a genuinely
mixed-commitment state. The spatial (UMAP) panel is illustrative only; the
separation claim it suggests is printed as high-dim statistics (within- vs
cross-basin F-distances) computed in the full space, never from the layout.

v1 (current data): W-committor head + empirical outcome mix from v1 tables
(draw ~= not-won in the toy). After round-2 v2 generation (terminal boards
labeled per surface), extend to the surface ATLAS: points ON each boundary,
kNN adjacency in F, visual contacts validated against true adjacency.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/sep/committor.pt")
    ap.add_argument("--whead", default="data/derived/sep/committor_whead.pt")
    ap.add_argument("--table", default="artifacts/experiments/certainty_table_r2_K16.json")
    ap.add_argument("--max-states", type=int, default=6000)
    ap.add_argument("--min-visits", type=int, default=8,
                    help="committed-basin markers need confident empirical P-hat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/generated/committor_surfaces.png")
    args = ap.parse_args()

    import torch
    from catspace.nn.fb import load_ckpt

    fb, _ = load_ckpt(Path(args.ckpt), "cpu")
    hp = torch.load(args.whead, map_location="cpu", weights_only=False)
    head = torch.nn.Sequential(torch.nn.Linear(hp["d_in"], 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, 1), torch.nn.Softplus())
    head.load_state_dict(hp["state"]); head.eval(); fb.eval()

    rows = json.loads(Path(args.table).read_text())["rows"]
    rng = np.random.default_rng(args.seed)
    if len(rows) > args.max_states:
        rows = [rows[i] for i in rng.choice(len(rows), args.max_states, replace=False)]
    print(f"{len(rows)} states from {args.table}")

    F = []
    with torch.no_grad():
        for i in range(0, len(rows), 512):
            ch = rows[i:i + 512]
            boards = [chess.Board(r["fen"]) for r in ch]
            packed = np.stack([encode_packed(b) for b in boards])
            meta = np.stack([encode_meta(b) for b in boards])
            om = omega_ids(np.full(len(ch), 1800), np.full(len(ch), 1800),
                           np.full(len(ch), np.nan))
            F.append(fb.embed_F(torch.from_numpy(feature_planes(packed, meta)),
                                torch.from_numpy(om)))
        F = torch.cat(F)
        d_w = head(F).squeeze(-1).numpy()
    P_learned = np.exp(-d_w)
    F = F.numpy()
    p_emp = np.array([r["p_hat"] for r in rows])
    n_vis = np.array([r["n"] for r in rows])

    # committed basins (confident empirical extremes) -- proxies for points
    # near each surface until v2 terminal boards exist
    win_c = (p_emp == 1.0) & (n_vis >= args.min_visits)
    draw_c = (p_emp == 0.0) & (n_vis >= args.min_visits)

    # HIGH-DIM separation statistics (the load-bearing numbers; the 2-D
    # panel below is illustrative only)
    def mean_pdist(A, B, k=2000):
        ia = rng.choice(len(A), min(k, len(A)))
        ib = rng.choice(len(B), min(k, len(B)))
        return float(np.linalg.norm(A[ia][:, None, :] - B[ib][None, :, :],
                                    axis=-1).mean()) if len(A) and len(B) else float("nan")
    d_ww = mean_pdist(F[win_c], F[win_c], 300)
    d_dd = mean_pdist(F[draw_c], F[draw_c], 300)
    d_wd = mean_pdist(F[win_c], F[draw_c], 300)
    print(f"VERDICT SURFACE_SEPARATION mean|F-F'| win-win {d_ww:.3f}  draw-draw {d_dd:.3f}  "
          f"cross {d_wd:.3f}  ratio cross/within {2*d_wd/(d_ww+d_dd):.2f} "
          f"(committed states: {int(win_c.sum())} win, {int(draw_c.sum())} draw)")

    import umap
    xy = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=args.seed).fit_transform(F)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    # Panel A: FIELD-NATIVE coordinates -- surfaces are the plot's edges by
    # construction; no projection artifacts possible
    from scipy.stats import spearmanr
    rho, _ = spearmanr(P_learned, p_emp)
    a1.hexbin(P_learned, p_emp, gridsize=40, cmap="Blues", mincnt=1, bins="log")
    a1.plot([0, 1], [0, 1], "--", color="#c0392b", lw=1.5, label="perfect calibration")
    a1.set_xlabel("learned committor  P_W = exp(−d_W(s))")
    a1.set_ylabel("empirical P̂ (own-play rollouts)")
    a1.set_title("Committor space (load-bearing view)\n"
                 "WIN surface = right edge, draw basin = left edge")
    a1.text(0.44, 0.5,
            f"rank ρ = {rho:+.2f} (good)\nbut absolutely COMPRESSED:\n"
            f"learned span [{P_learned.min():.2f}, {P_learned.max():.2f}]\n"
            f"vs empirical [0, 1] —\nordering learned, probability\nscale not yet calibrated",
            fontsize=9, color="#333")
    a1.legend(loc="lower right", fontsize=8, frameon=False)
    # Panel B: spatial map, explicitly illustrative
    o = np.argsort(rng.random(len(xy)))
    sc = a2.scatter(xy[o, 0], xy[o, 1], c=P_learned[o], s=4, cmap="coolwarm_r",
                    alpha=0.6, edgecolors="none")
    a2.scatter(xy[win_c, 0], xy[win_c, 1], s=14, facecolors="none",
               edgecolors="#1a7a1a", lw=0.8, label=f"committed WIN (P̂=1, n≥{args.min_visits})")
    a2.scatter(xy[draw_c, 0], xy[draw_c, 1], s=14, facecolors="none",
               edgecolors="#7a1a7a", lw=0.8, label=f"committed DRAW (P̂=0, n≥{args.min_visits})")
    plt.colorbar(sc, ax=a2, label="learned P_W")
    a2.set_title("F-space UMAP (ILLUSTRATIVE — 2-D cannot certify topology;\n"
                 f"high-dim separation: cross/within distance ratio "
                 f"{2*d_wd/(d_ww+d_dd):.2f})")
    a2.set_xticks([]); a2.set_yticks([])
    a2.legend(loc="lower right", fontsize=8, frameon=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
