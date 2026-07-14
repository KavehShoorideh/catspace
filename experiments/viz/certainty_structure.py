#!/usr/bin/env python
"""
experiments/viz/certainty_structure.py — the 3-panel "did the distill learn the
certainty field?" figure (JOURNAL 2026-07-14, memorization diagnosis).

Every panel uses the SAME states: the rollout certainty table (one row per toy
KRRvKBP state, P-hat = observed win rate under the eps-noised scaffold).
  (a) incumbent model:  learned d(s, mate goal)  vs  certainty target
  (b) distilled model:  same axes, train rows vs HELD-OUT rows marked separately
      -- the generalization gap (memorization) is visible as two Spearmans
  (c) UMAP of the distilled F embeddings, colored by P-hat -- is certainty
      spatially organized in the embedding at all?
Target = (plies + lam * (-ln P_clip)) / scale, identical to certainty_distill.py
(same lam/scale/horizon defaults, same seed-0 holdout split so panel (b)'s
held-out rows are exactly the rows the distill never saw).
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids
from experiments.certainty_distill import spearman_ci


def learned_d(fb, zW, rows, dev, batch=512):
    import torch
    out = []
    feats = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        boards = [chess.Board(r["fen"]) for r in chunk]
        packed = np.stack([encode_packed(b) for b in boards])
        meta = np.stack([encode_meta(b) for b in boards])
        om = omega_ids(np.full(len(chunk), 1800), np.full(len(chunk), 1800),
                      np.full(len(chunk), np.nan))
        with torch.no_grad():
            pl = torch.from_numpy(feature_planes(packed, meta)).to(dev)
            f = fb.embed_F(pl, torch.from_numpy(om).to(dev))
            out.append(fb.distance_matrix(f, zW[None, :])[:, 0].cpu().numpy())
            feats.append(f.cpu().numpy())
    return np.concatenate(out), np.concatenate(feats)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-incumbent", default="data/derived/lichess_fb_4gb_qm_plygap_only.pt")
    ap.add_argument("--ckpt-distilled", default="data/derived/sep/certainty_full.pt")
    ap.add_argument("--table", default="artifacts/experiments/certainty_table.json")
    ap.add_argument("--lam", type=float, default=8.0)
    ap.add_argument("--scale", type=float, default=50.0)
    ap.add_argument("--horizon", type=float, default=100.0)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0, help="must match certainty_distill.py")
    ap.add_argument("--out", default="artifacts/generated/certainty_structure.png")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from catspace.nn.fb import load_ckpt, pick_device

    dev = pick_device(args.device)
    rows = json.loads(Path(args.table).read_text())["rows"]
    p_hat = np.array([r["p_hat"] for r in rows])
    tgt = np.array([(r["plies"] if r["plies"] is not None else args.horizon)
                    + args.lam * (-np.log(max(r["p_hat"], 1.0 / (r["n"] + 2))))
                    for r in rows]) / args.scale
    # reproduce the distill's holdout split exactly (same seed, same order)
    order = np.random.default_rng(args.seed).permutation(len(rows))
    is_hold = np.zeros(len(rows), bool)
    is_hold[order[:int(len(rows) * args.holdout_frac)]] = True
    print(f"{len(rows)} table states ({is_hold.sum()} held out from the distill)")

    def zgoal(pay):
        z = pay["zgoals"]["MATE_W"]
        return (z.to(dev).float() if torch.is_tensor(z)
                else torch.as_tensor(np.asarray(z), dtype=torch.float32, device=dev))

    fb_i, pay_i = load_ckpt(Path(args.ckpt_incumbent), dev)
    d_inc, _ = learned_d(fb_i.eval(), zgoal(pay_i), rows, dev)
    del fb_i
    fb_d, pay_d = load_ckpt(Path(args.ckpt_distilled), dev)
    d_dis, F_dis = learned_d(fb_d.eval(), zgoal(pay_d), rows, dev)

    r_inc = spearman_ci(d_inc, tgt)
    r_tr = spearman_ci(d_dis[~is_hold], tgt[~is_hold])
    r_ho = spearman_ci(d_dis[is_hold], tgt[is_hold])
    print(f"Spearman incumbent(all) {r_inc[0]:+.3f}  "
          f"distilled train {r_tr[0]:+.3f} / held-out {r_ho[0]:+.3f}")

    from umap import UMAP
    xy = UMAP(n_neighbors=30, min_dist=0.3, random_state=args.seed).fit_transform(F_dis)

    fig, axes = plt.subplots(1, 3, figsize=(19, 7.6))
    fig.suptitle("Did distillation put the certainty field into the embedding?  "
                 "Same rollout-table states in every panel; color = P̂ "
                 "(observed win rate, red=0 → green=1)", fontsize=13, y=0.97)

    def scatter_vs_target(ax, d, title):
        sc = ax.scatter(tgt, d, c=p_hat, cmap="RdYlGn", s=8, alpha=0.6,
                        vmin=0, vmax=1, linewidths=0)
        ax.set_xlabel("certainty target  (plies + 8·(−ln P̂)) / 50")
        ax.set_ylabel("model's learned distance to mate goal")
        ax.set_title(title, fontsize=11, pad=10)
        return sc

    scatter_vs_target(axes[0], d_inc,
                      f"(a) INCUMBENT — Spearman {r_inc[0]:+.2f}")
    axes[1].scatter(tgt[is_hold], d_dis[is_hold], marker="x", c=p_hat[is_hold],
                    cmap="RdYlGn", s=22, vmin=0, vmax=1,
                    label=f"held-out (ρ={r_ho[0]:+.2f})")
    sc = axes[1].scatter(tgt[~is_hold], d_dis[~is_hold], c=p_hat[~is_hold],
                         cmap="RdYlGn", s=8, alpha=0.5, vmin=0, vmax=1,
                         linewidths=0, label=f"train rows (ρ={r_tr[0]:+.2f})")
    axes[1].set_xlabel("certainty target  (plies + 8·(−ln P̂)) / 50")
    axes[1].set_ylabel("model's learned distance to mate goal")
    axes[1].set_title(f"(b) DISTILLED — train ρ {r_tr[0]:+.2f} vs "
                      f"held-out ρ {r_ho[0]:+.2f}", fontsize=11, pad=10)
    axes[1].legend(loc="upper left", fontsize=8)

    axes[2].scatter(xy[:, 0], xy[:, 1], c=p_hat, cmap="RdYlGn", s=8, alpha=0.7,
                    vmin=0, vmax=1, linewidths=0)
    axes[2].set_title("(c) UMAP of distilled F embeddings", fontsize=11, pad=10)
    axes[2].set_xlabel("UMAP-1"); axes[2].set_ylabel("UMAP-2")
    fig.colorbar(sc, ax=axes[2], label="P̂ (rollout win rate)", shrink=0.8)

    captions = [
        "One dot per rollout-table state. x = where the state SHOULD sit: "
        "few plies to mate AND near-certain conversion = small x (the "
        "detached stripe at x≈2.3 is the P̂=0 rows, capped at the horizon). "
        "y = the incumbent's learned distance to the mate goal. A model that "
        "encodes certainty shows a rising diagonal; the incumbent shows none "
        "(ρ≈0) and squeezes every state into a narrow y band.",
        "Same axes after distillation. The cloud tilts upward (ρ +0.2) but is "
        "nowhere near the diagonal band a full fit would be — and train rows "
        "(dots) barely beat held-out rows (×), so this is NOT memorization: "
        "the fine-tune UNDERFIT the certainty target everywhere. It also "
        "navigates to a REBUILT goal vector, not the one it was calibrated "
        "against during training — that alone costs ~0.05 of ρ.",
        "The distilled F embeddings flattened to 2-D. Color is locally "
        "coherent (many small single-color patches: nearby states share a "
        "fate) but there is no global certain-win region a planner could aim "
        "at — the certainty signal never reorganized the large-scale "
        "geometry.",
    ]
    for ax, cap in zip(axes, captions):
        ax.text(0.5, -0.16, textwrap.fill(cap, 58), transform=ax.transAxes,
                ha="center", va="top", fontsize=8.2, color="0.25")
    fig.subplots_adjust(top=0.86, bottom=0.30, left=0.05, right=0.99, wspace=0.28)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
