#!/usr/bin/env python
"""catspace/research/tools/viz/viz_fb.py -- three FB-embedding visualizations (Kaveh 2026-07-21), per the FB / quasimetric-RL
viz literature (Touati-Ollivier FB 2103.07945; QRL Wang 2023; IQE 2211.15120; Dynamics-aware Embeddings
1908.09357; Successor Feature Landmarks 2111.09858). To show real reachability structure we span regimes --
MIDDLEGAME (lichess) + ENDGAME (won KRRvKBP tree) -- rather than the won-only set, which the field collapses.

  (a) F/B OVERLAY -- t-SNE of F(s) and B(s) stacked; F=circles, B=triangles, colored by piece count
      (middlegame=bright -> endgame=dark). Where F- and B-points coincide = the F-reach INTERSECT B-goal
      geography; the phase gradient shows the middlegame->endgame axis (Kaveh's transfer question).
  (b) REACHABILITY from a middlegame reference -- every point colored by d(F(ref), B(point)): does the
      endgame/mate region read as reachable from a middlegame position? (the successor-measure heatmap.)
  (c) ASYMMETRY -- scatter d(F(a),B(b)) vs d(F(b),B(a)); off-diagonal = one-way reachability.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
from catspace.io import paths
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device


def pcount(P, M):
    return np.array([len(board_from_packed(P[i], M[i]).piece_map()) for i in range(len(P))])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default=paths.sep("xfer_treat_20k.pt"))
    ap.add_argument("--dtm-npz", default=paths.derived("dtm_endgame.npz"))
    ap.add_argument("--lichess", default=paths.shards("lichess_db_standard_rated_2019-01.prefix256mb"))
    ap.add_argument("--n-mid", type=int, default=900)
    ap.add_argument("--n-end", type=int, default=900)
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--out", default=paths.experiment("viz_fb.png"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]

    # endgame (won KRRvKBP tree) + middlegame (lichess)
    dz = np.load(args.dtm_npz); ei = rng.permutation(len(dz["packed"]))[:args.n_end]
    Pe, Me = np.asarray(dz["packed"])[ei], np.asarray(dz["meta"])[ei]
    mz = np.load(glob.glob(args.lichess + "/shard_*.npz")[0]); mi = rng.permutation(len(mz["packed"]))[:args.n_mid]
    Pm, Mm = np.asarray(mz["packed"])[mi], np.asarray(mz["meta"])[mi]
    P = np.concatenate([Pm, Pe]); M = np.concatenate([Mm, Me])
    pc = pcount(P, M)

    with torch.no_grad():
        t = torch.from_numpy(feature_planes(P, M)).to(dev)
        F = fb.embed_F(t, torch.from_numpy(np.tile(om, (len(P), 1))).to(dev))
        B = fb.embed_B(t)
        Fn, Bn = F.cpu().numpy(), B.cpu().numpy()

    XY = TSNE(n_components=2, perplexity=30, init="pca", random_state=args.seed).fit_transform(np.vstack([Fn, Bn]))
    FX, BX = XY[:len(Fn)], XY[len(Fn):]

    ref = int(np.argmax(pc))                                               # a middlegame (many-piece) reference
    with torch.no_grad():
        d_ref = fb.distance_matrix(F[ref:ref + 1], B)[0].cpu().numpy()
    ai = rng.integers(0, len(Fn), args.pairs); bi = rng.integers(0, len(Fn), args.pairs)
    with torch.no_grad():
        d_ab = fb.distance_matrix(F[ai], B[bi]).diagonal().cpu().numpy()
        d_ba = fb.distance_matrix(F[bi], B[ai]).diagonal().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(20, 6.2))
    s0 = ax[0].scatter(FX[:, 0], FX[:, 1], c=pc, cmap="viridis", s=14, marker="o", alpha=0.7)
    ax[0].scatter(BX[:, 0], BX[:, 1], c=pc, cmap="viridis", s=28, marker="^", alpha=0.7)
    ax[0].set_title("(a) F/B overlay, colored by piece count\ncircles=F, triangles=B  (bright=middlegame -> dark=endgame)")
    fig.colorbar(s0, ax=ax[0], label="piece count")

    order = np.argsort(-d_ref)
    s1 = ax[1].scatter(BX[order, 0], BX[order, 1], c=d_ref[order], cmap="plasma_r", s=18, alpha=0.85)
    ax[1].scatter([BX[ref, 0]], [BX[ref, 1]], c="lime", s=240, marker="*", edgecolor="k",
                  label=f"reference ({int(pc[ref])} pieces)")
    ax[1].set_title("(b) reachability from a middlegame ref\nd(F(ref), B(·)): bright=near/reachable, dark=far")
    ax[1].legend(loc="upper right"); fig.colorbar(s1, ax=ax[1], label="learned quasimetric distance")

    lim = max(1e-3, np.quantile(np.concatenate([d_ab, d_ba]), 0.98))
    ax[2].scatter(d_ab, d_ba, s=8, alpha=0.35)
    ax[2].plot([0, lim], [0, lim], "r--", lw=1.5, label="symmetric (y=x)")
    ax[2].set_xlim(0, lim); ax[2].set_ylim(0, lim)
    ax[2].set_xlabel("d(F(a), B(b))"); ax[2].set_ylabel("d(F(b), B(a))")
    ax[2].set_title("(c) asymmetry of the quasimetric\noff-diagonal = one-way reachability")
    ax[2].legend(loc="upper left")

    asym = float(np.mean(np.abs(d_ab - d_ba)) / (np.mean(d_ab + d_ba) / 2 + 1e-9))
    fig.suptitle(f"FB embedding visualizations -- field={Path(args.field).stem}  "
                 f"(mean |asymmetry|={asym:.2f}; d_ref med={np.median(d_ref):.2f}, range {d_ref.min():.1f}-{d_ref.max():.1f})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"VERDICT VIZ_FB saved {args.out} | n_mid={args.n_mid} n_end={args.n_end} ref_pieces={int(pc[ref])} "
          f"mean_asymmetry={asym:.3f} d_ref[min/med/max]={d_ref.min():.2f}/{np.median(d_ref):.2f}/{d_ref.max():.2f}")


if __name__ == "__main__":
    main()
