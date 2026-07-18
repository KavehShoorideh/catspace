#!/usr/bin/env python
"""
experiments/viz/committor_atlas.py — region/surface visualization in the
theory's OWN coordinates (2026-07-18, replacing the unhelpful PCA panels).

Three views of the committor field over held-out games:
  1. OUTCOME SIMPLEX  each position is a point in the (P_W, P_D, P_L)
     triangle; the outcome surfaces ARE the corners; games are trajectories
     flowing toward them. Shows the draw basin, the contested ridge, and the
     conversion funnels directly.
  2. CERTAINTY PLANE  x = -ln P_win = d_certainty(s -> W surface),
     y = -ln P_loss. The coordinates the planner uses; surfaces on the axes.
  3. LEVEL SETS       committor contours (P_win = .25/.5/.75) over chess-
     native axes (material balance x ply) -- the "surfaces" as curves a chess
     player can read.

Usage: .venv/bin/python experiments/viz/committor_atlas.py \
    --ckpt data/derived/sep/cert_base_full.pt \
    --phead data/derived/sep/cert_base_full_phead.pt \
    --shards data/shards/lichess_db_standard_rated_2019-01.prefix4gb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phead_calibration import embed_games, load_holdout_games  # noqa: E402

# material balance from packed bitboards: planes 0-4 white P,N,B,R,Q; 6-10 black
_VALS = np.array([1, 3, 3, 5, 9], dtype=np.float32)

RES_COLOR = {1: "#2166ac", 0: "#888888", -1: "#b2182b"}   # won/draw/lost (White POV)


def material_balance(packed: np.ndarray) -> np.ndarray:
    w = np.stack([np.bitwise_count(packed[:, i]) for i in range(5)], 1).astype(np.float32)
    b = np.stack([np.bitwise_count(packed[:, i + 6]) for i in range(5)], 1).astype(np.float32)
    return (w - b) @ _VALS


def ternary_xy(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n,3) probs [W,D,L] -> 2D: corners W=(0,0), L=(1,0), D=(0.5, sqrt3/2)."""
    x = p[:, 2] * 1.0 + p[:, 1] * 0.5
    y = p[:, 1] * (np.sqrt(3) / 2)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--phead", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--max-games", type=int, default=400)
    ap.add_argument("--traj", type=int, default=120, help="trajectories to draw")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from catspace.nn.eval_head import EvalHead
    from catspace.nn.fb import load_ckpt, pick_device
    dev = pick_device(args.device)
    fb, _ = load_ckpt(Path(args.ckpt), dev)
    fb.eval()
    hp = torch.load(args.phead, map_location=dev, weights_only=False)
    phead = EvalHead(d_in=hp["d_in"]).to(dev)
    phead.load_state_dict(hp["state"]); phead.eval()

    games = load_holdout_games(Path(args.shards), args.max_games, args.seed)
    print(f"holdout games: {len(games)}  positions: {sum(len(g['ply']) for g in games)}")
    all_p, _ = embed_games(fb, phead, games, dev)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    # ---- 1. outcome simplex with trajectories --------------------------------
    ax = axes[0]
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(tri_x, tri_y, color="black", lw=1)
    ax.text(-0.03, -0.04, "WIN", ha="right", fontsize=11, color="#2166ac", weight="bold")
    ax.text(1.03, -0.04, "LOSS", ha="left", fontsize=11, color="#b2182b", weight="bold")
    ax.text(0.5, np.sqrt(3) / 2 + 0.03, "DRAW", ha="center", fontsize=11, color="#555", weight="bold")
    # density from all positions (faint), trajectories on top
    P_all = np.concatenate(all_p)
    x, y = ternary_xy(P_all)
    ax.hexbin(x, y, gridsize=45, cmap="Greys", bins="log", alpha=0.55, linewidths=0)
    rng = np.random.default_rng(3)
    for gi in rng.permutation(len(games))[: args.traj]:
        p = all_p[gi]
        if len(p) < 6:
            continue
        tx, ty = ternary_xy(p)
        c = RES_COLOR[games[gi]["result"]]
        ax.plot(tx, ty, color=c, alpha=0.30, lw=0.9)
        ax.plot(tx[-1], ty[-1], "o", color=c, ms=2.5, alpha=0.8)
    ax.set_title("outcome simplex: positions + game trajectories\n(surfaces = corners; dot = final position)")
    ax.set_aspect("equal"); ax.axis("off")

    # ---- 2. certainty plane: d_cert(W) vs d_cert(L) --------------------------
    ax = axes[1]
    eps = 1e-3
    dW = -np.log(np.clip(P_all[:, 0], eps, 1))
    dL = -np.log(np.clip(P_all[:, 2], eps, 1))
    res_all = np.concatenate([np.full(len(p), g["result"]) for p, g in zip(all_p, games)])
    for r in (1, 0, -1):
        m = res_all == r
        ax.scatter(dW[m], dL[m], s=2.5, alpha=0.25, c=RES_COLOR[r],
                   label={1: "won", 0: "drawn", -1: "lost"}[r])
    for gi in rng.permutation(len(games))[:18]:            # a few trajectories
        p = all_p[gi]
        ax.plot(-np.log(np.clip(p[:, 0], eps, 1)), -np.log(np.clip(p[:, 2], eps, 1)),
                color=RES_COLOR[games[gi]["result"]], alpha=0.5, lw=1.0)
    ax.axline((0, 0), slope=1, color="black", lw=0.6, ls=":")
    ax.set_xlabel(r"$-\ln P(\mathrm{win})$   (certainty distance to W surface)")
    ax.set_ylabel(r"$-\ln P(\mathrm{loss})$   (to L surface)")
    ax.legend(markerscale=5, loc="upper right")
    ax.set_title("certainty plane: surfaces sit ON the axes\n(below diagonal = White-favored)")

    # ---- 3. committor level sets over material x ply --------------------------
    ax = axes[2]
    mat = np.concatenate([material_balance(g["packed"]) for g in games])
    ply = np.concatenate([g["ply"].astype(float) for g in games])
    pw = P_all[:, 0]
    mb = np.arange(-9.5, 10.5, 1.0)
    pb = np.arange(0, 130, 10.0)
    H = np.full((len(mb) - 1, len(pb) - 1), np.nan)
    for i in range(len(mb) - 1):
        for j in range(len(pb) - 1):
            m = (mat >= mb[i]) & (mat < mb[i + 1]) & (ply >= pb[j]) & (ply < pb[j + 1])
            if m.sum() >= 30:
                H[i, j] = pw[m].mean()
    X, Y = np.meshgrid(0.5 * (pb[:-1] + pb[1:]), 0.5 * (mb[:-1] + mb[1:]))
    pc = ax.pcolormesh(X, Y, H, cmap="RdYlBu_r", vmin=0, vmax=1, shading="nearest")
    plt.colorbar(pc, ax=ax, label=r"mean $P(\mathrm{win})$")
    Hm = np.ma.masked_invalid(H)
    cs = ax.contour(X, Y, Hm, levels=[0.25, 0.5, 0.75], colors="black", linewidths=1.2)
    ax.clabel(cs, fmt="%.2f", fontsize=9)
    ax.set_xlabel("ply"); ax.set_ylabel("material balance (White − Black)")
    ax.set_title("committor level sets over chess-native axes\n(contours = the surfaces)")

    fig.tight_layout()
    out = args.out or f"artifacts/experiments/committor_atlas_{Path(args.ckpt).stem}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"ATLAS_PNG={out}")


if __name__ == "__main__":
    main()
