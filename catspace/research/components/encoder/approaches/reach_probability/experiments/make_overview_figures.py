#!/usr/bin/env python
"""make_overview_figures.py -- figures for the architecture overview artifact (2026-08-12).

  1. ribbon.png       -- concept profile (8 VQ codes) + committor over a REAL game
                         (Kaveh vs catspace: queen sac -> mating attack, 29.d4 Rh1#)
  2. interactions.png -- the 512x512 concept-interaction matrix, log-lift heatmap
  3. reliability.png  -- committor calibration: predicted P(white) vs empirical frequency
  4. descent.png      -- the three length-ruler pole distances along the same game

    .venv/bin/python -m ...make_overview_figures --ckpt artifacts/experiments/reach_v2_latest.pt
"""
from __future__ import annotations

import argparse
import os

import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.concept_vq import (
    ConceptVQ)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize

# Kaveh vs catspace, 2026-08-11 (from the live game log): queen sac into a mating attack.
GAME_SAN = ("e4 Nc6 a4 d5 exd5 Qxd5 Nc3 Qd8 Qe2 e6 Qh5 Nf6 Qg5 Qd4 Be2 Bd7 a5 O-O-O "
            "Qxg7 Bxg7 Nd1 Ne4 h3 Qxf2+ Nxf2 Nxf2 Kxf2 Bd4+ Kf3 Ne5+ Kf4 Ng6+ Kg3 Bc6 "
            "c3 Be5+ Kf2 Nf4 Bc4 Nd3+ Bxd3 Rxd3 Ne2 Rg8 Rg1 Bg3+ Nxg3 Rgxg3 h4 Kd7 c4 "
            "Kd6 Rd1 Rxg2+ Ke1 Rh3 d4 Rh1#").split()

INK, DIM, ACC, BG = "#1e2126", "#5c6370", "#1f6f54", "#fbfaf7"


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(DIM)
    ax.tick_params(colors=DIM, labelsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    out_dir = paths.experiment("overview_figs")
    os.makedirs(out_dir, exist_ok=True)
    base = args.ckpt[:-3]
    net, pay = load_net(args.ckpt, args.device)
    c = pay["cfg"]
    pv = torch.load(base + "_vq.pt", map_location=args.device, weights_only=False)
    vq = ConceptVQ(d_in=pv["d_in"], heads=pv["heads"], codes=pv["codes"]).to(args.device)
    vq.load_state_dict(pv["state_dict"]); vq.eval()
    pn = c["pole_names"]
    P = net.poles.poles.detach().float().to(args.device)
    pidx = [pn.index(k) for k in ("WIN", "DRAW", "LOSS")]

    # ---- embed the demo game ----
    b = chess.Board()
    boards = [b.copy()]
    for san in GAME_SAN:
        b.push_san(san)
        boards.append(b.copy())
    toks, globs = zip(*(tokenize(x) for x in boards))
    with torch.no_grad():
        phi = net.backbone(torch.from_numpy(np.array(toks).astype(np.int64)).to(args.device),
                           torch.from_numpy(np.array(globs).astype(np.float32)).to(args.device))
        _, ids, _ = vq(phi)
        z = net.encode_q(torch.from_numpy(np.array(toks).astype(np.int64)).to(args.device),
                         torch.from_numpy(np.array(globs).astype(np.float32)).to(args.device))
        DB = torch.stack([net.dB(z, P[[k]].expand(len(z), -1)) for k in pidx], 1)
        pr = torch.softmax(-DB / 5.0, 1).float().cpu().numpy()
        DA = torch.stack([net.dA(z, P[[k]].expand(len(z), -1)) for k in pidx], 1)
        DA = DA.float().cpu().numpy()
    ids = ids.cpu().numpy()
    E = pr[:, 0] + 0.5 * pr[:, 1]
    n = len(boards)

    # ---- 1. ribbon ----
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8.2, 3.6), dpi=150,
                                 gridspec_kw={"height_ratios": [1.2, 2.2], "hspace": 0.12})
    fig.patch.set_facecolor(BG)
    a0.fill_between(range(n), 0, pr[:, 0], color="#e8e5df", label="P(white)")
    a0.fill_between(range(n), pr[:, 0], pr[:, 0] + pr[:, 1], color="#b9b4aa", label="P(draw)")
    a0.fill_between(range(n), pr[:, 0] + pr[:, 1], 1, color="#3b3f45", label="P(black)")
    a0.set_xlim(0, n - 1); a0.set_ylim(0, 1); a0.set_xticks([])
    a0.set_ylabel("committor", fontsize=8, color=DIM)
    style(a0)
    a0.set_title("a real game through the model's eyes  (human queen-sacs on move 12 and mates on 29)",
                 fontsize=9.5, color=INK, loc="left")
    rng = np.random.default_rng(7)
    palette = rng.permutation(64)
    img = np.zeros((pv["heads"], n))
    for h in range(pv["heads"]):
        img[h] = palette[ids[:, h]]
    a1.imshow(img, aspect="auto", cmap="tab20", interpolation="nearest")
    a1.set_yticks(range(pv["heads"]))
    a1.set_yticklabels([f"codebook {h}" for h in range(pv["heads"])], fontsize=7)
    a1.set_xlabel("ply (half-move)", fontsize=8, color=DIM)
    style(a1)
    for t in (23, 56):                                    # queen sac ply ~23 (12.Qxf2+ ...), mate
        for ax in (a0, a1):
            ax.axvline(t, color=ACC, lw=0.8, ls="--", alpha=0.8)
    fig.savefig(os.path.join(out_dir, "ribbon.png"), bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)

    # ---- 2. interaction heatmap ----
    ix = np.load(base + "_interactions.npz")
    L = ix["lift"].copy()
    L[~np.isfinite(L)] = 1.0
    L[L <= 0] = 1e-3
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=150)
    fig.patch.set_facecolor(BG)
    im = ax.imshow(np.log10(L), cmap="RdBu_r", vmin=-1.2, vmax=1.2, interpolation="nearest")
    K = L.shape[0]; C = pv["codes"]
    for h in range(1, pv["heads"]):
        ax.axhline(h * C - .5, color=BG, lw=0.6)
        ax.axvline(h * C - .5, color=BG, lw=0.6)
    ax.set_xlabel("concept b (activates next)", fontsize=8, color=DIM)
    ax.set_ylabel("concept a (currently active)", fontsize=8, color=DIM)
    ax.set_title("concept-interaction matrix: log10 lift(a→b)\nred = synergy, blue = blocking; grid = the 8 codebooks",
                 fontsize=9.5, color=INK, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.ax.tick_params(labelsize=7, colors=DIM)
    fig.savefig(os.path.join(out_dir, "interactions.png"), bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    # ---- 3. reliability ----
    tr = T.build(n_human=0, n_sf=c["games"], seed=c["traj_seed"], max_plies=c["max_plies"],
                 n_piecedown=c.get("n_piecedown", 0), verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    game = tr.game_of_row(); y_all = tr.outcome_of_row_white()
    rows = np.flatnonzero(np.isin(game, np.flatnonzero(split == 1)) & (y_all >= 0))
    r2 = np.random.default_rng(0)
    rows = rows[r2.choice(len(rows), 6000, replace=False)]
    with torch.no_grad():
        zv = net.encode_q(torch.from_numpy(tr.tok[rows].astype(np.int64)).to(args.device),
                          torch.from_numpy(tr.glob[rows].astype(np.float32)).to(args.device))
        DBv = torch.stack([net.dB(zv, P[[k]].expand(len(zv), -1)) for k in pidx], 1)
        prv = torch.softmax(-DBv / 5.0, 1).float().cpu().numpy()
    Pw = prv[:, 0]; won = (y_all[rows] == 0).astype(float)
    bins = np.linspace(0, 1, 11)
    xs, ys, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (Pw >= lo) & (Pw < hi)
        if m.sum() >= 30:
            xs.append(Pw[m].mean()); ys.append(won[m].mean()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(3.9, 3.6), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.plot([0, 1], [0, 1], color=DIM, lw=0.8, ls="--")
    ax.plot(xs, ys, "o-", color=ACC, ms=4, lw=1.4)
    ax.set_xlabel("predicted P(white wins)", fontsize=8, color=DIM)
    ax.set_ylabel("empirical frequency", fontsize=8, color=DIM)
    ax.set_title("committor reliability (held-out)", fontsize=9.5, color=INK, loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    style(ax)
    fig.savefig(os.path.join(out_dir, "reliability.png"), bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    # ---- 4. descent ----
    fig, ax = plt.subplots(figsize=(8.2, 2.3), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.plot(DA[:, 0], color="#8a8580", lw=1.3, label="dA → white-wins")
    ax.plot(DA[:, 1], color=DIM, lw=1.0, ls=":", label="dA → draw")
    ax.plot(DA[:, 2], color=INK, lw=1.3, label="dA → black-wins")
    ax.axvline(23, color=ACC, lw=0.8, ls="--", alpha=0.8)
    ax.axvline(56, color=ACC, lw=0.8, ls="--", alpha=0.8)
    ax.text(23.5, ax.get_ylim()[1] * 0.9, "queen sac", fontsize=7, color=ACC)
    ax.set_xlabel("ply", fontsize=8, color=DIM)
    ax.set_ylabel("plies to ending", fontsize=8, color=DIM)
    ax.set_title("the length ruler along the same game: black's exit approaches as the attack lands",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, labelcolor=DIM)
    style(ax)
    fig.savefig(os.path.join(out_dir, "descent.png"), bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[figs] wrote 4 figures -> {out_dir}")


if __name__ == "__main__":
    main()
