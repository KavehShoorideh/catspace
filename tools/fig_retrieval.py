#!/usr/bin/env python
"""tools/fig_retrieval.py -- nearest-neighbour retrieval grid, FAIR convention
(I-JEPA/LeJEPA): each row = a query + its top-k cosine neighbours in frozen
embedding space, neighbour panels OUTLINED GREEN when their label matches the
query's and RED otherwise. Boards drawn as 8x8 glyph panels.

Source: the mined checkpoints npz (fens + labels ride along); encoder = a JEPA
checkpoint or the frozen trunk.

Usage:
  tools/fig_retrieval.py --checkpoints data/derived/checkpoints/checkpoints_v1_full.npz \
      --encoder jepa --ckpt artifacts/experiments/jepa_t1_latest.pt \
      --label region --queries 5 --k 5 --fig retrieval.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import figlib                                                # noqa: E402

GLYPH = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
         "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}
GOOD, BAD = "#2FA089", "#C0392B"          # status colors: match / mismatch


def draw_board(ax, fen, edge_color, title=""):
    board = chess.Board(fen)
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        light = (f + r) % 2 == 1
        ax.add_patch(__import__("matplotlib").patches.Rectangle(
            (f, r), 1, 1, facecolor="#f0e9dd" if light else "#b5a48a", lw=0))
        pc = board.piece_at(sq)
        if pc:
            ax.text(f + 0.5, r + 0.42, GLYPH[pc.symbol()], ha="center", va="center",
                    fontsize=11, color="#111")
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color(edge_color); spine.set_linewidth(2.5)
    ax.set_xlim(0, 8); ax.set_ylim(0, 8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if title:
        ax.set_title(title, fontsize=7, color=figlib.INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="data/derived/checkpoints/checkpoints_v1_full.npz")
    ap.add_argument("--encoder", choices=["jepa", "trunk"], default="jepa")
    ap.add_argument("--ckpt", default="artifacts/experiments/jepa_t1_latest.pt")
    ap.add_argument("--label", default="elo_victim",
                    help="ck_ column for match/mismatch (continuous cols get banded)")
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--queries", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fig", default="retrieval.png")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    d = dict(np.load(args.checkpoints, allow_pickle=True))
    n = len(d["ck_fen"])
    sel = np.sort(rng.choice(n, min(args.sample, n), replace=False))
    fens = d["ck_fen"][sel]
    lab = d["ck_" + args.label][sel] if "ck_" + args.label in d else d[args.label][sel]
    if lab.dtype.kind == "f" or len(np.unique(lab)) > 30:
        lab = np.digitize(lab, np.quantile(lab, [0.25, 0.5, 0.75]))   # band continuous
    if args.encoder == "trunk":
        from tools.embed import trunk_encode
        E = trunk_encode(list(fens))
    else:
        import torch
        from catspace.encoder.jepa import tokenize
        from tools.embed import jepa_encode
        from catspace.train.scaffold import resolve_device
        tg = [tokenize(chess.Board(f)) for f in fens]
        E = jepa_encode(args.ckpt, np.stack([t for t, _ in tg]),
                        np.stack([g for _, g in tg]), resolve_device("auto"))
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    qs = rng.choice(len(fens), args.queries, replace=False)
    fig, axes = figlib.new_fig(args.k + 1, args.queries, w=1.6, h=1.75)
    axes = np.atleast_2d(axes)
    match_rate = []
    for row, q in enumerate(qs):
        sim = E @ E[q]
        sim[q] = -np.inf
        nn = np.argsort(-sim)[:args.k]
        draw_board(axes[row, 0], fens[q], figlib.INK, f"query [{lab[q]}]")
        for j, i in enumerate(nn):
            ok = lab[i] == lab[q]
            match_rate.append(ok)
            draw_board(axes[row, j + 1], fens[i], GOOD if ok else BAD,
                       f"cos {sim[i]:.2f}")
    print(f"VERDICT retrieval[{args.encoder}, {args.label}]: top-{args.k} label-match "
          f"{np.mean(match_rate):.1%} over {args.queries} queries "
          f"(chance ~ {np.mean([np.mean(lab == v) for v in np.unique(lab)]):.1%})")
    figlib.save(fig, args.fig,
                f"Nearest-neighbour retrieval — {args.encoder}, outline = {args.label} match")


if __name__ == "__main__":
    main()
