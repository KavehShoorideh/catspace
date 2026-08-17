#!/usr/bin/env python
"""square_control.py -- who controls how much of the board, ply by ply? (Kaveh 2026-08-17)

Space is one of the few chess concepts with an unambiguous, engine-free definition, which makes
it a useful yardstick: if the learned concept vocabulary contains anything like "I have a space
advantage", it should track a curve like this one.

CONTROL IS DEFINED THREE WAYS, because the obvious one is not the interesting one:
    attacked      squares a side attacks at all (double-counts nothing, but ignores contest)
    dominated     squares where a side has STRICTLY MORE attackers than the other
                  -- this is the one that means "mine": I can win an exchange there
    contested     squares both sides attack equally, i.e. the fighting front

`board.attackers(colour, sq)` counts defenders of one's own pieces too, which is correct here:
controlling a square you occupy is still control.

    .venv/bin/python -m catspace.research.tools.viz.square_control --games 5
"""
from __future__ import annotations

import argparse

import numpy as np


def control_series(board, ucis, max_plies=400):
    """-> dict of per-ply arrays: white/black attacked, dominated, contested, material."""
    import chess
    VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
           chess.QUEEN: 9, chess.KING: 0}
    w_att, b_att, w_dom, b_dom, cont, mat = [], [], [], [], [], []
    for i, u in enumerate(ucis[:max_plies]):
        wa = ba = wd = bd = ct = 0
        for sq in range(64):
            nw = len(board.attackers(chess.WHITE, sq))
            nb = len(board.attackers(chess.BLACK, sq))
            wa += nw > 0; ba += nb > 0
            if nw > nb:
                wd += 1
            elif nb > nw:
                bd += 1
            elif nw > 0:
                ct += 1                     # equally attacked AND attacked at all
        m = sum(VAL[p.piece_type] * (1 if p.color else -1)
                for p in board.piece_map().values())
        w_att.append(wa); b_att.append(ba); w_dom.append(wd); b_dom.append(bd)
        cont.append(ct); mat.append(m)
        try:
            mv = chess.Move.from_uci(u)
            if mv not in board.legal_moves:
                break
            board.push(mv)
        except Exception:
            break
    return {k: np.array(v) for k, v in
            dict(w_att=w_att, b_att=b_att, w_dom=w_dom, b_dom=b_dom,
                 contested=cont, material=mat).items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--out", default="artifacts/experiments/square_control.png")
    args = ap.parse_args()

    import chess
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)

    rows = T.load_sf_games(400, args.seed, max_plies=args.max_plies)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(rows), args.games, replace=False)
    RES = {1: "white wins", 0: "draw", -1: "black wins"}

    fig, axes = plt.subplots(args.games, 2, figsize=(13, 2.7 * args.games), squeeze=False)
    for r, gi in enumerate(pick):
        g = rows[gi]
        gid, result, ucis = g[0], g[1], g[2]
        s = control_series(chess.Board(), ucis, args.max_plies)
        x = np.arange(len(s["w_dom"]))
        print(f"  game {gid} ({RES.get(result,'?'):<11}) {len(x):>3} plies | "
              f"dominated mean W {s['w_dom'].mean():5.1f} B {s['b_dom'].mean():5.1f} | "
              f"contested mean {s['contested'].mean():5.1f}", flush=True)

        ax = axes[r][0]
        ax.plot(x, s["w_dom"], color="#1f77b4", lw=1.6, label="white dominates")
        ax.plot(x, s["b_dom"], color="#d62728", lw=1.6, label="black dominates")
        ax.plot(x, s["contested"], color="#7f7f7f", lw=1.0, ls="--", label="contested")
        ax.set_ylabel("squares"); ax.set_title(
            f"game {gid} — {RES.get(result,'?')} — DOMINATED (strictly more attackers)",
            fontsize=9)
        ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=.25)

        ax = axes[r][1]
        ax.plot(x, s["w_att"], color="#1f77b4", lw=1.6, label="white attacks")
        ax.plot(x, s["b_att"], color="#d62728", lw=1.6, label="black attacks")
        ax2 = ax.twinx()
        ax2.plot(x, s["material"], color="#2ca02c", lw=1.2, alpha=.7, label="material (W−B)")
        ax2.axhline(0, color="#2ca02c", lw=.6, alpha=.4)
        ax2.set_ylabel("material", color="#2ca02c", fontsize=8)
        ax.set_title(f"game {gid} — ATTACKED (any attacker) + material", fontsize=9)
        ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=.25)
        if r == args.games - 1:
            axes[r][0].set_xlabel("ply"); axes[r][1].set_xlabel("ply")

    fig.suptitle("Square control through the game — 5 Stockfish-vs-Stockfish games",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(args.out, dpi=125)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
