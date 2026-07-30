#!/usr/bin/env python
"""tools/probe_quasimetric.py -- probe an asymmetric distance head (quasimetric).

The quasimetric literature (IQE/PQE/QRL: Wang & Isola 2022, Tongzhou Wang et al.)
defines what a GOOD learned quasimetric must show; each is a verdict here:

  asymmetry    : chess is irreversible (captures, pawn pushes) -- forward one-step
                 d(s_t -> s_{t+1}) should be systematically SMALLER than the
                 reverse d(s_{t+1} -> s_t). Report the asymmetry ratio distribution.
  monotonicity : along real games toward a fixed goal (the game's final position),
                 d(s_t -> s_T) should shrink as t -> T: per-game spearman(d, plies
                 remaining) -- the on-policy distance-recovery check.
  triangle     : sampled in-game triples t<u<v: violation margin
                 d(t->v) - [d(t->u) + d(u->v)] must be <= 0. IQE guarantees this
                 by construction -- a nonzero rate = implementation bug; learned
                 unconstrained heads (MRN-style) report their honest rate.

Distance heads: --head trunk (ReachabilityField.d, the T1-IQE field) or
--head clockfield (ClockField.d_pair via its committor ckpt).
Source: PGN file(s) of real games. Figure: 3-panel (--fig).

Usage: tools/probe_quasimetric.py --pgn artifacts/experiments/m5_read100.pgn \
           --head trunk --games 40 --fig /tmp/quasi.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess.pgn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_head(name, dev):
    if name == "trunk":
        from catspace.encoder import ReachabilityField
        rf = ReachabilityField()

        def emb(boards):
            return rf.phi(boards)

        def d_pair(es, eg):
            import torch
            with torch.no_grad():
                return rf.head.d_pair_emb(es, eg).cpu().numpy()
        return emb, d_pair
    else:
        import torch
        from catspace.value import CommittorGreedy
        cg = CommittorGreedy("artifacts/experiments/field_fullgame_v3_final.pt", dev)

        def emb(boards):
            x = torch.stack([torch.from_numpy(
                np.asarray(b.to_input_tensor().float().numpy())) for b in boards]).to(dev)
            return cg.net.phi(x)

        def d_pair(es, eg):
            return cg.net.iqe(es, eg).detach().cpu().numpy()
        return emb, d_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", nargs="+", required=True)
    ap.add_argument("--head", choices=["trunk", "clockfield"], default="trunk")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--triples", type=int, default=30, help="per game")
    ap.add_argument("--fig", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    from catspace.train.scaffold import resolve_device
    from lczerolens import LczeroBoard
    dev = resolve_device("auto")
    emb, d_pair = load_head(args.head, dev)

    fwd, bwd, monos, viol = [], [], [], []
    n_games = 0
    for path in args.pgn:
        with open(path) as fh:
            while n_games < args.games:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                boards = []
                b = LczeroBoard()
                for mv in game.mainline_moves():
                    boards.append(b.copy(stack=False)); b.push(mv)
                boards.append(b.copy(stack=False))
                if len(boards) < 20:
                    continue
                n_games += 1
                E = emb(boards)
                # one-step asymmetry
                fwd.extend(d_pair(E[:-1], E[1:]).tolist())
                bwd.extend(d_pair(E[1:], E[:-1]).tolist())
                # monotonicity toward the final position
                dg = d_pair(E[:-1], E[-1:].expand(len(boards) - 1, -1))
                rem = np.arange(len(dg))[::-1] + 1
                from scipy.stats import spearmanr
                monos.append(float(spearmanr(dg, rem).statistic))
                # triangle triples
                for _ in range(args.triples):
                    t, u, v = np.sort(rng.choice(len(boards), 3, replace=False))
                    if t == u or u == v:
                        continue
                    dv = d_pair(E[t:t+1], E[v:v+1])[0]
                    du = d_pair(E[t:t+1], E[u:u+1])[0] + d_pair(E[u:u+1], E[v:v+1])[0]
                    viol.append(float(dv - du))
        if n_games >= args.games:
            break
    fwd = np.array(fwd); bwd = np.array(bwd); viol = np.array(viol)
    ratio = bwd / np.maximum(fwd, 1e-9)
    print(f"VERDICT quasimetric[{args.head}] asymmetry: one-step fwd median "
          f"{np.median(fwd):.3f} vs bwd {np.median(bwd):.3f} | ratio median "
          f"{np.median(ratio):.2f} (>1 = irreversibility captured) | "
          f"P(bwd>fwd) {np.mean(bwd > fwd):.1%}")
    print(f"VERDICT quasimetric[{args.head}] monotonicity: per-game spearman(d, "
          f"plies-remaining) median {np.median(monos):+.3f} "
          f"(+1 ideal) | frac>0 {np.mean(np.array(monos) > 0):.1%} | n={len(monos)}")
    print(f"VERDICT quasimetric[{args.head}] triangle: violation rate "
          f"{np.mean(viol > 1e-6):.2%} | margin p95 {np.percentile(viol, 95):+.4f} "
          f"(IQE heads must be ~0 by construction)")
    if args.fig:
        from tools import figlib
        fig, ax = figlib.new_fig(3)
        ax[0].hist(np.log(np.clip(ratio, 1e-3, 1e3)), bins=40,
                   color=figlib.ACCENT, edgecolor="none")
        ax[0].axvline(0, color=figlib.MUTED, lw=1)
        ax[0].set_xlabel("log(bwd/fwd) one-step"); ax[0].set_title("Asymmetry")
        ax[1].hist(monos, bins=20, color=figlib.ACCENT, edgecolor="none")
        ax[1].axvline(0, color=figlib.MUTED, lw=1)
        ax[1].set_xlabel("spearman(d, plies remaining)"); ax[1].set_title("Monotonicity")
        ax[2].hist(viol, bins=40, color=figlib.ACCENT, edgecolor="none")
        ax[2].axvline(0, color=figlib.MUTED, lw=1)
        ax[2].set_xlabel("d(t,v) − [d(t,u)+d(u,v)]"); ax[2].set_title("Triangle margin")
        figlib.save(fig, args.fig, f"Quasimetric probe — {args.head}")


if __name__ == "__main__":
    main()
