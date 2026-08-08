#!/usr/bin/env python
"""selfplay_probe.py -- engine vs itself, logging the three WHITE-POV pole distances every ply.

Kaveh 2026-08-08: "see how the best distances keep flipping and not decreasing." A committor
that reads positions correctly is still not a PLAN: along an optimal line for the winning side,
d(->their win pole) should shrink ply over ply. This probe measures exactly that -- per ply:
d_W, d_D, d_L (white-POV), the bar probs, and whose move it was; then summarizes monotonicity
(fraction of winner-side plies that decreased their win distance, longest monotone streak,
sign flips of the leader).

    .venv/bin/python -m ...selfplay_probe --ckpt <ckpt.pt> [--depth 2] [--plies 120]
"""
from __future__ import annotations

import argparse

import chess
import numpy as np

from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--plies", type=int, default=120)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    eng = KittyChess(args.ckpt, args.device)
    b = chess.Board()
    hist = []                                            # (ply, san, d_W, d_D, d_L, pW,pD,pB)
    for ply in range(args.plies):
        if b.is_game_over(claim_draw=True):
            break
        (pW, pD, pB), dists = eng.wdl(b)
        rows = eng.search(b, depth=args.depth)
        if not rows:
            break
        mv = rows[0]["mv"]
        san = b.san(mv)
        d = dists if dists is not None else [float("nan")] * 3
        hist.append((ply, "w" if b.turn else "b", san, *d, pW, pD, pB))
        b.push(mv)
    res = b.outcome(claim_draw=True)
    verdict = ("draw" if res and res.winner is None else
               ("white wins" if res and res.winner else
                ("black wins" if res else f"unfinished after {len(hist)} plies")))

    print(f"ply mover move      d->Wwin  d->draw  d->Bwin   P(w)  P(d)  P(b)  leader")
    for (ply, mover, san, dw, dd, dl, pw, pd, pb) in hist:
        lead = "W" if dw == min(dw, dd, dl) else ("D" if dd <= dl else "B")
        print(f"{ply:3d}  {mover}   {san:8s} {dw:7.2f}  {dd:7.2f}  {dl:7.2f}   "
              f"{pw:.2f}  {pd:.2f}  {pb:.2f}   {lead}")
    print(f"\n[game] {verdict}  ({len(hist)} plies, depth {args.depth})")

    dw = np.array([h[3] for h in hist], float)
    dl = np.array([h[5] for h in hist], float)
    dd_ = np.array([h[4] for h in hist], float)
    lead = np.argmin(np.stack([dw, dd_, dl], 1), 1)
    flips = int((np.diff(lead) != 0).sum())
    print(f"[flips] leader (nearest pole) changed {flips}x over {len(hist)-1} transitions")
    for name, series in (("d->Wwin", dw), ("d->Bwin", dl)):
        steps = np.diff(series)
        dec = float((steps < 0).mean()) if len(steps) else float("nan")
        # longest monotone-decreasing streak
        best = cur = 0
        for s in steps:
            cur = cur + 1 if s < 0 else 0
            best = max(best, cur)
        print(f"[mono] {name}: decreased on {dec:.0%} of plies; longest decreasing streak "
              f"{best}; net {series[-1]-series[0]:+.2f} over the game")


if __name__ == "__main__":
    main()
