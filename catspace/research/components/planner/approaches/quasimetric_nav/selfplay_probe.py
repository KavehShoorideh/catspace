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
    ap.add_argument("--games", type=int, default=1,
                    help=">1: aggregate stats over games from random 4-ply opening prefixes "
                         "(the engine is deterministic; one game is one sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="suppress the per-ply table")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    eng = KittyChess(args.ckpt, args.device)

    def play_one(prefix_plies):
        b = chess.Board()
        for _ in range(prefix_plies):
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rng.choice(ms))
        hist = []
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
        return hist, verdict

    all_dec, all_flip, all_streak, results = [], [], [], []
    for gi in range(args.games):
        hist, verdict = play_one(0 if args.games == 1 else 4)
        results.append(verdict)
        if not args.quiet and args.games == 1:
            print("ply mover move      d->Wwin  d->draw  d->Bwin   P(w)  P(d)  P(b)  leader")
            for (ply, mover, san, dw, dd, dl, pw, pd, pb) in hist:
                lead = "W" if dw == min(dw, dd, dl) else ("D" if dd <= dl else "B")
                print(f"{ply:3d}  {mover}   {san:8s} {dw:7.2f}  {dd:7.2f}  {dl:7.2f}   "
                      f"{pw:.2f}  {pd:.2f}  {pb:.2f}   {lead}")
        if len(hist) < 6:
            continue
        dw = np.array([h[3] for h in hist], float)
        dl = np.array([h[5] for h in hist], float)
        dd_ = np.array([h[4] for h in hist], float)
        lead = np.argmin(np.stack([dw, dd_, dl], 1), 1)
        all_flip.append(float((np.diff(lead) != 0).mean()))
        for series in (dw, dl):
            steps = np.diff(series)
            all_dec.append(float((steps < 0).mean()))
            best = cur = 0
            for s in steps:
                cur = cur + 1 if s < 0 else 0
                best = max(best, cur)
            all_streak.append(best)
        print(f"[game {gi}] {verdict} ({len(hist)} plies)")

    print(f"\n[summary over {args.games} game(s), depth {args.depth}]")
    print(f"  results: {dict((r, results.count(r)) for r in set(results))}")
    if all_dec:
        print(f"  win-distance decreased on {np.mean(all_dec):.0%} of plies (chance ~50%)")
        print(f"  longest decreasing streak: mean {np.mean(all_streak):.1f}  max {max(all_streak)}")
        print(f"  leader flip rate: {np.mean(all_flip):.0%} of transitions")


if __name__ == "__main__":
    main()
