#!/usr/bin/env python
"""audit_data_balance.py -- corpus composition audit (Kaveh 2026-08-12: "make sure our training
data is balanced in terms of opening midgame endgame and endgame types and material classes and
win loss draw outcomes").

Criteria (stated, per the define-identifications rule):
  phase     opening = ply < 16; endgame = total pieces <= 10; middlegame = the rest.
  endgame   positions with <= 6 pieces, typed by their non-king material multiset
            (both colors pooled, e.g. "RP", "Q", "pawns-only", "none" = K vs K).
  material  white-minus-black point balance d in classes: even |d|<=1, minor 1<|d|<=3,
            piece 3<|d|<=5, heavy |d|>5 (each signed).
  outcome   per-POSITION W/D/L label (the game's outcome, white POV) -- what basin CE sees.

Prints per-class position counts and shares, the per-cell outcome mix, and an IMBALANCE verdict
per axis (max/min share ratio). Optionally --save writes per-row class ids + inverse-frequency
sampling weights for the trainer's balanced-sampling flag.

    .venv/bin/python -m ...audit_data_balance --games 4000 --sf-only --n-piecedown 27006
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T

VAL = {0: 0, 1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0, 7: 1, 8: 3, 9: 3, 10: 5, 11: 9, 12: 0}
LETTER = {2: "N", 3: "B", 4: "R", 5: "Q", 8: "N", 9: "B", 10: "R", 11: "Q"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--sf-only", action="store_true", default=True)
    ap.add_argument("--n-piecedown", type=int, default=27006)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true",
                    help="write per-row class ids + inverse-frequency weights npz")
    args = ap.parse_args()

    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, max_plies=args.max_plies,
                 n_piecedown=args.n_piecedown, verbose=False)
    tok = tr.tok
    ply = tr.ply_of_row()
    gor = tr.game_of_row()
    y = tr.outcome_of_row_white()                         # per position, white POV
    N = len(tok)
    v = np.vectorize(VAL.get)(tok)
    wm = np.where((tok >= 1) & (tok <= 6), v, 0).sum(1)
    bm = np.where(tok >= 7, v, 0).sum(1)
    d = wm - bm
    npc = (tok > 0).sum(1)

    print(f"[balance] {N:,} positions, {len(tr):,} games")

    def table(name, labels, ids, ysub=None):
        yy = y if ysub is None else ysub
        cnt = Counter(ids)
        tot = sum(cnt.values())
        shares = {}
        print(f"\n== {name} ==")
        for k in labels:
            n = cnt.get(k, 0)
            shares[k] = n / tot
            wdl = ""
            if n:
                m = ids == k
                wdl = (f"  W {np.mean(yy[m] == T.WIN):.0%} / D {np.mean(yy[m] == T.DRAW):.0%}"
                       f" / L {np.mean(yy[m] == T.LOSS):.0%}")
            print(f"  {k:<16s} {n:>10,}  ({n/tot:5.1%}){wdl}")
        nz = [s for s in shares.values() if s > 0]
        ratio = max(nz) / max(min(nz), 1e-9) if len(nz) > 1 else 1.0
        print(f"  VERDICT {name}: max/min share ratio {ratio:.1f}x "
              f"{'(BALANCED <3x)' if ratio < 3 else '(IMBALANCED)'}")
        return shares

    # phase
    phase = np.where(ply < 16, "opening", np.where(npc <= 10, "endgame", "middlegame"))
    table("phase", ["opening", "middlegame", "endgame"], phase)

    # outcome (per position)
    oname = np.where(y == T.WIN, "white-wins", np.where(y == T.DRAW, "draw",
                     np.where(y == T.LOSS, "black-wins", "censored")))
    table("outcome", ["white-wins", "draw", "black-wins", "censored"], oname)

    # material class (signed)
    mcls = np.select(
        [d > 5, d > 3, d > 1, d >= -1, d >= -3, d >= -5],
        ["W-heavy", "W-piece", "W-minor", "even", "B-minor", "B-piece"], default="B-heavy")
    table("material", ["W-heavy", "W-piece", "W-minor", "even",
                       "B-minor", "B-piece", "B-heavy"], mcls)

    # endgame types (<=6 pieces)
    eg = npc <= 6
    sig = np.full(N, "", dtype=object)
    if eg.any():
        rows = np.flatnonzero(eg)
        for r in rows:
            pcs = sorted(LETTER[t] for t in tok[r] if t in LETTER)
            pn = int(((tok[r] == 1) | (tok[r] == 7)).sum())
            s = "".join(pcs) + ("P" if pn else "")
            sig[r] = s if s else "KvK"
        top = [k for k, _ in Counter(sig[rows]).most_common(12)]
        table("endgame type (<=6 pc)", top, sig[rows], ysub=y[rows])
        print(f"  (endgame rows: {eg.sum():,} = {eg.mean():.1%} of corpus; "
              f"{len(set(sig[rows]))} distinct signatures)")

    if args.save:
        # joint stratum = phase x outcome x material-class; weight = 1/freq, clipped
        strat = np.char.add(np.char.add(phase.astype(str), "|" + oname.astype(str)),
                            "|" + mcls.astype(str))
        uniq, inv, cnts = np.unique(strat, return_inverse=True, return_counts=True)
        w = (N / (len(uniq) * cnts))[inv]
        w = np.clip(w, 0.1, 10.0).astype(np.float32)
        out = paths.derived(f"balance_weights_{N}.npz")
        np.savez(out, weight=w, stratum=inv.astype(np.int32),
                 names=np.array([str(u) for u in uniq]))
        print(f"\n[balance] saved inverse-frequency weights (clipped [0.1,10]) -> {out}")
        print(f"[balance] {len(uniq)} strata; weight quantiles "
              f"p10 {np.percentile(w,10):.2f} p50 {np.percentile(w,50):.2f} "
              f"p90 {np.percentile(w,90):.2f}")


if __name__ == "__main__":
    main()
