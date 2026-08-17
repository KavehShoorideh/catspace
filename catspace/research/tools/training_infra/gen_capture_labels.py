#!/usr/bin/env python
"""gen_capture_labels.py -- the next N captures, as (attacker, victim, square). (Kaveh 2026-08-17)

"have a set of concepts that allow us to predict the next five captures... I don't wanna capture
the entire board, but I wanna capture where the action is, and the action is the capture."
And: "both piece types, e.g. knight takes pawn on e5".

WHY CAPTURES. The measured failure of the current vocabulary is that it encodes VALUE but has
discarded MATERIAL COMPOSITION -- it knows a position is winning but not that it is heading for
KRP|KR (0.138 against material's 0.213 at 20 plies). Captures are precisely the events that
change material, so predicting them forces the concepts to keep what a tablebase class name is
made of. They are also dense (18.6% of plies) rather than terminal, and the 5th capture sits a
median 20 plies ahead -- the planning range.

WHY ALL THREE FIELDS. Predicting the victim type alone is nearly free: 49% of captures are
pawns, so "a pawn will go" is a strong constant baseline. The SQUARE is where the information
is (1/64), and the ATTACKER distinguishes 'knight takes pawn on e5' from 'queen takes pawn on
e5' -- a different proposition entirely. Emitted separately so each can be scored against its
own baseline rather than hidden inside a joint 4,608-way class.

Emitted per position row, for each of the next N captures:
    cap_sq     int8   destination square 0-63, -1 if fewer than N captures remain
    cap_vic    int8   victim piece id 1-12 (jepa vocab), -1 if none
    cap_att    int8   attacker piece id 1-12 AFTER the move (promotion shows the new piece)
    cap_ply    int16  plies from this row until that capture, -1 if none

    .venv/bin/python -m ...gen_capture_labels --n-caps 5 --save
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-caps", type=int, default=5)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    tok, n = tr.tok, len(tr.tok)
    gor = tr.game_of_row()
    gz = np.load(paths.derived(f"game_transitions_FULL_{n}.npz"))
    par, mid = gz["par"], gz["mid"]
    men = (tok != 0).sum(1)

    # ---- find every capture and describe it ------------------------------------------
    mv_to = np.full(n, -1, np.int64)
    mv_to[par] = mid[:, 1]
    is_cap = np.zeros(n, bool)
    is_cap[par] = men[par + 1] < men[par]
    caps = np.flatnonzero(is_cap)
    sq = mv_to[caps]
    vic = tok[caps, sq].astype(np.int8)                 # piece standing on the target square
    att = tok[caps + 1, sq].astype(np.int8)             # who is standing there afterwards
    # EN PASSANT: the target square was EMPTY yet a man vanished. The victim is a pawn beside
    # it, and its colour is the opposite of the capturer's -- reconstructed rather than dropped,
    # since ep is ~1% of captures and silently mislabelling them would poison the pawn class.
    ep = vic == 0
    vic[ep] = np.where(att[ep] <= 6, 7, 1)              # white captured -> black pawn, and back
    print(f"[cap] {len(caps):,} captures ({100*len(caps)/len(par):.1f}% of plies) | "
          f"en-passant {int(ep.sum()):,} ({100*ep.mean():.2f}%)", flush=True)

    # ---- for every row, the next N captures in its own game ---------------------------
    N = args.n_caps
    cap_sq = np.full((n, N), -1, np.int8)
    cap_vic = np.full((n, N), -1, np.int8)
    cap_att = np.full((n, N), -1, np.int8)
    cap_ply = np.full((n, N), -1, np.int16)
    # walk each game backwards, carrying the upcoming capture list
    order = np.argsort(caps)
    caps_s, sq_s, vic_s, att_s = caps[order], sq[order], vic[order], att[order]
    ptr = {}
    for gi in range(len(tr.start)):
        s0, ln = int(tr.start[gi]), int(tr.length[gi])
        lo = np.searchsorted(caps_s, s0); hi = np.searchsorted(caps_s, s0 + ln)
        gc, gsq, gv, ga = caps_s[lo:hi], sq_s[lo:hi], vic_s[lo:hi], att_s[lo:hi]
        if not len(gc):
            continue
        j = 0
        for r in range(s0, s0 + ln):
            while j < len(gc) and gc[j] < r:
                j += 1
            k = min(N, len(gc) - j)
            if k <= 0:
                continue
            cap_sq[r, :k] = gsq[j:j + k]
            cap_vic[r, :k] = gv[j:j + k]
            cap_att[r, :k] = ga[j:j + k]
            cap_ply[r, :k] = (gc[j:j + k] - r).astype(np.int16)

    have = (cap_sq[:, N - 1] >= 0)
    print(f"[cap] positions with all {N} captures ahead: {int(have.sum()):,} "
          f"({100*have.mean():.1f}%)", flush=True)
    d = cap_ply[have, N - 1]
    print(f"[cap] plies to the {N}th capture: median {np.median(d):.0f} "
          f"p25 {np.percentile(d,25):.0f} p75 {np.percentile(d,75):.0f} max {d.max()}")
    L = {1:"P",2:"N",3:"B",4:"R",5:"Q",6:"K",7:"p",8:"n",9:"b",10:"r",11:"q",12:"k"}
    c1 = Counter(cap_vic[have, 0].tolist()); t1 = sum(c1.values())
    ca = Counter(cap_att[have, 0].tolist())
    print(f"\n[cap] the NEXT capture -- baselines any model must beat:")
    print(f"    victim   most common {L[c1.most_common(1)[0][0]]} at "
          f"{100*c1.most_common(1)[0][1]/t1:.1f}%")
    print(f"    attacker most common {L[ca.most_common(1)[0][0]]} at "
          f"{100*ca.most_common(1)[0][1]/t1:.1f}%")
    sqc = Counter(cap_sq[have, 0].tolist())
    print(f"    square   most common {sqc.most_common(1)[0][0]} at "
          f"{100*sqc.most_common(1)[0][1]/t1:.1f}%  (chance 1.6%)")

    if args.save:
        p = paths.derived(f"capture_labels_{N}_{n}.npz")
        np.savez_compressed(p, cap_sq=cap_sq, cap_vic=cap_vic, cap_att=cap_att,
                            cap_ply=cap_ply)
        print(f"\n[cap] wrote {p}")


if __name__ == "__main__":
    main()
