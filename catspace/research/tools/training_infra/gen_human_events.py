#!/usr/bin/env python
"""gen_human_events.py -- capture events for a HUMAN corpus, straight from token diffs.

The SF pipeline built event labels through game_transitions_FULL, an artifact the repo
itself documents as ad-hoc. For captures none of that is needed: during the mover's ply
the opponent's pieces do not move, so ANY opponent-occupied square that is empty (or
mover-occupied) on the next row is a capture ON THAT SQUARE -- normal and en passant
uniformly, no board replay. Emits:

    event_labels_5_{n}.npz     the four keys gen_threat_execution consumes
                               (ev_sq/ev_ply/ev_type/ev_mover; type=1, mover=color marker)
    capture_within{k}_{n}.npz  (n,64) int8 -- occupant of square s at row r is captured
                               within k plies (the UNCONDITIONAL per-square capture head's
                               labels, the v2b L1 target; known label noise: a later
                               arrival on s captured within the window also counts)

    .venv/bin/python -m ...gen_human_events --n-human 30000 --k 8 --save
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-human", type=int, default=30000)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--n-piecedown", type=int, default=0)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-events", type=int, default=5)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    t0 = time.time()
    tr = T.build(n_human=args.n_human, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    n = len(tr.tok)
    E = args.n_events
    ev_sq = np.full((n, E), -1, np.int8)
    ev_ply = np.zeros((n, E), np.int16)
    ev_type = np.zeros((n, E), np.int8)
    ev_mover = np.zeros((n, E), np.int8)
    cap_k = np.zeros((n, 64), np.int8)

    for gi in range(len(tr.start)):
        s0, ln = int(tr.start[gi]), int(tr.length[gi])
        if ln < 2:
            continue
        tok = tr.tok[s0:s0 + ln]                     # (ln, 64) uint8, ids 1-6 W / 7-12 B
        stm = tr.glob[s0:s0 + ln, 0].astype(bool)    # True = White to move
        # vanished-opponent squares per transition t -> t+1
        opp = np.where(stm[:-1, None], tok[:-1] >= 7,
                       (tok[:-1] >= 1) & (tok[:-1] <= 6))
        gone = opp & (tok[1:] != tok[:-1])
        # a capture removes exactly one opponent piece; take that square
        tt, ss = np.nonzero(gone)
        # keep first vanished square per transition (multiple only if data glitch)
        keep = np.ones(len(tt), bool)
        keep[1:] = tt[1:] != tt[:-1]
        tt, ss = tt[keep], ss[keep]
        if not len(tt):
            continue
        movers = np.where(stm[tt], 1, 7).astype(np.int8)     # color marker id
        # next-E events per row: events at transition t land at ply offset (t - r) + 1
        ev_t = tt                                             # sorted ascending
        for r in range(ln):
            j0 = np.searchsorted(ev_t, r, side="left")
            take = min(E, len(ev_t) - j0)
            if take > 0:
                rr = s0 + r
                ev_sq[rr, :take] = ss[j0:j0 + take]
                ev_ply[rr, :take] = (ev_t[j0:j0 + take] - r) + 1
                ev_type[rr, :take] = 1
                ev_mover[rr, :take] = movers[j0:j0 + take]
        # unconditional: occupant of s at row r captured within k plies
        for j in range(len(ev_t)):
            t_ev, s_ev = int(ev_t[j]), int(ss[j])
            r_lo = max(0, t_ev - args.k + 1)
            cap_k[s0 + r_lo:s0 + t_ev + 1, s_ev] = 1
        if gi % 5000 == 0:
            print(f"  game {gi:,}/{len(tr.start):,} [{time.time()-t0:.0f}s]", flush=True)

    n_ev = int((ev_sq[:, 0] >= 0).sum())
    print(f"\n  rows with >=1 upcoming capture: {n_ev:,}/{n:,} ({100*n_ev/n:.0f} percent) | "
          f"cap-within-{args.k} positives: {100*cap_k.mean():.2f} percent of cells "
          f"[{time.time()-t0:.0f}s]")
    if args.save:
        p1 = paths.derived(f"event_labels_5_{n}.npz")
        np.savez_compressed(p1, ev_sq=ev_sq, ev_ply=ev_ply, ev_type=ev_type,
                            ev_mover=ev_mover)
        p2 = paths.derived(f"capture_within{args.k}_{n}.npz")
        np.savez_compressed(p2, cap=cap_k, k=np.int64(args.k))
        print(f"  wrote {p1}\n  wrote {p2}")


if __name__ == "__main__":
    main()
