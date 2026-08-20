#!/usr/bin/env python
"""gen_threat_execution.py -- threats vs executions: when does SF cash in an available capture?

"Anytime a piece is threatened to be taken -- that's a threat. But how often is it actually
taken? That is the execution. We don't always execute... that will give us insight into when is
a good time to execute the capture, even though the capture is available." (Kaveh 2026-08-19)

The gap between threat and execution IS tension, and its resolution timing is the tactical
knowledge the alignment pass found missing (hanging at 2x lift). The labels are FREE, and the
construction needs no board replay at all:

    THREAT     from the Rust basis (4.5 us/pos, zero-mismatch-verified): an enemy piece stands
               on s and the mover attacks s. Soundness graded by the basis' SEE.
    EXECUTION  from the existing event labels: the next-5 capture events per row carry their
               squares and ply offsets, so "was a capture executed ON s within k plies, and by
               whom" is a numpy join against artifacts already on disk.

Emitted per (row, threatened square): victim value, SEE, executed<=k, plies-to-execution,
executed-by-threatener. Plus the census that answers his question directly.

    .venv/bin/python -m ...gen_threat_execution --k 8 --save
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-human", type=int, default=0)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=8, help="execution window, plies")
    ap.add_argument("--n", type=int, default=1500000, help="rows to scan")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    import catspace_concepts as cc
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    t0 = time.time()
    tr = T.build(n_human=args.n_human, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    n = len(tr.tok)
    ev = np.load(paths.derived(f"event_labels_5_{n}.npz"))
    ev_sq, ev_ply = ev["ev_sq"], ev["ev_ply"]                 # (n,5) next events
    ev_type, ev_mover = ev["ev_type"], ev["ev_mover"]
    SQ_NAMES, _ = cc.feature_names()
    i_att_w, i_att_b = SQ_NAMES.index("att_w"), SQ_NAMES.index("att_b")
    i_see = SQ_NAMES.index("see")

    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(n, min(args.n, n), replace=False))
    recs = []
    CH = 100000
    for a in range(0, len(rows), CH):
        r = rows[a:a + CH]
        sq_feat, _, failed = cc.extract(np.ascontiguousarray(tr.tok[r]),
                                        np.ascontiguousarray(tr.glob[r]))
        assert failed == 0
        stm_white = tr.glob[r, 0].astype(bool)
        occ = tr.tok[r]                                        # (B,64) piece ids
        enemy = np.where(stm_white[:, None], occ >= 7, (occ >= 1) & (occ <= 6))
        att_by_mover = np.where(stm_white[:, None],
                                sq_feat[:, :, i_att_w] > 0, sq_feat[:, :, i_att_b] > 0)
        threat = enemy & att_by_mover                          # (B,64)
        bi, sqi = np.nonzero(threat)
        see = sq_feat[bi, sqi, i_see]
        vic = occ[bi, sqi]
        rr = r[bi]
        # execution: a CAPTURE event on this square within k plies (event types 1 and 3)
        e_sq, e_ply = ev_sq[rr], ev_ply[rr]
        e_cap = (ev_type[rr] == 1) | (ev_type[rr] == 3)
        hit = (e_sq == sqi[:, None]) & e_cap & (e_ply >= 1) & (e_ply <= args.k)
        executed = hit.any(1)
        first = np.where(executed, np.argmax(hit, 1), 0)
        t_exec = np.where(executed, e_ply[np.arange(len(rr)), first], -1)
        mover_white_at_exec = ev_mover[rr, first] <= 6
        by_threatener = executed & (mover_white_at_exec == stm_white[bi])
        recs.append(np.stack([rr, sqi, vic, see.astype(np.int64),
                              executed.astype(np.int64), t_exec,
                              by_threatener.astype(np.int64)], 1))
        print(f"  {min(a+CH,len(rows)):,}/{len(rows):,} rows "
              f"[{time.time()-t0:.0f}s]", flush=True)
    A = np.concatenate(recs)
    rr, sqi, vic, see, executed, t_exec, by_us = A.T
    print(f"\n[te] {len(A):,} threats on {len(rows):,} rows "
          f"({len(A)/len(rows):.2f} threats/position)")
    ex = executed.astype(bool)
    print(f"[te] P(executed within {args.k} plies | threat)          {ex.mean():.3f}")
    print(f"[te]   ... by the threatening side                    "
          f"{by_us[ex].mean():.3f} of executions")
    sound = see > 0
    even = see == 0
    print(f"[te] by soundness (Rust-basis SEE):")
    print(f"[te]   SEE>0 (winning capture available)  n={int(sound.sum()):,}  "
          f"P(exec) {ex[sound].mean():.3f}")
    print(f"[te]   SEE=0 (even trade available)       n={int(even.sum()):,}  "
          f"P(exec) {ex[even].mean():.3f}")
    print(f"[te]   SEE<0 (bad capture only)           n={int((~sound & ~even).sum()):,}  "
          f"P(exec) {ex[~sound & ~even].mean():.3f}")
    tt = t_exec[ex]
    print(f"[te] time-to-execution when executed: median {np.median(tt):.0f} plies, "
          f"P(immediate, 1 ply) {np.mean(tt == 1):.3f}")
    VAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0, 7: 1, 8: 3, 9: 3, 10: 5, 11: 9, 12: 0}
    vv = np.array([VAL[int(v)] for v in vic])
    for w, nm in ((1, "pawn"), (3, "minor"), (5, "rook"), (9, "queen")):
        m = vv == w
        if m.sum():
            print(f"[te]   victim {nm:<6} n={int(m.sum()):>9,}  P(exec) {ex[m].mean():.3f}  "
                  f"P(immediate|exec) {np.mean(t_exec[m & ex] == 1):.3f}")
    if args.save:
        p = paths.derived(f"threat_exec_k{args.k}_{len(A)}_{n}.npz")
        np.savez_compressed(p, row=rr, sq=sqi, victim=vic, see=see,
                            executed=executed, t_exec=t_exec, by_threatener=by_us)
        print(f"[te] wrote {p}")


if __name__ == "__main__":
    main()
