#!/usr/bin/env python
"""mine_concept_goals.py -- which concept patterns are worth being GOALS?

"So from this state that I'm in, how do I reach this concept state where my opponent is not
castled... and then I attack on the king's side." (Kaveh 2026-08-19)

A goal predicate has to earn three properties, all MEASURED here per candidate, none assumed:

    PERSISTENT   once true it stays true for a while (mean run length in plies). A predicate
                 that flips every ply is a tactic, not a destination -- measured on the current
                 vocabulary the global heads change code on ~98% of plies, so most global
                 patterns will fail this and that failure is worth having on record.
    REACHABLE    from states where it is false, it becomes true within the horizon often enough
                 to learn from (P(hit <= H)), but is not simply always true (base rate cap).
    RECURRENT    it appears in many games, so the reachability field generalises across games
                 rather than memorising one.

Candidates are the atomic patterns of the code space -- (global head h = code c) and
(square q = code c) -- because today's measurement showed RANDOM conjunctions are mostly dead
(22/48 never occur at all): compositions must be built from atoms that individually pass, not
enumerated blindly.

Engine-free by construction: every statistic is computed from the cached codes and game
boundaries alone.

    .venv/bin/python -m ...mine_concept_goals --codes data/derived/concept_codes_cef3_6545146.npz --save
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", required=True)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--eval-rows", type=int, default=150000)
    ap.add_argument("--min-run", type=float, default=4.0, help="mean plies a goal must persist")
    ap.add_argument("--min-hit", type=float, default=0.05)
    ap.add_argument("--max-base", type=float, default=0.5)
    ap.add_argument("--min-games", type=int, default=500)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    t0 = time.time()
    z = np.load(args.codes, allow_pickle=True)
    g, s = z["g"].astype(np.int16), z["s"].astype(np.int16)
    n, H = g.shape
    C = int(g.max()) + 1; SC = int(s.max()) + 1
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    assert len(tr.tok) == n
    gor = tr.game_of_row().astype(np.int32)
    g_end = np.zeros(n, np.int64)
    for gi in range(len(tr.start)):
        s0, ln = int(tr.start[gi]), int(tr.length[gi])
        g_end[s0:s0 + ln] = s0 + ln - 1
    new_game = np.zeros(n, bool); new_game[tr.start] = True
    rng = np.random.default_rng(args.seed)
    eval_rows = np.sort(rng.choice(n, min(args.eval_rows, n), replace=False))

    def stats(active):
        """(persist, base, p_hit, med_plies, n_games) for one boolean predicate."""
        # run structure: a run starts where active & (previous inactive or game boundary)
        starts = active & (~np.concatenate([[False], active[:-1]]) | new_game)
        n_runs = int(starts.sum())
        if n_runs == 0:
            return 0.0, 0.0, 0.0, np.nan, 0
        persist = float(active.sum()) / n_runs
        idx = np.flatnonzero(active)
        r = eval_rows[~active[eval_rows]]
        base = float(active[eval_rows].mean())
        if len(r) == 0:
            return persist, base, 0.0, np.nan, 0
        j = np.searchsorted(idx, r)
        nxt = np.where(j < len(idx), idx[np.minimum(j, len(idx) - 1)], n + 10**9)
        ok = (nxt - r <= args.horizon) & (nxt <= g_end[r])
        med = float(np.median((nxt - r)[ok])) if ok.any() else np.nan
        n_g = len(np.unique(gor[idx[:200000]]))
        return persist, base, float(ok.mean()), med, n_g

    rows = []
    for h in range(H):
        col = g[:, h]
        for c in range(C):
            rows.append(("g", h, c) + stats(col == c))
    print(f"[mine] global atoms done [{time.time()-t0:.0f}s]", flush=True)
    for q in range(64):
        col = s[:, q]
        for c in np.unique(col):
            rows.append(("s", q, int(c)) + stats(col == c))
    print(f"[mine] square atoms done [{time.time()-t0:.0f}s]", flush=True)

    dt = np.dtype([("kind", "U1"), ("slot", np.int16), ("code", np.int16),
                   ("persist", np.float32), ("base", np.float32), ("p_hit", np.float32),
                   ("med", np.float32), ("n_games", np.int32)])
    A = np.array(rows, dtype=dt)
    ok = ((A["persist"] >= args.min_run) & (A["p_hit"] >= args.min_hit)
          & (A["base"] <= args.max_base) & (A["n_games"] >= args.min_games))

    for kind, name in (("g", "GLOBAL-head atoms"), ("s", "SQUARE atoms")):
        m = A["kind"] == kind
        print(f"\n  {name}: {int(m.sum()):,} candidates")
        print(f"    persistence (mean run, plies): median {np.median(A['persist'][m]):.1f}  "
              f"p90 {np.percentile(A['persist'][m], 90):.1f}")
        print(f"    P(hit <= {args.horizon}): median {np.median(A['p_hit'][m]):.3f}")
        print(f"    PASS all gates (persist>={args.min_run}, hit>={args.min_hit}, "
              f"base<={args.max_base}, games>={args.min_games}): "
              f"{int((ok & m).sum()):,} ({100*(ok & m).mean():.1f}%)")

    top = A[ok][np.argsort(-A[ok]["persist"])][:12]
    print(f"\n  most persistent passing goals:")
    print(f"    {'kind':>4} {'slot':>4} {'code':>4} {'persist':>8} {'base':>6} "
          f"{'P(hit<=20)':>10} {'med plies':>9} {'games':>7}")
    for r in top:
        print(f"    {r['kind']:>4} {r['slot']:>4} {r['code']:>4} {r['persist']:>8.1f} "
              f"{r['base']:>6.3f} {r['p_hit']:>10.3f} {r['med']:>9.0f} {r['n_games']:>7,}")

    if args.save:
        tag = args.codes.split("concept_codes_")[-1].replace(".npz", "")
        p = paths.derived(f"goal_atoms_{tag}.npz")
        np.savez_compressed(p, atoms=A, passing=ok,
                            horizon=np.array(args.horizon))
        print(f"\n  wrote {p}  ({int(ok.sum()):,} passing goals)")


if __name__ == "__main__":
    main()
