#!/usr/bin/env python
"""eval_survival.py -- how long does a concept LAST? (Kaveh 2026-08-17: "I wanna see a histogram
of ply survival... how many survive two, three, four?")

A single "30.6% survive one ply" hides the shape. A concept vocabulary that is useful for
planning should have concepts that persist over the span of a plan -- 'my king is castled',
'the queenside is locked' are properties of many consecutive positions, not of one ply.

TWO DIFFERENT QUESTIONS, and only the first is a survival function:
    SURVIVAL   S(k) = P(code unchanged for EVERY ply from t to t+k)
               a concept that flickers off and back on has NOT survived
    RETURN     R(k) = P(code at t+k equals code at t), regardless of what happened between
               R(k) >= S(k) always; a large gap means concepts flicker rather than persist

Reported per granularity (global heads, square codes) and per head, because a mean over heads
would hide the case that matters: a few genuinely stable concepts alongside many volatile ones
is a very different vocabulary from uniformly middling ones, and only the first is useful.

    .venv/bin/python -m ...eval_survival --ckpt artifacts/experiments/cdf5_latest.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/cdf5_latest.pt")
    ap.add_argument("--k", type=int, default=12, help="plies to follow a concept for")
    ap.add_argument("--n", type=int, default=4000, help="starting positions")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_dynamics_field import ConceptDynamics
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"]
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    n = len(tr.tok)
    gor = tr.game_of_row()
    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nv = max(1, int(A["val_frac"] * len(tr.start))); nt = max(1, int(A["test_frac"] * len(tr.start)))
    sog = np.zeros(len(tr.start), np.int8); sog[perm[:nv]] = 1; sog[perm[nv:nv + nt]] = 2

    K = args.k
    ok = np.flatnonzero(sog[gor] == args.split)
    ok = ok[ok + K < n]
    ok = ok[gor[ok + K] == gor[ok]]                 # the whole span inside one game
    rng = np.random.default_rng(0)
    rows = ok[rng.choice(len(ok), min(args.n, len(ok)), replace=False)]
    print(f"[surv] {args.ckpt.split('/')[-1]} step {ck.get('step')} | {len(rows):,} spans of "
          f"{K} plies (split {args.split})", flush=True)

    net = ConceptDynamics(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
                          square_codes=A["square_codes"], hidden=A["hidden"], ema=A["ema"])
    net.load_state_dict(ck["state_dict"]); net.eval()

    # codes for every row of every span: (N, K+1, heads) and (N, K+1, 64)
    G = np.zeros((len(rows), K + 1, A["heads"]), np.int16)
    S = np.zeros((len(rows), K + 1, 64), np.int16)
    with torch.no_grad():
        for k in range(K + 1):
            rk = rows + k
            for a in range(0, len(rk), args.batch):
                r = rk[a:a + args.batch]
                _, ids, ids_sq, _ = net.encode(
                    torch.from_numpy(tr.tok[r].astype(np.int64)),
                    torch.from_numpy(tr.glob[r].astype(np.float32)))
                G[a:a + len(r), k] = ids.numpy()
                S[a:a + len(r), k] = ids_sq.numpy()

    def curves(C):
        """-> (survival, return_) each length K+1, index k = after k plies."""
        base = C[:, 0:1]                                  # code at t
        eq = (C == base)                                  # equal at each ply
        surv = np.ones((C.shape[0], K + 1, C.shape[2]), bool)
        run = np.ones_like(eq[:, 0], bool)
        for k in range(K + 1):
            run = run & eq[:, k]
            surv[:, k] = run
        return surv.reshape(-1, K + 1, C.shape[2]).mean((0, 2)), eq.mean((0, 2))

    gs, gr = curves(G)
    ss, sr = curves(S)
    print(f"\n  {'ply':>4} {'GLOBAL surv':>12} {'GLOBAL ret':>11} {'SQUARE surv':>12} "
          f"{'SQUARE ret':>11}")
    for k in range(1, K + 1):
        print(f"  {k:>4} {100*gs[k]:>11.1f}% {100*gr[k]:>10.1f}% {100*ss[k]:>11.1f}% "
              f"{100*sr[k]:>10.1f}%")
    print(f"\n  SURVIVAL = unchanged for EVERY ply up to k.  RETURN = equal at k, "
          f"regardless of the path.")
    gap = 100 * (gr[K] - gs[K])
    print(f"  gap at ply {K}: {gap:.1f} points -- concepts that flicker off and come back")

    # half-life: the ply at which survival crosses 50%
    def half(c):
        below = np.flatnonzero(c[1:] < 0.5)
        return (below[0] + 1) if len(below) else f">{K}"
    print(f"\n  half-life (survival < 50%):  global {half(gs)} plies | square {half(ss)} plies")

    # per-head spread -- a mean hides 'a few stable concepts among many volatile ones'
    ph = [( (G[:, :4+1] == G[:, 0:1]).all(1)[:, h].mean(), h) for h in range(A["heads"])]
    ph.sort(reverse=True)
    print(f"\n  per-head 4-ply survival, best to worst:")
    print("    " + "  ".join(f"h{h}:{100*v:.0f}%" for v, h in ph))
    print(f"    spread {100*ph[0][0]:.0f}% (best) to {100*ph[-1][0]:.0f}% (worst) -- a wide "
          f"spread means SOME concepts are genuinely durable")


if __name__ == "__main__":
    main()
