#!/usr/bin/env python
"""profile_step.py -- where does a training step actually go? (Kaveh 2026-08-16: "keeping track
of time it takes to do stuff and metrics, so we can address bottlenecks later")

End-to-end timings tell you a step costs 0.80 s. They do not tell you which 0.80 s to attack.
This breaks one step into its phases and reports both wall time and share, so the next
optimisation is chosen from evidence rather than intuition -- the last three were not what I
expected:
  * naive batching made the step SLOWER (2.37 vs 1.77), because the version it replaced had 10
    of 12 encodes under no_grad and the fix to persistence is what added backward cost
  * the win came from shrinking the persistence SUBSET, not from batching per se
  * torch was using 5 of 11 cores; setting 8 gave another 1.3x for free

Phases measured, in the order the step runs them:
    encode_start    live encode of the window start        (full batch, with grad)
    encode_pers     live encode of t..t+K for persistence  (subset, with grad)
    encode_target   EMA encode of t+1..t+K                 (no_grad)
    unroll          K x (predict_next + soft_state + value + CE)
    persistence     the t->t+2 term
    anchor+outcome  the two value terms
    backward        loss.backward()
    optimiser       opt.step() + EMA update

    .venv/bin/python -m ...profile_step --reps 8
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--unroll", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_dynamics_field import ConceptDynamics
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    t0 = time.time()
    tr = T.build(n_human=0, n_sf=args.games, seed=0, cache=True, max_plies=400,
                 n_piecedown=args.n_piecedown)
    n = len(tr.tok)
    syz = np.load(paths.derived(f"syzygy_wdl_{n}.npz"))["syz_wdl"].astype(np.int64)
    out_w = tr.outcome_of_row_white()
    res = np.where(out_w == T.WIN, 2, np.where(out_w == T.DRAW, 1, 0)).astype(np.int64)
    gz = np.load(paths.derived(f"game_transitions_FULL_{n}.npz"))
    par, mid = gz["par"], gz["mid"]
    print(f"[prof] corpus loaded [{time.time()-t0:.0f}s]  -- one-off, not part of a step",
          flush=True)

    gor = tr.game_of_row()
    mv_of_row = np.full((n, 3), -1, np.int64); mv_of_row[par] = mid
    K = args.unroll
    win = np.flatnonzero(mv_of_row[:, 0] >= 0)
    win = win[win + K + 1 < n]
    win = win[gor[win + K] == gor[win]]
    win = win[(mv_of_row[win[:, None] + np.arange(K)[None, :], 0] >= 0).all(1)]
    men = (tr.tok != 0).sum(1)
    quiet = np.zeros(n, bool); quiet[par] = men[par] == men[par + 1]
    anc = np.flatnonzero(syz >= 0)

    TOK = torch.from_numpy(tr.tok); GLB = torch.from_numpy(tr.glob)
    SYZ = torch.from_numpy(syz); RES = torch.from_numpy(res); MVR = torch.from_numpy(mv_of_row)
    def batch(rows):
        r = torch.as_tensor(rows); return TOK[r].long(), GLB[r].float()

    net = ConceptDynamics()
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-4)
    rng = np.random.default_rng(0)
    acc = defaultdict(float)

    class T_:
        def __init__(s, k): s.k = k
        def __enter__(s): s.t = time.perf_counter(); return s
        def __exit__(s, *a): acc[s.k] += time.perf_counter() - s.t

    for rep in range(args.reps + 1):          # first rep warms caches; discarded
        if rep == 1:
            acc.clear()
        net.train(); net.cache_books()
        wi = win[rng.integers(0, len(win), args.batch)]
        B = len(wi)
        with T_("encode_start"):
            zq0, ids0, sq0, vl = net.encode(*batch(wi))
        Bp = max(8, args.batch // 4); wp = wi[:Bp]
        with T_("encode_pers"):
            rows_p = (wp[:, None] + np.arange(K + 1)[None, :]).reshape(-1)
            zq_p, _, _, vl_p = net.encode(*batch(rows_p))
            zq_p = zq_p.view(Bp, K + 1, -1); vl = vl + vl_p
        with T_("encode_target"):
            with torch.no_grad():
                rows_t = (wi[:, None] + np.arange(1, K + 1)[None, :]).reshape(-1)
                _, t_ids, t_sq, _ = net.encode(*batch(rows_t), target=True)
                t_ids = t_ids.view(B, K, -1); t_sq = t_sq.view(B, K, 64)
        with T_("unroll"):
            zq_roll = zq0; l_dyn = 0.0; l_vnext = 0.0
            for k in range(K):
                rk = wi + k
                mv = MVR[torch.as_tensor(rk)]
                g_lg, s_lg = net.predict_next(zq_roll, mv)
                sd = F.cross_entropy(g_lg.reshape(-1, net.codes), t_ids[:, k].reshape(-1))
                sd = sd + F.cross_entropy(s_lg.reshape(-1, net.square_codes),
                                          t_sq[:, k].reshape(-1))
                _r1 = torch.as_tensor(rk + 1)
                y_k = torch.where(SYZ[_r1] >= 0, SYZ[_r1], RES[_r1])
                zq_roll = net.soft_state(g_lg, s_lg)
                l_dyn = l_dyn + sd / K
                l_vnext = l_vnext + F.cross_entropy(net.value(zq_roll), y_k) / K
        with T_("persistence"):
            l_pers = torch.zeros(()); npair = 0
            for k in range(K - 1):
                qm = torch.from_numpy((quiet[wp + k] & quiet[wp + k + 1]).astype(np.float32))
                if qm.sum() == 0: continue
                d = (zq_p[:, k] - zq_p[:, k + 2]).pow(2).mean(-1)
                l_pers = l_pers + (qm * d).sum() / qm.sum().clamp(min=1); npair += 1
            l_pers = l_pers / max(npair, 1)
        with T_("anchor+outcome"):
            a_rows = anc[rng.integers(0, len(anc), args.batch // 2)]
            zq_a, _, _, vl_a = net.encode(*batch(a_rows))
            l_anchor = F.cross_entropy(net.value(zq_a), SYZ[torch.as_tensor(a_rows)])
            l_out = F.cross_entropy(net.value(zq0), RES[torch.as_tensor(wi)])
        loss = (2.0*l_dyn + 1.0*l_vnext + 1.0*l_pers + 5.0*l_anchor + 1.0*l_out + vl + vl_a)
        with T_("backward"):
            opt.zero_grad(); loss.backward()
        with T_("optimiser"):
            opt.step(); net.update_ema()

    tot = sum(acc.values()) / args.reps
    print(f"\n  STEP BREAKDOWN  (mean of {args.reps} steps, {args.threads} threads, "
          f"batch {args.batch}, K={K})\n")
    print(f"  {'phase':<18} {'ms':>9} {'share':>8}")
    for k, v in sorted(acc.items(), key=lambda x: -x[1]):
        print(f"  {k:<18} {1000*v/args.reps:>9.1f} {100*v/sum(acc.values()):>7.1f}%")
    print(f"  {'-'*36}\n  {'TOTAL':<18} {1000*tot:>9.1f} {100.0:>7.1f}%")
    print(f"\n  -> 20,000 steps = {20000*tot/3600:.1f} h")
    enc = acc['encode_start'] + acc['encode_pers'] + acc['encode_target'] + acc['anchor+outcome']
    print(f"  -> ENCODES are {100*enc/sum(acc.values()):.0f}% of the step; the trunk is "
          f"45.7% of params and is run on\n     {args.batch}(start) + {Bp*(K+1)}(pers) + "
          f"{args.batch*K}(target) + {args.batch//2}(anchor) = "
          f"{args.batch + Bp*(K+1) + args.batch*K + args.batch//2} rows per step.")


if __name__ == "__main__":
    main()
