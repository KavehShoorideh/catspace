#!/usr/bin/env python
"""eval_rollout.py -- does the concept dynamics survive being COMPOSED? (2026-08-16)

The design's deliverable is a TRAJECTORY of concepts, not a single step. But the dynamics is
trained on one step only (MuZero unrolls K=5 and applies every loss at each step; we do not),
so nothing in the objective prevents error compounding over a rollout. This measures whether it
does, before anything else is built on top of the trajectory story.

TWO MODES, and the GAP between them is the whole point:

  TEACHER-FORCED   predict c_{t+k} from the TRUE c_{t+k-1}
                   -> per-step accuracy, unaffected by compounding
  FREE-RUNNING     predict c_{t+k} from its OWN previous prediction
                   -> what a planner actually gets, compounding included

If free-running tracks teacher-forced, the dynamics composes and a concept trajectory is
meaningful. If it collapses after 2-3 steps, the trajectory is a story the model cannot
support, and a k-step unroll loss is needed before going further.

Baselines are printed alongside, because "predict the codes do not change" is a strong
strategy in chess: most plies leave most concepts alone, so a persistence baseline is the bar,
not chance.

    .venv/bin/python -m ...eval_rollout --ckpt artifacts/experiments/cdf1_step5000.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=8, help="rollout depth in plies")
    ap.add_argument("--n", type=int, default=4000, help="starting positions")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_dynamics_field import ConceptDynamics
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.research.tools.training_infra import gen_game_transitions as GT
    from catspace.io import paths

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"]
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    # BUILD the full transition set rather than GT.ensure(): the cached file is a 1.5M
    # SUBSAMPLE of 4.65M rows (made for jqt), and requiring k CONSECUTIVE moves from a
    # subsample almost never succeeds -- it yielded 51 usable rollouts out of 3,000 requested,
    # which is not a sample anyone should draw a conclusion from.
    # BOTH pools, or 92% of the corpus (the piece-down games) is silently dropped and the
    # rollout is scored on 8% of it. Same bug as in training; missed here until the output
    # showed 'unmatched 45905'.
    try:
        _z = np.load(paths.derived(f"game_transitions_FULL_{len(tr.tok)}.npz"))
        par, mid = _z["par"], _z["mid"]
    except FileNotFoundError:
        par, mid = GT.build(tr, GT.all_rows(A["games"], A["n_piecedown"], A["seed"],
                                            A["max_plies"]))
    gor = tr.game_of_row()

    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nv = max(1, int(A["val_frac"] * len(tr.start))); nt = max(1, int(A["test_frac"] * len(tr.start)))
    sog = np.zeros(len(tr.start), np.int8); sog[perm[:nv]] = 1; sog[perm[nv:nv + nt]] = 2

    # move id per ROW, so a k-ply rollout can look up consecutive actions
    mv_of_row = np.full((len(tr.tok), 3), -1, np.int64)
    mv_of_row[par] = mid
    # starts: the whole k-ply window must stay inside one game AND inside the split
    ok = np.flatnonzero((sog[gor] == args.split) & (mv_of_row[:, 0] >= 0))
    ok = ok[(gor[np.clip(ok + args.k, 0, len(gor) - 1)] == gor[ok])]
    ok = ok[(mv_of_row[ok[:, None] + np.arange(args.k)[None, :], 0] >= 0).all(1)]
    rng = np.random.default_rng(0)
    rows = ok[rng.choice(len(ok), min(args.n, len(ok)), replace=False)]
    print(f"[roll] ckpt step {ck.get('step')} | {len(rows):,} rollouts of {args.k} plies "
          f"(split {args.split})", flush=True)

    net = ConceptDynamics(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
                          square_codes=A["square_codes"], hidden=A["hidden"], ema=A["ema"])
    net.load_state_dict(ck["state_dict"]); net.eval()

    tf_hit = np.zeros(args.k); fr_hit = np.zeros(args.k)
    per_hit = np.zeros(args.k); n_tot = np.zeros(args.k)
    with torch.no_grad():
        for a in range(0, len(rows), args.batch):
            r0 = rows[a:a + args.batch]
            tok0 = torch.from_numpy(tr.tok[r0].astype(np.int64))
            gl0 = torch.from_numpy(tr.glob[r0].astype(np.float32))
            zq_free, ids0, _, _ = net.encode(tok0, gl0)
            ids_prev_true = ids0
            for k in range(args.k):
                rk = r0 + k
                mv = torch.from_numpy(mv_of_row[rk].astype(np.int64))
                # truth at t+k+1, from the EMA branch (the same target training used)
                tokn = torch.from_numpy(tr.tok[rk + 1].astype(np.int64))
                gln = torch.from_numpy(tr.glob[rk + 1].astype(np.float32))
                _, ids_true, _, _ = net.encode(tokn, gln, target=True)

                # TEACHER-FORCED: step from the TRUE state at t+k
                zq_true, ids_at_k, _, _ = net.encode(
                    torch.from_numpy(tr.tok[rk].astype(np.int64)),
                    torch.from_numpy(tr.glob[rk].astype(np.float32)))
                g_tf, s_tf = net.predict_next(zq_true, mv)
                tf_hit[k] += float((g_tf.argmax(-1) == ids_true).float().sum())

                # FREE-RUNNING: step from our OWN previous prediction
                g_fr, s_fr = net.predict_next(zq_free, mv)
                fr_hit[k] += float((g_fr.argmax(-1) == ids_true).float().sum())
                zq_free = net.soft_state(g_fr, s_fr)

                # BASELINE: "nothing changes" -- carry the codes at t+k forward
                per_hit[k] += float((ids_at_k == ids_true).float().sum())
                n_tot[k] += ids_true.numel()

    print(f"\n  {'ply':>4} {'teacher-forced':>15} {'free-running':>14} {'persistence':>12}"
          f"   {'gap (compounding)':>18}")
    for k in range(args.k):
        tf, fr, pe = tf_hit[k]/n_tot[k], fr_hit[k]/n_tot[k], per_hit[k]/n_tot[k]
        print(f"  {k+1:>4} {tf:>15.3f} {fr:>14.3f} {pe:>12.3f}   {tf-fr:>18.3f}")
    print("\n  teacher-forced ~ free-running  -> the dynamics COMPOSES; trajectories are real")
    print("  free-running collapsing        -> one-step training does not survive rollout;")
    print("                                    a k-step unroll loss is needed (MuZero uses K=5)")
    print("  either below persistence       -> the model is worse than assuming nothing changed")


if __name__ == "__main__":
    main()
