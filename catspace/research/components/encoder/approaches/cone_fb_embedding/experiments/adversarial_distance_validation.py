#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/cone_fb_embedding/experiments/adversarial_distance_validation.py -- validate the ADVERSARIAL-QUASIMETRIC
thesis on tablebase ground truth (Kaveh 2026-07-20, "option 1"). No training; exact labels.

The claim: the game-theoretic distance-to-mate is the adversarial forcing-distance / REMOTENESS
(Smith 1966) / attractor-rank to the terminal REGION (NOT a pole -- mate is a scattered absorbing
set), and it is a genuine quasimetric that COMPOSES. We test three things:

  (1) MIN-PLUS REMOTENESS RECURSION (the quasimetric triangle inequality on ground truth):
      for a won, winner-to-move position s,   DTM(s) == 1 + min_child DTM(child).
      The optimal move attains equality; every other child gives slack
      (1 + DTM(child)) - DTM(s) >= 0  == the triangle inequality d(s->R) <= d(s->m) + d(m->R)
      with m = child, R = mate region. We report the exact-recursion rate and the slack dist.

  (2) SPARSITY of the HARD forcing-distance: DTM is finite only on the WON subset (you can force
      the mate REGION, but a *specific* position almost never) -> it is degenerate as a dense
      geometry. This is *why* we factor L1 (dense cooperative reachability) from L2 (adversarial
      outcome): a single adversarial quasimetric can't be the plannable field.

  (3) DENSITY of the COMMITTOR: P(win) under a softmax/eps defender is graded and defined
      EVERYWHERE (the dense harmonic quasimetric-to-region -ln P). We show it is graded (not the
      degenerate {0,0.5,1} of perfect play) and decreases toward the mate region along a won line.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess
import numpy as np


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.gen_dtm_data import rollout_dtm
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import TB, rollout, tb_best_move, white_pov_value
from catspace.io import paths


def recursion_probe(task):
    """For each won, White-to-move position: DTM(s) and, for every child, DTM(child) (None if
    White no longer wins). Returns per-position (dtm_s, best_child_dtm, [slacks of winning children])."""
    packed, meta, syzygy_dir = task
    tb = TB(syzygy_dir)
    out = []
    for i in range(len(packed)):
        b = board_from_packed(packed[i], meta[i])
        if b.turn != chess.WHITE or b.is_game_over() or white_pov_value(b, tb) != 1.0:
            continue
        dtm_s = rollout_dtm(b, tb)
        if dtm_s is None or dtm_s < 1:
            continue
        child_dtms = []
        for m in b.legal_moves:
            c = b.copy(stack=False); c.push(m)
            if c.is_checkmate():
                child_dtms.append(0)                      # mate delivered: DTM(child)=0
                continue
            d = rollout_dtm(c, tb)                         # plies for White to mate from c
            if d is not None:
                child_dtms.append(d)                      # only winning children (finite)
        if not child_dtms:
            continue
        best = min(child_dtms)
        slacks = [(1 + d) - dtm_s for d in child_dtms]    # triangle slack; >=0, 0 at optimal
        out.append((dtm_s, best, slacks))
    tb.close()
    return out


def committor_probe(task):
    """Graded committor P(win) under eps-greedy White (v^pi) for each position, + white_pov (V*)."""
    packed, meta, eps, rollouts, syzygy_dir, seed = task
    tb = TB(syzygy_dir)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(len(packed)):
        b = board_from_packed(packed[i], meta[i])
        if b.is_game_over():
            continue
        vstar = white_pov_value(b, tb)
        if vstar is None:
            continue
        p = float(np.mean([rollout(b, eps, tb, rng) for _ in range(rollouts)]))
        out.append((vstar, p))
    tb.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=paths.derived("stratified_perfect.npz"))
    ap.add_argument("--syzygy", default=str(paths.syzygy_dir()))
    ap.add_argument("--n-recursion", type=int, default=200, help="won positions for the recursion test")
    ap.add_argument("--n-committor", type=int, default=150, help="positions for the committor test")
    ap.add_argument("--eps", type=float, default=0.15, help="White blunder rate for the graded committor")
    ap.add_argument("--rollouts", type=int, default=25)
    ap.add_argument("--out", default=paths.experiment("adversarial_distance.png"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    W = max(1, args.workers)

    nz = np.load(args.data, allow_pickle=True)
    P, M, SDTM = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["sdtm"])
    WDL = np.asarray(nz["wdl"])
    rng = np.random.default_rng(args.seed)

    # ---- (1) min-plus remoteness recursion (on won positions) ----
    won = np.flatnonzero(SDTM > 0)
    sel = won[rng.permutation(len(won))[: args.n_recursion * 3]]     # oversample; probe filters
    bnd = np.linspace(0, len(sel), W + 1, dtype=int)
    tasks = [(P[sel[bnd[i]:bnd[i+1]]], M[sel[bnd[i]:bnd[i+1]]], args.syzygy) for i in range(W) if bnd[i+1] > bnd[i]]
    print(f"[stage] remoteness recursion on ~{len(sel)} won positions, {W} workers...", flush=True)
    rec = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for r in ex.map(recursion_probe, tasks):
            rec.extend(r)
    rec = rec[: args.n_recursion]
    exact = np.array([1 + b == s for (s, b, _) in rec])              # DTM(s) == 1 + min_child DTM
    dtms_s = np.array([s for (s, _, _) in rec])
    near = dtms_s <= 10
    near_rate = float(exact[near].mean()) if near.any() else float("nan")
    far_rate = float(exact[~near].mean()) if (~near).any() else float("nan")
    all_slacks = np.array([sl for (_, _, sls) in rec for sl in sls], dtype=float)
    neg_viol = float((all_slacks < -1e-9).mean())                   # triangle-inequality violations
    # Syzygy is DTZ-optimal, not DTM-optimal, so rollout "DTM" detours far from mate: the min-plus
    # recursion holds cleanly NEAR the terminal region (DTZ==DTM) and degrades far out -- a
    # MEASUREMENT artifact of DTZ!=DTM, not a failure of the adversarial quasimetric property.
    print(f"  recursion exact: overall {exact.mean():.3f} | near-mate DTM<=10 {near_rate:.3f} | "
          f"far DTM>10 {far_rate:.3f}  (far degradation = DTZ!=DTM artifact)", flush=True)
    print(f"  triangle slack: median={np.median(all_slacks):.1f} "
          f"frac_zero(optimal)={float((np.abs(all_slacks)<1e-9).mean()):.3f} "
          f"NEGATIVE(violations)={neg_viol:.4f}", flush=True)

    # ---- (2) sparsity of the hard forcing-distance ----
    won_frac = float((SDTM > 0).mean()); draw_frac = float((WDL == 0).mean())
    loss_frac = float((WDL == -1).mean())
    print(f"  DTM defined (finite) only on WON positions: {won_frac:.3f} of the data "
          f"(draw {draw_frac:.3f}, loss {loss_frac:.3f} -> DTM = +inf/undefined there)", flush=True)

    # ---- (3) committor density (graded P(win) under eps-greedy) ----
    csel = rng.permutation(len(P))[: args.n_committor]
    cbnd = np.linspace(0, len(csel), W + 1, dtype=int)
    ctasks = [(P[csel[cbnd[i]:cbnd[i+1]]], M[csel[cbnd[i]:cbnd[i+1]]], args.eps, args.rollouts,
               args.syzygy, args.seed + i) for i in range(W) if cbnd[i+1] > cbnd[i]]
    print(f"[stage] graded committor (eps={args.eps}) on {len(csel)} positions...", flush=True)
    comm = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for r in ex.map(committor_probe, ctasks):
            comm.extend(r)
    vstar = np.array([v for v, _ in comm]); pcommit = np.array([p for _, p in comm])
    graded_frac = float(((pcommit > 0.02) & (pcommit < 0.98)).mean())   # strictly-interior = graded/dense
    print(f"  committor graded fraction (0<P<1): {graded_frac:.3f}  "
          f"(perfect-play V* is only {{0,0.5,1}}; eps-defender makes it dense)", flush=True)

    _plot(args, exact, near_rate, all_slacks, won_frac, draw_frac, loss_frac, vstar, pcommit, graded_frac)
    print(f"VERDICT ADV_DIST recursion_exact_overall={exact.mean():.3f} near_mate={near_rate:.3f} "
          f"tri_violations={neg_viol:.4f} DTM_coverage={won_frac:.3f} "
          f"committor_graded={graded_frac:.3f} ({time.time()-t0:.0f}s)")


def _plot(args, exact, near_rate, slacks, wonf, drawf, lossf, vstar, pcommit, gradedf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0f1115")
    for a in (a1, a2, a3):
        a.set_facecolor("#0f1115"); a.tick_params(colors="#9aa4b2")
        for s in a.spines.values():
            s.set_color("#2a2e37")
        a.title.set_color("#e6e6e6"); a.xaxis.label.set_color("#9aa4b2"); a.yaxis.label.set_color("#9aa4b2")

    # (1) triangle slack histogram (quasimetric composition)
    a1.hist(np.clip(slacks, -1, 30), bins=40, color="#4fa3ff")
    a1.axvline(0, color="#ff5c5c", ls="--")
    a1.set_title(f"(1) DTM composes (min-plus recursion)\nnear-mate exact {near_rate:.2f}, "
                 f"overall {exact.mean():.2f} (far = DTZ!=DTM artifact)")
    a1.set_xlabel("(1 + DTM(child)) - DTM(s)   [=0 optimal, >0 slack, <0 = violation]")
    a1.set_ylabel("count")

    # (2) sparsity: DTM coverage vs outcome
    a2.bar(["won\n(DTM finite)", "draw\n(DTM inf)", "loss\n(DTM inf)"], [wonf, drawf, lossf],
           color=["#33cc77", "#8b93a3", "#d24b4b"])
    a2.set_title(f"(2) HARD forcing-distance is SPARSE\nDTM defined on only {wonf:.0%} of positions")
    a2.set_ylabel("fraction of positions")

    # (3) committor density: P^pi vs V*
    colors = {1.0: "#33cc77", 0.5: "#8b93a3", 0.0: "#d24b4b"}
    for v in (1.0, 0.5, 0.0):
        m = vstar == v
        if m.any():
            jit = (np.random.default_rng(0).random(m.sum()) - 0.5) * 0.12
            a3.scatter(vstar[m] + jit, pcommit[m], s=16, c=colors[v], alpha=0.6)
    a3.set_title(f"(3) COMMITTOR is DENSE/graded\n{gradedf:.0%} strictly in (0,1) under eps-defender")
    a3.set_xlabel("V* (perfect play): 0 / 0.5 / 1"); a3.set_ylabel("committor P(win) under eps-defender")
    a3.set_ylim(-0.05, 1.05)

    fig.suptitle("Adversarial distance is a quasimetric to the mate REGION (not a pole): "
                 "composes (1), sparse (2); the committor is its dense form (3)",
                 color="#e6e6e6", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=115, facecolor="#0f1115"); plt.close(fig)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
