#!/usr/bin/env python
"""
experiments/krkn_search_sweep.py — port of code/exp_search.py onto the shared
readout/arena stack: does k-ply MIN-readout minimax search on the learned
KRkn field close the gap to exact-DTM play, and does the goal-region ablation
(oracle {DTM<=3}∪{MATE} vs oracle-free B[MATE] alone) matter?

Every piece of exp_search.py's hand-rolled logic (dtm_full/B_opt, minimax_
backup, pol_from, evaluate) turned out to be exactly what Phase 2's shared
readout/scoring/opponents/arena modules already generalize -- this script is
almost pure composition of those, which is itself a proof the abstraction
was cut in the right place.

Requires experiments/train_krkn.py to have been run first (reads
dtm_krkn/krkn_F/krkn_B from data/derived/).
"""
from __future__ import annotations

import sys
import time

import numpy as np

from latentchess.arena import evaluate as arena_evaluate
from latentchess.domains import krkn
from latentchess.io.paths import generated_dir, load_array
from latentchess.opponents import TableOpponent, optimal_reply_table
from latentchess.planner.policy import TablePolicy
from latentchess.planner.readout import ReplyAgg, backup, greedy_policy
from latentchess.scoring import TerminalScores, fill_terminal_state_scores

BIG = 1e15
TS = TerminalScores(mate=BIG, draw=-BIG, bwin=-BIG)


def run_sweep(chain, dtm, F, zvec, b_opt, starts, label, t0):
    state_scores = fill_terminal_state_scores(F @ zvec, chain, TS)
    print(f"\n[{label}]  depth = 2k+1-ply minimax search on the learned field")
    print("  k | search plies | conversion | EXACT-DTM games | moves/optimal | rook-lost")
    V = state_scores.copy()
    for k in range(7):
        if k > 0:
            V = backup(V, chain, ReplyAgg.MIN, TS, 1)
        pol = greedy_policy(V, chain, ReplyAgg.MIN, TS)
        result = arena_evaluate(chain, dtm, TablePolicy(pol), TableOpponent(b_opt),
                                 starts, cap=70, seed=99)
        print(f"  {k} |     {2 * k + 1:2d}      |   {result.conversion:.3f}    |      "
              f"{result.exact_dtm_rate:.3f}      |     {result.tempo:.3f}     |   "
              f"{result.rook_loss:.3f}   ({time.time() - t0:.0f}s)")
    return V


def main():
    t0 = time.time()
    chain = krkn.build_chain(verbose=False)
    try:
        dtm = load_array("dtm_krkn")
        F = load_array("krkn_F")
        Bm = load_array("krkn_B")
    except FileNotFoundError:
        print("data/derived/{dtm_krkn,krkn_F,krkn_B}.npy not found -- "
              "run experiments/train_krkn.py first.", file=sys.stderr)
        return 1

    won = np.isfinite(dtm[:chain.n_live])
    b_opt = optimal_reply_table(chain, dtm)

    rng = np.random.default_rng(99)
    widx = np.where(won)[0]
    starts = widx[rng.integers(0, len(widx), size=400)]

    region_oracle = np.concatenate([np.where(dtm <= 3)[0], [chain.terminals.mate]])
    z_oracle = Bm[region_oracle].sum(0)
    z_pure = Bm[chain.terminals.mate]

    run_sweep(chain, dtm, F, z_oracle, b_opt, starts, "goal = oracle region {DTM<=3} ∪ {MATE}", t0)
    run_sweep(chain, dtm, F, z_pure, b_opt, starts, "goal = B[MATE] alone (oracle-free)", t0)

    # ---- map: where IS the DTM<=3 region? (viz only, not a scientific claim)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from latentchess.viz.projection import PCAProjection

        n2 = chain.strata["KRkn"].stop
        Fw = F[:chain.n_live]
        subs = np.random.default_rng(0).choice(chain.n_live, 20000, replace=False)
        proj = PCAProjection().fit(Fw[subs])
        P2 = proj.transform(Fw)
        near = dtm[:n2] <= 3

        fig, ax = plt.subplots(figsize=(8, 6.4), facecolor="#10151C")
        ax.set_facecolor("#171E27")
        ax.scatter(P2[:n2][won[:n2] & ~near, 0], P2[:n2][won[:n2] & ~near, 1], s=2, c="#4EC9B0",
                   alpha=.25, linewidths=0, label="won")
        ax.scatter(P2[:n2][~won[:n2], 0], P2[:n2][~won[:n2], 1], s=2, c="#6B7280",
                   alpha=.25, linewidths=0, label="drawn")
        ax.scatter(P2[n2:, 0], P2[n2:, 1], s=2, c="#C97B4E", alpha=.3, linewidths=0, label="KRk stratum")
        ax.scatter(P2[:n2][near, 0], P2[:n2][near, 1], s=9, c="#F0A83C", alpha=.9, linewidths=0.3,
                   edgecolors="#10151C", label=f"the goal region: DTM≤3 ({near.sum():,} states)")
        ax.legend(fontsize=9, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542", loc="lower right")
        ax.set_title("The plan target G, made visible — z_G = Σ_{g∈G} B(g)", color="#E8E4D9", fontsize=11)
        ax.tick_params(colors="#8B94A3", labelsize=7)
        for spn in ax.spines.values():
            spn.set_color("#2A3542")
        plt.tight_layout()
        out_path = generated_dir() / "goal_region.png"
        plt.savefig(out_path, dpi=140, facecolor="#10151C")
        print(f"\nwrote {out_path} ({time.time() - t0:.0f}s)")
    except ImportError:
        print("matplotlib not available -- skipping goal_region.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
