#!/usr/bin/env python
"""
experiments/generalization.py — does a neural FB generalize to states it has
NEVER seen in any training pair (familiar concept family, unfamiliar exact
position)? Port of code/exp_generalization.py onto the latentchess stack:
ChainRolloutSource for the neural-FB training pool, rollout_transitions +
empirical_P for the tabular-FB baseline, and greedy_policy/arena.evaluate for
the E2/E3 engine comparisons (instead of hand-rolled scoring/play loops).

E1 reach-ranking: spearman(F(s)@zG, exact reach) on holdout vs train states.
E2/E3: engine mate-rate from HELD-OUT starts, vs optimal and vs random black.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.stats as st

from latentchess.arena import evaluate
from latentchess.chain import exact_P, empirical_P
from latentchess.cone.neural import NeuralFB, absorbing_vec, one_hot_state
from latentchess.cone.tabular import fb_from_svd, randomized_svd_sm, sm_matvec
from latentchess.data.sources import ChainRolloutSource
from latentchess.domains import krk
from latentchess.game import rollout_transitions
from latentchess.opponents import EpsOptimalDTM, RandomOpponent, optimal_reply_table
from latentchess.planner.policy import RandomPolicy, TablePolicy
from latentchess.planner.readout import ReplyAgg, greedy_policy
from latentchess.scoring import TerminalScores, fill_terminal_state_scores


def _policy_from_live_scores(live_scores: np.ndarray, mate_score: float, draw_score: float, chain):
    ts = TerminalScores(mate=mate_score, draw=draw_score, bwin=draw_score)
    full = np.zeros(chain.n)
    full[:chain.n_live] = live_scores
    full = fill_terminal_state_scores(full, chain, ts)
    table = greedy_policy(full, chain, ReplyAgg.MEAN, ts)
    return TablePolicy(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=32000)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--gamma", type=float, default=0.92)
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=400)
    args = ap.parse_args()

    t0 = time.time()
    chain = krk.build_chain()
    W, B = krk.enumerate_states()
    dtm_w, _ = krk.compute_dtm(W, B)
    region_states = np.where(dtm_w <= 3)[0]
    region_full = np.concatenate([region_states, [chain.terminals.mate]])

    P_exact = exact_P(chain)
    e = np.zeros((chain.n, 1)); e[region_full] = 1.0
    reach_true = sm_matvec(P_exact, e, args.gamma).ravel()

    rng = np.random.default_rng(args.seed)
    holdout_mask = rng.random(chain.n_live) < args.holdout
    print(f"holdout states: {holdout_mask.sum()} / {chain.n_live}")

    # ---- tabular baseline: empirical P from a random-play rollout, holdout dropped
    starts = rng.integers(0, chain.n_live, size=args.n_games)
    rows, cols, _ = rollout_transitions(chain, RandomPolicy(), RandomOpponent(), starts,
                                        cap=200, rng=np.random.default_rng(args.seed + 1))
    rows_a, cols_a = np.array(rows), np.array(cols)
    keep = np.array([
        not (r < chain.n_live and holdout_mask[r]) and not (c < chain.n_live and holdout_mask[c])
        for r, c in zip(rows_a, cols_a)
    ])
    P_tab, seen_tab = empirical_P(rows_a[keep], cols_a[keep], chain.n, chain.terminals)
    U, S, V = randomized_svd_sm(P_tab, args.gamma, d=32, seed=0)
    F_tab, B_tab = fb_from_svd(U, S, V)
    zg_tab = B_tab[region_full].sum(0)
    print(f"tabular baseline: {len(rows_a)} raw transitions, {keep.sum()} holdout-free")

    # ---- neural FB training via ChainRolloutSource (the new pluggable data source)
    print("training neural FB...")
    source = ChainRolloutSource(chain, RandomPolicy(), RandomOpponent(), gamma=args.gamma,
                                 n_games=args.n_games, max_plies=200, holdout_mask=holdout_mask)
    anchors, goals = source.all_pairs(seed=args.seed + 10)
    print(f"training pairs (holdout-free): {len(anchors)}")

    net = NeuralFB(d=32, dh=256, seed=0, tau=0.1)
    X_all = np.stack([one_hot_state(s) for s in W]).astype(np.float32)
    X_mate = absorbing_vec(0)[None, :]
    X_draw = absorbing_vec(1)[None, :]

    def gvec(idx: int):
        if idx == chain.terminals.mate:
            return X_mate[0]
        if idx == chain.terminals.draw:
            return X_draw[0]
        return X_all[idx]

    BS = 256
    sched = [(0, 1e-3), (8000, 3e-4)]
    train_rng = np.random.default_rng(args.seed + 2)
    for step in range(args.steps):
        lr = [l for s0, l in sched if step >= s0][-1]
        idx = train_rng.integers(0, len(anchors), size=BS)
        Xs = X_all[anchors[idx]]
        Xg = np.stack([gvec(int(g)) for g in goals[idx]])
        loss = net.train_step(Xs, Xg, lr)
        if step % 2000 == 0:
            print(f"  step {step:6d}  loss {loss:.3f}  ({time.time() - t0:.0f}s)")
    print(f"training done ({time.time() - t0:.0f}s)")

    Fn = net.embed_F(X_all)
    Bn_states = net.embed_B(X_all)
    Bn_mate = net.embed_B(X_mate)[0]
    zg_n = Bn_states[region_states].sum(0) + Bn_mate

    # ---- E1: reach ranking at holdout vs train states
    r_n = Fn @ zg_n
    r_t = F_tab[:chain.n_live] @ zg_tab
    hi, ti = np.where(holdout_mask)[0], np.where(~holdout_mask)[0]
    holdout_rho = st.spearmanr(r_n[hi], reach_true[hi]).statistic
    train_rho = st.spearmanr(r_n[ti], reach_true[ti]).statistic
    print("\nE1 reach-ranking spearman vs exact reach:")
    print(f"  neural  train={train_rho:.3f}  HOLDOUT={holdout_rho:.3f}")
    print(f"  tabular train={st.spearmanr(r_t[ti], reach_true[ti]).statistic:.3f}  "
          f"HOLDOUT={st.spearmanr(r_t[hi], reach_true[hi]).statistic:.3f}   "
          f"(tabular has no rows for unseen states)")

    # ---- E2/E3: engines from HELD-OUT starts, via the shared readout (mean-of-outcomes)
    mate_n, draw_n = float(np.quantile(r_n, 0.999)), float(np.quantile(r_n, 0.001))
    neural_policy = _policy_from_live_scores(r_n, mate_n, draw_n, chain)
    tab_scores = np.where(seen_tab[:chain.n_live], r_t, 0.0)
    tab_policy = _policy_from_live_scores(tab_scores, mate_n, 0.0, chain)

    b_opt = optimal_reply_table(chain, dtm_w)
    optimal_black = EpsOptimalDTM(b_opt, eps=0.0)
    random_black = RandomOpponent()

    n_eval = min(args.n_eval, len(hi))
    hold_starts = rng.choice(hi, size=n_eval, replace=False)
    print(f"\nE2/E3 engines starting FROM {n_eval} HELD-OUT states:")
    for name, policy in (("neural ", neural_policy), ("tabular", tab_policy)):
        mo = evaluate(chain, dtm_w, policy, optimal_black, hold_starts, cap=80).conversion
        mr = evaluate(chain, dtm_w, policy, random_black, hold_starts, cap=80).conversion
        print(f"  {name}: vs OPTIMAL black mate-rate={mo:.3f}   vs random black={mr:.3f}")

    print(f"\ntotal time {time.time() - t0:.0f}s")
    print(f"\nVERDICT: HOLDOUT_SPEARMAN={holdout_rho:.3f} TRAIN_SPEARMAN={train_rho:.3f}")
    band_ok = 0.35 <= holdout_rho <= 0.60 and abs(holdout_rho - train_rho) < 0.10
    print("PASS" if band_ok else "FAIL", "(expected band: holdout in [0.35,0.60], |holdout-train|<0.10)")
    return 0 if band_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
