#!/usr/bin/env python
"""
experiments/krk_rung1.py — Milestone 1 rung-1 run on the latentchess stack:
rank probe on exact dynamics, a random-play learning curve, the spectral
concept audit, VQ plan tokens, and the cone-steering engine evaluation.
Port of code/experiment.py -- see artifacts/RESULTS-v3.md for the original
narrative (including the pre-registered gate's own documented mis-
registration: (a)/(b) fail for diagnosable reasons, (c) passes at 26x).

Simplification vs the original: the learned engine here scores every live
outcome via F(o)@zG uniformly (the original additionally excluded NEVER-
VISITED outcomes from each move's mean, which only matters at low game
counts / low state coverage -- at the 32000-game headline number coverage
is ~100% and the two conventions coincide).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.stats as st

from latentchess.chain import exact_P, empirical_P
from latentchess.concepts import KMeansVQ, usage_perplexity
from latentchess.cone.tabular import fb_from_svd, randomized_svd_sm, rank_error, sm_matvec
from latentchess.domains import krk
from latentchess.game import rollout_transitions
from latentchess.opponents import RandomOpponent
from latentchess.planner.policy import RandomPolicy, TablePolicy
from latentchess.planner.readout import ReplyAgg, greedy_policy
from latentchess.scoring import TerminalScores, fill_terminal_state_scores

GAMMA = 0.92


def buggy_dtm_ceiling_table(chain, dtm_w):
    """THE historical bug (RESULTS-v3 bug ledger): DRAW absorbing scored 0.0
    (same as MATE) instead of "bad", giving a ~42% ceiling instead of ~100%
    -- kept here deliberately, exactly as experiment.py originally computed
    it, since it's part of the documented finding, not a defect to silently
    fix in a script whose job is reproducing the historical run."""
    dtm_full = np.zeros(chain.n)
    dtm_full[:chain.n_live] = dtm_w
    vals_flat = dtm_full[chain.out_flat]
    V = np.add.reduceat(vals_flat, chain.op0) / chain.out_counts
    neg_smin = np.maximum.reduceat(-V, chain.mp0)
    is_min = (-V) == np.repeat(neg_smin, chain.move_counts)
    cand = np.where(is_min, chain.pos_idx, chain.n_moves)
    first = np.minimum.reduceat(cand, chain.mp0)
    return (first - chain.mp0).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-counts", type=int, nargs="+", default=[500, 2000, 8000, 32000])
    ap.add_argument("--d-learn", type=int, default=64)
    args = ap.parse_args()

    print("== setup ==")
    t0 = time.time()
    chain = krk.build_chain()
    W, B = krk.enumerate_states()
    dtm_w, _ = krk.compute_dtm(W, B)
    feats = krk.concept_features(W, dtm_w)
    P_exact = exact_P(chain)
    print(f"setup {time.time() - t0:.1f}s | W={chain.n_live}")

    region = np.concatenate([np.where(dtm_w <= 3)[0], [chain.terminals.mate]])
    print(f"near-mate region size: {len(region)}")

    e = np.zeros((chain.n, 1)); e[region] = 1.0
    reach_true = sm_matvec(P_exact, e, GAMMA).ravel()

    # ---------- (1) rank probe on exact M ----------
    print("\n== rank probe (exact dynamics) ==")
    rank_results = {}
    for d in (8, 16, 32, 64, 128):
        U, S, V = randomized_svd_sm(P_exact, GAMMA, d=d, seed=0)
        F, Bm = fb_from_svd(U, S, V)
        fro = rank_error(P_exact, GAMMA, F, Bm, n_probe=8)
        r_hat = F @ Bm[region].sum(0)
        reach_rel = np.linalg.norm(r_hat - reach_true) / np.linalg.norm(reach_true)
        rho = st.spearmanr(r_hat[:chain.n_live], reach_true[:chain.n_live]).statistic
        rank_results[d] = (fro, reach_rel, rho)
        print(f"d={d:4d}  frobenius={fro:.3f}  reach_rel_err={reach_rel:.4f}  reach_spearman={rho:.4f}")

    # ---------- (2) learning curve from random play ----------
    print("\n== learning from random play ==")
    learned = {}
    for ng in args.game_counts:
        rng = np.random.default_rng(11)
        starts = rng.integers(0, chain.n_live, size=ng)
        rows, cols, _ = rollout_transitions(chain, RandomPolicy(), RandomOpponent(), starts,
                                            cap=200, rng=rng)
        Phat, visited = empirical_P(rows, cols, chain.n, chain.terminals)
        U, S, V = randomized_svd_sm(Phat, GAMMA, d=args.d_learn, seed=0)
        F, Bm = fb_from_svd(U, S, V)
        r_hat = F @ Bm[region].sum(0)
        vis = visited[:chain.n_live]
        cov = vis.mean()
        rho = st.spearmanr(r_hat[:chain.n_live][vis], reach_true[:chain.n_live][vis]).statistic
        learned[ng] = dict(F=F, B=Bm, visited=visited, coverage=cov, rho=rho, n_tr=len(rows))
        print(f"games={ng:6d}  transitions={len(rows):7d}  state-coverage={cov:.2%}  "
              f"reach_spearman(visited)={rho:.4f}")

    U, S, V = randomized_svd_sm(P_exact, GAMMA, d=args.d_learn, seed=0)
    F_ex, B_ex = fb_from_svd(U, S, V)

    # ---------- (3) emergent concepts: spectral dims vs ground truth ----------
    print("\n== emergent concepts (spectral dims vs ground-truth features) ==")
    Fn = F_ex[:chain.n_live]
    concept_names = ["dtm", "kk_dist", "bk_edge", "box_area", "rook_bk_dist"]
    audit = np.zeros((16, len(concept_names)))
    for j, cn in enumerate(concept_names):
        for dim in range(16):
            audit[dim, j] = st.spearmanr(Fn[:, dim], feats[cn]).statistic
    n_strong, used_dims = 0, set()
    for j, cn in enumerate(concept_names):
        dim = int(np.nanargmax(np.abs(audit[:, j])))
        rho = audit[dim, j]
        print(f"concept {cn:>12s}: best dim {dim:2d}  rho={rho:+.3f}")
        if abs(rho) > 0.5 and dim not in used_dims:
            n_strong += 1; used_dims.add(dim)
    print(f"distinct dims with |rho|>0.5: {n_strong}")

    # ---------- (4) VQ plan tokens ----------
    print("\n== VQ plan tokens (k-means on cone shapes F) ==")
    Xf = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-9)
    K = 32
    vq = KMeansVQ(n_tokens=K, seed=5).fit(Xf[:, :16])
    assign = vq.tokens(Xf[:, :16])
    perp = usage_perplexity(assign, K)
    n_used = int((np.bincount(assign, minlength=K) > 0).sum())
    print(f"K={K}  codes used={n_used}  usage perplexity={perp:.1f}")
    print("code | size |  mean DTM | mean box_area")
    order = np.argsort([feats['dtm'][assign == k].mean() if (assign == k).any() else 99 for k in range(K)])
    for k in order[:10]:
        m = assign == k
        print(f"{k:4d} | {m.sum():4d} | {feats['dtm'][m].mean():8.2f} | {feats['box_area'][m].mean():8.2f}")

    # ---------- (5) engine + evaluation ----------
    print("\n== engine evaluation (vs uniform-random black, 100-ply cap) ==")
    ts_original = TerminalScores(mate=1e6, draw=0.0, bwin=0.0)   # the historically-weaker convention

    def play(policy_table, n_games=1000, cap=100, seed=21):
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, chain.n_live, size=n_games)
        mates = 0; plies = []
        for g in range(n_games):
            s = int(starts[g])
            for ply in range(cap):
                mid = int(chain.move_ptr[s]) + int(policy_table[s])
                k = int(chain.move_kind[mid])
                if k == 1:
                    mates += 1; plies.append(ply + 1); break
                if k == 2:
                    break
                outs = chain.outs_of(mid)
                nxt = int(outs[rng.integers(0, len(outs))])
                if nxt >= chain.n_live:
                    break
                s = nxt
        return mates / n_games, (float(np.mean(plies)) if plies else float("nan"))

    def engine_table(F, Bm):
        # F/Bm are built from the full n x n (live + absorbing) P, so F already
        # has one row per chain state -- no slicing needed before F @ zG.
        zG = Bm[region].sum(0)
        state_scores = fill_terminal_state_scores(F @ zG, chain, ts_original)
        return greedy_policy(state_scores, chain, ReplyAgg.MEAN, ts_original)

    def random_policy_play(n_games=1000, cap=100, seed=21):
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, chain.n_live, size=n_games)
        mates = 0; plies = []
        for g in range(n_games):
            s = int(starts[g])
            for ply in range(cap):
                a, b = int(chain.move_ptr[s]), int(chain.move_ptr[s + 1])
                mid = a + int(rng.integers(0, b - a))
                k = int(chain.move_kind[mid])
                if k == 1:
                    mates += 1; plies.append(ply + 1); break
                if k == 2:
                    break
                outs = chain.outs_of(mid)
                nxt = int(outs[rng.integers(0, len(outs))])
                if nxt >= chain.n_live:
                    break
                s = nxt
        return mates / n_games, (float(np.mean(plies)) if plies else float("nan"))

    base_rate, base_len = random_policy_play()
    print(f"random white       : mate-rate={base_rate:.3f}  mean-plies={base_len:.1f}")
    ceil_rate, ceil_len = play(buggy_dtm_ceiling_table(chain, dtm_w))
    print(f"DTM-guided ceiling : mate-rate={ceil_rate:.3f}  mean-plies={ceil_len:.1f}")

    rates = {}
    for ng in args.game_counts:
        L = learned[ng]
        table = engine_table(L["F"], L["B"])
        rate, mlen = play(table)
        rates[ng] = rate
        print(f"learned ({ng:6d} games, cov {L['coverage']:.0%}): "
              f"mate-rate={rate:.3f}  mean-plies={mlen:.1f}  (x{rate / max(base_rate, 1e-9):.1f} vs random)")
    ex_rate, ex_len = play(engine_table(F_ex, B_ex))
    print(f"exact-dynamics ref : mate-rate={ex_rate:.3f}  mean-plies={ex_len:.1f}")

    # ---------- GATE ----------
    print("\n== GATE G-M1 ==")
    ga = rank_results[64][1] < 0.05
    gb = n_strong >= 3
    last_ng = args.game_counts[-1]
    gc = rates[last_ng] >= 5 * base_rate
    print(f"(a) rank-64 reach error < 5%   : {'PASS' if ga else 'FAIL'} ({rank_results[64][1]:.4f})")
    print(f"(b) >=3 concept dims |rho|>0.5 : {'PASS' if gb else 'FAIL'} ({n_strong})")
    print(f"(c) engine >= 5x random        : {'PASS' if gc else 'FAIL'} "
          f"({rates[last_ng]:.3f} vs {base_rate:.3f})")
    print("G-M1:", "PASS" if (ga and gb and gc) else "FAIL")
    print(f"\ntotal time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
