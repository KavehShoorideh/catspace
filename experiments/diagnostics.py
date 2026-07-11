#!/usr/bin/env python
"""
experiments/diagnostics.py — post-hoc analyses after the G-M1 mixed result
(port of code/diagnostics.py). NOT the pre-registered gates -- these diagnose
WHY (a)/(b) failed while (c) passed decisively (see RESULTS-v3.md).

D1: the FIXED DTM ceiling (draw scored 90.0 = bad, not 0.0 = good -- the
    bug krk_rung1.py's buggy_dtm_ceiling_table deliberately preserves).
D2: rank probe with the DRAW column deflated (reach ranking is what
    planning uses; tests whether the L2 failure was a draw-mass artifact).
D3: concept audit via linear probes over the whole embedding (single-dim
    spearman was needlessly strict: SVD dims are rotation-arbitrary).
"""
from __future__ import annotations

import numpy as np
import scipy.stats as st

from latentchess.chain import exact_P
from latentchess.cone.tabular import sm_matvec
from latentchess.domains import krk
from latentchess.util import ridge_r2

GAMMA = 0.92


def fixed_dtm_ceiling_table(chain, dtm_w, draw_value=90.0):
    """Same MEAN-over-replies ceiling as krk_rung1's buggy_dtm_ceiling_table,
    but with the bug fixed: draw is scored `draw_value` (bad), not 0.0."""
    dtm_full = np.full(chain.n, draw_value)
    dtm_full[:chain.n_live] = np.where(np.isfinite(dtm_w), dtm_w, draw_value)
    dtm_full[chain.terminals.mate] = 0.0
    vals_flat = dtm_full[chain.out_flat]
    V = np.add.reduceat(vals_flat, chain.op0) / chain.out_counts
    neg_smin = np.maximum.reduceat(-V, chain.mp0)
    is_min = (-V) == np.repeat(neg_smin, chain.move_counts)
    cand = np.where(is_min, chain.pos_idx, chain.n_moves)
    first = np.minimum.reduceat(cand, chain.mp0)
    return (first - chain.mp0).astype(np.int32)


def play(chain, table, n_games=1000, cap=100, seed=21):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, chain.n_live, size=n_games)
    mates = 0; plies = []
    for g in range(n_games):
        s = int(starts[g])
        for ply in range(cap):
            mid = int(chain.move_ptr[s]) + int(table[s])
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


def sm_matvec_defl(P, X, gamma, draw_idx):
    mask = np.ones((P.shape[0], 1)); mask[draw_idx] = 0.0
    return sm_matvec(P, mask * X, gamma), mask


def rsvd_defl(P, gamma, d, draw_idx, seed=0, over=10):
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    Om = rng.standard_normal((n, d + over))
    Y, mask = sm_matvec_defl(P, Om, gamma, draw_idx)
    Q, _ = np.linalg.qr(Y)
    Z = mask * sm_matvec(P.T.tocsr(), Q, gamma)
    Ub, S, Vt = np.linalg.svd(Z.T, full_matrices=False)
    return (Q @ Ub)[:, :d], S[:d], Vt[:d].T


def main():
    chain = krk.build_chain()
    W, B = krk.enumerate_states()
    dtm_w, _ = krk.compute_dtm(W, B)
    feats = krk.concept_features(W, dtm_w)
    P = exact_P(chain)
    region = np.concatenate([np.where(dtm_w <= 3)[0], [chain.terminals.mate]])

    # ---------- D1 ----------
    table = fixed_dtm_ceiling_table(chain, dtm_w)
    rate, mlen = play(chain, table)
    print(f"[D1] DTM ceiling (fixed): mate-rate={rate:.3f}  mean-plies={mlen:.1f}")

    # ---------- D2 ----------
    e = np.zeros((chain.n, 1)); e[region] = 1.0
    reach_true = sm_matvec(P, e, GAMMA).ravel()
    print("[D2] draw-deflated rank probe (reach metrics on W states):")
    for d in (16, 32, 64, 128):
        U, S, V = rsvd_defl(P, GAMMA, d, chain.terminals.draw)
        F, Bm = U * S[None, :], V
        r_hat = F @ Bm[region].sum(0)
        rel = np.linalg.norm(r_hat[:chain.n_live] - reach_true[:chain.n_live]) / \
            np.linalg.norm(reach_true[:chain.n_live])
        rho = st.spearmanr(r_hat[:chain.n_live], reach_true[:chain.n_live]).statistic
        print(f"  d={d:4d}  reach_rel_err={rel:.4f}  reach_spearman={rho:.4f}")

    # ---------- D3 ----------
    U, S, V = rsvd_defl(P, GAMMA, 64, chain.terminals.draw)
    F = (U * S[None, :])[:chain.n_live]
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    print("[D3] linear-probe R^2 from 64-dim embedding (5-fold CV):")
    for cn in ("dtm", "kk_dist", "bk_edge", "box_area", "rook_bk_dist"):
        r2 = ridge_r2(F, feats[cn], folds=5, lam=1.0)
        print(f"  {cn:>12s}: R^2 = {r2:.3f}")


if __name__ == "__main__":
    main()
