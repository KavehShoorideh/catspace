"""
diagnostics.py — post-hoc analyses after G-M1 mixed result. Clearly labeled:
these are NOT the pre-registered gates; they diagnose why (a) and (b) failed
while (c) passed decisively.

D1: fix the DTM-ceiling baseline (bug: DRAW absorbing state was scored 0=good).
D2: rank probe with the DRAW column deflated (reach ranking is what planning
    uses; test whether the L2 failure was scale artifact from draw mass).
D3: concept audit via LINEAR PROBES over the whole embedding (single-dim
    spearman was needlessly strict: SVD dims are rotation-arbitrary).
"""
import numpy as np, scipy.stats as st, time
from sklearn_free import ridge_r2  # tiny local helper
from domain import enumerate_states, compute_dtm, concept_features
from learn import Chain, randomized_svd_sm, fb_from_svd, sm_matvec, reach_scores

GAMMA = 0.92
ch = Chain()
dtm_w, dtm_b = compute_dtm(ch.W, ch.B)
feats = concept_features(ch.W, dtm_w)
P = ch.exact_P_uniform()
region = np.array(list(np.where(dtm_w <= 3)[0]) + [ch.MATE_S])

# ---------- D1: fixed DTM ceiling ----------
def play(policy, n_games=1000, cap=100, seed=21):
    r = np.random.default_rng(seed)
    starts = r.integers(0, ch.nW, size=n_games)
    mates = 0; plies = []
    for g in range(n_games):
        s = int(starts[g])
        for ply in range(cap):
            mv = policy(s, r)
            outcomes = ch.moves[s][mv]
            nxt = int(outcomes[r.integers(0, len(outcomes))])
            if nxt == ch.MATE_S: mates += 1; plies.append(ply+1); break
            if nxt == ch.DRAW_S: break
            s = nxt
    return mates/n_games, (np.mean(plies) if plies else float("nan"))

def dtm_policy_fixed(s, r):
    best, bestv = 0, np.inf
    for mi, outcomes in enumerate(ch.moves[s]):
        vals = []
        for o in outcomes:
            o = int(o)
            if o == ch.MATE_S: vals.append(0.0)
            elif o == ch.DRAW_S: vals.append(90.0)      # draw is BAD (bug fixed)
            else:
                v = dtm_w[o]; vals.append(v if np.isfinite(v) else 90.0)
        v = float(np.mean(vals))
        if v < bestv: bestv, best = v, mi
    return best

rate, mlen = play(dtm_policy_fixed)
print(f"[D1] DTM ceiling (fixed): mate-rate={rate:.3f}  mean-plies={mlen:.1f}")

# ---------- D2: draw-deflated rank probe ----------
mask = np.ones((ch.n, 1)); mask[ch.DRAW_S] = 0.0
def sm_matvec_defl(Pm, X, gamma):  # (M @ diag(mask)) @ X = M @ (mask*X)
    return sm_matvec(Pm, mask * X, gamma)

def rsvd_defl(Pm, gamma, d, seed=0, over=10):
    r = np.random.default_rng(seed); n = Pm.shape[0]
    Om = r.standard_normal((n, d+over))
    Y = sm_matvec_defl(Pm, Om, gamma)
    Q, _ = np.linalg.qr(Y)
    Z = mask * sm_matvec(Pm.T.tocsr(), Q, gamma)   # (M D)^T Q = D (M^T Q)
    Ub, S, Vt = np.linalg.svd(Z.T, full_matrices=False)
    return (Q @ Ub)[:, :d], S[:d], Vt[:d].T

e = np.zeros((ch.n, 1)); e[region] = 1.0
reach_true = sm_matvec(P, e, GAMMA).ravel()
print("[D2] draw-deflated rank probe (reach metrics on W states):")
for d in (16, 32, 64, 128):
    U, S, V = rsvd_defl(P, GAMMA, d)
    F, Bm = U * S[None, :], V
    r_hat = reach_scores(F, Bm, region)
    rel = np.linalg.norm(r_hat[:ch.nW]-reach_true[:ch.nW])/np.linalg.norm(reach_true[:ch.nW])
    rho = st.spearmanr(r_hat[:ch.nW], reach_true[:ch.nW]).statistic
    print(f"  d={d:4d}  reach_rel_err={rel:.4f}  reach_spearman={rho:.4f}")

# ---------- D3: linear-probe concept audit ----------
U, S, V = rsvd_defl(P, GAMMA, 64)
F = (U * S[None, :])[:ch.nW]
F = (F - F.mean(0)) / (F.std(0) + 1e-9)
print("[D3] linear-probe R^2 from 64-dim embedding (5-fold CV):")
for cn in ("dtm", "kk_dist", "bk_edge", "box_area", "rook_bk_dist"):
    y = feats[cn]
    r2 = ridge_r2(F, y, folds=5, lam=1.0)
    print(f"  {cn:>12s}: R^2 = {r2:.3f}")
