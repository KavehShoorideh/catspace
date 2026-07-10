"""
exp_krrk.py — the richer domain, end to end:
  - PI curriculum training on the stratified union chain (KRRK + KRK strata)
  - evaluation vs TRUE optimal defense (mate rate, tempo ratio, stratum drops)
  - stratified region map: token territories, the KRK chute, tug-of-war games
  - filmstrip PNG of one learned game

Training diet per the directive: curriculum from round 0 (white eps-greedy on
current scores, black eps_b-optimal annealed), NOT pure random self-play.
"""
import numpy as np, scipy.sparse as sp, time, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from krrk import UnionChain, compute_dtm_union, rc, rook_attacks
from learn import randomized_svd_sm, fb_from_svd

GAMMA = 0.90
t0 = time.time()
uc = UnionChain(verbose=False)
try:
    dtm = np.load("dtm_union.npy")
except FileNotFoundError:
    dtm, _ = compute_dtm_union(uc)
print(f"chain ready ({time.time()-t0:.0f}s)")

region = np.concatenate([np.where(dtm <= 3)[0], [uc.MATE_S]])

# ---- optimal black reply per ongoing move (max dtm over outcomes)
B_opt = np.zeros(len(uc.move_kind), dtype=np.int32)
for mid in range(len(uc.move_kind)):
    if uc.move_kind[mid] == 0:
        outs = uc.outs_of(mid)
        vals = np.where(outs >= uc.nW, 1e6, dtm[np.minimum(outs, uc.nW-1)])   # DRAW_S outcome = escape = best
        B_opt[mid] = int(np.argmax(vals))
print(f"optimal replies precomputed ({time.time()-t0:.0f}s)")

def greedy_pol(scores):
    """Per-state best LOCAL move index. scores has length uc.n (absorbing rows included)."""
    pol = np.zeros(uc.nW, dtype=np.int32)
    mk, mp = uc.move_kind, uc.move_ptr
    for s in range(uc.nW):
        a, b = mp[s], mp[s+1]
        best, bv = 0, -np.inf
        for j, mid in enumerate(range(a, b)):
            k = mk[mid]
            if k == 1: best = j; bv = np.inf; break
            if k == 2: continue
            v = float(scores[uc.outs_of(mid)].mean())
            if v > bv: bv, best = v, j
        pol[s] = best
    return pol

def sample_round(pol, ew, eb, ng, seed, cap=120):
    r = np.random.default_rng(seed)
    rows, cols = [], []
    n_mate = 0
    starts = r.integers(0, uc.n2, size=ng)          # start in the KRRK stratum
    for s0 in starts:
        s = int(s0)
        for _ in range(cap):
            a, b = uc.move_ptr[s], uc.move_ptr[s+1]
            j = int(pol[s]) if r.random() > ew else int(r.integers(0, b - a))
            mid = a + j
            k = uc.move_kind[mid]
            if k == 1: nxt = uc.MATE_S
            elif k == 2: nxt = uc.DRAW_S
            else:
                outs = uc.outs_of(mid)
                bi = int(B_opt[mid]) if r.random() > eb else int(r.integers(0, len(outs)))
                nxt = int(outs[bi])
            rows.append(s); cols.append(nxt)
            if nxt == uc.MATE_S: n_mate += 1
            if nxt >= uc.nW: break
            s = nxt
    return rows, cols, n_mate

def estimate(all_rows, all_cols, d=64):
    counts = sp.coo_matrix((np.ones(len(all_rows)), (all_rows, all_cols)),
                           shape=(uc.n, uc.n)).tocsr()
    rowsum = np.asarray(counts.sum(1)).ravel(); seen = rowsum > 0; rowsum[rowsum == 0] = 1
    P = (sp.diags(1/rowsum) @ counts).tolil()
    for i in np.where(~seen)[0]: P[i, i] = 1.0
    for a in (uc.MATE_S, uc.DRAW_S): P[a, :] = 0; P[a, a] = 1.0
    U, S, V = randomized_svd_sm(P.tocsr(), GAMMA, d=d, seed=0)
    F, Bm = fb_from_svd(U, S, V)
    return F, Bm, (F @ Bm[region].sum(0))

def eval_vs_optimal(scores, n_eval=300, cap=40, seed=99):
    r = np.random.default_rng(seed)
    starts = r.integers(0, uc.n2, size=n_eval)
    mates, ratios, drops = 0, [], 0
    for s0 in starts:
        s = int(s0); d0 = dtm[s]; dropped = False
        for wm in range(cap):
            a, b = uc.move_ptr[s], uc.move_ptr[s+1]
            best, bv = a, -np.inf
            for mid in range(a, b):
                k = uc.move_kind[mid]
                if k == 1: best, bv = mid, np.inf; break
                if k == 2: continue
                v = float(scores[uc.outs_of(mid)].mean())
                if v > bv: bv, best = v, mid
            k = uc.move_kind[best]
            if k == 1:
                mates += 1; ratios.append((wm+1) / max(1.0, np.ceil(d0/2))); break
            if k == 2: break
            outs = uc.outs_of(best)
            nxt = int(outs[B_opt[best]])
            if nxt == uc.DRAW_S: break
            if nxt >= uc.n2 and nxt < uc.nW and not dropped: drops += 1; dropped = True
            if nxt >= uc.nW: break
            s = nxt
    return mates/n_eval, (float(np.mean(ratios)) if ratios else np.nan), drops/n_eval

print("\nround | black eps | data mates | vs-OPT mate | moves/opt | stratum-drop")
all_rows, all_cols = [], []
scores = np.zeros(uc.n)
for k, (ew, eb, ng) in enumerate([(1.0, 0.7, 15000), (0.3, 0.4, 15000),
                                   (0.25, 0.2, 15000), (0.2, 0.0, 15000),
                                   (0.2, 0.0, 15000)]):
    pol = greedy_pol(scores)
    rows, cols, nm = sample_round(pol, ew, eb, ng, seed=100+k)
    all_rows += rows; all_cols += cols
    F, Bm, scores = estimate(all_rows, all_cols)
    rate, ratio, drop = eval_vs_optimal(scores)
    print(f"  {k}   |   {eb:.2f}    |  {nm:6d}   |    {rate:.3f}    |   {ratio:.2f}   |   {drop:.3f}   ({time.time()-t0:.0f}s)")

np.save("krrk_scores.npy", scores); np.save("krrk_F.npy", F); np.save("krrk_B.npy", Bm)
print(f"training done ({time.time()-t0:.0f}s)")
