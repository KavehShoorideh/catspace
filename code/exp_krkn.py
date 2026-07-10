"""
exp_krkn.py — the two-sided domain, end to end.

New evaluation axes vs KRRK:
  - WIN/DRAW DISCOVERY: AUC of the learned reach score classifying
    game-theoretically won vs drawn positions (never labeled in training)
  - fork survival: rate of losing the rook from won positions
  - conversion route: fraction of mates that pass through the KRK stratum
    (does it learn to trade the knight off when winning?)
"""
import numpy as np, scipy.sparse as sp, time, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from krkn import KRKNChain, compute_dtm_krkn, KN_ATT, KNIGHT, rook_attacks, rc
from learn import randomized_svd_sm, fb_from_svd

GAMMA = 0.93
t0 = time.time()
uc = KRKNChain(verbose=False)
try: dtm = np.load("dtm_krkn.npy")
except FileNotFoundError: dtm, _ = compute_dtm_krkn(uc)
won = np.isfinite(dtm[:uc.n2])
print(f"chain ready | won {won.mean():.1%} ({time.time()-t0:.0f}s)")

region = np.concatenate([np.where(dtm <= 3)[0], [uc.MATE_S]])

# optimal black per ongoing move: DRAW/BWIN best, else max dtm
B_opt = np.zeros(len(uc.move_kind), dtype=np.int32)
mk = uc.move_kind
for mid in range(len(mk)):
    if mk[mid] == 0:
        outs = uc.outs_of(mid)
        vals = np.where(outs >= uc.nW, 1e6, np.where(np.isfinite(dtm[np.minimum(outs, uc.nW-1)]),
                        dtm[np.minimum(outs, uc.nW-1)], 1e6))
        B_opt[mid] = int(np.argmax(vals))
print(f"optimal replies ({time.time()-t0:.0f}s)")

def greedy_pol(scores):
    pol = np.zeros(uc.nW, dtype=np.int32)
    mp = uc.move_ptr
    for s in range(uc.nW):
        a, b = mp[s], mp[s+1]
        best, bv = 0, -np.inf
        for j, mid in enumerate(range(a, b)):
            k = mk[mid]
            if k == 1: best, bv = j, np.inf; break
            if k in (2, 3): continue
            v = float(scores[uc.outs_of(mid)].mean())
            if v > bv: bv, best = v, j
        pol[s] = best
    return pol

def sample_round(pol, ew, eb, ng, seed, cap=150):
    r = np.random.default_rng(seed)
    rows, cols, n_mate = [], [], 0
    for s0 in r.integers(0, uc.n2, size=ng):
        s = int(s0)
        for _ in range(cap):
            a, b = uc.move_ptr[s], uc.move_ptr[s+1]
            j = int(pol[s]) if r.random() > ew else int(r.integers(0, b - a))
            mid = a + j
            k = mk[mid]
            if k == 1: nxt = uc.MATE_S
            elif k == 2: nxt = uc.DRAW_S
            elif k == 3: nxt = int(uc.outs_of(mid)[0])
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
    counts = sp.coo_matrix((np.ones(len(all_rows)), (all_rows, all_cols)), shape=(uc.n, uc.n)).tocsr()
    rowsum = np.asarray(counts.sum(1)).ravel(); seen = rowsum > 0; rowsum[rowsum == 0] = 1
    P = (sp.diags(1/rowsum) @ counts).tolil()
    for a in (uc.MATE_S, uc.DRAW_S, uc.BWIN_S): P[a, :] = 0; P[a, a] = 1.0
    for i in np.where(~seen)[0]: P[i, i] = 1.0
    U, S, V = randomized_svd_sm(P.tocsr(), GAMMA, d=d, seed=0)
    F, Bm = fb_from_svd(U, S, V)
    return F, Bm, (F @ Bm[region].sum(0))

def auc(pos, neg):
    from numpy import concatenate as cat
    x = cat([pos, neg]); y = cat([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(x); ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x)+1)
    rp = ranks[y == 1].sum()
    return (rp - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

def eval_all(scores, n_eval=300, cap=70, seed=99):
    r = np.random.default_rng(seed)
    won_idx = np.where(won)[0]
    starts = won_idx[r.integers(0, len(won_idx), size=n_eval)]
    mates, ratios, rook_lost, via_krk = 0, [], 0, 0
    for s0 in starts:
        s = int(s0); d0 = dtm[s]; crossed = False
        for wm in range(cap):
            a, b = uc.move_ptr[s], uc.move_ptr[s+1]
            best, bv = a, -np.inf
            for mid in range(a, b):
                k = mk[mid]
                if k == 1: best, bv = mid, np.inf; break
                if k in (2, 3): continue
                v = float(scores[uc.outs_of(mid)].mean())
                if v > bv: bv, best = v, mid
            k = mk[best]
            if k == 1:
                mates += 1; via_krk += crossed
                ratios.append((wm+1)/max(1.0, np.ceil(d0/2))); break
            if k in (2, 3): break
            nxt = int(uc.outs_of(best)[B_opt[best]])
            if nxt == uc.DRAW_S: rook_lost += 1; break
            if nxt >= uc.nW: break
            if uc.n2 <= nxt < uc.nW: crossed = True
            s = nxt
    sc = scores[:uc.n2]
    a_wd = auc(sc[won], sc[~won])
    return mates/n_eval, (np.mean(ratios) if ratios else np.nan), rook_lost/n_eval, \
           (via_krk/max(mates,1)), a_wd

print("\nround | b-eps | data mates | vs-OPT mate (won starts) | mv/opt | rook-lost | via-KRK | WIN/DRAW AUC")
all_rows, all_cols = [], []
scores = np.zeros(uc.n)
for k, (ew, eb, ng) in enumerate([(1.0, 0.7, 12000), (0.3, 0.4, 12000),
                                   (0.25, 0.2, 12000), (0.2, 0.0, 12000),
                                   (0.2, 0.0, 12000)]):
    pol = greedy_pol(scores)
    rows, cols, nm = sample_round(pol, ew, eb, ng, seed=100+k)
    all_rows += rows; all_cols += cols
    F, Bm, scores = estimate(all_rows, all_cols)
    rate, ratio, rl, vk, a_wd = eval_all(scores)
    print(f"  {k}   | {eb:.2f}  |  {nm:6d}   |          {rate:.3f}           |  {ratio:.2f}  |   {rl:.3f}   |  {vk:.2f}   |   {a_wd:.3f}   ({time.time()-t0:.0f}s)")

np.save("krkn_scores.npy", scores); np.save("krkn_F.npy", F); np.save("krkn_B.npy", Bm)
print(f"training done ({time.time()-t0:.0f}s)")
