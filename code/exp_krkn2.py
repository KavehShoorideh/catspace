"""
exp_krkn2.py — vectorized + checkpointed KRKN training.
All per-move work is numpy segment ops (reduceat); games are index-chasing.
Resumes from checkpoint if interrupted; run repeatedly until done.
"""
import numpy as np, scipy.sparse as sp, time, os, pickle
from krkn import KRKNChain, compute_dtm_krkn
from learn import randomized_svd_sm, fb_from_svd

GAMMA = 0.93
CKPT = "krkn_ckpt.pkl"
t0 = time.time()
uc = KRKNChain(verbose=False)
dtm = np.load("dtm_krkn.npy")
won = np.isfinite(dtm[:uc.n2])
print(f"chain ready ({time.time()-t0:.0f}s)")

region = np.concatenate([np.where(dtm <= 3)[0], [uc.MATE_S]])
mk = uc.move_kind
mp0, mp1 = uc.move_ptr[:-1], uc.move_ptr[1:]
op0 = uc.out_ptr[:-1]
out_counts = np.diff(uc.out_ptr)
move_counts = np.diff(uc.move_ptr)
n_moves = len(mk)
pos_idx = np.arange(n_moves)

# ---- vectorized optimal-black reply per move
dtm_full = np.full(uc.n, 1e6)
dtm_full[:uc.nW] = np.where(np.isfinite(dtm), dtm, 1e6)
vals_flat = dtm_full[uc.out_flat]
seg_max = np.maximum.reduceat(vals_flat, op0)
is_max = vals_flat == np.repeat(seg_max, out_counts)
cand = np.where(is_max, np.arange(len(vals_flat)), len(vals_flat))
first = np.minimum.reduceat(cand, op0)
B_opt = (first - op0).astype(np.int32)
print(f"B_opt vectorized ({time.time()-t0:.0f}s)")

def move_values(scores):
    """Mean of scores over each move's outcome set, with kind overrides."""
    sums = np.add.reduceat(scores[uc.out_flat], op0)
    V = sums / out_counts
    V[mk == 1] = 1e18
    V[(mk == 2) | (mk == 3)] = -1e18
    return V

def greedy_pol(scores):
    V = move_values(scores)
    smax = np.maximum.reduceat(V, mp0)
    is_m = V == np.repeat(smax, move_counts)
    cand = np.where(is_m, pos_idx, n_moves)
    first = np.minimum.reduceat(cand, mp0)
    return (first - mp0).astype(np.int32)

def sample_round(pol, ew, eb, ng, seed, cap=120, dtm_cap=None):
    """Reverse-start curriculum: 70% of starts from WON states with
    dtm <= dtm_cap (annealed outward), 30% uniform. dtm_cap=None -> uniform."""
    r = np.random.default_rng(seed)
    if dtm_cap is not None:
        pool = np.where(won & (dtm[:uc.n2] <= dtm_cap))[0]
        n_cur = int(0.7 * ng)
        starts = np.concatenate([pool[r.integers(0, len(pool), size=n_cur)],
                                 r.integers(0, uc.n2, size=ng - n_cur)])
    else:
        starts = r.integers(0, uc.n2, size=ng)
    rows, cols, n_mate = [], [], 0
    rand01 = r.random  # local
    for s0 in starts:
        s = int(s0)
        for _ in range(cap):
            a = mp0[s]
            j = int(pol[s]) if rand01() > ew else int(r.integers(0, move_counts[s]))
            mid = a + j
            k = mk[mid]
            if k == 1: nxt = uc.MATE_S
            elif k == 2: nxt = uc.DRAW_S
            elif k == 3: nxt = int(uc.out_flat[op0[mid]])
            else:
                if rand01() > eb: bi = int(B_opt[mid])
                else: bi = int(r.integers(0, out_counts[mid]))
                nxt = int(uc.out_flat[op0[mid] + bi])
            rows.append(s); cols.append(nxt)
            if nxt == uc.MATE_S: n_mate += 1
            if nxt >= uc.nW: break
            s = nxt
    return rows, cols, n_mate

def estimate(rows_all, cols_all, d=48):
    counts = sp.coo_matrix((np.ones(len(rows_all)), (rows_all, cols_all)),
                           shape=(uc.n, uc.n)).tocsr()
    rowsum = np.asarray(counts.sum(1)).ravel(); seen = rowsum > 0; rowsum[rowsum == 0] = 1
    P = (sp.diags(1/rowsum) @ counts).tolil()
    for a in (uc.MATE_S, uc.DRAW_S, uc.BWIN_S): P[a, :] = 0; P[a, a] = 1.0
    diag_fix = np.where(~seen)[0]
    for i in diag_fix: P[i, i] = 1.0
    U, S, V = randomized_svd_sm(P.tocsr(), GAMMA, d=d, n_oversample=8, seed=0)
    F, Bm = fb_from_svd(U, S, V)
    return F, Bm, (F @ Bm[region].sum(0))

def auc(pos, neg):
    x = np.concatenate([pos, neg])
    order = np.argsort(x); ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x)+1)
    rp = ranks[:len(pos)][...] if False else ranks[np.arange(len(pos))]
    # careful: ranks are aligned to x's original order (pos first)
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

def eval_all(scores, n_eval=300, cap=70, seed=99):
    r = np.random.default_rng(seed)
    pol = greedy_pol(scores)
    won_idx = np.where(won)[0]
    starts = won_idx[r.integers(0, len(won_idx), size=n_eval)]
    mates, ratios, rook_lost, via_krk = 0, [], 0, 0
    for s0 in starts:
        s = int(s0); d0 = dtm[s]; crossed = False
        for wm in range(cap):
            mid = mp0[s] + int(pol[s])
            k = mk[mid]
            if k == 1:
                mates += 1; via_krk += crossed
                ratios.append((wm+1)/max(1.0, np.ceil(d0/2))); break
            if k in (2, 3): break
            nxt = int(uc.out_flat[op0[mid] + B_opt[mid]])
            if nxt == uc.DRAW_S: rook_lost += 1; break
            if nxt >= uc.nW: break
            if nxt >= uc.n2: crossed = True
            s = nxt
    sc = scores[:uc.n2]
    return (mates/n_eval, (float(np.mean(ratios)) if ratios else np.nan),
            rook_lost/n_eval, via_krk/max(mates, 1), auc(sc[won], sc[~won]))

SCHEDULE = [  # (white eps, black eps, games, start-dtm cap)
    (0.50, 1.00, 15000, 5),
    (0.40, 0.70, 15000, 9),
    (0.30, 0.50, 15000, 13),
    (0.25, 0.30, 15000, 19),
    (0.20, 0.15, 15000, 27),
    (0.20, 0.05, 15000, None),
    (0.15, 0.00, 15000, None),
    (0.15, 0.00, 15000, None),
]

if os.path.exists(CKPT):
    with open(CKPT, "rb") as f: st = pickle.load(f)
    rows_all, cols_all, k0, scores = st["rows"], st["cols"], st["round"], st["scores"]
    print(f"resumed at round {k0} with {len(rows_all)} transitions")
else:
    rows_all, cols_all, k0, scores = [], [], 0, np.zeros(uc.n)
    print("\nround | b-eps | dtm-cap | data mates | vs-OPT mate | mv/opt | rook-lost | via-KRK | WIN/DRAW AUC")

for k in range(k0, len(SCHEDULE)):
    ew, eb, ng, dcap = SCHEDULE[k]
    pol = greedy_pol(scores)
    rows, cols, nm = sample_round(pol, ew, eb, ng, seed=100+k, dtm_cap=dcap)
    rows_all += rows; cols_all += cols
    F, Bm, scores = estimate(rows_all, cols_all)
    rate, ratio, rl, vk, a_wd = eval_all(scores)
    print(f"  {k}   | {eb:.2f}  |  {str(dcap):>4s}   |  {nm:6d}   |    {rate:.3f}    |  {ratio:.2f}  |   {rl:.3f}   |  {vk:.2f}   |   {a_wd:.3f}   ({time.time()-t0:.0f}s)")
    with open(CKPT, "wb") as f:
        pickle.dump(dict(rows=rows_all, cols=cols_all, round=k+1, scores=scores), f)
    np.save("krkn_scores.npy", scores); np.save("krkn_F.npy", F); np.save("krkn_B.npy", Bm)

print(f"training done ({time.time()-t0:.0f}s)")
