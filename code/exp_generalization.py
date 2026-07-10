"""
exp_generalization.py — the generalization experiment.

Question: does a neural FB generalize to states it has NEVER seen in any
training pair (familiar concept family, unfamiliar exact positioning)?

Comparison, same filtered experience for both learners:
  - tabular-FB (SVD of empirical successor measure) -> undefined at unseen states
  - neural-FB (contrastive MLP)                     -> generalizes or doesn't

Evaluations AT HELD-OUT STATES ONLY:
  E1 reach-ranking: spearman(F(s)·z_G, exact reach) on holdout vs train states
  E2 engine from holdout starts vs OPTIMAL black (the hard test)
  E3 engine from holdout starts vs random black
"""
import numpy as np, scipy.stats as st, time, json
from domain import compute_dtm, concept_features, white_moves, black_moves
from learn import Chain, randomized_svd_sm, fb_from_svd, sm_matvec
from neural import (NeuralFB, one_hot_state, absorbing_vec, build_pairs,
                    sample_episodes)

GAMMA = 0.92
rng = np.random.default_rng(0)
t0 = time.time()

ch = Chain()
dtm_w, dtm_b = compute_dtm(ch.W, ch.B)
feats = concept_features(ch.W, dtm_w)
region_states = np.where(dtm_w <= 3)[0]

# ---- exact reach ground truth (for evaluation only)
region_full = np.array(list(region_states) + [ch.MATE_S])
e = np.zeros((ch.n, 1)); e[region_full] = 1.0
reach_true = sm_matvec(ch.exact_P_uniform(), e, GAMMA).ravel()

# ---- holdout: 15% of W states never enter training in ANY role
HOLD = 0.15
holdout_mask = rng.random(ch.nW) < HOLD
print(f"holdout states: {holdout_mask.sum()} / {ch.nW}")

# ---- experience: 32k random games, filtered
eps = sample_episodes(ch, 32000, seed=11)
pairs = build_pairs(ch, eps, holdout_mask, GAMMA, rng)
print(f"training pairs (holdout-free): {len(pairs)}")

# ---- tabular baseline on the SAME filtered experience
import scipy.sparse as sp
rows, cols = [], []
for ep in eps:
    for i in range(len(ep) - 1):
        s, nx = ep[i], ep[i + 1]
        if s < ch.nW and not holdout_mask[s] and not (nx < ch.nW and holdout_mask[nx]):
            rows.append(s); cols.append(nx)
counts = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(ch.n, ch.n)).tocsr()
rowsum = np.asarray(counts.sum(1)).ravel(); seen_tab = rowsum > 0
rowsum[rowsum == 0] = 1
Ptab = (sp.diags(1/rowsum) @ counts).tolil()
for i in np.where(~seen_tab)[0]: Ptab[i, i] = 1.0
for a in (ch.MATE_S, ch.DRAW_S): Ptab[a, :] = 0; Ptab[a, a] = 1.0
Ptab = Ptab.tocsr()
U, S, V = randomized_svd_sm(Ptab, GAMMA, d=32, seed=0)
Ftab, Btab = fb_from_svd(U, S, V)
zG_tab = Btab[region_full].sum(0)

# ---- neural FB training
print("training neural FB...")
X_all = np.stack([one_hot_state(s) for s in ch.W]).astype(np.float32)
X_mate = absorbing_vec(0)[None, :]
X_draw = absorbing_vec(1)[None, :]
def gvec(idx):
    if idx == ch.MATE_S: return X_mate[0]
    if idx == ch.DRAW_S: return X_draw[0]
    return X_all[idx]

net = NeuralFB(d=32, dh=256, seed=0, tau=0.1)
pairs = np.array(pairs)
BS, STEPS = 256, 12000
sched = [(0, 1e-3), (8000, 3e-4)]
for step in range(STEPS):
    lr = [l for s0, l in sched if step >= s0][-1]
    idx = rng.integers(0, len(pairs), size=BS)
    Xs = X_all[pairs[idx, 0]]
    Xg = np.stack([gvec(int(g)) for g in pairs[idx, 1]])
    loss = net.train_step(Xs, Xg, lr)
    if step % 2000 == 0:
        print(f"  step {step:6d}  loss {loss:.3f}  ({time.time()-t0:.0f}s)")
print(f"training done ({time.time()-t0:.0f}s)")

Fn = net.embed_F(X_all)                       # all states incl. holdout — net never saw them
Bn_states = net.embed_B(X_all)
Bn_mate = net.embed_B(X_mate)[0]
zG_n = Bn_states[region_states].sum(0) + Bn_mate

# ---- E1: reach ranking at holdout vs train states
r_n = Fn @ zG_n
r_t = Ftab[:ch.nW] @ zG_tab
hi, ti = np.where(holdout_mask)[0], np.where(~holdout_mask)[0]
print("\nE1 reach-ranking spearman vs exact reach:")
print(f"  neural  train={st.spearmanr(r_n[ti], reach_true[ti]).statistic:.3f}  "
      f"HOLDOUT={st.spearmanr(r_n[hi], reach_true[hi]).statistic:.3f}")
print(f"  tabular train={st.spearmanr(r_t[ti], reach_true[ti]).statistic:.3f}  "
      f"HOLDOUT={st.spearmanr(r_t[hi], reach_true[hi]).statistic:.3f}   "
      f"(tabular has no rows for unseen states)")

# ---- engines
def neural_score(si):
    out = []
    for mi, outs in enumerate(ch.moves[si]):
        v = 0.0
        for o in outs:
            o = int(o)
            if o == ch.MATE_S: v += float(Fn_absorb_mate)
            elif o == ch.DRAW_S: v += float(Fn_absorb_draw)
            else: v += float(Fn[o] @ zG_n)
        out.append(v / len(outs))
    return out

# absorbing "scores": mate should be max reach; use F of absorbing? F is over W states.
# For outcomes that are absorbing we score by construction: mate = max observed score, draw = min.
Fn_absorb_mate = float(np.quantile(r_n, 0.999))
Fn_absorb_draw = float(np.quantile(r_n, 0.001))

def tab_score(si):
    out = []
    for mi, outs in enumerate(ch.moves[si]):
        v = 0.0
        for o in outs:
            o = int(o)
            if o == ch.MATE_S: v += float(Ftab[o] @ zG_tab)
            elif o == ch.DRAW_S: v += 0.0
            else: v += float(Ftab[o] @ zG_tab) if seen_tab[o] else 0.0
        out.append(v / len(outs))
    return out

def optimal_black(bnode):
    """True optimal defense: capture the rook if possible (draw = black's best);
    otherwise MAXIMIZE white's distance-to-mate."""
    reps = black_moves(*bnode)
    best_i, best_v = 0, -np.inf
    for i, (nxt, cap) in enumerate(reps):
        if cap: return i
        v = dtm_w[ch.Wi[nxt]] if nxt in ch.Wi and np.isfinite(dtm_w[ch.Wi[nxt]]) else 1e6
        if v > best_v: best_v, best_i = v, i
    return best_i

def play(score_fn, starts, black="optimal", cap=80, seed=5):
    r = np.random.default_rng(seed)
    mates = 0
    for s0 in starts:
        s = int(s0)
        for _ in range(cap):
            sc = score_fn(s)
            mi = int(np.argmax(sc))
            bnodes = white_moves(*ch.W[s])
            bnode = bnodes[mi]
            from domain import classify_b, MATE as MATE_C, STALEMATE
            cls = classify_b(*bnode)
            if cls == MATE_C: mates += 1; break
            if cls == STALEMATE: break
            reps = black_moves(*bnode)
            bi = optimal_black(bnode) if black == "optimal" else int(r.integers(0, len(reps)))
            nxt, cap_r = reps[bi]
            if cap_r: break
            s = ch.Wi[nxt]
        # unfinished counts as non-mate
    return mates / len(starts)

n_eval = 400
hold_starts = rng.choice(hi, size=min(n_eval, len(hi)), replace=False)
print(f"\nE2/E3 engines starting FROM {len(hold_starts)} HELD-OUT states:")
for nm, fn in (("neural ", neural_score), ("tabular", tab_score)):
    mo = play(fn, hold_starts, black="optimal")
    mr = play(fn, hold_starts, black="random")
    print(f"  {nm}: vs OPTIMAL black mate-rate={mo:.3f}   vs random black={mr:.3f}")
mo = play(lambda s: list(np.random.default_rng(1).random(len(ch.moves[s]))), hold_starts, black="optimal")
print(f"  random : vs OPTIMAL black mate-rate={mo:.3f}")

np.save("/home/claude/toykrk/Fn_neural.npy", Fn)
np.save("/home/claude/toykrk/holdout_mask.npy", holdout_mask)
np.save("/home/claude/toykrk/reach_neural.npy", r_n)
json.dump(dict(zG=zG_n.tolist()), open("/home/claude/toykrk/zg_neural.json", "w"))
print(f"\nartifacts saved ({time.time()-t0:.0f}s total)")
