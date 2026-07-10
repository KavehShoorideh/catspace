"""
exp_policy_iteration.py — can cone-steering beat TRUE optimal defense if the
cone is estimated under the right dynamics?

Finding that motivated this: a cone estimated under RANDOM black supports
95.5% mating vs random black but 0% vs optimal black (distribution shift —
the successor measure is opponent-conditioned).

Fix under test: approximate policy iteration with an opponent curriculum.
  round k: white = eps-greedy on current reach scores, black = eps_b-optimal
           (eps_b annealed 1.0 -> 0.0); sample games; accumulate transitions;
           re-estimate successor measure; SVD -> new F,B -> new scores.
Evaluate every round: greedy engine vs TRUE optimal black, 300 starts, and
report mean white-moves-to-mate vs the DTM optimum.

Everything precomputed per round (stationary policies -> index chasing), so
each round is seconds.
"""
import numpy as np, scipy.sparse as sp, time
from domain import (compute_dtm, concept_features, white_moves, black_moves,
                    classify_b, MATE, STALEMATE)
from learn import Chain, randomized_svd_sm, fb_from_svd

GAMMA = 0.92
rng = np.random.default_rng(0)
t0 = time.time()

ch = Chain()
dtm_w, dtm_b = compute_dtm(ch.W, ch.B)
region = np.array(list(np.where(dtm_w <= 3)[0]) + [ch.MATE_S])

# ---------- precompute: for each W state, per white move -> (kind, payload) ----------
# kind: 0 ongoing (payload = array of black-reply next W-state indices and a
#       parallel "optimal reply" index), 1 mate, 2 stalemate
W_move_out = []       # W_move_out[s][m] = np.array of next chain-state per black reply (DRAW_S for capture)
W_move_kind = []      # 0/1/2
B_opt_reply = []      # per (s,m): index into replies chosen by optimal black
for si in range(ch.nW):
    outs_per_move, kinds, opts = [], [], []
    for bnode in white_moves(*ch.W[si]):
        cls = classify_b(*bnode)
        if cls == MATE:
            kinds.append(1); outs_per_move.append(np.array([ch.MATE_S])); opts.append(0)
        elif cls == STALEMATE:
            kinds.append(2); outs_per_move.append(np.array([ch.DRAW_S])); opts.append(0)
        else:
            reps = black_moves(*bnode)
            nxts, best_i, best_v = [], 0, -np.inf
            for i, (nxt, cap) in enumerate(reps):
                if cap:
                    nxts.append(ch.DRAW_S)
                    if best_v < 1e6: best_v, best_i = 1e6, i     # capture = draw = black's best
                else:
                    wi = ch.Wi[nxt]; nxts.append(wi)
                    v = dtm_w[wi] if np.isfinite(dtm_w[wi]) else 1e6
                    if v > best_v: best_v, best_i = v, i
            kinds.append(0); outs_per_move.append(np.array(nxts)); opts.append(best_i)
    W_move_out.append(outs_per_move); W_move_kind.append(kinds); B_opt_reply.append(opts)
print(f"precompute done ({time.time()-t0:.0f}s)")

def greedy_white_policy(scores):
    """scores: per-W-state reach score vector (len nW). Returns per-state best move
    under expectation over RANDOM black (consistent with the cone's own dynamics),
    with immediate mates preferred and stalemates avoided."""
    pol = np.zeros(ch.nW, dtype=np.int32)
    for s in range(ch.nW):
        best, bv = 0, -np.inf
        for m, outs in enumerate(W_move_out[s]):
            k = W_move_kind[s][m]
            if k == 1: best = m; bv = np.inf; break
            if k == 2: v = -1e9
            else:
                vals = np.where(outs == ch.DRAW_S, 0.0, scores[np.minimum(outs, ch.nW - 1)])
                vals = np.where(outs == ch.DRAW_S, 0.0, vals)
                v = float(vals.mean())
            if v > bv: bv, best = v, m
        pol[s] = best
    return pol

def sample_round(pol_w, eps_w, eps_b, n_games, seed):
    """White: eps_w-greedy on pol_w. Black: optimal w.p. 1-eps_b else uniform."""
    r = np.random.default_rng(seed)
    rows, cols = [], []
    starts = r.integers(0, ch.nW, size=n_games)
    n_mates = 0
    for g in range(n_games):
        s = int(starts[g])
        for _ in range(120):
            m = int(pol_w[s]) if r.random() > eps_w else int(r.integers(0, len(W_move_out[s])))
            k = W_move_kind[s][m]
            outs = W_move_out[s][m]
            if k == 1: nxt = ch.MATE_S
            elif k == 2: nxt = ch.DRAW_S
            else:
                bi = B_opt_reply[s][m] if r.random() > eps_b else int(r.integers(0, len(outs)))
                nxt = int(outs[bi])
            rows.append(s); cols.append(nxt)
            if nxt >= ch.nW:
                if nxt == ch.MATE_S: n_mates += 1
                break
            s = nxt
    return rows, cols, n_mates

def estimate_cone(all_rows, all_cols, d=64):
    counts = sp.coo_matrix((np.ones(len(all_rows)), (all_rows, all_cols)),
                           shape=(ch.n, ch.n)).tocsr()
    rowsum = np.asarray(counts.sum(1)).ravel(); seen = rowsum > 0
    rowsum[rowsum == 0] = 1
    P = (sp.diags(1/rowsum) @ counts).tolil()
    for i in np.where(~seen)[0]: P[i, i] = 1.0
    for a in (ch.MATE_S, ch.DRAW_S): P[a, :] = 0; P[a, a] = 1.0
    U, S, V = randomized_svd_sm(P.tocsr(), GAMMA, d=d, seed=0)
    F, Bm = fb_from_svd(U, S, V)
    return (F @ Bm[region].sum(0))[:ch.nW], seen

def eval_vs_optimal(scores, n_eval=300, cap=60, seed=99):
    """Greedy engine on `scores` vs TRUE optimal black. Returns mate rate and
    mean (white moves to mate) / DTM-optimal ratio."""
    r = np.random.default_rng(seed)
    starts = r.integers(0, ch.nW, size=n_eval)
    mates, ratios = 0, []
    for s0 in starts:
        s = int(s0); d0 = dtm_w[s]
        for wm in range(cap):
            best, bv = 0, -np.inf
            for m, outs in enumerate(W_move_out[s]):
                k = W_move_kind[s][m]
                if k == 1: best = m; bv = np.inf; break
                if k == 2: v = -1e9
                else:
                    vals = np.where(outs == ch.DRAW_S, 0.0, scores[np.minimum(outs, ch.nW-1)])
                    v = float(vals.mean())
                if v > bv: bv, best = v, m
            k = W_move_kind[s][best]
            if k == 1:
                mates += 1
                ratios.append((wm + 1) / max(1.0, np.ceil(d0 / 2)))
                break
            if k == 2: break
            outs = W_move_out[s][best]
            nxt = int(outs[B_opt_reply[s][best]])
            if nxt >= ch.nW: break
            s = nxt
    return mates / n_eval, (float(np.mean(ratios)) if ratios else float("nan"))

# ---------- the curriculum ----------
print("\nround | black eps | data mates | vs-OPTIMAL mate-rate | moves/optimal")
all_rows, all_cols = [], []
scores = np.zeros(ch.nW)                     # round 0 white = uniform-ish (greedy on zeros = first move; use eps_w=1)
schedule = [  # (eps_w, eps_b, n_games)
    (1.00, 1.00, 20000),   # pure random vs random  (the original data)
    (0.30, 0.50, 20000),
    (0.30, 0.25, 20000),
    (0.20, 0.10, 20000),
    (0.20, 0.00, 20000),   # black fully optimal
    (0.20, 0.00, 20000),
]
for k, (eps_w, eps_b, ng) in enumerate(schedule):
    pol = greedy_white_policy(scores)
    rows, cols, nm = sample_round(pol, eps_w, eps_b, ng, seed=100 + k)
    all_rows += rows; all_cols += cols
    scores, seen = estimate_cone(all_rows, all_cols)
    rate, ratio = eval_vs_optimal(scores)
    print(f"  {k}   |   {eps_b:.2f}    |   {nm:6d}   |        {rate:.3f}         |   {ratio:.2f}   ({time.time()-t0:.0f}s)")

np.save("/home/claude/toykrk/scores_pi.npy", scores)
print(f"\ndone ({time.time()-t0:.0f}s)")
