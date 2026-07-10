"""
gen_ui_data_pi.py — corrected viewer data: final policy-iteration engine vs
TRUE optimal black, plus random-engine baselines, with VQ tokens recomputed
from the final opponent-conditioned cone.
"""
import json, numpy as np, scipy.sparse as sp
from domain import (compute_dtm, concept_features, white_moves, black_moves,
                    classify_b, MATE, STALEMATE)
from learn import Chain, randomized_svd_sm, fb_from_svd

GAMMA = 0.92
rng = np.random.default_rng(4)
ch = Chain()
dtm_w, dtm_b = compute_dtm(ch.W, ch.B)
feats = concept_features(ch.W, dtm_w)
region = np.array(list(np.where(dtm_w <= 3)[0]) + [ch.MATE_S])

# ---- rebuild the PI run compactly (same schedule/seeds as exp_policy_iteration)
W_move_out, W_move_kind, B_opt_reply = [], [], []
for si in range(ch.nW):
    outs_per_move, kinds, opts = [], [], []
    for bnode in white_moves(*ch.W[si]):
        cls = classify_b(*bnode)
        if cls == MATE: kinds.append(1); outs_per_move.append(np.array([ch.MATE_S])); opts.append(0)
        elif cls == STALEMATE: kinds.append(2); outs_per_move.append(np.array([ch.DRAW_S])); opts.append(0)
        else:
            reps = black_moves(*bnode); nxts, bi_, bv_ = [], 0, -np.inf
            for i, (nxt, cap) in enumerate(reps):
                if cap:
                    nxts.append(ch.DRAW_S)
                    if bv_ < 1e6: bv_, bi_ = 1e6, i
                else:
                    wi = ch.Wi[nxt]; nxts.append(wi)
                    v = dtm_w[wi] if np.isfinite(dtm_w[wi]) else 1e6
                    if v > bv_: bv_, bi_ = v, i
            kinds.append(0); outs_per_move.append(np.array(nxts)); opts.append(bi_)
    W_move_out.append(outs_per_move); W_move_kind.append(kinds); B_opt_reply.append(opts)

def greedy_pol(scores):
    pol = np.zeros(ch.nW, dtype=np.int32)
    for s in range(ch.nW):
        best, bv = 0, -np.inf
        for m, outs in enumerate(W_move_out[s]):
            k = W_move_kind[s][m]
            if k == 1: best = m; bv = np.inf; break
            v = -1e9 if k == 2 else float(np.where(outs == ch.DRAW_S, 0.0,
                    scores[np.minimum(outs, ch.nW-1)]).mean())
            if v > bv: bv, best = v, m
        pol[s] = best
    return pol

def sample_round(pol_w, eps_w, eps_b, ng, seed):
    r = np.random.default_rng(seed); rows, cols = [], []
    for s0 in r.integers(0, ch.nW, size=ng):
        s = int(s0)
        for _ in range(120):
            m = int(pol_w[s]) if r.random() > eps_w else int(r.integers(0, len(W_move_out[s])))
            k = W_move_kind[s][m]; outs = W_move_out[s][m]
            if k == 1: nxt = ch.MATE_S
            elif k == 2: nxt = ch.DRAW_S
            else:
                bi = B_opt_reply[s][m] if r.random() > eps_b else int(r.integers(0, len(outs)))
                nxt = int(outs[bi])
            rows.append(s); cols.append(nxt)
            if nxt >= ch.nW: break
            s = nxt
    return rows, cols

def estimate(all_rows, all_cols, d=64):
    counts = sp.coo_matrix((np.ones(len(all_rows)), (all_rows, all_cols)), shape=(ch.n, ch.n)).tocsr()
    rowsum = np.asarray(counts.sum(1)).ravel(); seen = rowsum > 0; rowsum[rowsum == 0] = 1
    P = (sp.diags(1/rowsum) @ counts).tolil()
    for i in np.where(~seen)[0]: P[i, i] = 1.0
    for a in (ch.MATE_S, ch.DRAW_S): P[a, :] = 0; P[a, a] = 1.0
    U, S, V = randomized_svd_sm(P.tocsr(), GAMMA, d=d, seed=0)
    return fb_from_svd(U, S, V)

all_rows, all_cols, scores = [], [], np.zeros(ch.nW)
for k, (ew, eb, ng) in enumerate([(1,1,20000),(0.3,0.5,20000),(0.3,0.25,20000),
                                   (0.2,0.1,20000),(0.2,0,20000),(0.2,0,20000)]):
    rows, cols = sample_round(greedy_pol(scores), ew, eb, ng, seed=100+k)
    all_rows += rows; all_cols += cols
    F, Bm = estimate(all_rows, all_cols)
    scores = (F @ Bm[region].sum(0))[:ch.nW]
print("PI rebuilt")

Fn = F[:ch.nW]
zG = Bm[region].sum(0)

# ---- VQ tokens on the final opponent-conditioned cone
def kmeans(X, K, iters=50, seed=5):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        a = ((X[:, None, :] - C[None, :, :])**2).sum(-1).argmin(1)
        for k in range(K):
            m = a == k
            if m.any(): C[k] = X[m].mean(0)
    return C, a
Xf = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-9)
K = 32
_, assign = kmeans(Xf[:, :16], K)
token_legend = []
for k in range(K):
    m = assign == k
    token_legend.append(dict(id=k, size=int(m.sum()),
        meanDtm=float(feats['dtm'][m].mean()) if m.any() else 12.0,
        meanBox=float(feats['box_area'][m].mean()) if m.any() else 12.0))

def mv_fromto(s, bnode):
    (wk, wr, bk), (wk2, wr2, _) = s, bnode
    return ([wk, wk2] if wk2 != wk else [wr, wr2])

def move_scores(si):
    out = []
    for m, outs in enumerate(W_move_out[si]):
        k = W_move_kind[si][m]
        if k == 1: v, mates = float(scores.max()*1.5), True
        elif k == 2: v, mates = -1.0, False
        else:
            v = float(np.where(outs == ch.DRAW_S, 0.0, scores[np.minimum(outs, ch.nW-1)]).mean())
            mates = False
        out.append((v, m, mates))
    out.sort(reverse=True)
    return out

def play_game(start, engine, cap=40, seed=0):
    r = np.random.default_rng(seed)
    s = int(start); plies = []; result = "unfinished"
    for _ in range(cap):
        st = ch.W[s]
        bnodes = white_moves(*st)
        scored = move_scores(s)
        if engine == "learned":
            m = scored[0][1]
        else:
            m = int(r.integers(0, len(bnodes)))
        cands = [dict(name=ch.move_names[s][mi], score=float(v),
                      fromTo=mv_fromto(st, bnodes[mi]), mates=bool(mt),
                      played=(mi == m))
                 for v, mi, mt in scored[:6]]
        if not any(c["played"] for c in cands):          # played move outside top-6
            for v, mi, mt in scored:
                if mi == m:
                    cands.append(dict(name=ch.move_names[s][mi], score=float(v),
                                      fromTo=mv_fromto(st, bnodes[mi]),
                                      mates=bool(mt), played=True))
                    break
        k = W_move_kind[s][m]; outs = W_move_out[s][m]
        chosen_ft = mv_fromto(st, bnodes[m])
        black_ft, nxt = None, None
        if k == 1: nxt = ch.MATE_S
        elif k == 2: nxt = ch.DRAW_S
        else:
            bi = B_opt_reply[s][m]                      # TRUE optimal defense
            nxt = int(outs[bi])
            if nxt < ch.nW:
                black_ft = [bnodes[m][2], ch.W[nxt][2]]
        plies.append(dict(wk=st[0], wr=st[1], bk=st[2],
            move=ch.move_names[s][m], fromTo=chosen_ft, blackReply=black_ft,
            token=int(assign[s]), reach=float(scores[s]),
            concepts=dict(dtm=float(min(dtm_w[s], 60)), box=float(feats['box_area'][s]),
                          kk=float(feats['kk_dist'][s]), rookbk=float(feats['rook_bk_dist'][s]),
                          bkedge=float(feats['bk_edge'][s])),
            candidates=cands))
        if nxt == ch.MATE_S: result = "mate"; break
        if nxt == ch.DRAW_S: result = "draw"; break
        s = nxt
    return dict(result=result, plies=plies)

games = []
for lo, hi, seed in ((17, 20, 1), (15, 17, 2), (13, 15, 3), (11, 13, 4), (9, 11, 5), (17, 20, 6)):
    cand = np.where((dtm_w >= lo) & (dtm_w <= hi))[0]
    start = int(cand[rng.integers(0, len(cand))])
    g = play_game(start, "learned", seed=seed)
    g["startDtm"] = float(dtm_w[start]); g["engine"] = "learned"
    games.append(g)
    print(f"  PI-engine vs OPTIMAL: start DTM {dtm_w[start]:.0f} -> {g['result']} in {len(g['plies'])} white moves")
for lo, hi, seed in ((17, 20, 7), (13, 15, 8), (9, 11, 9)):
    cand = np.where((dtm_w >= lo) & (dtm_w <= hi))[0]
    start = int(cand[rng.integers(0, len(cand))])
    g = play_game(start, "random", seed=seed)
    g["startDtm"] = float(dtm_w[start]); g["engine"] = "random"
    games.append(g)
    print(f"  random    vs OPTIMAL: start DTM {dtm_w[start]:.0f} -> {g['result']} in {len(g['plies'])} white moves")

data = dict(N=5, gamma=GAMMA, K=K, tokens=token_legend, games=games,
            meta=dict(trainedGames=120000, rank=64, regionDef="DTM<=3 ∪ {mate}",
                      opponent="TRUE optimal defense (max-DTM black, captures hanging rooks)",
                      engine="opponent-conditioned cone via policy iteration (6 rounds)"))
json.dump(data, open("ui_data_pi.json", "w"))
print("wrote ui_data_pi.json")
