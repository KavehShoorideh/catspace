"""
gen_ui_data.py — produce JSON for the interactive viewer.

For each of several games played by the learned engine (32k random-play games),
record per white-ply: position, candidate moves + learned scores, chosen move,
black's reply, plan token, concept values, reach score.
"""
import json, numpy as np
from domain import compute_dtm, concept_features, rc
from learn import Chain, randomized_svd_sm, fb_from_svd

GAMMA = 0.92
rng = np.random.default_rng(4)

ch = Chain()
dtm_w, _ = compute_dtm(ch.W, ch.B)
feats = concept_features(ch.W, dtm_w)
region = np.array(list(np.where(dtm_w <= 3)[0]) + [ch.MATE_S])

# learned model (same recipe as experiment.py, 32k games)
tr = ch.sample_games(32000, seed=11)
Phat, visited = ch.empirical_P(tr)
U, S, V = randomized_svd_sm(Phat, GAMMA, d=64, seed=0)
F, Bm = fb_from_svd(U, S, V)
zG = Bm[region].sum(axis=0)

# VQ tokens (same as experiment.py: k-means on normalized leading F dims)
def kmeans(X, K, iters=50, seed=5):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d2.argmin(1)
        for k in range(K):
            m = a == k
            if m.any(): C[k] = X[m].mean(0)
    return C, a

Fn = F[:ch.nW]
Xf = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-9)
K = 32
C, assign = kmeans(Xf[:, :16], K)

token_legend = []
for k in range(K):
    m = assign == k
    token_legend.append(dict(
        id=k, size=int(m.sum()),
        meanDtm=float(feats['dtm'][m].mean()) if m.any() else None,
        meanBox=float(feats['box_area'][m].mean()) if m.any() else None))

def mv_fromto(s, bnode):
    (wk, wr, bk), (wk2, wr2, _) = s, bnode
    return ([wk, wk2] if wk2 != wk else [wr, wr2])

def score_moves(si):
    """Learned score per white move (mean over black replies), and outcome info."""
    out = []
    for mi, outcomes in enumerate(ch.moves[si]):
        v, has_mate = 0.0, False
        for o in outcomes:
            o = int(o)
            if o == ch.MATE_S: v += float(F[o] @ zG); has_mate = True
            elif o == ch.DRAW_S: v += 0.0
            else: v += float(F[o] @ zG) if visited[o] else 0.0
        out.append((v / len(outcomes), mi, has_mate))
    out.sort(reverse=True)
    return out

def play_game(start, cap=60, seed=0):
    r = np.random.default_rng(seed)
    s = int(start)
    plies = []
    result = "unfinished"
    for ply in range(cap):
        st = ch.W[s]
        scored = score_moves(s)
        best_v, best_mi, _ = scored[0]
        cands = []
        for v, mi, hm in scored[:6]:
            bnode = None
            # recover the bnode for from/to: recompute white_moves order == moves order
            pass
        # recompute white move bnodes in order (same order as ch.moves construction)
        from domain import white_moves
        bnodes = white_moves(*st)
        for v, mi, hm in scored[:6]:
            cands.append(dict(name=ch.move_names[s][mi], score=float(v),
                              fromTo=mv_fromto(st, bnodes[mi]), mates=bool(hm)))
        # step
        outcomes = ch.moves[s][best_mi]
        nxt = int(outcomes[r.integers(0, len(outcomes))])
        chosen_ft = mv_fromto(st, bnodes[best_mi])
        # black reply arrow: bk square before vs after
        black_ft = None
        if nxt < ch.nW:
            bk_before = bnodes[best_mi][2]
            bk_after = ch.W[nxt][2]
            black_ft = [bk_before, bk_after]
        plies.append(dict(
            wk=st[0], wr=st[1], bk=st[2],
            move=ch.move_names[s][best_mi], fromTo=chosen_ft, blackReply=black_ft,
            token=int(assign[s]),
            reach=float(Fn[s] @ zG),
            concepts=dict(dtm=float(min(dtm_w[s], 60)),
                          box=float(feats['box_area'][s]),
                          kk=float(feats['kk_dist'][s]),
                          rookbk=float(feats['rook_bk_dist'][s]),
                          bkedge=float(feats['bk_edge'][s])),
            candidates=cands))
        if nxt == ch.MATE_S: result = "mate"; break
        if nxt == ch.DRAW_S: result = "draw"; break
        s = nxt
    return dict(result=result, plies=plies)

games = []
# varied starting difficulty: DTM bands
for lo, hi, seed in ((15, 20, 1), (13, 20, 2), (11, 14, 3), (9, 12, 4), (7, 10, 5), (16, 20, 6)):
    cand = np.where((dtm_w >= lo) & (dtm_w <= hi))[0]
    start = int(cand[rng.integers(0, len(cand))])
    g = play_game(start, seed=seed)
    g["startDtm"] = float(dtm_w[start])
    games.append(g)
    print(f"game: start DTM {dtm_w[start]:.0f} -> {g['result']} in {len(g['plies'])} plies")

data = dict(N=5, gamma=GAMMA, K=K, tokens=token_legend, games=games,
            meta=dict(trainedGames=32000, rank=64,
                      regionDef="DTM<=3 ∪ {mate}",
                      engine="greedy cone-steering, learned from random play"))
with open("ui_data.json", "w") as f:
    json.dump(data, f)
print("wrote ui_data.json,", len(json.dumps(data)) // 1024, "KB")
