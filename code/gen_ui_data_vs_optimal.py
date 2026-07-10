"""
gen_ui_data_vs_optimal.py — learned engine vs optimal opponent.
Compare to random engine for baseline.
"""
import json, numpy as np
from domain import compute_dtm, concept_features, white_moves, black_moves, classify_b, MATE, STALEMATE, rc, sq
from learn import Chain, randomized_svd_sm, fb_from_svd

GAMMA = 0.92
rng = np.random.default_rng(4)

ch = Chain()
dtm_w, dtm_b = compute_dtm(ch.W, ch.B)
feats = concept_features(ch.W, dtm_w)
region = np.array(list(np.where(dtm_w <= 3)[0]) + [ch.MATE_S])

# learned model
tr = ch.sample_games(32000, seed=11)
Phat, visited = ch.empirical_P(tr)
U, S, V = randomized_svd_sm(Phat, GAMMA, d=64, seed=0)
F, Bm = fb_from_svd(U, S, V)
zG = Bm[region].sum(axis=0)

# VQ tokens
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
    """Learned score per white move."""
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

def play_game(start, engine_policy, cap=60, seed=0):
    """engine_policy(s, r) returns white move index."""
    r = np.random.default_rng(seed)
    s = int(start)
    plies = []
    result = "unfinished"
    for ply in range(cap):
        st = ch.W[s]
        scored = score_moves(s)  # always compute for the learned scores (even if engine doesn't use them)
        best_v, best_mi_scored, _ = scored[0]
        
        # get the move the engine actually chooses
        best_mi = engine_policy(s, r)
        
        # recover bnodes
        bnodes = white_moves(*st)
        cands = []
        for v, mi, hm in scored[:6]:
            cands.append(dict(name=ch.move_names[s][mi], score=float(v),
                              fromTo=mv_fromto(st, bnodes[mi]), mates=bool(hm)))
        
        # step: white move
        outcomes = ch.moves[s][best_mi]
        nxt = int(outcomes[r.integers(0, len(outcomes))])
        chosen_ft = mv_fromto(st, bnodes[best_mi])
        
        # black reply: OPTIMAL DEFENSE (maximize DTM; capture rook = draw = best)
        black_ft = None
        if nxt < ch.nW:
            bnode = bnodes[best_mi]
            black_replies = black_moves(*bnode)
            if black_replies:
                best_black_i = 0
                best_bv = -np.inf
                for bi, (nxt_try, captured) in enumerate(black_replies):
                    if captured:
                        best_black_i = bi; best_bv = np.inf; break  # draw: best outcome for black
                    if nxt_try in ch.Wi:
                        v_try = float(dtm_w[ch.Wi[nxt_try]]) if np.isfinite(dtm_w[ch.Wi[nxt_try]]) else 1e6
                        if v_try > best_bv:
                            best_bv, best_black_i = v_try, bi
                nxt_actual, captured = black_replies[best_black_i]
                if captured:
                    nxt = ch.DRAW_S
                elif nxt_actual in ch.Wi:
                    nxt = ch.Wi[nxt_actual]
                else:
                    nxt = ch.DRAW_S
                bk_before = bnode[2]
                bk_after = ch.W[nxt][2] if nxt < ch.nW else bnode[2]
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

def learned_policy(s, r):
    """Greedy cone-steering."""
    best_v = -np.inf
    best_mi = 0
    for mi, outcomes in enumerate(ch.moves[s]):
        v = 0.0
        for o in outcomes:
            o = int(o)
            if o == ch.MATE_S: v += float(F[o] @ zG)
            elif o == ch.DRAW_S: v += 0.0
            else: v += float(F[o] @ zG) if visited[o] else 0.0
        v /= len(outcomes)
        if v > best_v: best_v, best_mi = v, mi
    return best_mi

def random_policy(s, r):
    """Uniform random white move."""
    return int(r.integers(0, len(ch.moves[s])))

def optimal_policy(s, r):
    """Proper minimax with ground-truth DTM: black MAXIMIZES delay (capture=draw
    is black's best), white picks the move minimizing black's best reply value."""
    best_v = np.inf
    best_mi = 0
    bnodes = white_moves(*ch.W[s])
    for mi, bnode in enumerate(bnodes):
        cls = classify_b(*bnode)
        if cls == MATE: return mi                 # immediate mate
        if cls == STALEMATE: continue             # never stalemate on purpose
        worst = -np.inf                           # black's best (max-DTM or draw)
        for nxt, captured in black_moves(*bnode):
            if captured: worst = 1e6; break       # rook hangs: black draws
            if nxt in ch.Wi:
                v = float(dtm_w[ch.Wi[nxt]]) if np.isfinite(dtm_w[ch.Wi[nxt]]) else 1e6
                worst = max(worst, v)
        if worst < best_v:
            best_v, best_mi = worst, mi
    return best_mi

print("generating games: learned vs optimal opponent")
games = []
for lo, hi, seed in ((15, 20, 1), (13, 20, 2), (11, 14, 3), (9, 12, 4), (7, 10, 5), (16, 20, 6)):
    cand = np.where((dtm_w >= lo) & (dtm_w <= hi))[0]
    start = int(cand[rng.integers(0, len(cand))])
    g = play_game(start, learned_policy, seed=seed)
    g["startDtm"] = float(dtm_w[start])
    g["engine"] = "learned"
    games.append(g)
    print(f"  learned: start DTM {dtm_w[start]:.0f} -> {g['result']} in {len(g['plies'])} plies")

print("generating games: random vs optimal opponent (baseline)")
for lo, hi, seed in ((15, 20, 7), (13, 20, 8), (11, 14, 9), (9, 12, 10), (7, 10, 11), (16, 20, 12)):
    cand = np.where((dtm_w >= lo) & (dtm_w <= hi))[0]
    start = int(cand[rng.integers(0, len(cand))])
    g = play_game(start, random_policy, seed=seed)
    g["startDtm"] = float(dtm_w[start])
    g["engine"] = "random"
    games.append(g)
    print(f"  random:  start DTM {dtm_w[start]:.0f} -> {g['result']} in {len(g['plies'])} plies")

data = dict(N=5, gamma=GAMMA, K=K, tokens=token_legend, games=games,
            meta=dict(trainedGames=32000, rank=64,
                      regionDef="DTM<=3 ∪ {mate}",
                      opponent="optimal (minimax with ground-truth DTM)",
                      engine="learned cone-steering vs random vs optimal"))
with open("ui_data_vs_optimal.json", "w") as f:
    json.dump(data, f)
print("wrote ui_data_vs_optimal.json")
