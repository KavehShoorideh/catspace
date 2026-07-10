"""
gen_krkn_viewer.py — data for the interactive linked viewer.

Per start (several, varied DTM): planner game AND oracle game vs optimal
defense. Per node: pieces (KRkn or KRk stratum), DTM, fork flag, all white
moves with learned minimax scores (played flagged), black reply, t-SNE xy,
and a cone spray (subsampled MC futures under black eps=0.25).
Plus a downsampled background cloud (won/drawn/KRk classes).
"""
import numpy as np, json, pickle, time
import domain as K1
from krkn import KRKNChain, KN_ATT, black_moves, white_moves, rc, bk_in_check as bk_chk2
t0 = time.time()

uc = KRKNChain(verbose=False)
dtm = np.load("dtm_krkn.npy"); won = np.isfinite(dtm[:uc.n2])
scores = np.load("krkn_scores.npy")
F = np.load("krkn_F.npy")
mk = uc.move_kind; mp0 = uc.move_ptr[:-1]; op0 = uc.out_ptr[:-1]
out_counts = np.diff(uc.out_ptr)

dtm_full = np.full(uc.n, 1e6); dtm_full[:uc.nW] = np.where(np.isfinite(dtm), dtm, 1e6)
vf = dtm_full[uc.out_flat]
sm_ = np.maximum.reduceat(vf, op0)
B_opt = (np.minimum.reduceat(np.where(vf == np.repeat(sm_, out_counts),
        np.arange(len(vf)), len(vf)), op0) - op0).astype(np.int32)

Fw = F[:uc.nW].astype(np.float32)
Fn = (Fw - Fw.mean(0)) / (Fw.std(0) + 1e-9)
with open("tsne_cache.pkl", "rb") as f:
    emb, fit_idx = pickle.load(f)
P = np.asarray(emb)
print(f"setup ({time.time()-t0:.0f}s)")

def sq_name(x): r, c = rc(x); return f"{'abcde'[c]}{r+1}"

def state_pieces(s):
    if s < uc.n2:
        wk, wr, bk, bn = uc.W[s]
        return dict(wk=wk, wr=wr, bk=bk, bn=bn, stratum="KRkn")
    wk, wr, bk = uc.W1[s - uc.n2]
    return dict(wk=wk, wr=wr, bk=bk, bn=None, stratum="KRk")

def move_list(s):
    """All white moves at union state s with fromTo, in flatten order.
    Names carry chess suffixes: '#' mate, '+' check."""
    a = mp0[s]
    out = []
    if s < uc.n2:
        wk, wr, bk, bn = uc.W[s]
        for j, (kind, pay) in enumerate(white_moves(wk, wr, bk, bn)):
            mid = a + j
            if kind == 'xN':
                wk2, wr2, _ = pay
                ft = [wk, bn] if wk2 != wk else [wr, bn]
                nm = ("Kx" if wk2 != wk else "Rx") + sq_name(bn)
                chk = K1.black_in_check(*pay)          # KRk bnode: rook check on bk
            else:
                wk2, wr2, _, _ = pay
                ft = [wk, wk2] if wk2 != wk else [wr, wr2]
                nm = ("K" if wk2 != wk else "R") + sq_name(ft[1])
                chk = bk_chk2(*pay)
            if mk[mid] == 1: nm += "#"
            elif chk: nm += "+"
            out.append((mid, nm, ft))
        return out
    wk, wr, bk = uc.W1[s - uc.n2]
    for j, bn_ in enumerate(K1.white_moves(wk, wr, bk)):
        mid = a + j
        wk2, wr2, _ = bn_
        ft = [wk, wk2] if wk2 != wk else [wr, wr2]
        nm = ("K" if wk2 != wk else "R") + sq_name(ft[1])
        if mk[mid] == 1: nm += "#"
        elif K1.black_in_check(*bn_): nm += "+"
        out.append((mid, nm, ft))
    return out

def minimax_score(mid):
    k = mk[mid]
    if k == 1: return 1e9, True
    if k in (2, 3): return -1e9, False
    return float(np.min(scores[uc.outs_of(mid)])), False

def planner_pick(s):
    best, bv = None, -np.inf
    for mid, nm, ft in move_list(s):
        v, mate = minimax_score(mid)
        if mate: return mid
        if v > bv: bv, best = v, mid
    return best

def oracle_pick(s):
    best, bv = None, np.inf
    for mid, nm, ft in move_list(s):
        k = mk[mid]
        if k == 1: return mid
        if k in (2, 3): continue
        w = dtm_full[int(uc.out_flat[op0[mid] + B_opt[mid]])]
        if w < bv: bv, best = w, mid
    return best if best is not None else mp0[s]

def fork_threat(s):
    if s >= uc.n2: return False
    wk, wr, bk, bn = uc.W[s]
    for kind, pay in black_moves(wk, wr, bk, bn):
        if kind == 'm' and wk in KN_ATT[pay[3]] and wr in KN_ATT[pay[3]]: return True
    return False

def cone_spray(s0, picker, n_roll=120, horizon=20, eps=0.25, seed=7, keep=90):
    r = np.random.default_rng(seed)
    got = []
    for _ in range(n_roll):
        s = int(s0)
        for t_ in range(horizon):
            mid = picker(s); k = mk[mid]
            if k != 0: break
            outs = uc.outs_of(mid)
            bi = int(B_opt[mid]) if r.random() > eps else int(r.integers(0, len(outs)))
            nxt = int(outs[bi])
            if nxt >= uc.nW: break
            got.append((nxt, t_ + 1)); s = nxt
    if len(got) > keep:
        got = [got[i] for i in np.random.default_rng(1).choice(len(got), keep, replace=False)]
    return got

def play_record(start, picker, cap=40, seed=0):
    r = np.random.default_rng(seed)
    s = int(start); nodes = []; result = "unfinished"
    for _ in range(cap):
        mvs = move_list(s)
        mid = picker(s)
        cands = []
        for m2, nm, ft in mvs:
            v, mate = minimax_score(m2)
            cands.append(dict(name=nm, score=(None if abs(v) > 1e8 else round(v, 4)),
                              mates=bool(mk[m2] == 1), fromTo=ft, played=(m2 == mid)))
        cands.sort(key=lambda c: (not c["mates"], -(c["score"] if c["score"] is not None else -1e9)))
        cands = [c for c in cands if c["played"] or True][:8] + \
                ([c for c in cands[8:] if c["played"]])
        chosen = next(c for c in cands if c["played"])
        k = mk[mid]
        node = dict(**state_pieces(s), dtm=(None if not np.isfinite(dtm_full[s]) or dtm_full[s] > 1e5 else int(dtm_full[s])),
                    fork=bool(fork_threat(s)), move=chosen["name"], fromTo=chosen["fromTo"],
                    cands=cands, sIdx=int(s))
        black_ft, black_nm = None, None
        if k == 1: result = "mate"; nxt = uc.MATE_S
        elif k in (2, 3): result = "draw"; nxt = uc.DRAW_S
        else:
            outs = uc.outs_of(mid)
            nxt = int(outs[B_opt[mid]])
            if nxt == uc.DRAW_S: result = "draw (rook lost)"
            elif nxt < uc.nW:
                b4, af = state_pieces(s), state_pieces(nxt)
                if s < uc.n2 and nxt < uc.n2:
                    if af["bk"] != b4["bk"]:
                        black_ft = [b4["bk"], af["bk"]]; black_nm = "K" + sq_name(af["bk"])
                    elif af["bn"] != b4["bn"]:
                        black_ft = [b4["bn"], af["bn"]]; black_nm = "N" + sq_name(af["bn"])
                    # does black's reply give check to the white king?
                    if black_nm:
                        wk2, wr2, bk2, bn2 = uc.W[nxt]
                        if wk2 in KN_ATT[bn2]: black_nm += "+"
                else:
                    if af["bk"] != b4["bk"]:
                        black_ft = [b4["bk"], af["bk"]]; black_nm = "K" + sq_name(af["bk"])
        node["blackReply"] = black_ft
        node["blackMove"] = black_nm
        # ---- alternatives for the fan/toggle visualization ----
        # white alternatives: destination under modeled (optimal) black
        walts = []
        for m2, nm2, ft2 in mvs:
            if m2 == mid: continue
            v2, _ = minimax_score(m2)
            k2 = mk[m2]
            if k2 == 1: walts.append(dict(name=nm2, score=None, term="mate"))
            elif k2 in (2, 3): walts.append(dict(name=nm2, score=None, term="draw"))
            else:
                nxt2 = int(uc.out_flat[op0[m2] + B_opt[m2]])
                if nxt2 >= uc.nW:
                    walts.append(dict(name=nm2, score=round(v2, 3), term="draw"))
                else:
                    walts.append(dict(name=nm2, score=round(v2, 3), sIdx=int(nxt2)))
        walts.sort(key=lambda a: -(a["score"] if a["score"] is not None else 1e9))
        node["whiteAlts"] = walts[:7]
        # black alternatives from the played move's B-node
        balts = []
        if k == 0:
            # reconstruct the B-node pieces by applying the played move
            b4 = state_pieces(s)
            bp = dict(b4)
            frm, to = chosen["fromTo"]
            if frm == b4["wk"]: bp["wk"] = to
            else: bp["wr"] = to
            if b4.get("bn") is not None and to == b4["bn"]: bp["bn"] = None   # xN capture
            outs = uc.outs_of(mid)
            for bi in range(len(outs)):
                nxt2 = int(outs[bi])
                if nxt2 == uc.DRAW_S:
                    balts.append(dict(name="xR", score=None, term="draw",
                                      optimal=bool(bi == int(B_opt[mid]))))
                    continue
                af2 = state_pieces(nxt2)
                if bp.get("bn") is not None and af2.get("bn") is not None and af2["bn"] != bp["bn"]:
                    nm2 = "N" + sq_name(af2["bn"])
                    if nxt2 < uc.n2:
                        wk2, wr2, bk2, bn2 = uc.W[nxt2]
                        if wk2 in KN_ATT[bn2]: nm2 += "+"
                elif af2["bk"] != bp["bk"]:
                    nm2 = "K" + sq_name(af2["bk"])
                else:
                    nm2 = "?"
                balts.append(dict(name=nm2, score=round(float(scores[nxt2]), 3),
                                  sIdx=int(nxt2), optimal=bool(bi == int(B_opt[mid]))))
        node["blackAlts"] = balts
        nodes.append(node)
        if nxt >= uc.nW: break
        s = nxt
    return nodes, result

# ---- pick starts: varied DTM, planner must terminate within cap; include one failure
rng = np.random.default_rng(9)
games = []
bands = [(15, 19), (13, 15), (11, 13), (9, 11), (7, 9)]
for lo, hi in bands:
    pool = np.where(won & (dtm[:uc.n2] >= lo) & (dtm[:uc.n2] < hi))[0]
    for t_ in range(60):
        st = int(pool[rng.integers(0, len(pool))])
        pn, pr = play_record(st, planner_pick)
        if pr == "mate":
            on, orr = play_record(st, oracle_pick)
            games.append(dict(startDtm=int(dtm[st]), planner=dict(nodes=pn, result=pr),
                              oracle=dict(nodes=on, result=orr)))
            break
# one honest failure
pool = np.where(won & (dtm[:uc.n2] >= 13))[0]
for t_ in range(200):
    st = int(pool[rng.integers(0, len(pool))])
    pn, pr = play_record(st, planner_pick)
    if pr != "mate" and len(pn) >= 3:
        on, orr = play_record(st, oracle_pick)
        games.append(dict(startDtm=int(dtm[st]), planner=dict(nodes=pn, result=pr),
                          oracle=dict(nodes=on, result=orr)))
        break
for g in games:
    print(f"  DTM {g['startDtm']}: planner {g['planner']['result']} in {len(g['planner']['nodes'])} | "
          f"oracle {g['oracle']['result']} in {len(g['oracle']['nodes'])}")
print(f"games recorded ({time.time()-t0:.0f}s)")

# ---- cones per node + transforms
need = set()
for g in games:
    for side, picker in (("planner", planner_pick), ("oracle", oracle_pick)):
        for i, nd in enumerate(g[side]["nodes"]):
            spray = cone_spray(nd["sIdx"], picker, seed=7 + i)
            nd["cone"] = [[int(sx), int(tt)] for sx, tt in spray]
            need |= {sx for sx, _ in spray}
            need.add(nd["sIdx"])
            for a in nd.get("whiteAlts", []):
                if "sIdx" in a: need.add(a["sIdx"])
            for a in nd.get("blackAlts", []):
                if "sIdx" in a: need.add(a["sIdx"])
need = sorted(need)
pos = {s: i for i, s in enumerate(need)}
E = np.asarray(emb.transform(Fn[np.array(need)]))
E = np.round(E, 2)
print(f"transformed {len(need)} points ({time.time()-t0:.0f}s)")

for g in games:
    for side in ("planner", "oracle"):
        for nd in g[side]["nodes"]:
            nd["xy"] = [float(E[pos[nd["sIdx"]], 0]), float(E[pos[nd["sIdx"]], 1])]
            nd["cone"] = [[float(E[pos[sx], 0]), float(E[pos[sx], 1]), tt] for sx, tt in nd["cone"]]
            for a in nd.get("whiteAlts", []):
                if "sIdx" in a:
                    a["xy"] = [float(E[pos[a["sIdx"]], 0]), float(E[pos[a["sIdx"]], 1])]
                    del a["sIdx"]
            bx = []
            for a in nd.get("blackAlts", []):
                if "sIdx" in a:
                    a["xy"] = [float(E[pos[a["sIdx"]], 0]), float(E[pos[a["sIdx"]], 1])]
                    bx.append(a["xy"]); del a["sIdx"]
            # opponent node = centroid of black's possible replies ("we don't
            # know what black plays" is a point-cloud mean, not a state)
            if bx:
                nd["bnodeXY"] = [round(float(np.mean([p[0] for p in bx])), 2),
                                 round(float(np.mean([p[1] for p in bx])), 2)]
            del nd["sIdx"]

# ---- background cloud (downsampled, classed)
r0 = np.random.default_rng(0)
sub = r0.choice(len(fit_idx), 6000, replace=False)
bidx = fit_idx[sub]
cls = np.where(bidx >= uc.n2, 2, np.where(won[np.clip(bidx, 0, uc.n2-1)], 0, 1))
bg = [[float(round(P[i, 0], 2)), float(round(P[i, 1], 2)), int(c)] for i, c in zip(sub, cls)]

data = dict(N=5, games=games, bg=bg,
            meta=dict(domain="KRkn on 5x5", opponent="optimal defense (max-DTM, capture-aware)",
                      planner="learned cone, 1-ply minimax readout",
                      oracle="DTM-perfect play (tablebase)",
                      cone="MC futures, black eps=0.25, colored by ply depth"))
conv = lambda o: o.item() if hasattr(o, "item") else str(o)
with open("krkn_viewer_data.json", "w") as f:
    json.dump(data, f, default=conv)
print(f"wrote krkn_viewer_data.json ({len(json.dumps(data, default=conv))//1024} KB, {time.time()-t0:.0f}s)")
