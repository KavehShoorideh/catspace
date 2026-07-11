"""
viz/payload.py — builds the KRkn linked-viewer JSON payload (games, cones,
alt fans, background cloud) on top of the new chain/readout/projection
stack. Ports gen_krkn_viewer.py's move-list/candidate/cone-spray logic
faithfully; the only seam this module depends on for its 2D coordinates is
FittedMap.project(F, idx) -> (m, 2) -- swap the projection kind and every
xy/cone/alt-fan coordinate in the payload follows.
"""
from __future__ import annotations

import numpy as np

from catspace.board import rc
from catspace.domains import krk as K1
from catspace.domains import krkn as K2
from catspace.scoring import dtm_filled


def json_default(o):
    """The numpy-JSON fix, in one place (was a repeated bug source)."""
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def sq_name(x: int) -> str:
    r, c = rc(x)
    return f"{'abcde'[c]}{r + 1}"


class KrknViewerBuilder:
    """Ports gen_krkn_viewer.py's per-node payload logic onto TransitionChain."""

    def __init__(self, chain, dtm, scores, b_opt):
        self.chain = chain
        self.dtm = dtm
        self.scores = scores
        self.b_opt = b_opt
        self.dtm_full = dtm_filled(dtm, chain.n)
        self.n2 = chain.strata["KRkn"].stop

    def state_pieces(self, s: int) -> dict:
        return K2.describe_state(self.chain, s)

    def move_list(self, s: int):
        chain = self.chain
        a = int(chain.move_ptr[s])
        out = []
        if s < self.n2:
            wk, wr, bk, bn = chain.W[s]
            for j, (kind, pay) in enumerate(K2.white_moves(wk, wr, bk, bn)):
                mid = a + j
                if kind == 'xN':
                    wk2, wr2, _ = pay
                    ft = [wk, bn] if wk2 != wk else [wr, bn]
                    nm = ("Kx" if wk2 != wk else "Rx") + sq_name(bn)
                    chk = K1.black_in_check(*pay)
                else:
                    wk2, wr2, _, _ = pay
                    ft = [wk, wk2] if wk2 != wk else [wr, wr2]
                    nm = ("K" if wk2 != wk else "R") + sq_name(ft[1])
                    chk = K2.bk_in_check(*pay)
                if chain.move_kind[mid] == 1: nm += "#"
                elif chk: nm += "+"
                out.append((mid, nm, ft))
            return out
        wk, wr, bk = chain.W1[s - self.n2]
        for j, bn_ in enumerate(K1.white_moves(wk, wr, bk)):
            mid = a + j
            wk2, wr2, _ = bn_
            ft = [wk, wk2] if wk2 != wk else [wr, wr2]
            nm = ("K" if wk2 != wk else "R") + sq_name(ft[1])
            if chain.move_kind[mid] == 1: nm += "#"
            elif K1.black_in_check(*bn_): nm += "+"
            out.append((mid, nm, ft))
        return out

    def minimax_score(self, mid: int):
        k = self.chain.move_kind[mid]
        if k == 1: return 1e9, True
        if k in (2, 3): return -1e9, False
        return float(np.min(self.scores[self.chain.outs_of(mid)])), False

    def planner_pick(self, s: int) -> int:
        best, bv = None, -np.inf
        for mid, nm, ft in self.move_list(s):
            v, mate = self.minimax_score(mid)
            if mate: return mid
            if v > bv: bv, best = v, mid
        return best

    def oracle_pick(self, s: int) -> int:
        chain = self.chain
        best, bv = None, np.inf
        for mid, nm, ft in self.move_list(s):
            k = chain.move_kind[mid]
            if k == 1: return mid
            if k in (2, 3): continue
            w = self.dtm_full[int(chain.out_flat[chain.op0[mid] + self.b_opt[mid]])]
            if w < bv: bv, best = w, mid
        return best if best is not None else int(chain.mp0[s])

    def fork_threat(self, s: int) -> bool:
        if s >= self.n2: return False
        wk, wr, bk, bn = self.chain.W[s]
        for kind, pay in K2.black_moves(wk, wr, bk, bn):
            if kind == 'm' and wk in K2.KN_ATT[pay[3]] and wr in K2.KN_ATT[pay[3]]:
                return True
        return False

    def cone_spray(self, s0: int, picker, n_roll=120, horizon=20, eps=0.25, seed=7, keep=90):
        chain = self.chain
        r = np.random.default_rng(seed)
        got = []
        for _ in range(n_roll):
            s = int(s0)
            for t_ in range(horizon):
                mid = picker(s); k = chain.move_kind[mid]
                if k != 0: break
                outs = chain.outs_of(mid)
                bi = int(self.b_opt[mid]) if r.random() > eps else int(r.integers(0, len(outs)))
                nxt = int(outs[bi])
                if nxt >= chain.n_live: break
                got.append((nxt, t_ + 1)); s = nxt
        if len(got) > keep:
            got = [got[i] for i in np.random.default_rng(1).choice(len(got), keep, replace=False)]
        return got

    def play_record(self, start: int, picker, cap: int = 40, seed: int = 0):
        chain = self.chain
        s = int(start); nodes = []; result = "unfinished"
        for _ in range(cap):
            mvs = self.move_list(s)
            mid = picker(s)
            cands = []
            for m2, nm, ft in mvs:
                v, mate = self.minimax_score(m2)
                cands.append(dict(name=nm, score=(None if abs(v) > 1e8 else round(v, 4)),
                                   mates=bool(chain.move_kind[m2] == 1), fromTo=ft, played=(m2 == mid)))
            cands.sort(key=lambda c: (not c["mates"], -(c["score"] if c["score"] is not None else -1e9)))
            chosen = next(c for c in cands if c["played"])
            k = chain.move_kind[mid]
            node = dict(**self.state_pieces(s),
                        dtm=(None if not np.isfinite(self.dtm_full[s]) or self.dtm_full[s] > 1e5
                             else int(self.dtm_full[s])),
                        fork=bool(self.fork_threat(s)), move=chosen["name"], fromTo=chosen["fromTo"],
                        cands=cands[:8], sIdx=int(s))
            black_ft, black_nm = None, None
            if k == 1:
                result = "mate"; nxt = chain.terminals.mate
            elif k in (2, 3):
                result = "draw"; nxt = chain.terminals.draw
            else:
                outs = chain.outs_of(mid)
                nxt = int(outs[self.b_opt[mid]])
                if nxt == chain.terminals.draw:
                    result = "draw (rook lost)"
                elif nxt < chain.n_live:
                    b4, af = self.state_pieces(s), self.state_pieces(nxt)
                    if s < self.n2 and nxt < self.n2:
                        if af["bk"] != b4["bk"]:
                            black_ft = [b4["bk"], af["bk"]]; black_nm = "K" + sq_name(af["bk"])
                        elif af["bn"] != b4["bn"]:
                            black_ft = [b4["bn"], af["bn"]]; black_nm = "N" + sq_name(af["bn"])
                        if black_nm:
                            wk2, wr2, bk2, bn2 = chain.W[nxt]
                            if wk2 in K2.KN_ATT[bn2]: black_nm += "+"
                    else:
                        if af["bk"] != b4["bk"]:
                            black_ft = [b4["bk"], af["bk"]]; black_nm = "K" + sq_name(af["bk"])
            node["blackReply"] = black_ft
            node["blackMove"] = black_nm

            walts = []
            for m2, nm2, ft2 in mvs:
                if m2 == mid: continue
                v2, _ = self.minimax_score(m2)
                k2 = chain.move_kind[m2]
                if k2 == 1: walts.append(dict(name=nm2, score=None, term="mate"))
                elif k2 in (2, 3): walts.append(dict(name=nm2, score=None, term="draw"))
                else:
                    nxt2 = int(chain.out_flat[chain.op0[m2] + self.b_opt[m2]])
                    if nxt2 >= chain.n_live:
                        walts.append(dict(name=nm2, score=round(v2, 3), term="draw"))
                    else:
                        walts.append(dict(name=nm2, score=round(v2, 3), sIdx=int(nxt2)))
            walts.sort(key=lambda a: -(a["score"] if a["score"] is not None else 1e9))
            node["whiteAlts"] = walts[:7]

            balts = []
            if k == 0:
                b4 = self.state_pieces(s)
                bp = dict(b4)
                frm, to = chosen["fromTo"]
                if frm == b4["wk"]: bp["wk"] = to
                else: bp["wr"] = to
                if b4.get("bn") is not None and to == b4["bn"]: bp["bn"] = None
                outs = chain.outs_of(mid)
                for bi in range(len(outs)):
                    nxt2 = int(outs[bi])
                    if nxt2 == chain.terminals.draw:
                        balts.append(dict(name="xR", score=None, term="draw",
                                           optimal=bool(bi == int(self.b_opt[mid]))))
                        continue
                    af2 = self.state_pieces(nxt2)
                    if bp.get("bn") is not None and af2.get("bn") is not None and af2["bn"] != bp["bn"]:
                        nm2 = "N" + sq_name(af2["bn"])
                        if nxt2 < self.n2:
                            wk2, wr2, bk2, bn2 = chain.W[nxt2]
                            if wk2 in K2.KN_ATT[bn2]: nm2 += "+"
                    elif af2["bk"] != bp["bk"]:
                        nm2 = "K" + sq_name(af2["bk"])
                    else:
                        nm2 = "?"
                    balts.append(dict(name=nm2, score=round(float(self.scores[nxt2]), 3),
                                       sIdx=int(nxt2), optimal=bool(bi == int(self.b_opt[mid]))))
            node["blackAlts"] = balts
            nodes.append(node)
            if nxt >= chain.n_live:
                break
            s = nxt
        return nodes, result


def build_games(builder: KrknViewerBuilder, bands, seed: int = 9, cap: int = 40):
    """Pick starts across DTM bands; planner must reach mate within cap
    (plus one honest recorded failure), each paired with the oracle game."""
    dtm, won = builder.dtm, np.isfinite(builder.dtm)
    rng = np.random.default_rng(seed)
    games = []
    for lo, hi in bands:
        pool = np.where(won & (dtm >= lo) & (dtm < hi))[0]
        if len(pool) == 0:
            continue
        for _ in range(60):
            st = int(pool[rng.integers(0, len(pool))])
            pn, pr = builder.play_record(st, builder.planner_pick, cap=cap)
            if pr == "mate":
                on, orr = builder.play_record(st, builder.oracle_pick, cap=cap)
                games.append(dict(startDtm=int(dtm[st]), planner=dict(nodes=pn, result=pr),
                                   oracle=dict(nodes=on, result=orr)))
                break
    pool = np.where(won & (dtm >= bands[0][0]))[0]
    for _ in range(200):
        st = int(pool[rng.integers(0, len(pool))])
        pn, pr = builder.play_record(st, builder.planner_pick, cap=cap)
        if pr != "mate" and len(pn) >= 3:
            on, orr = builder.play_record(st, builder.oracle_pick, cap=cap)
            games.append(dict(startDtm=int(dtm[st]), planner=dict(nodes=pn, result=pr),
                               oracle=dict(nodes=on, result=orr)))
            break
    return games


def attach_cones(games, builder: KrknViewerBuilder):
    for g in games:
        for side, picker in (("planner", builder.planner_pick), ("oracle", builder.oracle_pick)):
            for i, nd in enumerate(g[side]["nodes"]):
                spray = builder.cone_spray(nd["sIdx"], picker, seed=7 + i)
                nd["cone"] = [[int(sx), int(tt)] for sx, tt in spray]


def finalize_with_xy(games, F: np.ndarray, fmap, round_ndigits: int = 2):
    """One combined FittedMap.project() call over every state index the
    payload references (node states, cone points, alt-fan targets), then
    fills xy/cone/alt coordinates and drops the raw state indices."""
    need = set()
    for g in games:
        for side in ("planner", "oracle"):
            for nd in g[side]["nodes"]:
                need.add(nd["sIdx"])
                need |= {sx for sx, _ in nd.get("cone", [])}
                for a in nd.get("whiteAlts", []):
                    if "sIdx" in a: need.add(a["sIdx"])
                for a in nd.get("blackAlts", []):
                    if "sIdx" in a: need.add(a["sIdx"])
    need = sorted(need)
    pos = {s: i for i, s in enumerate(need)}
    E = np.round(fmap.project(F, np.array(need)), round_ndigits)

    for g in games:
        for side in ("planner", "oracle"):
            for nd in g[side]["nodes"]:
                nd["xy"] = [float(E[pos[nd["sIdx"]], 0]), float(E[pos[nd["sIdx"]], 1])]
                nd["cone"] = [[float(E[pos[sx], 0]), float(E[pos[sx], 1]), tt] for sx, tt in nd.get("cone", [])]
                for a in nd.get("whiteAlts", []):
                    if "sIdx" in a:
                        a["xy"] = [float(E[pos[a["sIdx"]], 0]), float(E[pos[a["sIdx"]], 1])]
                        del a["sIdx"]
                bx = []
                for a in nd.get("blackAlts", []):
                    if "sIdx" in a:
                        a["xy"] = [float(E[pos[a["sIdx"]], 0]), float(E[pos[a["sIdx"]], 1])]
                        bx.append(a["xy"]); del a["sIdx"]
                if bx:
                    nd["bnodeXY"] = [round(float(np.mean([p[0] for p in bx])), round_ndigits),
                                      round(float(np.mean([p[1] for p in bx])), round_ndigits)]
                del nd["sIdx"]
    return games


def build_background(F: np.ndarray, fmap, won: np.ndarray, n2: int, n_bg: int = 6000, seed: int = 0):
    """Downsampled, classed background cloud from the fit sample (0=won, 1=drawn, 2=KRk stratum)."""
    r0 = np.random.default_rng(seed)
    fit_idx = fmap.fit_idx
    sub = r0.choice(len(fit_idx), min(n_bg, len(fit_idx)), replace=False)
    bidx = fit_idx[sub]
    P = fmap.fit_points()
    cls = np.where(bidx >= n2, 2, np.where(won[np.clip(bidx, 0, n2 - 1)], 0, 1))
    return [[float(round(P[i, 0], 2)), float(round(P[i, 1], 2)), int(c)] for i, c in zip(sub, cls)]
