"""
domains/krrk.py — King + two Rooks vs King on 5x5, as a STRATIFIED union chain.

Strata: KRRK (both rooks alive) --capture--> KRK (one rook) --capture--> DRAW.
The union state space is [KRRK W states][KRK W states][MATE][DRAW]; a black
rook-capture is an irreversible stratum drop, i.e. a chute in the region graph.

DTM is computed stratified: KRK first (reused from domains/krk.py), then KRRK
retrograde with capture edges feeding into the KRK values.
"""
from __future__ import annotations

import time

import numpy as np

from latentchess.board import rc, sq, chebyshev, KING_MOVES
from latentchess.chain import Terminals, TransitionChain
from latentchess.domains import krk as K1

N, NSQ = K1.N, K1.NSQ


def rook_slides(rook, blockers):
    r, c = rc(rook); out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        while 0 <= rr < N and 0 <= cc < N:
            t = sq(rr, cc)
            if t in blockers: break
            out.append(t)
            rr += dr; cc += dc
    return out


def rook_attacks(rook, target, blockers):
    r, c = rc(rook); tr, tc = rc(target)
    if r != tr and c != tc: return False
    if r == tr:
        lo, hi = sorted((c, tc))
        return not any(sq(r, x) in blockers for x in range(lo + 1, hi))
    lo, hi = sorted((r, tr))
    return not any(sq(x, c) in blockers for x in range(lo + 1, hi))


def bk_in_check(wk, ra, rb, bk):
    return rook_attacks(ra, bk, {wk, rb}) or rook_attacks(rb, bk, {wk, ra})

def w_legal(wk, ra, rb, bk):
    if len({wk, ra, rb, bk}) < 4: return False
    if chebyshev(wk, bk) <= 1: return False
    return not bk_in_check(wk, ra, rb, bk)

def b_legal(wk, ra, rb, bk):
    if len({wk, ra, rb, bk}) < 4: return False
    return chebyshev(wk, bk) > 1

def white_moves(wk, ra, rb, bk):
    """B-nodes reachable by white. Rooks canonicalized ra<rb in outputs."""
    out = []
    for t in KING_MOVES[wk]:
        if t in (ra, rb, bk): continue
        if chebyshev(t, bk) <= 1: continue
        out.append((t, ra, rb, bk))
    for (mv_r, other) in ((ra, rb), (rb, ra)):
        for t in rook_slides(mv_r, {wk, other, bk}):   # cannot capture/pass bk or own men
            a, b = (t, other) if t < other else (other, t)
            out.append((wk, a, b, bk))
    return out

def black_moves(wk, ra, rb, bk):
    """Black replies from a B-node. Returns list of (kind, payload):
       kind 'm' -> payload = next KRRK W tuple
       kind 'c' -> payload = KRK W tuple (wk, remaining_rook, new_bk)"""
    out = []
    for t in KING_MOVES[bk]:
        if t == wk or chebyshev(t, wk) <= 1: continue
        if t in (ra, rb):
            cap, rem = (ra, rb) if t == ra else (rb, ra)
            # capture legal iff landing square not defended AFTER removal
            if chebyshev(t, wk) <= 1: continue
            if rook_attacks(rem, t, {wk}): continue
            out.append(('c', (wk, rem, t)))
        else:
            if bk_in_check(wk, ra, rb, t): continue
            out.append(('m', (wk, ra, rb, t)))
    return out

MATE, STALEMATE, ONGOING = 0, 1, 2
def classify_b(wk, ra, rb, bk):
    if black_moves(wk, ra, rb, bk): return ONGOING
    return MATE if bk_in_check(wk, ra, rb, bk) else STALEMATE

def enumerate_states():
    W, B = [], []
    for wk in range(NSQ):
        for ra in range(NSQ):
            for rb in range(ra + 1, NSQ):
                for bk in range(NSQ):
                    if not b_legal(wk, ra, rb, bk): continue
                    B.append((wk, ra, rb, bk))
                    if not bk_in_check(wk, ra, rb, bk):
                        W.append((wk, ra, rb, bk))
    return W, B


def _nm2(s, b):
    (wk, ra, rb, bk) = s; (wk2, ra2, rb2, _) = b
    def nm(x): r, c = rc(x); return f"{'abcde'[c]}{r + 1}"
    if wk2 != wk: return f"K{nm(wk2)}"
    old, new = ({ra, rb} - {ra2, rb2}), ({ra2, rb2} - {ra, rb})
    return f"R{nm(new.pop())}" if new else "R?"


def build_chain(verbose: bool = True) -> TransitionChain:
    """[KRRK W][KRK W][MATE][DRAW], flattened transition structure."""
    t0 = time.time()
    W2, B2 = enumerate_states()
    W2i = {s: i for i, s in enumerate(W2)}
    W1, B1 = K1.enumerate_states()
    W1i = {s: i for i, s in enumerate(W1)}
    n2, n1 = len(W2), len(W1)
    MATE_S = n2 + n1
    DRAW_S = MATE_S + 1
    n = DRAW_S + 1
    if verbose:
        print(f"KRRK W={n2} B={len(B2)} | KRK W={n1} | union n={n} ({time.time() - t0:.0f}s)")

    mp, mk, op, of, names = [0], [], [0], [], []
    # ---- KRRK stratum
    for si, s in enumerate(W2):
        bnodes = white_moves(*s)
        for bn in bnodes:
            cls = classify_b(*bn)
            names.append(_nm2(s, bn))
            if cls == MATE: mk.append(1); of.append(MATE_S)
            elif cls == STALEMATE: mk.append(2); of.append(DRAW_S)
            else:
                mk.append(0)
                for kind, pay in black_moves(*bn):
                    if kind == 'c':
                        of.append(n2 + W1i[pay])
                    else:
                        of.append(W2i[pay])
            op.append(len(of))
        mp.append(len(mk))
        if verbose and si % 30000 == 0 and si:
            print(f"  flatten KRRK {si}/{n2} ({time.time() - t0:.0f}s)")
    # ---- KRK stratum (reuse K1 movegen; remap indices)
    for si, s in enumerate(W1):
        for bn in K1.white_moves(*s):
            cls = K1.classify_b(*bn)
            names.append(_nm1(s, bn))
            if cls == K1.MATE: mk.append(1); of.append(MATE_S)
            elif cls == K1.STALEMATE: mk.append(2); of.append(DRAW_S)
            else:
                mk.append(0)
                for nxt, captured in K1.black_moves(*bn):
                    of.append(DRAW_S if captured else n2 + W1i[nxt])
            op.append(len(of))
        mp.append(len(mk))

    chain = TransitionChain(
        n=n, n_live=n2 + n1,
        move_ptr=np.array(mp, dtype=np.int64), move_kind=np.array(mk, dtype=np.int8),
        out_ptr=np.array(op, dtype=np.int64), out_flat=np.array(of, dtype=np.int32),
        terminals=Terminals(mate=MATE_S, draw=DRAW_S),
        move_names=names,
        strata={"KRRk": range(0, n2), "KRk": range(n2, n2 + n1)},
    )
    chain.W2, chain.W1 = W2, W1
    if verbose:
        print(f"flattened: {len(mk)} moves, {len(of)} outcomes ({time.time() - t0:.0f}s)")
    return chain


def _nm1(s, b):
    (wk, wr, bk), (wk2, wr2, _) = s, b
    def nm(x): r, c = rc(x); return f"{'abcde'[c]}{r + 1}"
    return f"K{nm(wk2)}" if wk2 != wk else f"R{nm(wr2)}"


def compute_dtm(chain: TransitionChain) -> np.ndarray:
    """Stratified DTM over union W states (plies). KRK values from K1, then
    KRRK retrograde with capture edges into KRK."""
    n2 = chain.strata["KRRk"].stop
    W1_full, B1 = K1.enumerate_states()
    dtm1_w, _ = K1.compute_dtm(W1_full, B1)
    dtm = np.full(chain.n_live, np.inf)
    dtm[n2:] = dtm1_w
    changed = True; it = 0
    while changed:
        changed = False; it += 1
        for s in range(n2):
            best = dtm[s]
            for mid in chain.moves_of(s):
                k = chain.move_kind[mid]
                if k == 1: v = 1.0
                elif k == 2: continue
                else:
                    outs = chain.outs_of(mid)
                    vals = dtm[outs]                 # all outs are union W indices here
                    if np.any(~np.isfinite(vals)): continue
                    v = 2.0 + float(vals.max())      # white ply + black ply
                if v < best: best = v
            if best < dtm[s] - 1e-9:
                dtm[s] = best; changed = True
        if it > 60: raise RuntimeError("union DTM did not converge")
    return dtm


def describe_state(chain: TransitionChain, s: int) -> dict:
    n2 = chain.strata["KRRk"].stop
    if s < n2:
        wk, ra, rb, bk = chain.W2[s]
        return dict(wk=wk, ra=ra, rb=rb, bk=bk, stratum="KRRk")
    wk, wr, bk = chain.W1[s - n2]
    return dict(wk=wk, wr=wr, bk=bk, stratum="KRk")


if __name__ == "__main__":
    t0 = time.time()
    uc = build_chain()
    dtm = compute_dtm(uc)
    n2 = uc.strata["KRRk"].stop
    fin2 = np.isfinite(dtm[:n2])
    print(f"KRRK forcible: {fin2.sum()}/{n2} "
          f"max={np.nanmax(np.where(fin2, dtm[:n2], np.nan)):.0f} plies "
          f"({time.time() - t0:.0f}s)")
    np.save("dtm_union.npy", dtm)
