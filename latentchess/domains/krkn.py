"""
domains/krkn.py — King+Rook vs King+Knight on 5x5: the first two-sided domain.

New physics vs KRRK:
  - black has a real piece: knight forks, pins (knight can't move if it exposes
    bk to the rook), strategic rook attacks
  - white can be IN CHECK (knight) and must evade; white can even be mated or
    stalemated (rare, handled)
  - genuine game-theoretic DRAWS exist: dtm = inf on positions optimal white
    cannot win -- the map gains a WIN/DRAW frontier
Strata: KRKN --white captures N--> KRK (existing stratum)
        KRKN --black captures R--> DRAW (K+N cannot mate)

Union chain: [KRKN W][KRK W][MATE][DRAW][BLACKWIN], flattened.
"""
from __future__ import annotations

import time

import numpy as np

from latentchess.board import rc, sq, chebyshev, KING_MOVES
from latentchess.chain import Terminals, TransitionChain
from latentchess.domains import krk as K1

N, NSQ = K1.N, K1.NSQ


def knight_moves(s):
    r, c = rc(s); out = []
    for dr, dc in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < N and 0 <= cc < N: out.append(sq(rr, cc))
    return out
KNIGHT = [knight_moves(s) for s in range(NSQ)]
KN_ATT = [set(m) for m in KNIGHT]


def rook_attacks(rook, target, blockers):
    r, c = rc(rook); tr, tc = rc(target)
    if r != tr and c != tc: return False
    if r == tr:
        lo, hi = sorted((c, tc))
        return not any(sq(r, x) in blockers for x in range(lo + 1, hi))
    lo, hi = sorted((r, tr))
    return not any(sq(x, c) in blockers for x in range(lo + 1, hi))


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

# state tuple: (wk, wr, bk, bn)
def bk_in_check(wk, wr, bk, bn):     # black king attacked (by rook only)
    return rook_attacks(wr, bk, {wk, bn})
def wk_in_check(wk, wr, bk, bn):     # white king attacked (by knight only)
    return wk in KN_ATT[bn]

def w_legal(wk, wr, bk, bn):
    if len({wk, wr, bk, bn}) < 4: return False
    if chebyshev(wk, bk) <= 1: return False
    return not bk_in_check(wk, wr, bk, bn)      # side not to move can't be in check

def b_legal(wk, wr, bk, bn):
    if len({wk, wr, bk, bn}) < 4: return False
    if chebyshev(wk, bk) <= 1: return False
    return not wk_in_check(wk, wr, bk, bn)      # now WHITE is the side not to move

def white_moves(wk, wr, bk, bn):
    """Returns list of ('m', bnode) | ('xN', krk_bnode). Every move must leave
    wk out of knight check (kings can never be adjacent by construction)."""
    out = []
    in_check = wk in KN_ATT[bn]
    # king moves
    for t in KING_MOVES[wk]:
        if t == wr: continue
        if chebyshev(t, bk) <= 1: continue
        if t == bn:
            out.append(('xN', (t, wr, bk)))        # capture resolves any check
        elif t not in KN_ATT[bn]:
            out.append(('m', (t, wr, bk, bn)))
    # rook slides: blockers wk, bk (hard stop); bn capturable (stop after)
    r, c = rc(wr)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        while 0 <= rr < N and 0 <= cc < N:
            t = sq(rr, cc)
            if t == wk or t == bk: break
            if t == bn:
                out.append(('xN', (wk, t, bk)))    # rook takes knight: check gone
                break
            if not in_check:                        # rook can't parry a knight check
                out.append(('m', (wk, t, bk, bn)))
            rr += dr; cc += dc
    return out

def black_moves(wk, wr, bk, bn):
    """From a KRKN B-node. Returns list of ('m', wnode)|('xR', None).
    All moves must leave bk out of rook check (pins!)."""
    out = []
    for t in KING_MOVES[bk]:
        if t == bn or chebyshev(t, wk) <= 1: continue
        if t == wr:
            if chebyshev(wr, wk) <= 1: continue            # defended by king
            out.append(('xR', None))                        # K+N vs K: draw
        else:
            if rook_attacks(wr, t, {wk, bn}): continue
            out.append(('m', (wk, wr, t, bn)))
    for t in KNIGHT[bn]:
        if t == bk: continue
        if t == wk: continue                                # can't capture king
        if t == wr:
            # knight takes rook -> draw; legal iff bk not left in check (no rook after!)
            out.append(('xR', None))
        else:
            # ordinary knight move: bk must not be exposed to the rook (pin check)
            if rook_attacks(wr, bk, {wk, t}): continue
            out.append(('m', (wk, wr, bk, t)))
    return out

MATE, STALEMATE, ONGOING = 0, 1, 2
def classify_b(wk, wr, bk, bn):
    """Black to move in KRKN."""
    if black_moves(wk, wr, bk, bn): return ONGOING
    return MATE if bk_in_check(wk, wr, bk, bn) else STALEMATE

def classify_w(wk, wr, bk, bn):
    """White to move: can white be mated/stalemated? (rare, but handle)"""
    if white_moves(wk, wr, bk, bn): return ONGOING
    return MATE if wk_in_check(wk, wr, bk, bn) else STALEMATE   # MATE here = BLACK WINS

def enumerate_states():
    W, B = [], []
    for wk in range(NSQ):
        for wr in range(NSQ):
            for bk in range(NSQ):
                if wk == wr or wk == bk or wr == bk: continue
                if chebyshev(wk, bk) <= 1: continue
                for bn in range(NSQ):
                    if bn in (wk, wr, bk): continue
                    if not bk_in_check(wk, wr, bk, bn):
                        W.append((wk, wr, bk, bn))
                    if not wk_in_check(wk, wr, bk, bn):
                        B.append((wk, wr, bk, bn))
    return W, B


def _nm(s, kind, pay):
    def nm(x): r, c = rc(x); return f"{'abcde'[c]}{r + 1}"
    wk, wr, bk, bn = s
    if kind == 'xN':
        wk2, wr2, _ = pay
        return f"Kx{nm(wk2)}" if wk2 != wk else f"Rx{nm(wr2)}"
    wk2, wr2, _, _ = pay
    return f"K{nm(wk2)}" if wk2 != wk else f"R{nm(wr2)}"


def build_chain(verbose: bool = True) -> TransitionChain:
    """[KRKN W][KRK W][MATE][DRAW][BLACKWIN] flattened. Black replies folded
    per white move (uniform sampling / policy chooses index)."""
    t0 = time.time()
    W, B = enumerate_states()
    Wi = {s: i for i, s in enumerate(W)}
    W1, B1 = K1.enumerate_states()
    W1i = {s: i for i, s in enumerate(W1)}
    n2, n1 = len(W), len(W1)
    MATE_S = n2 + n1
    DRAW_S = MATE_S + 1
    BWIN_S = DRAW_S + 1
    n = BWIN_S + 1
    nW = n2 + n1
    if verbose:
        print(f"KRKN W={n2} | KRK W={n1} | union n={n} ({time.time() - t0:.0f}s)")

    def resolve_krk_bnode(bnode):
        """White just captured N -> KRK B-node. Return outcome list (union idx)."""
        cls = K1.classify_b(*bnode)
        if cls == K1.MATE: return [MATE_S]
        if cls == K1.STALEMATE: return [DRAW_S]
        outs = []
        for nxt, captured in K1.black_moves(*bnode):
            outs.append(DRAW_S if captured else n2 + W1i[nxt])
        return outs

    mp, mk, op, of, names = [0], [], [0], [], []
    for si, s in enumerate(W):
        wcls = classify_w(*s)
        if wcls != ONGOING:
            # white has no moves: encode a single pseudo-move to the terminal
            mk.append(3)      # kind 3 = white-terminal
            of.append(BWIN_S if wcls == MATE else DRAW_S)
            op.append(len(of)); names.append("—")
            mp.append(len(mk)); continue
        for kind, pay in white_moves(*s):
            names.append(_nm(s, kind, pay))
            if kind == 'xN':
                mk.append(0); of.extend(resolve_krk_bnode(pay))
            else:
                cls = classify_b(*pay)
                if cls == MATE: mk.append(1); of.append(MATE_S)
                elif cls == STALEMATE: mk.append(2); of.append(DRAW_S)
                else:
                    mk.append(0)
                    for bkind, bpay in black_moves(*pay):
                        of.append(DRAW_S if bkind == 'xR' else Wi[bpay])
            op.append(len(of))
        mp.append(len(mk))
        if verbose and si % 50000 == 0 and si:
            print(f"  flatten {si}/{n2} ({time.time() - t0:.0f}s)")
    # KRK stratum (as in domains/krrk.py)
    for s in W1:
        for bn_ in K1.white_moves(*s):
            cls = K1.classify_b(*bn_)
            names.append("krk")
            if cls == K1.MATE: mk.append(1); of.append(MATE_S)
            elif cls == K1.STALEMATE: mk.append(2); of.append(DRAW_S)
            else:
                mk.append(0)
                for nxt, captured in K1.black_moves(*bn_):
                    of.append(DRAW_S if captured else n2 + W1i[nxt])
            op.append(len(of))
        mp.append(len(mk))

    chain = TransitionChain(
        n=n, n_live=nW,
        move_ptr=np.array(mp, np.int64), move_kind=np.array(mk, np.int8),
        out_ptr=np.array(op, np.int64), out_flat=np.array(of, np.int32),
        terminals=Terminals(mate=MATE_S, draw=DRAW_S, bwin=BWIN_S),
        move_names=names,
        strata={"KRkn": range(0, n2), "KRk": range(n2, nW)},
    )
    chain.W, chain.W1 = W, W1
    if verbose:
        print(f"flattened: {len(mk)} moves, {len(of)} outcomes ({time.time() - t0:.0f}s)")
    return chain


def compute_dtm(chain: TransitionChain) -> np.ndarray:
    """DTM over union W (plies). inf = white cannot force mate (draw or worse).
    KRK values seeded from K1; KRKN by value iteration on the flattened chain
    (white min, black max, terminal moves as encoded)."""
    n2 = chain.strata["KRkn"].stop
    W1 = chain.W1
    W1_full, B1 = K1.enumerate_states()
    dtm1_w, _ = K1.compute_dtm(W1_full, B1)
    dtm = np.full(chain.n_live, np.inf)
    dtm[n2:] = dtm1_w
    MATE_S, DRAW_S, BWIN_S = chain.terminals.mate, chain.terminals.draw, chain.terminals.bwin
    changed, it = True, 0
    while changed:
        changed = False; it += 1
        for s in range(n2):
            best = dtm[s]
            a, b = chain.move_ptr[s], chain.move_ptr[s + 1]
            for mid in range(a, b):
                k = chain.move_kind[mid]
                if k == 1: v = 1.0
                elif k in (2, 3): continue           # stalemate / white-terminal: not a win
                else:
                    outs = chain.outs_of(mid)
                    if len(outs) == 1 and outs[0] == MATE_S:
                        v = 1.0                       # capture that mates on the spot
                    else:
                        worst = 0.0; ok = True
                        for o in outs:
                            if o == MATE_S: vv = 0.0
                            elif o >= chain.n_live: ok = False; break     # DRAW/BWIN reachable by black
                            else:
                                vv = dtm[o]
                                if not np.isfinite(vv): ok = False; break
                            if vv > worst: worst = vv
                        if not ok: continue
                        v = 2.0 + worst
                if v < best: best = v
            if best < dtm[s] - 1e-9:
                dtm[s] = best; changed = True
        if it > 80: raise RuntimeError("KRKN DTM did not converge")
    return dtm


def describe_state(chain: TransitionChain, s: int) -> dict:
    if s < chain.strata["KRkn"].stop:
        wk, wr, bk, bn = chain.W[s]
        return dict(wk=wk, wr=wr, bk=bk, bn=bn, stratum="KRkn")
    wk, wr, bk = chain.W1[s - chain.strata["KRkn"].stop]
    return dict(wk=wk, wr=wr, bk=bk, bn=None, stratum="KRk")


if __name__ == "__main__":
    t0 = time.time()
    uc = build_chain()
    dtm = compute_dtm(uc)
    n2 = uc.strata["KRkn"].stop
    fin = np.isfinite(dtm[:n2])
    print(f"DTM sweeps done | WON: {fin.sum()}/{n2} ({fin.mean():.1%}) "
          f"DRAWN: {(~fin).sum()} | max DTM {np.nanmax(np.where(fin, dtm[:n2], np.nan)):.0f} plies "
          f"({time.time() - t0:.0f}s)")
    np.save("dtm_krkn.npy", dtm)
