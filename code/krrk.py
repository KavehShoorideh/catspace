"""
krrk.py — King + two Rooks vs King on 5x5, as a STRATIFIED union chain.

Strata: KRRK (both rooks alive) --capture--> KRK (one rook) --capture--> DRAW.
The union state space is [KRRK W states][KRK W states][MATE][DRAW]; a black
rook-capture is an irreversible stratum drop, i.e. a chute in the region graph.

DTM is computed stratified: KRK first (reused from domain.py), then KRRK
retrograde with capture edges feeding into the KRK values.

All per-state transition structure is FLATTENED into int32 arrays (memory-safe
at ~100k states on a 3GB box).
"""
import numpy as np
import domain as K1                      # the existing KRK domain (5x5)
from domain import rc, sq, chebyshev, KING_MOVES

N, NSQ = K1.N, K1.NSQ

def rook_slides(rook, blockers):
    r, c = rc(rook); out = []
    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
        rr, cc = r+dr, c+dc
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
        return not any(sq(r, x) in blockers for x in range(lo+1, hi))
    lo, hi = sorted((r, tr))
    return not any(sq(x, c) in blockers for x in range(lo+1, hi))

def bk_in_check(wk, ra, rb, bk):
    return rook_attacks(ra, bk, {wk, rb}) or rook_attacks(rb, bk, {wk, ra})

def w_legal(wk, ra, rb, bk):
    if len({wk, ra, rb, bk}) < 4: return False
    if chebyshev(wk, bk) <= 1: return False
    return not bk_in_check(wk, ra, rb, bk)

def b_legal(wk, ra, rb, bk):
    if len({wk, ra, rb, bk}) < 4: return False
    return chebyshev(wk, bk) > 1

def white_moves2(wk, ra, rb, bk):
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

def black_moves2(wk, ra, rb, bk):
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
def classify_b2(wk, ra, rb, bk):
    if black_moves2(wk, ra, rb, bk): return ONGOING
    return MATE if bk_in_check(wk, ra, rb, bk) else STALEMATE

def enumerate2():
    W2, B2 = [], []
    for wk in range(NSQ):
        for ra in range(NSQ):
            for rb in range(ra+1, NSQ):
                for bk in range(NSQ):
                    if not b_legal(wk, ra, rb, bk): continue
                    B2.append((wk, ra, rb, bk))
                    if not bk_in_check(wk, ra, rb, bk):
                        W2.append((wk, ra, rb, bk))
    return W2, B2

class UnionChain:
    """[KRRK W][KRK W][MATE][DRAW], flattened transition structure."""
    def __init__(self, verbose=True):
        import time
        t0 = time.time()
        self.W2, self.B2 = enumerate2()
        self.W2i = {s: i for i, s in enumerate(self.W2)}
        self.k1 = K1
        self.W1, self.B1 = K1.enumerate_states()
        self.W1i = {s: i for i, s in enumerate(self.W1)}
        self.n2, self.n1 = len(self.W2), len(self.W1)
        self.MATE_S = self.n2 + self.n1
        self.DRAW_S = self.MATE_S + 1
        self.n = self.DRAW_S + 1
        if verbose: print(f"KRRK W={self.n2} B={len(self.B2)} | KRK W={self.n1} | union n={self.n} ({time.time()-t0:.0f}s)")
        self._flatten(verbose, t0)

    def _flatten(self, verbose, t0):
        """Per union W state: moves -> outcome lists (black replies uniform)."""
        mp, mk, op, of = [0], [], [0], []      # move_ptr, move_kind, out_ptr, out_flat
        names = []
        # ---- KRRK stratum
        for si, s in enumerate(self.W2):
            bnodes = white_moves2(*s)
            for bn in bnodes:
                cls = classify_b2(*bn)
                names.append(self._nm2(s, bn))
                if cls == MATE: mk.append(1); of.append(self.MATE_S)
                elif cls == STALEMATE: mk.append(2); of.append(self.DRAW_S)
                else:
                    mk.append(0)
                    for kind, pay in black_moves2(*bn):
                        if kind == 'c':
                            of.append(self.n2 + self.W1i[pay])
                        else:
                            of.append(self.W2i[pay])
                op.append(len(of))
            mp.append(len(mk))
            if verbose and si % 30000 == 0 and si: print(f"  flatten KRRK {si}/{self.n2} ({__import__('time').time()-t0:.0f}s)")
        # ---- KRK stratum (reuse K1 movegen; remap indices)
        for si, s in enumerate(self.W1):
            for bn in K1.white_moves(*s):
                cls = K1.classify_b(*bn)
                names.append(self._nm1(s, bn))
                if cls == K1.MATE: mk.append(1); of.append(self.MATE_S)
                elif cls == K1.STALEMATE: mk.append(2); of.append(self.DRAW_S)
                else:
                    mk.append(0)
                    for nxt, captured in K1.black_moves(*bn):
                        of.append(self.DRAW_S if captured else self.n2 + self.W1i[nxt])
                op.append(len(of))
            mp.append(len(mk))
        self.move_ptr = np.array(mp, dtype=np.int64)
        self.move_kind = np.array(mk, dtype=np.int8)
        self.out_ptr = np.array(op, dtype=np.int64)
        self.out_flat = np.array(of, dtype=np.int32)
        self.move_names = names
        self.nW = self.n2 + self.n1
        if verbose: print(f"flattened: {len(mk)} moves, {len(of)} outcomes ({__import__('time').time()-t0:.0f}s)")

    @staticmethod
    def _nm2(s, b):
        (wk, ra, rb, bk) = s; (wk2, ra2, rb2, _) = b
        def nm(x): r, c = rc(x); return f"{'abcde'[c]}{r+1}"
        if wk2 != wk: return f"K{nm(wk2)}"
        old, new = ({ra, rb} - {ra2, rb2}), ({ra2, rb2} - {ra, rb})
        return f"R{nm(new.pop())}" if new else "R?"
    @staticmethod
    def _nm1(s, b):
        (wk, wr, bk), (wk2, wr2, _) = s, b
        def nm(x): r, c = rc(x); return f"{'abcde'[c]}{r+1}"
        return f"K{nm(wk2)}" if wk2 != wk else f"R{nm(wr2)}"

    # convenience accessors
    def moves_of(self, s):
        a, b = self.move_ptr[s], self.move_ptr[s+1]
        return range(a, b)                       # global move ids
    def outs_of(self, mid):
        return self.out_flat[self.out_ptr[mid]:self.out_ptr[mid+1]]

def compute_dtm_union(uc):
    """Stratified DTM over union W states (plies). KRK values from K1, then
    KRRK retrograde with capture edges into KRK."""
    dtm1_w, _ = K1.compute_dtm(uc.W1, uc.B1)
    dtm = np.full(uc.nW, np.inf)
    dtm[uc.n2:] = dtm1_w
    # value iteration on KRRK W states via the flattened structure:
    # W value = 1 + min over moves of (move value), where
    #   mate move -> 0-after... careful: mate move = 1 ply total
    #   ongoing move value = 1 + max over black replies of dtm[next]  (black delays)
    #   capture reply value uses dtm at the KRK index (already the plies-to-mate there)
    changed = True; it = 0
    while changed:
        changed = False; it += 1
        for s in range(uc.n2):
            best = dtm[s]
            for mid in uc.moves_of(s):
                k = uc.move_kind[mid]
                if k == 1: v = 1.0
                elif k == 2: continue
                else:
                    outs = uc.outs_of(mid)
                    vals = dtm[outs]                 # all outs are union W indices here
                    if np.any(~np.isfinite(vals)): continue
                    v = 2.0 + float(vals.max())      # white ply + black ply
                if v < best: best = v
            if best < dtm[s] - 1e-9:
                dtm[s] = best; changed = True
        if it > 60: raise RuntimeError("union DTM did not converge")
    return dtm, it

if __name__ == "__main__":
    import time
    t0 = time.time()
    uc = UnionChain()
    dtm, iters = compute_dtm_union(uc)
    fin2 = np.isfinite(dtm[:uc.n2])
    print(f"DTM: {iters} sweeps | KRRK forcible: {fin2.sum()}/{uc.n2} "
          f"max={np.nanmax(np.where(fin2, dtm[:uc.n2], np.nan)):.0f} plies "
          f"({time.time()-t0:.0f}s)")
    np.save("dtm_union.npy", dtm)
