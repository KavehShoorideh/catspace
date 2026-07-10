"""
domain.py — 5x5 KRK (King+Rook vs King) toy chess domain.

White: king + rook (the planner). Black: king (the opponent).
The planning chain is over WHITE-TO-MOVE states; a transition is
(white move) -> black-to-move node -> classify -> (black reply) -> next W state.

Everything is exact and enumerable (~14k legal W states), so every learned
quantity in the experiment has ground truth.
"""
import numpy as np
from dataclasses import dataclass
from core import N, NSQ, KING_MOVES, rc, rook_moves, rook_attacks, chebyshev, sq


def black_in_check(wk, wr, bk):
    return rook_attacks(wr, bk, {wk})

# ---------- state spaces ----------
# W state: (wk, wr, bk), white to move.  Legal iff pieces distinct,
# kings not adjacent, and black NOT in check (side not to move can't be in check).
# B node:  (wk, wr, bk), black to move.  Legal iff pieces distinct, kings not adjacent.
#          (black MAY be in check here — must respond or be mated).

def w_legal(wk, wr, bk):
    if wk == wr or wk == bk or wr == bk: return False
    if chebyshev(wk, bk) <= 1: return False
    if black_in_check(wk, wr, bk): return False
    return True

def b_legal(wk, wr, bk):
    if wk == wr or wk == bk or wr == bk: return False
    if chebyshev(wk, bk) <= 1: return False
    return True

def enumerate_states():
    W, B = [], []
    for wk in range(NSQ):
        for wr in range(NSQ):
            for bk in range(NSQ):
                if b_legal(wk, wr, bk):
                    B.append((wk, wr, bk))
                    if not black_in_check(wk, wr, bk):
                        W.append((wk, wr, bk))
    return W, B

def white_moves(wk, wr, bk):
    """Legal white moves from a W state. Returns list of resulting B nodes."""
    out = []
    for t in KING_MOVES[wk]:
        if t == wr or t == bk: continue          # can't capture own rook; bk capture impossible (never adjacent)
        if chebyshev(t, bk) <= 1: continue        # king can't move next to enemy king
        out.append((t, wr, bk))
    for t in rook_moves(wr, wk):
        if t == bk: continue                      # rook may never capture the lone king (would be illegal position)
        out.append((wk, t, bk))
    return out

def black_moves(wk, wr, bk):
    """Legal black replies from a B node. Returns list of (next_state, rook_captured)."""
    out = []
    for t in KING_MOVES[bk]:
        if chebyshev(t, wk) <= 1: continue
        if t == wr:
            # capture rook: legal iff rook is undefended by white king
            if chebyshev(wr, wk) <= 1: continue
            out.append(((wk, wr, t), True))
        else:
            if rook_attacks(wr, wk, t): continue  # can't move into check
            out.append(((wk, wr, t), False))
    return out

# ---------- terminal classification of B nodes ----------
MATE, STALEMATE, ONGOING = 0, 1, 2

def classify_b(wk, wr, bk):
    replies = black_moves(wk, wr, bk)
    if replies: return ONGOING
    return MATE if black_in_check(wk, wr, bk) else STALEMATE

# ---------- exact DTM (distance-to-mate in plies, optimal both sides) ----------
def compute_dtm(W, B):
    """Returns (dtm_w, dtm_b): plies to mate with white minimizing, black maximizing.
    inf where mate is not forcible."""
    Wi = {s: i for i, s in enumerate(W)}
    Bi = {s: i for i, s in enumerate(B)}
    INF = np.inf
    dtm_b = np.full(len(B), INF)
    dtm_w = np.full(len(W), INF)

    b_replies = [black_moves(*s) for s in B]
    b_class = [classify_b(*s) for s in B]
    w_children = [ [Bi[c] for c in white_moves(*s)] for s in W ]

    # black-to-move node -> list of successor W indices (rook capture => absorbing draw, excluded)
    b_children = []
    for reps in b_replies:
        ch = []
        for (nxt, captured) in reps:
            if not captured and nxt in Wi:
                ch.append(Wi[nxt])
        b_children.append(ch)
    b_can_capture = [any(c for (_, c) in reps) for reps in b_replies]

    for i, s in enumerate(B):
        if b_class[i] == MATE: dtm_b[i] = 0

    # iterate to fixpoint (bounded by max DTM, small board => small)
    changed = True
    it = 0
    while changed:
        changed = False; it += 1
        # W nodes: white to move, minimizes over children B values
        for i in range(len(W)):
            best = INF
            for j in w_children[i]:
                v = dtm_b[j]
                if v + 1 < best: best = v + 1
            if best < dtm_w[i]:
                dtm_w[i] = best; changed = True
        # B nodes: black maximizes; if black can capture rook or reach a non-mating
        # line (inf), value is inf; else 1 + max over children
        for i in range(len(B)):
            if b_class[i] != ONGOING: continue
            if b_can_capture[i]: continue  # stays inf: black escapes into a draw
            worst = 0.0; all_finite = True
            for j in b_children[i]:
                v = dtm_w[j]
                if not np.isfinite(v): all_finite = False; break
                if v > worst: worst = v
            if all_finite and b_children[i]:
                v = worst + 1
                if v < dtm_b[i]:
                    dtm_b[i] = v; changed = True
        if it > 200: raise RuntimeError("DTM did not converge")
    return dtm_w, dtm_b

# ---------- ground-truth concept features on W states (EVALUATION ONLY) ----------
def box_area(wk, wr, bk):
    """Squares the black king could roam treating the rook's lines as walls
    (classic 'box' the rook cuts). Flood fill from bk; rook square + its
    attacked squares (with wk blocking) are walls."""
    walls = {wr}
    for t in range(NSQ):
        if t != wk and rook_attacks(wr, wk, t):
            walls.add(t)
    if bk in walls:  # shouldn't happen in W states (not in check) but be safe
        return NSQ
    seen = {bk}; stack = [bk]
    while stack:
        cur = stack.pop()
        for t in KING_MOVES[cur]:
            if t not in walls and t not in seen:
                seen.add(t); stack.append(t)
    return len(seen)

def concept_features(W, dtm_w):
    feats = {}
    feats['dtm'] = np.array([min(dtm_w[i], 60.0) for i in range(len(W))])  # cap inf for corr
    feats['dtm_finite'] = np.isfinite(dtm_w).astype(float)
    feats['kk_dist'] = np.array([chebyshev(wk, bk) for (wk, wr, bk) in W], float)
    feats['bk_edge'] = np.array([min(rc(bk)[0], rc(bk)[1], N-1-rc(bk)[0], N-1-rc(bk)[1]) for (_, _, bk) in W], float)
    feats['box_area'] = np.array([box_area(*s) for s in W], float)
    feats['rook_bk_dist'] = np.array([chebyshev(wr, bk) for (_, wr, bk) in W], float)
    return feats

if __name__ == "__main__":
    W, B = enumerate_states()
    print(f"W states: {len(W)}, B nodes: {len(B)}")
    dtm_w, dtm_b = compute_dtm(W, B)
    finite = np.isfinite(dtm_w)
    print(f"forcible-mate W states: {finite.sum()} / {len(W)}  "
          f"max DTM: {dtm_w[finite].max() if finite.any() else '—'}")
    n_mates = sum(1 for s in B if classify_b(*s) == MATE)
    n_stale = sum(1 for s in B if classify_b(*s) == STALEMATE)
    print(f"mate B-nodes: {n_mates}, stalemate B-nodes: {n_stale}")
