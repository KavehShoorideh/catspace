"""
learn.py — transitions, random-play data, successor-measure learning via rank-d SVD.

The chain lives on W states plus two absorbing pseudo-states: MATE_S, DRAW_S
(stalemate or rook captured). One chain step = white move (policy) + black reply
(uniform random). The successor measure under discount g:
    M = (1-g) * sum_t g^t P^t
is factorized at rank d by randomized SVD:  M ~ F @ B.T  with F = U*S, B = V.
That is exactly the tabular analogue of a Forward-Backward representation.
Learned-from-data version uses the empirical P-hat from sampled games.
"""
import numpy as np
import scipy.sparse as sp
from domain import (enumerate_states, white_moves, black_moves, classify_b,
                    MATE, STALEMATE, ONGOING)

rng = np.random.default_rng(0)

class Chain:
    def __init__(self):
        self.W, self.B = enumerate_states()
        self.Wi = {s: i for i, s in enumerate(self.W)}
        self.nW = len(self.W)
        self.MATE_S = self.nW      # absorbing indices appended
        self.DRAW_S = self.nW + 1
        self.n = self.nW + 2
        self._build()

    def _build(self):
        """For each W state, for each white move: outcome distribution over
        next chain states (uniform over black replies)."""
        self.moves = []       # moves[i] = list of arrays: outcome state indices per white move
        self.move_names = []  # for filmstrips
        for i, s in enumerate(self.W):
            per_move, names = [], []
            for bnode in white_moves(*s):
                cls = classify_b(*bnode)
                if cls == MATE:
                    outcomes = np.array([self.MATE_S])
                elif cls == STALEMATE:
                    outcomes = np.array([self.DRAW_S])
                else:
                    outs = []
                    for nxt, captured in black_moves(*bnode):
                        outs.append(self.DRAW_S if captured else self.Wi[nxt])
                    outcomes = np.array(outs)
                per_move.append(outcomes)
                names.append(self._mv_name(s, bnode))
            self.moves.append(per_move)
            self.move_names.append(names)

    @staticmethod
    def _mv_name(s, b):
        (wk, wr, bk), (wk2, wr2, _) = s, b
        import domain as D
        def nm(sq_): r, c = D.rc(sq_); return f"{'abcde'[c]}{r+1}"
        return (f"K{nm(wk2)}" if wk2 != wk else f"R{nm(wr2)}")

    # ----- exact transition matrix under (white policy = uniform, black = uniform)
    def exact_P_uniform(self):
        rows, cols, vals = [], [], []
        for i in range(self.nW):
            k = len(self.moves[i])
            for outcomes in self.moves[i]:
                p_move = 1.0 / k
                p_out = p_move / len(outcomes)
                for o in outcomes:
                    rows.append(i); cols.append(int(o)); vals.append(p_out)
        for a in (self.MATE_S, self.DRAW_S):
            rows.append(a); cols.append(a); vals.append(1.0)
        P = sp.coo_matrix((vals, (rows, cols)), shape=(self.n, self.n)).tocsr()
        P.sum_duplicates()
        return P

    # ----- random-play games, empirical P-hat
    def sample_games(self, n_games, max_plies=200, seed=1):
        r = np.random.default_rng(seed)
        transitions = []  # (state, next_state)
        starts = r.integers(0, self.nW, size=n_games)
        for g in range(n_games):
            s = int(starts[g])
            for _ in range(max_plies):
                mv = r.integers(0, len(self.moves[s]))
                outcomes = self.moves[s][mv]
                nxt = int(outcomes[r.integers(0, len(outcomes))])
                transitions.append((s, nxt))
                if nxt >= self.nW: break
                s = nxt
        return transitions

    def empirical_P(self, transitions):
        rows = np.array([t[0] for t in transitions])
        cols = np.array([t[1] for t in transitions])
        counts = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                               shape=(self.n, self.n)).tocsr()
        visited = np.asarray(counts.sum(axis=1)).ravel() > 0
        rowsum = np.asarray(counts.sum(axis=1)).ravel()
        rowsum[rowsum == 0] = 1.0
        Dinv = sp.diags(1.0 / rowsum)
        P = Dinv @ counts
        # unvisited rows: self-loop (they contribute nothing; flagged by mask)
        fix = sp.coo_matrix((np.ones((~visited).sum()),
                             (np.where(~visited)[0], np.where(~visited)[0])),
                            shape=(self.n, self.n))
        P = (P + fix).tocsr()
        for a in (self.MATE_S, self.DRAW_S):
            P[a, :] = 0; P[a, a] = 1.0
        P.eliminate_zeros()
        return P.tocsr(), visited

# ----- successor measure matvecs and randomized SVD -----
def sm_matvec(P, X, gamma, T=None):
    """(1-g) * sum_{t=0..T} g^t P^t  applied to columns of X."""
    if T is None:
        T = int(np.ceil(np.log(1e-6) / np.log(gamma)))
    acc = X.copy().astype(np.float64)
    cur = X.astype(np.float64)
    for _ in range(T):
        cur = gamma * (P @ cur)
        acc += cur
        if np.abs(cur).max() < 1e-9: break
    return (1.0 - gamma) * acc

def randomized_svd_sm(P, gamma, d, n_oversample=10, seed=0):
    """Rank-d SVD of the successor measure M without forming it."""
    r = np.random.default_rng(seed)
    n = P.shape[0]
    k = d + n_oversample
    Omega = r.standard_normal((n, k))
    Y = sm_matvec(P, Omega, gamma)                 # M @ Omega
    Q, _ = np.linalg.qr(Y)
    Z = sm_matvec(P.T.tocsr(), Q, gamma)           # M.T @ Q
    Bsmall = Z.T                                    # Q.T @ M
    Ub, S, Vt = np.linalg.svd(Bsmall, full_matrices=False)
    U = Q @ Ub
    return U[:, :d], S[:d], Vt[:d, :].T            # U, S, V

def fb_from_svd(U, S, V):
    """F = U*S (cone shape per state), B = V (goal embedding per state)."""
    return U * S[None, :], V

def rank_error(P, gamma, F, Bm, n_probe=20, seed=3):
    """Relative error ||M - F B^T||_F / ||M||_F via Hutchinson probes."""
    r = np.random.default_rng(seed)
    n = P.shape[0]
    X = r.standard_normal((n, n_probe))
    MX = sm_matvec(P, X, gamma)
    RX = MX - F @ (Bm.T @ X)
    return np.linalg.norm(RX) / np.linalg.norm(MX)

def reach_scores(F, Bm, region_idx):
    """P(cone mass into region) per state = F @ (sum of B rows over region)."""
    zG = Bm[region_idx].sum(axis=0)
    return F @ zG
