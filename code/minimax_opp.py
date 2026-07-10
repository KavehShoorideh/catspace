"""
minimax_opp.py — optimal play via alpha-beta minimax, evaluated by DTM.
Fast enough on 5×5 KRK for per-ply decisions (no search tree explosion).
"""
import numpy as np
from domain import classify_b, MATE, DRAW, ONGOING

class OptimalOpponent:
    def __init__(self, dtm_w, dtm_b, W, B, ch):
        """dtm_w, dtm_b are ground-truth distances; W, B are state lists; ch is the Chain."""
        self.dtm_w = dtm_w
        self.dtm_b = dtm_b
        self.W = W
        self.B = B
        self.ch = ch
        self.Wi = {s: i for i, s in enumerate(W)}
        self.Bi = {s: i for i, s in enumerate(B)}

    def eval_w(self, si, alpha=-np.inf, beta=np.inf):
        """Minimax: white to move (maximize reach). Returns best value."""
        dtm = self.dtm_w[si]
        if np.isfinite(dtm):
            return min(dtm, 60.0)  # cap for numerical stability
        # Should not reach here in won positions
        return 60.0

    def eval_b(self, bi, alpha=-np.inf, beta=np.inf):
        """Minimax: black to move (minimize reach). Returns best value."""
        cls = classify_b(*self.B[bi])
        if cls == MATE: return 0.0
        if cls == DRAW: return 60.0  # draw is bad for white
        # Traverse black moves
        best_v = np.inf
        from domain import black_moves
        for nxt, captured in black_moves(*self.B[bi]):
            if captured: return 60.0  # black escapes via capture
            if nxt in self.Wi:
                v = self.eval_w(self.Wi[nxt])
                best_v = min(best_v, v)
                beta = min(beta, best_v)
                if alpha >= beta: break
        return best_v if best_v < np.inf else 60.0

    def best_white_move(self, si):
        """Return index of white's best move from state si."""
        best_v = -np.inf
        best_mi = 0
        from domain import white_moves
        bnodes = white_moves(*self.W[si])
        for mi, bnode in enumerate(bnodes):
            v = self.eval_b(self.Bi[bnode])
            if v > best_v:
                best_v, best_mi = v, mi
        return best_mi

    def best_black_move(self, bi):
        """Return index into black_moves list."""
        from domain import black_moves
        best_v = np.inf
        best_i = 0
        for i, (nxt, captured) in enumerate(black_moves(*self.B[bi])):
            if captured: return i  # black should take the rook (instant draw)
            if nxt in self.Wi:
                v = self.eval_w(self.Wi[nxt])
                if v < best_v:
                    best_v, best_i = v, i
        return best_i
