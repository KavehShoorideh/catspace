"""catspace/planner/optionality.py -- OPTIONALITY-PRESERVING subgoal navigation + denial, the
mechanism behind MULTIPURPOSE moves (attack + defend + more). Kaveh 2026-07-26.

WHERE THIS LIVES: this is a MOVE-PRIOR shaper, never a value term. The engine's ValueModel stays the
GLOBAL objective E[V] (committor-backed expected score, ARCHITECTURE.md 11); subgoals enter ONLY the
prior/alpha-mixture (interfaces.py: "Subgoals enter HERE ... never the value"). So nothing here
competes with E[V] -- it SHAPES which moves search explores first.

THE IDEA (rigorous framing, not a bolted-on heuristic). We are uncertain about (a) which of our
candidate subgoals will actually pan out and (b) the opponent's z / replies. Under that uncertainty
the value of a position is an expectation over WHICH PLAN succeeds. So we do NOT commit to the single
nearest subgoal (hard-min distance); we aggregate the portfolio SOFTLY:

    soft_reach(s; G, beta) = (1/beta) * logsumexp_k[ beta * (-d(s, g_k)) + log w_k ]

a soft-max of (-distance) over the subgoal set G with weights w_k (>= flux * density). A finite beta
VALUES having several subgoals close at once -- two live subgoals beat one at the same nearest
distance (optionality = a Jensen / value-of-information effect on E[V], NOT a new objective). As
beta -> inf it collapses to -min_k d (hard nearest-subgoal).

MOVE SHAPING. For my move s -> s' (after which the OPPONENT is to move):
    gain_me  = soft_reach(s'; G_me)  - soft_reach(s; G_me)     # got closer to MANY of my subgoals?
    gain_opp = soft_reach(s'; G_opp) - soft_reach(s; G_opp)    # did I let them near THEIR subgoals?
    score(move) = gain_me  -  lam * gain_opp  -  mu * self_blunder(s')
DENIAL: raising the opponent's distance to ALL their subgoals lowers gain_opp -> -lam*gain_opp rises.
MULTIPURPOSE EMERGES: the move that advances MANY of my subgoals (attack) AND raises the barrier to
MANY of theirs (defend) maximizes ONE uncertainty-aware score -- with no rule that says "prefer
multipurpose." self_blunder(s') is the search-complexity proxy for t_loss(s',z_me) (my own error
map; a learned z_me replaces it later).

RISK KNOB. beta couples to sharpness sigma (11): need a win -> raise beta (commit to the single
sharpest subgoal); flexible / winning -> low beta (keep options open).

The FLUX weights w_k and the opponent set G_opp come from T(s,z) once it exists; until then callers
pass heuristic weights (e.g. committor-gain, uniform). The core math here is field-agnostic: it
operates on DISTANCE MATRICES d[move, subgoal], so it is unit-tested in isolation and plugs into any
field (single-space IQE d(phi(s),phi(g)) or the FB field's distance) without change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def logsumexp(a, axis=None, b=None):
    """Stable log(sum(b*exp(a))). b optional (weights). No scipy dependency."""
    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    if b is not None:
        tmp = np.asarray(b, dtype=float) * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    out = np.log(s) + a_max
    return np.squeeze(out, axis=axis) if axis is not None else float(out)


def soft_reach(dists, beta: float, weights=None):
    """soft_reach = (1/beta) logsumexp_k[ beta*(-d_k) + log w_k ]  over the LAST axis (subgoals).
    dists: (..., K) distances to the K subgoals. Returns (...,). Higher = closer to the SET; a
    finite beta rewards MULTIPLE subgoals being close (optionality). beta -> inf gives -min_k d."""
    dists = np.asarray(dists, dtype=float)
    K = dists.shape[-1]
    if K == 0:                                        # empty subgoal set -> zero reach contribution
        return np.zeros(dists.shape[:-1]) if dists.ndim > 1 else 0.0
    w = np.ones(K) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()                                   # normalize so beta alone controls sharpness
    return logsumexp(beta * (-dists), axis=-1, b=w) / beta


def self_blunder_proxy(n_legal: int, in_check: bool, n_captures: int, n_giving_check: int) -> float:
    """Search-complexity proxy for t_loss(s, z_me) -- MY error risk at s (placeholder for a learned
    z_me). Rises with branching (more ways to go wrong), being in check (forced/tactical), and
    tactical tension (captures + checks available = sharp, blunder-prone). Normalized ~[0,1]."""
    branch = min(n_legal / 40.0, 1.0)
    tension = min((n_captures + n_giving_check) / 10.0, 1.0)
    return float(0.5 * branch + 0.3 * tension + 0.2 * float(in_check))


@dataclass
class ShapeWeights:
    beta: float = 1.0        # optionality sharpness (high = commit to nearest subgoal)
    lam: float = 1.0         # denial weight (how much to avoid the opponent's subgoals)
    mu: float = 0.5          # self-blunder-avoidance weight
    temp: float = 1.0        # softmax temperature turning scores into a move prior


def move_scores(d_me_before, d_me_after, d_opp_before, d_opp_after, w: ShapeWeights,
                weights_me=None, weights_opp=None, self_blunder=None):
    """Per-move shaping score (higher = better). Shapes:
      d_me_before  (K_me,)              my soft-distance components at s (pre-move)
      d_me_after   (M, K_me)            ... at each of the M candidate successors s'
      d_opp_before (K_opp,)             opponent's soft-distances at s
      d_opp_after  (M, K_opp)           ... at each s' (opponent to move there)
      self_blunder (M,) or None         self-error proxy at each s'
    Returns scores (M,). gain_me - lam*gain_opp - mu*self_blunder."""
    r_me_before = soft_reach(d_me_before, w.beta, weights_me)
    r_me_after = soft_reach(d_me_after, w.beta, weights_me)          # (M,)
    r_opp_before = soft_reach(d_opp_before, w.beta, weights_opp)
    r_opp_after = soft_reach(d_opp_after, w.beta, weights_opp)       # (M,)
    gain_me = r_me_after - r_me_before
    gain_opp = r_opp_after - r_opp_before
    sb = np.zeros(d_me_after.shape[0]) if self_blunder is None else np.asarray(self_blunder, float)
    return gain_me - w.lam * gain_opp - w.mu * sb


def move_prior(scores, temp: float = 1.0):
    """Softmax the shaping scores into a {index: prob} prior (the alpha-mixture entry)."""
    scores = np.asarray(scores, float)
    p = np.exp((scores - scores.max()) / max(temp, 1e-6))
    p = p / p.sum()
    return p


def board_self_blunder(board) -> float:
    """self_blunder_proxy from a python-chess board (my-error risk at this position)."""
    import chess
    legal = list(board.legal_moves)
    n_captures = sum(1 for m in legal if board.is_capture(m))
    n_checks = sum(1 for m in legal if board.gives_check(m))
    return self_blunder_proxy(len(legal), board.is_check(), n_captures, n_checks)


class PortfolioPrior:
    """A MovePrior (engine interfaces.py) that shapes the search toward a SET of my subgoals G_me and
    AWAY FROM the opponent's set G_opp, valuing optionality + emergent multipurpose moves. This is
    the integration seam Kaveh asked to 'wire up': subgoals enter the PRIOR here, never the value.

    distance_fn(boards, subgoal) -> np.ndarray of field distances d(board -> subgoal) for each board.
    FIELD-AGNOSTIC: pass the FB field's distance now, the single-space IQE d(phi(s),phi(g)) later.
    G_me / G_opp: lists of subgoals (anything distance_fn accepts -- Region, exemplar board, ...).
    weights_me / weights_opp: per-subgoal flux*density weights (uniform until T(s,z) exists)."""

    def __init__(self, distance_fn, g_me, g_opp=None, weights: "ShapeWeights" = None,
                 weights_me=None, weights_opp=None, self_blunder: bool = True):
        self.distance_fn = distance_fn
        self.g_me = list(g_me)
        self.g_opp = list(g_opp or [])
        self.w = weights or ShapeWeights()
        self.weights_me = weights_me
        self.weights_opp = weights_opp
        self.use_self_blunder = self_blunder

    def _dist_to_set(self, boards, subgoals):
        if not subgoals:
            return np.zeros((len(boards), 0))
        return np.stack([np.asarray(self.distance_fn(boards, g), float) for g in subgoals], axis=1)

    def priors(self, board) -> dict:
        moves = list(board.legal_moves)
        if not moves:
            return {}
        succ = []
        for m in moves:
            b = board.copy(stack=False); b.push(m); succ.append(b)
        d_me_before = self._dist_to_set([board], self.g_me)[0]
        d_me_after = self._dist_to_set(succ, self.g_me)                 # (M, K_me)
        if self.g_opp:
            d_opp_before = self._dist_to_set([board], self.g_opp)[0]
            d_opp_after = self._dist_to_set(succ, self.g_opp)
        else:                                                          # no opponent set -> no denial term
            d_opp_before = np.zeros(0); d_opp_after = np.zeros((len(moves), 0))
        sb = np.array([board_self_blunder(b) for b in succ]) if self.use_self_blunder else None
        scores = move_scores(d_me_before, d_me_after, d_opp_before, d_opp_after, self.w,
                             self.weights_me, self.weights_opp, sb)
        p = move_prior(scores, self.w.temp)
        return {m: float(pi) for m, pi in zip(moves, p)}


def select_active_plan(values, incumbent: int | None = None, switch_margin: float = 0.0) -> int:
    """OPPORTUNISM with hysteresis (Kaveh 2026-07-26): re-select which subgoal to EMPHASIZE from the
    CURRENT position each ply. `values` = per-subgoal value (flux * reachability), including any
    transition point the opponent's slip just opened. Forward-looking (Markov, no sunk cost): switch
    to the best plan, but only if it beats the incumbent by `switch_margin` -- so a clear opportunity
    is seized while marginal noise doesn't cause thrash. Returns the chosen subgoal index.

    NOTE the SET stays live in the soft portfolio (keep options open); this only picks the emphasis /
    main plan for move-ordering coherence. With switch_margin=0 it is pure argmax (always opportunistic)."""
    values = np.asarray(values, float)
    best = int(np.argmax(values))
    if incumbent is None or incumbent < 0 or incumbent >= len(values):
        return best
    if values[best] >= values[incumbent] + switch_margin:
        return best
    return incumbent


def multipurpose_index(d_me_before, d_me_after, d_opp_before, d_opp_after, adv: float = 0.25):
    """Diagnostic: per move, (n of MY subgoals advanced) + (n of OPPONENT subgoals denied), where
    'advanced' = distance dropped by >= adv and 'denied' = opponent distance rose by >= adv. A high
    value IS a multipurpose move (serves several of my plans and blocks several of theirs)."""
    d_me_before = np.asarray(d_me_before, float); d_opp_before = np.asarray(d_opp_before, float)
    advanced = (d_me_before[None, :] - np.asarray(d_me_after, float)) >= adv        # (M,K_me)
    denied = (np.asarray(d_opp_after, float) - d_opp_before[None, :]) >= adv         # (M,K_opp)
    return advanced.sum(axis=1) + denied.sum(axis=1)


# --------------------------------------------------------------------------------------------------
def _tests():
    ok = True

    def check(name, cond):
        nonlocal ok; ok &= bool(cond)
        print(f"  {'OK ' if cond else 'FAIL'} {name}")

    # 1. OPTIONALITY: two subgoals at distance 1 beat ONE subgoal at distance 1 (same nearest dist).
    one = soft_reach(np.array([1.0, 9.0]), beta=1.0)
    two = soft_reach(np.array([1.0, 1.0]), beta=1.0)
    check("optionality: 2 near subgoals > 1 near subgoal at equal min-dist", two > one)

    # 2. beta -> inf collapses to hard nearest (-min d).
    hard = soft_reach(np.array([1.0, 1.0, 5.0]), beta=1e6)
    check("beta->inf -> -min(d)", abs(hard - (-1.0)) < 1e-3)

    # 3. finite beta keeps optionality gap; large beta shrinks it (options matter less when committed)
    gap_soft = soft_reach(np.array([1.0, 1.0]), 0.5) - soft_reach(np.array([1.0, 9.0]), 0.5)
    gap_sharp = soft_reach(np.array([1.0, 1.0]), 5.0) - soft_reach(np.array([1.0, 9.0]), 5.0)
    check("higher beta -> smaller optionality bonus", gap_soft > gap_sharp > 0)

    # 4. MULTIPURPOSE EMERGENCE. 2 of my subgoals + 2 opponent subgoals. Moves:
    #    A = multipurpose (advances BOTH mine, denies BOTH theirs)
    #    B = pure attack (advances both mine, ignores theirs)
    #    C = pure defense (denies both theirs, no progress on mine)
    #    D = single-purpose (advances ONE of mine only)
    d_me_before = np.array([3.0, 3.0]); d_opp_before = np.array([3.0, 3.0])
    d_me_after = np.array([
        [2.0, 2.0],   # A advances both mine
        [2.0, 2.0],   # B advances both mine
        [3.0, 3.0],   # C no progress mine
        [2.0, 3.0],   # D advances one mine
    ])
    d_opp_after = np.array([
        [4.0, 4.0],   # A denies both theirs (their dist rose)
        [3.0, 3.0],   # B ignores theirs
        [4.0, 4.0],   # C denies both theirs
        [3.0, 3.0],   # D ignores theirs
    ])
    w = ShapeWeights(beta=1.0, lam=1.0, mu=0.0)
    scores = move_scores(d_me_before, d_me_after, d_opp_before, d_opp_after, w)
    mp = multipurpose_index(d_me_before, d_me_after, d_opp_before, d_opp_after, adv=0.25)
    check("multipurpose move A is top-ranked", int(np.argmax(scores)) == 0)
    check("A beats pure-attack B (denial adds value)", scores[0] > scores[1])
    check("A beats pure-defense C (progress adds value)", scores[0] > scores[2])
    check("A beats single-purpose D", scores[0] > scores[3])
    check("multipurpose_index flags A highest", int(np.argmax(mp)) == 0 and mp[0] == 4)

    # 5. DENIAL sign: with only-defense info, a move that raises opp distance scores > one that lowers it
    dmb = np.array([5.0]); dma = np.array([[5.0], [5.0]])
    dob = np.array([3.0]); doa = np.array([[5.0], [1.0]])   # move0 denies (dist up), move1 helps them
    s = move_scores(dmb, dma, dob, doa, ShapeWeights(beta=1.0, lam=1.0, mu=0.0))
    check("denial: raising opp distance scores higher", s[0] > s[1])

    # 6. move_prior is a valid distribution, monotone in score
    p = move_prior(scores)
    check("prior sums to 1 & argmax matches best score", abs(p.sum() - 1) < 1e-9 and np.argmax(p) == np.argmax(scores))

    # 7. self_blunder_proxy monotonicity
    lo = self_blunder_proxy(10, False, 0, 0); hi = self_blunder_proxy(40, True, 5, 5)
    check("self_blunder proxy rises with complexity/tactics", hi > lo and 0 <= lo <= 1 and 0 <= hi <= 1)

    # 8. OPPORTUNISM (hysteresis): incumbent plan value 1.0. A big new opportunity (value 2.0) -> SWITCH;
    #    a marginal alternative (1.05) within the margin -> STAY.
    check("opportunism: seize a clearly-better opened transition point",
          select_active_plan([1.0, 2.0], incumbent=0, switch_margin=0.2) == 1)
    check("hysteresis: ignore a marginal fluctuation (no thrash)",
          select_active_plan([1.0, 1.05], incumbent=0, switch_margin=0.2) == 0)
    check("no incumbent -> pure argmax", select_active_plan([0.3, 0.9, 0.5]) == 1)

    # 9. BOARD-LEVEL PortfolioPrior wiring (real chess board, synthetic distance_fn = king-distance to
    #    a target square). Valid distribution over legal moves; a move toward BOTH my targets outranks
    #    a move toward one.
    import chess
    def kdist_to_square(boards, sq):                 # field-agnostic stand-in for d(phi(s), phi(g))
        return np.array([chess.square_distance(b.king(chess.WHITE), sq) for b in boards], float)
    board = chess.Board("8/8/8/8/4K3/8/8/k6R w - - 0 1")   # white K on e4, targets pull the king
    g_me = [chess.G6, chess.C6]                            # two subgoals (pull king NE and NW)
    prior = PortfolioPrior(kdist_to_square, g_me, weights=ShapeWeights(beta=1.0, lam=1.0, mu=0.2))
    pr = prior.priors(board)
    legal = list(board.legal_moves)
    check("PortfolioPrior: valid distribution over legal moves",
          abs(sum(pr.values()) - 1.0) < 1e-9 and len(pr) == len(legal) and all(p >= 0 for p in pr.values()))
    # king move toward both targets (e4->d5, reduces dist to both g6 & c6) should beat e4->f3 (away)
    d5 = chess.Move.from_uci("e4d5"); f3 = chess.Move.from_uci("e4f3")
    check("PortfolioPrior: multipurpose king step (toward both targets) out-priors a step away",
          pr.get(d5, 0) > pr.get(f3, 0))

    print("ALL OPTIONALITY TESTS PASSED" if ok else "OPTIONALITY TESTS FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _tests() else 1)
