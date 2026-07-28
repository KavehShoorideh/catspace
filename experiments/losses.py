#!/usr/bin/env python
"""experiments/losses.py -- CANONICAL, UNIT-TESTED field loss terms (2026-07-26).

Motivation: repeatedly bolting new, untested loss terms onto the objective caused a bug
(margin_ranking with y=sign(0)=0 on 39% tied sibling pairs -> constant unsatisfiable margin
floored the loss; misread as a 'fundamental wall'). Rule now: NO new loss term enters a
training run without a passing test here, and terms are IMPORTED from here, not re-implemented.

Run `python experiments/losses.py` to execute the self-tests.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def quasimetric_regression(d, target_log):
    """Huber on log1p(d) toward a log-space target. d>=0."""
    return F.huber_loss(torch.log1p(d.clamp(min=0)), target_log, delta=1.0)


def wdl_hinge(d, is_won, log_margin):
    """One-sided ∞-barrier: push NON-won (draw/loss) d UP to >= log_margin (bounded repeller).
    is_won: float mask (1 won, 0 draw/loss)."""
    dl = torch.log1p(d.clamp(min=0))
    lost = (1.0 - is_won)
    return (F.relu(log_margin - dl) * lost).sum() / lost.sum().clamp(min=1)


# Ending-type categories (Kaveh's categorical head: "what kind of end is approaching").
# Order is the label index. Draws in the middle, decisive at the ends.
ENDINGS = ["WIN_MATE", "DRAW_FIFTY", "DRAW_STALEMATE", "DRAW_INSUFFICIENT",
           "DRAW_REPETITION", "LOSS_MATE"]
N_ENDINGS = len(ENDINGS)


def categorical_ending_loss(logits, labels):
    """Cross-entropy for the categorical ending-type head. logits (N,N_ENDINGS), labels (N,) int
    in [0,N_ENDINGS). Predicts P(which terminal this position leads to / represents)."""
    return F.cross_entropy(logits, labels)


def anchored_pairwise_rank(d_close, d_far, log_gap):
    """Tie-safe, scale-anchored 1-ply order. Enforces log1p(d_far) - log1p(d_close) >= log_gap,
    where log_gap is the TRUE per-pair target gap (0 for ties -> no push). Caller passes pairs
    with d_close the truly-closer position. Never uses sign()/±1 labels (that was the bug)."""
    dl_c = torch.log1p(d_close.clamp(min=0)); dl_f = torch.log1p(d_far.clamp(min=0))
    return F.relu(log_gap - (dl_f - dl_c)).mean()


def reachability_target(n_moves, path_surprisal):
    """PROBABILITY-ADJUSTED reachability target (Kaveh 2026-07-27): the log-space distance the IQE
    regresses to, folding the transition probability into the move-gap. Reachability isn't the raw
    (player-independent) ply-gap; it's the gap stretched by how UNLIKELY this player is to walk the path:
        d(s->g) <- log1p( n_moves / P(path|z) ) ,  P(path|z)=prod P_player(move_i|z)
                =  log1p( n_moves * exp(path_surprisal) ) ,  path_surprisal = -sum log P >= 0
                =  logaddexp(0, log(n_moves) + path_surprisal)   (numerically safe)
    Constraints (see tests): forced single move (n=1, P=1) -> gap EXACTLY 1 (never 0); forced sequence
    (P=1) -> gap = n_moves; unlikely path -> gap = n_moves/P; the never-taken case is NOT here -- the
    repulsion term (wdl_hinge / random-pair repel) carries it to infinity. Inputs floored: n_moves>=1,
    path_surprisal>=0. Works on floats or numpy arrays (this builds TARGET data, not a differentiable
    loss); import it -- do not re-implement (the losses.py rule)."""
    n = np.asarray(n_moves, dtype=np.float64)
    s = np.asarray(path_surprisal, dtype=np.float64)
    return np.logaddexp(0.0, np.log(np.maximum(n, 1.0)) + np.maximum(s, 0.0))


# --------------------------------------------------------------------------------------------
def _tests():
    torch.manual_seed(0); ok = True

    # anchored_pairwise_rank: correct order (far>close by >= gap) -> ~0; violation -> >0; tie -> 0
    dc = torch.tensor([1.0, 5.0]); df = torch.tensor([3.0, 20.0]); gap = torch.tensor([0.3, 0.3])
    lo = anchored_pairwise_rank(dc, df, gap)
    hi = anchored_pairwise_rank(df, dc, gap)                    # swapped -> violation
    tie = anchored_pairwise_rank(dc, dc, torch.zeros(2))       # ties, gap 0 -> exactly 0
    assert lo.item() < 1e-3, f"correct order should be ~0, got {lo.item()}"
    assert hi.item() > 0.3, f"violation should be large, got {hi.item()}"
    assert tie.item() == 0.0, f"ties (gap=0) must be exactly 0, got {tie.item()}"
    print(f"  anchored_pairwise_rank: order {lo.item():.4f} | violation {hi.item():.3f} | tie {tie.item():.4f}  OK")

    # the OLD bug caught: margin_ranking with y=0 (ties) returns constant margin, not 0
    old = F.margin_ranking_loss(dc, dc, torch.zeros(2), margin=0.5)
    assert abs(old.item() - 0.5) < 1e-6, "sanity: old buggy form returns the margin on ties"
    print(f"  [regression guard] old margin_ranking on ties = {old.item():.2f} (the bug); anchored form = 0.0  OK")

    # wdl_hinge: won stays (mask=1 -> 0 contribution); draw below margin -> pushed
    d = torch.tensor([2.0, 2.0]); won = torch.tensor([1.0, 0.0]); lm = torch.log1p(torch.tensor(400.0))
    h = wdl_hinge(d, won, lm)
    assert h.item() > 4.0, f"draw at d=2 should hinge hard toward logM~6, got {h.item()}"
    print(f"  wdl_hinge: draw d=2 -> {h.item():.2f} (toward logM {lm.item():.2f})  OK")

    # quasimetric_regression: perfect prediction -> 0
    d = torch.tensor([9.0]); t = torch.log1p(torch.tensor([9.0]))
    assert quasimetric_regression(d, t).item() < 1e-6
    print("  quasimetric_regression: exact -> ~0  OK")

    # categorical_ending_loss: confident-correct -> ~0; confident-wrong -> large; N_ENDINGS shape
    lab = torch.tensor([0, 1])
    conf = torch.zeros(2, N_ENDINGS); conf[0, 0] = 20.0; conf[1, 1] = 20.0
    wrong = torch.zeros(2, N_ENDINGS); wrong[0, 3] = 20.0; wrong[1, 4] = 20.0
    assert categorical_ending_loss(conf, lab).item() < 1e-3, "confident-correct -> ~0"
    assert categorical_ending_loss(wrong, lab).item() > 10.0, "confident-wrong -> large"
    assert N_ENDINGS == len(ENDINGS) == 6
    print(f"  categorical_ending_loss: correct {categorical_ending_loss(conf,lab).item():.4f} | "
          f"wrong {categorical_ending_loss(wrong,lab).item():.1f} | {N_ENDINGS} endings  OK")

    # reachability_target: the probability-adjusted reachability constraints (Kaveh 2026-07-27).
    from math import log
    egap = lambda n, s: float(np.expm1(reachability_target(n, s)))   # recovered EFFECTIVE gap
    assert abs(egap(1, 0.0) - 1.0) < 1e-9, f"forced single move must be gap EXACTLY 1, got {egap(1,0.0)}"
    assert egap(1, 0.0) > 0.5, "a certain move must NOT collapse toward 0 (the ply-gap floor stays)"
    assert abs(egap(5, 0.0) - 5.0) < 1e-9, f"forced 5-move path -> gap 5, got {egap(5,0.0)}"
    assert abs(egap(1, log(1000)) - 1000.0) < 1.0, f"Kaveh's 1/1000 -> gap ~1000, got {egap(1,log(1000))}"
    assert egap(1, 2.0) > egap(1, 0.5) > egap(1, 0.0), "gap must grow monotonically with unlikeliness"
    assert abs(egap(3, log(4)) - 12.0) < 1e-6, f"n=3 at P=1/4 -> gap 12 (=n/P), got {egap(3,log(4))}"
    assert reachability_target(1, 50.0) > reachability_target(1, 10.0) > 5.0, "never-taken -> ->inf"
    assert abs(egap(1, -3.0) - 1.0) < 1e-9 and abs(egap(0, 0.0) - 1.0) < 1e-9, "inputs floored (n>=1,s>=0)"
    au, ug = 0.7, 1.1                                             # path surprisal is additive over moves
    assert abs(float(reachability_target(2, au + ug)) - float(np.logaddexp(0.0, log(2) + au + ug))) < 1e-9
    # vectorized (numpy array in -> array out)
    v = reachability_target(np.array([1, 5, 1]), np.array([0.0, 0.0, log(1000)]))
    assert v.shape == (3,) and abs(float(np.expm1(v[2])) - 1000.0) < 1.0
    print(f"  reachability_target: forced 1->{egap(1,0.0):.3f} | forced-5->{egap(5,0.0):.1f} | "
          f"1/1000->{egap(1,log(1000)):.0f} | n3@P1/4->{egap(3,log(4)):.1f} | monotone+additive+floored  OK")

    print("ALL LOSS TESTS PASSED" if ok else "TESTS FAILED")


if __name__ == "__main__":
    _tests()
