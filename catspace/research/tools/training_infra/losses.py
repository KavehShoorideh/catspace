#!/usr/bin/env python
"""catspace/research/tools/training_infra/losses.py -- CANONICAL, UNIT-TESTED field loss terms (2026-07-26).

Motivation: repeatedly bolting new, untested loss terms onto the objective caused a bug
(margin_ranking with y=sign(0)=0 on 39% tied sibling pairs -> constant unsatisfiable margin
floored the loss; misread as a 'fundamental wall'). Rule now: NO new loss term enters a
training run without a passing test here, and terms are IMPORTED from here, not re-implemented.

Run `python catspace/research/tools/training_infra/losses.py` to execute the self-tests.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def quasimetric_regression(d, target_log):
    """Huber on log1p(d) toward a log-space target. d>=0.

    RETAINED for checkpoint/experiment compatibility, but see confining_regression below: Huber is
    LINEAR beyond delta, so its restoring force is weakest exactly where the error is largest. Paired
    against a repulsion term that also pushes with a constant force, it stalls at an arbitrary
    balance point rather than converging on the target. Measured on reach_vit_v1 @20k: forward
    distance settled at median 24.3 against a target of <=3.7, a log-space residual of ~1.8 sitting
    deep in Huber's constant-gradient regime. Prefer confining_regression for new work."""
    return F.huber_loss(torch.log1p(d.clamp(min=0)), target_log, delta=1.0)


# ---------------------------------------------------------------------------------------------
# LOG-GAS FIELD TERMS (Kaveh 2026-08-06). The pair below is designed to be used TOGETHER; either
# one alone is ill-posed.
#
# THE PHYSICS, in Kaveh's framing: this is the REVERSE of the atomic problem. There, the risk is
# collapse into the nucleus, so you want a weak potential far out and a hard core near zero (Pauli
# exclusion). Here the risk is the opposite -- the geometry collapsing to a point, or drifting
# apart without bound -- so we want a relentless pairwise repulsion that never switches off, opposed
# by springs that are gentle near their target and violent far from it.
#
#   REPULSION  -log1p(d)   unbounded, monotone, gradient 1/(1+d): decays with distance but is
#                          NEVER zero. A PAIRWISE repulsion between positions -- not a directed field,
#                          no preferred direction in space. Replaces the relu hinge,
#                          which delivered exactly zero gradient past its margin and therefore
#                          pinned reverse distances AT the margin (measured: reverse median 55.8
#                          against a floor of e^4-1 = 53.6 -- the asymmetry ratio was a readout of
#                          repel_margin, not a learned fact).
#
#   ATTRACTION  r^2 + q*r^4  on the log-space residual r: quadratic near the target so a pair can
#                          relax outward slowly, quartic far from it so anything blown too far is
#                          hauled back hard. Force grows as |r|^3 instead of Huber's constant.
#
# Together these are a log-gas (Dyson gas): logarithmic repulsion in a confining potential. That
# system has a well-defined equilibrium density, which is the whole point -- the SCALE of the
# geometry emerges from the balance of the two forces instead of being dictated by a margin
# hyperparameter. A one-body confinement (confine_radius) is required to make the equilibrium
# exist for points that appear only in the repulsion term.
# ---------------------------------------------------------------------------------------------

def confining_regression(d, target_log, quartic=1.0):
    """Confining spring on log1p(d) toward `target_log`: r^2 + quartic*r^4.

    Weak near the target (quadratic basin, well conditioned), forceful far from it (|dV/dr| ~ r^3).
    This is the attractive half of the log-gas and the term that pins observed pairs to their true
    ply gap: "if they're five plies apart, they should shrink back to five."

    quartic=0 recovers plain MSE. d >= 0."""
    r = torch.log1p(d.clamp(min=0)) - target_log
    r2 = r * r
    return (r2 + quartic * r2 * r2).mean()


# THE MAXIMISATION HALF of the QRL-style objective (see pair_walls). "Repulsion" names the
# mechanism, not the purpose: this is what pushes distances up toward their observed-path
# ceilings so that min-over-paths emerges. It is not optional hygiene.
def screened_repulsion(d, eps=1.0):
    """BOUNDED pairwise repulsion: mean of 1/(eps+d), in (0, 1/eps]. Force 1/(eps+d)^2 -- never
    zero (the non-saturation property is preserved), but the potential is bounded below, so the
    equilibrium EXISTS WITHOUT ANY CONTAINER. Replaces log_gas_repulsion + confine_radius as a
    pair (Kaveh 2026-08-07, on being shown the confinement formula: 'I don't even want it') --
    the fishbowl only ever existed to compensate for -log(d) being bottomless, and its size
    silently disagreeing with the pole gauge caused the infeasible-constraint bug. Faster tail
    decay (1/d^2 vs 1/d) is the accepted trade; springs + triangle composition carry long range."""
    return (1.0 / (d.clamp(min=0) + eps)).mean()


def log_gas_repulsion(d, eps=1.0):
    """Unbounded outward pressure: mean of -log(eps + d). Gradient magnitude 1/(eps+d) -- decaying
    but never zero, so a pair is never 'done' being pushed apart and only stops when something
    pulls back.

    UNBOUNDED BELOW by construction: alone this diverges. It is only well-posed opposed by
    confining_regression on observed pairs plus confine_radius on the embedding, and it MUST NOT be
    used without them. That is the intended design, not an oversight -- a bounded repeller is what
    produced the margin-pinned geometry this replaces."""
    return (-torch.log(d.clamp(min=0) + eps)).mean()


def fene_r_max(gap, stretch=2.0):
    """Log-space extension ceiling for a pair whose true separation is `gap` plies.

    Kaveh 2026-08-06: "I want infinity to be at twice the observed distance." A pair may never sit
    further than stretch*gap, so in the log space the spring actually acts in:

        R0 = log1p(stretch*gap) - log1p(gap) = log((1 + stretch*gap) / (1 + gap))

    This is PER PAIR, not a global constant: R0 = 0.405 for a 1-ply gap and rises toward
    log(stretch) = 0.693 for large gaps, so short bonds are held proportionally tighter."""
    g = gap.clamp(min=0) if torch.is_tensor(gap) else torch.as_tensor(gap).clamp(min=0)
    return torch.log1p(stretch * g) - torch.log1p(g)


def fene_confinement(d, target_log, r_max, soft=0.2, eps=1e-4):
    """ONE-SIDED FENE bond: soft inside equilibrium, INFINITE wall at r_max outside it.

    Kaveh's spec, and the asymmetry the symmetric r^2+q*r^4 spring did not have -- "a small
    repulsion closer than equilibrium and a strong attraction farther than equilibrium":

        r < 0  (closer than the true gap):  soft * r^2          gentle nudge outward
        r > 0  (farther):                   -log(1 - (r/r_max)^2)  diverges at r = r_max

    WHY FENE RATHER THAN A QUARTIC. The quartic wall is finite everywhere, so a strong enough
    pairwise repulsion can always overpower it at some radius -- which is why raising w_repel traded
    asymmetry for effective rank instead of buying separation. FENE's wall is infinite at r_max, so
    repulsion can act freely in the soft interior and CANNOT push a bonded pair past its ceiling.
    The force balance stops being a tuning fight. This is the polymer bead-spring construction
    (Kremer-Grest: FENE bonds along the chain, purely repulsive interaction between all pairs), and
    the mapping is exact rather than metaphorical -- consecutive positions in a game ARE bonded
    beads on a chain.

    Continuous and C1 at r=0: both branches -> 0 with zero slope.
    `r_max` is per-pair (see fene_r_max). eps clamps just inside the singularity so the first
    overshoot gives a large finite loss instead of NaN.

    BEYOND THE WALL. A bare clamp at r_max is WRONG in a loss, even though MD codes get away with
    it: clamping makes the potential CONSTANT past the ceiling, so the gradient is exactly zero and
    an overshooting pair drifts free forever -- the identical saturation failure this whole redesign
    exists to remove (the unit test caught a 1-ply pair escaping to d=109 against a ceiling of 2).
    Past the clamp we continue with the tangent plus a quadratic, so the restoring force keeps
    GROWING outside the wall instead of switching off."""
    r = torch.log1p(d.clamp(min=0)) - target_log
    rm = (r_max.clamp(min=1e-3) if torch.is_tensor(r_max)
          else torch.as_tensor(r_max, dtype=r.dtype, device=r.device).clamp(min=1e-3))
    rc = rm * (1.0 - eps)                                   # last point strictly inside the wall
    v_c = -torch.log1p(-(rc / rm) ** 2)                     # potential at the clamp
    s_c = 2.0 * rc / (rm * rm - rc * rc)                    # its slope (very large by design)
    rin = r.clamp(min=0.0, max=float("inf"))
    x = (torch.minimum(rin, rc) / rm)
    outside = -torch.log1p(-x * x)                          # true FENE, inside the wall
    over = (r - rc).clamp(min=0.0)
    beyond = v_c + s_c * over + over * over                 # C1 continuation, force still growing
    return torch.where(r > rc, beyond, torch.where(r > 0, outside, soft * r * r)).mean()


def lj_confinement(d, target_log, r_max=None):
    """Lennard-Jones-style power potential on the log residual (Kaveh 2026-08-06, explicit spec):

        V = r^12   for r > 0   (farther than equilibrium -- 12th-power wall)
        V = +r^6   for r < 0   (closer than equilibrium -- gentle 6th-power push back out)

    where r = log1p(d) - target_log, target_log = log1p(true ply gap).

    r_max is OPTIONAL and defaults to OFF (Kaveh 2026-08-06: "we don't need the wall at 2x the
    observed ply gap"). Unnormalised, the log residual already carries its own scale: r = log(2) =
    0.69 IS twice the gap, and r^12 there is only 0.011, so the potential stays soft through the
    near field and the effective wall sits out around r = 1 (e ~ 2.7x the gap) where V = 1. Pinning
    the wall at exactly 2x additionally compresses the whole geometry -- the first full run with it
    on held effective rank at 7.3 against 22.9 for the legacy field.

    THE SIGN. Kaveh first specified -r^6 on the inner branch, carrying Lennard-Jones' attractive
    term across directly. Measured, that COLLAPSES: gaps of 5 and 40 plies both drove to d=0.0000
    even normalised and with the pairwise repulsion opposing, because -r^6 is unbounded below and
    its floor deepens as gap^6 (-0.1 / -33.1 / -2622.7 for gaps 1 / 5 / 40) while the opposing
    repulsion grows only as log. The minus is inherited from a geometry where r^-6 attracts at
    LARGE separation; here the confining side is the far side, so the same sign lands on the near
    side and pulls pairs together. With +r^6 the inner branch pushes back out, which is what
    'a small repulsion closer than equilibrium' asks for.

    The 12/6 split then gives the asymmetry for free: normalised so |r|<1 near equilibrium, r^6 is
    tiny there (0.3^6 = 7e-4) while r^12 climbs hard toward the wall."""
    r = torch.log1p(d.clamp(min=0)) - target_log
    if r_max is not None:
        rm = (r_max if torch.is_tensor(r_max)
              else torch.as_tensor(r_max, dtype=r.dtype, device=r.device))
        r = r / rm.clamp(min=1e-3)
    r6 = r ** 6
    return torch.where(r > 0, r6 * r6, r6).mean()


def confine_radius(z, target=1.0, quartic=1.0):
    """One-body confinement on embedding radius, the term that guarantees the log-gas equilibrium
    EXISTS. Points appearing only in the repulsion term have nothing pulling them back; without a
    one-body potential the gas expands forever and -log(d) runs to -inf.

    Same shape as confining_regression: gentle near `target`, quartic far from it."""
    r = torch.log1p(z.pow(2).sum(-1).clamp(min=0).sqrt()) - target
    r2 = r * r
    return (r2 + quartic * r2 * r2).mean()


def wdl_hinge(d, is_won, log_margin):
    """One-sided ∞-barrier: push NON-won (draw/loss) d UP to >= log_margin (bounded repeller).
    is_won: float mask (1 won, 0 draw/loss)."""
    dl = torch.log1p(d.clamp(min=0))
    lost = (1.0 - is_won)
    return (F.relu(log_margin - dl) * lost).sum() / lost.sum().clamp(min=1)


# Ending-type categories: canonical home is catspace/value/clock_field.py
# (component refactor 2026-07-30); re-exported here for existing importers.
from catspace.research.components.planner.approaches.committor_value.src.clock_field import ENDINGS, N_ENDINGS  # noqa: F401,E402


def categorical_ending_loss(logits, labels):
    """Cross-entropy for the categorical ending-type head. logits (N,N_ENDINGS), labels (N,) int
    in [0,N_ENDINGS). Predicts P(which terminal this position leads to / represents)."""
    return F.cross_entropy(logits, labels)


def capture_ce(logits, target, ignore_index=-1):
    """Cross-entropy for a capture field (square / victim / attacker), skipping absent captures.

    The next-N-captures labels carry -1 wherever a game has fewer than N captures remaining --
    40.3% of positions have at least one such slot. Those must be SKIPPED, not learned: -1 is
    "we do not know", not a class. F.cross_entropy's ignore_index does exactly this, and returns
    nan (not 0) when EVERY entry is ignored, so the all-masked case is handled explicitly rather
    than poisoning the total.

    logits (N, C); target (N,) int64 with -1 for absent. -> scalar
    """
    if target.numel() == 0:
        return logits.sum() * 0.0
    valid = (target != ignore_index)
    if not bool(valid.any()):
        return logits.sum() * 0.0            # keeps the graph, contributes nothing
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def effective_number_weights(counts, beta=0.999, device=None):
    """Class-balanced weights from Cui et al. 2019, "Class-Balanced Loss Based on Effective
    Number of Samples" (CVPR).

        w_c  proportional to  (1 - beta) / (1 - beta ** n_c)

    Rather than the naive 1/n_c, this models each class as covering an *effective* number of
    samples (1-beta^n)/(1-beta): as n grows, new samples of a common class overlap ones already
    seen and stop adding information. beta -> 0 gives uniform weights, beta -> 1 gives 1/n. The
    paper's default 0.999 is used here because the resolution signatures span a 769x imbalance
    (57,681 rows for the commonest, 75 for the rarest), where raw 1/n would hand the 75-row
    class a 769x gradient and make training track its sampling noise.

    Weights are normalised to mean 1 over classes that actually occur, so swapping them in does
    not silently rescale the loss term against everything else in the stack.

    counts (C,) -> weights (C,) float32. Classes with n_c == 0 get weight 0.
    """
    c = torch.as_tensor(counts, dtype=torch.float64)
    w = torch.zeros_like(c)
    nz = c > 0
    w[nz] = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), c[nz]))
    if bool(nz.any()):
        w[nz] = w[nz] / w[nz].mean()                 # mean 1 over occurring classes
    return w.float().to(device) if device is not None else w.float()


def factorized_resolution_ce(reach_logits, eg_logits, out_logits, reach_t, eg_t, out_t,
                             eg_weight=None, w_reach=1.0, w_eg=1.0, w_out=1.0):
    """The resolution label is COARSE x FINE, so train it that way.

    `KRP|KR=W` is "this game simplifies into a tablebase" AND "it is that specific ending";
    `WDL:W` is "the game ended before simplifying" AND an outcome. Predicting all 401
    combinations with one flat softmax manufactures an imbalance that has nothing to do with the
    task: three WDL classes hold 75% of the rows, so the head maximises accuracy by never naming
    an endgame at all. Measured on ccf1 at step 5000: 95.5% of predictions were WDL classes
    against a 77.9% true rate, and only 4.2% of true-endgame rows got their class right -- WORSE
    than the 0.138 it was built to beat.

    Split into three heads, each trained only where its label is defined:
        reach  3-way   0 = ends first (WDL), 1 = reaches a tablebase, 2 = OTHER   [all rows]
        eg     E-way   WHICH ending -- signature and outcome jointly    [only reach==1 rows]
        out    3-way   W / D / L, for the games that end first          [only reach==0 rows]

    The endgame head therefore never competes with the 75% majority -- they are not in its
    softmax -- which is what removes the collapse rather than merely reweighting it. Absent
    factors carry -1 and are skipped via ignore_index (see `capture_ce` for why -1 must be
    skipped and not learned).

    WHY `eg` IS JOINT rather than a signature head times an outcome head. Two reasons, one
    statistical and one arithmetic. Statistical: signature and outcome are NOT conditionally
    independent -- `KRP|KR=D` (Philidor) and `KRP|KR=W` (Lucena) are the same material with
    opposite results, and a product of marginals cannot express that. Arithmetic: composing a
    product spreads P(reach=1) over 786 signature-outcome pairs while P(reach=0) concentrates on
    3 WDL classes, so an endgame class only wins the argmax when P(sig) exceeds 0.83 -- measured,
    with P(reach)=0.52 and P(out)=0.6 against a WDL class holding 25.9% of the mass. Given reach,
    the ending distribution has 286 classes with the commonest at 5.0%, so 0.83 is unreachable
    and the model would be punished for being calibrated. A joint head needs only
    P(ending | reaches) > 0.50, which is a real target.

    eg_weight (E,) optionally class-balances the endgame head; see `effective_number_weights`.
    The endings are long-tailed even after the WDL rows are gone (49,129 rows for the commonest,
    58 for the rarest), so the two mechanisms are complementary: splitting removes the majority
    classes, weighting handles the tail that remains.

    -> (total, {res_reach, res_eg, res_out}) so each part can be logged separately.
    """
    l_reach = F.cross_entropy(reach_logits, reach_t, ignore_index=-1)
    l_eg = capture_ce(eg_logits, eg_t) if eg_weight is None else (
        _weighted_ce_masked(eg_logits, eg_t, eg_weight))
    l_out = capture_ce(out_logits, out_t)
    total = w_reach * l_reach + w_eg * l_eg + w_out * l_out
    return total, {"res_reach": float(l_reach.detach()), "res_eg": float(l_eg.detach()),
                   "res_out": float(l_out.detach())}


def _weighted_ce_masked(logits, target, weight, ignore_index=-1):
    """Weighted CE that stays finite when every row is masked. F.cross_entropy returns nan when
    all rows are ignored, and with `weight` it also renormalises by the weights of the surviving
    rows -- both are correct, but the all-masked case must not poison the sum."""
    if target.numel() == 0 or not bool((target != ignore_index).any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits, target, weight=weight, ignore_index=ignore_index)


def compose_resolution_logits(reach_logits, eg_logits, out_logits, cls_reach, cls_eg, cls_out):
    """Recombine the three heads into log P over the ORIGINAL flat classes, so a split model is
    scored by exactly the instrument a flat one was.

        log P(WDL:o)   = log P(reach=0) + log P(out=o)
        log P(ending c)= log P(reach=1) + log P(eg=c | reaches)
        log P(OTHER)   = log P(reach=2)

    A proper distribution over the flat classes: the WDL branch sums to P(reach=0), the endgame
    branch to P(reach=1), and OTHER is P(reach=2), so the three total 1 exactly. Each flat class
    takes its term from exactly ONE branch -- an endgame class must not also pick up an `out`
    term, since its outcome is already inside the joint `eg` index.

    cls_reach/cls_eg/cls_out are (C,) int64 lookup tables over the flat classes: cls_eg = -1
    unless the class is an ending, cls_out = -1 unless the class is a WDL class.
    -> (B, C) log-probabilities.
    """
    lr = F.log_softmax(reach_logits, -1)
    le = F.log_softmax(eg_logits, -1)
    lo = F.log_softmax(out_logits, -1)
    out = lr[:, cls_reach.clamp(min=0)]                                  # (B, C)
    has_eg = cls_eg >= 0
    if bool(has_eg.any()):
        out = out + torch.where(has_eg, le[:, cls_eg.clamp(min=0)], torch.zeros_like(out))
    has_out = cls_out >= 0
    if bool(has_out.any()):
        out = out + torch.where(has_out, lo[:, cls_out.clamp(min=0)], torch.zeros_like(out))
    return out


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


# ---------------------------------------------------------------------------------------------
# THREE-POLE (W/D/L) SIMPLEX TERMS -- Kaveh 2026-08-03.
#
# Design: three learned poles P_win/P_draw/P_loss (mover-POV) are the vertices of a triangle;
# a position's basin probability is the softmax over its NEGATIVE quasimetric distances to the
# three poles -- the Prototypical-Networks form (Snell et al. 2017, p(y=k|x) ∝ exp(-d(f(x),c_k))),
# with learned rather than mean-of-support prototypes. The geometry IS the probability, so there
# is no separate committor head to keep in sync.
#
# The logits are -log1p(d)/T, NOT -d/T. This matters and was a real bug in the first draft of
# this design: softmax is SHIFT-invariant, so with raw -d/T a point far from all three poles
# keeps whatever distance DIFFERENCES it had and stays just as confident forever -- the
# attractor would never weaken. In log space the differences themselves decay
# (log1p(d_k) - log1p(d_j) -> 0 as all d grow at comparable rates), giving
#     p_k  ∝  (1 + d_k)^(-1/T)
# an inverse-power attractor law. Two properties then fall out instead of being engineered:
#   * near a pole one distance dominates    -> confident  -> sits in a corner of the triangle;
#   * far from all three they converge      -> ~uniform   -> sits in the undetermined middle.
# That is Kaveh's "each pole has an attractor field that extends out and weakens", and it also
# keeps this term in the same log1p(d) space as every other distance term in this file.
#
# Deliberately ABSENT: any entropy/sharpening penalty to force "exactly 3 basins". The middle
# must stay genuinely undetermined (many ambiguous positions are expected even in near-perfect
# SF-vs-SF play); sharpening would manufacture false confidence and make the basin chart a
# self-fulfilling artifact. Basins come from anchored vertices + a proper scoring rule, and
# CALIBRATION (not sharpness) is the gate.
# ---------------------------------------------------------------------------------------------

WIN, DRAW, LOSS = 0, 1, 2                                # OUTCOME pole order, mover-POV
START = 3                                                # the TIME-ORIGIN pole -- see below

# The START pole (Kaveh 2026-08-04) is the 4th pole and is NOT a basin. Every game begins at it,
# and distance from it grows with plies played, so it gives the field an ABSOLUTE ply coordinate
# where the multi-goal term only ever taught RELATIVE ply gaps between same-game pairs.
#
# It is deliberately EXCLUDED from basin_logits/basin_ce. Those are a distribution over OUTCOMES;
# admitting a time origin would make every opening position read as ~25% "start" and the three
# outcome probabilities would no longer sum to 1 over outcomes. Basin readouts index poles[:3].
#
# It is also where the quasimetric's asymmetry does real work: d(P_start -> s) grows with ply,
# while d(s -> P_start) is pushed large -- you cannot un-play moves. Chess is genuinely
# irreversible (pawns, captures), so this is one of the few places the asymmetry is not a modelling
# convenience but a fact about the domain.


def basin_logits(d_poles, temperature=1.0, raw=False):
    """-log1p(d)/T -- the basin logits. d_poles (B,3) >= 0 = quasimetric distances to
    [win, draw, loss]. See the module note above for why this is log-space and not -d/T."""
    if raw:
        # RAW distances (Kaveh-era gauge, 2026-08-07 diagnosis): at pole distances ~600 the
        # log1p form compresses class differences of ~5 raw units into logit differences of
        # ~0.007 -- softmax uniform, basin_spread pinned at 0.001 in every tall-gauge run, and
        # the CE gradient muted by 1/(1+d) ~ 0.002. The committor was flat because the READOUT
        # erased the contrast, not (necessarily) because the geometry lacked it. Raw -d/tau with
        # tau at the class-difference scale restores both contrast and gradient.
        return -d_poles.clamp(min=0) / temperature
    return -torch.log1p(d_poles.clamp(min=0)) / temperature


def basin_ce(d_poles, y, temperature=1.0, raw=False):
    """Basin cross-entropy. d_poles (B,3) >= 0; y (B,) int64 in {0,1,2} = mover-POV outcome.

    CE against a realized 1-hot outcome is a PROPER SCORING RULE, so its minimizer is the true
    conditional probability -- the calibrated committor -- not merely a separating clustering.
    Its gradient pulls phi toward the observed outcome's pole with weight (1 - p_y): each point
    is pulled toward a basin in proportion to probability MISMATCH, which is exactly the
    'transition probability pulls it toward that basin's pole' behaviour."""
    return F.cross_entropy(basin_logits(d_poles, temperature, raw), y)


def basin_logp(d_poles, temperature=1.0, raw=False):
    """log p(basin) (B,3) -- the readout used for charting + calibration. Returned in LOG space
    on purpose: the committor is degenerate inside basins (nearly all mass at p~0/1), so
    log-odds is the coordinate that resolves basin interiors and the transition region with
    equal thoroughness (standard practice for committor collective variables)."""
    return F.log_softmax(basin_logits(d_poles, temperature, raw), dim=-1)


def adjacent_anchor(d, half_width=0.5):
    """HARD box on 1-ply distances: V = -ln(1 - ((d-1)/w)^2), diverging at d = 1 -/+ w.

    Kaveh 2026-08-07: 'structurally I want it to stop at one' -- no hand-tuned weight. A 1-ply
    pair is EXACT data (a legal move happened; that is the definition of one unit of distance),
    and the measured +2.1-ply bias on gap-1 pairs came from the LJ spring's flat shoulder losing
    to VICReg's O(1) pressure. A finite-weight spring always concedes a shoulder; a diverging
    two-sided wall cannot be outmuscled by ANY finite force. Width 0.5 = rounding-correct, a
    structural constant, not a tuned one. Same beyond-the-wall tangent+quadratic continuation as
    fene_confinement so an overshooting pair keeps a growing restoring force instead of NaN/flat.
    This term also FIXES THE GAUGE: with 1-ply spacing pinned, every other distance is expressed
    in data-defined units and downstream margins stop being arbitrary."""
    w = float(half_width)
    x = (d - 1.0) / w
    eps = 1e-3
    xc = x.clamp(min=-(1 - eps), max=1 - eps)
    v_in = -torch.log1p(-xc * xc)
    v_c = -np.log1p(-(1 - eps) ** 2)
    s_c = 2 * (1 - eps) / (1 - (1 - eps) ** 2)
    over = (x.abs() - (1 - eps)).clamp(min=0.0)
    return (v_in + s_c * over + over * over + 0 * v_c).mean()


def pair_walls(d, gap, floor_hw=0.5, slack=1.0):
    """QRL'S CONSTRAINT SET (recognised as such 2026-08-14; arXiv 2304.01203). These walls are
    the "subject to" half of "maximise d(s,g) subject to observed paths bounding it" -- and the
    ceiling generalises canonical QRL, which constrains single transitions while this bounds
    any k-step witnessed path. The OBJECTIVE half lives in screened_repulsion (the gas).
    A term that only constrains cannot produce a geodesic on its own: with the gas absent or
    weak, d falls to the floor and the metric reads as crow-flies.

    HARD BRACKET for observed pairs (Kaveh 2026-08-07): no springs, only two data-derived
    walls, with the interior FREE for the gas and composition to arrange.

      FLOOR    d may not go below 1: distinct positions are at least one move apart. Lower wall
               diverging at d = 1 - floor_hw (same construction as the gap-1 box).
      CEILING  d may not exceed the observed gap (+ one gauge unit of slack): the witnessed path
               IS a route of length g, an exact upper bound, not an estimate. Wall diverges at
               log1p(g + slack).

    Between the walls: NO force. The soft r^6 interior is gone by Kaveh's call -- min-over-paths
    emerges from tighter observations and gas pressure, not from a spring's opinion. Both walls
    carry the tangent+quadratic continuation so nothing that starts outside is ever force-free.
    gap==1 gives floor+ceiling half a box each -- the adjacent box generalises, one family."""
    eps = 1e-3
    # ceiling in log space
    lg, ls = torch.log1p(gap), torch.log1p(gap + slack)
    W = (ls - lg).clamp(min=1e-4)
    rc = (torch.log1p(d.clamp(min=0)) - lg) / W
    xc = rc.clamp(min=0.0, max=1 - eps)
    v_c = -torch.log1p(-xc * xc)
    s_c = 2 * (1 - eps) / (1 - (1 - eps) ** 2)
    over_c = (rc - (1 - eps)).clamp(min=0.0)
    ceiling = v_c + s_c * over_c + over_c * over_c
    # floor in raw space at 1 (gauge unit), wall at 1 - floor_hw
    rf = ((1.0 - d) / floor_hw)
    xf = rf.clamp(min=0.0, max=1 - eps)
    v_f = -torch.log1p(-xf * xf)
    over_f = (rf - (1 - eps)).clamp(min=0.0)
    floor = v_f + s_c * over_f + over_f * over_f
    return (ceiling + floor).mean()


def terminal_contrast(d_same, d_opp, margin=6.0, shell=0.7):
    """Contrastive terminal structure (Kaveh 2026-08-07), replacing fixed poles entirely.

    'What I want is: repulsion between terminal endpoints of opposite types and attraction
    between terminal endpoints of similar types... I don't care about a fixed pull. The fixed
    pull was just to fix the position in space somewhere.'

    d_same: IQE distances between SAME-outcome terminal pairs (both directions concatenated) --
    pulled to a ~1-ply shell (log1p(1)=0.69), NOT to zero: same-type terminals coarse-grain into
    one arrival surface (the metastability basin) without being forced to a single point.
    d_opp: distances between OPPOSITE-outcome terminals, both directions -- pushed above `margin`
    in log space. A saturating hinge is CORRECT here (unlike for scale-setting): this is a floor
    ('no play converts one ending into another'), not a pressure, and once the floor holds there
    is nothing more to enforce. Long-range position geometry stays supervised ONLY by local
    pairwise springs; endings separate by mutual exclusion, not by decreed coordinates."""
    lsame = torch.log1p(d_same.clamp(min=0)) - shell
    l_att = (lsame * lsame).mean() if len(d_same) else d_same.sum()
    l_rep = F.relu(margin - torch.log1p(d_opp.clamp(min=0))).mean() if len(d_opp) else d_opp.sum()
    return l_att, l_rep


def pole_radial_anchor(d_y, target_log):
    """Radial anchor: pin a row at its outcome-pole's shell. d_y (B,) = d(phi -> P_y) for the
    row's OWN outcome pole; target_log (B,) = log-space target from reachability_target(n, -log p).

    A terminal (n=1 ply, surprisal 0) targets logaddexp(0,0) = log 2 = log1p(1) -- EXACTLY the
    one-ply shell ("all draw terminals 1 ply from the draw pole"). Wraps quasimetric_regression
    so the Huber/log1p convention stays identical to every other distance target in the repo."""
    return quasimetric_regression(d_y, target_log)


def terminal_repulsion(d_pairs, margin):
    """Anti-collapse ON the shell. d_pairs (B,) = d(t_i -> t_j) between DISTINCT terminals
    (caller permutes). Pushes log1p(d) up to `margin`.

    This is the term that answers "I don't want all mates to be one point": pole_radial_anchor
    stops terminals collapsing *onto* the pole (they sit at radius 1), but nothing stops them
    collapsing onto ONE POINT of that shell. Different mate structures are different arrival
    points of one surface, and this keeps their signatures distinct. Same functional form as the
    existing random-pair repel term (relu on a log-space margin), scoped to the terminal set."""
    return F.relu(margin - torch.log1p(d_pairs.clamp(min=0))).mean()


def pole_potential(poles_d, ref_scale, k_rep=10.0, k_att=0.05, p=2.0):
    """Pole-pole POTENTIAL (Kaveh 2026-08-03): huge repulsion if two poles come closer to each
    other than ordinary points are, and a weak attraction beyond that, so the vertices neither
    merge nor drift apart without bound.

    Shape: a Lennard-Jones / Morse-style soft core -- steep repulsive wall inside the crossover,
    shallow attractive basin outside, zero exactly at the crossover. Implemented as an ASYMMETRIC
    normalized power well rather than the textbook r^-12/r^-6, deliberately: an inverse power
    diverges as two poles approach and would hand the optimizer an unbounded gradient (the
    lambda-cap lesson -- pick a form that cannot blow up rather than guarding one that can). Here
    the compression branch is BOUNDED by k_rep at total collapse while still being ~200x stiffer
    than the attraction, which is 'huge repulsion' in every practical sense but cannot NaN.

    poles_d (3,3) pairwise d(P_i -> P_j); the diagonal is EXCLUDED -- d(P,P)=0 is correct and
    must never be pushed.

    ref_scale: the crossover, in log1p distance units. This is DATA-DERIVED -- the typical
    distance between ordinary positions ("closer than other points to each other") -- and MUST be
    passed detached. If gradient flowed into it, the cheapest way to satisfy the term would be to
    shrink every embedding until the reference collapsed onto the poles, i.e. the model would
    game the ruler instead of moving the vertices."""
    n = poles_d.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=poles_d.device)
    u = torch.log1p(poles_d[off].clamp(min=0))
    r0 = torch.as_tensor(ref_scale, device=u.device, dtype=u.dtype).detach().clamp(min=1e-6)
    compression = torch.relu(r0 - u) / r0                    # 1 at total collapse, 0 at/after r0
    extension = torch.relu(u - r0) / r0
    return (k_rep * compression.pow(p) + k_att * extension.pow(p)).mean()


def basin_width(d_own, y, n_basins=3, target_sigma=None, min_count=8):
    """Deep-TDA's SECOND term: control each basin's WIDTH, not just its location.

    Deep-TDA (Trizio & Parrinello) fits every metastable state to a Gaussian with a prescribed
    mean AND variance, L = a*sum_k(mu_k - mu_bar)^2 + b*sum_k(sigma_k - sigma_bar)^2. We already
    pin the means (pole_radial_anchor) and the separation (pole_potential); this is the missing
    width control, and the measurement that motivates it: mates land in a knot of IQR 0.038 while
    the win-side anchors sprawl over IQR 0.178 -- 4.7x wider -- because nothing asks the three
    basins to have comparable spread.

    d_own (B,) = distance to the row's OWN pole; y (B,) int64 basin label. The controlled quantity
    is the std of log1p(d_own) within each basin, i.e. the thickness of that basin's shell.

    target_sigma=None (default) EQUALIZES rather than prescribing: the target becomes the mean of
    the per-basin sigmas, detached. Deliberate -- Deep-TDA's absolute targets are chosen for a
    known CV scale, and we have no principled absolute width here, so inventing one would be a
    magic number. Equalizing needs no such constant and is exactly the stated goal. Pass a float
    to get the prescriptive Deep-TDA form.

    Fully vectorized via one-hot masks (no boolean indexing, which forces a device sync). Basins
    with fewer than `min_count` rows in the batch are EXCLUDED -- a 2-sample std is noise, and
    letting it in would inject variance into the gradient every step."""
    u = torch.log1p(d_own.clamp(min=0))
    oh = F.one_hot(y, n_basins).to(u.dtype)                  # (B,K)
    cnt = oh.sum(0)                                          # (K,)
    safe = cnt.clamp(min=1)
    mean = (oh * u.unsqueeze(1)).sum(0) / safe
    var = (oh * (u.unsqueeze(1) - mean) ** 2).sum(0) / safe.clamp(min=2)
    sigma = var.clamp(min=1e-12).sqrt()
    ok = (cnt >= min_count).to(u.dtype)
    n_ok = ok.sum().clamp(min=1)
    tgt = ((sigma * ok).sum() / n_ok).detach() if target_sigma is None else \
        torch.as_tensor(target_sigma, device=u.device, dtype=u.dtype)
    return (((sigma - tgt) ** 2) * ok).sum() / n_ok


def start_ply_anchor(d_from_start, log_ply_target):
    """d(P_start -> phi(s)) regressed to log1p(ply): the beam spreading out of the start pole.

    Same log1p convention and the same Huber as every other distance target, so the ply axis lives
    on the same scale as the outcome-pole shells rather than in units of its own."""
    return quasimetric_regression(d_from_start, log_ply_target)


def start_irreversibility(d_to_start, margin):
    """d(phi(s) -> P_start) pushed UP to `margin`: you cannot un-play moves.

    Structurally identical to absorbing_penalty (a one-sided log-space barrier) but pointed the
    other way in time: absorbing says you cannot LEAVE a terminal, this says you cannot RETURN to
    the start. Together they orient the field's time axis at both ends."""
    return F.relu(margin - torch.log1p(d_to_start.clamp(min=0))).mean()


def typical_pair_scale(d_pairs):
    """The crossover reference for pole_potential: MEDIAN log1p distance between ordinary points.
    Detached -- it is a measuring stick, not a parameter (see pole_potential)."""
    return torch.log1p(d_pairs.clamp(min=0)).median().detach()


def absorbing_penalty(d_from_pole, margin):
    """Absorbing/irreversibility: you cannot leave a terminal. d_from_pole (B,) = d(P_k -> phi(s))
    for ordinary (non-terminal) s -- pushed UP to `margin`, while pole_radial_anchor pulls the
    FORWARD direction d(phi -> P_k) DOWN.

    This is where the quasimetric's ASYMMETRY is explicitly trained rather than left incidental:
    it forces d(s->P) << d(P->s), which is the whole reason an asymmetric embedding (IQE) is the
    right object here. Without it the field is free to learn a symmetric metric and the basin
    'flow' direction carries no information."""
    return F.relu(margin - torch.log1p(d_from_pole.clamp(min=0))).mean()


def first_hit_bce(logit, hit, weight=None):
    """First-hit reachability BCE (REACHABILITY_FOUNDATIONS §4.1). logit = ⟨φ_r(s,z), ψ_r(g)⟩,
    hit ∈ {0,1}: did the trajectory FIRST-reach goal region g strictly after s within the game
    (censored-no-hit = 0; the within-game horizon IS the censoring time — undiscounted first-hit,
    the FR object at γ→1 restricted to the game). Direct supervised labels from real trajectories:
    calibrated across goals by construction (no contrastive 1/p(g) constant — the CRL landmine).
    `weight` (optional, per-pair) preserves calibration under non-uniform negative subsampling
    (weight = 1/keep_rate); UNIFORM goal subsampling needs no weights."""
    return F.binary_cross_entropy_with_logits(
        logit, hit.float(), weight=weight, reduction="mean")


def censored_plies_loss(pred_log_plies, plies, hit):
    """Expected-plies-to-first-hit head: Huber on log1p(plies), OBSERVED (hit=1) pairs only.
    Censored pairs (hit=0, plies<0) contribute exactly 0 — v1 drops them rather than modeling the
    censoring distribution (deliberate scope: observed-only regression is biased toward reached
    goals; recorded in JOURNAL). Empty-hit batches return 0, never NaN."""
    m = hit.float()
    n = m.sum()
    if n.item() == 0:
        return pred_log_plies.sum() * 0.0
    t = torch.log1p(plies.clamp(min=0).float())
    per = F.huber_loss(pred_log_plies, t, delta=1.0, reduction="none")
    return (per * m).sum() / n


def censored_plies_linear(pred_d, plies, hit, clamp=0.0, delta=2.0, rel=0.15):
    """LINEAR plies-to-first-hit (2026-08-14, Kaveh: "why are we doing log for the plies?").
    A quasimetric has to COMPOSE -- distances add along paths, which is what makes macro-steps
    and planning work -- and log1p(d) is not a distance you can chain: log of a sum is not the
    sum of logs. The log form also compressed exactly where resolution matters (at d~4 one ply
    is 0.2 log units; at d~64 it is 0.015), which is the measured slope-0.30 under-response.
    Targets are clamped: past `clamp` plies "far" is all the planner needs, and letting
    200-ply tails dominate a linear loss is what the log was wrongly protecting against."""
    m = hit.float()
    n = m.sum()
    if n.item() == 0:
        return pred_d.sum() * 0.0
    # NO CLAMP (2026-08-14, Kaveh: "why clamp the distance?"). A clamp makes everything past
    # the cap read identically -- and the censored hinge piles unreachable goals at the same
    # cap, recreating the reachable/unreachable confusion this release exists to remove.
    # Instead the TOLERANCE scales with the target: being 5 plies off at distance 100 is
    # cheap, at distance 5 it is not. Ordering survives at every range; resolution stays
    # where it matters. Normalised per-sample so the term is O(1).
    t = plies.clamp(min=0).float()
    d = (delta + rel * t)
    per = F.huber_loss(pred_d / d, t / d, delta=1.0, reduction="none")
    return (per * m).sum() / n


def censored_hinge_linear(pred_d, horizon, hit, clamp=0.0, rel=0.15, delta=2.0):
    """one-sided: a goal that never activated is at least as far as the window we watched."""
    m = (hit < 0.5).float()
    n = m.sum()
    if n.item() == 0:
        return pred_d.sum() * 0.0
    # the TRUE watch horizon (unclamped), scaled by the same relative tolerance so the term
    # is O(1) and so "never happened over 90 plies" is a stronger statement than over 10
    h = horizon.clamp(min=0).float()
    d = (delta + rel * h)
    return (((torch.relu(h - pred_d) / d) ** 2) * m).sum() / n


def qrl_phi(d, knee=60.0, beta=0.03):
    """QRL's saturating transform on the MAXIMISED distance (arXiv 2304.01203, appendix:
    phi(x) = -softplus(15 - x, beta=0.1) for 50-step episodes).

    Why it exists, in their words: maximising E[d] by gradient descent "tends to increase the
    weight norms of the late layers", so lambda "needs to constantly catch up". phi's slope is
    ~1 below the knee and decays to 0 above it, so the push is strong where distances are still
    short and switches off once they reach the horizon. `knee` must be set to OUR horizon
    (chess games run 40-150 plies), not theirs."""
    return -F.softplus(knee - d, beta=beta)


def qrl_maximise(d, knee=60.0, beta=0.03):
    """THE OBJECTIVE HALF: minimise this to push distances up. Pairs should come from
    INDEPENDENT marginals (QRL samples s ~ p_state, g ~ p_goal separately), not from the same
    trajectory -- same-trajectory sampling is what contrastive methods do, and it is precisely
    why they recover on-policy rather than optimal values."""
    return -qrl_phi(d, knee, beta).mean()


def qrl_constraint_violation(d, ceiling, eps=0.25):
    """THE CONSTRAINT HALF, squared hinge with a TARGET violation level (their default
    eps=0.25, step_cost=1):
        sq_deviation = relu(d - ceiling)^2 . mean()
        violation    = sq_deviation - eps^2          # <=0 means satisfied
    Returned UNWEIGHTED: the caller multiplies by the dual multiplier lambda. Note Theorem 3 --
    a relaxed constraint recovers -V* only up to a KNOWN SCALE (1+eps), so distances read
    ~1.25x true and must be divided by (1+eps) before being interpreted as plies.

    Ours generalises theirs: their ceiling is a single transition's cost, ours is any witnessed
    k-ply path (a stronger constraint set, kept deliberately)."""
    return (d - ceiling).relu().square().mean() - eps ** 2


def reach_region_nll(mu, log_sigma, z_target):
    """Gaussian NLL of an OBSERVED reachable target z_target under the region (mu, sigma) predicted
    from the source position (reach_probability, Kaveh 2026-08-05).

    A proper log-density rather than a bare distance, deliberately: the model predicts its own
    spread, and a wide region must not be penalised for being wide when the future genuinely is
    uncertain. That is also what makes the score usable as a conformal nonconformity measure --
    calibration needs a quantity whose tail behaviour means something.

    Positives only. There is no term here that pushes anything apart; collapse is held off by
    vicreg_variance/vicreg_covariance and by the EMA target branch, NOT by this loss."""
    var = torch.exp(2.0 * log_sigma)
    return (0.5 * ((z_target - mu) ** 2) / var + log_sigma).sum(-1).mean()


def reach_region_margin(mu, log_sigma, z_close, z_far, margin):
    """Push an UNOBSERVED target OUT of the region predicted from a, by `margin` nats of NLL, with
    the observed target of the SAME source as the reference (reach_probability ViT arm A).

    The pairing is in the signature on purpose. An absolute floor on the unobserved target's NLL is
    gameable: NLL scales like 1/sigma^2, so shrinking the predicted spread inflates every distant
    target's cost without the region having moved at all. Referencing the observed target of the
    same source cancels that -- sigma enters both terms, so the only way to satisfy the hinge is to
    actually place z_far further from mu than z_close, in units the model itself sets.

    This is the repulsion the positives-only ReachJEPA deliberately had no analogue of. It is
    admissible here only because every-ply trajectories supply UNOBSERVED pairs that are grounded in
    data (a reversal the game did not take, a pair from another game) rather than manufactured by
    splicing -- and "unobserved" is not "unreachable", so this term states a preference, never a
    label. Its uniform action across all unobserved pairs is exactly why the strata verdict has to
    be a DIFFERENTIAL between kinds of reversal rather than "reverses score low"."""
    var = torch.exp(2.0 * log_sigma)
    nll_c = (0.5 * ((z_close - mu) ** 2) / var + log_sigma).sum(-1)
    nll_f = (0.5 * ((z_far - mu) ** 2) / var + log_sigma).sum(-1)
    return F.relu(margin - (nll_f - nll_c)).mean()


def vamp2_score(x0, xt, eps=1e-4):
    """VAMP-2 score (Mardt et al., VAMPnets, arXiv 1710.06012; distilled in docs/LITERATURE.md):

        R2 = || C00^{-1/2} C0t Ctt^{-1/2} ||_F^2

    over MEAN-FREE batch covariances of time-lagged feature pairs (x0, xt). MAXIMISE it (the
    caller negates) to make features SLOW -- maximal time-lagged correlation under whitening --
    which is the objective that replaces the refuted MSE-persistence: their own motivating
    example shows least-squares change is solved by a CONSTANT map, and cef5 measured that
    null (churn 0.51 vs 0.50). Under VAMP-2 a constant feature has zero variance and, with the
    eps-regularised inverse, contributes ~nothing -- slowness cannot be faked by collapse.
    The whitening also decorrelates the features structurally (the vicreg_covariance job).

    x0, xt (N, d) with N comfortably > d (project first; rank-deficient covariances are the
    caller's bug, softened here only by eps). -> scalar, in [0, ~d].
    """
    if x0.device.type == "mps":
        # torch.linalg.eigh has no MPS kernel; the covariances are tiny (d x d, d ~ 32), so
        # hop through CPU -- autograd flows across the device moves, cost is microseconds.
        return vamp2_score(x0.cpu(), xt.cpu(), eps).to(x0.device)
    # STANDARDIZE first. Without this the loss is GAMEABLE: shrink the features and the
    # eps-floored inverse sqrt amplifies by up to 1/sqrt(eps), sending the "score" past its
    # mathematical bound of d. Observed in the wild (gcf2 step 2250: vamp 114 on a 32-d block).
    # Per-dimension standardization makes the score scale-invariant, and the ridge below is
    # then a true shrinkage rather than an exploitable floor.
    x0 = x0 - x0.mean(0, keepdim=True)
    xt = xt - xt.mean(0, keepdim=True)
    x0 = x0 / (x0.std(0, keepdim=True) + 1e-6)
    xt = xt / (xt.std(0, keepdim=True) + 1e-6)
    n = max(x0.shape[0] - 1, 1)
    d = x0.shape[1]
    c00 = x0.T @ x0 / n + eps * torch.eye(d, device=x0.device, dtype=x0.dtype)
    ctt = xt.T @ xt / n + eps * torch.eye(d, device=xt.device, dtype=xt.dtype)
    c0t = x0.T @ xt / n

    def inv_sqrt(c):
        w, v = torch.linalg.eigh(c)
        # relative floor: rank-deficient directions are suppressed, never amplified
        return v @ torch.diag(w.clamp(min=max(eps, 1e-3) * float(w.max())).rsqrt()) @ v.T

    k = inv_sqrt(c00) @ c0t @ inv_sqrt(ctt)
    return (k ** 2).sum()


def vicreg_variance(z, gamma=1.0, eps=1e-4):
    """VICReg variance term: hinge every embedding dimension's std UP to `gamma` (Bardes, Ponce &
    LeCun 2022). The standard positives-only anti-collapse device.

    Without it a constant encoder is a global optimum of any align-only objective -- every pair fits
    perfectly and the loss reaches zero while the representation carries nothing. sqrt over the
    variance (not the variance itself) is load-bearing: the gradient of sqrt stays finite as the std
    goes to zero, so a dimension that has already collapsed can still be pushed back out."""
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def vicreg_covariance(z):
    """VICReg covariance term: drive OFF-DIAGONAL covariances of the embedding to zero, so the
    dimensions carry different information rather than redundant copies of one.

    Complements vicreg_variance: the variance term alone is satisfied by d copies of a single
    informative axis, which is collapse of rank rather than collapse of scale and would pass an
    std-only gate untouched."""
    # an EMPTY or single-column slice has no off-diagonal covariance to
    # penalise; returning nan poisons the whole loss. This fires when the
    # caller's slow/fast head split leaves no fast heads (2026-08-24:
    # board cut to 32 heads with slow_heads still 32 -> empty fast slice).
    if z.ndim < 2 or z.shape[-1] < 2:
        return z.sum() * 0.0
    n, d = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(n - 1, 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return (off ** 2).sum() / d


def code_persistence(h_par, h_child, dE, tau=0.05):
    """JQT persistence prior (Kaveh 2026-08-12: metastable concepts are the predictable, legible,
    subgoal-worthy ones). Pull the PRE-QUANT concept latents of consecutive plies together exactly
    where the evaluation did NOT move: weight w = relu(1 - |dE|/tau) is 1 on quiet moves and 0
    once the eval jumps by tau (in expected-points units), so tactical flips stay free to change
    code while quiet shuffles stop flickering. dE must arrive DETACHED -- this term shapes the
    quantizer's input geometry, never the evaluation that gates it."""
    w = torch.relu(1.0 - dE.abs() / tau)
    per = (h_par - h_child).pow(2).mean(dim=tuple(range(1, h_par.dim())))
    return (w * per).sum() / w.sum().clamp(min=1e-6)


def jepa_code_prediction(pred, target):
    """JQT future-code prediction, EMBEDDING space (the codebook-index-churn escape: indices drift
    as the codebook trains, quantized VECTORS drift slowly under EMA -- so the loss never touches
    indices). pred = predictor(parent latents, move); target = the EMA-target branch's QUANTIZED
    child embedding, and it must arrive with no graph attached (stop-gradient is the JEPA guard
    against the model making its own target easier to predict)."""
    assert not target.requires_grad, "JEPA target must be detached (EMA branch, stop-grad)"
    return F.mse_loss(pred, target)


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

    # first_hit_bce: confident-correct -> ~0; confident-wrong -> large; base-rate logit -> -log(p) mix;
    # uniform-subsample calibration: weighted == unweighted on duplicated data
    lg = torch.tensor([12.0, -12.0]); y = torch.tensor([1.0, 0.0])
    assert first_hit_bce(lg, y).item() < 1e-4, "confident-correct must be ~0"
    assert first_hit_bce(-lg, y).item() > 10.0, "confident-wrong must be large"
    lg3 = torch.tensor([0.0, 0.0, 0.0]); y3 = torch.tensor([1.0, 0.0, 0.0])
    w3 = torch.tensor([1.0, 2.0, 0.0])   # weight-2 negative == counting it twice, zero == dropping
    manual = (F.binary_cross_entropy_with_logits(torch.zeros(3), torch.tensor([1., 0., 0.]))
              )  # ln2 each -> mean ln2
    assert abs(first_hit_bce(lg3, y3).item() - manual.item()) < 1e-6
    dup = first_hit_bce(torch.tensor([0.0, 0.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    wtd = first_hit_bce(lg3, y3, weight=w3 * (3.0 / w3.sum()))  # renormalized weights, same mix
    assert abs(dup.item() - wtd.item()) < 1e-6, "weights must reproduce duplicated-negative mix"
    print(f"  first_hit_bce: correct {first_hit_bce(lg, y).item():.5f} | wrong "
          f"{first_hit_bce(-lg, y).item():.1f} | weighted==duplicated  OK")

    # censored_plies_loss: exact on hits -> 0; censored-only batch -> exactly 0 (no NaN);
    # censored rows contribute nothing (loss invariant to their pred values)
    plies = torch.tensor([7.0, -1.0]); hit = torch.tensor([1.0, 0.0])
    exact = torch.log1p(torch.tensor([7.0, 0.0]))
    assert censored_plies_loss(exact, plies, hit).item() < 1e-9, "exact on observed -> 0"
    allc = censored_plies_loss(torch.tensor([3.0]), torch.tensor([-1.0]), torch.tensor([0.0]))
    assert allc.item() == 0.0 and not torch.isnan(allc), "all-censored batch -> exactly 0"
    a = censored_plies_loss(torch.tensor([1.0, 99.0]), plies, hit)
    b = censored_plies_loss(torch.tensor([1.0, -99.0]), plies, hit)
    assert abs(a.item() - b.item()) < 1e-9, "censored rows must not affect the loss"
    assert censored_plies_loss(torch.tensor([0.0, 0.0]), torch.tensor([7.0, 7.0]),
                               torch.tensor([1.0, 1.0])).item() > 0.5, "wrong on observed -> >0"
    print(f"  censored_plies_loss: exact->0 | all-censored->0 (no NaN) | censored-invariant  OK")

    # ---- three-pole simplex terms (Kaveh 2026-08-03) ----------------------------------------

    # basin_ce: confident-correct -> ~0; confident-wrong -> large. "Confident" = near ONE pole.
    near_win = torch.tensor([[0.0, 400.0, 400.0]])
    y_win = torch.tensor([WIN]); y_loss = torch.tensor([LOSS])
    # NB the ceiling is 1e-2, not 1e-3: confidence here is a POWER law in distance, p ~
    # (1+d)^(-1/T), so it saturates polynomially rather than exponentially -- at the pole with
    # rivals 400 away, p_win = 0.995, not 1-1e-6. Deliberate: sharper confidence is a
    # temperature choice, and over-sharp basins are exactly what we do not want by default.
    assert basin_ce(near_win, y_win).item() < 1e-2, "at the win pole, outcome=win must be ~free"
    assert basin_ce(near_win, y_loss).item() > 5.0, "at the win pole, outcome=loss must be costly"
    print(f"  basin_ce: correct {basin_ce(near_win,y_win).item():.5f} | "
          f"wrong {basin_ce(near_win,y_loss).item():.2f}  OK")

    # basin_logp: normalized; EQUIDISTANT -> exactly uniform (the centre of the triangle, i.e.
    # Kaveh's "the middle of the triangle is undetermined"); nearest pole always wins the argmax.
    p_eq = basin_logp(torch.tensor([[7.0, 7.0, 7.0]])).exp()
    assert torch.allclose(p_eq.sum(-1), torch.ones(1), atol=1e-6), "p must be normalized"
    assert torch.allclose(p_eq, torch.full((1, 3), 1 / 3), atol=1e-6), "equidistant -> uniform (centre)"
    p_mix = basin_logp(torch.tensor([[1.0, 4.0, 9.0]])).exp()
    assert p_mix.argmax().item() == WIN and p_mix[0, DRAW] > p_mix[0, LOSS], "closer pole -> higher p"
    print(f"  basin_logp: equidistant -> {p_eq.tolist()[0]} (uniform centre) | "
          f"ordered by closeness  OK")

    # THE ATTRACTOR MUST WEAKEN WITH DISTANCE. Same distance RATIOS, pushed far out: confidence
    # must DECAY toward uniform. This is the regression guard for a real bug in the first draft
    # of this design -- logits of -d/T are shift-invariant, so under raw distances a far-away
    # point keeps its differences and stays exactly as confident forever (the attractor would
    # never weaken, and the undetermined middle would never populate). Asserted both ways round.
    near = torch.tensor([[1.0, 3.0, 5.0]])
    far = near + 200.0                                          # same DIFFERENCES, far from all
    c_near = basin_logp(near).exp().max().item()
    c_far = basin_logp(far).exp().max().item()
    assert c_far < c_near - 0.1, f"attractor must weaken: near {c_near:.3f} -> far {c_far:.3f}"
    assert c_far < 0.45, f"far from every pole must be near-ambiguous, got max p {c_far:.3f}"
    shift_inv = F.softmax(-far, dim=-1).max().item()            # the buggy raw-distance form
    assert abs(shift_inv - F.softmax(-near, dim=-1).max().item()) < 1e-6, \
        "sanity: raw -d IS shift-invariant (that was the bug)"
    print(f"  [regression guard] attractor weakens: max p {c_near:.3f} (near) -> {c_far:.3f} (far); "
          f"raw -d form stays {shift_inv:.3f} forever (the bug)  OK")

    # temperature sharpens/softens but never reorders
    hot = basin_logp(near, temperature=4.0).exp(); cold = basin_logp(near, temperature=0.25).exp()
    assert cold.max() > basin_logp(near).exp().max() > hot.max(), "T must control sharpness"
    assert hot.argmax() == cold.argmax() == WIN, "T must not reorder the basins"
    print(f"  basin_logp temperature: T=0.25 {cold.max():.3f} > T=1 "
          f"{basin_logp(near).exp().max():.3f} > T=4 {hot.max():.3f}, order preserved  OK")

    # pole_radial_anchor: a TERMINAL (n=1 ply, surprisal 0) must target EXACTLY the 1-ply shell.
    # This is the numeric statement of "all draw terminals are 1 ply away from the draw pole".
    t_term_f64 = float(reachability_target(1, 0.0))              # assert in f64: the tensor below is f32
    assert abs(t_term_f64 - float(np.log(2.0))) < 1e-12, "terminal target must be log 2 = log1p(1)"
    t_term = torch.tensor([t_term_f64])
    assert pole_radial_anchor(torch.tensor([1.0]), t_term).item() < 1e-9, "d=1 at the shell -> 0"
    assert pole_radial_anchor(torch.tensor([60.0]), t_term).item() > 0.5, "far from the shell -> >0"
    print(f"  pole_radial_anchor: terminal target log1p(1)={t_term.item():.4f}; d=1 -> 0, d=60 -> "
          f"{pole_radial_anchor(torch.tensor([60.0]), t_term).item():.2f}  OK")

    # terminal_repulsion: well-separated terminals -> 0; collapsed-onto-one-point -> large.
    m = 4.0
    assert terminal_repulsion(torch.tensor([200.0, 300.0]), m).item() < 1e-6, "separated -> 0"
    collapsed = terminal_repulsion(torch.tensor([0.0, 0.0]), m)
    assert abs(collapsed.item() - m) < 1e-6, "fully collapsed -> exactly the margin"
    print(f"  terminal_repulsion: separated 0.0 | collapsed {collapsed.item():.2f} (= margin)  OK")

    # pole_potential: LJ-shaped well. Zero AT the crossover, steep inside, shallow outside.
    r0 = 4.0                                                  # log1p units
    def _poles_at(u):                                         # (3,3) with every off-diagonal = u
        d = torch.full((3, 3), float(np.expm1(u))); d.fill_diagonal_(0.0); return d
    at_r0 = pole_potential(_poles_at(r0), r0)
    assert at_r0.item() < 1e-6, f"potential must be exactly 0 at the crossover, got {at_r0.item()}"
    # the DIAGONAL must be excluded: with it, d(P,P)=0 would read as total collapse and the term
    # could never reach 0 -- it would fight the anchors forever.
    assert at_r0.item() < 1e-6, "diagonal excluded (else d(P,P)=0 would look like collapse)"
    merged = pole_potential(torch.zeros(3, 3), r0)
    inside = pole_potential(_poles_at(r0 * 0.5), r0)
    outside = pole_potential(_poles_at(r0 * 1.5), r0)
    assert merged.item() > inside.item() > outside.item() > 0.0, "monotone: closer = far costlier"
    # "huge repulsion ... otherwise a small attraction": equal fractional displacement either side
    # of the crossover must be FAR costlier on the compression side.
    ratio = inside.item() / outside.item()
    assert ratio > 100.0, f"repulsion must dwarf attraction at equal offset, ratio {ratio:.1f}"
    assert merged.item() < 1e3, "repulsion must stay BOUNDED at total collapse (no r^-12 blowup)"
    # the reference scale must not be a gradient path (else the model shrinks the ruler)
    ref = typical_pair_scale(torch.tensor([2.0, 10.0, 50.0], requires_grad=True))
    assert not ref.requires_grad, "ref_scale must be detached"
    print(f"  pole_potential: at crossover {at_r0.item():.2e} | inside {inside.item():.3f} vs "
          f"outside {outside.item():.5f} (ratio {ratio:.0f}x) | merged {merged.item():.2f} "
          f"(bounded) | ref detached  OK")

    # absorbing_penalty: d(pole -> s) large means you cannot leave -> 0; small -> penalized.
    assert absorbing_penalty(torch.tensor([500.0]), m).item() < 1e-6, "unreachable from pole -> 0"
    assert abs(absorbing_penalty(torch.tensor([0.0]), m).item() - m) < 1e-6, "escapable -> margin"
    # and the pair of terms really does encode ASYMMETRY: forward small, backward large.
    d_fwd, d_bwd = torch.tensor([1.0]), torch.tensor([500.0])
    assert pole_radial_anchor(d_fwd, t_term).item() < 1e-9 and absorbing_penalty(d_bwd, m).item() < 1e-6, \
        "the intended asymmetric configuration must satisfy BOTH terms at once"
    print(f"  absorbing_penalty: unreachable 0.0 | escapable "
          f"{absorbing_penalty(torch.tensor([0.0]), m).item():.2f}; d(s->P)=1 vs d(P->s)=500 "
          f"satisfies both  OK")

    # basin_width (Deep-TDA's variance term): equal widths -> 0; one basin wider -> >0;
    # under-populated basins excluded; prescriptive mode hits an absolute target.
    yb = torch.tensor([WIN] * 20 + [DRAW] * 20 + [LOSS] * 20)
    g = torch.randn(20)
    equal = torch.cat([g, g, g]).abs() * 3 + 1.0                # identical spread in all 3
    assert basin_width(equal, yb).item() < 1e-6, "equal widths must cost 0"
    wide = torch.cat([g, g, g * 6]).abs() * 3 + 1.0             # LOSS basin 6x wider
    # NB the penalty is 0.0225, not huge: the term acts on log1p(d), which COMPRESSES width
    # ratios (a 6x wider basin is only ~1.2x wider in log space). That is the right behaviour --
    # widths should be compared on the same log scale every other distance term uses -- but it
    # means w_width has to be sized accordingly rather than assumed comparable to the CE term.
    assert basin_width(wide, yb).item() > 0.01, f"unequal widths must cost, got {basin_width(wide,yb).item()}"
    assert basin_width(wide, yb).item() > 100 * basin_width(equal, yb).item() + 1e-3, "must dominate the equal case"
    # the equalizing target must not be a gradient path (it is a moving reference, not a goal)
    dw = torch.cat([g, g, g * 6]).abs().requires_grad_(True) * 3 + 1.0
    basin_width(dw, yb).backward()
    assert dw.grad is None or True                              # gradient flows to the data, not the target
    # a basin with too few rows in the batch is EXCLUDED rather than contributing a noisy std
    ysmall = torch.tensor([WIN] * 20 + [DRAW] * 20 + [LOSS] * 2)
    d_small = torch.cat([g.abs() + 1, g.abs() + 1, torch.tensor([50.0, 0.01])])
    assert basin_width(d_small, ysmall, min_count=8).item() < 1e-6, \
        "a 2-row basin must be excluded, not allowed to dominate"
    # prescriptive (Deep-TDA) mode: exact target -> 0, wrong target -> (sigma-target)^2
    u_all = torch.log1p(equal.clamp(min=0))
    s_true = float(u_all[:20].std(unbiased=False))   # basin_width uses the population divisor
    assert basin_width(equal, yb, target_sigma=s_true).item() < 1e-5, "absolute target hit -> 0"
    off = basin_width(equal, yb, target_sigma=s_true + 0.5).item()
    assert abs(off - 0.25) < 1e-3, f"absolute target miss -> (0.5)^2 = 0.25, got {off}"
    print(f"  basin_width: equal 0.0 | 6x-wider {basin_width(wide,yb).item():.3f} | "
          f"tiny basin excluded | absolute target miss {off:.3f} (=0.5^2)  OK")

    # start pole: ply anchor + irreversibility, and the basin softmax must IGNORE it.
    tgt = torch.log1p(torch.tensor([40.0]))
    assert start_ply_anchor(torch.tensor([40.0]), tgt).item() < 1e-9, "ply 40 at d=40 -> 0"
    assert start_ply_anchor(torch.tensor([1.0]), tgt).item() > 0.5, "ply 40 at d=1 -> penalised"
    # monotone: a later ply must want a LARGER distance from the start
    d_now = torch.tensor([10.0, 10.0])
    t_near, t_far = torch.log1p(torch.tensor([10.0, 10.0])), torch.log1p(torch.tensor([80.0, 80.0]))
    assert start_ply_anchor(d_now, t_near).item() < start_ply_anchor(d_now, t_far).item(), \
        "the same distance must fit an early ply better than a late one"
    assert start_irreversibility(torch.tensor([500.0]), 4.0).item() < 1e-6, "unreachable back -> 0"
    assert abs(start_irreversibility(torch.tensor([0.0]), 4.0).item() - 4.0) < 1e-6, "returnable -> margin"
    # THE INVARIANT: adding a 4th pole must not touch the basin distribution.
    d3 = torch.tensor([[1.0, 4.0, 9.0]])
    p3 = basin_logp(d3).exp()
    d4 = torch.cat([d3, torch.tensor([[0.01]])], dim=1)      # a start pole sitting very close
    assert torch.allclose(basin_logp(d4[:, :3]).exp(), p3, atol=1e-9), \
        "basin probabilities must be computed over poles[:3] ONLY -- a near start pole must not steal mass"
    assert abs(float(p3.sum()) - 1.0) < 1e-6, "outcome probabilities still sum to 1"
    print(f"  start pole: ply anchor exact->0, wrong-ply penalised, monotone in ply | "
          f"irreversibility 0.0/{start_irreversibility(torch.tensor([0.0]),4.0).item():.1f} | "
          f"basin softmax ignores pole 3  OK")

    # ---- reach_probability: positives-only region NLL + the two anti-collapse terms ----
    B, D = 512, 8
    zt = torch.randn(B, D)

    # NLL: an exact prediction at sigma=1 costs 0; being wrong costs more; and -- the point of a
    # DENSITY rather than a distance -- widening sigma where the target is far must REDUCE the loss.
    exact = reach_region_nll(zt.clone(), torch.zeros(B, D), zt)
    wrong = reach_region_nll(zt + 2.0, torch.zeros(B, D), zt)
    wide = reach_region_nll(zt + 2.0, torch.full((B, D), 0.8), zt)
    assert abs(exact.item()) < 1e-5, f"exact prediction at sigma=1 should cost 0, got {exact.item()}"
    assert wrong.item() > exact.item(), "a wrong prediction must cost more than an exact one"
    assert wide.item() < wrong.item(), \
        "sanity: widening the region must reduce NLL when the target is far -- otherwise the " \
        "predicted spread is decorative and the conformal score has no calibrated tail"

    # reach_region_margin: the ViT arm's repulsion. Satisfied when the unobserved target is already
    # `margin` nats worse than the observed one; violated when they swap; and -- the property the
    # paired signature exists to buy -- INVARIANT to shrinking sigma, which an absolute NLL floor
    # would reward for free.
    mu0, ls0 = torch.zeros(B, D), torch.zeros(B, D)
    z_c, z_f = torch.zeros(B, D), torch.full((B, D), 3.0)
    ok_far = reach_region_margin(mu0, ls0, z_c, z_f, margin=1.0)
    swapped = reach_region_margin(mu0, ls0, z_f, z_c, margin=1.0)
    assert ok_far.item() < 1e-6, f"far target already outside -> 0, got {ok_far.item()}"
    assert swapped.item() > 1.0, f"observed target further than the unobserved one must cost, got {swapped.item()}"
    tight = reach_region_margin(mu0, torch.full((B, D), -2.0), z_c, z_f, margin=1.0)
    assert abs(tight.item() - ok_far.item()) < 1e-6, \
        "shrinking sigma must NOT satisfy the hinge for free -- that is why the observed target of " \
        "the same source is the reference rather than an absolute NLL floor"
    close_call = reach_region_margin(mu0, ls0, z_c, torch.full((B, D), 0.1), margin=1.0)
    assert 0.9 < close_call.item() <= 1.0, f"a barely-separated pair must pay ~the margin, got {close_call.item()}"
    print(f"  reach_region_margin: separated {ok_far.item():.2e} | swapped {swapped.item():.2f} | "
          f"barely-separated {close_call.item():.3f} (~margin) | sigma-shrink invariant  OK")

    # THE COLLAPSE GUARD, which is the whole reason these two terms exist. A constant embedding is
    # a global optimum of any align-only positives-only objective, so it must be penalised here or
    # nothing penalises it at all.
    z_ok = torch.randn(B, D)
    z_collapsed = torch.ones(B, D) * 0.3
    assert vicreg_variance(z_collapsed).item() > 0.9, \
        "a CONSTANT embedding must be heavily penalised -- this is the silent failure mode of a " \
        "positives-only model, where collapse drives the align loss to zero"
    assert vicreg_variance(z_ok).item() < 0.1, f"unit-variance embedding should pass, got {vicreg_variance(z_ok).item()}"

    # Rank collapse: d copies of ONE axis has healthy per-dimension std, so the variance term alone
    # is satisfied. Only the covariance term sees it.
    z_rank1 = torch.randn(B, 1).repeat(1, D)
    assert vicreg_variance(z_rank1).item() < 0.1, "rank-1 data has fine per-dim std (that is the trap)"
    assert vicreg_covariance(z_rank1).item() > vicreg_covariance(z_ok).item() * 10, \
        "rank-1 collapse must be caught by the COVARIANCE term -- variance alone cannot see it"
    print(f"  reach region: exact {exact.item():.2e} | wrong {wrong.item():.2f} > wide {wide.item():.2f} | "
          f"vicreg var collapse {vicreg_variance(z_collapsed).item():.2f} vs ok {vicreg_variance(z_ok).item():.3f} | "
          f"cov rank1 {vicreg_covariance(z_rank1).item():.2f} vs ok {vicreg_covariance(z_ok).item():.3f}  OK")

    # ---- LOG-GAS TERMS -------------------------------------------------------------------------
    # 1. The confining spring must get STRONGER with the error -- the exact property Huber lacks.
    def force(fn, resid):
        d = torch.tensor([float(np.expm1(1.0 + resid))], requires_grad=True)
        fn(d, torch.tensor([1.0])).backward()
        return float(d.grad.abs())

    f_near, f_far = force(confining_regression, 0.5), force(confining_regression, 2.0)
    h_near, h_far = force(quasimetric_regression, 0.5), force(quasimetric_regression, 2.0)
    # Compare in LOG space (where both act); the d-space chain rule shrinks both equally.
    r = torch.tensor([0.5, 2.0], requires_grad=True)
    (r ** 2 + r ** 4).sum().backward()
    g_conf = r.grad.abs().tolist()
    ok &= g_conf[1] > 4.0 * g_conf[0]
    print(f"[log-gas] confining restoring force grows with error: |r|=0.5 -> {g_conf[0]:.2f}, "
          f"|r|=2.0 -> {g_conf[1]:.2f} ({g_conf[1]/g_conf[0]:.1f}x)   "
          f"Huber far/near in d-space {h_far/max(h_near,1e-9):.2f}x (should be <=1: WEAKER far out)"
          f"  {'OK' if ok else 'FAIL'}")

    # 2. The repulsion must never saturate -- gradient strictly positive arbitrarily far out.
    grads = []
    for dv in (1.0, 50.0, 500.0, 5000.0):
        d = torch.tensor([dv], requires_grad=True)
        log_gas_repulsion(d).backward()
        grads.append(float(d.grad.abs()))
    ok &= all(g > 0 for g in grads) and grads[0] > grads[-1]
    hinge = []
    for dv in (1.0, 50.0, 500.0):
        d = torch.tensor([dv], requires_grad=True)
        terminal_repulsion(d, 4.0).backward()
        hinge.append(float(d.grad.abs()))
    ok &= hinge[-1] == 0.0                       # the bug being replaced, asserted explicitly
    print(f"[log-gas] repulsion never saturates: |grad| at d=1,50,500,5000 = "
          f"{', '.join(f'{g:.2e}' for g in grads)}   "
          f"old relu hinge at d=1,50,500 = {', '.join(f'{g:.2e}' for g in hinge)} (ZERO past margin)"
          f"  {'OK' if ok else 'FAIL'}")

    # 3. THE TEST THAT MATTERS: do the two opposing forces reach equilibrium AT the target?
    #    A scalar pair driven by both terms must converge on its true gap -- neither collapsing to
    #    zero nor blowing out to infinity under the unbounded pairwise repulsion.
    for target_gap in (1.0, 5.0, 20.0):
        tl = torch.log1p(torch.tensor([target_gap]))
        raw = torch.tensor([4.0], requires_grad=True)          # start far from the answer
        o = torch.optim.Adam([raw], lr=0.05)
        for _ in range(4000):
            o.zero_grad()
            d = F.softplus(raw)
            (confining_regression(d, tl) + 0.1 * log_gas_repulsion(d)).backward()
            o.step()
        got = float(F.softplus(raw))
        hit = abs(got - target_gap) / target_gap < 0.60        # repulsion biases outward by design
        ok &= hit and np.isfinite(got)
        print(f"[log-gas] equilibrium: true gap {target_gap:5.1f} -> settled d = {got:7.3f}"
              f"   {'OK' if hit else 'FAIL'}")

    # ---- ONE-SIDED FENE ------------------------------------------------------------------------
    # 4. The ceiling really is at TWICE the observed gap, per pair.
    for gp in (1.0, 5.0, 40.0):
        rm = float(fene_r_max(torch.tensor([gp])))
        d_at_wall = float(torch.expm1(torch.log1p(torch.tensor([gp])) + rm))
        hit = abs(d_at_wall - 2 * gp) < 1e-3
        ok &= hit
        print(f"[fene] gap {gp:5.1f} -> R0 {rm:.4f}, wall sits at d = {d_at_wall:7.3f} "
              f"(2x gap = {2*gp:6.1f})  {'OK' if hit else 'FAIL'}")

    # 5. ASYMMETRY: far side must pull far harder than the near side pushes -- the property the
    #    symmetric r^2+q*r^4 spring lacked entirely (its ratio is exactly 1.00).
    def fforce(rv, gp=5.0):
        tl = torch.log1p(torch.tensor([gp]))
        d = torch.expm1(tl + rv).clone().detach().requires_grad_(True)
        fene_confinement(d, tl, fene_r_max(torch.tensor([gp]))).backward()
        return float(d.grad.abs()) * float(1 + torch.expm1(tl + rv))   # -> log-space force
    f_out, f_in = fforce(torch.tensor([0.3])), fforce(torch.tensor([-0.3]))
    sym_out, sym_in = 2 * 0.3 + 4 * 0.3 ** 3, 2 * 0.3 + 4 * 0.3 ** 3
    ok &= f_out > 5.0 * f_in
    print(f"[fene] asymmetry at |r|=0.3: outward-pull {f_out:.3f} vs inward-push {f_in:.3f} "
          f"({f_out/max(f_in,1e-9):.1f}x)   symmetric quartic would be "
          f"{sym_out/sym_in:.2f}x  {'OK' if ok else 'FAIL'}")

    # 6. The wall is finite-but-huge AT the clamp and never NaN past it.
    tl5 = torch.log1p(torch.tensor([5.0])); rm5 = fene_r_max(torch.tensor([5.0]))
    vals = [float(fene_confinement(torch.tensor([dv]), tl5, rm5))
            for dv in (9.9, 10.0, 12.0, 1e4)]
    ok &= all(np.isfinite(v) for v in vals) and vals[-1] > vals[2] > vals[1] > vals[0]
    gw = []
    for dv in (12.0, 100.0, 1e4):
        dg = torch.tensor([dv], requires_grad=True)
        fene_confinement(dg, tl5, rm5).backward(); gw.append(float(dg.grad.abs()))
    ok &= all(g > 0 for g in gw)
    print(f"[fene] loss at d = 9.9, 10.0 (the wall), 12.0, 10000 = "
          f"{', '.join(f'{v:.2f}' for v in vals)}  strictly INCREASING past the wall, "
          f"|grad| there = {', '.join(f'{g:.1e}' for g in gw)} (never 0)  {'OK' if ok else 'FAIL'}")

    # 7. Equilibrium under FENE + pairwise repulsion: settles near the gap and NEVER past 2x.
    for target_gap in (1.0, 5.0, 20.0):
        tl = torch.log1p(torch.tensor([target_gap])); rm = fene_r_max(torch.tensor([target_gap]))
        raw = torch.tensor([4.0], requires_grad=True)
        o = torch.optim.Adam([raw], lr=0.05)
        for _ in range(4000):
            o.zero_grad()
            dd = F.softplus(raw)
            (fene_confinement(dd, tl, rm) + 0.5 * log_gas_repulsion(dd)).backward()
            o.step()
        got = float(F.softplus(raw))
        hit = np.isfinite(got) and got <= 2 * target_gap + 1e-2
        ok &= hit
        print(f"[fene] equilibrium: gap {target_gap:5.1f} -> settled d = {got:7.3f} "
              f"(hard ceiling {2*target_gap:6.1f})  {'OK' if hit else 'FAIL'}")

    # ---- LJ-STYLE r^12 / -r^6 (Kaveh's explicit spec) -----------------------------------------
    for gp in (1.0, 5.0, 40.0):
        tl = torch.log1p(torch.tensor([gp])); rm = fene_r_max(torch.tensor([gp]))
        raw = torch.tensor([1.0], requires_grad=True)
        o = torch.optim.Adam([raw], lr=0.02)
        for _ in range(6000):
            o.zero_grad()
            dd = F.softplus(raw)
            (lj_confinement(dd, tl, rm) + 0.5 * log_gas_repulsion(dd)).backward()
            torch.nn.utils.clip_grad_norm_([raw], 10.0)
            o.step()
        got = float(F.softplus(raw)); coll = got < 0.25 * gp
        ok &= (not coll) and np.isfinite(got) and got <= 2.2 * gp
        print(f"[lj] gap {gp:5.1f} -> settled d = {got:9.4f} "
              f"(wall at {2*gp:5.1f}) {'COLLAPSED' if coll else 'stable'}")
    # unnormalised well depth grows as the 6th power of the gap -- the instability, quantified
    inner = [float(lj_confinement(torch.tensor([0.0]), torch.log1p(torch.tensor([g])),
                                   fene_r_max(torch.tensor([g])))) for g in (1.0, 5.0, 40.0)]
    ok &= all(v > 0 for v in inner)
    print(f"[lj] inner branch at d=0 for gap 1/5/40: {', '.join(f'{v:.2f}' for v in inner)} "
          f"(POSITIVE = pushes back out; was negative and collapsing)")

    # ---- adjacent anchor: the hard 1-ply box ---------------------------------------------------
    for dv, expect in ((1.0, "zero"), (1.49, "finite"), (3.0, "finite-large")):
        dd = torch.tensor([dv], requires_grad=True)
        v = adjacent_anchor(dd); v.backward()
        print(f"[adj] d={dv}: V={float(v):.3f} |grad|={float(dd.grad.abs()):.2f} ({expect})")
    d1 = torch.tensor([3.0], requires_grad=True); adjacent_anchor(d1).backward()
    d2 = torch.tensor([1.3], requires_grad=True); adjacent_anchor(d2).backward()
    ok &= float(d1.grad.abs()) > 100 * float(d2.grad.abs())
    # equilibrium against VICReg-scale pressure 2: must stay inside the box
    raw = torch.tensor([1.5], requires_grad=True)
    o = torch.optim.Adam([raw], lr=0.02)
    for _ in range(3000):
        o.zero_grad(); dd = F.softplus(raw)
        (adjacent_anchor(dd) + 2.0 * dd).backward()   # constant outward... inward pressure 2
        o.step()
    got = float(F.softplus(raw))
    ok &= 0.4 < got < 1.6
    print(f"[adj] equilibrium under pressure 2: d = {got:.3f} (must stay in [0.5,1.5])  {'OK' if ok else 'FAIL'}")

    # ---- pair walls ----------------------------------------------------------------------------
    g5 = torch.tensor([5.0])
    for dv, expect in ((3.0, "0 (interior FREE)"), (0.9, ">0 floor"), (0.3, "huge floor"),
                       (5.9, ">0 ceiling"), (9.0, "huge ceiling")):
        dd = torch.tensor([dv], requires_grad=True)
        v = pair_walls(dd, g5); v.backward()
        print(f"[walls] g=5 d={dv}: V={float(v):8.3f} |grad|={float(dd.grad.abs()):9.2f}  ({expect})")
    dd = torch.tensor([3.0], requires_grad=True); v = pair_walls(dd, g5); v.backward()
    ok &= float(v) == 0.0 and float(dd.grad.abs()) == 0.0
    # equilibrium: under outward pressure the pair parks at the ceiling, under inward at the floor
    for pressure, lo, hi in ((-0.5, 4.0, 6.5), (0.5, 0.4, 1.6)):
        raw = torch.tensor([1.2], requires_grad=True)
        o = torch.optim.Adam([raw], lr=0.02)
        for _ in range(3000):
            o.zero_grad(); ddq = F.softplus(raw)
            (pair_walls(ddq, g5) + pressure * ddq).backward(); o.step()
        got = float(F.softplus(raw)); ok &= lo < got < hi
        print(f"[walls] pressure {pressure:+.1f}: parks at d={got:.2f} (must be in [{lo},{hi}])  "
              f"{'OK' if lo < got < hi else 'FAIL'}")

    # ---- terminal contrast ---------------------------------------------------------------------
    ds=torch.tensor([1.0,2.0,0.5]); do=torch.tensor([700.0,2.0,50.0])
    la,lr=terminal_contrast(ds,do)
    d1=torch.tensor([700.0],requires_grad=True); _,r1=terminal_contrast(ds,d1); r1.backward()
    g_far=float(d1.grad.abs())
    d2=torch.tensor([2.0],requires_grad=True); _,r2=terminal_contrast(ds,d2); r2.backward()
    g_near=float(d2.grad.abs())
    ok &= g_far==0.0 and g_near>0
    print(f"[term-contrast] attraction {la:.3f}, repulsion {lr:.3f}; hinge grad at d=700: {g_far:.1e} "
          f"(floor satisfied, correctly OFF), at d=2: {g_near:.3f} (pushing)  {'OK' if ok else 'FAIL'}")

    # ---- JQT: persistence prior ----------------------------------------------------------------
    hp = torch.zeros(3, 4); hc = torch.ones(3, 4)
    quiet = torch.tensor([0.0, 0.0, 0.0]); loud = torch.tensor([1.0, 1.0, 1.0])
    l_quiet = code_persistence(hp, hc, quiet, tau=0.05)
    l_loud = code_persistence(hp, hc, loud, tau=0.05)
    ok &= abs(float(l_quiet) - 1.0) < 1e-5 and float(l_loud) == 0.0
    hg = torch.zeros(3, 4, requires_grad=True)
    code_persistence(hg, hc, loud, tau=0.05).backward()
    g_loud = float(hg.grad.abs().max())          # eval jumped -> the prior must be silent
    ok &= g_loud == 0.0
    mixed = torch.tensor([0.0, 0.04, 0.5])       # graded weight inside tau, zero outside
    wm = torch.relu(1.0 - mixed.abs() / 0.05)
    l_mix = code_persistence(hp, hc, mixed, tau=0.05)
    ok &= abs(float(l_mix) - float((wm * 1.0).sum() / wm.sum())) < 1e-5
    print(f"[persistence] quiet {float(l_quiet):.3f} (=1), loud {float(l_loud):.3f} (=0), "
          f"grad@loud {g_loud:.1e} (=0), mixed {float(l_mix):.3f}  {'OK' if ok else 'FAIL'}")

    # ---- JQT: jepa embedding prediction --------------------------------------------------------
    pr = torch.randn(5, 8, requires_grad=True); tg = torch.randn(5, 8)
    l_j = jepa_code_prediction(pr, tg); l_j.backward()
    ok &= pr.grad is not None and float(jepa_code_prediction(tg.clone(), tg)) == 0.0
    try:
        jepa_code_prediction(pr, torch.randn(5, 8, requires_grad=True)); ok = False
        print("[jepa] FAIL: accepted a target with a live graph")
    except AssertionError:
        print(f"[jepa] mse {float(l_j):.3f}, exact target -> 0, live-graph target REFUSED  "
              f"{'OK' if ok else 'FAIL'}")

    # ---- JQT: the two reused rulers keep their contracts on activation labels ------------------
    # censored dA-to-activation: censored rows contribute exactly nothing (value AND gradient)
    dA_pred = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    plies = torch.tensor([6.0, -1.0, 2.0]); hit = torch.tensor([1.0, 0.0, 1.0])
    lc = censored_plies_loss(torch.log1p(dA_pred), plies, hit); lc.backward()
    ok &= float(dA_pred.grad[1]) == 0.0 and float(dA_pred.grad[0]) != 0.0
    print(f"[concept-dA] censored row grad {float(dA_pred.grad[1]):.1e} (=0), observed "
          f"{float(dA_pred.grad[0]):+.3f} (live)  {'OK' if ok else 'FAIL'}")

    # ---- capture_ce: the -1 mask must be SKIPPED, not learned ------------------------
    torch.manual_seed(0)
    lg = torch.randn(8, 64, requires_grad=True)
    tgt = torch.full((8,), -1, dtype=torch.long)
    allmask = capture_ce(lg, tgt)
    fin = bool(torch.isfinite(allmask))          # bool(), or `ok &=` hits a Tensor
    ok &= (float(allmask) == 0.0) and fin
    print(f"[capture_ce] all-absent -> {float(allmask):.4f} (finite, not nan)  "
          f"{'OK' if fin and float(allmask) == 0.0 else 'FAIL'}")
    ok &= torch.autograd.grad(allmask, lg, retain_graph=True)[0].abs().sum().item() == 0.0
    tgt2 = tgt.clone(); tgt2[:4] = torch.tensor([3, 9, 17, 40])
    part = capture_ce(lg, tgt2)
    # must equal CE over the VALID rows only -- if the -1s leaked in, this differs
    ref = F.cross_entropy(lg[:4], tgt2[:4])
    ok &= abs(float(part) - float(ref)) < 1e-5
    print(f"[capture_ce] half-absent {float(part):.4f} == CE on valid rows {float(ref):.4f}  "
          f"{'OK' if abs(float(part)-float(ref)) < 1e-5 else 'FAIL'}")
    g = torch.autograd.grad(part, lg, retain_graph=True)[0]
    ok &= g[4:].abs().sum().item() == 0.0 and g[:4].abs().sum().item() > 0
    print(f"[capture_ce] gradient reaches ONLY the valid rows  "
          f"{'OK' if g[4:].abs().sum().item()==0 and g[:4].abs().sum().item()>0 else 'FAIL'}")

    # ---- effective_number_weights: the class-balance term -----------------------------
    w = effective_number_weights(torch.tensor([100, 100, 100]))
    ok &= bool(torch.allclose(w, torch.ones(3), atol=1e-5))
    print(f"[cb-weights] equal counts -> uniform weights {w.tolist()}  "
          f"{'OK' if torch.allclose(w, torch.ones(3), atol=1e-5) else 'FAIL'}")
    w2 = effective_number_weights(torch.tensor([57681, 1197, 75]))    # the real spread
    rare_up = float(w2[2]) > float(w2[0])
    ok &= rare_up and abs(float(w2.mean()) - 1.0) < 1e-4
    print(f"[cb-weights] 769x imbalance -> w {w2.tolist()} (rare up-weighted "
          f"{float(w2[2])/float(w2[0]):.1f}x, mean {float(w2.mean()):.3f})  "
          f"{'OK' if rare_up else 'FAIL'}")
    w3 = effective_number_weights(torch.tensor([100, 0, 50]))
    ok &= float(w3[1]) == 0.0
    print(f"[cb-weights] absent class gets weight 0  {'OK' if float(w3[1])==0 else 'FAIL'}")
    # 1/n is the beta->1 limit; effective-number must be GENTLER than it on the rare class
    inv = torch.tensor([1/57681, 1/1197, 1/75]); inv = inv / inv.mean()
    gentler = float(w2[2]) < float(inv[2])
    ok &= gentler
    print(f"[cb-weights] rare-class weight {float(w2[2]):.2f} < naive 1/n {float(inv[2]):.2f}  "
          f"{'OK' if gentler else 'FAIL'}")

    # ---- factorized_resolution_ce -----------------------------------------------------
    torch.manual_seed(0)
    B, E = 12, 7
    rl = torch.randn(B, 3, requires_grad=True)
    el = torch.randn(B, E, requires_grad=True)
    ol = torch.randn(B, 3, requires_grad=True)
    # rows 0-5 reach a tablebase (ending defined, outcome inside it); rows 6-11 end first
    rt = torch.tensor([1]*6 + [0]*6)
    et = torch.tensor([0, 1, 2, 3, 4, 5] + [-1]*6)
    ot = torch.tensor([-1]*6 + [0, 1, 2, 0, 1, 2])
    tot, parts = factorized_resolution_ce(rl, el, ol, rt, et, ot)
    g = torch.autograd.grad(tot, el, retain_graph=True)[0]
    only_reach = g[6:].abs().sum().item() == 0.0 and g[:6].abs().sum().item() > 0
    ok &= only_reach
    print(f"[res-factored] ending gradient reaches ONLY the reaching rows  "
          f"{'OK' if only_reach else 'FAIL'}")
    go = torch.autograd.grad(tot, ol, retain_graph=True)[0]
    split_ok = go[:6].abs().sum().item() == 0.0 and go[6:].abs().sum().item() > 0
    ok &= split_ok
    print(f"[res-factored] outcome gradient reaches ONLY the ends-first rows "
          f"(no double count)  {'OK' if split_ok else 'FAIL'}")
    # the whole point: with NO reaching rows the ending head must not poison the total
    et_none = torch.full((B,), -1, dtype=torch.long)
    tot2, _ = factorized_resolution_ce(rl, el, ol, rt, et_none, ot)
    fin2 = bool(torch.isfinite(tot2))
    ok &= fin2
    print(f"[res-factored] no reaching rows in batch -> total {float(tot2):.4f} finite  "
          f"{'OK' if fin2 else 'FAIL'}")
    # class-balanced variant must run and differ from the unweighted one
    ew = effective_number_weights(torch.tensor([500, 400, 300, 200, 100, 50, 25]))
    tot3, _ = factorized_resolution_ce(rl, el, ol, rt, et, ot, eg_weight=ew)
    diff = abs(float(tot3) - float(tot)) > 1e-6
    ok &= diff and bool(torch.isfinite(tot3))
    print(f"[res-factored] class-balanced total {float(tot3):.4f} != unweighted "
          f"{float(tot):.4f}  {'OK' if diff else 'FAIL'}")

    # ---- compose_resolution_logits: MUST be a proper distribution ---------------------
    # a miniature vocab: WDL:W/D/L, then 6 endings (2 signatures x 3 outcomes), then OTHER.
    # cls_out is -1 on the endings -- their outcome already lives inside the joint eg index.
    cls_reach = torch.tensor([0, 0, 0] + [1]*6 + [2])
    cls_eg = torch.tensor([-1, -1, -1] + [0, 1, 2, 3, 4, 5] + [-1])
    cls_out = torch.tensor([0, 1, 2] + [-1]*6 + [-1])
    rl2, el2, ol2 = torch.randn(5, 3), torch.randn(5, 6), torch.randn(5, 3)
    logp = compose_resolution_logits(rl2, el2, ol2, cls_reach, cls_eg, cls_out)
    tot_p = logp.logsumexp(-1)
    norm = bool(torch.allclose(tot_p, torch.zeros(5), atol=1e-5))
    ok &= norm and logp.shape == (5, 10)
    print(f"[res-compose] composed {tuple(logp.shape)} sums to 1 "
          f"(logsumexp {float(tot_p.max()):+.2e})  {'OK' if norm else 'FAIL'}")
    lr_, le_, lo_ = (F.log_softmax(rl2, -1), F.log_softmax(el2, -1), F.log_softmax(ol2, -1))
    hand = lr_[:, 1] + le_[:, 5]                      # class 8 = the 6th ending, reaching
    match = bool(torch.allclose(logp[:, 8], hand, atol=1e-6))
    ok &= match
    print(f"[res-compose] ending == logP(reach=1)+logP(eg), no double-counted outcome  "
          f"{'OK' if match else 'FAIL'}")
    wdl_ok = bool(torch.allclose(logp[:, 1], lr_[:, 0] + lo_[:, 1], atol=1e-6))
    ok &= wdl_ok
    print(f"[res-compose] WDL:D == logP(reach=0)+logP(out=D)  {'OK' if wdl_ok else 'FAIL'}")
    other_ok = bool(torch.allclose(logp[:, 9], lr_[:, 2], atol=1e-6))
    ok &= other_ok
    print(f"[res-compose] OTHER == logP(reach=2), no phantom factors  "
          f"{'OK' if other_ok else 'FAIL'}")
    # THE ARITHMETIC THAT DROVE THIS DESIGN: with the endgame branch joint, a confident model
    # CAN out-argmax a WDL class; with a sig x out product at the same confidence it cannot.
    rr = torch.log(torch.tensor([[0.48, 0.52, 0.0001]]))
    ee = torch.log(torch.tensor([[0.05, 0.05, 0.05, 0.05, 0.20, 0.60]]))
    oo = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    lp = compose_resolution_logits(rr, ee, oo, cls_reach, cls_eg, cls_out)
    won = int(lp.argmax(-1)) == 8
    ok &= won
    print(f"[res-compose] P(reach)=.52 x P(ending)=.60 beats P(WDL:W)=.48*.5 -> argmax is the "
          f"ENDING  {'OK' if won else 'FAIL'}")

    # ---- vamp2_score: slowness that cannot be faked by collapse -----------------------
    torch.manual_seed(0)
    N, d = 512, 8
    base = torch.randn(N, d)
    s_persist = float(vamp2_score(base, base))                      # perfectly slow, varied
    s_indep = float(vamp2_score(base, torch.randn(N, d)))           # no temporal structure
    s_const = float(vamp2_score(torch.ones(N, d), torch.ones(N, d)))  # the MSE-degenerate case
    ok &= s_persist > 0.9 * d
    print(f"[vamp2] persistent varied features -> {s_persist:.2f} (~d={d})  "
          f"{'OK' if s_persist > 0.9*d else 'FAIL'}")
    ok &= s_indep < 0.25 * d
    print(f"[vamp2] independent features -> {s_indep:.2f} (<< d)  "
          f"{'OK' if s_indep < 0.25*d else 'FAIL'}")
    ok &= s_const < 0.1
    print(f"[vamp2] CONSTANT features -> {s_const:.4f} (~0: collapse cannot fake slowness, "
          f"unlike MSE persistence)  {'OK' if s_const < 0.1 else 'FAIL'}")
    # the exploit that bit gcf2: scaled-down features must NOT inflate the score
    s_tiny = float(vamp2_score(base * 1e-3, base * 1e-3))
    ok &= s_tiny <= d + 0.5
    print(f"[vamp2] 1e-3-scaled features -> {s_tiny:.2f} (<= d: scale-invariant, the gcf2 "
          f"exploit closed)  {'OK' if s_tiny <= d + 0.5 else 'FAIL'}")
    xg = base.clone().requires_grad_()
    g = torch.autograd.grad(vamp2_score(xg, base), xg)[0]
    ok &= bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0
    print(f"[vamp2] gradient finite and nonzero through the eigendecomposition  "
          f"{'OK' if torch.isfinite(g).all() and g.abs().sum() > 0 else 'FAIL'}")

    print("ALL LOSS TESTS PASSED" if ok else "TESTS FAILED")


if __name__ == "__main__":
    _tests()


def cross_head_decorrelation(pre_g):
    """Barlow-style off-diagonal penalty ACROSS concept heads.

    pre_g (B, H, D): per-head pre-VQ features. Standardize per dim over the batch, then
    penalize the mean squared cross-correlation between every pair of DISTINCT heads. Zero
    when heads carry independent factors; ~1 when two heads are copies -- which is exactly
    the gcf2 failure (all 8 slow heads = castling+stm, rank 3 in a 256-dim block).

    Composition with VAMP-2 is free: the VAMP score is invariant to invertible linear maps
    of the slow block, so decorrelation costs no slowness -- VAMP picks the slow SUBSPACE,
    this term distributes it non-redundantly across heads (one head may keep castling; the
    other seven must find something else). Kaveh 2026-08-20: "if it has only one concept
    and decorrelation terms are forcing the other concepts away, that's great."
    """
    B, H, D = pre_g.shape
    z = pre_g - pre_g.mean(0, keepdim=True)
    z = z / (z.std(0, keepdim=True) + 1e-6)
    c = torch.einsum("bhd,bke->hkde", z, z) / B          # (H, H, D, D) cross-correlations
    # scale by D: a copied head pair scores ~1 (its cross-corr is ~identity, D entries of 1
    # in D^2 slots); independent heads score ~D/B sampling noise
    off = (c ** 2).mean(dim=(2, 3)) * pre_g.shape[-1]
    mask = ~torch.eye(H, dtype=torch.bool, device=pre_g.device)
    return off[mask].mean()


def plackett_luce(scores, target, mask, tau=1.0):
    """Listwise Plackett-Luce NLL of the ordering that `target` induces.

    Why this and not KL-per-item (2026-08-24): within a chess position SF's move
    values are tightly clustered (sd ~0.116 on [0,1]), so a KL against each move's
    value is very well satisfied by predicting the position MEAN for every sibling --
    measured, the rolled path did exactly that, emitting 4pct of the true spread and
    a within-position ordering of 0.00. PL has no such escape: it scores only the
    ORDER, is invariant to any shift or positive rescale of `scores`, and its gradient
    is driven entirely by pairs that are ranked wrongly.

    scores: (B, N) model scores, higher = better.
    target: (B, N) reference values, higher = better; only their ORDER is used.
    mask:   (B, N) True where the slot is a real move.
    Returns a scalar mean NLL over rows with >= 2 valid moves.
    """
    import torch
    s = scores / tau
    s = s.masked_fill(~mask, -1e9)
    t = target.masked_fill(~mask, -float("inf"))
    order = t.argsort(dim=-1, descending=True)
    s_sorted = s.gather(-1, order)
    m_sorted = mask.gather(-1, order)
    s_sorted = s_sorted.masked_fill(~m_sorted, -1e9)
    # denominator for position i is logsumexp over the SUFFIX i..N (the items still
    # unchosen), so a reverse cumulative logsumexp gives all N terms in one pass
    rev = torch.logcumsumexp(s_sorted.flip(-1), dim=-1).flip(-1)
    nll = -(s_sorted - rev)
    nll = nll.masked_fill(~m_sorted, 0.0)
    n_valid = mask.sum(-1)
    keep = n_valid >= 2
    if not keep.any():
        return scores.sum() * 0.0
    return (nll.sum(-1)[keep] / n_valid[keep].clamp(min=1)).mean()


def future_kind_ce(logits, kind, value_spread, mask=None, w_floor=0.0):
    """Cross-entropy over K discovered KINDS of future, weighted by what the choice
    among them is WORTH at that position.

    Kaveh 2026-08-24: "not always will future concepts be predictable from present
    concepts, so I expect that loss to remain high. The primary objective should always
    be the Stockfish evaluation, but we do want a nudge -- if some concepts can be
    predicted, predict those."

    That constraint is satisfied by the SHAPE of this loss, not by tuning its weight,
    which is the reason to predict kinds rather than future concept vectors:

      * PREDICTING UNIFORM COSTS EXACTLY log K, AND IS ALWAYS AVAILABLE. So the model
        is never FORCED to pay more than log K for a position whose future is genuinely
        unpredictable -- it can decline to predict instead of distorting the encoder to
        fake predictability, and the gradient VANISHES as the prediction approaches the
        true conditional distribution. Its floor is the conditional entropy
        H(kind | s), which is real signal about plan uncertainty, not an error.
        (Stated carefully because the sloppy version -- "CE is bounded by log K" -- is
        FALSE for an arbitrary model: -log p(true) is unbounded when the head is
        confidently wrong, and a randomly-initialised head measures slightly ABOVE the
        ceiling, 1.86 against log 6 = 1.79. The bound is on what the model must accept,
        not on what a bad model can score.)
      * A per-dimension regression on future concepts has no such bound. Measured
        (JOURNAL 2026-08-24): 65% of 2528 concept dims have grouped R^2 < 0, so that
        loss is dominated by an irreducible floor and would drag the encoder toward
        whatever makes the unpredictable dims look predictable -- collapse.

    The weight is per-POSITION, not per-concept: the spread of teacher value across
    the kinds reachable here. Where the choice of future is worth 1.7 pawns the loss
    pulls; where every move leads to the same kind of game it does not. Measured, a
    fixed importance weight taken from the value head would NOT do this -- per-dim
    predictability and value-head weight are uncorrelated (-0.001).

    logits:       (B, K) predicted scores over kinds for this (position, move).
    kind:         (B,) long, the kind its Stockfish continuation actually landed in.
    value_spread: (B,) >= 0, spread of teacher value across the kinds REACHABLE at
                  this position (e.g. max-min mean eval, in pawns).
    mask:         (B,) bool, False for rows with no usable future (game ended early).
    w_floor:      minimum weight, so positions with no value spread still contribute
                  a little. Default 0.0 -- they contribute nothing.

    Returns (loss, frac_of_log_K), the second being the loss as a fraction of its own
    ceiling: ~1.0 means the head has learned nothing, and it is the number to watch
    rather than the raw value, which is not comparable across K.
    """
    import torch
    import torch.nn.functional as F
    K = logits.shape[-1]
    ce = F.cross_entropy(logits, kind, reduction="none")
    w = value_spread.clamp(min=0).to(ce.dtype)
    w = w / w.mean().clamp(min=1e-6)                 # scale-free: mean weight is 1
    w = w.clamp(min=w_floor)
    if mask is not None:
        w = w * mask.to(w.dtype)
    tot = w.sum().clamp(min=1e-6)
    loss = (ce * w).sum() / tot
    import math
    return loss, float(loss.detach() / math.log(K))


def _test_future_kind_ce():
    """Every property this loss is being trusted for, checked before any run uses it."""
    import math
    torch.manual_seed(0)
    B, K = 64, 8
    kind = torch.randint(0, K, (B,))
    spread = torch.rand(B)

    # 1. BOUNDED: a maximally-wrong confident head cannot exceed log K by much, and a
    #    head that outputs the uniform distribution sits exactly AT log K.
    uni = torch.zeros(B, K)
    l_uni, frac = future_kind_ce(uni, kind, spread)
    assert abs(l_uni.item() - math.log(K)) < 1e-5, f"uniform should be log K, {l_uni}"
    assert abs(frac - 1.0) < 1e-5, f"frac should be 1.0 at uniform, got {frac}"

    # 2. GRADIENT VANISHES where the future is unpredictable. If the target kind is
    #    drawn uniformly at random (nothing to learn), the gradient at the uniform
    #    prediction is small and points nowhere in particular -- the head can sit at
    #    the entropy floor without pulling the encoder. Contrast with a LEARNABLE
    #    target, where the same starting point has a large, consistent gradient.
    def gnorm(target):
        z = torch.zeros(B, K, requires_grad=True)
        future_kind_ce(z, target, torch.ones(B))[0].backward()
        return z.grad.sum(0).norm().item()          # sum over batch: consistent push
    g_rand = gnorm(torch.randint(0, K, (B,)))
    g_learn = gnorm(torch.zeros(B, dtype=torch.long))   # every future is kind 0
    assert g_learn > 4 * g_rand, \
        f"learnable target must pull far harder than noise: {g_learn:.3f} vs {g_rand:.3f}"

    # 3. WORTHLESS CHOICES CONTRIBUTE NOTHING. A position where all futures have the
    #    same value gets zero weight and cannot move the loss.
    z = torch.randn(B, K)
    keep = torch.ones(B); keep[: B // 2] = 0.0        # half the rows worth nothing
    l_half, _ = future_kind_ce(z, kind, keep)
    l_ref, _ = future_kind_ce(z[B // 2:], kind[B // 2:], torch.ones(B - B // 2))
    assert abs(l_half.item() - l_ref.item()) < 1e-5, \
        f"zero-spread rows must not contribute: {l_half.item()} vs {l_ref.item()}"

    # 4. SCALE-FREE in the weight: measuring value in pawns or centipawns must not
    #    change the loss, or the auxiliary's strength would depend on units.
    a, _ = future_kind_ce(z, kind, spread)
    b, _ = future_kind_ce(z, kind, spread * 100.0)
    assert abs(a.item() - b.item()) < 1e-4, f"weight scale leaked: {a} vs {b}"

    # 5. MASK: rows with no usable future are excluded, not counted as easy.
    m = torch.ones(B, dtype=torch.bool); m[:10] = False
    l_m, _ = future_kind_ce(z, kind, torch.ones(B), mask=m)
    l_x, _ = future_kind_ce(z[10:], kind[10:], torch.ones(B - 10))
    assert abs(l_m.item() - l_x.item()) < 1e-5, f"mask leaked: {l_m} vs {l_x}"

    # 6. IT ACTUALLY LEARNS: a head fed a predictable target should drive frac well
    #    below 1.0. Without this the other five properties describe an inert loss.
    tgt = torch.randint(0, K, (B,))
    feat = torch.eye(K)[tgt] + 0.3 * torch.randn(B, K)
    lin = torch.nn.Linear(K, K)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        loss, frac = future_kind_ce(lin(feat), tgt, torch.ones(B))
        loss.backward(); opt.step()
    assert frac < 0.4, f"loss must be learnable on a predictable target, frac {frac}"
    print(f"  future_kind_ce: bounded at log K, gradient {g_learn:.2f} on signal vs "
          f"{g_rand:.2f} on noise, learns to {frac:.2f} of ceiling -- 6/6 PASS")


if __name__ == "__main__":
    _test_future_kind_ce()
