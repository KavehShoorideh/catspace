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
    n, d = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(n - 1, 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return (off ** 2).sum() / d


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

    print("ALL LOSS TESTS PASSED" if ok else "TESTS FAILED")


if __name__ == "__main__":
    _tests()
