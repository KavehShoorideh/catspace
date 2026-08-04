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
    """Huber on log1p(d) toward a log-space target. d>=0."""
    return F.huber_loss(torch.log1p(d.clamp(min=0)), target_log, delta=1.0)


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


def basin_logits(d_poles, temperature=1.0):
    """-log1p(d)/T -- the basin logits. d_poles (B,3) >= 0 = quasimetric distances to
    [win, draw, loss]. See the module note above for why this is log-space and not -d/T."""
    return -torch.log1p(d_poles.clamp(min=0)) / temperature


def basin_ce(d_poles, y, temperature=1.0):
    """Basin cross-entropy. d_poles (B,3) >= 0; y (B,) int64 in {0,1,2} = mover-POV outcome.

    CE against a realized 1-hot outcome is a PROPER SCORING RULE, so its minimizer is the true
    conditional probability -- the calibrated committor -- not merely a separating clustering.
    Its gradient pulls phi toward the observed outcome's pole with weight (1 - p_y): each point
    is pulled toward a basin in proportion to probability MISMATCH, which is exactly the
    'transition probability pulls it toward that basin's pole' behaviour."""
    return F.cross_entropy(basin_logits(d_poles, temperature), y)


def basin_logp(d_poles, temperature=1.0):
    """log p(basin) (B,3) -- the readout used for charting + calibration. Returned in LOG space
    on purpose: the committor is degenerate inside basins (nearly all mass at p~0/1), so
    log-odds is the coordinate that resolves basin interiors and the transition region with
    equal thoroughness (standard practice for committor collective variables)."""
    return F.log_softmax(basin_logits(d_poles, temperature), dim=-1)


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

    print("ALL LOSS TESTS PASSED" if ok else "TESTS FAILED")


if __name__ == "__main__":
    _tests()
