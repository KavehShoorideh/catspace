# Distance should price risk: certainty-weighted quasimetrics for planning

*One topic from the catspace project: the single reframe that produced our
first confirmed play improvement — redefining "close" so that it accounts for
whether you'll actually get there.*

## The bug in "distance to goal"

Goal-conditioned planners learn some d(s, g): how far state s is from goal g.
Nearly every way of learning it — shortest-path supervision, contrastive
objectives on observed trajectories, temporal-difference distances — bakes in
**min semantics**: d is small if *there exists* a short path. That is the
right semantics for an infallible executor and the wrong one for everything
else.

Chess makes the failure concrete. Consider a guaranteed mate-in-15 — quiet,
forcing, any reasonable move order works — versus a mate-in-7 that threads a
single needle you'll find maybe a third of the time. Min-semantics distance
says the mate-in-7 is *closer*. A planner steering by that distance walks
into needle positions, misses the needle, and bleeds the win. We didn't just
theorize this; we measured it: our learned field's distance was
**anti-correlated** with empirical conversion probability on held-out states
(Spearman −0.099, CI [−0.175, −0.027]) — the geometry was systematically
optimistic exactly where positions were hardest to hold.

## The fix: put −ln P in the metric

Define

    d(s, g) = plies(s → g) + λ · (−ln P(reach g from s))

where P is the probability that *this agent, with its actual fallibility*,
converts s into g. Two properties make this more than a heuristic:

- **It's still a quasimetric.** Probabilities multiply along a path, so
  −ln P adds; adding it to a ply count preserves subadditivity (the triangle
  inequality). Plans still compose: pin, then win the piece, then mate.
- **It degrades gracefully to the classical case.** For a forced line
  (P = 1) the risk term vanishes and d is pure move count. Certainty and
  hops live in one currency, with λ the exchange rate.

Where does P come from? Not from an oracle — from **the agent's own play**.
We roll out the current policy (with exploration noise ε) from visited
states, aggregate per-state conversion frequencies P̂, and distill
`plies + λ(−ln P̂)` back into the geometry. The estimator needs the standard
hygiene: a Laplace floor so P̂ = 0 doesn't produce infinities, visit-count
thresholds, and held-out splits — the targets are statistics, and they will
happily be memorized if you let them.

## What the decomposition taught us

Estimating P̂ at several noise levels ε lets you regress −ln P̂ on ε per
state and identify two separate quantities: an **existence** intercept (is
there a path at all, at ε → 0) and a **sharpness** slope S (how fast the
position punishes fallibility). The measured surprise: S is essentially
uncorrelated with distance-to-mate (Spearman ≈ +0.04 over 4,373 states).
**Risk does not accumulate with path length — it concentrates in
bottlenecks.** A long smooth technique is safe under time pressure; a short
sharp one is not. Human intuition says this ("simple mate-in-15 under time
pressure: fine; complex mate-in-5: I'll blunder"); the rollout statistics
confirm it, which also means any *constant* λ fusing plies and risk is
structurally wrong at the tails, and sharpness ultimately deserves its own
channel.

One cautionary result on that last point: our first attempt to split the
channels — purify the geometry back to plies, train a separate frozen
sharpness head, apply risk only when reading the field out — was **falsified
at play** (−0.24 conversion, CI-real). The certainty term earns its keep *in*
the geometry the search descends, where it reshapes every intermediate
comparison, not as a post-hoc correction at the root. Decompose the signal
for analysis; don't decompose the metric until each channel is strong enough
to stand alone.

## Did it work?

Yes — with two conditions that took us weeks to isolate. The certainty
distill only produced a confirmed play gain once (a) the P̂ table was built
from **on-distribution own-play states** at sufficient scale (~10k states;
below that, null), and (b) evaluation ran at a search budget deep enough that
field quality, not search depth, was the binding constraint. Under those
conditions: conversion against a tablebase-optimal defender 0.400 → 0.608
(CI [+0.108, +0.317], e = 185) on a pre-registered, never-touched position
set — and the approach then transferred out of the toy by training the
certainty term directly into the full-board objective, where it beat the
prior incumbent head-to-head decisively (composed e = 539).

## Portable summary

If your planner executes imperfectly — a robot with actuation noise, an
agent with a fallible low-level controller, a search with a budget — then
"how far" and "will I make it" are not separate concerns to be traded off at
decision time. Fold the log-probability of successful traversal into the
distance itself: it keeps the metric axioms, it's estimable from your own
rollouts with no oracle, and it converts "optimistic geodesics" into routes
you can actually hold. Then measure your sharpness separately — it's not the
same axis as length, and pretending it is will cost you at exactly the
states that decide outcomes.
