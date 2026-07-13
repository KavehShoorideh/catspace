# Uncertainty-driven planning — design spec

Status: **approved-in-principle 2026-07-13** (Kaveh's reframe + B/C direction).
Supersedes the *axis* of TWO_HORIZON_DESIGN.md (ply/depth) while reusing its
machinery as a baseline. Authored on Opus.

## The reframe (why depth is the wrong axis)

Tactical vs positional is not temporal distance — it's **local sharpness of the
value landscape** (curvature). A *sharp* position: one tempo flips the result, so
you cannot prune. A *smooth* position: many move-orders converge, so a coarse
estimate suffices. A forcing tactical line can run 20 ply deep; a position can be
quiet at ply 2. So any handover keyed on ply is mis-specified — and that
mis-specification is why the two ply-stratified heads fight: at a fixed horizon
you force one scalar to be a sharp estimate and a smooth one at once.

Our own evidence agrees: the node-budget sweep was non-monotonic (200=.062
400=.100 800=.062) — depth is not the lever.

**Fix:** drive the tactical/positional handover on **uncertainty the model
emits**, not on depth. Two kinds map onto the two desiderata:
- **Aleatoric** (irreducible branch volatility — "this line is genuinely sharp")
  → *don't prune where spread is high*.
- **Epistemic** (model hasn't mapped this region — rare/deep/OOD) → grows with
  depth as a *consequence*, not by construction.

## Options (all scored on the sharpness benchmark before touching search)

- **A — head-disagreement gate** (near-free): where the two ply-heads disagree on
  the best move, you're in the sharp regime. Validates the sharpness hypothesis
  using the two-horizon run we already have. Two point estimates, no new training.
- **B — distributional reachability head** (the signal producer): predict the
  distance-to-goal as a **distribution**, not a mean. Spread = the regime signal.
- **C — uncertainty-gated expansion** (the signal consumer): replace the depth
  schedule with per-node quiescence-by-sharpness — high spread → don't prune,
  expand checks/captures locally; low spread → allow far subgoal jumps. This is
  classical quiescence + singular extensions, driven by B's spread.
- **D — γ-ensemble** (optional): value at a spectrum of discount factors; read off
  which horizon the decision is sensitive to. Still horizon-parameterized, so it
  half-inherits the wrong-axis problem. Lower priority.

**Chosen path: B produces, C consumes.** A is the cheap validator run first.

## FOSS-first check (Kaveh's standing rule: don't rebuild what good FOSS provides)

Before building B/C, the existing open-source homes for each piece. Kaveh's
balancing clause (2026-07-13): FOSS-first UNLESS it's hard to incorporate or
distracts from the research core AND the thing is simple to hand-roll.
- **Quasimetric head** — `torchqmet` (Tongzhou Wang, the IQE/MRN/PQE reference
  impl) exists, but per the balancing clause we KEEP our hand-rolled
  `score = r - d`: it's ~5 lines, it's core research, and adopting the lib adds a
  dependency + changes numerics (forcing a re-baseline) for no clear win. (Moving
  to IQE for its greater expressivity stays available as a deliberate RESEARCH
  choice later — not a "don't reinvent" obligation.)
- **Distributional head** — `torch.distributions.Categorical` + built-in
  `cross_entropy` (and pinball loss for the quantile fallback). No bespoke
  distributional-RL framework needed for a softmax-over-bins head.
- **Already FOSS, kept**: python-chess (board/legal moves — the search's substrate),
  `chess.syzygy` (tablebase probing for the sharpness/calibration ground truth),
  scipy.stats (Spearman/Wilcoxon), torch. What we genuinely build is only the
  research composition: the FB/quasimetric loss, the chess readout, the gated search.

## B — distribution choice: CATEGORICAL (Kaveh, 2026-07-13)

- **Categorical, not Gaussian.** Chess distance-to-goal is *bounded and
  integer-valued* (tablebase DTM/DTZ caps out), so a fixed set of distance bins
  has no "where do the edges go for an unbounded continuous quantity" problem.
  Predict a softmax over distance bins; train by cross-entropy against the
  observed ply-gap's bin. **Gaussian is explicitly rejected**: bimodality ("either
  3 ply or 30 ply depending on the line") is *precisely* the tactical signal, and
  a Gaussian cannot represent it. **Quantile regression** is the safe fallback if
  we don't want to commit to a fixed bin range.
- **Axiom preservation (the load-bearing constraint).** Whatever we predict, the
  **mean/point-estimate used as the PLANNING DISTANCE must still satisfy the IQE
  quasimetric axioms** (triangle inequality, identity) — otherwise multi-hop plans
  stop composing. The **spread** (histogram entropy / inter-quantile range) rides
  on top as an *auxiliary* output and does NOT need to satisfy the axioms — it's a
  regime signal, not a distance.
  - v1 decision (safe): keep the existing IQE/quasimetric `d(s,g)` as the planning
    distance (axioms free by construction), and add the categorical head purely
    for its **spread**. Its mean need not be the planning distance.
  - v2 (the open problem): make the categorical's mean the axiom-respecting
    planning distance — measure its triangle-violation on the fitness probe; adopt
    only if violations stay ~0.

## C — uncertainty-gated expansion

Feed B's per-node spread into the search's expansion/pruning:
- high spread → widen the beam / quiescence-extend (expand all checks & captures),
- low spread → narrow / allow far jumps (AdaSubS-style: take the farthest subgoal
  that still verifies; high spread makes far subgoals fail verification and fall
  back to short dense ones — the behavior emerges endogenously, per-node, with no
  depth schedule).

## Rigorous-testing infrastructure (the actual ask)

1. **Sharpness benchmark** (`experiments/sharpness_bench.py`, BUILT): exact
   tablebase ground-truth sharpness (DTZ progress-cost curvature over legal
   moves), scores any uncertainty signal by rank-correlation (ρ) vs truth.
   Baseline: incumbent point-head score-spread ρ=+0.14. Each of A/B/D must beat it.
2. **Common signal interface**: every candidate emits `(planning_distance,
   uncertainty)` per position, so the benchmark and the gated search consume them
   uniformly.
3. **Gated-search knob** on FBSearchPolicy: expansion width/quiescence as a
   function of the uncertainty input, so C is A/B-agnostic.
4. **Reuse** the existing paired-stats arena (KRRvKBP + ACPL, EValueTest +
   bootstrap) and the fitness probe for the final play gate.

## Sequence

1. Finish the ply-stratified two-horizon run → baseline + **run A** (head-
   disagreement) on the sharpness benchmark: does it beat ρ=+0.14? (validates the
   whole reframe cheaply).
2. Build **B** (categorical head, spread output, quasimetric μ kept as distance) →
   score its spread ρ on the benchmark.
3. Build **C** (gated search consuming the best signal) → play gate vs incumbent.
4. Every step: pre-registered criterion, matched comparison, journal + commit.
