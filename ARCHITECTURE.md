# catspace — Architecture

2026-07-18. The working architecture document: motivation, approach, the
settled method stack with its chess-grounded reasons and its mathematics,
implementation and health machinery, results to date, and open work.
Supersedes ARCHITECTURE_REVIEW.md (2026-07-17 literature-verification pass)
and the old package-layout ARCHITECTURE.md; code-level verification lives in
MATH_AUDIT.md; day-by-day evidence in JOURNAL.md. Soft limit 8 pages.

## 1. Motivation

We want to understand what it takes for an agent to **plan** — to pick
destinations, decompose them into subgoals, commit to a line, notice when it
is refuted, and know when to fight, simplify, or resign — rather than merely
search well. Chess is the laboratory because it is the rare domain that is
simultaneously *rich enough* to demand real planning and *verifiable enough*
to grade it: the rules define exact terminal surfaces, small endgames have
tablebase oracles, and a century of human play at every skill level is on
record.

The hypotheses under test:

- **H1 (geometry):** the game's reachability structure can be *learned* as a
  geometry — an asymmetric distance between positions — and that geometry is
  useful for planning (subgoal selection, plan composition) in a way flat
  similarity is not.
- **H2 (value is derived):** "how good is this position" is not a primitive.
  It is the probability of reaching each outcome surface under a **play
  measure** — a model of what these players would actually do. Change the
  players and the value changes; the geometry does not.
- **H3 (planning is metareasoning):** a planner is an agent whose *internal*
  actions are computations (probe a region with search, adopt a plan) and
  whose *game* actions are moves, draw offers, and resignation. Search is a
  tool the planner calls, not the planner itself.
- **H4 (fallibility is the bridge):** the gap between best-case geometry and
  realizable value is governed by the probability that the right moves
  actually get played — ours and the opponent's. This "coherence" determines
  how deep a plan can be trusted and where compute should be spent.

The project is for fun, and for transfer: the same structure — irreversible
actions, resource budgets, an adversarial or uncertain environment,
categorical outcomes — is the shape of agentic planning and robotics
problems generally.

## 2. Main approach

Treat the game as a directed graph and learn two independent objects over it:

1. **Reachability geometry.** A quasimetric d(s, g): the best-case number of
   plies to reach g from s, for *any* pair — learned from position pairs,
   with provably-unreachable pairs pushed apart. This is the map.
2. **The play measure μ(move | s, ω).** The probability each legal move
   actually gets played, conditioned on the players and their state ω —
   skill, and exogenous factors such as time pressure or tilt. Perfect play
   is the degenerate measure; human play spreads. This is the traffic model.

Everything else is computed from these two. The **value** of a position is
the probability of hitting the win surface before the draw/loss surfaces when
μ drives the dynamics over the graph — a *committor*, in the language of
statistical mechanics, with the outcome surfaces as absorbing boundaries.
Goals are **surfaces and regions**, never single positions: "mate" is a
boundary you cross (touchdown semantics), "rook endgame up a pawn" is a
region you enter. **Coherence** — how far ahead the best-case map can be
trusted — is the probability the intended corridor survives both sides'
choices, and it controls search depth and backup trust. The **planner** sits
on top, choosing regions to probe and plans to commit to, with search as its
probing instrument.

## 3. Environment and resources

- **Claude LLM loop.** The research is conducted by an LLM agent (Claude)
  under human direction: designs, implements, runs, and journals
  experiments; runs autonomously overnight with scheduled check-ins;
  operates under standing lab rules (§4b).
- **Local laptop.** Apple-silicon machine (MPS backend). Models are small
  (~9M parameters); a 40k-step training run takes a few hours; the GPU is
  used by one job at a time.
- **Experimentation harness.** Deterministic paired A/B playouts against a
  tablebase-optimal defender, fixed start sets, bootstrap confidence
  intervals, e-value sequential tests, a leakage audit, and a running
  JOURNAL.md where no number enters without a printed script verdict.
- **Toy scenarios.** The MVP task: convert KRRvKBP winning positions against
  an optimal defender. Small enough for tablebase ground truth, hard enough
  that conversion is not free (the failure modes are draws: repetition,
  wrong trades).
- **Mathematical diagnostics.** Invariant test suites, collapse detectors,
  effective-rank checks, calibration instruments (§4b).
- **Data.** Lichess open data (2019-01, ~4GB prefix): positions, outcomes,
  Elo, clock. Every 50th game is held out and never trained on.

## 4. Details

### 4a1. The method stack in words — and why, in chess terms

**State: board plus the rules' counters.** A chess position is not just piece
placement: the threefold-repetition surface only exists in board ×
repetition-count space, and the 50-move clock is part of what makes a
position drawn or alive. The encoder input is bitboard planes plus these
augmented-state fields.

**Two towers, F and B.** F(s, ω) embeds the state *with play context* (Elo
bins, clock bucket); B(g) embeds a goal as a position only. Goals are
destinations — they don't have a clock.

**A quasimetric, because chess does not go back.** Captures cannot be
undone, pawns never retreat, castling rights only disappear. Reachability is
therefore *asymmetric*: you can reach the endgame from the middlegame, never
the reverse. The distance head must be a quasimetric — asymmetric, but with
the triangle inequality so multi-hop plans compose ("get the rook behind the
pawn, then advance" costs no more than the sum of its legs). We use IQE
(Interval Quasimetric Embedding): quasimetric *by construction* and
universal (§4a2.1).

**Training the metric: unit steps, a push to "far," and the arrow of time.**
Every move costs exactly one ply, so an observed transition's true distance
is exactly 1 and is pinned two-sidedly. Random pairs are pushed toward a
finite offset — unreachable pairs "want" ∞, but no consumer ever needs ∞,
only separation above every reachable distance; true infinity lives in
probability space as −ln P(reach). The sharpest training signal comes from
**irreversibility itself**: from one parent, two divergent irreversible moves
(different pawn advanced, different capture) yield near-identical boards that
are *provably mutually unreachable* — certified by monotone resources
(piece counts, pawn-advancement budget, castling rights) that no legal move
can increase (§4a2.3). Chess's arrow of time, turned directly into geometry.
Confluence is not a contradiction: two forks that trade down into the same
endgame are far from *each other* yet close to their common *future* —
exactly what asymmetry permits and a symmetric metric would forbid.

**Value: a committor to the outcome surfaces.** A 3-way head on F(s)
predicts P(win/draw/loss) — the probability of absorption at each surface
under the play measure that generated the data (§4a2.4). This is
deliberately *measure-dependent*: trained on human games it is a human-play
committor, and skill/exogenous conditioning is the designed refinement, not
a confounder. Under perfect play the committor collapses toward {0, 1};
under fallible play it grades — which is the point.

**Coherence: how far to trust the map.** The geometry is best-case; the
opponent has a say. A KRvK technique position is *coherent* to depth 12+ —
many legal defenses, one outcome — while a sharp middlegame with one winning
only-move is coherent to depth 2–3. Coherence is therefore grounded in
**probability, not move count** (§4a2.5). Value asks "how good is the
destination"; coherence asks "how far ahead is the map reliable" — two
readouts of one committor field, and independent axes: a forced perpetual is
low-value, maximum-coherence.

**Search: the adversarial step no metric can take.** Shortest-path geometry
composes by min-sum; game value composes by min-max, and "can I *force* this
region" is attractor membership in a two-player reachability game — a
different fixpoint from any distance (§4a2.6). A PUCT MCTS performs the
minimax composition over the field: leaf values from the committor readout,
draws neutral with a separate *clearance* term for winning-side draw
avoidance, repetition detected along the search path, faster mates strictly
preferred via per-ply discounts.

**Planner (designed; post-MVP).** A metareasoning agent with **internal
actions** — probe_region(r) (bounded MCTS toward a candidate region),
set_plan(subgoal) — and **game actions** — make_move, offer_draw, resign.
It probes candidate regions (including the draw surface when losing),
commits to the best forceable one, and acknowledges hopeless positions
rather than playing on forever. Its decision rule is the value × coherence
quadrant: bank what is won and forced; search what is winning but fragile;
offer draws from dead equality; resign what is confidently lost. Plan memory
blocks refuted regions with wake conditions. Ultimately this layer is its
own RL problem (probes cost compute; reward = outcome − thinking cost),
trained on top of the frozen executor.

### 4a2. The mathematics

**1. The quasimetric head (IQE).** A latent u ∈ ℝᵈ is reshaped to
(C components × K dims). For an ordered pair (u, v), per component c:

    d_c(u→v) = | ∪ₖ [u_ck, max(u_ck, v_ck)] |        (Lebesgue measure)

— length accumulates exactly where v exceeds u ("must climb"); u dominating
v costs nothing ("already reached"). Components combine as
d = e^s · (α·max_c d_c + (1−α)·mean_c d_c), α = σ(α₀). Each piece preserves
the axioms d(u,u)=0, asymmetry, and d(u→w) ≤ d(u→v) + d(v→w), so the head is
a quasimetric for *any* parameters, and universal (Wang & Isola). Score =
−d; distances between F(s) and B(g).

**2. The training objective (constrained optimization, no ranking loss).**

    minimize    E_{s,g~pool} [ softplus(offset − d(F(s), B(g)), β) ]      (push far pairs up)
    subject to  E_{(s,s′) observed} [ (d(F(s), B(s′)) − 1)² ] ≤ ε²        (unit step, two-sided)

The two-sided pin is exact for unit-cost game graphs: the observed move is a
witness that d*(s,s′) = 1 (nothing shorter exists between distinct
positions). Long distances are never supervised — they assemble by chaining
unit steps through the triangle inequality; the push provides the outward
pressure that makes chained distances tight. The constraint is enforced by a
PID-controlled multiplier on the violation c_t = sq_dev − ε²:

    I_t = max(I_{t−1} + k_i c_t, 0),   λ_t = max(k_p c_t + I_t + k_d (c_t − c_{t−1})₊, 0)

(one-sided derivative: damping opposes rising violation only). Additional
terms: a VICReg variance floor per embedding dimension (escape gradient at
the constant-embedding fixed point), and the sibling hinge

    L_sib = relu(F − d(a→b)) + relu(F − d(b→a)),   F ≈ 2 × offset,

applied only to certificate-incomparable fork siblings (below) — a floor,
not a target: nothing pulls d back down.

**3. Provable unreachability (monotone certificates).** Define
C(s) = (white pawns, black pawns, white total, black total, white pawn
budget, black pawn budget), where pawn budget = Σ ranks-to-promotion. Every
legal move — including promotion, en passant, castling — leaves every
coordinate non-increasing (piece *counts*, not values: promotion converts,
never creates). Hence s →* t implies C(t) ≤ C(s) componentwise, and if
C(a), C(b) are **incomparable** (each strictly smaller somewhere), then
neither position can reach the other: d(a→b) = d(b→a) = ∞. Fork siblings
tying on all certificates (fungible captures) are *rejected* — piece
identity is not conserved, and such pairs can transpose. The provable-bound
ladder: side-to-move parity makes any same-parity distance even (so ≥ 2);
same-capturer forks admit no 2-path (occupancy argument), giving d ≥ 4 with
a machine-checked tight witness; incomparability gives ∞.

**4. The committor (value as a boundary-value problem).** With absorbing
surfaces W, D, L and play measure μ:

    P(s) = Σ_m μ(m | s, ω) · P(s·m),    P|_W = 1,  P|_{D∪L} = 0

— the harmonic function of μ on the game graph (Dirichlet problem); the
phead is its learned (amortized) solution, and search is its on-demand
evaluation. The certainty distance is d_cert(s) = −ln P(s): additive along
corridors, +∞ at impossibility, numerically tame. Two consequences used as
instruments: P is a **Doob martingale** along real play,
E[P(s_{t+1}) | s_t] = P(s_t) (its residual is a calibration/leakage test),
and value = OR over corridors while a single plan's survival is an AND —
their gap is redundancy (robust positions have many corridors; sharp ones,
one).

**5. Coherence.** Per node, γ(s) = e^{−k(1−P(s))}; along a backup path the
discounts compound: Π_i γ(s_i) = e^{−k Σ(1−P_i)} ≈ (Π P_i)^k — approximately
"the probability the whole line is realized," raised to the trust gain k.
Forced lines (P ≈ 1 per ply) reach the root undiscounted regardless of
branching factor; fragile lines arrive attenuated. Value is P evaluated at a
point; coherence is P's decay along a path — amplitude vs correlation
length of one field.

**6. Why search is necessary (min-sum ≠ min-max).** The quasimetric computes
shortest paths: d(s,g) = min over move sequences, *both sides steering*.
Forceability is the attractor fixpoint

    A₀ = W,   A_{i+1} = A_i ∪ { s : our move ∃→A_i }  ∪ { s : opponent to move, ∀ moves →A_i }

— an alternating (∃/∀) fixpoint that no path-additive quantity computes. The
committor under *optimal* play is the indicator of the attractor; MCTS
minimax is its sampled, anytime approximation, with the field as leaf prior.

**7. Verdicts.** Play claims: paired mate-rate difference on a fixed start
set vs a deterministic optimal defender; bootstrap CI over starts; e-values
for anytime-valid sequential decisions (peeking-safe). Diagnostics carry
bootstrap CIs (per-game where outcomes are shared within a game).

### 4b. Implementation: health machinery, evaluation, and lab rules

**Health tests** (all in-repo, run continuously):

| Test | Looks for | How |
|---|---|---|
| Invariant suite (`tests/test_invariants.py`) | value-scale symmetry (draw negates to itself; win = −loss; loss < draw < win); quasimetric *direction* semantics (d(dominating→dominated)=0, reverse large — the failure axiom tests cannot see, since a transposed quasimetric still satisfies the axioms); IQE self-zero, monotonicity-in-gap, triangle inequality; monotone certificates non-increasing under captures, promotion, en passant, castling | property assertions with constructed witnesses |
| Collapse auto-detector (in training) | *local collapse* (unit-step distances → 0: the degenerate all-states-identical solution) and *small-world* (random-pair distances ≈ unit-step: metric not spreading) | rolling means over 2k steps, checked every 1k after warmup; warns, or halts the run under a flag — turns a silent multi-hour degeneration into a 20-minute failure |
| Effective rank (bootstrapped) | dimensional collapse of the embedding | participation ratio of the singular spectrum over a fixed state sample, with bootstrap CI |
| Calibration gate | committor overconfidence (poisons search termination, coherence, resign decisions) | reliability curves + ECE with **per-game** bootstrap CIs (positions within a game share one outcome); **martingale residuals** — E[P(s_{t+1}) | s_t] = P(s_t) along held-out play, tested at endpoints and per game-phase; systematic drift is miscalibration and, via adaptedness, doubles as a leakage detector |
| Leakage audit | oracle contamination (the toy is fully tablebased; a tablebase in the play loop is cheating) | static purity check over the training path, provenance stamps in checkpoints, hard 1/50 game holdout |
| Regression suite | previously-fixed failure modes staying fixed | 194 tests, incl. mate-over-draw selection for both colors, per-ply mate-discount dominance, path-aware threefold |

**Evaluation harness.** All play claims come from `playout_ab`: the model
plays White from a fixed set of tablebase-verified winning starts; Black is
a **tablebase-optimal deterministic defender**, so the only variance is
which starts were sampled — per-start results are exact and reproducible.
Verdicts are paired mate-rate differences with bootstrap CIs and e-values.
Budgets are matched in *counted network evaluations*. The design premise:
proxy metrics (retrieval accuracy, loss curves) inform, but **play is the
arbiter** — every mechanism is accepted or rejected on conversion.

**Major LLM-loop guidances** (the lab rules the agent operates under):
validate on a short run before any long run; check long runs at 1 minute and
every ~5 after (output-growth watchdogs); no number enters the journal
without a printed script verdict; when stuck >15 minutes on something that
should work, search the literature before tuning; directional builds need
human sign-off — measurements don't; rejections are conditional on the field
version (retest shelved mechanisms after promotions); "X is a Y" claims must
state Y's criteria and verify them; no sycophancy — claims graded
proven / provable / plausible, retractions explicit; and **one scalar field
per question** — geometry, value, trust, and policy each live in their own
layer, and history shows every major failure was one layer doing another's
job.

**Visualization** in the theory's own coordinates (`committor_atlas`): the
outcome **simplex** (positions and game trajectories in P(W/D/L) space —
surfaces are the corners), the **certainty plane** (−ln P_win vs −ln P_loss
— the planner's coordinates), and **committor level sets** over material ×
ply (the surfaces as contour lines). Linear projections of the embedding
(PCA) were tried and rejected as uninformative.

## 5. Results

### 5a. Current

- **The toy converts, visibly.** The current field + search converts
  **0.60** of KRRvKBP winning starts against optimal defense (n=100, 800
  evals/move). Example conversion, five plies, no shuffling:
  `1.Rxb6+ Ke7 2.Rb7+ Kf8 3.Rc8#`. The remaining 40% fail as draws
  (repetition, wrong trades) — the failure mode the roadmap targets. A
  first attempt to carry draw-avoidance through the clearance term (β=0.5)
  was not significant (0.51 vs 0.60, CI [−0.19, 0.00], e=1.0); the right β
  or mechanism is open.
- **The committor field is real but miscalibrated.** Its level sets track
  material sensibly and game trajectories flow to the correct corners of
  the simplex; but it is overconfident (predicted 0.85 → realized 0.72 in
  the top bins), and it has a **draw-confidence ceiling**: max P(draw) =
  0.49 over 22k holdout positions, consistent with the 5%-draw human-game
  training measure. Consequence: draw-region recognition and draw-offers
  are blind until the committor sees draw-rich data — a concrete instance
  of measure-dependence (H2), found by the atlas visualization.
- **Quasimetric training: stability solved, spread open.** The unit-step
  constraint now holds robustly (no degenerate collapse across runs; the
  detector's recent catches were all the *small-world* mode). Distances
  between random positions remain compressed (~1.5–1.8 vs the ~15 target);
  a force-balance experiment (stronger irreversibility-repulsion) is in
  progress. The IQE head itself is mathematically verified.
- **Coherence and the region-recognizer are built and gated** — shelved
  pending calibration; see 5b's last entries for why that gate exists.

### 5b. Selected past results and the choices they invalidated

- **Flat similarity is not planning geometry.** Cosine/dot-product
  reachability could rank neighbors, but multi-hop plans did not compose;
  waypoint decomposition only started helping once the distance had metric
  structure. → A by-construction (quasi)metric.
- **Retrieval objectives cannot train interval geometries.** Contrastive
  ranking (InfoNCE) trained the encoder but left absolute distances
  arbitrarily small — the interval quasimetric stayed flat and retrieval
  plateaued far below the bilinear baseline. → Constrained-optimization
  training (unit-step pin + separation push); ranking alone rejected.
- **Absolute ply-gap regression fights the metric.** Calibrating distances
  to observed ply gaps imposed a *trajectory-length* scale on a
  *shortest-path* quantity and suppressed the geometry. → Dropped; scale
  emerges from chaining unit steps. Relatedly, distances between reasonable
  positions live at shortest-path scale (~15 plies), not trajectory scale
  (~50–130): separation floors sit just above the reachable band, not at
  "horizon" scales.
- **Goals are surfaces, not points.** Steering to a single mate-centroid
  embedding converted measurably worse than reading the committor to the
  whole mate surface. → Goal-as-region adopted; poles retired.
- **Outcome-agnostic reachability is tactically blind.** A field trained
  without game results cannot distinguish "mate is near" from "mate
  *against me* is near," and play showed it. → Outcome conditioning (the
  committor) is load-bearing, not auxiliary.
- **Move-count entropy is not coherence.** Discounting deep values by
  branching factor punished *forced mates with many legal defenses* — the
  most trustworthy lines in chess. → Coherence must be grounded in
  P(outcome), not option counts (H4 in its final form).
- **An uncalibrated recognizer must not gate search.** Letting the
  committor declare regions "resolved" at 0.9 confidence collapsed
  conversion 0.60 → 0.20 (e = 4×10⁶): it stopped the search exactly where
  conversion still required work, because its confidence exceeds its
  accuracy. → Calibration is a *hard precondition* for search-termination,
  coherence weighting, and any future resign/draw decisions.
- **The draw value must be neutral — and draw-avoidance must be explicit.**
  Valuing draws as near-losses secretly performed draw-avoidance for the
  winning side; making the value symmetric (draw = 0, required for coherent
  minimax) exposed that the avoidance work needs its own mechanism, and the
  first candidate (clearance β=0.5) is not yet sufficient. → Value
  semantics and steering pressure are separate layers.

## 6. Work still to be done

1. **Spread the metric.** Resolve the small-world plateau (repulsion-weight
   sweep in progress; offset sweep {15, 30, 60}; an alternative
   published trainer is shelf-ready if constrained-max stalls) — then the
   first full QRL-IQE vs incumbent conversion head-to-head.
2. **Calibrate the committor** (temperature scaling or retraining), re-run
   the calibration gate, and only then re-test the obvious-region
   soft-terminal and coherence-weighted backup in play.
3. **Draw-rich training mass** — self-play from the toy region or
   draw-upweighted losses — to lift the draw-confidence ceiling; then
   D-surface recognition and the resign/draw-offer layer (with a standard
   no-resign holdout to calibrate false positives).
4. **Win the last 40%:** a draw-avoidance mechanism that actually converts
   (clearance β sweep or a dedicated repulsion-from-draw-basin term),
   evaluated on the fixed toy set.
5. **Build the planner-as-prober** (probe interface → candidate regions →
   hand-coded decide → plan memory → resign/draw), then its RL phase
   (value-of-information over internal actions).
6. **The play measure ω** — skill and exogenous conditioning (clock
   pressure, tilt), two-sided (ours and theirs): the fallibility layer that
   turns perfect-play committors into human ones.
7. **Memory across timescales** — persistent search trees within a game,
   committor sharpening from probe results, plan memory across games.
8. **Sensitivity analyses** — embedding dimension, IQE component count,
   offset; the parity invariant (even/odd distance classes) as a further
   health check.
9. **Publication pipeline** — refresh the papers in `writing/` from
   journaled verdicts once the QRL-IQE verdict lands.
