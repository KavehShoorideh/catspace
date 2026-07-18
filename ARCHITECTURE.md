# catspace — Architecture

2026-07-18. The working architecture document: motivation, approach, the
settled method stack (each component: description, mathematics, and the
reason it was chosen over alternatives), implementation and health machinery,
results to date, and open work. Code-level verification lives in
MATH_AUDIT.md; day-by-day evidence in JOURNAL.md. Soft limit 8 pages.

Notation used throughout: s and s′ denote positions (states), g a goal
position or region, m a legal move, s·m the position after playing m in s,
t a ply index, and ω the play context (player skill and exogenous state).
All other symbols are defined where they first appear.

## 1. Motivation

We want to understand what it takes for an agent to plan — to pick
destinations, decompose them into subgoals, commit to a line, notice when it
is refuted, and know when to fight, simplify, or resign — rather than merely
search well. Chess is the laboratory because it is simultaneously rich
enough to demand real planning and verifiable enough to grade it: the rules
define exact terminal outcomes, small endgames have tablebase oracles
(precomputed perfect-play tables), and a large record of human play exists
at every skill level.

The hypotheses under test:

- **H1 (geometry).** The game's reachability structure can be learned as a
  geometry — an asymmetric distance between positions — and that geometry is
  useful for planning (subgoal selection, plan composition) in a way flat
  similarity is not.
- **H2 (value is derived).** "How good is this position" is not a primitive
  quantity. It is the probability of reaching each outcome surface under a
  play measure — a model of what these players would actually do. Change the
  players and the value changes; the geometry does not.
- **H3 (planning is metareasoning).** A planner is an agent whose internal
  actions are computations (probe a region with search, adopt a plan) and
  whose game actions are moves, draw offers, and resignation. Search is a
  tool the planner calls, not the planner itself.
- **H4 (fallibility is the bridge).** The gap between best-case geometry and
  realizable value is governed by the probability that the correct moves
  actually get played — by both sides. This "coherence" determines how deep
  a plan can be trusted and where computation should be spent.

The intended transfer: the same structure — irreversible actions, resource
budgets, an adversarial or uncertain environment, categorical outcomes — is
the shape of agentic-planning and robotics problems generally.

## 2. Main approach

Treat the game as a directed graph over positions and learn two independent
objects:

1. **Reachability geometry.** A quasimetric d(s, g) — an asymmetric distance
   satisfying the triangle inequality — estimating the best-case number of
   plies to reach g from s, for any pair of positions. Learned from position
   pairs, with provably-unreachable pairs pushed apart.
2. **The play measure μ(m | s, ω).** The probability that move m is actually
   played in position s given context ω: the players' skill and exogenous
   factors such as time pressure or emotional state. Perfect play is the
   degenerate case in which μ concentrates on the best move; human play
   spreads.

Everything else is computed from these two. The value of a position is the
probability of hitting the win surface before the draw or loss surfaces when
μ drives the dynamics over the graph — a committor, in the language of
statistical mechanics, with the outcome surfaces as absorbing boundaries.
Goals are surfaces and regions, never single positions: "mate" is a boundary
whose crossing ends the process; "rook endgame up a pawn" is a region one
enters. Coherence — how far ahead the best-case map can be trusted — is the
probability that an intended line survives both sides' choices, and it
controls search depth and backup trust. The planner sits on top, choosing
regions to probe and plans to commit to, with search as its probing
instrument.

## 3. Environment and resources

- **Claude loop.** The research is conducted by a Large Language Model (LLM)
  agent (Claude) under human direction: it designs, implements, runs, and
  journals experiments; it runs autonomously overnight with scheduled
  check-ins; it operates under standing lab rules (§4.11).
- **Local hardware.** An Apple-silicon laptop using the Metal Performance
  Shaders (MPS) graphics backend. Models are small (~9M parameters); a
  40,000-step training run takes a few hours; the Graphics Processing Unit
  (GPU) is used by one job at a time.
- **Experimentation harness.** Deterministic paired two-condition (A/B)
  playouts against a tablebase-optimal defender, fixed start sets, bootstrap
  confidence intervals (CIs), e-values (evidence values: anytime-valid
  sequential test statistics that remain valid under repeated peeking), a
  leakage audit, and a running JOURNAL.md in which no number is recorded
  without a printed script verdict.
- **Toy scenario.** The current milestone task: convert winning positions of
  King and two Rooks versus King, Bishop, and Pawn (KRRvKBP) against an
  optimal defender. Small enough for tablebase ground truth; hard enough
  that conversion is not free (the failure modes are draws: repetition,
  incorrect trades).
- **Mathematical diagnostics.** Invariant test suites, collapse detectors,
  effective-rank checks, calibration instruments (§4.9).
- **Data.** Lichess open data (January 2019, ~4 GB prefix): positions,
  outcomes, Elo ratings, clock information. Every 50th game is held out and
  never trained on.

## 4. The stack, piece by piece

Each component below is given in three parts: what it is, its mathematics,
and why this choice rather than the alternatives.

### 4.1 State encoding

**Description.** A chess position is encoded as bitboard planes (one 8×8
plane per piece type and color) plus the rule counters that are genuinely
part of the state: the repetition count and the 50-move clock. The
threefold-repetition draw exists only in the product space of board ×
repetition-count, and the 50-move clock changes whether a position is alive
or dead; omitting them makes distinct states indistinguishable.

**Why not board-only.** Early experiments used board-only encodings; the
search could then neither see repetition draws forming nor value clock
pressure. The counters were added as explicit input planes/fields.

### 4.2 The two encoders, F and B

**Description.** Two separate networks embed positions for the two distinct
roles a position can play. The **forward encoder** F embeds the *current
state together with its play context*: F(s, ω) ∈ ℝ^{d_e}, where d_e is the
embedding dimension (currently 512) and ω carries Elo bins for both players
and a clock bucket. The **backward encoder** B embeds a position in its role
as a *goal*: B(g) ∈ ℝ^{d_e}, board only. The reachability estimate between a
state and a goal is the quasimetric distance d(F(s, ω), B(g)) of §4.3. The
names F and B follow the forward–backward factorization used in
representation learning for reinforcement learning (RL): F answers "where
can I go from here, as the player I am," B answers "what does arriving at g
look like."

**Why two encoders rather than one.** A state has a context — whose move it
is matters, and the players' condition matters (H2) — while a destination
does not: a goal is a position, not a position-being-played-under-time-
pressure. Sharing one encoder would force these two roles into one
representation and would prevent conditioning the state side on ω without
also (incorrectly) conditioning the goal side.

### 4.3 The reachability head: a quasimetric (IQE)

**Description.** Distances between F- and B-embeddings are computed by an
Interval Quasimetric Embedding (IQE) head. A quasimetric is a distance that
is asymmetric — d(u → v) ≠ d(v → u) in general — but still satisfies the
triangle inequality. Chess requires exactly this: captures cannot be undone,
pawns never retreat, castling rights only disappear, so reachability is
one-directional (the endgame is reachable from the middlegame, never the
reverse), while the triangle inequality is what makes multi-hop plans
compose ("reach the rook endgame, then convert" costs no more than the sum
of its legs).

**Mathematics.** Let u, v ∈ ℝ^{d_e} be two embeddings. Reshape each into M
components of K dimensions each (d_e = M·K; currently M = 32, K = 16). For
component index c ∈ {1..M} the component distance is the length of a union
of intervals on the real line:

    d_c(u → v) = | ∪_{k=1}^{K} [ u_{ck}, max(u_{ck}, v_{ck}) ] |,

where |·| denotes the Lebesgue measure (total length). Each interval is
nonempty exactly where v exceeds u, so distance accumulates in the "must
climb" direction and is zero in the "already past it" direction. Components
are combined with a learned mixture and scale:

    d(u → v) = e^{η} · ( α · max_c d_c + (1 − α) · mean_c d_c ),

with α = σ(a₀) ∈ (0,1) a learned mixing weight (σ is the logistic sigmoid,
a₀ a learned scalar) and η a learned log-scale. Every part of this
construction preserves the quasimetric axioms — d(u,u) = 0, asymmetry, and
d(u→w) ≤ d(u→v) + d(v→w) — for all parameter values, and the family is
universal: any quasimetric on a finite set can be approximated. The in-repo
verification covers the union-measure computation, the direction convention,
and the axioms under adversarial parameter search.

**Why IQE and not the alternatives.** A plain inner product or cosine score
has no triangle inequality, and multi-hop plans built on it did not compose
(§5b). A symmetric metric contradicts irreversibility. A metric-plus-free-
residual construction (a symmetric metric with an unconstrained bilinear
correction) leaves the asymmetric part unconstrained — the correction can
violate composition. IQE is the simplest head that is asymmetric *and*
triangle-consistent by construction rather than by training.

### 4.4 Training the quasimetric

**Description.** The metric is trained as a constrained optimization, not by
ranking. Observed one-ply transitions are pinned to distance exactly 1 (a
move costs one ply). All other pressure is outward: random position pairs
are pushed toward a finite separation target, and provably-unreachable pairs
(§4.5) are pushed above a floor. Long-range distances are never supervised
directly — they assemble by chaining unit steps through the triangle
inequality.

**Mathematics.** With θ the network parameters, the objective is

    minimize_θ   E_{(s,g)} [ softplus( ρ − d(F(s,ω), B(g)), β_s ) ]
    subject to   E_{(s,s′) observed} [ ( d(F(s,ω), B(s′)) − 1 )² ] ≤ ε²,

where ρ is the separation target ("offset," ~15 plies), softplus(x, β_s) =
(1/β_s)·ln(1 + e^{β_s x}) with sharpness β_s, and ε is the constraint
tolerance. The pin is two-sided — the observed move witnesses that the true
shortest-path distance between distinct adjacent positions is exactly 1, so
penalizing both directions is exact for a unit-cost game graph, and it is
what makes the degenerate everything-at-distance-zero solution costly. The
constraint is enforced by a multiplier λ_t driven by a Proportional–
Integral–Derivative (PID) controller on the violation c_t (the constraint
excess at step t):

    I_t = ( I_{t−1} + k_i·c_t )₊ ,   λ_t = ( k_p·c_t + I_t + k_d·(c_t − c_{t−1})₊ )₊ ,

where (x)₊ = max(x, 0) and k_p, k_i, k_d are the controller gains; the
derivative term is one-sided so that damping opposes rising violation only.
Two auxiliary terms: a per-dimension variance floor on the embeddings (in
the style of Variance-Invariance-Covariance Regularization, VICReg), which
supplies an escape gradient at the constant-embedding degenerate point; and
the unreachable-pair hinge

    L_sib = ( Φ − d(a → b) )₊ + ( Φ − d(b → a) )₊ ,

with floor Φ ≈ 2ρ, applied to the certified pairs (a, b) of §4.5. A hinge
is a floor-and-release: nothing pulls d back down to Φ, so no consumer ever
needs an infinite distance — only separation above every reachable one.
True impossibility is represented in probability space (§4.6), where
−ln P → ∞ smoothly.

**Why not ranking or regression.** Contrastive ranking objectives (of the
Information Noise-Contrastive Estimation family, InfoNCE) enforce only
*relative* order within a batch; absolute distances can stay arbitrarily
small, and interval-based heads in particular stay flat under them (§5b).
Regressing distances onto observed ply gaps imposes a trajectory-length
scale on a shortest-path quantity — games meander, so the gap between two
positions in a game is an upper bound, not the distance — and measurably
suppressed the geometry (§5b).

### 4.5 Provably unreachable pairs (monotone certificates)

**Description.** The hardest and most informative training pairs are
positions that look nearly identical but are mutually unreachable. Chess
supplies them: from one parent position, two divergent irreversible moves
(a different pawn advanced, a different piece captured) produce siblings
that can never transpose into each other. Certification must be done
carefully — piece identity is not conserved (a position is a set of
piece-on-square facts), so "different capture square" alone is unsound;
two captures of same-type pieces can shuffle back into each other.

**Mathematics.** Define the certificate vector

    C(s) = ( white pawn count, black pawn count, white piece total,
             black piece total, white pawn budget, black pawn budget ),

where pawn budget = Σ over that side's pawns of ranks-remaining-to-
promotion. Every legal move — including promotion, en passant, and castling
— leaves every coordinate non-increasing (counts, not point values:
promotion converts a pawn into a piece and never creates one). Therefore
reachability implies componentwise dominance: s →* t requires C(t) ≤ C(s).
If C(a) and C(b) are *incomparable* — each strictly smaller than the other
in some coordinate — then neither position can reach the other, and
d(a→b) = d(b→a) = ∞ is proven. Sibling pairs that tie on all certificates
(fungible captures) are rejected. The full ladder of provable lower bounds:
side-to-move parity forces every same-parity distance to be even (hence
≥ 2); same-capturer forks admit no two-ply path (an occupancy argument),
giving d ≥ 4 with a machine-checked tight witness; certificate
incomparability gives ∞.

**Why this and not heuristic negatives.** Random negatives are far in
feature space and teach little; heuristic "probably unreachable" rules risk
training falsehoods into the geometry (a falsely-repelled pair whose true
distance is 4 corrupts the metric at exactly the resolution that matters).
The certificate criterion admits only pairs whose infinite distance is a
theorem of the rules.

### 4.6 Value: the committor

**Description.** A three-way head on F(s, ω) predicts the probabilities
P_W, P_D, P_L of the game ending in a win, draw, or loss (W/D/L) for the
side of interest. This is a committor: the absorption probability at each
outcome surface under the play measure that generated the training games.
It is deliberately measure-dependent — trained on human games it is a
human-play committor; conditioning on ω (skill, clock) is the designed
refinement. Under perfect play the committor collapses toward {0, 1}; under
fallible play it grades.

**Mathematics.** With absorbing surfaces W, D, L and play measure μ, the
win-committor P satisfies the boundary-value problem

    P(s) = Σ_m μ(m | s, ω) · P(s·m),    P ≡ 1 on W,    P ≡ 0 on D ∪ L,

i.e., P is the harmonic function of μ on the game graph. The learned head
is an amortized solution of this problem; tree search (§4.8) is its
on-demand evaluation; the two must agree where both are available. The
certainty distance d_cert(s) = −ln P(s) is additive along independent
survival events, tends to +∞ at impossibility, and is numerically tame.
Two structural consequences are used as instruments: (i) P is a Doob
martingale along real play, E[ P(s_{t+1}) | s_t ] = P(s_t), so systematic
per-ply drift on held-out games indicates miscalibration (and, through the
requirement that P(s_t) use only information available at ply t, doubles as
a leakage detector); (ii) the value of a position is an OR over winning
corridors while a single plan's survival is an AND along one corridor —
the gap between them is redundancy, distinguishing robust positions (many
corridors) from sharp ones (one).

**Why a committor and not a scalar evaluation.** A scalar evaluation hides
its play-strength assumptions; a committor makes the measure explicit,
which is required by H2 and by the planned fallibility layer. Outcome
surfaces (not single goal states) are required because "mate" is thousands
of positions; steering to a single mate-centroid embedding converted
measurably worse than reading the committor to the surface (§5b). An
outcome-agnostic reachability signal is disqualified outright: it cannot
distinguish "mate is near" from "mate against me is near" (§5b).

### 4.7 Coherence: how far to trust the map

**Description.** The quasimetric is best-case: it reports the shortest
route assuming both sides cooperate with the plan. The opponent does not.
Coherence quantifies how far ahead the best-case map remains reliable. It
is grounded in probability, not in the number of legal moves: a textbook
King-and-Rook versus King (KRvK) mate is trustworthy twelve plies out even
though the defender has many legal moves (all lead to the same outcome),
while a sharp middlegame with a single winning continuation is trustworthy
for two or three.

**Mathematics.** Per position, γ(s) = e^{−κ·(1 − P(s))}, where κ ≥ 0 is a
trust gain and P is the committor confidence. During search backup the
per-node factors compound along the path: Π_i γ(s_i) = e^{−κ·Σ_i (1−P(s_i))}
≈ (Π_i P(s_i))^κ — approximately the probability that the entire line is
realized, raised to the gain. A proven forced line (P ≈ 1 at every node)
reaches the root undiscounted; a fragile line arrives attenuated. Value is
P evaluated at a point; coherence is P's decay along a path.

**Why not branching-factor entropy.** An earlier version discounted by the
entropy of the move distribution; it penalized forced mates with many legal
defenses — the most trustworthy lines in the game — and was replaced by the
probability-grounded form (§5b).

### 4.8 Search: the adversarial composition step

**Description.** A Monte Carlo Tree Search (MCTS) in the Predictor + Upper
Confidence bounds applied to Trees (PUCT) style runs minimax over the
learned field: leaf values come from the committor readout, selection
alternates maximizing and minimizing players, and terminal rules are exact
(draws valued at zero; repetition detected along the search path; faster
mates strictly preferred through a per-ply discount). A separate clearance
term provides steering pressure away from the draw basin when winning —
deliberately kept out of the value scale, which must remain symmetric for
minimax to be consistent.

**Mathematics (why search cannot be replaced by the metric).** Shortest-path
distance composes by minimum-over-paths of sums (min-sum). Forceability —
"can I reach this region against best resistance" — is membership in the
attractor of a two-player reachability game, defined by the alternating
fixpoint

    A₀ = W,   A_{i+1} = A_i ∪ { s : side-to-move is ours and ∃m, s·m ∈ A_i }
                          ∪ { s : side-to-move is theirs and ∀m, s·m ∈ A_i } ,

an ∃/∀ alternation that no path-additive quantity computes. The committor
under optimal play is exactly the indicator of this attractor; MCTS minimax
is its sampled, anytime approximation with the field as prior.

**Why this readout.** Beam-style fixed-shape search was the earlier
executor; visit-guided PUCT with the same evaluation budget is the current
one. Search budgets are counted in network evaluations so that comparisons
are compute-matched.

### 4.9 Planner (designed; post-milestone)

**Description.** A metareasoning agent with two action sets. Internal
actions (computations): probe_region(r) — run a bounded MCTS toward
candidate region r and return its forced value, confidence, and coherence —
and set_plan(subgoal). Game actions: make_move, offer_draw, resign. A
decision step is a sequence of internal actions ending in exactly one game
action; the planner plays the move the winning probe surfaced. Candidate
regions come from the geometry (outcome surfaces and waypoint regions);
plan memory blocks refuted regions and re-wakes them on drift or on the
recurrence of the refuting resource. The decision rule is the value ×
coherence quadrant: bank what is won and forced; search what is winning but
fragile; offer a draw from dead equality; resign what is confidently lost.
In its final form this layer is itself an RL problem — internal actions
cost compute, the reward is the game outcome minus thinking cost — trained
on top of the frozen executor.

### 4.10 Health machinery

| Test | Looks for | How |
|---|---|---|
| Invariant suite (`tests/test_invariants.py`) | value-scale symmetry (a draw negates to itself; win = −loss; loss < draw < win); quasimetric direction semantics (distance from a dominating to a dominated embedding is 0, the reverse is large — a failure mode axiom tests cannot see, since a transposed quasimetric still satisfies all axioms); IQE self-distance zero, monotonicity in gap, triangle inequality; certificate coordinates non-increasing under captures, promotion, en passant, castling | property assertions with constructed witnesses |
| Collapse auto-detector (in training) | local collapse (unit-step distances → 0: the all-states-identical degenerate solution) and small-world collapse (random-pair distances ≈ unit-step: the metric not spreading) | rolling means over 2,000 steps, checked every 1,000 after warmup; warns, or halts the run under a flag |
| Effective rank (bootstrapped) | dimensional collapse of the embedding | participation ratio of the singular-value spectrum over a fixed state sample, with bootstrap CI |
| Calibration gate | committor overconfidence (it poisons search termination, coherence, and any resign decision) | reliability curves and Expected Calibration Error (ECE) with per-game bootstrap CIs (all positions of a game share one outcome, so games are the unit of evidence); martingale residuals at endpoints and per game-phase |
| Leakage audit | oracle contamination (the toy is fully tablebased; a tablebase inside the play loop would be cheating) | static purity check over the training path, provenance stamps in checkpoints, the 1-in-50 game holdout |
| Regression suite | previously-fixed failure modes staying fixed | 194 tests, incl. mate-over-draw selection for both colors, per-ply mate-discount dominance, path-aware repetition detection |

### 4.11 Evaluation harness and lab rules

All play claims come from paired playouts: the model plays White from a
fixed set of tablebase-verified winning starts; Black is a tablebase-optimal
deterministic defender, so the only variance is which starts were sampled —
per-start results are exact and reproducible. Verdicts are paired mate-rate
differences with bootstrap CIs and e-values; budgets are matched in counted
network evaluations. Proxy metrics (retrieval accuracy, loss curves) inform,
but play is the arbiter: every mechanism is accepted or rejected on
conversion.

The LLM loop operates under standing rules: validate on a short run before
any long run; check long runs at one minute and every few minutes after; no
number enters the journal without a printed script verdict; when stuck more
than ~15 minutes on something that should work, search the literature before
tuning further; directional builds need human sign-off, measurements do not;
rejections are conditional on the field version (shelved mechanisms are
retested after promotions); an "X is a Y" claim must state Y's criteria and
verify them; claims are graded proven / provable / plausible and loose ones
retracted explicitly; and one scalar field per question — geometry, value,
trust, and policy live in separate layers, because every major failure to
date was one layer doing another's job.

Visualization is done in the theory's own coordinates (`committor_atlas`):
the outcome simplex (positions and game trajectories in (P_W, P_D, P_L)
space — the outcome surfaces are the corners), the certainty plane
(−ln P_W versus −ln P_L), and committor level sets over material × ply.
Linear projections of the embedding (Principal Component Analysis, PCA)
were tried and rejected as uninformative.

## 5. Results

### 5a. Current

- **The toy converts.** The current field with the corrected search
  converts 0.60 of KRRvKBP winning starts against optimal defense (n = 100,
  800 evaluations per move). An example conversion, five plies:
  1.Rxb6+ Ke7 2.Rb7+ Kf8 3.Rc8#. The remaining 40% end in draws
  (repetition, incorrect trades) — the failure mode the roadmap targets. A
  first attempt to carry draw-avoidance through the clearance term (weight
  0.5) was not significant (0.51 vs 0.60, CI [−0.19, 0.00], e = 1.0); the
  correct weight or mechanism is open.
- **The committor field is real but miscalibrated.** Its level sets track
  material sensibly and game trajectories flow to the correct corners of
  the outcome simplex; but it is overconfident (predicted 0.85 → realized
  0.72 in the top bins), and it has a draw-confidence ceiling: max P_D =
  0.49 over 22,283 holdout positions, consistent with the 5% draw rate of
  the human-game training measure. Draw-region recognition and draw offers
  are blind until the committor sees draw-rich data — a concrete instance
  of measure-dependence (H2), found by the atlas visualization.
- **Quasimetric training: stability solved, spread open.** The unit-step
  constraint now holds robustly across runs (no degenerate collapse; the
  detector's recent halts were all the small-world mode). Distances between
  random positions remain compressed (~1.5–1.8 versus the ~15 target); a
  force-balance experiment (stronger unreachable-pair repulsion) is in
  progress. The IQE head itself is mathematically verified.
- **Coherence and the region-recognizer are built and gated** — shelved
  pending calibration; the last entries of 5b record why that gate exists.

### 5b. Selected past results and the choices they invalidated

- **Flat similarity is not planning geometry.** Inner-product reachability
  ranked neighbors, but multi-hop plans did not compose; waypoint
  decomposition began helping only once the distance had metric structure.
  → A by-construction quasimetric.
- **Ranking objectives cannot train interval geometries.** Contrastive
  ranking left absolute distances arbitrarily small; the interval head
  stayed flat and retrieval plateaued far below the bilinear baseline.
  → Constrained optimization (unit-step pin plus separation push).
- **Ply-gap regression fights the metric.** Games meander; the ply gap
  between two positions is an upper bound on their distance, not the
  distance. Calibrating to it imposed a trajectory scale on a
  shortest-path quantity and suppressed the geometry. → Dropped; scale
  emerges from chaining unit steps. Distances between typical positions
  live at shortest-path scale (~15 plies), not trajectory scale (~50–130).
- **Goals are surfaces, not points.** Steering to a single mate-centroid
  embedding converted measurably worse than the committor-to-surface
  readout. → Goal-as-region adopted.
- **Outcome-agnostic reachability is tactically blind.** A field trained
  without game results cannot distinguish "mate is near" from "mate against
  me is near," and play showed it. → The committor is load-bearing.
- **Branching entropy is not coherence.** It penalized forced mates with
  many legal defenses — the most trustworthy lines in the game. → The
  probability-grounded form (H4).
- **An uncalibrated recognizer must not gate search.** Allowing the
  committor to declare regions resolved at 0.9 confidence collapsed
  conversion from 0.60 to 0.20 (e = 4×10⁶): search stopped exactly where
  conversion still required work, because the committor's confidence
  exceeds its accuracy. → Calibration is a hard precondition for
  search-termination, coherence weighting, and future resign decisions.
- **The draw value must be neutral, and draw-avoidance explicit.** Valuing
  draws near the loss value performed hidden draw-avoidance for the winning
  side; restoring the symmetric value (draw = 0, required for consistent
  minimax) exposed that avoidance needs its own mechanism, and the first
  candidate (clearance at weight 0.5) is not yet sufficient. → Value
  semantics and steering pressure are separate layers.

## 6. Work still to be done

1. **Spread the metric.** Resolve the small-world plateau (repulsion-weight
   sweep in progress; offset sweep {15, 30, 60}; an alternative published
   trainer is shelf-ready if the constrained objective stalls) — then the
   first full quasimetric-field versus incumbent conversion head-to-head.
2. **Calibrate the committor** (temperature scaling or retraining), re-run
   the calibration gate, and only then re-test the region-recognizer and
   coherence-weighted backup in play.
3. **Draw-rich training mass** — self-play from the toy region or
   draw-upweighted losses — to lift the draw-confidence ceiling; then
   draw-surface recognition and the resign/draw-offer layer (with a
   no-resign holdout fraction to calibrate false positives).
4. **Win the remaining 40%:** a draw-avoidance mechanism that converts
   (clearance weight sweep or a dedicated repulsion from the draw basin),
   evaluated on the fixed toy set.
5. **Build the planner-as-prober** (probe interface → candidate regions →
   hand-coded decision rule → plan memory → resign/draw), then its RL phase
   (value-of-information over internal actions).
6. **The play measure ω** — skill and exogenous conditioning (clock
   pressure, emotional state), for both sides: the fallibility layer that
   turns perfect-play committors into human ones.
7. **Memory across timescales** — persistent search trees within a game,
   committor sharpening from probe results, plan memory across games.
8. **Sensitivity analyses** — embedding dimension d_e, component counts
   (M, K), offset ρ; the parity invariant (even/odd distance classes) as a
   further health check.
9. **Publication pipeline** — refresh the papers in `writing/` from
   journaled verdicts once the quasimetric-field verdict lands.
