# Planning by Reachability: A Committor-Field Architecture for Learned Search

*catspace technical report — 2026-07-16. This is a design paper: it lays out
the architecture, the mathematics, and the design decisions of a
reachability-based planner, and states precisely what we believe is novel. Some
components are empirically validated (cited inline to journaled, e-value-gated
experiments); others are motivated but not yet proven at scale, and are marked
as such. The goal is to specify the best architecture we currently have for a
system that **plans** rather than **evaluates**.*

---

## 1. Problem and thesis

Modern game-playing systems learn a **value** function `V(s)` — "how good is
this position" — and search over it (AlphaZero, Leela, Stockfish/NNUE). Value
is a scalar summary of a position's worth; it says nothing directly about *what
to do* or *where you are trying to go*. Planning is bolted on afterward as
tree search over `V`.

We pursue a different primitive. We learn a **reachability field**: an
embedding in which the distance from a state `s` to a goal region `g` is the
(negative log) probability that play starting at `s` actually reaches `g`.
Planning is then **navigation** — descend the field toward the goal — and the
value of a position is a *derived* quantity (how reachable is winning), not the
learned object. The bet, stated as a falsifiable claim: a distance geometry
over *outcomes* is a better substrate for planning than a scalar value, because
it is compositional (multi-step plans chain), it is goal-conditioned (the same
field supports "mate", "reach a draw", "win the bishop"), and it exposes
*where you are on the path*, which a scalar cannot.

This connects to quasimetric goal-conditioned RL (Wang & Isola 2022; Liu et
al. 2023, MRN; Myers et al. 2025) — the line that establishes the optimal
goal-conditioned value function *is* a quasimetric — but applies it to a
discrete, adversarial, DAG-structured domain (chess) where those methods have
no prior validation, and adds two ingredients that domain forces on us: a
**probability-first metric** and **outcome surfaces as goals**.

---

## 2. The reachability field

### 2.1 Two encoders

A state `s` is encoded to feature planes `x(s) ∈ {0,1}^{20×8×8}` (12 piece
planes, side-to-move, castling, en-passant, halfmove clock, and — new — a
repetition-count plane; see §6). Two convolutional trunks with residual blocks
produce a forward embedding and a goal embedding:

```
F(s, ω) = headF( trunk_F(x(s)) ⊕ e(ω) ) ∈ R^d       (state, conditioned on ω)
B(g)    = headB( trunk_B(x(g)) )        ∈ R^d       (goal, board-only)
```

`ω` is the **opponent/player context** (Elo bins for both players, a clock
bucket), embedded and concatenated into the forward head only. Reachability is
a property of a state *under a policy* — a maneuver a strong player converts
and a weak one blunders are at different distances from mate — so the forward
field is conditioned on who is generating the dynamics; goals are board
identities and are not.

### 2.2 The score: a quasimetric with an asymmetric residual

Reaching `g` from `s` is not the same as reaching `s` from `g` (captures are
one-way doors), so the score cannot be a symmetric metric alone. Following MRN
(Liu et al. 2023) we split it:

```
score(F, B) = F · W · B  −  d(F, B)                                    (1)
d(F, B)     = ‖ σ ⊙ F  −  σ ⊙ B ‖₂                                     (2)
```

- `d` is a **genuine metric** on the per-dimension-rescaled embedding: `σ ∈
  R^d_{>0}` (`metric_scale`) is a learned per-coordinate scale, and `d` is a
  Euclidean norm, so non-negativity, symmetry, and the **triangle inequality
  hold by construction** — verified numerically to persist after training, not
  just at init (journaled 2026-07-11). This is what makes plans compose:
  `d(a,c) ≤ d(a,b) + d(b,c)`.
- `W ∈ R^{d×d}` is an unconstrained bilinear **residual** carrying the
  asymmetry `d` cannot. It initializes to `0`, so training starts from a pure
  metric (`score = −‖F−B‖`) and grows asymmetry only as the data demands it.

`σ` is the load-bearing quantity for §7: the metric uses a *distance* dimension
`i` only insofar as `σ_i` is large. Taxing `‖σ‖₁` therefore prices the
dimensionality of the geometry directly.

### 2.3 Base objective: contrastive reachability + scale calibration

The field is trained on real games. For an anchor `s` and a goal `g` that
actually occurred `Δ` plies later in the same game, in-batch InfoNCE pulls the
true future above the batch's other futures:

```
L_NCE = CrossEntropy( score_matrix(F, B) / τ ,  diag )                 (3)
```

InfoNCE fixes only *relative ranking* within a batch — nothing anchors the
absolute scale of `d`. So a second term regresses the true anchor→goal
distance toward the real ply gap (a proper, if coarse, unit of "how far"):

```
L_ply = ‖ d(F(s), B(g)) − Δ / c ‖²      (c a fixed scale)             (4)
```

Both winning and losing trajectories are kept: a goal that is a mate *against*
the mover is a genuine far-and-bad future, and the geometry of "no way back"
can only be learned if unrecoverable positions and their continuations appear
in training (a lesson paid for — an early "winner-POV" filter that censored
losing trajectories was measured to help single-move safety yet was retired
because it starves this calibration; journaled 2026-07-12).

`L_base = L_NCE + λ_ply · L_ply`.

---

## 3. The central reframe: distance is negative-log-reachability

The naive learned distance has **min-semantics**: `d(s,g)` is small if *there
exists* a short path. That is correct for an infallible executor and wrong for
every real agent. In chess it fails concretely: a guaranteed mate-in-15 (quiet,
any move order works) and a mate-in-7 that threads a single needle you find a
third of the time sit at similar min-distance — yet the second is *farther* for
anyone who can blunder. We measured this directly: the incumbent field's
distance was **anti-correlated** with empirical conversion probability on
held-out states (Spearman −0.099, CI [−0.175,−0.027]; journaled 2026-07-14).

The fix is to make the distance a **certainty**:

```
d(s, g)  =  − ln P( reach g from s under the acting policy )           (5)
```

Because segment-wise reach probabilities multiply along a path,
`−ln P` is **additive**, hence subadditive across concatenations — it is a
quasimetric *by the same algebra* as a hop count, and a forced win (`P=1`)
reduces to zero excess distance. Certainty and distance are one currency
(nats). Earlier we wrote this as `plies + λ·(−ln P)` with a hand-set `λ`; the
`plies` term is now understood as itself absorbed into `−ln P` (a longer path
is farther only because a fallible agent has more moves to slip, or because
constraint/epistemic hazard accrues — see §6), so the honest object is pure
`−ln P` and `λ` disappears.

`P` is not an oracle — it is **estimated from the agent's own play**: roll out
the current policy (with exploration noise) from visited states, aggregate
per-state conversion frequency `P̂`, and regress the field toward `−ln P̂`.
Estimation hygiene is a Laplace floor `P̂ ← max(P̂, 1/(n+2))` — which is not a
numerical hack but the **epistemic-hazard term**: finite evidence can never
certify `P=0` or `P=1`, so the field's honest range is bounded away from the
poles (only the rules engine emits exact 0/1, at real terminals).

**Decomposition (measured, 2026-07-15).** Estimating `P̂` at several noise
levels `ε` and regressing `−ln P̂(ε) ≈ E(s) + S(s)·ε` per state identifies a
*path-existence* term `E` and a *sharpness* term `S` (how fast fallibility is
punished). We found **`S` is essentially uncorrelated with distance-to-mate**
(Spearman ≈ +0.04): risk does not accumulate with length, it concentrates in
bottlenecks — the human intuition that a long quiet technique is safe while a
short sharp one is not, quantified. This means any *constant* trade between
length and risk is structurally wrong, and sharpness deserves its own channel.

---

## 4. Goals are surfaces, not points

### 4.1 Why the point-goal fails

A goal like "checkmate" was historically represented as a single vector `z` (a
centroid of embedded mate positions), and reach read out as `−d(F(s), z)`.
Two measured failures killed this: (a) averaging mate exemplars into any
centroid **destroys** the distance structure (centroid calibration against true
mate-distance is flat; nearest-exemplar calibration is real — journaled
2026-07-13); (b) a bank of exemplars scored by soft-min plays *worse* than the
blurry centroid, because "closest exemplar" changes move-to-move and injects
goal-switching noise into move ranking (three decisive play rejections). Point
and set readouts are both wrong: one is uninformative, the other unstable.

### 4.2 Committors: hitting probabilities of terminal surfaces

The correct object treats each terminal outcome as an **absorbing surface** in
augmented state space, and learns, for each, the probability of hitting *that*
surface before the others under the acting policy — a **committor** (the
first-passage / harmonic function of the play process):

```
d_W(s) = − ln P( hit the mate-for-us surface  first )                 (6)
d_D(s) = − ln P( hit a draw surface           first )
d_L(s) = − ln P( hit the mate-against-us surface first )
```

There is **no goal vector**. The surface enters only as a boundary condition
supplied by the rules engine (a rollout terminated; the rules say which surface
it crossed, *anywhere* on that surface — full credit, "touchdown semantics").
This dissolves an entire bug class (there is no goal vector to mismatch or to
average) and it is what makes "reach the draw region" or "reach mate" a single
uniform mechanism. The search descends `d_W`; the draw field `d_D` is the
"out-of-bounds" surface a losing side navigates *toward* and a winning side
keeps clearance *from*.

The win surface is itself a **union** — mate, opponent flag-fall, opponent
resignation — recorded as distinct sub-classes because they are reached by
different plans (conversion vs. survive-on-the-clock), while `d_W` targets
`P(any win)`. Human-game boundaries (resignation, draw-by-agreement) are
*belief-actions*, not rule surfaces, and carry exogenous noise ("wanted a
London, got an Italian" resignations are not positional) — routed to an "other"
class rather than taken as ground truth.

### 4.3 Two ways to realize a committor

We have two concrete realizations, and — importantly — the cheaper one already
beats the point-goal at play:

1. **Distilled committor head** `d_W(s) = softplus(h_W(F(s)))`, trained by
   regressing `−ln P̂` from own-play rollout tables (§5). Calibration
   (absolute scale) is fixed *post hoc* by a **monotone** recalibration —
   isotonic regression on held-out frequencies (rank-exact, so play is
   unchanged; expected calibration error 0.23 → 0.06 in one run). Attempting
   to get calibration for free by training the head with a proper (binomial
   likelihood) loss instead **failed** — it collapsed rank to the base rate,
   because with a shared trunk the fastest likelihood descent is predicting the
   marginal. The clean recipe is *rank from regression, scale from monotone
   post-calibration* (journaled 2026-07-16).

2. **Outcome-head committor** `d_W(s) = −ln softmax(head(F(s)))_win`, where
   `head` is a 3-class win/draw/loss classifier trained on game results *in the
   base objective* (`L = L_base + λ_p · CrossEntropy(head(F(s)), result)`).
   This is the strongest thing we have (§8): a full-board-trained outcome head,
   read as a committor, converts a tablebase-won toy endgame **0.78 vs 0.64**
   for the point-goal readout at deep search (n=200, CI [+0.070,+0.215],
   e=91.5), with faster mates — a promotion-grade effect that cost **zero
   additional training** (it reinterprets a head the model already had).

Crucially this is *not* a value head reintroduced. A value head learns a
scalar `V` end-to-end and searches on it. Here the learned objects are three
*navigable geometries* with plan semantics; the scalar we act on
(`P_W + ½P_D`, §9) is composed at readout from the game's own known scoring
rule, not trained. You can still ask "what is the plan" and get "reach the
repetition region via this corridor" — which `V(s)` can never answer.

---

## 5. The training loop, and where it works

Two learning signals feed the committors:

- **Monte-Carlo distillation.** Roll out from the single canonical start under
  the current policy + ε-noise vs. a fixed defender; aggregate per-state
  `(wins, visits, which-surface)` into a table; regress the heads toward the
  empirical `−ln P̂`. This compounds *data* extremely well (own-play tables
  grow richer each round; within-won gradient reaches Spearman ≈ +0.7).
- **Base-objective co-training.** Train the outcome-head committor jointly with
  `L_base` on human games (realization 2 above). This is the only regime that
  has produced a *confirmed* play win in the certainty program.

A cautionary, load-bearing negative: **post-hoc distillation into the
embedding does not compound into play.** A 12-round closed loop (generate from
the root → distill → gate on paired play and field metrics) improved field
calibration every round while root-conversion stayed flat
(0.719 → 0.719 → 0.672 across pre-loop / round-1 / round-12; journaled
2026-07-16). Only round 1 — a small, dense, *fresh* table — ever cleared a play
gate. The lesson is that repeatedly re-fitting a fixed embedding on an
accumulating table warps the regions play traverses; the training signal wants
**fresh pulses or base-objective co-training**, not cumulative-table
repetition. This is why realization 2 (train the committor *in* the base
objective) is the recommended architecture, not iterative distillation.

---

## 6. Augmented state, walls, and why the near-mate field is flat

A subtle, decisive finding (2026-07-16). In a *guaranteed-win* region every
move still wins under perfect play, so per-position conversion probability is
near-constant — the committor target looks flat. Yet the target is **not**
actually flat: the gradient exists and is **generated by the draw walls**.
Under fallible rollouts, wasting tempo risks the 50-move and threefold
surfaces; that is the *only* thing that makes a slow move worse than a fast one
when the win is otherwise secure. We measured empirical `P̂` falling
0.87 → 0.73 across mate-distance 1 → 8 (Spearman +0.29, CI [+0.16,+0.51]) —
real, wall-generated signal — while the learned field was flat against true
distance (Spearman −0.01).

The field misses it because the wall is erased in three places, each now a
named fix:

1. **Aggregation.** `P̂` keyed by board FEN alone averages a position reached
   fresh with the same position reached mid-shuffle, blurring the repetition
   history the threefold wall lives in. → key targets by `(position,
   repetition-count)`.
2. **Representation.** A repetition-count input plane now exists (state is
   `board × counters`, so the threefold surface is *representable*), but heads
   trained on rep-blind targets leave it inert. → train on rep-keyed targets.
3. **Search.** In-tree positions historically carried no move history, so the
   search cannot detect a threefold forming *inside its own lines*. → path-aware
   terminal detection in search.

This reframes "the near-mate field is flat" from a mysterious capacity failure
to a **specific, fixable signal-erasure** — and it makes `d_D` (the draw
committor) not a luxury but the *sensor* that keeps the win field non-flat: a
clearance term `reach = −d_W + β·d_D` uses distance-from-the-draw-basin to
break ties exactly where `d_W` saturates.

---

## 7. Capacity: wide representation, sparse metric

We measured that the embedding operates in a **~7-dimensional effective
subspace regardless of width** (effective rank 5.7→6.9 across a full training
run, of 64 dims), and that late training — whose gradients contain *zero*
rare-regime information — drifts rare-regime (endgame) features **1.7×** more
than common-regime (middlegame) features. The rare regime is *undefended
collateral*: nothing in the frequent data anchors it, so shared-parameter
updates drag it. This is the mechanistic explanation for the recurring
"play peaks early, then degrades" phenomenon (a 5k-step checkpoint mates a rook
endgame that its own 155k-step continuation shuffles into a draw).

Whether ~7 dims is the *objective's* demand or a *collapse pathology* of
contrastive training (a known InfoNCE stationary point) is the open question.
The architecture's response, either way:

```
L = L_base + λ_p·L_outcome + λ_1·‖σ‖₁                                  (7)
```

a **much wider embedding** (`d`: 64 → 512, trunk 64→256 channels, ~2M → ~32M
params — Leela-classic scale) with an **L1 tax on the metric scales `σ`**,
ramped in after a warmup so the wide field explores before the tax bites. The
principle: **the representation is free; the metric is priced.** Dimensions are
allocated to the *distance* only where a pattern pays for them, so specializing
on the frequent regime no longer requires *overwriting* the rare one — regimes
can occupy disjoint sparse supports. A complementary defense is a small replay
anchor (a few percent endgame data) so rare features have gradients defending
them, not merely capacity insulating them. Pre-registered gates: effective rank
must *rise* and scale with width; the rare/common drift ratio must flatten
toward 1; rook-endgame competence must *survive* to late steps; field-only
mate-in-1 must beat the current 0.18.

---

## 8. What is validated, and how

Every play number is a paired comparison against a **deterministic
tablebase-optimal defender** (removing opponent randomness), reported with a
bootstrap CI and an **anytime-valid e-value** (a nonnegative supermartingale
under the null — monitorable under optional stopping; `e ≥ 1/α = 20` rejects;
independent-run e-values multiply). Selected effects require a **pre-registered
confirmatory** on a fresh single-use position set. A source-inspecting leakage
audit gates every run: engines and tablebases may *grade and oppose*, never
*train*.

Confirmed / strong:

- **Certainty geometry** (train `−ln P̂` into the base objective) beats the
  pre-certainty field head-to-head at full board: composed e = 539.
- **Committor readout** of a full-board outcome head beats the point-goal
  readout on the toy: +0.14 conversion, e = 91.5 at n=200 (single fresh-set
  confirmatory ns at +0.067; composed e ≈ 49 — treated as strong, not
  promoted, pending the purpose-built checkpoint).
- **MCTS over the field** beats beam-minimax at matched compute (confirmed,
  e-gated) — the search that reads the geometry matters as much as the
  geometry.
- **Sharpness ⊥ distance** (Spearman +0.04) and **wall-generated near-mate
  gradient** (Spearman +0.29) — both CI-clean.

Not yet proven (motivated, gated, in progress):

- That the **wide + L1-sparse** architecture raises effective rank and
  decouples regimes (the run this report accompanies).
- That the **three wall-visibility fixes** convert the flat near-mate field
  into reliable conversion.
- That `d_L` and the **goal-selection layer** (§9) produce competent
  losing-side and full-game play (blocked on loss-abundant data, i.e. full
  board).

---

## 9. The full architecture (target)

Putting it together, the planner we are building:

```
Encoders:   F(s,ω), B(g)   — wide trunks (§7), ω = (Elo_w, Elo_b, clock)
State:      augmented — board × {halfmove clock, repetition count} (§6)
Geometry:   score = F·W·B − ‖σ⊙F − σ⊙B‖   — quasimetric + asym residual (§2)
Committors: d_W, d_D, d_L  — −ln P(hit surface first), outcome-head-in-base (§4)
Objective:  L_NCE + λ_ply·L_ply + λ_p·CE(outcome) + λ_1·‖σ‖₁            (§2,3,7)
Readout:    reach = −d_W + β·d_D   (draw clearance keeps the win field
            non-flat at the rim; §6)
Search:     PUCT MCTS over reach, path-aware terminal detection (§6,8)
Goal layer: pick the plan maximizing P_W + ½·P_D from the game's own scoring
            rule — win when you can, steer to the draw surface when you can't;
            NOT a learned value (§4.3)
```

The **plan** is a trajectory through this space: a corridor of soft regions the
play passes through, wide and uncertain far out (high epistemic hazard on every
unverified continuation), narrowing as search verifies specific lines
(epistemic hazard collapses on the verified ones, the aleatoric bottleneck
stands out), a bottleneck where a few states must be searched exactly, then
widening again — and replanning when the corridor's total `−ln P` degrades past
an alternative's. The draw walls, made visible, are what a winning side erects
*behind* itself to funnel play down the mate corridor.

---

## 10. Contribution

We claim four things as novel, in decreasing order of confidence:

1. **Probability-first reachability distance.** Defining the planning distance
   as `−ln P(reach the goal under the acting, fallible policy)` — a quantity
   that is a quasimetric by the multiplicativity of path probabilities, reduces
   to hop-count under perfect play, and is estimable from the agent's own
   rollouts with no oracle. This makes "certainty" and "distance" one currency
   and repairs the min-semantics optimism that a shortest-path or contrastive
   distance bakes in. *Evidence: the certainty geometry's confirmed full-board
   win (e=539) and the measured anti-correlation it fixes.*

2. **Outcome surfaces as goals (committor fields).** Representing terminal
   outcomes as absorbing surfaces and learning per-surface *hitting
   probabilities* (committors) instead of point or set goal-embeddings — with
   touchdown semantics, no goal vector, and the draw surface as a first-class
   navigable field that both repels the winner and attracts the loser. To our
   knowledge, committor / first-passage functions have not been used as the
   goal representation for a learned game planner. *Evidence: the committor
   readout beats point and set readouts at play (e=91.5); draw-committor learned
   at rank +0.68.*

3. **Wall-generated gradient and augmented-state committors.** The observation
   — with tablebase-ground-truth measurement — that in a guaranteed-win region
   the *only* thing making a slow move worse than a fast one is proximity to the
   draw walls, so the near-goal planning gradient is a first-passage property of
   the *augmented* (history-bearing) state, and is destroyed by
   history-blind aggregation, representation, and search. This turns a
   notorious "flat near the goal" failure into a precise, fixable signal-flow
   problem. *Evidence: `P̂` gradient +0.29 vs. flat learned field −0.01.*

4. **Free representation, priced metric.** Decoupling representational capacity
   from geometric capacity: a wide embedding with an L1 tax on the per-dimension
   *metric* scales, so distance dimensions are allocated per-pattern and rare
   regimes need not be overwritten to serve frequent ones. *Evidence so far:
   the diagnosis (effective rank ~7 regardless of width; rare/common drift
   ratio 1.71); the intervention is under test.*

The methodological spine underneath all four — deterministic-defender paired
playouts, anytime-valid e-values, pre-registered single-use confirmatories,
and a source-inspecting no-leakage audit — is what lets a single overnight run
generate a *week* of trustworthy verdicts, and is itself a contribution to how
this kind of research is run.

**What we do not yet claim:** a system that reliably converts even a simple
won endgame under a small search budget. That is the standing gate, and every
element above is in service of clearing it. The honest status is: we have a
principled, measured, and largely-built architecture for planning-by-navigation
whose individual mechanisms are validated, whose known failure (rim flatness)
now has a mechanism and a fix, and whose central open risk (representational
capacity) has a diagnosis and an intervention now under way.

---

*Provenance: JOURNAL.md (round-by-round verdicts), GLOSSARY.md (definitions),
`catspace/nn/fb.py` (score/loss), `experiments/` (all named experiments). Every
quantitative claim is a printed script VERDICT; nothing is asserted in prose
alone.*
