# catspace — Architecture synthesis & the representation decision

Status: **synthesis 2026-07-20** (Opus). Companion to `ARCHITECTURE.md` (the standing spec)
and `UNCERTAINTY_DESIGN.md` / `PLANNER_PROBE_DESIGN.md` / `FIELD_PLAN.md`. Reconciles the
*battle-tested* 2026-07-19/20 geometry-redesign findings with the earlier, largely-untested
design corpus. Bias per Kaveh: **trust what's verified over what was only designed.**

---

## 0. The real problem: we kept oscillating over *how to represent the game*

Kaveh's own diagnosis, and it's correct: the churn hasn't been about tuning — it's been
**going back and forth between different representations of the game.** Every single-object
representation we tried was *right about one thing and failed at another*, which pushed us to
the next one, which failed differently. That's a cycle, not progress. Here is the whole map:

| Representation tried | The question it actually answers | Why it failed *as THE representation* |
|---|---|---|
| **InfoNCE FB embedding** (occupancy / successor-measure) | "does play *later* resemble g?" | Outcome-blind → **tactically blind** (hangs pieces); it learned trajectory occupancy, never win/loss. |
| **Cosine `F@B` score** | soft similarity | Not a metric (no triangle inequality) → multi-hop plans don't compose. |
| **Quasimetric MRN → IQE distance `d(s,g)`** | "can I *get there*, how far?" (cooperative) | Correct object, but *by itself* says nothing about who wins or who forces it. |
| **DTM-hinge** (fit `d` = tablebase DTM) | "plies to mate under *optimal defense*" (adversarial) | Crammed an **adversarial** quantity into a **cooperative** metric → mismatch; flattened global order, collapsed cross-material. |
| **Committor `P(W/D/L)`** | "what's the *outcome probability*?" | Correct object — but it's a *value*, not a *geometry*; can't route/compose subgoals on its own. |
| **Certainty `d = plies + λ(−ln P)`** | distance *and* outcome fused into one scalar | A **readout**, not a base object; fusing them hid which part was wrong (and the λ-cap was a bug). |
| **Two-horizon near/far** | short vs long distance | A **readout split**, not a new representation. |
| **Distributional / categorical** | outcome as a *distribution* (spread = sharpness) | The right **form** for the outcome head — not a competing base object. |
| **Strata / one-way** | "is this move *undoable*?" | A **property of the geometry**, not a separate representation. |

### The resolution: these aren't rival answers — they're **different questions → different layers**

The oscillation comes entirely from trying to make **one field answer three different
questions at once.** It can't, and the failures above are the proof. There are three genuinely
distinct objects, and the discipline is to **assign each question to its own layer and stop
trying to merge them:**

- **"Can I reach it, and how far — ignoring the opponent?"** → **L1**, the reachability
  geometry `d(s,g)` (cooperative, policy-independent).
- **"What is the outcome, and how far to a *forced* result — with the opponent resisting?"**
  → **L2**, the outcome head (committor / categorical, supervised on the tablebase's
  already-adversarial labels).
- **"What would *actually be played* here, at this strength?"** → **L3**, the play measure
  `μ(m|s,ω)` (occupancy).

They *feel* like one object because the **planner uses all three together** — but
coupling-in-use is not sameness-of-representation. You **compose** them at readout/search
time (the certainty distance, coherence `γ`, uncertainty-gated search are all *compositions*
of L1+L2+L3), while the **base objects stay separate**. That composition-not-fusion rule is
what ends the cycle: the moment a design pressure appears ("but the field also needs to know
who wins / who'd play this"), the answer is *"that's a different layer,"* not *"change the
field."*

The one non-obvious insight that makes the split *safe* (no circularity): **optimal-adversarial
DTM is policy-independent, and the tablebase already computed it offline (retrograde search).**
So L2's adversarial labels don't require L3's policy at train time — L1 and L2 can be built
without a working policy. That's why the layering isn't a chicken-and-egg.

**Decision: commit to the three-layer representation. Do not re-litigate "should the field be
reachability or value or occupancy" — it is all three, as separate layers, composed at use.**

---

## 1. L1 — reachability geometry

Verdict key: **KEEP** (proven this session) · **ADOPT** (motivated, low risk) · **EVALUATE**
(needs a kill-test) · **DEFER** · **DROP** (tried, superseded).

| Element | Verdict | Why |
|---|---|---|
| **GroupNorm, not BatchNorm** | **KEEP** | *Proven* (`prove_batchnorm.py`): BN's train/eval statistic mismatch erased a learned 68× one-way asymmetry to 1.1× at inference; BN was the *entire* train/eval gap; GroupNorm restored 47×. Non-negotiable for any directional/strata structure. |
| **IQE quasimetric** (triangle inequality) | **KEEP** | Makes multi-hop plans compose ("stitching") and caps reachable-but-non-adjacent pairs at their path length. Symmetric embeddings *cannot* represent one-way reachability. |
| **Forward hinge `d(F(s)→B(s'))≈1`** | **KEEP** | The known "ply-gap calibration term" — sets absolute scale (InfoNCE only ranks). |
| **Targeted `is_irreversible` hard negative** (push `d(child→parent)` far, irreversible edges only) | **KEEP** | *Proven*: `iqe_geom.pt` reached irrev 8.3× / rev 1.09× at inference. |
| **Pure "huge random push" for strata** | **DROP** | *Refuted twice.* **Turn parity**: a move's reverse is never a 1-ply edge (the turn flips), so a random push can't tell a reversible reverse (reachable in a few plies) from an irreversible one (∞) — it inflates both (observed sep < 1, cross-material collapsing). |
| **Material-reachability repel** (push far only for count-vector-*unreachable* material pairs) | **ADOPT** | Fixes the measured cross-material collapse (KRk→KQk-mate read 1.05, but KQk is *unreachable* from KRk — no pawn to promote). Tune the floor gently (aggressive floors re-inflate the whole scale — observed). |
| **Pool coverage** (nucleus ∪ 1-ply children) | **ADOPT** | Necessary — strata are probed on *derived children*; if they're not in the pool their F-embeddings are OOD and never shaped. |
| **Board-only geometry** (clock+repetition as *separate potentials*) | **KEEP** | Shuffle-equivalent positions cluster; the clock is a second monotone potential the planner reads, not a dimension of `d`. |
| **Per-material DTM order, not a global mate pole** | **KEEP** | *Proven incompatibility*: separation clusters materials, so one pole sits at cluster-determined distances regardless of DTM. `iqe_geom.pt`: per-material mean ρ=+0.454 (all classes positive) vs overall ≈+0.025. |
| **Nucleation** (rigid tablebase nucleus, far field propagates) | **KEEP (roadmap)** | `iqe_nucleus_gn` trained (ρ+0.39); explains why the DTM-hinge (anchor everything at once) failed. |
| **Spectral norm (not a λ hard-cap) for scale** | **KEEP** | The λ-cap was a non-paper-supported bug; spectral norm is the correct Lipschitz control. |
| **Region goals** — Gaussian pooling → box/hyperbolic | **EVALUATE** | For "corner the king" surfaces; Gaussian pooling plugs into F@B almost unchanged. Gate on a concrete cross-material-transfer win. |

**Net L1 = `train_geometry_l1.py`** (targeted irreversibility + material-repel + pool
coverage, board-only, per-material). `train_geometry_min.py` (pure push) is kept only as the
recorded negative result.

## 2. L2 — outcome head (categorical) + uncertainty

| Element | Verdict | Why |
|---|---|---|
| **Categorical over distance bins + draw/loss, CE on exact tablebase labels** | **KEEP** | Bounded/integer distance → fixed bins are clean. Gaussian *rejected* (bimodality "3 or 30 ply" is the signal it can't hold). |
| **Committor reading** (`P(W/D/L)` harmonic, `d_cert=−ln P` additive, Doob-martingale leakage detector; perfect play → the two-player-attractor indicator = the tablebase target) | **KEEP** | The backbone that makes L2 planner-usable (leaf value + leakage audit). |
| **Use the SPREAD, not just the mean** | **ADOPT** | The tactical/positional regime signal → drives search quiescence (§4). I'd dropped it; it's the head's near-free second output. |
| **Aleatoric readout = *across-move* divergence, not within-position spread** | **ADOPT** | *Diagnosed*: within-position spread measures *epistemic* depth-uncertainty; aleatoric = how much the outcome distribution changes *across the legal moves*. No retraining — a readout change. |
| **Axiom preservation** (keep IQE `d` as the planning distance; categorical mean need not be) | **KEEP** | If the categorical mean became the distance and broke the triangle inequality, multi-hop plans stop composing. |
| **Acceptance test** (mate-within-5 guidance vs Syzygy defense) | **KEEP** | The concrete near-mate correctness gate. |

## 3. Co-training vs frozen — the one place to be careful

- **Frozen L1 → L2**: the codebase default; but my frozen-L2 smoke on the *pure-reachability*
  L1 was poor (draw-recall 0.03) — a reachability-only encoder doesn't encode win/draw.
- **Full-weight co-train of the outcome head**: **battle-tested to COLLAPSE the geometry** —
  the committor at weight 1.0 pulled the embedding into 3 W/D/L clusters that *fought the
  metric's need to spread* (the documented "small-world collapse"). **DROP full-weight
  co-training.**
- **Resolution (ADOPT with a guardrail):** (1) re-test frozen L2 on the *corrected* L1 first —
  the poor result was largely an un-converged, strata-less L1, not proof that frozen is wrong;
  (2) if still weak, co-train **small-weight** with an **effective-rank guardrail**
  (`check_representational_collapse` rule — back off the head weight if rank drops); (3) keep
  `d` axiom-clean throughout — only the categorical head co-trains.
- Honest answer to "didn't you want to co-train?": **naïve co-training collapses the metric;
  do frozen-first-on-the-fixed-L1, small-weight-with-rank-guard only if the probe demands it.**

## 4. The planner (search) — the job is *efficiency*

Established reframe: **conversion is SEARCH-limited, not value-limited** (0.567@400 →
0.767@1600 nodes; ties ~0.567 because the value is fine and depth is the bottleneck). So the
planner exists to **shorten the horizon via subgoals.**

| Element | Verdict | Why |
|---|---|---|
| **Landmark / SoRB / ProQ coarse plan** | **KEEP (built)** | `planner_landmark.py` — the horizon-shortening skeleton; it surfaced the cross-material hallucination that motivated the material-repel. |
| **Adversarial min-max search** (forceability is an ∃/∀ two-player attractor → search irreducible); pluggable `MinimaxAStar (AO*/LAO*)` + `MCTS`, **no** single-agent A\* | **KEEP (built)** | Field *proposes*, adversarial search *disposes*. `catspace/planner/search.py`. |
| **Uncertainty-gated quiescence** (high across-move spread → widen/extend; low → far AdaSubS jumps) | **ADOPT** | Replaces the empirically non-monotonic depth schedule; L2 spread produces, search consumes. |
| **Coherence** `γ=e^{−κ(1−P)}` (≈ P(realize the whole line)) | **KEEP** | Trust-the-map / backup depth; the earlier branching-entropy discount is retired. |
| **Certified-label store + `probe()`** (only Wilson/bootstrap-CI / tablebase / deep-search labels terminate a probe; probes deposit compounding boundaries; resign/draw = store lookup) | **EVALUATE→ADOPT** | The one genuinely new component; fixes the "0.60→0.20 soft-terminal" disaster. Build after the executor. |
| **Region progressive-widening** (coarse-probe all, deep-probe few; reuse trees) | **ADOPT (with store)** | The PUCT-over-regions efficiency lever the search-limited finding calls for. |
| **Subgoal density prior** `S=forceability×reachability×density^γ` | **EVALUATE (kill-test)** | Density=epistemics, but density ∝ **draw mass** in human data → could steer into the draw basin. Correlate density vs reliability + realized-draw-rate before building. |
| **Hierarchical primitives** (precondition vectors, MoveIdentity, PlanSelector-RL, give-up rules, BlockReason+wake) | **EVALUATE, graft selectively** | Graft the cheap high-value parts first — **give-up rules + BlockReason/wake** (bounded, re-entrant planning) and **executable-verification** (a hop is real only if a bounded probe reaches it). **Defer** the RL PlanSelector until a non-RL executor exists to bootstrap it. |
| **Planner-as-RL / metareasoning** (VoI over probe/move/resign) | **DEFER** | The endgame; `decide()` loop first. |

## 5. L3 — strength / play measure

- **ω-conditioned `μ(m|s,ω)`** (realized policy per Elo) — **KEEP (roadmap), DEFER build.**
- **Descriptive vs normative eval heads, divergence = trap regions** — **EVALUATE** (cheap probe once L1/L2 land).
- **Resign/draw as belief-actions** (free weak supervision of human committor estimates) — **ADOPT (data labeling).**

## 6. Data

| Element | Verdict | Why |
|---|---|---|
| **Nucleus = Lichess ≤5-piece + exact tablebase DTM**, `result` ≠ `dtm` | **KEEP (built)** | 15,869 tb-wins were *drawn* in the actual game → train L2 on `dtm`, not `result`. |
| **The DTM-data gap: no losing trajectories** (0 black-wins; dtm=0 conflates draw & mate-0) | **FIX — high priority** | Starves *both* L2's loss classes *and* strata ("winner-pov was removed because the ply-gap term needs unrecoverable losing trajectories to learn 'no way back'"). Fix by color-swapping wins→losses and mining black-winning ≤5-piece positions; disambiguate draw vs mate-0. |
| **Far-field propagation** (streaming/pgx on-the-fly gen + persistence) | **DEFER** | Not the bottleneck (encoder-compute-bound; precompute is 0.6s). Slate for self-play; import `pgx` over hand-rolling GPU move-gen. |

## 7. What this session corrected in the prior design

1. **Turn parity** breaks "reverse is its own edge" → strata need *targeted* negatives, not emergence.
2. **Cooperative ≠ adversarial** → don't fit `d` to DTM; DTM → L2 label.
3. **Single mate pole is a category error** → per-material order + tablebase cross-cluster.
4. **Co-training the outcome head collapses the metric** → frozen-first / small-weight-guarded.
5. **Cross-material distances are untrained → collapse** → material-reachability repel.
6. **Aleatoric sharpness = across-move divergence**, not within-position spread.
7. **Data: no losses** starves strata *and* L2's loss classes.

## 8. Recommended build sequence

1. **Corrected L1** (`train_geometry_l1.py`) → verify strata (irr≫rev) + cross-material separation + per-material DTM at inference.
2. **Fix the loss-data gap** (color-swap losses; draw/mate-0 disambiguation) — unblocks strata *and* L2.
3. **Frozen L2 on the corrected L1** + the mate-within-5 acceptance test; small-weight co-train with rank-guard only if needed.
4. **Sharpness readout** (across-move divergence) → search quiescence; score on `sharpness_bench.py` (beat ρ=+0.14).
5. **Executor**: adversarial min-max + coherence over the landmark plan; then **certified-label store + `probe()`** + region progressive-widening.
6. **Graft give-up rules + BlockReason/wake**; defer RL PlanSelector and L3.
7. **Kill-test the density prior** before building it.

Through-line (memory rules): no metric enters the journal without a printed script verdict;
validate short before long; every mechanism is *found*, not hand-coded; and — the meta-lesson
of this doc — **compose the three layers at use; never fuse them back into one representation.**
