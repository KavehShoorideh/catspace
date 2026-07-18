# Planner-as-Prober — the hierarchical planner probes regions with bounded MCTS

Design scope (2026-07-17, Kaveh + Opus). Status: **POST-MVP — scoping only, not
to be built until the MVP lands.** The MVP remains: convert the winning toy vs an
optimal defender (executor + committor readout + coherence, single goal = the W
surface). This planner-as-prober layer sits on top of a *working* executor.
Decisions marked ⟐ need Kaveh's sign-off before implementation.

## 1. The shift

Today the planner *sets* a goal and MCTS drives to it. The new structure: the
planner **probes** several candidate regions by running a *bounded* MCTS toward
each, reads back "can I force a good outcome in this region?", and **decides**
among them — including deciding to **offer a draw** or **resign** when every
probe is hopeless.

> MCTS stops being only the executor and becomes an **evaluation primitive the
> planner calls**. The planner must do more than set goals: it must PROBE goals.

## 2. Why probing is necessary (not optional)

The quasimetric gives **best-case, cooperative** reachability — `d(s, region)` is
"plies to the region if both sides steer there." It structurally **cannot** tell
whether a region is *adversarially forceable* (min-sum ≠ min-max, see
GLOSSARY / the cooperative-vs-adversarial note). Only search supplies the
minimax. So to choose among candidate regions the planner has to actually
**search each one a little** — the probe is how it gets adversarial ground truth
before committing. The quasimetric proposes; the probe disposes.

## 3. Components and interfaces

- **Candidate generator** `candidates(s) -> [Region]`. From the geometry:
  (a) the outcome surfaces **W / D / L**; (b) intermediate **waypoint regions**
  on the best-case path to W (`planner/decompose.py` already produces these);
  (c) **fallback** regions (the D surface) when W looks blocked. A `Region` is a
  target descriptor the MCTS leaf can score against — a goal-embedding set
  (goal_bank), a committor head, or a surface.
  **DENSITY PRIOR (Kaveh 2026-07-18): candidate subgoals are ranked with a
  point-density factor — prefer regions where the embedding holds a large
  density of known positions.** Those are known clusters: the field is
  well-estimated there (epistemic error shrinks with sample density), plans
  through them are trustworthy, and maneuvering inside them is easy (many
  known neighbors/transpositions to route through). Score shape:
  `S(region) = forceability x reachability x density(region)^gamma`, with
  density measured over the B-embeddings of the training/experience sample
  (cheap: k-NN radius or kernel estimate on the goal pool; later: density of
  CERTIFIED labels in the label store — the epistemically meaningful count,
  since certified points are where we've actually verified the field).
  Ties into the competence map (density ~ where the field has been reliable)
  and milestoning (dense clusters are natural milestone sites: labels there
  compose most often). Implementation post-spread, with decompose.py's
  `score_pairs` as the plug point.
- **Probe** (API v2, Kaveh 2026-07-18)
  `probe(s, region, opponent, terminate, budget, reuse_tree=None) -> ProbeResult`.
  A bounded minimax MCTS from `s` toward the region, with:
  - `opponent ∈ {perfect*, self, μ_ω}` — the policy plugged into min nodes.
    perfect* = best-response under our deepest search (exact only where
    tablebases are LEGAL, i.e. evaluation contexts; fidelity measurable on the
    toy). μ_ω = the conditioned opponent model (deferred layer; slot exists
    from day one). Labels certified under perfect* are one-sided BOUNDS valid
    against any weaker μ; weak-μ labels never bound perfect play — every
    stored label carries its μ.
  - `terminate` — a tunable termination SPEC over certified surfaces:
    (quantity ∈ {P_W, P_D, P_L}, direction, threshold, max CI width), e.g.
    "stop where lower-bound(P_loss) ≥ 0.95" (the resign surface) or
    "lower-bound(P_win) ≥ 0.99, CI ≤ ±0.02" (a conversion milestone). The
    probe terminates at rules-terminals OR at any region in the LABEL STORE
    meeting the spec. Chess is Markov, so this is exact composition
    (milestoning): labeled regions act as additional absorbing surfaces with
    boundary value P̂(R); P(W|s) = P(hit W) + Σ_R P(hit R first)·P̂(W|R).
    DISCIPLINE (paid for at 0.60→0.20): only CERTIFIED labels terminate —
    empirical CIs from real probe outcomes, tablebase anchors where legal,
    deep-search bounds. Raw network confidence may PROPOSE surfaces, never
    CLOSE them.
  - returns `ProbeResult`:
    - `value interval [P_lo, P_hi]` — backup propagates intervals (min-max
      over frontier CIs), so output uncertainty is honest about the labels
      it stood on
    - `hit surface` — which terminal/labeled region ended each line
    - `coherence` — trust depth at the searched horizon
    - `tree` — cached for reuse / deepening
  Every completed probe DEPOSITS a labeled region (value interval, μ used,
  budget, provenance) into the store: probes compound across time — search
  results become boundary conditions (the three-timescale memory made
  concrete), milestone spacing ≈ the coherence length ξ, and the store's
  certified "P_loss ≥ θ" frontier is an incremental, on-demand attractor
  computation under the chosen μ. Move-quality layering (Kaveh 2026-07-18):
  good(m) ⟺ probe-vs-perfect* preserves the game-theoretic class Δ*(m)=0
  (against perfect play advantage can only be kept or conceded, never
  grown); ranking among good moves = P_μ growth under the modeled opponent
  — Q(s,m) = (Δ*(m), P_μ(W|s·m)), lexicographic.
- **Aggregator / decision** `decide({region: ProbeResult}) -> Action`. Picks the
  best-valued region to commit to; if the best is only ≤ draw, deepens probes or
  generates new candidates; **all ≤ draw → offer draw; all ≤ loss → resign.**
- **Plan memory** (`planner/plans.py`, EXISTS): remembers a probed-and-hopeless
  region with its block-reason + wake-trigger (re-probe only when the enabling
  drift Δ or the refutation key recurs), so we don't re-probe dead regions every
  ply.

## 4. Control flow (the plan–probe–decide loop)

```
at each decision point (state s):
  cands   = candidates(s)                          # geometry + memory
  cands   = drop_blocked(cands, plan_memory)       # skip known-dead unless woken
  coarse  = { c: probe(s, c, COARSE_BUDGET) for c in cands }     # cheap first pass
  hot     = top_M(coarse, by=value·confidence)     # promising few
  deep    = { c: probe(s, c, DEEP_BUDGET, reuse=coarse[c].tree) for c in hot }
  best    = argmax_value(deep)
  if   value(best) is a forced WIN:   commit(best)                 # drive there
  elif all values <= DRAW:
        if value(best) == DRAW:       offer_draw()  /  steer to D-surface
        else:                         resign()
  else:                               commit(best)                 # best available
  plan_memory.update(block the hopeless, set wake conditions)
```

The **resign / draw-offer is the terminal case of the same loop**: once every
region MCTS says "no way to avoid loss/draw," the planner acts on it (Kaveh: a
proper engine plays forever; ours acknowledges the outcome). Confidence for that
call is the committor P(loss)/P(draw) at the probe roots.

## 5. Depth = coherence; budget = promise

- Each probe searches until the region's outcome is **resolved** (committor
  confident + coherence long — the obvious-region soft-terminal) or the budget is
  spent. Forced regions resolve **shallow**; divergent ones need **depth**.
- Budget is allocated across candidates by promise: a cheap COARSE pass on all,
  then DEEP passes only on the top few (progressive widening **over regions**,
  mirroring PUCT's progressive widening over moves).

## 6. Memory / three-timescale (reuse, don't recompute)

- **within a probe**: the tree.
- **across candidates**: transpositions share the eval cache (already in MCTS).
- **across plies**: reuse the previous ply's probe trees (MCTS `tree_reuse`).
- **across the game**: resolved probes sharpen the committor (the recognizer gets
  tighter where we've actually searched — the sandwich-bound / episodic-memory
  idea).
- `plans.py` persists blocked plans with wake-triggers — reuse verbatim for
  hopeless regions.

## 7. What exists vs what's new

**Exists:** `decompose.py` (waypoint candidate generation, pluggable
`score_pairs`), `plans.py` (plan memory + block/wake/persist), `mcts.py`
(minimax + coherence backup + obvious-region soft-terminal), the committor
readout, `goal_bank` (region-as-exemplar-set), MCTS eval cache + tree reuse.

**New:**
1. the **`probe()` interface** — bounded MCTS returning a structured
   `ProbeResult` (value, confidence, coherence, tree). (Small wrapper over MCTS.)
2. the **outer loop** — candidates → coarse probe → deepen → aggregate → decide.
3. **resign / draw-offer** as a game-level action from the decision.
4. **probe-tree caching across candidates and plies**, and deepening-by-promise
   budget allocation.
5. wiring `decompose.py` candidates + `plans.py` memory into the loop.

## 8. The planner is itself an RL agent (its own loop)

The hand-coded loop in §4 is a **bootstrap**; the real planner is an RL agent
whose policy *learns* those meta-decisions. This is the meta-game / plan-
optimization layer (the `PlanSelector` from the hierarchical-planning notes),
sitting ABOVE the game's own RL.

- **Two action sets** (Kaveh 2026-07-17 reframe — this is exactly Russell &
  Wefald's rational-metareasoning split between computations and external
  actions):
  - **INTERNAL actions** (computations; cost tokens, change only the planner's
    knowledge): `probe_region(r)` — run a bounded MCTS probe of region r;
    `set_plan(subgoal)` — adopt/revise the active plan in plan memory.
  - **GAME actions** (external; end the decision episode): `make_move(m)`,
    `offer_draw`, `resign`.
  The planner is the SINGLE agent owning both sets: MCTS is a *computation*
  whose result informs `make_move` — the planner plays the move the probe/search
  surfaced; the search never acts on its own. A decision step is a sequence of
  internal actions terminated by exactly one game action.
- **Meta-state**: the position embedding + the accumulated `ProbeResult`s so far
  this decision + plan memory (what's been tried/blocked). I.e. what the planner
  currently *knows*.
- **Meta-reward**: the eventual game outcome (win +1 / draw 0 / loss −1) **minus a
  cost per INTERNAL action** (probes priced by their node budget). The cost term is what makes it a
  genuine **value-of-information** problem: the policy learns to probe *only* when
  the expected decision improvement outweighs the compute — not to probe
  exhaustively. Resigning a lost position / drawing a dead one avoids wasted
  probes and is positively reinforced vs. flailing.
- **Two-level (hierarchical) RL**: high-level policy over regions/plans (options
  / temporal abstraction), low-level **MCTS minimax inside a region** as the
  primitive. The high level chooses *which region and how hard to probe*; the low
  level *executes/evaluates* within it. `decide()` in §4 is exactly the function
  the meta-policy replaces once learned.
- **Training signal**: MCTS probe outcomes are cheap self-supervision (a probe
  that resolves to a forced win/loss labels the region), and full games give the
  terminal reward. Curriculum: start on the toy (few regions, W/D/L), grow the
  candidate set as the policy improves.

Relation to the executor: the low-level MCTS is unchanged (minimax + coherence +
recognizer); the RL is entirely at the **region/plan** level. So the MVP executor
is a prerequisite, and this layer is trained *on top* of it — it never needs the
executor to be re-derived.

## 9. Open decisions ⟐ (need Kaveh)

- ⟐ **Candidate set**: W/D/L only, or also K waypoint intermediates from
  `decompose`? How many regions per decision (probe cost scales with it)?
- ⟐ **Budget split**: fixed per candidate vs promise-weighted; COARSE/DEEP sizes.
- ⟐ **Resolution thresholds**: what committor confidence counts as a "forced"
  win / draw / loss for the decision (ties to `certainty_stop`)?
- ⟐ **Draw/resign policy**: offer draw only when *losing-and-can-hold*, or also
  when *winning-but-can't-convert*? Resign at what P(loss) confidence?
- ⟐ **Probe = executor?**: is the winning probe's tree **reused as the actual
  move choice** (the probe that wins IS the move), or is the executor a separate
  deeper search? Reuse is much cheaper and unifies probe/execute.

## 10. Phased build (once decisions are set)

- **A** — `probe()` primitive: bounded MCTS → `ProbeResult`. Validate on the toy
  (single region = W surface; does the probe value track true forceability?).
- **B** — multi-candidate generation (W/D/L + a couple waypoints) + hand-coded
  `decide()`; probe = executor (reuse the winning tree).
- **C** — `plans.py` memory integration (block/wake hopeless regions) + probe-tree
  caching across plies.
- **D** — game-level resign / draw-offer.
- **E** — replace hand-coded `decide()` with the learned **meta-policy** (§8): RL
  over `A_meta`, value-of-information reward (outcome − probe cost). Hierarchical
  RL on top of the frozen executor.

A–D are the mechanism; **E is where "the planner is an RL agent" becomes real.**
Each phase is e-value-gated on conversion (and, for D, on not-resigning-won
positions / not-offering-draw-in-won positions; for E, on probe efficiency —
fewer probes for equal-or-better conversion).
