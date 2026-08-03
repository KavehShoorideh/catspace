# MILESTONES.md — the locked 30,000-foot plan (Kaveh, 2026-07-27)

**This file is the roadmap.** Details inside a milestone may change; the milestone structure and
the LOCKED DECISIONS must not. Re-litigating a locked decision requires Kaveh's explicit say-so,
recorded here with a date. (Origin: Kaveh's 2026-07-27 reset — "the details can change, the
overall 30k-foot plan shouldn't.")

## North star

An engine that beats fallible opponents by **planning through their errors**: use a
**reachability field** to map the outcome basins and their **transition points under different
player models**; train a **high-level planner** to navigate to those transition points via
**subgoals** (transition points are themselves subgoals on the way to mate); use **MCTS to probe**
when near a subgoal or uncertain, guided by the reachability field under the opponent model and
clock; hand off to **tablebases at ≤7 pieces**. Transitions are a function of the ENTIRE game
state: position+history, both players (Elo + style), clocks, possibly past game outcomes.
Chess is the laboratory; the machinery (fields, transition estimators, subgoal planners) is the
research product, headed for publication.

## Empirical foundation (M0 — DONE 2026-07-27)

- Outcome basins are REAL and sharp under perfect play: SF-vs-SF Win↔Loss barrier ≈ **0.00**
  (absorbing); perfect WDL bimodal at ALL material; 36% of outcome entropy explained by position.
- Human play (1400–1800) LEAKS: Win↔Loss ≈ **0.27/0.29** per ~6 plies; only 7% explained by
  position; the midgame is a jumble that **crystallizes at ~15–22 pieces** (bimodality 0.39→0.97).
- ⇒ **The exploitable edge IS the human leak**, concentrated in the crystallization zone.
- Stopgap transition estimator (Maia policy × SF committor-loss) **predicts real blunders**
  (ρ≈+0.54 smoke; top-B quartile 37.5% real-blunder rate vs 0% bottom).
- Scripts: `transition_map/bands/time`, `msm_basins`, `committor_by_material`,
  `sf_wdl_by_material`, `sf_vs_human_bands`, `engine_vs_human_basins`, `blunder_model`.

## Locked decisions

1. **Geometry-first navigation.** The engine navigates on reachability distances — d(s, subgoal
   region), terminating in d(s, TB-won→mate). WDL/committor outputs are permitted ONLY as (a)
   analysis instrumentation and (b) transition **labels** (a transition = a crossing between
   outcome basins). Never as the engine's primary navigation value.
   *Clarification (Kaveh 2026-07-27): the IQE is the MAP, not the basin object — a human-measure
   reachability metric is basin-PERMEABLE by design (a crossing is 1 ply away; the MSM found phase,
   not outcome). Basins/transitions live in T; the subgoal score is the product flux(T) ×
   reachability(d); an adversarial metric would wall off exactly the crossings we exploit.
   "Resisted vs typical reach" (forceability) derives later as a T-weighted path property over d —
   not a second metric.*
2. **Pretrained trunk + IQE head.** The board encoder is a **frozen pretrained Leela-family
   trunk** (Maia/lc0, via `lc0 leela2onnx` → `lczerolens` torch module — VERIFIED 2026-07-27:
   batched forward 277 pos/s CPU, policy+wdl heads, trunk hookable). New head = IQE quasimetric.
   We do not hand-roll board encoders. Encoder input = lc0 112 planes (8-position history
   included; **no clock in the encoder**).
3. **Context enters the TRANSITION ESTIMATOR, not the encoder.**
   `T(φ(s), clock_both, Elo_both, z_both, ply, [past outcomes]) → per-side crossing risk`
   (t_win, t_loss → net flux Φ = t_win − t_loss, sharpness σ = t_win + t_loss).
   *AMENDMENT (Kaveh 2026-07-27) — T is ABSORBED INTO the IQE head; d and T MERGE into ONE
   CONTEXT-CONDITIONED reachability head. Input = [φ(s), φ(g), context=(clock_both, Elo_both, z,
   z_uncertainty, ply)] → `d(s→g | context)`. Context FiLM-modulates the head (encoder φ stays frozen —
   decision 2 intact); zero-init so context=base reproduces the current field (identity, geometry
   preserved). ONE head → MANY reachability maps (per opponent-z / clock / rating). The transition
   probability FALLS OUT of the conditioned geometry: flux Φ and transition points = the balance of
   `d(s→win-boundary|ctx)` vs `d(s→loss-boundary|ctx)`. Supersedes "d z-independent / z only in T"
   (decision 1's basin-permeable-map framing). z + z_uncertainty come from the M2c estimator's Laplace
   posterior. [[infer_then_condition_z]]*
   *WALK-BACK (Kaveh 2026-07-28, recorded per this file's rules): the d/T-merge amendment above
   was tried and EMPIRICALLY REJECTED — z-lift ≈ 0 on every target, and structurally a
   MIN/quasimetric object is best-case reach, hence policy-invariant (cannot carry z). The locked
   replacement: the learned reachability object is a z-conditioned FIRST-HIT PROBABILITY field
   `P(reach g | s, z_self, z_opp) ≈ ⟨φ_r(s,z), ψ_r(g)⟩` (retrieval-factored, z-free goal tower);
   forced objects come from legal-move search (df-pn/minimax/TB) with the field as move-ordering
   only; z_opp is fed by the M2c estimator's CAUSAL in-game posterior ẑ_opp(t). See
   docs/THESIS.md §3 and memory [[z_conditioned_field_two_evaluators]]. Decision 1's
   geometry-first language now reads probability-first: navigation runs on P(reach) × basin
   quality; WDL/committor remain labels + analysis, never the primary navigation value.*
4. **Player model = known Elo part + unknown residual z.** Population prior p(z | Elo);
   per-player offline embeddings where history exists (name-masked ids); **in-game posterior
   tightening** of z from observed moves (cold start = prior). Matilda-style residual design.
5. **Transition points ARE subgoals.** Plans are chains:
   … → transition region (THEIR error zone, not ours) → won region → TB-won (≤7p) → mate.
   The planner holds a PORTFOLIO of subgoals (optionality + denial + opportunism —
   `catspace/planner/optionality.py`, built & 16/16 tested). MCTS probes when near or uncertain,
   guided by the d-field under the opponent model + clock. Tablebase plays ≤7p directly.
6. **One best line.** Always work on the current best-guess architecture. A superseded approach
   is killed immediately — runs stopped, docs marked — no parallel maintenance of inferior lines.
7. **Ops bar.** Every dataset DVC-tracked; every training MLflow-tracked (`catspace/train/
   scaffold.py`); batched tensor ops (MPS/GPU) over per-position subprocesses wherever possible;
   every comparison statistically rigorous (anytime-valid e-values / SPRT / bootstrap CIs, n
   pre-registered). No journal numbers without a printed script VERDICT.
   *Scope note (Kaveh 2026-07-27): the rigor budget applies to OUR OWN novel claims (exploitation
   dividend, planner effects, T quality) — NOT to re-validating the community's solved rankings.*
8. **Adopt before build (Kaveh 2026-07-27).** For any subproblem someone else has solved well,
   ADOPT their best attempt on trust (guided by web search), build on it, and move on — no
   re-validation A/Bs, no bootstrap battles over decisions that aren't ours. Spend our energy
   exclusively where nobody has a good solution (the transition estimator, opponent-conditioned
   exploitation, subgoal planning, armed tactics). Concretely: the field trunk = the best
   community DISTILLATE that runs within laptop constraints — T1-256x10-distilled now (distilled
   from lc0's strongest teachers), upgrade to T1-512x15-distilled when disk allows — BY FIAT,
   trunk A/B testing closed.

## Milestone philosophy & Definitions of Done (Kaveh 2026-07-27)

**Every milestone is an MVP milestone.** "Done" = the DoD floor below is met and its VERDICT is
printed by a script and journaled — NOT "fully formed / perfect." Iteration passes on a done
milestone are separate, recorded work items (M1.1, M2.1, …). DoD numbers are floors, not targets.
If a DoD proves unreachable with the MVP design, that is a plan-level conversation with Kaveh —
never a silent redefinition.

**The cumulative play ladder (the spine). Amendment (Kaveh 2026-07-27): TIME-CONTROL pressure
falls at the END of the project (M7/M8) — intermediate bars run at fixed NODE budgets, untimed.
Time-AWARENESS (clock as an input to T and the planner) stays throughout; only the LATENCY/speed
pressure is deferred.**
| after | minimum play bar |
|---|---|
| M1–M3 | no play bar — substrate/estimator/map quality only (M5 runs parallel to M3) |
| M5 | beats the 0.125 shallow-search baseline vs Maia-1100 at equal NODE budget; node-scaling monotone |
| M4 | **parity or better vs Maia-1100** at a fixed recorded node budget, untimed (SPRT ≥ 0 Elo) + steering demonstrated |
| M6 | **≥ 0.5 score vs Maia-1200** at fixed nodes, untimed (CI floor ≥ 0.45) + exploitation dividend > 0 |
| M7 | **BEATS Maia-1200 UNDER TIME CONTROL: SPRT accepts H1 ≥ +25 Elo** (no regression vs 1100) — the end-of-project bar |
| M8 | compression holds strength under TC (non-inferiority; speed = Elo work lives HERE) |

**Two standard match protocols.** Common to both: Maia-X = lc0 `maia-X.pb.gz` at nodes=1;
alternating colors; diversified openings; claim-draw rules on; SPRT (elo0=0, elo1=+25) or
anytime-valid e-process, n pre-registered; MLflow-logged, PGNs kept.
- **NODE-BUDGET match (intermediate bars M4–M6):** our side at a FIXED, RECORDED node budget per
  move, untimed — no latency pressure before M7. Generous budgets are fine; record them.
- **TIMED match (final bars M7–M8):** blitz 3+2, both clocks live; time management is the engine's
  job (flag = loss; a faster stack buys more nodes — speed converts to strength); hardware pinned
  + recorded (this laptop, MPS).

## Milestones

### M1 — Substrate: Leela-trunk IQE reachability field
Freeze a pretrained trunk (Maia-1500/1900 and/or a strong small lc0 net — pick by gate metrics);
attach an IQE head (+ thin adapter). Train the head on (i) same-game ply-gap pairs from mixed
human + engine corpora (full-phase, openings included), (ii) tablebase DTZ anchors (d_mate),
(iii) repulsion/unreachability.
- **Gates:** pair-order strong (community-distillate trunk, incumbent-beating on the fair all-phase
  protocol); d_progress-vs-DTZ ranks conversion-forcing progress (the anchor is DTZ = distance-to-
  ZEROING/irreversible-progress-to-the-TB-boundary, NOT DTM = distance-to-MATE; DTZ resets on
  pawn-push/capture so rho has a real ceiling < 1 even for a perfect field — the gate is "ranks
  DTZ well," not "matches mate"); opening values sane (opening-blindness fixed structurally);
  eff-rank healthy. NOTE (decision 8): trunk chosen by fiat = best community distillate; gates
  confirm health, they are not a re-validation contest.
- **Kills on green:** ClockField line (v2/v3/v4 plans), committor-greedy readouts.
- Verified 2026-07-27: ONNX conversion + lczerolens torch load + batched forward (277 pos/s CPU)
  + trunk-feature hook (B,64,8,8) — all working.
- **DoD (MVP):** all gates green on the v3 eval protocol, printed by ONE eval-script VERDICT;
  trunk choice recorded with its comparison table; field ckpt DVC'd + MLflow'd; ClockField kill
  executed and journaled. Informational, non-blocking: 2-ply d-guided play ≥ the 0.125 baseline.

### M2 — Transition estimator T(s, context) — the centerpiece
A context head over frozen φ(s): input [clock_mover, clock_opp, Elo_mover, Elo_opp, z, ply, …] →
per-side crossing risk (Φ, σ).
- **M2a** rating + clock conditioning. Requires per-move clocks → extend Stage-A game records
  with `[%clk]` arrays (raw PGNs have them; re-run records build). Train on real games:
  label = realized committor swings (SF-labeled subset) + game outcomes; distill the Maia×SF
  stopgap where labels are thin.
- **M2b** per-player z offline — Matilda-style residual over a FROZEN Maia-2 rating base + frozen φ:
  `logit_P(m|s) = log p_maia(m|s,Elo) + z_P·U(s,m)`, z ∈ R¹⁶, linear in z so recovery is a CONVEX MAP
  fit (Laplace posterior = the M2c hook). Prior `z=μ(Elo)+Δ`: players ≥40 games own a free Δ; the
  <20-game PROVISIONAL pool has Δ=0 so their moves ESTIMATE μ(Elo) (Kaveh 2026-07-27 — "a prior for all
  of them"). Single time control by construction (rapid, to match Maia-2's base; the `time_control`
  column is already in records — no PGN rebuild). Per Kaveh (2026-07-27) z is ALLOWED to carry strength
  + structure-competence — no purity firewall; only validity controls kept (see DoD).
  [[style_z_allows_strength]]
- **M2c** — the ONLINE OPPONENT-STATE ESTIMATOR: a single filter over **(Elo, z)** run for EVERY
  opponent (Kaveh 2026-07-27), fed by the player's own moves — HISTORY as prior + LIVE moves as they
  arrive, **recency-weighted** (style drifts → recent moves count more). Uniform for all: <20-game
  players get a personalized estimate from their few games; >20-game players keep updating live; nobody
  is frozen. The rating prior is only the true cold-start (zero history).
  - **z**: recover from observed moves → INFER-THEN-CONDITION (retrieve k-NN nearest CLEAN training
    styles, Elo-banded), NOT the overfit additive point-estimate. [[infer_then_condition_z]]
  - **Elo**: KNOWN → tight prior; UNKNOWN → broad population prior, ESTIMATED from moves (Maia-2's 11
    rating buckets ARE a rating estimator — the bucket best explaining the moves is the Elo). Prediction
    marginalizes the base over the Elo posterior; the retrieval band widens with Elo uncertainty.
    Graceful by construction: unknown Elo = widest band (still a positive lift — global retrieval +0.006
    vs Maia; Elo-banded ±100 +0.009).
  - Because the move-logit is LINEAR in z, the (Elo,z) belief stays tractable (recursive Laplace/Kalman).
- **Infra:** batched ONNX Maia policies (tensor ops; 277+ pos/s vs ~1/s subprocess) for both
  training targets and MCTS opponent models.
- **Gates:** beats the stopgap B(s,r) on held-out real-blunder prediction (ρ, quartile lift,
  calibration); clock effect real on matched positions (risk ↑ as clock ↓); rating monotonic;
  z adds statistically significant lift over Elo-only.
- **DoD (MVP) — REFRAMED 2026-07-27 (Kaveh; original single-move rho>=0.60 / "beats stopgap" bars
  were mis-specified: a single realized crossing is a noisy zero-inflated Bernoulli so single-move
  rho ceilings ~0.45 even when the RATE effect is large; and the Maia2xSF stopgap has per-move SF
  LOOKAHEAD that T deliberately trades for speed + clock/z-conditioning). Reframed:**
  M2a — (a) matched clock + rating effects significant [MET 2026-07-27: sharp-blitz crossing rate
  58%@low-clock -> 43%@high; sharp Elo 58%@low -> 47%@high; correct signs, controlled for
  sharpness x time-control]; (b) T's context-conditioned crossing-RATE ranks positions/regions
  correctly (rate-level calibration / AUC, not single-move rho); (c) T approximates the stopgap
  ranking within tolerance while clock/z-conditionable and orders-of-magnitude faster.
  M2b — REFRAMED 2026-07-27 (Kaveh relaxed the confound firewall: strength + structure-competence
  ARE exploitable signal, so no anti-strength / anti-repertoire purge). Gate = VALIDITY only, on
  HELD-OUT PLAYERS, player-clustered CIs (stats.paired_nll_ci): (a) recovered z beats raw Maia-2 base
  (A2>A0, NLL-lift CI floor>0); (b) z is THIS player, not generic capacity — beats a rating-matched
  OTHER player's z (A2>A3, the wrong-z placebo); single-TC (rapid); identity-init reproduces base.
  **[MET 2026-07-27 — 3k-player run, 525 held-out players / 62.9k query moves. The DIRECT additive z
  discriminates identity (+0.017 vs wrong-z) but overfits as a predictor (net −0.042 vs base). Kaveh's
  INFER-THEN-CONDITION fix (recover z from the player's own history → retrieve nearest CLEAN training
  styles k≈50 → predict with the blend) flips it: beats raw Maia +0.006 nats (P=1.00) AND player-specific
  +0.005–0.010 vs wrong-player conditioning (P=1.00). z-consumer = retrieve-and-condition,
  experiments/m2b_condition.py. See memory infer_then_condition_z.]**
  M2c — z from few observed moves beats prior-only. **[MET (base capability) 2026-07-27 — 525 held-out
  players. Cold-start break-even (m2c_ingame.py): identity (beats rating-matched wrong-z) from ~10
  observed moves; conditioned z beats the raw-Maia prior from ~40–80 moves; immediate for warm
  (history-prior) opponents. Elo-banded retrieval ±100 lifts the win to +0.009 vs Maia (m2b_condition.py).
  Unknown-Elo (m2c_elo_id.py): Elo IS recoverable from moves — Elo-MAE 142 @ 40 moves vs 205 no-info
  (coarse but monotone-tightening), so the wide-band fallback degrades gracefully. Estimator packaged:
  catspace/style/estimator.py (online (Elo,z) filter, 6/6 self-test). Recency/drift mechanism BUILT;
  validating its benefit needs MULTI-MONTH timestamped data (single-month 2019-01 has ~no drift) — follow-up.]**
  Batched-Maia infra: Maia-2 adopted (~10ms/pos). All via eval-script VERDICTs, CI'd (stats.py).

### M3 — Transition atlas + subgoal generator (the map)
For an opponent context, map reachable high-flux transition regions:
score(region) = crossing flux (T) × reachability (d) [× exemplar density]. Deliverables:
per-context atlas visualizations + a queryable API `(s, context) → ranked subgoal regions`.
- **M3b — Concept mining (Kaveh 2026-07-27):** matched case-control on similar positions that
  did vs didn't transition → **attacking factors** (pins, hanging pieces, king exposure,
  tension, …) vs **protective factors**; hand-coded extractors first, SAE/CAV stack later.
- **Gates:** out-of-sample validation — games passing through predicted-high-flux regions show
  elevated actual crossing rates; concept effects significant under matching.
- **DoD (MVP):** subgoal API live — (s, context) → ranked regions, fast enough for per-move use at
  play budgets (latency measured + recorded); out-of-sample: top-decile predicted-flux regions show
  ≥ 2× base crossing rate, for ≥ 2 rating bands, and the bands' maps measurably differ. M3b —
  ≥ 5 attacking + ≥ 5 protective factors significant under matching. Atlas artifact committed.
  **[MET 2026-07-29.** API = catspace/subgoals.py (approach + AVOID lists, 0.10 ms/query);
  regions live in COMPOSITE (φ-pattern × committor-band) space — pure φ-partitions capped at
  ~1.7× because the transition ridge (c≈0.5) cuts across pattern space; composite cells hit
  3.32×/3.04× out-of-sample on both bands, band maps differ (Spearman 0.573, Jaccard 0.06).
  M3b via the NO-HAND-CODING route (Kaveh's directive): TopK-SAE atoms over frozen-trunk square
  tokens + the matched case-control harness with directional select-even/confirm-odd —
  **16 attacking + 16 protective learned atoms**, all held-out-confirmed (attacking max +0.22 SD;
  protective to −0.47 SD). Standing finding: visible danger PROTECTS, error-affordance is hidden
  and diffuse (three independent methods concur) — hand-coded "attacking features" are
  structurally capped at ~4. Atlas artifact docs/figures/m3_atlas.png; atom catalog
  artifacts/experiments/m3b_atom_catalog.npz. Field substrate = reach_v2 two-z first-hit
  probability field (both style slots CI-positive at 60k games; ECE 0.0005).]**

### M4 — Planner: subgoal-chain navigation (the strategist)
Wire the M3 generator into the built portfolio planner (optionality/denial/opportunism);
chain through TB-won regions to mate; re-plan opportunistically each ply.
- **Gates:** vs fixed Maia — planner-on steers play into predicted-high-flux regions (mean T of
  reached positions ↑ vs planner-off, e-value significant) AND lifts score.
- **DoD (MVP, first play bar):** integrated planner+probe, standard NODE-BUDGET match vs
  Maia-1100 (untimed, budget recorded): SPRT accepts ≥ 0 Elo (parity or better) AND steering
  demonstrated (mean predicted flux of reached positions ↑ vs planner-off, e-value significant).

### M5 — MCTS as the probe (the prober)
Reachability-guided search: node signal = progress on d-to-active-subgoal (+ flux shaping; NO
WDL leaf values); expansion weighted by the OPPONENT MODEL (expectimax over Maia/z policy —
already measured better than minimax: 0.125 vs 0.094); clock-aware via T; TB handoff ≤7p
(built). Planner triggers probes when near a subgoal or uncertain.
- **Gates:** strength-per-node curve vs the Maia ladder; ≥ 0.125 shallow baseline at equal
  budget, scaling with nodes; beats a WDL-guided ablation at equal nodes.
- **DoD (MVP):** vs Maia-1100 at equal budget, beats the 0.125 shallow baseline (significant);
  node-scaling monotone across ≥ 3 budgets (e.g. 200/800/1600); beats the WDL-guided ablation at
  equal nodes (significant); TB handoff + expectimax expansion on. Strength-per-node VERDICT table.

### M6 — Close the loop: the exploiter
Full-stack play vs the Maia ladder (then other bots): measure the **exploitation dividend** =
score(with opponent model) − score(opponent-agnostic) at equal node budget, SPRT/e-values,
in-game z tightening on. Publishable evaluation + digest write-up.
- **DoD (MVP):** dividend > 0 (both variants at the SAME fixed node budget, untimed),
  pre-registered and significant, on ≥ 2 Maia levels; full-stack score ≥ 0.5 vs Maia-1200 in
  standard node-budget matches (95% CI floor ≥ 0.45); in-game z tightening ON; eval write-up
  drafted from MLflow-logged matches.

### M7 — Armed tactics: the conditional-activation store (Kaveh 2026-07-27)

**SEQUENCING OVERRIDE (Kaveh, 2026-08-03):** starting M7's detect/store/feedback machinery now,
ahead of M4/M6 and before M5 clears its gate — M5 plateaued at 0.085-0.095 vs the 0.125 gate
(node-scaling flat, "algorithmic plateau" per the 2026-07-30 close-out) and that plateau is a
separate open problem, not a blocker for this: the armed-tactic mechanism doesn't structurally
need WINNING search, just search that produces candidate lines to watch. Detection reuses
`catspace/controlfield/wdl_decay.py`'s validated SF-search decay check (parked control-field
work, but that one utility is generic and proven — not un-parking the ascent-cone thread).
Formal DoD (beats Maia-1200 under TC) is still gated on the full stack; this is infra work
toward it, not an early MET.

When search finds a tactic that ALMOST works — a transition point about to cross but not ready —
store it instead of discarding it, together with WHY it is not ready:
- **Armed-tactic record:** (region/pattern, the tactical line, payoff estimate, and the BLOCKING
  CONDITION — the specific defensive resource that refutes it, e.g. "the Nf6 guards h7").
- The blocking condition is precisely a **protective factor** (M3b vocabulary): an armed tactic =
  a near-transition whose flux is gated by one identifiable protective factor.
- **Activation watch:** each ply, cheaply check whether the blocking condition was removed
  (defender left, guard broken, pin released). If yes → the tactic ACTIVATES → high-priority
  pounce subgoal for the planner / first-probe line for MCTS.
- Dual use: (a) exploit the opponent's removal of their own protective factor the instant it
  happens; (b) inversely, protect OUR OWN blocking conditions that the opponent's armed tactics
  depend on (feeds the denial/self-blunder term).
- Also a search-efficiency win: discover once, arm, watch the trigger — instead of re-finding the
  tactic every ply.
- **Prereqs:** M3b (protective-factor vocabulary = the "why not ready" language) + M5 (the probing
  search that finds the tactics). Lineage: refines the old "tactics tracking → pounce" idea.
- **Gates:** activation detection correct (unit tests: blocker removed ⇒ fires, else not); in play
  vs Maia, pounce-on-activation converts opportunities the re-search-every-ply baseline misses at
  equal node budget (e-value significant), and/or equal strength at lower budget (efficiency);
  defensive side measurable (fewer careless releases of our own blockers).
- **DoD (MVP — the roadmap's minimum bar):** armed-tactics gates green AND the full stack
  **BEATS Maia-1200 UNDER TIME CONTROL** (standard 3+2 protocol): SPRT accepts H1 ≥ +25 Elo, no
  regression vs Maia-1100 (parity retained). When this VERDICT prints, the roadmap's minimum is met.

### M8 — Distill, optimize, discretize: the small fast engine (Kaveh 2026-07-27)
Once the research prototype exists (through M7), compress it until it is SMALLER and FASTER with
minimal strength loss — under time control, speed IS Elo, so compression climbs the ladder without
new learning:
- **Distill:** teacher → student. The full stack (d-field, transition estimator, planner+probe
  behavior) distilled into smaller nets — including AZ-style amortization of planner+search
  decisions into a single fast policy, and trunk+head into a smaller trunk.
- **Optimize:** inference engineering — quantization-aware ops, fusion, batching, caching, native
  Metal / CoreML or ONNX-runtime export, pruning.
- **Discretize:** (a) weight quantization (fp16 → int8 where it holds); (b) discretizing the FIELD
  itself — the atlas/regions as discrete lookup structures (region graphs, distance tables,
  transition maps) that replace net calls at play time.
- **Gates:** every compression step measured under the standard TC protocol; non-inferiority SPRT
  vs the uncompressed stack.
- **DoD (MVP):** ≥ 5× lower per-move compute (measured pos/s or ms/move on pinned hardware) with
  strength within −25 Elo of the uncompressed stack (SPRT non-inferiority), AND still beats
  Maia-1200 under the standard time control. Stretch (informational): beat Maia-1200 at HALF the
  clock (1.5+1).

**Sequencing:** M1 → M2 → (M3 ∥ M5) → M4 → M6 → M7 → M8. M3b concept mining can run any time after M2.

## Deferred / out of scope (no work without a recorded plan change)
Dockerized service stack; RL-trained plan selector (revisit after M4); non-board endings
(time/resign as outcome classes); viz niceties beyond the atlas.

## Acronyms & symbols (canonical — GLOSSARY.md defers to this)

**Chess & engines**
| term | meaning |
|---|---|
| WDL | Win / Draw / Loss — the three outcomes; a "WDL head" outputs (p_win, p_draw, p_loss). |
| SF | Stockfish — strongest open engine; our near-perfect reference/oracle. |
| lc0 / Leela | Leela Chess Zero — open AlphaZero-style neural engine; "Leela-family trunk" = the body of an lc0-format net. |
| T30/T40/T60/T70/T79/T80, BT4 | lc0 TRAINING-RUN generations — each a from-scratch self-play RL run at some net size; "T70 net 703810" = the last (strongest) network of run 70, a 128×10 conv SE-ResNet (~3100-Elo class) — our chosen field trunk. Newest runs (T80/BT4) are transformers: stronger but ~35× slower on our torch path. |
| Maia | (name, not acronym) lc0-format nets trained to predict HUMAN moves at a rating band (maia-1100…1900); our rating-conditioned human policy. |
| AZ | AlphaZero — DeepMind's self-play engine; "AZ-style" = policy+value net + PUCT search. |
| TB | (Syzygy) tablebase — precomputed PERFECT play for all ≤7-piece positions. |
| DTZ | Distance To Zeroing — TB metric: plies to an irreversible move (capture/pawn push/mate) under perfect play. Our exact distance anchor. |
| DTM | Distance To Mate — plies to checkmate under perfect play (older experiments). |
| Elo | rating scale (named after Arpad Elo — NOT an acronym). |
| PGN / FEN | Portable Game Notation (game text) / Forsyth–Edwards Notation (single-position text). |
| UCI / SAN | Universal Chess Interface (engine protocol) / Standard Algebraic Notation (move text). |
| CCRL / TCEC | Computer Chess Rating Lists / Top Chess Engine Championship — engine-game archives. |
| KQvK, KRRvKBP… | material classes: White's pieces "v" Black's (K king, Q queen, R rook, B bishop, N knight, P pawn). |
| ply | one half-move (one player's move). |
| ACPL | Average CentiPawn Loss — mean eval loss per move, hundredths of a pawn (legacy metric here). |

**Our method (math / ML)**
| term | meaning |
|---|---|
| IQE | Interval Quasimetric Embedding — the head that turns two embeddings into an ASYMMETRIC distance with the triangle inequality guaranteed by construction (Wang & Isola). Our d(·,·). |
| quasimetric | a distance where d(a,b) ≠ d(b,a) is allowed (chess reachability is directional). |
| φ (small phi) | the board EMBEDDING vector from the trunk. |
| d(s,g), d_mate | learned reachability distances: position→region, position→mate boundary. |
| committor c(s) | from transition-path theory: P(hit the win boundary before the others) UNDER A PLAY MEASURE. Play-measure-dependent (human c ≠ perfect c). |
| T(s, context) | the TRANSITION ESTIMATOR (M2): per-side basin-crossing risk given clocks/Elos/z/ply. |
| Φ (capital Phi) | net favorable flux = t_win − t_loss. ⚠ not the same symbol as φ the embedding. |
| σ (sigma) | sharpness = t_win + t_loss (how swingy a position is). |
| z | the UNKNOWN part of the player model — a style-residual embedding on top of known Elo. |
| MCTS | Monte Carlo Tree Search — build a tree by repeated select/expand/evaluate/backup. |
| PUCT | Prior + Upper Confidence bound applied to Trees — the AZ selection rule MCTS uses. |
| expectimax / minimax | back up the EXPECTED value over opponent replies (fallible foe) vs the WORST-case (perfect foe). |
| MSM | Markov State Model — discretize states, count transitions, analyze the matrix. |
| PCCA(+) | Perron Cluster Cluster Analysis — spectral grouping of an MSM into metastable basins. |
| TPT | Transition Path Theory — committors, reactive flux, transition rates. |
| MFPT | Mean First Passage Time — expected steps to first reach a target set. |
| SAE | Sparse AutoEncoder — unsupervised concept dictionary (M3b later stage). |
| CAV / TCAV | (Testing with) Concept Activation Vectors — supervised concept directions (M3b). |
| NLL | Negative Log-Likelihood (training objective). |
| OOD | Out-Of-Distribution — inputs unlike the training data. |
| UMAP / t-SNE | nonlinear 2-D projections for the maps (Uniform Manifold Approx. & Projection / t-dist. Stochastic Neighbor Embedding). |
| KDE | Kernel Density Estimate — the smooth histograms in ridgeline plots. |

**Metrics & statistics**
| term | meaning |
|---|---|
| ECE | Expected Calibration Error — mean gap between predicted probability and observed frequency (0 = perfectly calibrated). |
| MAE | Mean Absolute Error. |
| ρ / Spearman | rank correlation (−1…+1); our ordering-quality metric. |
| eff_rank | effective rank — how many dimensions an embedding really uses; the collapse gate. |
| CI | Confidence Interval. |
| SPRT | Sequential Probability Ratio Test — stop-when-evidence-suffices A/B for engine matches (fishtest-style). |
| e-value / anytime-valid | evidence measure you may inspect at ANY time without p-hacking; our A/B harness. |
| A/B, h2h | two-variant controlled comparison; head-to-head. |

**Infrastructure**
| term | meaning |
|---|---|
| DVC | Data Version Control — git-style versioning of datasets/models (pointer files in git, bytes in cache). |
| MLflow | experiment tracker — logs params/metrics/artifacts per training run (`mlflow ui`). |
| ONNX | Open Neural Network Exchange — portable net format; `lc0 leela2onnx` → loadable in PyTorch via lczerolens. |
| MPS | Metal Performance Shaders — Apple-silicon GPU backend for PyTorch (our `device="mps"`). |
| HP | hyperparameter. |
| WAL | Write-Ahead Log — SQLite journal mode (banned for the probe cache; grew unbounded, filled the disk). |
| npz / parquet | NumPy zipped arrays / columnar table format (our tensors / our game records). |

**Legacy (appears in old files & journal history only)**
| term | meaning |
|---|---|
| FB, F(s)/B(g) | the OLD two-encoder Forward/Backward field (superseded by single-space φ; survives in filenames like `lichess_fb*.pt`). |
| ω (omega) | the old strength/time conditioning of F (superseded; context now enters T, not the encoder). |
| InfoNCE | the contrastive loss the legacy FB field used (Noise-Contrastive Estimation family). |
