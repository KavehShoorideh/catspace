# Reachability foundations — whose math we borrow, and how it composes (2026-07-28)

Companion to `REACHABILITY_STRUCTURE_BRIEF.md` (the question) — this is the answer. Sources were
gathered by a deep-research pass (2026-07-27/28); the full adversarial verification stage was cut
short by usage limits, so every entry is **graded**:

- **[V]** — survived a 3-vote adversarial refutation panel (3-0), quote-anchored.
- **[Q]** — extracted from the fetched primary source with a supporting quote; refutation panel did
  not run. Treat as reliable citation, unverified interpretation.
- **[K]** — knowledge-based, source not fetched this pass. Verify before citing in a paper.

Architecture context (locked, memory `z_conditioned_field_two_evaluators`): the only learned field is
z-conditioned `P(reach g | s, z_self, z_opp)`; objective/forced reachability comes from search over
legal moves; subgoals = better parts of the outcome basin given both z's.

---

## 1. The two quantifier regimes and the knob between them

**Regime A — forced (∃∀, support-quantified).** `∃ π_us ∀ opponent legal moves: reach G within k`.
Probability-irrelevant. Formal object: the k-step **attractor**; logic: ATL `⟨⟨us⟩⟩◇G`; practical
prover: proof-number search; ground truth: tablebases (retrograde analysis **is** attractor
computation — §2.1). The learned quasimetric could never be this object (it is single-agent MIN, not
minimax), and no measure field can be either.

**Regime B — navigation (measure-quantified).** `P(reach G, avoid B | s, z-kernel) ≥ 1−ε` under the
joint stochastic play kernel. Formal object: MDP max-reachability / stochastic safety–reach-avoid
value; learned instrument: the z-conditioned field. "Take the center where all likely paths are good"
= sit deep in an almost-invariant (metastable) set of the kernel with high committor.

**The knob.** mean → CVaR_α → ambiguity-set radius → ε-support → full legal support (= forced).
CVaR's confidence level *is* an ambiguity-set radius (§2.7), so this is one principled dial, not a
heuristic ladder. Even at the forced end, **our own execution risk multiplies along the line** —
a proven mate is worth `P(we execute | z_self)`, never 1.0; risk budget composes multiplicatively
along a trajectory exactly as in the CVaR augmented-state formulation.

**The reduction that keeps us honest.** With the opponent fixed to a stochastic policy (the z-model),
the two-player stochastic game **degenerates to an MDP** (1.5-player): opponent nodes stop being
adversarial nondeterminism and become chance nodes. So the navigation object is MDP `P_max(◇G)`
(PCTL / value iteration), *not* game semantics — rPATL is what we cite for the unified object, MDP
model checking is what we actually compute/learn. [Q — §2.4]

---

## 2. Whose math we borrow

### 2.1 Forced side (regime A)

| Source | Exact object we borrow | Role here | Grade |
|---|---|---|---|
| Mazala, "Infinite Games," in Grädel–Thomas–Wilke (eds.), *Automata, Logics, and Infinite Games*, LNCS 2500, Springer 2002, pp. 23–38 | i-attractor `attr_i(G,R)`; winning distance `d` (0 on target; min 1+d at our nodes; max 1+d at opponent nodes); attractor = ⋃_k {d ≤ k}; O(n+m) backward BFS with successor counters | "Mate-in-k" ≡ membership in the k-th attractor level — the formal name for the forced object | [Q] |
| Ströhlein 1970 (TU Munich dissertation); Thompson, "Retrograde Analysis of Certain Endgames," *ICCA J.* 9(3):131–139, 1986; Zermelo 1912/13; Bellman 1965 (PNAS) | Retrograde tablebase generation: B_i/W_i strata built by the exact ∃/∀ predecessor iteration | Tablebase construction **is** attractor computation with winning distances — exactly, not morally (successor-counter algorithm is structurally identical) | [Q] |
| Alur, Henzinger, Kupferman, "Alternating-Time Temporal Logic," *JACM* 49(5):672–713, 2002 | `⟨⟨A⟩⟩◇g`; model checking = iterated controllable-predecessor fixpoint (linear time); turn-based synchronous case | The logic of forcedness; bridges attractor ↔ logic; CTL's ∃/∀ = degenerate coalitions (the cooperative map is the ⟨⟨Σ⟩⟩ corner — degenerate, as decided) | [Q] |
| Allis, van der Meulen, van den Herik, "Proof-Number Search," *AIJ* 66(1):91–123, 1994 (thesis: Allis 1994); forerunner McAllester conspiracy numbers (1985 TR; *AIJ* 1988) | pn/dn with min/sum backups; proof tree = ≥1 child at OR nodes, ALL children at AND nodes | The practical ∃-prover; structurally prefers slim (forcing) subtrees — exactly the tactical evaluator of the two-evaluator design | [Q] |
| Nagai 2002 (df-pn); Kishimoto & Müller, *Inf. Sci.* 175(4):296–314, 2005 (GHI fix); Kishimoto et al. 2012 survey | df-pn thresholds; **GHI/repetition incompleteness on cyclic graphs** | Deployment caveat: a "certified forced mate" in chess (repetitions!) must handle GHI explicitly | [Q] |
| Carmel & Markovitch 1993 ("M1 search") **and** Iida, Uiterwijk, van den Herik(, Herschberg) 1993 — independent, cite both | OM-search: asymmetric evals by node ownership | The deterministic opponent-model branch (what AOEE Module 5 rediscovers) — known-inferior to the probabilistic branch for our purpose | [Q] |
| Donkers, Uiterwijk, van den Herik, "Probabilistic opponent-model search," *Inf. Sci.* 135(3–4):123–149, 2001 (+ Donkers thesis) | PrOM: opponent = mixture of minimax types → behavior strategy; **expectimax backup at MIN nodes**; Thesis Thm 3: OM value ≥ minimax value; failure modes: imperfect prediction + type-I errors (own-eval overestimates act as attractors) → admissibility condition | Direct hand-crafted ancestor of expectimax-over-(Elo,z). Two standing lessons: (1) **search-value inflation is structural — never report exploitation gains from search values, only realized outcomes**; (2) our-side execution risk must discount speculative lines | [Q] |
| Jansen, CMU thesis 1992 (advisor: H. Simon); ICCA J. 15(3), 15(4) 1992, 16(1) 1993 (KQKR trilogy; speculative play from 1989) | Trap-setting vs a fallible human in a tablebase-solved endgame | The founding precedent for "planning through errors." Sobering: Donkers' later KQKR replication with *perfect* opponent knowledge + tablebases was **inconclusive** — our bar is realized-game evidence | [Q] |
| Ballard, "*-minimax," *AIJ* 1983 | Pruning in max-vs-chance trees | Search efficiency for the expectimax layer | [K] |

### 2.2 The unifying object and its degenerate case

| Source | Exact object | Role | Grade |
|---|---|---|---|
| Chen, Forejt, Kwiatkowska, Parker, Simaitis, TACAS **2012** (logic; journal *FMSD* 2013); PRISM-games tool = TACAS **2013** | rPATL `⟨⟨C⟩⟩P≥q[◇G]`: ∃ coalition strategy s.t. ∀ opponent strategies the probability bound holds; solved by max/min value iteration; determinacy via Martin 1998 (Blackwell games); NP∩coNP via Condon 1993 | The single formal object unifying strategic existence with a probability certificate — the name of our "ε-forced win with P ≥ q" | [Q] |
| Baier & Katoen, *Principles of Model Checking*, MIT Press 2008 — Defs 10.91/10.92; **Thm 10.100** (p. 851); **Lemma 10.102** (p. 852); Thm 10.105 (p. 855f); §10.6.2 pp. 866–867 | Fixing a scheduler → Markov chain; Bellman system for `Pr_max(◇B)` (unique with the 0-boundary condition); **memoryless schedulers attain the max**; LP formulation; PCTL-over-MDP ∀-scheduler semantics | The degenerate case we actually live in: opponent-as-fixed-stochastic-policy ⇒ MDP; the learned field approximates this fixed point; memoryless sufficiency licenses a stationary field | [Q] |
| Sound value iteration line (interval iteration / optimistic VI) | Naive VI stopping criteria can return arbitrarily wrong reachability values; OVI is the recommended certified default | **Certificate hygiene**: any number we call a "certified" bound must come from sound stopping rules, not vanilla convergence | [Q] |
| Martin 1998; Maitra & Sudderth 1998; Condon 1992/93 | Determinacy of stochastic games; MD optimal strategies in finite turn-based games (fails for concurrent — Everett) | Foundations; the guarantees are specific to finite turn-based structure — which chess satisfies | [Q] |

### 2.3 Measure side (regime B) — the field family

| Source | Exact object | Role | Grade |
|---|---|---|---|
| Eysenbach, Zhang, Levine, Salakhutdinov, "Contrastive Learning as Goal-Conditioned RL," NeurIPS 2022 | Critic `f(s,a,g)=φ(s,a)ᵀψ(g)`; optimal critic = goal-conditioned Q = **discounted state-occupancy probability** (Lemma 4.1, Prop 1); `exp(f*) = Q/p(g)` (per-goal constant); explicitly retrieval-factored | Proof that a factored dot-product critic is a genuine reachability estimator; also the **calibration landmine** (per-goal 1/p(g)) our supervised labels sidestep | **[V]** |
| — same paper, caveats | Occupancy is conditioned on the **behavior policy** that generated the data; on-policy; no player-embedding mechanism | For us a feature: human z-kernels ARE the behavior policies; explicit z-input conditioning is our extension | [Q] |
| Eysenbach et al., C-learning, ICLR 2021 (arXiv:2011.08909) | Goal-reaching as recursive classification; Bayes-rule-normalized future-state density | The properly-normalized classifier route (fallback if we ever drop explicit labels) | **[V]** |
| Moskovitz et al., "First-Occupancy Representation," ICLR 2022 | `F^π(s,s′) = E[γ^T_first]` — first-hit, vs SR's ever/cumulative occupancy; γ couples P(reach) with time | The **first-hit vs ever-reach fork**; our head is the first-hit object (basins: they coincide; subgoals: they don't) | **[V]** |
| Janner et al., γ-models, NeurIPS 2020 | Generative discounted occupancy `μ(s|s_t,a_t)`; TD-as-generative-training; policy-conditioned by construction | Confirms SUM/Bellman-expectation character; the generative sibling we don't need (we need scores, not samples) | **[V]** |
| Touati & Ollivier, Forward-Backward representations, NeurIPS 2021 | Successor measure `M^π(s0,a0,X)=Σγᵗ Pr(∈X)` factored as `F(s0,a0,z)ᵀB(s′,a′)ρ` | The factored z-conditioned successor object — our template, with z's meaning swapped (their z = reward-task projection, π_z reward-greedy) | **[V]** (factorization) / [Q] (z caveat) |
| Borsa et al., USFA, ICLR 2019 (arXiv:1812.07626) | SFs conditioned on a **policy encoding** + GPI | Nearest prior art for policy-embedding-conditioned successor estimates | [Q] |
| InFOM (intention-conditioned flow occupancy models, 2025) | Occupancy conditioned on latent user "intention" z inferred variationally from transitions; generative flow; sample-based readout (N≈16); no first-hit, no MFPT | Nearest prior art for *user*-embedding-conditioned occupancy; establishes the trade-off vs factored critics (not retrieval-shaped) | [Q] |
| Dayan 1993 (SR); Kemeny & Snell, *Finite Markov Chains* | `M=(I−γP)⁻¹`; fundamental matrix `N=(I−Q)⁻¹`: absorption probs `NR` (=WDL), expected steps `N·1` (=MFPT) | The WDL↔MFPT identity — one absorbing chain, two readouts; committor = competing-sets hitting probability | [K] (textbook; panels errored) |

### 2.4 Reach-avoid and the safety measure

| Source | Exact object | Role | Grade |
|---|---|---|---|
| Abate, Prandini, Lygeros, Sastry, *Automatica* 44(11), 2008 | **Safety/invariance** (not reach-avoid — attribution correction): P(stay in safe set) as multiplicative-cost DP; Thm 1: maximal safety via sup-over-controls recursion; "maximal probabilistic safe set at level 1−ε" | The formal "all likely paths stay good" object; the 1−ε safe set = our basin-quality criterion | [Q] |
| Summers & Lygeros, *Automatica* 2010 | Reach-avoid proper ("sum-multiplicative" DP): P(reach target while avoiding bad) | The navigation objective with an avoid-set (don't cross bad basins en route) | [Q] (split) / [K] (details) |
| Hsu, Rubies-Royo, Fisac et al., RSS 2021 | Discounted reach-avoid Bellman (DRABE): contraction for γ<1; conservative under-approx → true set as γ→1 (tabular); deep version = **untrusted oracle** (6.6–23.3% false-success) requiring shielding | If we ever TD-learn reach-avoid values: the sound discounting trick + the warning that the deep net needs a verifying search layer (which we have) | [Q] |

### 2.5 Metastability (already in use)

PCCA+ (Deuflhard & Weber), almost-invariant sets (Froyland & Dellnitz), TPT committor/flux (E &
Vanden-Eijnden) — the "region the kernel doesn't leave" formalization behind `msm_basins.py`. [K —
not re-fetched this pass; already load-bearing in M0/M2a.]

### 2.6 Search-architecture evidence

| Source | Finding | Role | Grade |
|---|---|---|---|
| Ramanujan, Sabharwal, Selman ~2010 | MCTS averaging fails at shallow tactical traps | Empirical case for the two-evaluator split | [K] |
| Baier & Winands 2013–15 | MCTS-minimax hybrid backups | Integration pattern for M5 | [K] |
| ALLIE (Zhang, Jacob, Lai, Fried, Ippolito, arXiv:2410.03893, 2024); Jacob et al. 2022 | Human-trained policy/value/time transformer + AlphaZero MCTS; search-free imitation underperforms 2400+ players by ≥200 Elo, modest search closes it | Independent support: a behavioral field alone is insufficient in tactical regimes — it must be paired with search | [Q] |

### 2.7 The interpolation knob

| Source | Exact object | Role | Grade |
|---|---|---|---|
| Chow, Tamar, Mannor, Pavone, NeurIPS 2015 | **Prop 1: CVaR_α = worst case under budget-constrained multiplicative kernel perturbations** (CVaR level ≡ ambiguity radius); augmented-state (s,y) Bellman contraction; one VI run yields all α; risk budget multiplies along the trajectory; time-inconsistency ⇒ augmentation required | The theorem making "risk level" and "adversary budget" one dial; the multiplicative-budget structure matches our ∏(1−δᵢ) certificate shape | [Q] |
| Nilim & El Ghaoui, *OR* 53(5), 2005; Iyengar, *MOR* 2005 | Robust DP: min-over-ambiguity-set inside the Bellman backup; KL/likelihood sets ≈ classical cost | The worst-case end of the knob as a cheap backup modification (usable at opponent nodes in search) | [Q] / [K] |
| Jacobson 1973; Whittle; Fleming & McEneaney, *SICON* 33(6):1881–1915, 1995 | Risk-sensitive (exponential) control → deterministic game / H∞ in the small-noise limit (continuous-time) | Conceptual license that mean→worst-case is a continuous deformation | [Q] |
| Asadi, Chatterjee, Goharshady, Karrabi, Shafiee, arXiv 2025 (preprint) | Qualitative RMDP reachability (P=1 against every kernel in an uncertainty set), oracle-access algorithms | A third rung — "almost-sure against the whole ε-ball" — between fixed-model P≥1−ε and forced; small-scale only | [Q] |

---

## 3. Novelty ledger (default-refute stance; full adversarial pass incomplete)

- **Opponent-strength conditioning per se: NOT novel.** Maia-2 conditions on both players' Elos and has
  a skill-conditioned WDL head; ALLIE conditions on both Elos via soft tokens; a rating+clock WDL
  model (AUC 0.78) exists. [Q]
- **Style-z per player: Matilda** (arXiv:2606.25176) is the nearest z prior art — 32-d residual over
  frozen Maia-3, Elo-disentangled (rating probes R²≈0.12–0.16) — but it is a one-ply move-policy
  re-ranker only: no reachability, no occupancy, no time head. [Q]
- **Policy/user-embedding-conditioned occupancy: mechanism exists** (USFA policy encodings; FB's
  z-indexed F; InFOM's inferred user intentions) — none with an exogenous opponent strength/style
  embedding, none in a two-player game, none retrieval-factored + first-hit + expected-time. [Q]
- **z-conditioned P(reach g|s,z) to arbitrary indexed goals + MFPT, from human games:** unpreempted in
  everything fetched, including a GCRL survey current through ICLR 2026 (no opponent/rating-conditioned
  goal-reaching, no MFPT entries). Grade: **plausibly novel, pending a completed adversarial pass.** [Q]
- **ε-support-pruned proof search with a ∏(1−δᵢ) certificate:** absent from the PNS survey (nearest
  in-family: threat-space search, Yoshizoe's dynamic widening, MC-PNS — all heuristic or
  eventually-exact, none opponent-model-based, none with a probability certificate) and from the
  OM/PrOM line (expectimax values, no certificates; Jansen's speculative play = hand-crafted).
  Grade: **plausibly novel.** [Q]
- **Subgoals = basin regions conditioned on both players' embeddings:** nothing found; weak negative
  evidence only. [Q]

---

## 4. How it composes (the two-evaluator architecture)

1. **Measure field (learned):** `P̂(reach g | s, z_self, z_opp, c_t) = σ(⟨φ_r(s, z, c_t), ψ_r(g)⟩)` —
   first-hit-within-game BCE on real trajectory labels (bank is known at training time ⇒ no 1/p(g)
   correction needed), plus a censored expected-plies head; WDL = 3-way competing-risks readout
   sharing φ_r. Formally: an amortized `Pr(◇g)` of the induced Markov chain (Baier–Katoen §2.2),
   first-hit variant (FR, §2.3). Serves bank-wide proposal through the existing vector-DB.
2. **Existence prover (search):** df-pn/minimax over legal moves with GHI handling; the field and
   trunk WDL are move-ordering heuristics only — proofs come from the tree. Output at the ε-support
   setting: proof + certificate `P(line holds) ≥ ∏(1−δᵢ)` over pruned opponent mass, multiplied by
   `P(we execute | z_self)`.
3. **The knob** sets opponent-node backups in search: expectation (nominal z-model) → CVaR_α /
   robust-ball → ε-support → full legal (forced). One dial, per §2.7.
4. **Subgoal score (M3):** `P̂(reach g) × basin quality(g)` where quality = committor level ×
   invariance (1−ε safety of g's neighborhood under the kernel, §2.4) — "better parts of the basin
   for us, given our z and their z."
5. **Honesty gates** (standing): search-value inflation is structural (PrOM Thm 3) ⇒ exploitation
   claims only from realized game outcomes; certified bounds only from sound stopping criteria;
   deep value nets are untrusted oracles unless a verifying search layer confirms (RSS 2021 lesson —
   our prover is that layer).

---

## 5. AOEE spec disposition (2026-07-28 review of `chess_agent_spec.pdf`)

| AOEE element | Decision | Reason |
|---|---|---|
| Elo-conditioned prior; metastable-basin framing; A(s) asymmetry | Already ours | = frozen Maia-2 (μ=0), locked metastability plan, crossing-risk primitive (built, ρ≈0.64) |
| Reachability estimator P_φ(s,g,τ,z), BCE | Convergent | Same object as §4.1 minus factorization, censoring, time head, competing-risks WDL |
| **c_t context (clock, Δτ, tilt)** | **ADOPT** | Promote into M2c estimator state + field inputs; blunder rates are strongly clock-dependent |
| **Blunder-severity mining (Q*−Q_actual > δ) as goal channel** | **ADOPT** | Complements density/committor-swing criteria in the atlas pipeline |
| Per-player LoRA | SHELF (trigger: opponents with ≥~1k games) | M2b evidence: smaller per-player capacity already overfit (−0.042 nats); retrieval fix won (+0.006–0.009); kills online cold-start |
| Dual-accumulator asymmetric alpha-beta (deterministic W_opp at MIN nodes) | REJECT as spine; SHELF the distillation kernel (trigger: search depth becomes the bottleneck) | Deterministic opponent = OM-search branch; models only error bias, discards variance (contradicts the thermal-noise framing); known self-delusion failure mode; as written the LoRA→NNUE weight fusion is dimensionally incoherent |
| No existence prover / no certificates in AOEE | Keep ours | Test 2 (prefer sharp traps under time pressure) is unsound without an objective floor |

---

## 6. Next actions

1. Prototype the v1 field head (§4.1) — SHORT run first (fail-fast), on the existing dense player
   trajectory cache; validation gates: held-out players, wrong-z placebo, effective-rank, calibration
   vs realized hit frequencies.
2. Add c_t (clock, Δτ, tilt) to the M2c estimator state and the field inputs.
3. Blunder-severity mining channel in the atlas builder.
4. ε-pruned df-pn prover with GHI handling + certificate arithmetic — after the field v1.
5. Re-run the aborted verification pass on the [Q]/[K] rows before any external write-up (budget
   permitting; the extraction is cached — only vote panels + synthesis remain).
6. **END OF THREAD (Kaveh 2026-07-28): online ẑ_opp.** Opponent-Elo conditioning accepted for now;
   the z_opp slot is ultimately fed by the M2c estimator's CAUSAL in-game posterior ẑ_opp(t)
   (identity-free: ~10 moves discriminates, ~40–80 beats the rating prior cold, immediate with
   history). Train-time must condition on the same causal ẑ_opp(t) (moves ≤ t only — no full-game
   leakage); cold start = the verified ẑ=0 population fallback; train across observation counts so
   calibration holds along the whole game arc.
