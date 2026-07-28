# Reachability structure — research brief (for a fresh literature-search session)

**Purpose.** Load this cold into a new session (this one exhausted its web-search budget). It states the
exact object we want, the conceptual conclusions we reached, the prior-work leads to verify, what we
already have to build on, and ready-to-run search queries. Everything below marked *(knowledge-based)*
was reasoned from training knowledge, **not** verified against live sources — verifying/deepening it is
the job of the fresh session.

---

## 0. Project context (one paragraph)

**catspace**: an engine that beats *fallible* opponents by planning *through their errors*. A
**reachability field** maps outcome basins (Win/Draw/Loss) and their transition points under different
**player models**; a planner navigates to those transitions as subgoals; MCTS probes; tablebases finish
≤7 pieces. Substrate is a **frozen Leela/lc0 trunk** (T1-256x10) → board embedding `φ(s)` (64-d); the
trunk also exposes **WDL** and **Moves-Left (MLH)** heads. The player model is **Elo + a style residual
`z`** (Matilda-style residual over a frozen Maia-2 human-move base; we built an online `(Elo,z)`
estimator via "infer-then-condition": recover `z` from a player's moves, retrieve nearest *clean*
training styles). We also built a **crossing-risk primitive**: `Σ_moves P_player(move|z) ·
committor-swing`, refereed by Stockfish — validated (Spearman ≈ 0.64 vs realized crossings; weaker
opponents cross ~1.4–3× more).

---

## 1. THE EXACT OBJECT WE WANT

Given a current state `s` and a **memory bank of stored points** `{g₁ … g_N}` (positions / regions /
embeddings we can index), compute for each:

> **`P(reach gᵢ | s, z)`** — the probability that play *reaches* point `gᵢ` starting from `s`, under the
> (fallible, policy-`z`) player and opponent.

WDL is the **special case** where the goals are the 3 outcome basins. We want it for **arbitrary indexed
points**. Optionally we also want the **expected time** to each point (expected plies = MFPT) alongside
the probability. It must be **retrieval-shaped**: embed the bank once, query any `s` fast.

---

## 2. CONCEPTUAL CONCLUSIONS WE REACHED (the shape constraints)

1. **No absolute "base."** There is no perfect reference (Stockfish isn't perfect; our data is human), so
   reachability was never a player-independent structural distance. It is **intrinsically probabilistic,
   policy-conditioned, and two-player** (the opponent can *veto* — steer away from a target). Human
   trajectory data is not a compromise — it **is** the reachability distribution we want to model.
2. **WDL and expected-plies are the SAME object, two readouts.** *(textbook — absorbing Markov chains,
   Kemeny & Snell.)* For the policy's transition kernel split into transient `Q` and absorbing outcomes
   `R`, the **fundamental matrix** `N = (I − Q)⁻¹` gives **absorption probability `N·R`** (= WDL /
   committor) **and expected steps to absorption `N·1`** (= MFPT). Same matrix, over the policy that
   generated it.
3. **Composition is the expected value (SUM / Bellman expectation), NOT the min (shortest path).**
   `d(s) = 1 + E_{s'∼π}[d(s')]` (MFPT recursion) / `P(reach g|s) = Σ_{s'} P(s→s'|z) P(reach g|s')`. A
   state reachable by many mediocre paths is *more* reachable than one reachable by a single sharp path;
   MIN throws that away.
4. **This is the EXPECTED-under-policy branch, not the shortest-path/quasimetric branch.** *(This is the
   key architectural fork.)*

---

## 3. EMPIRICAL FINDINGS THIS SESSION (why the IQE was the wrong tool)

- The M1 field is an **IQE quasimetric** (Interval Quasimetric Embedding, Wang & Isola) over frozen `φ`.
  It gives a metric with the **triangle inequality by construction = MIN / shortest-path**.
- We tried to make reachability `z`-dependent by FiLM-conditioning the IQE. It conditions *mechanically*
  (pair-order ≈ 0.85–0.90) but **`z`-lift ≈ 0 on every target**: raw ply-gap (structurally
  player-independent), probability-adjusted multi-step target `log1p(ply_gap/P(path|z))` (the observed
  path's surprisal isn't recoverable from the endpoints the field sees), and 1-step (degenerate).
- **Root cause:** `φ` encodes *structural* reachability; the `z`-dependent transition *probability* lives
  in the move policy (the estimator), and a MIN/triangle-inequality metric **cannot represent
  sum-over-paths (probability) reachability**. Wrong operator, not wrong `z`.
- The **crossing-risk primitive** we built is effectively a **1-step slice** of the probability object and
  it *works* — which is consistent with all of the above.

---

## 4. THE STRUCTURE THE LITERATURE POINTS TO *(knowledge-based — verify)*

The retrieval-shaped, policy-conditioned, probability form is the **Successor Representation / contrastive
RL** family:

- **Successor Representation (SR)** — Dayan 1993. `M = (I − γP)⁻¹` is the discounted fundamental matrix =
  policy-conditioned expected occupancy of `g` from `s` = "probability of reaching `g`."
- **Contrastive RL** — Eysenbach et al., *Contrastive Learning as Goal-Conditioned Reinforcement
  Learning*, NeurIPS 2022. Learns a **factored critic `f(s,g) = ⟨φ(s), ψ(g)⟩ ≈ P(reach g from s)`**,
  trained **contrastively**: positive pairs = `(s, g)` where `g` occurs *later in the same trajectory`;
  negatives = random `g`. **This is the direct fit for "index points from memory"** — embed the bank as
  `ψ(g)`, embed `s` as `φ(s|z)`, one dot-product returns `P(reach)` to every stored point.
- **Successor Features** — Barreto et al., NeurIPS 2017 (transfer / generalized policy improvement).
- **Absorbing Markov chains / fundamental matrix** — Kemeny & Snell, *Finite Markov Chains* (the WDL↔MFPT
  link, §2 above).
- **Transition Path Theory** — committor + MFPT + reactive flux (E & Vanden-Eijnden; MSM/PCCA). We already
  use this (`msm_basins.py`).
- **lc0 Moves-Left Head (MLH)** — a *learned expected-moves-to-terminal* head, **already in our frozen
  trunk**. Closest existing artifact for the *time* readout; but moves-to-*any*-terminal under lc0
  self-play, **not per-basin, not opponent-conditioned**.
- **Maia / Maia-2** — McIlroy-Young et al. (*Aligning Superhuman AI with Human Behavior*, KDD 2020) and the
  unified Maia-2. The rating-conditioned human policy = the *kernel* whose reachability we want.
- **Quasimetric RL (the branch to AVOID here)** — Wang & Isola (*quasimetric learning* / IQE); Wang, Pinto,
  Isola, *Optimal Goal-Reaching RL via Quasimetric Learning* (QRL). This is the **MIN/optimal/shortest-path**
  branch — the wrong operator for probabilistic reachability, though possibly still right for a *planner's*
  best-forcing-route object.
- Also worth checking: **temporal-distance / actionable-distance learning**, **C-learning**, **γ-models /
  successor models** (Janner), **first-occupancy representation** (Moskovitz et al.).

---

## 5. NOVELTY TO VERIFY IN THE SEARCH

The *combination* appears novel *(knowledge-based)*: **strength/`z`-conditioned goal-reaching probability
(and MFPT) to arbitrary memorized points — including outcome basins — learned from human game data, used
as the exploitable flux.** Components exist separately (MLH = moves-left; Maia = rating-conditioned
policy; contrastive RL / SR = goal-reaching occupancy; TPT = committor/MFPT) but we're not aware of them
fused. **Key questions to answer:** Has anyone done *opponent-/strength-conditioned* successor features or
goal-reaching probability in chess/games? Any "expected-time-to-outcome, rating-conditioned" model beyond
lc0 MLH? Any contrastive-RL work with a *policy-index* (condition the representation on a player embedding)?

---

## 6. WHAT WE ALREADY HAVE TO BUILD ON (don't reinvent)

- Frozen Leela trunk (T1-256x10) → `φ(s)` (64-d); trunk WDL + **MLH** heads available.
- Online **`(Elo, z)` estimator** (`catspace/style/estimator.py`): style `z` + Maia-2-conditioned move probs
  (infer-then-condition: recover from moves → retrieve nearest clean training styles; Laplace posterior;
  Elo-from-moves via Maia's rating buckets).
- **Crossing-risk primitive** (`catspace/transition.py`): SF-refereed `Σ P_player(move|z)·committor-swing`.
- **`d_to_goal` / `d_to_region` vector-DB retrieval primitive** (already built) — the memory-index + query
  half of "index points from memory."
- The **atlas** (t-SNE/UMAP of `φ`) — a candidate memory bank of `g`'s.
- **Human trajectory data** (lichess 2019-01), player-labeled → `z` (IDs masked); Stockfish committor/WDL
  labels for a subset (M2a `transition_data_labeled.npz`); dense player-decision trajectories cached.

---

## 7. RECOMMENDED HYPOTHESIS TO TEST (after the search confirms the frame)

Build a **contrastive successor-features reachability head**:
`⟨ φ(s | z), ψ(g) ⟩ ≈ P(reach g | s, z)`,
trained **contrastively** on human `(state, future-state-in-same-trajectory)` pairs vs random negatives,
**conditioned on `z`** (player, and ideally opponent), and **served through the existing vector-DB
retrieval** so any memorized point returns its reachability probability. WDL, outcome basins, subgoals,
and armed-tactic triggers all become the **same query** ("reach this point"). Optionally add an MLH-style
**expected-time** head for the plies readout. **Keep the IQE/quasimetric only** if the planner separately
needs a shortest-forcing-route (MIN) object — a different question from probabilistic reachability.

---

## 8. READY-TO-RUN SEARCH QUERIES (for the fresh session)

- successor representation goal-conditioned reinforcement learning hitting probability
- contrastive learning goal-conditioned reinforcement learning Eysenbach 2022 critic factored
- successor features transfer reinforcement learning Barreto generalized policy improvement
- opponent-conditioned / policy-conditioned successor features games chess
- mean first passage time absorbing Markov chain fundamental matrix Kemeny Snell
- Leela Chess Zero moves left head MLH training expected moves to end
- expected game length prediction / moves to outcome chess rating-conditioned
- quasimetric reinforcement learning optimal goal reaching Wang Isola QRL
- transition path theory committor mean first passage time metastability MSM
- temporal distance learning reinforcement learning expected vs shortest path
- gamma-models successor models first-occupancy representation reinforcement learning
- Maia chess human move prediction McIlroy-Young rating conditioned

---

*Written 2026-07-27 at the end of a long design session. The through-line: reachability is a
**policy-conditioned probability (SR / contrastive-RL family), retrieval-factored `⟨φ(s|z), ψ(g)⟩`** —
NOT a shortest-path quasimetric — and WDL is its outcome-basin special case.*
