# THESIS — what catspace claims, and the architecture of record

*Unified 2026-07-28 from METASTABILITY_PLAN.md, ARCHITECTURE.md (⭐ section), and
REACHABILITY_FOUNDATIONS.md (all preserved in `docs/archive/`). This document is the current
truth; where an archived doc disagrees, this one wins. Roadmap & locked decisions:
`../MILESTONES.md`. Chronology & evidence: `../JOURNAL.md`.*

---

## 1. The claim

Chess engines assume a perfect adversary; humans aren't one. Model the game's outcomes
{Win, Draw, Loss} as **metastable basins**: under optimal play the barriers between them are
infinite; under real play every basin crossing is someone's **error**. Then playing a fallible
opponent well is a navigation problem — steer toward reachable regions where *this* opponent is
likely to make the outcome-flipping error and we are not. The edge is an **information
asymmetry**: we know better than they do where the transition zones are, for them specifically.

Chess is the laboratory; the machinery — reachability fields, opponent-conditioned transition
estimators, subgoal planners — is the research product (headed for publication; see
[[project_purpose_and_publication]]).

**Empirical footing (M0, done — verdicts in JOURNAL):** basins are real and sharp under perfect
play (SF-vs-SF Win↔Loss barrier ≈ 0.00; WDL bimodal at all material; 36% of outcome entropy
explained by position). Human play (1400–1800) leaks (Win↔Loss ≈ 0.27–0.29 per ~6 plies; only 7%
explained by position) and **crystallizes at ~15–22 pieces** (bimodality 0.39→0.97). The
exploitable edge IS the human leak, concentrated in the crystallization zone. Figures:
`docs/figures/engine_vs_human_basins.png`, `docs/figures/committor_by_material.png`.

## 2. Epistemics — no oracle

There is no true midgame V* (chess is unsolved above 7 pieces). Stockfish is a strong but
fallible **reference**, not truth; tablebases are truth only at ≤7 pieces; human outcomes are
truth but sparse and noisy. Three standing consequences:

1. **The committor is outcome-defined, hence oracle-free.** `c(s) = P(win | s, play-measure)` is
   a property of real play, not of an engine's opinion; engines only *estimate* it. It is also
   play-measure-dependent by construction — the human committor ≠ the perfect-play committor,
   and that difference is signal, not error.
2. **The value signal is a swappable module** (weak→strong): single engine → ensemble →
   tablebase (exact, ≤7p) → actual outcomes/self-play. Upgrading the source is a config change,
   not a redesign. Stockfish estimates are weighted by the **SF reliability map** (calibrated
   against tablebase truth; ~97% agreement, errors concentrated on the win/draw boundary).
3. **Grade every claim.** Nothing enters the record without a printed script VERDICT; foundation
   claims carry verification grades (§6); retractions are loud (see `docs/TESTING.md`).

## 3. The architecture of record: two evaluators and a knob

The 2026-07-28 lock ([[z_conditioned_field_two_evaluators]]). Two quantifier regimes require
two different instruments; collapsing them into one was this project's most instructive failure
(§7).

**Regime A — forced (∃∀, existence).** "Mate in 5" = ∃ our strategy such that ∀ opponent legal
moves we reach mate within 5. Probability-irrelevant, quantified over the full legal support.
Formal object: the k-step **attractor** (retrograde tablebase generation *is* attractor
computation); logic ⟨⟨us⟩⟩◇G (ATL); practical prover: **proof-number / minimax search over legal
moves**. No learned field can be this object, and none needs to be: the objective map is
search-computable pointwise, and its optimal-play amortization already exists in the frozen lc0
trunk's WDL/MLH heads. The learned field's only role here is move ordering.

**Regime B — navigation (measure).** "Take the center, where all likely paths are good" =
`P(reach good region, avoid bad | s, play-kernel) ≥ 1−ε`. A measure statement needs a kernel,
and **the only honest kernel is the players'** — so every measure map is player-conditioned *by
definition*. This dissolves the old worry that human data "taints" the field: an opponent-free
measure map does not exist even in principle. The learned instrument is the **z-conditioned
first-hit reachability field**

    P(first-reach g within game | s, z_self, z_opp, c_t)  ≈  σ( ⟨φ_r(s, z, c_t), ψ_r(g)⟩ )

retrieval-factored (goal tower z-free: embed the bank once, sweep it with dot products), trained
with direct first-hit labels from real trajectories (calibrated across goals by construction —
no contrastive 1/p(g) constant), with a censored expected-plies head beside it, and WDL as the
competing-risks special case (the three outcome basins are just three more goals). Context c_t
(clock, time-per-move, tilt) is adopted into the estimator and field inputs (AOEE review, §6).

**The knob.** The two regimes are endpoints of one dial over how much opponent probability mass
is quantified over: **mean → CVaR_α → ε-support of the opponent model → full legal support
(forced)**. CVaR's confidence level is provably an ambiguity-set radius, so this is one
principled dial, not a heuristic ladder. The ε-support point yields **"practically forced" wins
with certificates**: prune opponent branching to moves with model probability ≥ ε, prove the
line, and carry `P(line holds) ≥ ∏(1−δᵢ)` over the pruned mass — "mate in 9 unless the 1350
finds one of three engine-only defenses, P ≥ 0.94."

**Execution risk is never waived.** Even a truly forced line is worth `P(we execute | z_self)`,
not 1.0 — our own fallibility (the self-blunder model) multiplies along every line, exactly as
the crossing-risk primitive's denial term measures.

**The reduction that keeps the math honest.** With the opponent fixed to a stochastic policy
(the z-model), the two-player game degenerates to an MDP (opponent nodes become chance nodes);
the navigation object is MDP max-reachability (Bellman fixed point; memoryless policies suffice
— which licenses a stationary learned field). Keep game semantics (rPATL) only for the
adversarial end of the knob.

## 4. The player model (M2, built)

- **z (16-d) = Matilda-style residual** over a frozen Maia-2 rating base; μ=0 (raw Maia at the
  player's Elo IS the universal prior). z is allowed to carry strength and structure-competence
  — exploitable signal, no purity firewall ([[style_z_allows_strength]]).
- **Infer-then-condition** ([[infer_then_condition_z]]): the recovered z overfits as an additive
  predictor (−0.042 nats held-out) but discriminates identity; used as a *retriever* over k≈50
  nearest clean training styles (Elo-banded) it beats raw Maia (+0.006–0.009 nats) and a
  rating-matched wrong player's z.
- **Online (Elo, z) estimator** (M2c, `catspace/style/estimator.py`): one filter per opponent,
  history-prior + live moves, recency-weighted; identity from ~10 observed moves, beats the
  prior from ~40–80 cold (immediately with history); Elo recoverable from moves alone (MAE 142 @
  40 moves vs 205 uninformed). **The field's z_opp slot is fed by this estimator's *causal*
  in-game posterior ẑ_opp(t)** — identity-free; train-time conditioning must equal play-time
  conditioning (moves ≤ t only); cold start = the z=0 population fallback.
- **Crossing-risk primitive** (`catspace/transition.py`): expected committor swing under a move
  model, refereed by Stockfish — `risk(s|model) = Σ_m p_model(m)·max(0, c(s)−c(s·m))`. One
  primitive, both directions: opponent model in → their exploitable flux; our model in → the
  self-blunder/denial term. Validated ρ≈0.64 vs realized crossings; weaker opponents cross
  1.4–3× more. Where errors happen is position-driven (ranking rating-invariant, Spearman 0.95);
  who errs is strength-driven (magnitude).

## 5. Planning on top (M3–M8, per MILESTONES)

Subgoal score = **crossing flux (T) × reachability (P̂) × basin quality** (committor level ×
invariance of the region under the kernel). Transition points ARE subgoals; plans chain
… → their-error zone → won region → TB-won (≤7p) → mate. The **optionality portfolio**
(`catspace/planner/optionality.py`, built, 16/16 tested) holds a soft set of subgoals for both
sides — multipurpose moves (advance many of ours + deny many of theirs − our own blunder risk)
emerge as the argmax, with opportunistic re-selection each ply and hysteresis against thrash.
MCTS probes when near a subgoal or uncertain, expectimax over the opponent model (measured
better than minimax vs fallible play), minimax vs perfect defense, tablebase handover at ≤7
pieces. Armed tactics (M7): store almost-working tactics with their blocking condition (a
protective factor); watch for its removal; pounce.

Canonical value identities (verified): the ending head gives a distribution over 6 terminal
types; committor c = P(WIN_MATE) is the basin coordinate; expected score V = Σ p_e·score_e is
the planner value; V and c are two readouts of one distribution, conditioned on BOTH agents
(V(s | z_me, z_opp)); flux Φ = t_win(z_opp) − t_loss(z_me) is the *attribution* the planner uses
to pick subgoals, never a competing objective; sharpness σ = t_win + t_loss drives the risk
knob (need-a-win → sharp; winning → quiet = principled contempt).

## 6. Foundations — whose math we borrow, and what's actually new

Grades: **[V]** survived a 3-0 adversarial verification panel · **[Q]** quote-backed extraction,
panel incomplete · **[K]** knowledge-based, verify before external citation. (Full graded tables
with quotes: `docs/archive/REACHABILITY_FOUNDATIONS.md` §2; condensed here.)

**Borrowed:**

| Source | Object we borrow | Grade |
|---|---|---|
| Mazala ch. in Grädel–Thomas–Wilke (eds.) 2002; Ströhlein 1970; Thompson ICCA 1986 | k-step attractor; winning distance; retrograde tablebase = attractor computation (exactly) | [Q] |
| Alur–Henzinger–Kupferman, JACM 2002 (ATL) | ⟨⟨A⟩⟩◇g forcedness; model checking = attractor fixpoint | [Q] |
| Chen–Forejt–Kwiatkowska–Parker–Simaitis, TACAS 2012 (rPATL) | ⟨⟨C⟩⟩P≥q[◇G]: strategy-existence wrapped around a probability bound | [Q] |
| Baier & Katoen 2008, Thm 10.100 / Lemma 10.102 | MDP P_max(◇B) Bellman system; memoryless sufficiency (licenses a stationary field) | [Q] |
| Allis et al., AIJ 66(1) 1994; Nagai 2002; Kishimoto–Müller 2005 | proof-number search; df-pn; GHI/repetition caveat for certified mates | [Q] |
| Carmel–Markovitch AND Iida et al. 1993 (independent); Donkers et al., Inf. Sci. 2001 (PrOM); Jansen 1989–93 | opponent-model search ancestry; expectimax over opponent types; speculative/trap play | [Q] |
| Eysenbach et al., NeurIPS 2022; C-learning 2021 | factored critic ⟨φ,ψ⟩ provably = discounted occupancy (not similarity); the 1/p(g) calibration landmine our labels sidestep | **[V]** |
| Moskovitz et al., ICLR 2022 (FR) | first-hit vs ever-reach distinction; our head is the first-hit object | **[V]** |
| Touati–Ollivier, NeurIPS 2021 (FB); Borsa et al. 2019 (USFA); Janner et al. 2020 (γ-models); Dayan 1993 | z-indexed factored successor measures; policy-embedding conditioning mechanism | **[V]**/[Q] |
| Kemeny & Snell | fundamental matrix: WDL (absorption) and MFPT (expected steps) = two readouts of one chain | [K] |
| Abate et al. 2008 (safety); Summers–Lygeros 2010 (reach-avoid); Hsu–Fisac RSS 2021 | 1−ε safe sets; reach-avoid DP; discounted-contraction trick (+ deep-net untrusted-oracle warning) | [Q] |
| Chow et al., NeurIPS 2015; Nilim–El Ghaoui 2005; Fleming–McEneaney 1995 | CVaR ≡ ambiguity radius (the knob theorem); robust Bellman backups; risk→worst-case limit | [Q] |
| E–Vanden-Eijnden (TPT); Deuflhard–Weber (PCCA+); Froyland–Dellnitz | committor, reactive flux, almost-invariant (metastable) sets | [K, in production use] |
| lc0/Leela trunk + Maia-2 + Stockfish + syzygy; Matilda (arXiv:2606.25176) | substrate, human prior, referee, endgame truth; residual-z design | adopted |

**Plausibly novel** (default-refute stance; panel incomplete — nearest prior art in the archived
ledger): (a) **opponent-conditioned reachability** — P(first-reach g | s, z_self, z_opp) to
arbitrary memorized goals + expected time, learned from human games (nearest: task-latent
successor features, InFOM's intention-conditioned occupancy, Maia-2's both-Elo WDL head — none
exogenous-opponent, retrieval-factored, first-hit, and timed); (b) **ε-support forced wins with
∏(1−δᵢ) certificates** (absent from the PNS and OM-search literatures); (c) **infer-then-
condition** as the correct use of a recovered style vector; (d) the **two-sided crossing-risk
primitive**. Not novel: opponent-Elo conditioning per se (Maia-2, ALLIE).

**Standing honesty gates from the literature:** opponent-model search values are structurally
inflated (PrOM Thm 3) ⇒ exploitation claims come only from realized game outcomes; "certified"
probabilities require sound value-iteration stopping rules; deep value nets are untrusted
oracles unless a verifying search layer confirms (ours is the prover).

## 7. Design history — reversals that define the current shape

Each was killed by a measurement (chronology in JOURNAL; artifacts in `docs/archive/`,
`experiments/archive/`):

| Was | Is | Why |
|---|---|---|
| Two-encoder F(s)/B(g) field | single-space φ | 10.9% triangle violations → 0.00% |
| d=512 embedding | d=64 | endgame eff-rank ~6; widen only on saturation |
| Hand-trained ClockField encoder | frozen Leela-family trunk + heads | adopt-before-build (MILESTONES decision 8) |
| Casual-pool prior μ(Elo) | μ=0, raw Maia = the prior | population mismatch + gauge bug |
| Additive per-player z | z as retriever (infer-then-condition) | −0.042 → +0.006–0.009 nats |
| Conditioned-IQE reachability (d,T merged) | z-conditioned first-hit **probability** field | z-lift ≈ 0, structurally: MIN object = best-case reach = policy-invariant; MILESTONES decision-3 amendment walked back |
| WDL-guided navigation | geometry/probability-first; WDL as labels & analysis | locked decision 1 |
| Trunk-WDL as crossing referee | Stockfish referee | locked; trunk WDL is not the oracle |

## 8. Reading order

Visitor: `README.md` → this file → JOURNAL.md. Builder: `../MILESTONES.md` (locked plan) →
`COMPONENTS.md` (what exists) → `TESTING.md` (how claims are made) → `RUNBOOK.md` (how to run).
Definitions: `GLOSSARY.md`. History: `docs/archive/` + git.
