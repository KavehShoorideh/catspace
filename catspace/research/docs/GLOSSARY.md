# GLOSSARY — terms of the current architecture

*Rewritten 2026-07-28. Acronyms & symbols are canonical in `../MILESTONES.md` §Acronyms — this
file defers to it and adds the method terms that need more than one line. Pre-rebuild
vocabulary (FB field, sharpness sensors, two-horizon, …) lives in `docs/archive/GLOSSARY.md`.*

## The physics frame

- **Metastable basin** — an outcome class {Win, Draw, Loss} viewed as a region the play-kernel
  rarely leaves; under optimal play the barriers are infinite (absorbing), under real play
  crossings are errors. Formally: almost-invariant sets of the game's Markov kernel (PCCA+).
- **Committor c(s)** — P(reach the win basin | s, play-measure); the outcome coordinate. c≈0.5
  is the transition ridge. Play-measure-dependent: the human committor ≠ the perfect one, and
  the gap is exploitable signal. Outcome-defined ⇒ oracle-free (engines only estimate it).
- **Crossing / transition** — a basin change between consecutive positions, refereed by
  Stockfish (locked). A blunder is a crossing charged to the mover.
- **Crossing risk** (the primitive) — `Σ_m p_model(m)·max(0, c(s)−c(s·m))`: expected committor
  swing under a move model. Opponent's model in → exploitable flux; ours in → self-blunder /
  denial term.
- **Flux Φ / sharpness σ** — Φ = t_win(z_opp) − t_loss(z_me) (net favorable crossing rate; the
  planner's subgoal attribution); σ = t_win + t_loss (swinginess; drives the risk knob).
- **Crystallization zone** — ~15–22 pieces, where human outcome bimodality rises 0.39→0.97;
  where the exploitable leak concentrates.

## The two evaluators (THESIS §3)

- **Forced / ∃∀ object** — ∃ our strategy ∀ opponent legal moves: reach the target. Mate-in-k =
  k-th attractor level. Provable only by search (df-pn/minimax/tablebase); no field represents
  it.
- **Measure / navigation object** — P(reach G, avoid B | s, kernel) ≥ 1−ε. Requires a kernel ⇒
  every measure map is player-conditioned by definition.
- **First-hit field** — the learned `P(first-reach g within game | s, z_self, z_opp, c_t)`
  ≈ σ(⟨φ_r(s,z,c_t), ψ_r(g)⟩); first-hit (not ever-visit occupancy); censored expected-plies
  head beside it; WDL = the competing-risks special case.
- **Retrieval-factored** — goal tower ψ_r is z-free, so the goal bank embeds once and any state
  sweeps it with dot products.
- **The knob** — mean → CVaR_α → ε-support of the opponent model → full legal support (forced):
  one dial over how much opponent probability mass is quantified over. CVaR level ≡ ambiguity-
  set radius.
- **ε-forced win / certificate** — a line proved with opponent branching pruned to model-
  probability ≥ ε, carrying P(line holds) ≥ ∏(1−δᵢ) over the pruned mass.
- **Execution risk** — P(we play the line | z_self); multiplies every value, including truly
  forced ones.

## The player model

- **z** — 16-d per-player residual over the frozen Maia-2 rating base (Matilda-style); allowed
  to carry strength & structure-competence (no purity firewall). μ=0: raw Maia at the player's
  Elo is the universal prior.
- **Infer-then-condition** — the correct use of a recovered z: not as an additive predictor
  (overfits) but as a retriever of the k≈50 nearest clean training styles (Elo-banded);
  predictions come from the blend.
- **ẑ_opp(t) (causal)** — the online estimator's posterior from the opponent's moves up to ply
  t only; what the field's z_opp slot consumes at train AND play time (no future leakage).
  Cold start = z=0 = population-at-Elo.
- **n_obs** — how many opponent moves the estimate has seen; fed with ẑ so the model knows how
  much to trust the slot.
- **Wrong-z placebo** — the identity control: swap in a rating-matched other player's z; a real
  style effect must vanish. The opponent-slot version permutes ẑ_opp trajectories within Elo
  bands at matched n_obs.

## Planning vocabulary

- **Subgoal** — a transition-point region (their error zone, not ours); score = flux ×
  reachability × basin quality. Transition points ARE subgoals; plans chain to TB-won → mate.
- **Optionality portfolio** — soft set of subgoals both sides
  (`soft_reach = (1/β)·logsumexp(...)`); multipurpose moves (advance mine + deny theirs − my
  blunder risk) emerge as the argmax; re-selected every ply with hysteresis.
- **Armed tactic (M7)** — an almost-working tactic stored with its blocking condition (a
  protective factor); watched each ply; activates when the blocker is removed.
- **Expectimax vs minimax** — expected value over the opponent model (fallible foe) vs worst
  case (perfect foe); vs perfect defense the planner reduces to minimax, vs humans expectimax
  measured stronger.
- **Tablebase handover** — at ≤7 pieces the tablebase IS the engine (WDL short-circuit + DTZ
  move); the committor's exact boundary condition.

## Verification vocabulary (TESTING.md)

- **VERDICT line** — the printed script output that is the only admissible source of a number.
- **[V]/[Q]/[K] grades** — adversarially verified / quote-backed unverified / knowledge-based.
- **Acceptance instrument** — the pre-registered eval written before launch; its miniature runs
  in every smoke.
- **eff_rank** — entropy-of-singular-values effective rank; the collapse gate.
- **paired_nll_ci** — per-position paired lift with player-clustered bootstrap (resample
  players, never positions).
