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

## The lemma-discovery vocabulary (2026-10, the analyst's discovery loop)

*Added 2026-10-13 (Kaveh: "you need some sort of a long-term project memory ... in terms of glossary"). The sections above
are the July architecture; this one is the October code (`catspace/incubator/analyst/discovery/`, `rust/analyst_core/src/`
`vocab.rs plan.rs goals.rs`, Lean `lean/ideas/Ideas/`). Where the code still uses an older word, it is named.*

**The words**
- **Base vocabulary** — the deterministic, interpretable primitives every concept is built from (Lean `Ideas.Base.Vocabulary`,
  Rust `vocab.rs`, Python `base_vocabulary.py`): the pieces by name (W our king, L theirs; P, P2, P3 … ours and O, O2 … theirs,
  strongest first, pawns the most advanced first), sets of squares (attacked, hit, box, file, line, between, ahead, passer,
  beyond, xray, moves …), numbers (count, dist, fileof, rankof, dfile, drank, pushes, attackers, defenders, see, material …),
  the 64 square names (from our side), time `(ago k T)`, combinators (and, or, not, sum, min, max, diff, eq).
- **Term** — an expression over the base vocabulary, written as an S-expression: `(count (and (nearer L W) (not (lines P))))`.
- **Chess term** — a named composition of base words, never a primitive: pinned, en prise, skewer, discovery, passed pawn,
  outside the square, opposition, key square … (`CHESS_TERMS`; unfolded by the evaluator and in Lean).
- **Without** (`(without X T)`, 2026-10-14) — T read on the board with every non-king piece on X's squares taken off; X is resolved on the
  board as it is, the names inside T on the board that is left; built relaxed (a king left attacked is allowed). Not enumerated, only in
  compositions: **job** `(job X Y)` = how much worse the exchange on Y gets without X (a guard or blocker gives > 0).
- **If / after / delta / gained / lost** (2026-10-14; not enumerated) — the language of conditionals and change, one family with `ago` (back),
  `without` / `after` (sideways), `if` (branch). `(if C A B)`: C a number term, nonzero holds (a comparison is written `(eq (max t m) t)` for
  `t >= m`, `(eq (min t m) t)` for `t <= m`, `&&` the min); `(after F P T)`: T on the board after the piece on F goes to P (whatever stood
  there taken; pawn to the last rank a queen; the other side to move; built relaxed; "us" unchanged); `(delta T)`: T now minus T a ply ago,
  on an offset of 64 (64 unchanged; no earlier position: 64); `(gained S)` / `(lost S)`: squares in S now and not a ply ago, and the reverse.
- **Layers** — the order the proposers search the terms in: layer 0 the original words, layer 1 the completed static words,
  layer 2 the expensive ones (the short window's tactics, time).

**What the planner follows**
- **Frame** — in the discovery loop: WHERE it applies (a material, e.g. `KRRvKR`, and optionally a CONDITION), WHAT it
  pushes (a ranked list of measures), and its CLAIM: on every move it chooses, the measure falls within k moves of ours
  against every reply, or the goal (a mate, a capture we keep). Stored by key in `generated_frames.json`; `<material>:check`
  is the frame used when in check.
- **Conditional frame** — a frame WITH a condition: key `<material>:if:<condition>` (the condition: terms with `>=` / `<=`,
  joined by `&&`); while the condition holds its measure replaces the material's frame. Formerly called a "stage" (the
  challenger's word, after Sadikov & Bratko 2006); retired 2026-10-13 (Kaveh: "frame and conditional frame") -- old logs and
  the journal still say stage. ("Stage" elsewhere in the code means the verifier's two checks, or a round of the race.)
- **Loops** — not a kind of frame: a frame whose measure falls once per cycle has a longer horizon k; `forced_by_measure`
  covers every k.
- **Condition = partial pattern** — the few words that say where a frame applies ("a rook one rank in front of their king");
  the pattern memory is the set of frames and their conditions.
- **Measure** — a term the frame drives down; ranked lists compare lexicographically.
- **Claim, horizon k** — "falls within k against every reply"; k = 2 for every frame today (`CLAIM_K`).
- **Lemma** — a claim that is PROVED (a Lean theorem) or checked exactly on every position it covers; until then a
  CONJECTURE. The loop also says "lemma" for a promoted frame measure: that is a conjecture with evidence.
- **Earlier meanings of "frame"** — in September, `lemma_gen.py`'s frame was a pattern of piece slots and relations (a
  partial pattern of pieces); in `FrameLemma.lean` a frame is the set of squares bearing on a question; `mating.rs`'s frames
  (K+Q, K+R) are hand-derived measures with proved edge conditions -- the discovery frame descends from these.

**How frames are found and judged**
- **Surprise** — a claim that did not hold (the measure did not fall), or a game that ended below what our own search proved.
- **Hole** — the boards of a surprise, for the pipeline's proposers (graph network, transformer, the known lemmas) to fill
  with a measure.
- **Promotion, e-value** — a mined measure is kept when held-out games favour it: one-sided paired sign e-value >= 20.
- **Challenger** — mines regions where a frame fails (traps, blind spots) and answers with a conditional frame.
- **Explorer** (`explorer.py`) — from ONE surprise, no statistics: LOCATE the decision that threw a forced gain away;
  EXPLORE the positions around it, each labelled exactly (forced / stopped / unknown); SEPARATE what the planner missed;
  REPAIR with a measure; then the guards: CORNERSTONES (every right decision ever labelled, the case base), a COMMITTEE of
  candidates, COUNTEREXAMPLES, HELD-OUT positions, SUPPORT from two surprises, UTILITY on fresh games.
- **Forced / stopped / unknown** — the exact solver's labels (`goals.rs forced_moves`, `forced_goal`): forced within n moves
  against every defence (with its line), proved not within n, or out of budget.
- **Cycle lemma** (`patterns.py`) — an arrangement of the pieces around their king that comes back shifted k moves later;
  checked at every placement on the board; the hypothesis of the proved Lean theorem `forced_by_measure` -- a frame whose
  condition is the arrangement, whose measure is their king's distance to the edge, whose horizon is the period.

**The tower (2026-10-14, `tower.py`, ANALYST_PLAN section 38)**
- **Floor** — a conditional frame `<material>:if:<condition>` with `horizon:k`; its number n is the measure; its claim: from
  every position of the floor some lower floor (or a mate) is forced within k of our moves against every reply. Floor 0 is
  the mate, described from the side that suffers it ("our king has no square the enemy does not hit").
- **Tower** — the floors of one material at one horizon, numbered from 0; a frame dict in the loop's own format, so the chooser
  plays it unchanged. A floor is a LEMMA when its claim was checked exactly on every placement inside its condition (a finite
  check, not a Lean proof) and a CONJECTURE when sampled and held out.
- **Arrangement** (`tower.py`, phase 2) — a floor read as relations: pieces by kind at an offset from their king, order-free, in three views
  (full, ranks, files), with our king and their king's place (edge, corner) as further items; an arrangement and its mirror are one. A
  floor's frame is an arrangement with no unforced position inside; the forced positions none covers are the REMAINDER.
- **Floor number measure** — the lowest floor whose condition holds, as one term over the vocabulary's own combinators
  (indicator `(eq (max t m) t)`, penalty by `diff` and doubling `sum`, `min` over floors): the tower is ONE frame per material.
- **Sharp band** — a material and horizon where the goal (a mate) is forced from some drawn positions and not from others; the
  place where conditions live. The ladder finds it: none forced, bring our strongest piece in; all forced, bring one of theirs.

**The reasoning ledger (2026-10-14, `reasoning.rs`, `discovery/ledger.py`, ANALYST_PLAN section 39)**
- **Fact / question / finding** — a fact is a vocabulary word evaluated with its READ SET (`vocab_reads`: every square it can depend on); a question is
  typed and about named pieces (SAFE, JOB, RECAP, BLOCKS, PIN); a finding is an answered question: a claim, a value interval in pawn units for the
  piece's owner, a status (PROVEN, REFUTED, CONDITIONAL on a premise, UNKNOWN over budget, ESTIMATED for a closed cycle), the squares read, the findings cited.
  Keyed by a canonical form (type, side that acts, the pieces as sorted kind+square, extra squares), hashed.
- **Retraction, stale** — a move touching a square a finding read retracts it, and what cites it; it is marked STALE with the ply, never deleted.
- **Listener / back pocket** — the index premise key -> the CONDITIONAL findings waiting on it ("e5 leaves"); the conjectures not yet true.
- **Candidate, interval, B\* stop, stake** (phase 3) — the moves the findings suggest (captures, checks, promotions, moves of an attacked piece, blocks, kicks,
  discoveries, pawns far advanced, jobless pieces; in check every legal reply) plus the unsuggested legal moves, each with an interval [lo, hi] in pawn units
  (mate 1000): the rule bounds first, then the gain minus the opponent's value of the position after it, found by rolling the ledger forward along the FORCING lines
  (checks, captures, promotions; every reply in check) with the stand-pat value where nothing is forcing. A question counts only on a candidate whose interval still
  overlaps the best's lower bound (its stake); B*: decide when the best lower bound >= every other upper bound (PROVEN), else on the midpoints (ESTIMATED).
- **Threat** (Kaveh 2026-10-14: "Threats are material loss. Or mate.") — against side S in a position P (S to move, not in check): a move m of the other
  side O that O could make if S PASSED (pass(P) = the same board, O to move) after which S loses material with its best defence, or is mated. Its size is the material
  lost (pawn 1, minor 3, rook 5, queen 9; mate = all). Exact form, for checking: our solver, forcing_them_* / forced_moves on pass(P) (a gain of >= 1 forced within k).
  The ledger's REASONED form (reasoning.rs size_kick): m mates -> mate; an unstoppable mate in one after m (every reply of S leaves one) -> mate; else S's stand-pat after m,
  where S saves at most one problem and only with a move that is safe where it lands (a piece with no such move is TRAPPED); a threat when the size is >= 1. Which moves of O are
  sized: every one at the root; deeper, those that take, give check, or leave a piece of ours newly attacked (a discovery included). The threat is DEFINED, not learned; what
  is learned (by watching Stockfish) is how to avoid losing material (the defence frame).
- **Because-chain** — the dependency path from the decision (the pin) down to the facts.

**Other machinery**
- **Goal field** — goals imagined for both sides against a NATURAL OPPONENT (a mate, else the best immediate gain, else a
  pass); each CONTESTED: forced, stopped (with the INTERCEPTORS that stop it) or unknown; serves a forced goal (off by default).
- **Tactic profile / short window** — the fewest moves (1-3) in which each side forces a gain of 1, 3, 5, 9 or a mate.
- **Outcome floor** — the smallest gain that matters at a material balance (the census's outcome curve); judged at the
  decision's balance.
- **Mirror lemma** — `lean/ideas/Ideas/Mirror.lean`: the left-right reflection keeps legality, play and forced wins
  (no castling rights): search one position, know its mirror.
