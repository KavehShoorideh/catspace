# THINKING_AGENT — a game-playing agent that thinks in discrete machine language

**Status: PARKED (2026-09-01).** Captured from the 9/1 design discussion so it does not
derail current work. This is NOT part of the locked MILESTONES.md roadmap; activating it
is a deliberate pivot that requires Kaveh's explicit unlock. Nothing here is built.

---

## Why this document exists (the empirical trigger)

Two measurements on 9/1 reframed the qgt line's plateau:

- **Oracle-dynamics ceiling probe** (`qgt_hang_probe --oracle-dyn`): scoring candidate
  moves by *encoding the true child board* (perfect dynamics, same readout, same replies)
  moves blunder detection only 26.6% → 27.5% (qgt25) and 27.5% → 37.6% (qgt21), vs the
  ~71% the arena threshold requires. **The linear-readout-on-quantized-bits value pathway
  is the binding constraint, not transition fidelity.** The qgt25 "fidelity ceiling"
  story is falsified; qgt26 (2x steps) targets the wrong constraint.
- **Variance reality** (qgt15b: 23.9→0.0; qgt19b: 24.8→7.3 on seed alone): 3-seed
  protocols are not affordable on this hardware. **Doctrine: only levers with much
  bigger jumps count as evidence.** Small deltas are noise by construction.

Kaveh's resulting inclination: give up the linear value readout and state-level
(bit-legibility) interpretability. Relocate interpretability elsewhere (see below).

---

## The four ideas (in the order they were developed)

### 1. Layered, goal-conditioned concepts that transfer across game variants

Learn the game in layers: **base = mechanics** (how pieces move; the GNN), **middle =
concepts** that aid reaching objectives, **top = the objective**. Change the objective
and *some* concepts keep their meaning while their **valence** flips: in antichess a pin
still exists mechanically, but it becomes desirable. Express that reasoning explicitly
and make it transfer across variants.

Refinement (Kaveh): do **not** hand-code the transferable concepts (python-chess
predicates break on e.g. atomic chess). Instead **let the system discover the
partition**: generate a family of variants by turning rule knobs, train across them,
and let shared vs variant-specific concepts emerge. On an unseen variant: lean on shared
concepts, and use exact-rules mental simulation / MCTS to learn the changed and new
concepts on the fly (a few imagined rounds → a mapping old↔new concepts → goal-conditioned
play toward subgoals).

Literature anchors: successor features (value = ψ·w; w is the objective — the pin's
valence is literally a w entry), forward-backward representations (zero-shot new
rewards), HiP-MDP / contextual world models (family of MDPs indexed by knobs; infer θ in
a handful of trials), sparse mechanism shift (Schölkopf: rule changes alter few
mechanisms — "some concepts flip, most stay"), EMPA/theory-based RL (mental-simulation
adaptation at human speed), domain randomization/XLand (knob diversity → transfer),
Ludii's General Board Game Concepts (hand-defined version; ours is learned).

Design cautions agreed:
- **One shared-trunk model conditioned on the knob vector** — never N separate models
  compared post hoc (concept spaces won't align).
- The partition must be **incentivized** (sparsity price on knob-sensitivity), not hoped
  for; IRM-style penalties are finicky — prefer measured sensitivity.
- **Separate reward-knobs from dynamics-knobs** (antichess = reward-knob, easy case;
  atomic = dynamics-knob, hard case). Condition mechanics layer on dynamics-knobs,
  valence layer on reward-knobs.
- Chess advantage over the general literature: variant rules are *executable*, so mental
  rollouts are exact — only valences/concepts need re-estimating, never the dynamics.

**Gauge (big-jump-shaped): leave-one-variant-out.** Train on K−1 variants; measure
games-to-competence on the held-out variant relearning only valences/adapters vs from
scratch. Transfer shows up dramatically or not at all. Start knobs on reward variants
(antichess, KOTH, racing kings) before dynamics variants (atomic, crazyhouse).

### 2. Hierarchical latent chain-of-thought (thoughts in machine language)

An autoregressive **thought generator** emits thoughts conditioned on recent thoughts
(bounded, *decaying* window — thoughts fade) and on inputs from the board. A **pool of
persistent threads** is held (limited-capacity workspace: we never attend all thoughts).
Thoughts can trigger lower-level **actions/tools**: look at the board, run the
shared-concept trunk, run MCTS, train new concepts on the fly, make a move. The chain of
thought is explicitly **not natural language** — discrete machine tokens.

Prior art per organ (no prior art for the organism): latent CoT (COCONUT — continuous
thoughts implement implicit parallel BFS; 2025 survey field), HRM/TRM (two coupled
recurrent modules at different timescales, 27M params, latent reasoning = internal
policy improvement), sliding-window + compressive memory (decay), options/Feudal/
Director + slot memories + global workspace (threads; limited capacity is a *feature*),
metareasoning/metacontrol (Hamrick metacontrol chooses how much to imagine AND which
model to consult; Thinker makes planning steps actions in an augmented MDP — note this
is already our locked search-as-RL-action design), cognitive architectures (Soar/ACT-R:
the symbolic ancestors).

**Distinctive card: the thought alphabet is the concept space.** COCONUT-style thoughts
are opaque vectors; ours are tokens in a quantized, transfer-validated concept
vocabulary — machine-speed latent reasoning that stays legible. Unifies three standing
roadmap items: subgoals-as-tokens, planning-in-code-space, search-as-action.

### 3. Generator + validator + reorganization

As thoughts are emitted, a higher-level **validator** tries to fit each new thought into
the self-consistent organization of the thought history. Three outcomes:
1. **Fits** → assimilate; the thought adds value.
2. **Doesn't fit and is dismissible** (irrational / low score) → reject.
3. **Doesn't fit but cannot be rejected** → it **breaks the organization** and forces a
   reorganization (= a plan change), which feeds back into the generator, which now
   thinks along the reorganized line.

This exact three-way loop is named prior art in five traditions: Piaget
(assimilation vs accommodation), Kuhn (anomaly → paradigm shift), AGM belief revision +
TMS (epistemic entrenchment decides reject-vs-revise — the axiomatized version),
Thagard's explanatory coherence (constraint network; one strong element flips the global
assignment), ART (vigilance threshold: resonate-and-learn vs reset-and-reorganize),
predictive processing (precision decides discount-vs-model-update), attractor dynamics
(reorganization is punctuated basin-jumping), Soar impasses, Minsky's critic–selector,
plan-repair literature.

**The gap that is ours:** modern ML validators (PRMs, Tree-of-Thoughts, Reflexion)
*prune and veto — they do not restructure*. No learned latent implementation exists of
"keep the anomalous thought, break the framework, re-derive the organization." The
reorganization operator is the novel contribution. **The literature's one warning:**
every formalism reduces the fork to a single threshold (vigilance / precision /
entrenchment / coherence delta) and its calibration is the whole game — too low =
paradigm thrash, too high = dogmatism.

### 4. Structure, not method: the pluggable framework + knob harness

Fix the **structure** and make the thinking algorithm pluggable. The three interfaces to
pin (everything else swaps):
1. **Thought-token space** (discrete vocabulary + decaying window + thread slots);
2. **Tool/action API** (sense/encode board, run trunk, search, move, learn);
3. **Trace format** (the auditable trail).

Prior art: CoALA fixes exactly this skeleton but is language-locked and a taxonomy, not
a trainable harness; Thinker's augmented MDP is the formalism (thoughts = inner
actions); **Searchformer / Stream of Search is the working training recipe** (supervised
distillation on serialized search traces as discrete tokens → then RL/self-improvement;
models improve *beyond* the teacher search; solution-only training scales worse).
Cognitive architectures' adoption lesson: **the framework must be the byproduct; a
demonstrated result is the product** (Gym won via minimal interface + benchmark).

Used as a harness: plug in a game as a **knob family**, plug in a thought algorithm, get
standardized outputs — auditable traces, transfer curves, interpretability metrics.
Substrates to borrow, not build: OpenSpiel, Ludii (ludemes = compositional rule-knobs),
python-chess + Fairy-Stockfish (chess variants + oracles).

**The unifying instrument: the thought trace across knob settings.** Tokens that persist
across knobs = shared concepts (idea 1, measured directly); tokens that appear/vanish/
flip = variant-specific; token-remapping speed on a held-out variant = the adaptation
claim. One trace serves interpretability and transfer simultaneously.

---

## Architecture requirements (settled in discussion)

- **A continuous latent substrate is required**; discrete thought tokens are a *readout*
  of it. Two coupled loops: continuous inner dynamics, discrete outer trail (the LM
  pattern: continuous residual stream, discrete tokens at the boundary).
- **The load-bearing decision: what crosses a thought boundary without passing through a
  token.** All-through-tokens recreates the qgt capacity failure; free continuous
  carryover makes the trail decorative (CoT-unfaithfulness). Chosen middle:
  **continuous within a thought (HRM-style fast iterations), tokenized between thoughts**
  plus a bounded, decaying continuous residual; threads hold tokens, not vectors.
- **Faithfulness is a printed gauge, not a claim:** periodically roll forward from the
  trail alone (hidden state dropped) and require competence; the trail-only vs
  full-state gap is the faithfulness number. (Closure/Dreamer discipline transplanted to
  thoughts.)
- **Interpretability = post-hoc labeling of tokens** (they are discrete; a trail exists;
  map tokens to text by hand/probing later). Vocabulary lives in the continuous space →
  labeling via probes against oracle predicates (rule-miner retargeted), and vocabulary
  growth on a new variant = allocating new codes, not retraining.
- Vocabulary pressures so tokens stay labelable: small vocab, capacity limits (the
  decay + limited workspace help), and **seed meanings by distilling from search traces
  and concept tokens** — emergent codes without pressure come out polysemous (qgt15
  lesson; emergent-communication literature).
- Training recipe: **distill-from-search-traces first, then RL** — never RL-from-scratch
  over discrete thought choices (variance).
- All quantizer scar tissue transfers: dead-code monitoring, effective-rank gates,
  collapse barriers, temporal-coherence alarms.

## Requirements distilled (Kaveh's own constraints)

1. Thoughts in **discrete machine language** — not natural language; hand-labelable
   post hoc; the trail is the observability/interpretability object.
2. **Big-jump gauges only** (3-seed protocols unaffordable): leave-one-variant-out
   transfer; faithfulness gap; trace-follows-teacher → beats-teacher → transfers.
3. **Staging: small game → robust → chess → variant transfer.** Recommended start:
   Gardner 5×5 minichess family (concept space continuous with chess; natural knobs:
   forced captures, win conditions, board size) or Countdown (Stream-of-Search
   precedent).
4. Framework released (if ever) only alongside a result.
5. **Asset conservation** (nothing from the current project is discarded): encoder trunk
   → the "encode board" tool; concept tokens → seed thought vocabulary; MCTS + oracle
   fleet → the search tool and distillation teacher; rule miner → trace interpreter;
   health-gate stack → quantizer monitoring.

## Relationship to current work

Near-term (active, NOT parked): qgt26 verdict; then the readout-replacement experiment
(deep value trunk in place of the linear readout on the same states) — the one-run
big-jump test the ceiling probe licenses. This parked program is the destination if/when
Kaveh calls the pivot; until then the qgt/gcf lines proceed under MILESTONES.md.

## Key references

Successor features (Barreto et al.) · FB representations (Touati & Ollivier) · HiP-MDP
(Doshi-Velez & Konidaris; GHP-MDP) · sparse mechanism shift (Schölkopf et al.) · EMPA
(Tsividis et al., arXiv 2107.12544) · Ludii concepts (arXiv 2107.01078) / variant
transfer (arXiv 2102.12375) · COCONUT (arXiv 2412.06769) · HRM (arXiv 2506.21734) ·
Metacontrol (arXiv 1705.02670) · Thinker (arXiv 2307.14993) · CoALA (arXiv 2309.02427) ·
Searchformer (arXiv 2402.14083) · Stream of Search · AGM belief revision · Thagard,
*Explanatory Coherence* · Grossberg, ART · chess-transformer OOD (arXiv 2510.20783).
