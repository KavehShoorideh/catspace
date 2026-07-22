# INQUIRY: Tactics — definition, detection, and the map-native signature (opened 2026-07-23)

Kaveh's framing: *"a tactic is effectively an opportunity outside of our plan afforded by a
mistake by our opponent."* This document turns that into definitions, testable signatures, and
an experiment ladder. Status: inquiry open; E1 runnable with existing code.

---

## 1. How the field defines a tactic

**Chess theory (classical):** a forced, concrete sequence (checks, captures, threats) exploiting
short-term features — pin, fork, skewer, discovered attack, deflection, overload, back-rank —
that wins material or mates. Opposed to *strategy* (long-term, non-forcing). Two essentials:
**forcingness** (opponent's replies are constrained) and **discontinuity** (the evaluation jumps
once the line resolves; "compensation" = the jump arrives a few plies late).

**Engine practice (operational definitions, convergent):**
- **Quiescence**: a position is tactical exactly when *static evaluation is unreliable* until
  pending forcing moves resolve. The oldest operational definition — engines literally cannot
  evaluate a non-quiet position without searching it.
- **Depth instability**: eval(depth d) vs eval(depth d+k) swings — tactical sharpness as
  disagreement between shallow and deep readings of the same position.
- **Only-move narrowness**: gap between best and second-best move. Lichess's puzzle generator
  (300M analyzed games, Stockfish at 40 meganodes) requires every solution move to be an
  "only move" — and puzzles are mined at positions typically *created by a mistake*. That
  pipeline IS Kaveh's definition industrialized: opportunity + created-by-error + forced
  conversion. The resulting labeled dataset (millions of puzzles, tagged fork/pin/sacrifice/...)
  is public (HuggingFace `Lichess/chess-puzzles`) — import, don't re-mine.
- **WDL sharpness (Leela)**: an "equal" position that is 100% draw is calm; an "equal" position
  that is 50% win / 50% loss is sharp. Sharpness = win+loss probability mass hiding under a
  flat scalar eval — scalar evaluation *obscures* sharpness by construction.
- **Human-modeling (Maia line)**: sharp positions are where human move-match accuracy collapses
  and blunder probability spikes — sharpness as *human fragility*, which is also chess.com/
  lichess "accuracy" mechanics (win-chance loss per move).

**Synthesis:** every operational definition is a **disagreement between a smooth reading and a
forced reading** of the same position — static vs deep, scalar vs WDL, plan vs concrete line.
That abstraction is what ports to our map.

## 2. The map-native formalization (what WE add)

Our vocabulary gives the definition a form none of the above have, because we model
*reachability two ways*:

- **Tactic = a veto lapse, cashed by force.** Measured 2026-07-23: from won positions, 87% of
  exact winning targets are adversarially DENIED (opponent steers away); regions become 99%
  forceable only at neighborhood granularity. A **mistake flips specific regions from denied to
  forceable** — that flip IS the "opportunity suddenly making itself available." A tactic is
  the forcing line that cashes it before it closes. "Outside of our plan" = the newly forceable
  region was not the current subgoal — which is exactly why the engine's α-mixture prior keeps
  a (1−α) global tail: tactics are definitionally what focused search misses (DECISIONS §8).
- **Sharpness = smooth-vs-forced disagreement, in our pieces:** |field/value-head reading −
  short forced-search reading| at the same position. This is quiescence re-expressed for a
  learned map, and it is computable with existing components (field d, DTM head, mate_stop
  MCTS). The competence head and compute_layer's `should_search` flag were built for exactly
  this seam.
- **Predicted manifestation in B:** around a genuinely tactical region (e.g. post-sacrifice),
  *winning approaches are narrow and funneled* (forced lines) while general approaches are
  broad; for a pseudo-tactic the winning-approach set is ~empty. Sharp regions should look like
  **bottlenecks** in approach-space, not clouds.

## 3. The three-set contrast (Kaveh's Bxh3 design)

Anchor: a candidate tactical strike s* (say Bxh3). Take B(s*)'s neighborhood (the index/goal-bank
machinery). Build three predecessor sets:

| set | who reaches the region | machinery |
|---|---|---|
| **G** — general | cooperative random-walk predecessors | measure_adversarial_veto walks |
| **H** — human | positions from real games that entered the region | shard scan + index query |
| **W** — winning | entries that then mate / convert (tb-exact in toy; game outcome + puzzle labels in human data) | continuation harvests |

Contrastive readouts:
- **Opportunity ratio** = |W∩region| / |G∩region| — is this region one that winning play flows
  through at all? (sound vs pseudo sacrifice at the region level)
- **Approach concentration** — directional coherence of W's approach vectors vs G's (forced
  funnel vs diffuse cloud; the mate-directions machinery, master-direction-removed)
- **Conversion gap** = H∩W vs H\W among entries — did humans who *got there* cash it? (the
  converted / missed axis)
- **Temporal jump** — all of the above computed per-ply along a game: a tactic event is the
  ply where the opportunity ratio / forceability of the region JUMPS (the veto lapse), which
  is Kaveh's "suddenly made itself available," measurable.

Taxonomy for one shared surface move (same Bxh3):
1. **Sound + converted** — region forceable, W-dense, follow-up played (puzzle-DB positive).
2. **Sound + missed** — region forceable, W-dense, human left the line; eval decays back.
3. **Unsound (pseudo-tactic / blunder)** — W-thin, refutation exists; forceability never flipped.

Ground truth: toy = tablebase-exact (free, unlimited); human-scale = lichess puzzle DB labels
(externally computed; note audit stance — engine-derived labels are for *evaluation and probes*,
not field-training signal, unless deliberately reversed).

## 4. Experiment ladder (cheap → deep; instruments exist)

- **E1 — exact tactic events on the toy (runnable now).** Along tb/SF/human toy trajectories,
  compute per-ply which good regions are forceable (forceable() DFS + rollout_dtm). Label veto
  lapses exactly: opponent move → region flips denied→forceable. Output: base rates, the exact
  tactic-event dataset, and the first measured "opportunity appears" curves.
- **E2 — do our maps see sharpness? (probe).** Import the lichess puzzle DB; probe every field
  (cooperative, human d=64, human d=512-in-training) for puzzle-vs-matched-quiet separation
  with the in-stratum controls (TRAINING_STANDARDS #11). Prediction, from the pattern results:
  current fields are blind; the sharpened human field is the interesting read.
- **E3 — the Bxh3 three-set contrast.** Sacrifice-themed puzzles (sound) vs mined same-move
  unsound sacs; compute opportunity ratio / approach concentration / conversion gap. The
  deliverable is the separation curve between sound and pseudo.
- **E4 — sharpness = field-vs-search disagreement.** Correlate |smooth reading − forced
  reading| against E1/E2 labels; if it holds, this is the engine's tactic ALARM (gates the
  α-dial down / triggers search-more) — no hand-coded tactic features, honoring the
  diagnostics-only rule: the alarm is a *disagreement between learned components*.
- **E5 — close the old loop.** The original UI request ("when MCTS finds tactics, track them and
  ask what has to happen for this tactic to become workable"): with the veto-lapse formalism,
  "what has to happen" = which precondition makes region G forceable — the precondition-vector
  idea from the hierarchical-planning design, now with an exact toy laboratory.

## 5. What this plugs into

Sharpness alarm → engine (α-dial + search budget); veto-lapse events → blunder-mined training
data (the sparse precious set from the adversarial-reachability discussion); puzzle DB → the
concept/pattern contrast recipe extended from mates to tactics; E1's exact events → the
contrast-tuple generator's third branch (denied vs forceable futures).

---

## 6. The latent-tactic portfolio (Kaveh 2026-07-23, second pass — the core formalization)

A tactic is not a per-position property; it is a PERSISTENT OBJECT with a lifecycle:

    LatentTactic:
      basin          B-region of POST-EXECUTION advantage positions (dust settled = quiescent,
                     advantage banked -- material OR structural: bishop pair, center, tempo).
                     Discovered as a cluster; defined by OUTCOME, not by the move.
      strike         the surface move/line that enters it (Bxh3).
      preconditions  the refutation at sensing time, inverted -- DISJUNCTIVE routes to
                     activation (defender overloaded / moves away / gets kicked / outnumbered:
                     4-7 distinct ways). = the precondition-vectors-from-refutations idea of the
                     hierarchical-planning design, instantiated.
      monitor        cheap per-ply check: did any route complete / did their move drop the veto?

    Lifecycle: SENSE (expensive search, once, when first noticed) -> REGISTER (extract
    preconditions from the refutation) -> MONITOR (O(k) proxies per ply over the watchlist,
    NO deep re-search) -> ALARM (a route completed = veto lapse) -> CASH (search once, pounce).

**Geometry (adopt viability-theory naming for the robotics port): the basin is a CAPTURE
BASIN** -- states from which the target is reachable under our control against their
interference. WATERSHED = the entry-point set (predecessors from which the basin is forceable;
exact on toy via forceable()). RIDGE = optimal defense = walking so no watched watershed is
entered; a blunder = stepping off the ridge; "seeing them go down into the basin" = watershed
membership test against the watchlist (cheap). Sound-but-MISSED tactics are WATCHLIST failures,
not search failures -- monitoring, not search, is the human-efficiency mechanism.

**Field-native WDL (the success predicate):** two-sided comparison, existing pieces (zgoals
MATE_W/MATE_B/MATE_DIFF; two_field.py two-perspective runtime; committor line):

    dA(s) = d(F(s), Bank_their_mate) - d(F(s), Bank_my_mate)
    success  <=>  dA(after resolution) - dA(before strike) >= margin      (material may be WORSE)

**E1 extensions (all tb-exact on toy):**
  E1a  do basins exist as B-clusters? harvest successful resolutions, cluster, in-stratum controls.
  E1b  watershed extraction + ENTRY-ROUTE MULTIPLICITY per basin (the "4-7 ways", counted).
  E1c  watchlist economics: per-ply proxy-monitor cost vs re-search cost -- the number that
       justifies the spidey-sense architecture on the strength-per-node frontier.

## 7. Watershed shells & the geodesic corridor (Kaveh 2026-07-23, third pass — the E1b mechanism)

Given current position s and a basin bank B_G, over the INDEX pool compute a(x)=d(F(s),B(x))
(my reach to x) and b(x)=d(F(x),B_G) (x's distance into the basin; min over bank):

  SHELLS     bucket by b(x): shell_k = k plies from entry ("1 ply out, 2 plies out, ...").
  CORRIDOR   triangle inequality: a(x)+b(x) >= d(F(s),B_G), equality ~ on geodesics ->
             rank x by SLACK a(x)+b(x)-d(s->G); low-slack, low-a = the promising entries.
             Third factor = density (S = forceability x reachability x density).
  MONITOR    per ply, ONE distance call per watched basin -> current shell; the spidey-sense
             alarm = a shell-CROSSING event (3->1). O(k) per move, no re-search (E1c).
  REUSE      planner_landmark.py Dijkstra/K-shortest-paths with basin bank substituted for MATE.

Caveats (all from measurements): (1) corridor is COOPERATIVE/optimistic -> forceability filters,
at REGION granularity (87%/99% veto result); (2) shell_k vs shell_{k+1} needs +-1-ply field
resolution -- historically at chance; promote SHELL RESOLUTION to an acceptance instrument for
the contrast-trained field; (3) the open asym inversion leaks backward shells through one-way
doors (capture-dense regions!) -- land the repulsion-exemption fix before trusting watersheds
near captures.
