# planner approaches

Registry for the **planner** component: `propose_subgoals(state) -> ranked goals`.

Every directory under `approaches/` must have an entry here, and every entry must have a
directory — `scripts/check_approaches.py` enforces both directions. Schema is defined in
`repo_structure.md` § "approaches.md schema".

`status` is one of `active`, `parked`, `superseded-by:<name>`.

---

## subgoal_cascade

- **folder** — `approaches/subgoal_cascade/`
- **status** — active
- **hypothesis** — The planner owns *where* to go and the navigator owns *how*; a hand-coded
  decide loop organized around the energy objective `E_mu[score] - c * compute` — probe,
  decompose, select, commit — plans better per unit of search than raw search does.
- **definition of done** — At matched node budgets the cascade beats plain search on the
  probe suite, and every decision it makes is attributable to a named stage.
- **notes** — The largest approach here: `probe.py` (bounded MCTS as an *evaluation*),
  `decompose.py` (meet-in-the-middle geodesic midpoints), `plans.py` (plans as persisted
  first-class objects that remember why they were blocked), `optionality.py` (multipurpose
  moves), `trap_trace.py` (recognize -> verify -> commit, trace as the product),
  `readout.py`/`policy.py`/`selector.py` (the MEAN-vs-MIN and plan-choice seams).
- **results** — JOURNAL.md (traced-engine MVP: 57% trap-moves, 34% confirmed / 66% honestly
  refuted on the smoke harness)
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## endgame_groundtruth

- **folder** — `approaches/endgame_groundtruth/`
- **status** — active
- **hypothesis** — Tablebase truth (WDL/DTZ), material signatures and a DTM CNN give an exact
  reference the learned stack can be measured against — and a *logged* fallback at play time,
  never a silent crutch.
- **definition of done** — Every fallback consultation is logged and attributable; the DTZ
  conversion move makes strict progress rather than shuffling (the g017 autopsy trap).
- **notes** — `tb.py` is the canonical tablebase utility module (~10 scripts previously
  cross-imported private copies). Certification is offline/referee-only.
- **results** — JOURNAL.md (g017 autopsy; DTZ != DTM)
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## opponent_model

- **folder** — `approaches/opponent_model/`
- **status** — active
- **hypothesis** — Who you are playing changes which plan is right. A frozen Maia-2 rating base
  plus a per-player style residual `z` — recoverable convexly from their own moves, and
  updatable online — makes `P(move | position, opponent)` a first-class planner input.
- **definition of done** — The online estimator's `(Elo, z)` belief converges from a real
  opponent's moves, and conditioning on it beats the population-average prior on held-out
  move prediction.
- **notes** — Merges the former `predictor/opponent` and `style/` trees: `style_model.py`
  (M2b residual `z`), `style_recover.py` (convex recovery), `style_estimator.py` (M2c online
  belief), `style_live.py` (play-time wiring), `maia2_policy.py` (poison-guarded priors).
- **results** — JOURNAL.md M2b/M2c
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## atlas_region_stats

- **folder** — `approaches/atlas_region_stats/`
- **status** — active
- **hypothesis** — Region-level statistics from the M3 composite table — shrunk committor
  quality, chute fall rates, badness, crossing risk — are enough to rank subgoal regions:
  `score(region) = P_reach x net_flux x quality`.
- **definition of done** — The ranking correlates with realized outcome on held-out games
  better than reach alone.
- **results** — artifacts/region-discovery-feasibility-standalone.md (see
  `research/docs/archive/`)
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## reach_field

- **folder** — `approaches/reach_field/`
- **status** — active
- **hypothesis** — A `z`-conditioned first-hit reachability head gives calibrated region-level
  first-hit probabilities and plies at a fixed play-time context.
- **definition of done** — Predicted first-hit plies are calibrated against realized first-hit
  on held-out rollouts.
- **results** — —
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## committor_value

- **folder** — `approaches/committor_value/`
- **status** — active
- **hypothesis** — The committor `c(s) = P(win)` used greedily (1-ply/2-ply expectimax) is a
  sensible value oracle and the honest baseline every planner must beat.
- **definition of done** — Remains reproducible as the reference baseline (the 0.125 number)
  that planner approaches are scored against.
- **results** — JOURNAL.md (committor-greedy vs maia)
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## armed_tactics

- **folder** — `approaches/armed_tactics/`
- **status** — active
- **hypothesis** — Moves whose immediate committor gain looks good but *decays* over the next
  few plies are detectable, and they are exactly the armed tactics a planner should either
  avoid or aim the opponent into.
- **definition of done** — Detection returns well-formed candidate records on real
  Stockfish-vs-Stockfish mid-game positions, with the decay signature separating armed
  candidates from ordinary good moves.
- **results** — MILESTONES.md §M7; M7 detection demo on real SF-vs-SF mid-game positions
- **added** — 2026-08-03 · **owner** — Kaveh Shoorideh

## two_perspective_scoring

- **folder** — `approaches/two_perspective_scoring/`
- **status** — active
- **hypothesis** — Two perspectives over one slow embedding — my evidence and theirs, each a
  `MemoryField` — score candidate moves better than a single-perspective field.
- **definition of done** — Two-perspective scoring beats single-perspective at equal budget,
  and is selected by at least one end-to-end config.
- **notes** — Split from the old top-level `two_field.py` in the 2026-08-03 restructure: the
  scoring half (`score_components()`, `TwoFieldPolicy`) stayed here; the fast-field re-pricing
  half moved to `search/puct_mcts/src/repricing.py`. Not yet wired into an end-to-end config.
- **results** — —
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh
