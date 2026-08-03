# end-to-end approaches

Registry for `catspace/approaches/` — the named wirings of the four research components.
An end-to-end approach picks one planner, one-or-more searches, one-or-more encoders and
one-or-more memories, and owns the glue. This is the layer `catspace/deployment/` consumes;
deployment never reaches into a research component directly.

Every directory under `approaches/` must have an entry here, and every entry must have a
directory — `scripts/check_approaches.py` enforces both directions. Schema is defined in
`repo_structure.md` § "approaches.md schema".

Two kinds live here:

- **config** — has `config.py` exposing `CONFIG` (an `EndToEndConfig`) and `build()`;
  resolvable by name via `catspace.approaches.load(name)`.
- **harness** — runs configs against something (an opponent, a tournament) rather than being
  one. No `config.py`.

Component slots are named by string (`"search:puct_mcts"`), never by deep import path; only
`catspace/registry.py` imports approach modules. `CONFIG.validate()` fails loudly if a slot
names an approach that does not exist.

`status` is one of `active`, `parked`, `superseded-by:<name>`.

---

## field_mcts_default

- **folder** — `approaches/field_mcts_default/` · **kind** — config
- **status** — active
- **wiring** — `planner:subgoal_cascade` · `search:puct_mcts` · `encoder:reachability_field` ·
  `memory:vector_store_retrieval`
- **hypothesis** — The incumbent wiring: a trained reachability field supplies the leaf value,
  subgoal regions shape the move prior *only* (never the value), PUCT runs the plan phase, and
  the finisher runs pure because the field value degrades play near mate.
- **definition of done** — Reproduces pre-restructure `LayeredEngine` play exactly, so it can
  serve as the control every other config is compared against.
- **notes** — `field_ckpt=None` gives a pure-search engine, the honest default when no trained
  checkpoint is supplied.
- **results** — JOURNAL.md (pre-restructure LayeredEngine baselines)
- **added** — 2026-08-03 · **owner** — Kaveh Shoorideh

## bootstrap_mate

- **folder** — `approaches/bootstrap_mate/` · **kind** — config
- **status** — active
- **wiring** — `planner:endgame_groundtruth` · `search:puct_mcts` ·
  `encoder:reachability_field` · `memory:experience_store`
- **hypothesis** — No external mate bank at all. The engine starts knowing nothing; MCTS
  (energy prior + `mate_stop`) probes the reachability field, every checkmate leaf the search
  touches is harvested into an online bank of its *own* experience, and the value becomes
  distance-to-discovered-mates. One knob: search budget.
- **definition of done** — Identify the search budget at which KRRvK-central hits 100%.
- **notes** — `src/` is shipping code — both deployment shells build their engine from
  `build_engine()`. The budget sweep lives in `experiments/bootstrap_mate_engine.py` and is
  not packaged. Tablebase truth is a logged fallback, never consulted at play.
- **results** — JOURNAL.md 2026-07-24 (bootstrap pivot)
- **added** — 2026-07-24 · **owner** — Kaveh Shoorideh

## gauntlet_harness

- **folder** — `approaches/gauntlet_harness/` · **kind** — harness
- **status** — active
- **hypothesis** — Strength claims must come from the frameworks where chess bots actually
  compete (fastchess/cutechess SPRT at fixed TC), not from bespoke scoring — and every game
  loop must emit the same VERDICT lines so logs stay comparable across refactors.
- **definition of done** — A config-vs-config SPRT run completes and reports a verdict; the
  VERDICT instrumentation is byte-comparable with the pre-modular `m5_mcts_probe` output.
- **notes** — `src/play.py` is the reusable loop; `experiments/` holds the drivers
  (`arena_real.py`, `play_vs_maia.py`, `play_traced.py`, `gauntlet.sh`). `gauntlet.sh` needs a
  third-party fastchess binary — set `FASTCHESS`, it is not vendored.
- **results** — JOURNAL.md (traced-engine MVP smoke; arena/maia runs)
- **added** — 2026-07-30 · **owner** — Kaveh Shoorideh
