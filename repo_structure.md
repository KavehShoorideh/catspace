# repo_structure.md — where things go, and why

This is the instruction doc for the layout adopted on 2026-08-03. If you are adding code,
data, an experiment or a doc, the answer to "where does this go" is here. `AGENTS.md`
points at this file and states the rules that are actually enforceable.

Anything dated before 2026-08-03 (JOURNAL.md, MILESTONES.md, `research/docs/archive/`,
`research/docs/writing/`, `research/weekly_report/`) describes the OLD layout and was
deliberately not rewritten — see [Historical documents](#historical-documents).

---

## The one-paragraph version

`catspace/` is the wrapper package. It owns the engine backbone and the plugin Protocols,
and it wires together four research components — **encoder, planner, search, memory** —
each of which holds several competing **approaches**. A named **end-to-end approach**
(`catspace/approaches/<name>/`) says which approach fills each slot. `catspace/deployment/`
consumes an end-to-end approach and never reaches into a component itself. Everything is
`import catspace.*`; there are no other top-level packages.

---

## The tree

```
/                                repo root
  repo_structure.md              this file
  AGENTS.md                      the short, enforceable version
  pyproject.toml                 packaging, entry points, pytest config
  config.json                    fastchess tournament config
  scripts/                       symlinks to the entry points a newcomer needs
  tests/                         cross-component and integration tests
  contrib/                       third-party code, unchanged
  data/                          generated datasets, DVC-tracked
  assets/                        THIRD-PARTY only: tablebases, engines, corpora, weights
  artifacts/                     experiment records + curated results
  mlruns/, mlflow.db             the one tracking store
  README.md JOURNAL.md MILESTONES.md human-written-alt-arch.md    (historical, banner-noted)

  catspace/                      THE WRAPPER PACKAGE
    interfaces.py                the Protocols an implementation must satisfy
    engine_core.py               LayeredEngine, the composer
    registry.py                  resolve "component:approach" -> module
    orchestrator.py introspection.py watchlist.py priors.py fields.py values.py
    io/paths.py                  THE PATH REGISTRY (see Data)
    incubator/                   explicit holding pen: work with no home yet
      analyst/discovery/         the analyst's discovery code; the wake/sleep loop (plan 78.14-78.17):
        wake.py sleep.py         self-play logging (play_self(..., starts=None): game g starts from starts[g % n], a position string the kernel accepts, None = the game's own; `game` rows carry `start`; read_starts(path); CLI --starts); sleep (surprises, cards, lemmas; rank sources, one shape `rank(variant, moves, cap=None, start=None)` (start passed only when not None): parse_multipv (+ parse_multipv_lines: the whole principal line per rank), disagreement, break_ties, plan_tools (one source or a list: the human source first when coarse), source_of (mcts | human[:path] | gnn[:path]), UciTeacher (gap = 100, start-aware; `grade(score)` (mate exact, else score // gap); `lines(variant, moves, cap=None, start=None) -> {move: principal line}` from the same cached analysis as `rank`; CLI --teacher-ms, repeatable --teacher-option NAME=VALUE); Settings.regret; plan 78.23: Settings.judge_by ("store" | "source") + judge_positions, CLI --judge-by / --judge-positions; read_log keeps `start`); plan 78.24: Settings.vocabulary (a NAMED lemma vocabulary: spec/vocabulary, known lemmas, meta.vocabulary and publish all use it), UciTeacher.mate_bucket (plies, default 1: a mate's distance floored to a multiple), CLI --vocabulary NAME, repeatable --gate NAME=VALUE (dev and held-out gates) / --held-gate NAME=VALUE (held-out only), --teacher-mate-bucket N; plan 78.25: play_self(..., reason=False) and CLI --reason pass the reasoning chain through to solve_batch (as --ties does); tests/test_plan78_22.py, tests/test_plan78_24.py, tests/test_plan78_25_reason.py
        detectors.py             plan 78.22: surprise detectors as modules (Context, Hindsight (no oracle), Disagreement(source), Regret(source) (the move played against the source's ranking, by the source's own `gap`, else top-k order), confirm (shared deeper-card confirmation: confirmed / tied (OPEN) / unconfirmed / cut), run, detector_of(name, source=None), REGISTRY); sleep calls them; Context.continuations (plan 78.23); `ask` takes cap; curriculum ChessCodec.kinds (optional codec method); tests/test_plan78_22.py
        judges.py                plan 78.23: a rank source as the JUDGE of the cards (optional source methods `grade(score) -> int`, `lines(...)`; default_grade, grader, describe, revalue(cards, positions, source, variant, indices=None, *, cap=None, say=None, context=None) -> (CardSet via CardSet.revalued, report incl. the optimal share), regret_seeds (regret confirmed directly on the revalued card; rep["regret"]["confirmed_rows"] and meta.why = the source's lines after the played / best move), consequence(variant, start, moves, line, codec=None) (replay a line: plies, outcome, units, first change), say_consequence; describe also reports mate_bucket; plan 78.24: Cached(source, path) = any rank source remembered in a JSON file keyed (variant, start, moves, cap), so the sleep, the audit and a re-run read one verdict per position; the wrapped source's own attributes show through); tests/test_plan78_23.py, tests/test_plan78_24_loop.py
        curriculum.py            plan 78.22, generic: the material curriculum (PositionCodec protocol, ChessCodec (the one place chess appears; optional `kinds`), CODECS by game name, required(game) found by test, starts(game, size, limit=, seed=), survey, CLI -> a --starts file under paths.derived); tests/test_plan78_22.py
        apply_audit.py          plan 78.24 B: the audit of application (generic): render(lemma) / mined(game) (a lemma read as IF body THEN best move removes/adds, with stats, judge, meta.why); audit(game, log, *, snapshot=None, variant=None, source=None, vocabulary=None, limit=None, say=print) -> per decision where a lemma fired: the moves it proposed (recovered with the ENGINE's matcher: CardSet.revalued one-hot per move + CardSet.eval, since the log row holds only the `fired` fact and the count), played, followed, optimal by the rank source; per-lemma and overall summaries as OBS lines; CLI --log --snapshot --teacher --out; tests/test_plan78_24_loop.py
        initialize.py           plan 78.24 C: the loop for one size on a SANDBOX store (GS.LEARNED_ROOT / SNAPSHOT_ROOT swapped in and restored; vocabulary copied from the real rules/learned, never written back): run_size(game, size, *, source, store_root=None, judge=None, games_per_start=4, ..., sleep_settings=None, vocabulary=None, tag=None) = starts -> wake a -> sleep (source is the judge) -> wake b -> audit -> before/after compare (a mechanism check); settings_pairs / settings_from (Settings fields typed from the dataclass); CLI --size --teacher --store-root --set NAME=VALUE --vocabulary --gate NAME=VALUE --held-gate NAME=VALUE --teacher-mate-bucket N --teacher-cache FILE; tests/test_plan78_24_loop.py
        diagnose.py              game-free diagnostics over a wake log (ties, surprises by kind, cost, depth, gap share); tests/test_plan78_17.py
        rank_bench.py            plan 78.18: a rank source's agreement with a reference on logged positions, and a match of two rank sources (positions carry their start); tests/test_plan78_18.py
        human_index.py           plan 78.19 stage E, TRAINING ONLY: the human-games index; HumanSource.rank/detail take start= (answered by the kernel key, [] for a source keyed otherwise) (sqlite: position key -> move n, mover's points; build, lichess_records_iter, kernel_key, signed, points_of, default_path, main CLI)
        fact_graph.py          plan 78.20, generic: the typed fact graph (graph_of(facts, moves, vocabulary, bindings) -> HeteroData: term / fact / move / claim nodes, arg<i> / removes / adds / binds edges + reverses; Vocab (UNK = 0), encode, edge_types(max_arity, max_parts)) and graph_hash (Weisfeiler-Lehman abstraction key of 78.19 reading ii); the claim type is built but nothing supplies bindings yet. Plan 78.21: from_engine(CE.graph dict, values, lost) -> HeteroData, the graphs the net trains on are BUILT IN RUST (decision/src/graph.rs); tests/test_plan78_21.py
        graph_prior.py         plan 78.20 / 78.21: GraphNet (torch; HeteroConv SAGE over arg<i>/part<i>/removes/adds/binds edges, scores the move nodes), train(cards, out, log=...) (graphs built by CE.graph from the card's position, cross-entropy against the optimal set, dev/held by group, OBS gnn lines incl. the torch-vs-Rust parity line), export(checkpoint, weights.json) (the Rust weights file), GraphSource(weights_path, specs=, vocabulary=, game=) (rank source `gnn`, inference kit, ALL IN RUST through CE.GraphModel, no torch needed; .ms, .timing()), build-cards / train CLI; tests/test_plan78_20.py, tests/test_plan78_21.py
                                 and HumanSource (rank/detail/__call__/close; source "human"); read by sleep's training kit only; tests/test_plan78_19e.py
    approaches.md                registry of end-to-end approaches
    approaches/<name>/           config.py + src/ + experiments/
    research/
      components/<component>/    encoder | planner | search | memory
        approaches.md            registry of this component's approaches
        approaches/<name>/
          src/                   SHIPPING code
          experiments/           lab scripts (not packaged, no __init__.py)
          tests/                 unit tests for this approach
          logs/ artifacts/ data/ non-shipping output
      tools/                     by TYPE: probes figures embeddings ablations viz
                                 stats_eval training_infra chess_specific
      docs/                      research docs + archive/ + writing/
      infra/                     preemption, checkpointing, observability, cloud
      weekly_report/             phase reports
      assets/ logs/              (assets live at the repo root; see Deviations)
    deployment/
      server/                    assistant_server, uci_engine, banksync
      web/                       board UI
      docker/                    Dockerfile, compose, nginx, prometheus, grafana
```

---

## Imports

**No new top-level packages. Everything is `catspace.*`.** `experiments/`, `tools/` and
`infra/` are gone as top-level names; the restructure deleted ~294 `sys.path.insert` lines
that existed only to make bare imports of them work.

- An approach's `src/` is a regular package and gets an `__init__.py` that re-exports the
  approach's public names, so callers can write
  `from ...approaches.reachability_field.src import ReachabilityField`.
- `experiments/`, `logs/`, `artifacts/` and `data/` **must not** have an `__init__.py`.
  That is what keeps them out of `setuptools.packages.find()` and out of the wheel.
- They are still importable — a directory without `__init__.py` inside a regular package is
  a PEP 420 namespace portion — so research scripts can import each other the way a lab
  notebook does. This is deliberate, not an accident of packaging.
- End-to-end configs name approaches **by string** (`"search:puct_mcts"`), never by deep
  import path. Only `catspace/registry.py` does deep imports.
- Do not re-export a function whose name collides with its own module. `subgoal_cascade`
  does not re-export `decompose`, because `from ...src import decompose` must keep meaning
  the module.

---

## approaches.md — the schema

Every component directory and `catspace/approaches/` has an `approaches.md`. Every folder
must have an entry and every entry must have a folder;
`scripts/check_approaches.py` enforces both directions and runs in P13 / CI.

Each entry is a level-2 heading (`## <folder_name>`) followed by:

| field | meaning |
|---|---|
| **folder** | path, relative to the registry |
| **status** | `active` \| `parked` \| `superseded-by:<name>` |
| **hypothesis** | one line: what this approach claims |
| **definition of done** | the measurement that would settle it |
| **results** | link to where the numbers are, or `—` if it has not reported |
| **added** | date |
| **owner** | owner of record |

Optional: **wiring** (end-to-end approaches only), **notes**, **why parked**.

Write `—` for results when an approach has not reported. Do not invent a link: an entry
that overclaims is worse than one that admits it has no numbers yet.

---

## The component contract

`catspace/interfaces.py` holds the Protocols. Anything satisfying one can be injected into
`LayeredEngine`:

```python
class ValueModel(Protocol):        # leaf evaluation, white-POV, in [-1, 1]
    def values(self, boards: list) -> np.ndarray: ...

class MovePrior(Protocol):         # board -> {move: prob}
    def priors(self, board: chess.Board) -> dict: ...

class SubgoalSelector(Protocol):   # propose the current Region, or None
    def select(self, board: chess.Board) -> Region | None: ...
```

plus the `Region` (goal-as-region) and `SearchOutcome` dataclasses.

The rule these encode, and it is the important one: **subgoals enter through the prior,
never through the value.** The value is the global objective. See DECISIONS §8.

> **Known gap.** `catspace/registry.py` also defines a uniform constructor —
> `registry.build("search:puct_mcts", **kwargs)` expects the approach's `src` package to
> expose `build(**kwargs)`. **No approach implements it yet** (0 of 22). `registry.load()`
> works and is what everything currently uses. New approaches should provide `build()`;
> retrofitting the existing 22 needs a real decision per approach about what its
> constructor arguments are, and was not guessed at during the restructure.

---

## Production vs research

`catspace/research/components/<c>/approaches/<name>/src/` is **SHIPPING CODE** despite
living under `research/`. It goes in the wheel and runs in the container.

Non-shipping, under any approach: `experiments/`, `logs/`, `artifacts/`, `data/`.

The practical consequence, and the one that actually bit during the restructure: **shipping
code must never import from an `experiments/` directory.** `bootstrap_mate/src/engine.py`
was lazily importing three model classes (`DTMTok`, `DTMNet`, `PlanNet`) out of training
scripts that the Docker image does not copy. Model architectures the engine loads at play
time belong in `src/`; the trainer imports them from there, which also stops the training-
time and play-time featurization from drifting apart.

---

## Tests

- **Cross-component and integration** → root `tests/`.
- **Unit tests for one approach** → `approaches/<name>/tests/`.

`pytest` `testpaths` covers both. `*/experiments` is excluded from collection: lab scripts
with `test_`-shaped names (`test_field_fullgame.py` — "test the field on full games") are
not tests and must not be imported at collection time.

---

## Data

**Generated data** lives under `data/`, DVC-tracked, and is **always** reached through the
path registry — never a literal relative path.

```python
from catspace.io import paths

paths.derived("dtm_endgame.npz")     # -> /abs/.../data/derived/dtm_endgame.npz
paths.sep("lichess_mc2.pt")
paths.experiment("krrkbp_test_n200.json")
paths.figure("reach_curve.png")
paths.syzygy_dir(), paths.engine("maia/maia-1900.pb.gz")
```

Why this is a rule and not a preference: `"data/derived/foo.npz"` only resolves when the
process happens to have been launched from the repo root. That is why cross-module artifact
handoff kept breaking, and why the container silently pointed at nothing. The restructure
replaced 901 such literals across 294 files.

- The **directory** is declared once, in `catspace/io/paths.py`. The **filename** stays at
  the call site, where it is meaningful.
- File accessors return `str`, so they drop into every place a literal used to sit.
- `REPO_ROOT` is found by walking up for `pyproject.toml`/`.git`, not by counting
  `parents[N]`, so moving a module deeper cannot break it.
- **Third-party** downloads (tablebases, engine binaries, corpora, Maia weights) are not
  data: they go under `assets/`, never `data/`.

---

## Historical documents

**Never rewrite a dated or archived document to match the new layout.** JOURNAL.md,
MILESTONES.md, README.md, `research/docs/archive/*`, `research/docs/writing/*` and
`research/weekly_report/*` are the research record; paths in them were correct when
written. They carry a banner pointing here instead. The migration tooling skips `.md`
entirely for this reason.

This does not apply to Python docstrings that name their own file path — those are code
comments and were updated.

---

## Deviations from the refactor plan

Recorded so the difference is a decision rather than a discrepancy:

1. **`assets/` and `data/` stayed at the repo root** rather than moving under
   `research/`. Deployment needs `assets/engines` (compose mounts it), so they are not
   research-only; the path registry, written in P1, already resolved them at the root.
2. **The path registry is `catspace/io/paths.py`**, not `research/tools/io/paths.py`. It
   is imported by shipping code and by deployment, so it belongs at wrapper level.
3. **Directory names use underscores**, not hyphens (`gauntlet_harness`,
   `weekly_report`) — they have to be importable.
4. **`uci-engine` and `http-assistant` did not become end-to-end approach folders.** Their
   shared glue did: it is `catspace/approaches/bootstrap_mate/`, which both deployment
   shells build from. Splitting the 949-line `assistant_server.py` into analysis logic and
   HTTP shell was judged a separate change, not part of a move.
5. **Generated data was not physically scattered** into per-approach `data/` folders. The
   registry is the indirection that makes location a one-line change; relocating ~100
   DVC-tracked outputs is a separate, riskier operation and the plan's access rule (always
   via the registry) is satisfied without it.

## Rust engine (rust/engine, a Cargo workspace; game-free layer crates)

Plan 78's end-to-end system (docs/ANALYST_PLAN.md 78). One crate per layer; every layer crate reads only the `Game` trait and names no chess object
(tests/test_plan78.py's generic test scans their sources). The build is offline: `cargo test --release --offline --workspace --exclude cs_engine`;
the Python extension `cs_engine` is built from `py/` with `maturin develop --release --offline` (the venv's python).

```
rust/engine/
  gdl/                           cs_gdl: the GDL reader and spec evaluator (compose a variant's delta onto a base description); plan 78.19 A: `Program::run_for(wanted)`
                                 (only the dependency cone's strata; `cone`, `cone_rules`, `run_in` for the audit's wrong cone; `run` stays the oracle), statics shared by
                                 `Rc` with prebuilt indexes (`Db` layered over them), indexes kept current on `add`; tests/demand.rs (run == run_for on the cone, negative control)
  game/                          cs_game: the Game trait (roles, mover, moves, apply, outcome, key, base_facts), GdlGame (any description),
    src/kernel.rs                  KernelGame over shakmaty's variants -- THE ONE CHESS MODULE (rules ingestion); src/outside.rs the outside engines
    src/certify.rs                 co-play certification of a kernel against its description (states, moves, facts); src/rng.rs the seeded RNG
    src/gdl_game.rs                FactView: the vocabulary's derived relations evaluated over base facts; `with_demand(bool)` (off = full `run`, the oracle), `rows` (derived
                                   rows interned, no strings), `cone_size`, string<->term memos (plan 78.19 A)
    src/fact_audit.rs              plan 78.19 A: `positions` (seeded random play of any Game) and `audit` (demand view vs full view: equal/unequal, ms each); py `facts_audit`
    tests/demand.rs                demand == full facts over random play, the cone's size, a negative control (a wrong cone is caught), rows vs facts, statics' copy cost
  operator/src/lib.rs            cs_operator: THE SHARED INTERFACE -- Goal (+ its Mode; plan 78.25 adds Goal::Reason { role }), Budget, Outcome (status, move, value, candidates, evidence, line, events,
                                 measured cost in parts, facts), Operator, Controller, drive; Kit (Inference | Training) is a property of each Operator and
                                 drive = drive_in(Kit::Inference, ...): an inference drive refuses a training operator in the type (plan 78.17)
  tactics/                       cs_tactics: forcing (proof), the ledger search (events along the line, a Judge turns them into a value: Count or Worth), census;
                                 src/operators.rs their Operator wrappers (ForcingOp, LedgerOp, CensusOp, plan 78.25 ReasonOp)
    src/reason.rs                  plan 78.25 THE REASON OPERATOR, game-free: `reason(g, s, judge, depth, fuse) -> Report` (+ `reason_within(.., seconds)`: ONE deadline for the whole chain, through `census::census_within(.., seconds)`; a reading the clock cut or left no time for is counted in `cut`, the chain is then a GAP) = the threats against the side to move by stake (the census of the other side after a pass, no defect; no pass: `must_answer`, every move read as an answer), each threat's FRAME (the base facts its line removes or adds, the facts every legal move changes left out), the exact NET of every move of ours one ply deeper (one ledger search, `root_values`), the ANSWERS (net beats `-stake`; the question asked of each is whether they can still carry the threat out: the threat's first move, if still legal, is read exactly (`Search::gain` is pub for this) and its line searched for the targets -- Parry = they cannot, Counter = they can and our net if they do (`if_executed`, with `executed_line`) is above 0 (a greater threat), Cover = it is 0 (the taker is taken back or matched), Lessen = it is below 0 but better than passing; a goal threat has parries only; a threat's line is cut at its last ledger event so the frame holds its own facts only), the IRRELEVANT units (the facts of ours a move removes, owner read off the roles' names in the facts; a unit none of whose moves answers and none of whose facts is in the frame; moves left unread by a cut are reported unread, never irrelevant), with no threat OUR tactics (census, one defect) as the candidates; `explain(&Report) -> Vec<String>` = the chain as plain `reason:` lines; named REASON_DEPTH / REASON_FUSE / ANSWER_FUSE_FACTOR. An order and an explanation, never a verdict; py `reason`
  planning/                      cs_planning: keep, the density prior; src/mcts.rs UCT Monte Carlo tree search (Rollout pluggable; MctsOp answers Goal::Estimate);
                                 tests/mcts.rs on tic-tac-toe in GDL
    src/rollout.rs                 the stronger rank source as pluggable pieces: Leaf (Draw, Balance: value of a cut playout), Greedy rollout, Prior + Weighted rollout,
                                   GainPrior (one notion of gain for Greedy and Weighted)
    tests/rollout.rs               each piece alone, on a toy units-only Game
  memory/                        cs_memory: content-addressed snapshots, the lemma language, the one matcher, cards (the corpus lemmas are verified on), LemmaOp; src/op.rs also `delta` and plan 78.24's `Effects` (the one reader of a move's effect, state or derived delta relations)
  decision/                      cs_decision: Params (plan 78.25: `reason`, default off), the Fixed controller (lemma, proof, [reason: REASON_SHARE of the clock, its candidates join the lemmas' as the ledger's `prefer`], ledger, prior), batch (JSON Lines log), cards (the card builder)
    src/lemma_prior.rs             LemmaPrior: a snapshot's bound lemmas as a playout Prior (cs_planning::rollout), matched as LemmaOp does
    tests/lemma_prior.rs           the lemma prior on a small GDL game; an empty snapshot gives uniform weights
    src/graph.rs                   plan 78.21: GraphSpec::new(description text, graph vocabulary JSON), .graph(g, s) -> Graph: the typed fact graph of a position, generic (terms typed by sort,
                                   product sorts fused into objects with part<i> edges, always-present constants, static / state / derived fact nodes, move nodes with removes / adds, an empty claim type, reverses;
                                   constant term / relation vocabularies read from the description's sorts and signatures)
    src/graph_net.rs               plan 78.21: GraphNet::load / from_json / validate(spec) / forward(&Graph) -> one score a move: the torch GraphNet's forward (SAGEConv per edge type, sum, residual LayerNorm, MLP head) from the JSON weights file, plain-loop matrix products
    src/graph_prior.rs             plan 78.21: GraphPrior {spec, net}: scores(g, s) -> (move names, scores), a cs_planning Prior (softmax x n; uniform on error), timing counters (facts / graph / forward)
    tests/graph.rs                 graph.rs on a made-up GDL game (sorts, an object, a scale, a static / state / derived relation): typing, fusion, constants, identical statics, edgeless constants, move effects; the forward pass on a hand-computed 1-layer dim-2 net; the prior
  learn/                         cs_learn: THE TRAINING KIT -- watching a teacher (src/watch.rs; Teacher is training-only in the type), the valuation miner,
    src/advice.rs                  Advisor (an outside source's ordering of the legal moves), TeacherOp (Kit::Training), ScriptedAdvisor for tests.
                                 No decision-path crate depends on cs_learn (tests/test_plan78_14.py reads the manifests)
  py/src/lib.rs                  cs_engine: the one Python entry (best_move, solve_batch (both take `reason=False`), plan 78.25: `reason(variant, fen=None, moves=[], depth=2, fuse=200000, worth={}, ours_first=False, valuation="") -> dict` (threats, answers, irrelevant, unread, leaving, ours, candidates, candidate_moves, evidence, readings, cut, nodes), build_cards, CardSet (card, eval, eval_many, save, load, merge, plan 78.23: revalued(values) -> CardSet: per-card {move: value}, a missing move cuts the card, None keeps it), estimate, facts, snapshots, certify, watch, mine, plan 78.21: graph, graph_vocab, GraphModel (rank / scores / timing))
```

The Python side of the loop is `catspace/incubator/analyst/discovery/` (wake, sleep, diagnose, abstraction, game_store);
plan 78.24 A: `game_store` named vocabularies (`vocabulary_file/vocabulary/spec_texts(game, name=None)` read `vocabulary.<name>.json`, whose optional `concepts` is a rules-delta path or list composed in order; `read_lemmas(game, vocabulary=None)` returns the lemmas of that vocabulary by `meta.vocabulary` (None = default, `GS.ANY` = all); `build/publish(game, ..., vocabulary=None)`); `abstraction` gains the step DROP A RESPONSE LITERAL (`drop_response(lemma, side, i)`; `size` counts response literals; `Gates.override(["NAME=VALUE"])`); a delta relation may be DERIVED: `cs_memory::op::Effects::at(view, relations, base).of(after_base)` reads a move's effect (base facts when every delta relation is a state relation, else the full fact views before and after), used by card_of, LemmaOp, LemmaPrior and GraphSpec::graph; tests/test_plan78_24.py (+ rust/engine/decision/tests/lemmas.rs); the learned knowledge is
`rules/learned/<game>/` (vocabulary, lemmas) and its immutable snapshots under `data/derived/snapshots/<game>/`.

Plan 78.21, the human concept vocabulary as rules: `rules/concepts/chess.gdl` is a DELTA (additive only) on `rules/chess.gdl` holding the concepts (defended, hanging,
pinned, fork, open_file, half_open_file, passed_pawn, isolated_pawn, doubled_pawn, absent, bishop_pair) and the static `diagonal_of` (26 named diagonals);
`rules/learned/chess/graph_vocabulary.json` names what a graph reader sees (state, derived, static relations, `objects`, `scales`, the `concepts` delta), read by
`game_store.graph_vocabulary(game)` / `graph_spec_texts(game)`; each concept is tested against a python-chess oracle in `tests/test_plan78_21_concepts.py`.

Plan 78.24 B, relations between units: `rules/concepts/chess_units.gdl` is a second DELTA composed after `rules/concepts/chess.gdl` (additive only): the sort `distance` (d0..d7, never the rank
numerals) with static `file_gap` / `rank_gap` / `dist_max` / `closer` / `reach`, and `apart` (king-step distance between a king and another unit), `flight`, `opposition`,
`diagonal_opposition`, `to_promote`, `king_ahead`, `in_square`. `rules/learned/chess/vocabulary.relational.json` is the second lemma vocabulary (the keys of `vocabulary.json` plus
`concepts`, a list of deltas composed in order); each relation is tested against a python-chess oracle in `tests/test_plan78_24_concepts.py`.
