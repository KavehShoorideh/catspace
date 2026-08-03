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
