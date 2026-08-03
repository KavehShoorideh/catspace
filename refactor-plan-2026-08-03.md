# Plan: catspace repo restructure (FINAL — approved, ready to execute)

## Status
User approved all 20 review findings + gave final constraints. Ready to execute (Plan mode cannot edit files — needs mode switch).

## Final user constraints (round 4)
- Do NOT rewrite past documentation (JOURNAL.md, MILESTONES.md, docs/archive/*, writing/*, reports/*, artifacts/RESULTS-*.md). Only PREPEND a short banner: "a refactor occurred 2026-08-03; paths below are historical and no longer valid; see <refactor plan link>". Contents otherwise untouched.
- Refactor plan doc goes under docs, dated: `catspace/research/docs/2026-08-03-refactor-plan.md`.
- ALL data pointers and ALL MLflow pointers must be adjusted.
- `repo_structure.md` at REPO ROOT + root `AGENTS.md` so agents actually load the rules.
- Execute the refactor.

## Accumulated decisions (rounds 1-3)
- Root `catspace/` = the WRAPPER package. `research/` and `deployment/` live INSIDE it. Still `import catspace`.
- `research/components/` = exactly 4: planner, search, memory, encoder. No `wrapper` component.
- End-to-end configs in `catspace/approaches/<config>/`: each selects which planner + search(es) + encoder(s) + memory approach(es) to wire, holds the glue, owns its experiments/logs/artifacts/data. This layer is what `deployment/` consumes.
- predictor/* + style/* -> ALL into planner approaches (incl. opponent modeling).
- MCTS MERGE: `nn/mcts.py` + `search/mcts.py` -> ONE `puct-mcts` approach taking an externally-supplied guidance function (reach-guided / random / value-guided pluggable).
- `two_field.py` SPLIT: `score_components()` + `TwoFieldPolicy` -> planner approach `two-perspective-scoring`; `effective_distance()` (fast MemoryField re-pricing) -> search, as a re-pricing hook in `puct-mcts`.
- Shared chess-specific + generic infra -> `research/tools/` organized BY TYPE. No separate `common/`.
- Legacy cleanup: delete alias-shim dirs/files, empty `probe/`, stale `latent_chess_planner.egg-info/`.
- Generated data lives near its generator under a `data/` subfolder, DVC-tracked; ACCESS routed through the paths registry.

## TARGET STRUCTURE

```
/ (repo root)
  repo_structure.md            # THE instruction doc (root, per user)
  AGENTS.md                    # short, points to repo_structure.md + enforceable rules
  pyproject.toml, requirements.txt
  .gitignore (rewritten), .dockerignore (NEW), .dvcignore, .dvc/
  .git-blame-ignore-revs (NEW)
  README.md, JOURNAL.md, MILESTONES.md, human-written-alt-arch.md   # banner-noted only
  config.json                  # fastchess eval config; engine path updated
  scripts/                     # symlinks recreated to new paths
  tests/                       # cross-component / integration only
  contrib/                     # unchanged
  mlruns/, mlflow.db           # tracking stores stay at root; pointers fixed to resolve here
  catspace/                    # THE WRAPPER PACKAGE
    __init__.py
    interfaces.py              # from engine/interfaces.py — the 4 component plugin Protocols
    engine.py                  # LayeredEngine composer (engine/engine.py)
    orchestrator.py, introspection.py, watchlist.py, priors.py, fields.py, values.py
    registry.py                # NEW: resolve component approach by string name
    incubator/                 # explicit "no home yet" holding pen
    approaches.md              # registry of end-to-end configs
    approaches/
      <config-name>/{config.(py|yaml), wiring.py, experiments/, logs/, artifacts/, data/}
      uci-engine/              # from experiments/uci_engine.py
      http-assistant/          # analysis logic behind assistant_server (server shell in deployment)
      gauntlet-harness/        # harness/play.py, arena_real.py, play_vs_maia.py, play_traced.py, gauntlet.sh
    research/
      components/
        planner/  {README.md, approaches.md, approaches/<name>/{src,experiments,logs,artifacts,data,tests}}
          approaches: subgoal-cascade, atlas-region-stats, endgame-groundtruth, reach-field,
                      committor-value, opponent-model (predictor/opponent + style/),
                      two-perspective-scoring, armed-tactics (armed/)
        search/   approaches: puct-mcts (merged, pluggable guidance + fast-field re-pricing hook),
                              anytime-path (nn/anytime.py)
        memory/   approaches: vector-store-retrieval, goal-region-bank, plan-ledger,
                              checkpoint-trap-bank, fast-field-knn, experience-store, competence-map
        encoder/  approaches: reachability-field, jepa-tokenizer (nn/ representation stack),
                              control-field-wdl (status: parked), cone-fb-embedding,
                              concept-quantization (concepts.py)
      tools/
        probes/ figures/ embeddings/ ablations/ viz/ stats-eval/ training-infra/ io/ chess-specific/
        (each may have its own data/; search/memo.py -> tools/ as generic LRU)
      logs/            # orchestration-level logs not owned by one approach
      docs/            # docs/*.md + 2026-08-03-refactor-plan.md + archive/ + writing/
      infra/           # infra/ moved unchanged
      weekly-report/   # reports/phase-2.md + future weekly reports
      assets/          # THIRD-PARTY ONLY
        models/maia2/ engines/{lc0,maia}/ tablebases/syzygy/ corpora/{raw-pgn,eco,lichess-raw}/
      data/            # shared cross-cutting generated datasets (registry-resolved)
    deployment/
      server/          # assistant_server.py promoted; uci entrypoint; banksync entrypoint
      web/             # ui/README.md + future board UI
      docker/          # docker-compose.yml, Dockerfile, nginx.conf, prometheus.yml, grafana/
```

### Import strategy (resolves the deep-path problem)
- No new top-level packages. Everything is `catspace.*`.
- Approach source dirs (`approaches/<name>/src/`) get `__init__.py`. `experiments/`, `logs/`, `artifacts/`, `data/` must NOT (keeps them out of the wheel and out of setuptools find()).
- Each component `__init__.py` exposes `get_approach(name)`; `catspace/registry.py` aggregates. End-to-end configs reference approaches BY STRING, never deep import paths. Only the registry does deep imports.
- Deleted alias shims are NOT reintroduced.

## THE 20 REVIEW ITEMS — all in scope

### Code with no home in the earlier draft (now assigned)
- `catspace/engine/*` -> backbone of root `catspace/` (interfaces.py = plugin Protocols; engine.py = LayeredEngine; orchestrator/introspection/watchlist/priors/fields/values alongside). `engine/search.py` folds into merged puct-mcts.
- `catspace/armed/` -> planner approach `armed-tactics`.
- `catspace/concepts.py` -> encoder approach `concept-quantization`.
- `catspace/harness/` -> `catspace/approaches/gauntlet-harness/`.
- `catspace/incubator/` -> stays at wrapper level, documented.
- Root md docs stay at root (banner-noted). `artifacts/*.md` (RESULTS-v2/v3, roadmap-v2, filmstrip, region-discovery-feasibility) -> `research/docs/archive/` (banner-noted, untouched).
- `artifacts/vendor/` (node_modules for viz) -> `research/tools/viz/vendor/`.
- `contrib/` unchanged at root. `config.json` stays at root, engine invocation path updated.

### Mechanical breakage fixes
1. **~250 `sys.path.insert(0, parents[1])` lines** — DELETE all (catspace is installed editable). First audit for `from experiments.` sibling imports and rewrite those.
2. **`io/paths.py` `REPO_ROOT = parents[2]`** and **`tracking.py` `parents[1]`** — replace with upward walk for `pyproject.toml`/`.git` marker so nesting depth stops mattering.
3. **Dataset path registry** — `research/tools/io/paths.py` = single registry, named accessor per dataset, physical location declared in ONE table. All ~53 files with hardcoded CWD-relative argparse defaults (`data/derived/...`, `artifacts/experiments/...`, `data/shards/...`) rewritten to call it. This is what makes "data near its generator" survivable.
4. **MLflow pointers** — `tracking.py` (`parents[1]/mlflow.db`), `experiments/register_incumbents.py` (CWD-relative `Path('mlflow.db')`), `train/scaffold.py` docstring, `deploy/docker-compose.yml` (`../mlflow.db` bind mount), `infra/observability/run_metrics.py` — all repointed to registry-resolved root `mlflow.db` / `mlruns/`.
5. **`.gitignore` rewrite** — `/data/**` no longer covers scattered data dirs. Move to `**/data/**` + `!**/data/**/` + `!**/data/**/*.dvc`, plus `**/logs/**`, `**/artifacts/generated/**`; keep `*.pt`/`*.npz`; update `maia2_models/` rule to assets path.
6. **Packaging leak** — `include = ["catspace*"]` would sweep the whole research tree into the wheel. Add explicit `exclude`; no-`__init__.py` rule is second line of defense. `deployment/` = importable subpackage (`python -m catspace.deployment.server`).
7. **`.dockerignore` (new)** — exclude research/, data, artifacts, tablebases, mlruns, .venv, .git.
8. **`scripts/` symlinks** — recreate (not re-point); update `scripts/README.md`.
9. **Phase ordering** — one component at a time, tests green after each. No big-bang.
10. **`.git-blame-ignore-revs`** — add, listing bulk-move commits.

### Conventions repo_structure.md must define
- **`approaches.md` schema** (mandatory per approach): name, folder, status (`active` | `parked` | `superseded-by:<x>`), one-line hypothesis, definition-of-done, results link, date added, owner-of-record.
- **Component plugin contract** — Protocol signatures from `catspace/interfaces.py`: encoder `encode(board, clock, ...) -> embedding`; planner `propose_subgoals(state) -> ranked goals`; search `find_move(state, goal, guidance_fn) -> move`; memory `query(...)`/`store(...)`. New approaches must satisfy their component Protocol to be selectable in an end-to-end config.
- **Test placement** — component unit tests co-locate at `approaches/<name>/tests/`; root `tests/` keeps cross-component + integration; `pytest testpaths` updated to both.
- **Production vs research** — state explicitly that `approaches/*/src/` is SHIPPING code despite living under `research/`; only `experiments/`, `logs/`, `artifacts/`, `data/` under an approach are non-shipping.
- **Data rule** — generated data under the generating approach's `data/`, DVC-added, ALWAYS accessed via the registry, never literal relative paths.
- **Historical docs rule** — never rewrite dated/archived docs; add a banner instead.

## EXECUTION PHASES

**P0 Prep** — branch `refactor/repo-structure-2026-08-03`; baseline `pytest` + `dvc status` + file inventory manifest committed.

**P1 Foundations (before any moves)** — marker-based `REPO_ROOT`; build dataset path registry; rewrite `.gitignore`; add `.dockerignore`, `.git-blame-ignore-revs`. Verify tests green, no moves yet.

**P2 Skeleton** — create `catspace/{research/{components/{planner,search,memory,encoder},tools/*,logs,docs,infra,weekly-report,assets,data},deployment/{server,web,docker},approaches}` with README.md + approaches.md stubs.

**P3 Wrapper backbone** — promote `engine/*` into root `catspace/`; add `registry.py`. Verify `import catspace` + tests.

**P4-P7 One component per phase** (encoder -> memory -> search -> planner; green tests each):
move source into `approaches/<name>/src/`, delete alias shims, repoint imports, move tests, move data under registry control, update `approaches.md`.
- P6 (search) also performs the MCTS MERGE.
- P7 (planner) also performs the `two_field.py` split.

**P8 Wrapper approaches** — create `catspace/approaches/{uci-engine,http-assistant,gauntlet-harness}` + one named end-to-end config reproducing today's default wiring.

**P9 Deployment** — `deploy/*` -> `catspace/deployment/docker/`; `experiments/viz/assistant_server.py` -> `catspace/deployment/server/`; `ui/README.md` -> `deployment/web/`. Rewrite Dockerfile COPY/CMD (`python -m catspace.deployment.server`), compose volumes, prometheus target. Verify build + compose smoke.

**P10 Tools/docs/assets/remaining experiments** — `tools/*` + generic `experiments/*` -> `research/tools/<type>/`; remaining experiments -> owning approach's `experiments/`; `docs/*` -> `research/docs/`; `writing/*` -> `research/docs/writing/`; `reports/*` -> `research/weekly-report/`; `infra/` -> `research/infra/`; third-party -> `research/assets/`. Delete `latent_chess_planner.egg-info/`, empty `probe/`. Strip the ~250 `sys.path.insert` lines; rewrite ~53 hardcoded data paths to registry calls.

**P11 Packaging/entrypoints** — pyproject include/exclude + testpaths + console entrypoints; reinstall editable; recreate `scripts/` symlinks; update `config.json` engine path.

**P12 Docs & instructions** — root `repo_structure.md` + root `AGENTS.md`; `research/docs/2026-08-03-refactor-plan.md`; prepend historical banner ONLY to JOURNAL.md, MILESTONES.md, README.md, human-written-alt-arch.md, `research/docs/archive/README.md`, `research/docs/writing/README.md`, `research/weekly-report/phase-2.md`. No content rewrites.

**P13 Verification** (below), then merge with blame-ignore revs recorded.

## VERIFICATION
1. `python -c "import catspace"`; `catspace.registry` resolves every approach listed in every `approaches.md`.
2. `pytest` green (root + co-located approach tests).
3. Zero grep hits for dead import paths (`catspace.predictor|atlas|navigator|reach|value|opponent|endgame|subgoals|field|goal_bank|tb|vectordb`), excluding banner-noted historical docs.
4. Zero hits for `sys.path.insert(0, str(Path(__file__).resolve().parents`.
5. Zero literal `"data/derived/`, `"data/shards/`, `"artifacts/experiments/` outside the registry.
6. `dvc status` clean; all `.dvc` pointers resolve (moves via `git mv`/`dvc move`, never copy+delete).
7. `docker build -f catspace/deployment/docker/Dockerfile .` succeeds; `docker compose up` brings up engine+web+qdrant; `/health` 200; prometheus scrapes engine.
8. MLflow: `tracking.py` and `register_incumbents.py` write to the SAME root store; `mlflow ui` shows pre-existing runs (no orphaned store).
9. Script-checked: every approach folder on disk appears in its `approaches.md` and vice versa.
10. Built wheel contains no `experiments/`, `logs/`, `artifacts/`, `data/` payload.
11. Spot-run 10 migrated experiment scripts with `--help` — no import/path errors.
12. All `scripts/*` symlinks resolve.
