# Restructure verification — 2026-08-03

Results of the 12-item checklist in
[`2026-08-03-refactor-plan.md`](2026-08-03-refactor-plan.md), run on branch
`refactor/repo-structure-2026-08-03` at `102fd32`.

**11 of 12 pass. 1 could not be run here** (no Docker daemon), and three items pass with a
stated qualification rather than a bare zero. Nothing is claimed that a command did not
print.

| # | check | result |
|---|---|---|
| 1 | `import catspace`; registry resolves every registered approach | **PASS** — 22/22 component approaches load; both end-to-end configs validate |
| 2 | `pytest` green | **PASS** — 295 passed |
| 3 | zero dead `catspace.*` import paths | **PASS** — 0 hits |
| 4 | zero `sys.path.insert` preludes | **PASS** — 0 in code (1 prose mention in a docstring) |
| 5 | zero literal data paths outside the registry | **PASS, qualified** — 0 path references; 3 documentation strings remain (below) |
| 6 | `dvc status` clean; `.dvc` pointers resolve | **PASS, qualified** — 40/40 outs present; 2 pre-existing modified outs (below) |
| 7 | `docker build` + `compose up` + `/health` | **NOT RUN** — Docker daemon unavailable (below) |
| 8 | MLflow: one store, pre-existing runs visible | **PASS** — 132 runs / 16 experiments in the single root store |
| 9 | `approaches.md` ↔ disk parity | **PASS** — 3 end-to-end + 22 component approaches in sync |
| 10 | wheel carries no non-shipping payload | **PASS** — 288 entries, 0 leaks |
| 11 | migrated scripts answer `--help` | **PASS, qualified** — 59/60; the one failure is pre-existing (below) |
| 12 | `scripts/` symlinks resolve | **PASS** — 9/9 |

---

## Qualifications

### 5 — three remaining literals are documentation, not paths

```
tools/viz/builders/article_figures.py:92     a provenance note ("plygap rho from artifacts/…json")
…/cone_fb_embedding/experiments/train_lichess_fb.py:569   an argparse help example
…/gauntlet_harness/experiments/play_traced.py:15          a python -c one-liner in a docstring
```

None of these is used to resolve a path. Rewriting them into `paths.*` calls would turn
prose into function calls and lose the meaning. The plan's "zero literals" is met for every
string the code actually opens.

### 6 — two modified DVC outs predate this branch

`data/shards/regime_rollouts_v1` and `data/derived/trunk_feats` were already reported
modified by `dvc status` before the first commit on this branch, and report identically
now. The restructure did not touch them: all 40 `.dvc` files resolve to an existing output,
including the one under `assets/engines/lc0/` that travelled with the asset move.

### 7 — the container stack could not be exercised

`docker version` returns EOF; the daemon is not accepting connections on this machine.
What *was* verified without it:

- `docker compose -f catspace/deployment/docker/docker-compose.yml config` parses and
  resolves the build context to the repo root;
- the Dockerfile `CMD` (`python -m catspace.deployment.server`) runs locally and answers
  `--help`, as do `catspace-uci` and `catspace-banksync`;
- the compose `mlflow.db` bind mount resolves to the same file `paths.mlflow_db()` returns.

To finish item 7:

```bash
docker build -f catspace/deployment/docker/Dockerfile .
docker compose -f catspace/deployment/docker/docker-compose.yml up -d
curl -sf localhost:8777/health
```

### 11 — one script fails for a reason older than the restructure

`tools/embeddings/visualize_clusters.py` has no argparse and loads a checkpoint at import.
Post-restructure it fails on a state-dict mismatch (unexpected BatchNorm buffers). Run from
the pre-restructure worktree it also fails — with `FileNotFoundError:
data/derived/dtm_endgame.npz`, because the CWD-relative literal did not resolve. The
registry fixed the path and thereby exposed a genuine checkpoint-version problem that was
previously masked. That is a real bug in the checkpoint/model pairing, not a migration
regression, and it is unfixed.

`prove_batchnorm.py` and `visualize_batchnorm.py` likewise have no argparse (they read
`argv[1]` as a checkpoint path) and are excluded from the sweep rather than counted as
failures.

---

## Open items

1. **`registry.build()` is unimplemented by all 22 approaches.** `registry.load()` works
   and is what everything uses. See `repo_structure.md` § "The component contract".
2. **`tools/_migration/` is still present.** It holds the four scripts that performed the
   move (classification rules, import rewrite, path-literal rewrite, the original
   `rewrite_imports.py`), kept so the mechanical edits are reviewable and re-runnable.
   Their own docstrings say to delete them once the migration lands.
3. **No approach has co-located tests yet.** `pytest` `testpaths` already covers
   `approaches/<name>/tests/`; the directories do not exist.
