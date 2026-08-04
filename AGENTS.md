# AGENTS.md

Read **`repo_structure.md`** before adding or moving anything. This file is the short,
enforceable subset.

## Layout in one line

`catspace/` is the wrapper. It wires four components — encoder, planner, search, memory —
each holding competing **approaches**. A named end-to-end approach in
`catspace/approaches/` picks one per slot; `catspace/deployment/` consumes an end-to-end
approach and never reaches into a component directly.

## Rules

1. **No new top-level packages.** Everything is `catspace.*`. Never add a
   `sys.path.insert` — the package is installed editable; if an import fails, the import
   is wrong.

2. **Never write a relative data path.** Use the registry:
   ```python
   from catspace.io import paths
   paths.derived("x.npz")   paths.experiment("y.json")   paths.figure("z.png")
   ```
   `"data/derived/x.npz"` only works when the process was launched from the repo root.
   Third-party downloads go under `assets/`, never `data/`.

3. **`src/` ships; `experiments/` does not.** An approach's `src/` goes in the wheel and
   runs in the container. Shipping code must never import from an `experiments/` directory
   — if the engine loads it at play time, it belongs in `src/`.

4. **`experiments/`, `logs/`, `artifacts/`, `data/` never get an `__init__.py`.** That is
   what keeps them out of the wheel. They stay importable as PEP 420 namespace portions.

5. **Every approach folder has an entry in its `approaches.md`, and vice versa.** Schema
   in `repo_structure.md`. Run `python scripts/check_approaches.py` — it fails on drift
   and validates that every end-to-end config's slots resolve.

6. **Name approaches by string** (`"search:puct_mcts"`) in end-to-end configs. Only
   `catspace/registry.py` does deep imports.

7. **Subgoals enter through the prior, never the value.** The value is the global
   objective (`catspace/interfaces.py`, DECISIONS §8).

8. **Never rewrite dated or archived docs** to match the current layout — JOURNAL.md,
   MILESTONES.md, `research/docs/archive/`, `research/docs/writing/`,
   `research/weekly_report/`. They are the research record. Add a banner if needed.

9. **Tests**: cross-component and integration in root `tests/`; one-approach unit tests in
   `approaches/<name>/tests/`.

10. **No number is quoted in a doc unless a script printed it.** Every run emits `VERDICT`
    lines. An `approaches.md` entry with no results says `—`.

## Commands

```bash
.venv/bin/python -m pytest -q             # 295 tests, ~4 min
.venv/bin/python scripts/check_approaches.py
catspace-uci | catspace-server | catspace-banksync
docker compose -f catspace/deployment/docker/docker-compose.yml up
```
