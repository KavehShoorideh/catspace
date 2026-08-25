# RUNBOOK — run, reproduce, monitor

*Rewritten 2026-07-28 for the current line; the pre-rebuild runbook (toy-era field runs, play
atlas) is preserved at `docs/archive/RUNBOOK.md` — those commands still run but reference
superseded checkpoints. Python is always `.venv/bin/python` from the repo root. GPU rule: one
training job at a time; one disk-heavy job at a time; eval tooling on CPU while MPS is busy.*

## 0. Durable launch (multi-hour jobs)

Long runs must survive the launching terminal/session closing. Use the wrapper — detaches to
launchd, blocks idle sleep for the job's lifetime, writes a timestamped log + stable symlink +
pidfile:

```bash
experiments/launch.sh <name> -- .venv/bin/python -u experiments/<script>.py ...
tail -f artifacts/experiments/<name>.log        # monitor (symlink -> newest)
kill "$(cat artifacts/experiments/<name>.pid)"  # stop
```

Do NOT rely on `&` from an interactive shell or a session-tied background mode for multi-hour
jobs. Watch cadence per TESTING §2.8: first check ~1 min, then ~5 min.

## 1. The reachability-field pipeline (current line)

```bash
# 1. opponent decision reconstruction (fast, ~1 min)
.venv/bin/python experiments/build_opp_positions.py

# 2. feature caches (Maia-2 candidates + trunk phi; resumable shards; ~5 min per 85k on MPS)
.venv/bin/python -u experiments/m2b_cache.py \
  --positions data/derived/m2b/positions_dense_opp.parquet \
  --out data/derived/m2b/cache_dense_opp --shard-size 8192

# 3. reach dataset (goal bank + first-hit labels; audits gate the write; ~10 s)
.venv/bin/python experiments/build_reach_data.py

# 4. train the head — smoke first (600 steps, ~5 min), then full
.venv/bin/python -u experiments/train_reach_head.py \
  --out artifacts/experiments/reach_v1_smoke --steps 600
.venv/bin/python -u experiments/train_reach_head.py \
  --out artifacts/experiments/reach_v1_full --steps 8000
```

The trainer prints the pre-registered VERDICT block (z-lift CIs, wrong-z placebo, calibration
bins, plies-MAE, eff-rank). Numbers go to JOURNAL only from those lines.

## 2. Player model (M2) — evaluation entry points

```bash
.venv/bin/python experiments/m2b_condition.py     # infer-then-condition lift (held-out players)
.venv/bin/python experiments/m2c_ingame.py        # cold-start break-even curves (5..160 moves)
.venv/bin/python experiments/m2c_elo_id.py        # Elo-from-moves recovery
.venv/bin/python catspace/style/estimator.py      # online (Elo,z) filter self-test (6/6)
```

## 3. Crossing risk & basins

```bash
.venv/bin/python catspace/transition.py           # SF-refereed crossing-risk demo (weaker crosses more)
.venv/bin/python experiments/msm_basins.py        # MSM/PCCA basins
.venv/bin/python experiments/engine_vs_human_basins.py   # the thesis figure
.venv/bin/python experiments/committor_by_material.py    # the crystallization figure
```

## 4. Play & A/B

```bash
.venv/bin/python experiments/play_vs_maia.py --maia 1100        # vs the Maia ladder
.venv/bin/python experiments/playout_ab.py --n 100 ...          # paired playout A/B (PLAYOUT_AB line)
bash gauntlet.sh ...                                            # fastchess SPRT
```

Protocols (node-budget vs timed) and bars: `../MILESTONES.md` §match protocols. Every match:
alternating colors, diversified openings, SPRT/e-process with n pre-registered, PGNs kept,
MLflow-logged.

## 5. Tests

```bash
.venv/bin/python -m pytest tests/ -q              # suite (~4.5 min)
.venv/bin/python experiments/losses.py            # loss invariants
.venv/bin/python experiments/endgame_handover.py  # tablebase handover
```

## 6. Tracking, data, disk

- **MLflow**: local `./mlruns`; `mlflow ui` to browse. Every training run logs params/metrics/
  tags via the scaffold.
- **DVC**: `dvc add <dataset>` after any build (autostage on); commit the `.dvc` pointer.
  Never commit raw bytes; never re-ingest tracked data (lichess 2019-01 shards: ~10 GB,
  ingested once).
- **Disk**: check `df -h` before cache builds (m2b_cache pre-flights and refuses); the
  tablebase probe cache is sqlite `journal_mode=DELETE` (WAL banned — filled the disk twice).
- **Engines/nets**: syzygy under `data/syzygy/`; Maia lc0 nets + Maia-2 ONNX under
  `maia2_models/`; Stockfish on PATH. `maia2_models/` is gitignored (the weights are 267 MB,
  over GitHub's 100 MB limit) — a fresh clone starts without it and
  `maia2.from_pretrained(type, device)` gdown-downloads into `./maia2_models`, skipping if
  present.

## 7. Where things live

- Chronology & verdicts: `../JOURNAL.md`. Design: `THESIS.md`. Map: `COMPONENTS.md`.
- Training logs/checkpoints: `artifacts/experiments/` (gitignored patterns; DVC for datasets).
- Session auto-memory (rules that persist across sessions):
  `~/.claude/projects/-Users-kav-code-remote-github-catspace/memory/`.

---

## 7. The concept-bottleneck line (gko, current)

Kaveh 2026-08-25: *"record the new way to run the training and testing so that we get
the same behaviour."* Everything below is one command; no flag lives only in a shell
history.

### Where the architecture is defined

| | |
|---|---|
| the model, and the contract its heads obey | `.../reach_probability/experiments/gko_model.py` |
| the contract in prose, with shapes | `docs/ARCHITECTURE.md` |
| a specific run's flags | `experiments/configs/<run>.sh` — version-controlled, not a temp file |

`gko_model.py` holds `HEADS`, a table where every head declares what it reads. The enum
it draws from has **no member for pre-quantisation features**, and `check_head_widths()`
fails at startup if a head's real input width disagrees with its declaration. Add a head
by adding a row; do not wire one up ad hoc.

### Running a training

```bash
experiments/configs/gko9a.sh gates    # unit gates, exits non-zero on failure
experiments/configs/gko9a.sh smoke    # 60 steps end to end
experiments/configs/gko9a.sh          # the full 8000-step run
```

The three modes share one flag list, so the smoke and the real run cannot drift apart.
Copy the file to start a new run; edit the flags there, never at the prompt.

**Never run two model-loading processes at once.** Not a gate suite beside a trainer,
not a benchmark, not a thirty-second diagnostic. This has now killed one run outright
(gko7a, step 4000 of 8000) and paged another down to 100 MB resident. Check first:

```bash
vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} \
  /Pages purgeable/{p=$3} END{printf "%.1f GB free\n",(f+i+p)*ps/1073741824}'
```

Note `ps` — the page size is **16 KB** on this machine, not 4 KB. Hardcoding 4096
under-reports free memory by 4x, which it did in every measurement I took before
2026-08-25 12:00.

### Verifying a run behaves as intended

```bash
# every parameter name and shape -- the equivalence test for any refactor.
# The list must not change, or existing checkpoints stop loading.
experiments/configs/gko9a.sh 2>/dev/null | grep -E "^  [a-z_]+\."   # with --dump-arch

# what each head actually reads, generated from HEADS (printed by every run)
grep -E "^\[gko0\] heads|CONTRACT VIOLATION" artifacts/experiments/<run>.log
```

### Test suites

```bash
.venv/bin/python catspace/research/tools/training_infra/losses.py         # every loss term
.venv/bin/python -m catspace.research.tools.training_infra.train.test_resume
.venv/bin/python -m catspace.research.tools.lint_patterns                 # silent-failure ratchet
```

`test_resume` covers the resume path *and* `frozen()`; each test carries a negative
control, because a resume test passes just as happily on an implementation that never
needed the state.

`lint_patterns` is a ratchet against `lint_baseline.json`: known instances are
tolerated, any NEW one fails. Fix a site, then `--baseline` to lower the count.

### Measurement entry points

```bash
# how much accuracy an INTERPRETABLE readout costs vs the incumbent head
.venv/bin/python -m catspace.research.tools.training_infra.interpretable_wdl --ckpt <ckpt>

# how hard each loss term pushes the ENCODER (needs --resume-self for a trained model)
experiments/configs/gko9a.sh   # with --grad-audit N --resume-self <ckpt>

# within-position sibling discrimination; --pv-cache makes two checkpoints
# comparable on IDENTICAL positions and identical Stockfish futures
.venv/bin/python -m catspace.research.tools.training_infra.future_concepts \
    --ckpt <ckpt> --pv-cache artifacts/experiments/fut_pv_cache.json --n_pos 120

# what the discovered future-kinds actually ARE, in chess terms (no model, no GPU)
.venv/bin/python -m catspace.research.tools.training_infra.name_the_kinds
```

### Future-kind labels (only when regenerating them)

```bash
future_kind_labels --stage pv      # Stockfish PVs; CPU-only, safe beside a trainer
future_kind_labels --stage gate --ref-ckpt <ckpt> --device mps
future_kind_labels --stage fit     --ref-ckpt <ckpt> --device mps --jobs 4
future_kind_labels --stage assign  --ref-ckpt <ckpt> --device mps --jobs 4
```

`assign` skips shards that already carry labels — **do not pass `--force` to resume**,
or it relabels from shard 0 and four restarts overlap on the same work.
