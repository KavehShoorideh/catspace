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
  `maia2_models/`; Stockfish on PATH.

## 7. Where things live

- Chronology & verdicts: `../JOURNAL.md`. Design: `THESIS.md`. Map: `COMPONENTS.md`.
- Training logs/checkpoints: `artifacts/experiments/` (gitignored patterns; DVC for datasets).
- Session auto-memory (rules that persist across sessions):
  `~/.claude/projects/-Users-kav-code-remote-github-catspace/memory/`.
