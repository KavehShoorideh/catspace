# RUNBOOK — reproduce the train / eval / analyze commands

Every command here was actually run this session (2026-07-18) and its verdict
is recorded in JOURNAL.md. Run python as `.venv/bin/python` from the repo root.
GPU rule: **one job at a time**, and interactive/eval tooling runs on CPU while
a training job holds MPS. Numbers only count when they come from a printed
`VERDICT`/`PLAYOUT_AB` line (see [[journal_numbers_must_be_verdicts]] rule).

Checkpoints referenced:
- Incumbent (plays at 0.600 conversion): `data/derived/sep/cert_base_full.pt`
  + phead `data/derived/sep/cert_base_full_phead.pt`
- Spread field, best healthy snapshot: `data/derived/sep/qrl_iqe_leak_step10000.pt`
  (+ `_phead.pt`) — saved BEFORE the step-18000 collapse; the largest-spread
  field produced so far. `qrl_iqe_leak.pt` itself is the post-collapse halt-save
  (do NOT use it as the field).
- Test set of winning KRRvKBP starts: `artifacts/experiments/krrkbp_test_n200.json`

---

## 0. Durable launch (survive terminal / VSCode / Claude Code close)

Long runs must NOT depend on the launching terminal staying open. Use the
wrapper — it detaches the job so it reparents to `launchd` (fully independent of
the terminal AND of Claude Code), blocks idle sleep only for the job's lifetime,
and writes a timestamped log + stable symlink + pidfile:
```
experiments/launch.sh <name> -- <command...>
# e.g. the full field run:
experiments/launch.sh qrl_iqe_sn_full -- \
  .venv/bin/python -u experiments/train_lichess_fb.py \
  --ckpt data/derived/sep/qrl_iqe_sn_full.pt --steps 40000 ...   # (full flags in 1b)
```
- Monitor: `tail -f artifacts/experiments/<name>.log` (symlink -> newest run)
- Stop:    `kill "$(cat artifacts/experiments/<name>.pid)"`
- Mechanism: `nohup` (ignore SIGHUP) + `disown` (drop from job table ->
  reparents to PID 1) + `caffeinate -i -w <pid>` (no-sleep, auto-exits with the
  job). Verify detachment with `ps -o ppid= -p <pid>` == `1`, TTY `??`.
- Do NOT rely on a plain background job (`&`) from an interactive shell, nor on
  Claude Code's own background-run mode, for multi-hour jobs: those stay tied to
  the session and can be reaped when it ends. The wrapper is session-independent.

## 1. Field training (the spread runs)

The objective: an IQE quasimetric F/B field that SPREADS (unreachable pairs far
apart) without collapsing to the all-F-above-all-B dead zone. Key mechanisms
added this session: leaky IQE (`--iqe-leak-beta`), PID spike guards
(`--qrl-pid-eclip`, `--qrl-lambda-max`, smaller `--qrl-pid-kd`), certified
unreachability repulsion (`--qrl-unreach-weight/-floor`), two-sided step pin.

### 1a. Smoke (fail-fast, ~10 min, validate before the long run)
```
caffeinate -i .venv/bin/python -u experiments/train_lichess_fb.py \
  --shards data/shards/lichess_db_standard_rated_2019-01.prefix4gb \
  --ckpt data/derived/sep/qrl_iqe_leak_smoke.pt --steps 2000 --fresh \
  --quasimetric --iqe --iqe-components 32 --iqe-embed-scale 1.0 \
  --iqe-leak-beta 10 \
  --qrl-objective --qrl-push-offset 15 --qrl-goal-pool 8192 --qrl-two-sided \
  --qrl-use-pid --qrl-pid-kd 0.25 --qrl-pid-eclip 3.0 --qrl-lambda-max 20 \
  --qrl-halt-on-collapse --qrl-var-weight 1.0 \
  --qrl-unreach-weight 8.0 --qrl-unreach-floor 30 \
  --committor-base --phead-weight 0.1 --lr 2e-4 \
  --d 512 --channels 128 --blocks 10 --enc-out 512 --batch 128 --seed 0
```

### 1b. Full run (40k steps, ~3.4 h at 3.4 it/s on MPS)
Same as 1a but: `--ckpt data/derived/sep/qrl_iqe_leak.pt --steps 40000
--ckpt-every 10000` (drop the `_smoke`). The `--ckpt-every 10000` is what saved
the healthy `qrl_iqe_leak_step10000.pt` before the collapse.

### 1c. Read the vitals (per-step `qrl push ...` line in the log)
Healthy: `d_step` ~1.0-1.3 (the unit-step pin), `d_rand`/`d_unr` large and
GROWING (spread; this run hit 129 at step 6000), `lam` <= the cap (20),
`var` > 0, no `COLLAPSE`. Dead zone: `d_step` -> 0, `d_rand` -> 0, `lam`
ratchets with no effect. The `--qrl-halt-on-collapse` detector halts on
`d_step ~ 0 over 2000 steps`.

### 1d. Flag glossary (the ones that matter for spread)
- `--iqe-leak-beta 10` — leaky (relaxed-relu) interval max; keeps a live escape
  gradient at the collapse SURFACE (repulsive), NOT deep in the zone. beta=0 =
  exact paper IQE. Bias d(x,x)~0.54 < step_cost.
- `--qrl-pid-eclip 3.0` — clip the violation fed to the PID; a transient sq_dev
  spike (45.5) otherwise drove lam 24->134 in one step (the implosion).
- `--qrl-lambda-max 20` — hard cap + anti-windup on the multiplier.
- `--qrl-pid-kd 0.25` — derivative gain (was 2.0; it was 2/3 of the spike).
- `--qrl-unreach-weight 8 --qrl-unreach-floor 30` — certified-unreachable pairs
  (nn/unreachable.py) pushed to distance >= 30, both directions (anchor-anchor).
- `--qrl-two-sided` — pin d(s,s') to EXACTLY 1 (penalize below too), the main
  ordering-collapse guard.

---

## 2. Evaluation & analysis (all CPU-safe; use --device mps only when GPU free)

### 2a. Conversion A/B vs the incumbent (the PRIMARY strength metric)
Paired, deterministic tablebase-optimal defender, n=100 winning starts.
`PLAYOUT_AB` line = mate-rate A vs B, paired diff, bootstrap CI, e-value.
```
.venv/bin/python -u experiments/playout_ab.py \
  --ckpt-a data/derived/sep/cert_base_full.pt \
  --ckpt-b <candidate>.pt \
  --search-a mcts --search-b mcts \
  --phead-a data/derived/sep/cert_base_full_phead.pt \
  --phead-b <candidate>_phead.pt \
  --nodes 800 --n 100 --label <NAME>
```
Variants used this session (flags on side B):
- `--mate-stop-b` — certified mate-stop ONLY. Proven move-identical => diff MUST
  be 0.000 exactly (proof-check). Verdict: diff +0.000 CI[0,0].
- `--decision-stop-b` — BOTH early stops (stability heuristic + mate-stop).
  Verdict: -0.040 conv (e=4.38) => SHELVED.
- `--search-b mctsplan` — two-budget plan persistence. Verdict: 0.440 vs 0.600
  => SHELVED (dominated by mcts@200).

### 2b. Energy Pareto (compute vs strength — the planner objective instrument)
`VERDICT ENERGY` line = conversion, rows/move (embed_F/B forwards), util, ms.
```
.venv/bin/python -u experiments/energy_baseline.py \
  --n 100 --policies mcts beam plan --budgets 200 800 1600 --device cpu
```
Single-config energy re-measures (append `--early-stop` for both stops,
`--mate-stop` for the certified stop alone):
```
.venv/bin/python -u experiments/energy_baseline.py \
  --n 100 --policies mcts --budgets 800 --mate-stop --device mps
```

### 2c. phead calibration gate (ECE + martingale, held-out games)
```
.venv/bin/python -u experiments/phead_calibration.py \
  --ckpt data/derived/sep/cert_base_full.pt \
  --phead data/derived/sep/cert_base_full_phead.pt \
  --shards data/shards/lichess_db_standard_rated_2019-01.prefix4gb
```
Verdict this session: ECE 0.052 CI[0.032,0.084], martingale drift ~0 (all phase
bins), overconfidence localized to the 0.8-0.9 band (0.849->0.717).

### 2d. Decision-flip probe (is deep search worth it / is it targetable?)
```
.venv/bin/python experiments/decision_flip_probe.py --n 30 --device mps
```
Verdict: 51% flip rate 200n-vs-800n, low-gap tercile captures 44.7% of flips
=> difficulty homogeneous => escalation NO-BUILD.

### 2e. Show the mate (SAN of the first converted toy game)
```
.venv/bin/python experiments/show_mate.py \
  --ckpt data/derived/sep/cert_base_full.pt \
  --phead data/derived/sep/cert_base_full_phead.pt --nodes 800
```

---

## 3. Tests
```
.venv/bin/python -m pytest tests/ -q                       # full suite (~4.5 min)
.venv/bin/python -m pytest tests/test_probe.py tests/test_cascade.py -q   # planner
.venv/bin/python -m pytest tests/test_leaky_iqe.py tests/test_invariants.py -q  # field
.venv/bin/python -m pytest tests/test_unreachable.py -q    # oracle
```

---

## 4. Planner primitives (built this session; play-validation pending a spread field)
- `catspace/planner/probe.py` — `probe(mcts, board, budget)` -> ProbeResult with
  CERTIFIED [lo,hi] (game-truth only; network confidence never certifies).
- `catspace/planner/cascade.py` — `DecisionCascade` (LUCB coarse->deepen,
  certified-dominance stop, resign/draw on certified bounds only).
- `catspace/nn/mcts.py` — `mate_stop` (certified, kept) and `decision_stop`
  (stability heuristic, shelved) flags; `game_truth()` provenance helper.
- `catspace/nn/mcts.py::FBPlanMCTSPolicy` — two-budget plan persistence (shelved).

---

## 5. Interactive play-atlas interface (play + analyze + t-SNE map + memory)
Local server (incumbent model on CPU, GPU untouched). Three steps:
```
# 1. build the atlas (CPU, ~40s at n=4000). Default = CERTIFIED games only
#    (mate|draw|winner up >=3 pts; --all-outcomes to disable).
.venv/bin/python experiments/viz/build_play_atlas.py --n 4000
# 2. seed the position memory (CPU, ~2.5 min at n=200k; once per field)
.venv/bin/python experiments/build_position_memory.py --n 200000
# 3. start the server (durable), then open http://localhost:8000
experiments/launch.sh play_server -- .venv/bin/python experiments/viz/play_server.py \
  --port 8000 --c-puct 1.0 --pw-c 1.5 --nodes 400
```
Server flags: `--c-puct` (exploration; 1.0 interactive, 1.5 = training default),
`--pw-c` (progressive widening; 0 = full-width), `--prior-tau`, `--memory <dir>`
(position-memory dir; '' disables), `--nodes` (engine-move budget).
UI: Engine toggle cycles Black → White → **Manual (both sides)** (analysis-board
mode); depth dropdown (100..2000) sets the Analyze budget; Analyze is start/stop
with checkpoint (a move or nav interrupts it and nav auto-resumes); **⟳ Rebuild
map** re-fits the t-SNE (iter/perp/exag/n) live; **🧠 Memory** lists the nearest
SEEN positions with outcomes/provenance (click → mini-board). Completed UI games
and every search line reaching a rules-certified terminal are appended to the
memory automatically (sources play_ui / mcts_sim).
To view a DIFFERENT field: rebuild atlas + memory with `--ckpt <field>.pt
[--phead <field>_phead.pt]` and start the server with the same ckpt (the memory
carries a ckpt tag and warns on mismatch).

---

## 6. Where things live
- Journal (chronological verdicts + decisions): `JOURNAL.md`
- Design: `ARCHITECTURE.md`, `PLANNER_PROBE_DESIGN.md`, `MATH_AUDIT.md`, `GLOSSARY.md`
- Training logs: `artifacts/experiments/*.log` (e.g. `qrl_iqe_leak.log`)
- Eval artifacts: `artifacts/experiments/*.json*`
- Shards (data): `data/shards/lichess_db_standard_rated_2019-01.prefix4gb`
- Auto-memory (rules/context, persists across sessions):
  `~/.claude/projects/-Users-kav-code-remote-github-catspace/memory/`
