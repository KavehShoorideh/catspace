# catspace — a metastability planner for exploiting fallible chess opponents

**A live research project.** Claude (the AI) does the building and experiments;
Kaveh directs the research — the questions, the reframes, and most of the core
ideas are his, iterated turn by turn. **[JOURNAL.md](JOURNAL.md)** is the running
lab notebook (newest entry last) and the most honest picture of how the work
actually happened — hypotheses, dead ends, reversals, and reasoning.

---

## The idea (current)

Chess outcomes {Win, Draw, Loss} are **metastable basins**: under optimal play
the barriers are infinite, under real play the rare crossings are **errors**. The
goal is to play a *fallible* opponent by **exploiting the information asymmetry
about where they err** — steer toward reachable positions where the opponent is
likely to make the outcome-flipping error and we are not ("pose problems").

The machinery (design-of-record: **[docs/REACHABILITY_FOUNDATIONS.md](docs/REACHABILITY_FOUNDATIONS.md)**)
is a **two-evaluator** stack: a **z-conditioned first-hit reachability field**
`P(reach g | s, z_self, z_opp)` = `⟨φ_r(s,z), ψ_r(g)⟩` with a **trained committor**
for navigation (the measure side), and legal-move **search** (df-pn/minimax,
tablebases) for forced objects (the existence side) — plus the **learned player
embedding `z`** + online `(Elo,z)` estimator and a **transition/crossing-risk
predictor** that says where *this* opponent slips. The system is **two-part**, exactly like Stockfish/Lc0:

1. **Full-board model** — the opponent-exploitation model (the thesis).
2. **Tablebase handover** — at **≤7 pieces the tablebase *is* the endgame**; the
   committor is grounded there and the goal region is the tablebase-won configs.
   (No learned endgame model — nets are measurably worse at conversion.)

> Note: earlier design commitments in the git history — a two-encoder `F(s)/B(g)`
> field, "no WDL/value head", `d=512` — have been **reversed** by the from-scratch
> rebuild (single-space `φ`, a trained committor, `d=64`). Trust the docs below,
> not the older `*.md` in history.

## Start here — the canonical docs

| file | what it's for |
|---|---|
| **[MILESTONES.md](MILESTONES.md)** | **the locked 30k-foot roadmap (M0–M6) + locked decisions. Read FIRST; do not deviate without a recorded plan change.** |
| **[JOURNAL.md](JOURNAL.md)** | the lab notebook — every experiment, result, reversal, with reasoning. **Read for the story & the reasoning.** |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | the technical spec. **Jump to the `⭐ CURRENT ARCHITECTURE (2026-07-26)` section** — input (lc0 112-plane), layers & sizes, objective functions, tablebase handover, opponent layer, and §10 the DVC/Ray/A-B **infrastructure**. |
| **[docs/METASTABILITY_PLAN.md](docs/METASTABILITY_PLAN.md)** | the strategy — the staged plan (S1 reliability map → field → opponent layer → eval) and the design rationale. |
| **[docs/TRAINING_STANDARDS.md](docs/TRAINING_STANDARDS.md)** | standing do's/don'ts for every training run. |
| **[docs/GLOSSARY.md](docs/GLOSSARY.md)** | plain-language definitions of every term/metric. |
| **[docs/COMPONENTS.md](docs/COMPONENTS.md)** · **[docs/RUNBOOK.md](docs/RUNBOOK.md)** | code map · ops runbook. |
| `experiments/losses.py` · `experiments/endgame_handover.py` | the **unit-tested primitives** (run them: `python experiments/losses.py`). No new loss enters a run without a passing test here. |
| **[docs/REACHABILITY_FOUNDATIONS.md](docs/REACHABILITY_FOUNDATIONS.md)** | **the current design-of-record**: whose-math table (graded), two-evaluator architecture, novelty ledger, AOEE disposition. Supersedes the archived brief/handoff (`docs/archive/`). |

## Component map

| Component | Where |
|---|---|
| Player model (M2b/c): style `z`, recovery, online `(Elo,z)` estimator | `catspace/style/` |
| Crossing-risk primitive (SF-refereed committor swing) | `catspace/transition.py` |
| Reachability head v1 (first-hit field) + dataset builders | `experiments/train_reach_head.py`, `experiments/build_reach_data.py`, `experiments/build_opp_positions.py` |
| Fields / embeddings (frozen lc0 trunk φ, IQE history) | `catspace/field.py`, trainers in `experiments/` |
| Goal bank + vector retrieval | `catspace/goal_bank.py`, `catspace/vectordb.py`, `catspace/memory/` |
| Canonical tested losses | `experiments/losses.py` |
| Training scaffold (MLflow / ladders / Tune / gates) | `catspace/train/scaffold.py` |
| Basins / committor / tablebases | `experiments/msm_basins.py`, `catspace/tb.py` |
| Planner / engine / arena / UCI | `catspace/planner/`, `catspace/engine/`, `catspace/arena.py`, `catspace/uci.py` |
| Superseded experiments (kept for provenance) | `experiments/archive/`, `docs/archive/` |


## Status (2026-07-26)

Field is architecturally complete and validated on ground truth: single-space IQE
+ multi-goal + repulsion + mate + WDL-hinge + distributional ending head +
committor over all W/D/L outcomes; lc0 112-plane real-history input; tablebase
handover primitive; SF reliability map. **Next:** the full-board opponent model —
player embedding `z`, transition predictor `T(s,z)`, cohort-regret/KL exploitation,
exploitation planner — on the **already-ingested** lichess shards.

## Data & infrastructure (already exists — see ARCHITECTURE.md §10)

- **Lichess** already ingested + DVC-tracked: `data/shards/lichess_db_standard_rated_2019-01.full/`
  (~10 GB, via `build_lichess_shards.py`), puzzles, and trained `data/derived/lichess_fb_*.pt`.
  Reuse — do not re-ingest.
- **DVC** versions the shards/derived datasets. **Ray** (`catspace/engine/orchestrator.py`)
  does probe memoization/coalescing. **A/B harnesses**: `ab_test.py` (anytime-valid),
  `move_ab.py`, `playout_ab.py`, `arena_real.py`, `gauntlet.sh` (fastchess SPRT),
  `experiment_report.py` (+ Stockfish-leakage audit). Tablebases under `data/syzygy/`.

## Running it

```bash
pip install -e .[nn]          # torch is the [nn] extra; lczerolens for the lc0 encoder
pytest                        # test suite
python experiments/losses.py            # unit-test the loss module
python experiments/endgame_handover.py  # unit-test the tablebase handover

# regenerate clean endgame basin data (both sides tb-optimal, all outcomes) + train the field
python experiments/gen_traj_lc0.py --games 3000 --eps 0.0 --out data/derived/traj_lc0_endgame.npz
python experiments/train_lc0_field.py --data data/derived/traj_lc0_endgame.npz --steps 18000
```

Every training run prints a `VERDICT` line (pair-order / eff_rank / committor-MAE /
ending-acc / W-D-L) copied into JOURNAL.md; `eff_rank(φ)` is the standing health gate.
