# catspace — interpretable chess by steering toward the opponent's weaknesses

**Central hypothesis (Kaveh Shoorideh, 2026):** planning *toward your opponent's
weaknesses and away from your own* produces **interpretable** chess — because that
is how humans think: recognize a known trap shape, verify it's sound, steer there.
Interpretability is the product, not a side effect.

**Corollary:** humans play strong chess on 10–100 positions of search. A system
that captures the human planning structure should need a human-scale search
budget — orders of magnitude below engines. It will likely never beat Stockfish
or Leela, and that is the accepted trade: **we sell interpretability, not Elo.**
Every "why this move" has an inspectable, causally-audited answer: *which* trap
structure, with *what* probability, *how far* out, priced against *what*
concession.

This is a hypothesis project, and also an experiment in how research gets done:
an AI (Claude) builds and runs everything; Kaveh directs — the questions, the
reframes, the core calls. The whole process, including the dead ends, is in the
open: **[JOURNAL.md](JOURNAL.md)** is the lab notebook written as the work
happened — if you read one thing, read that.

## The engine — five components

The architecture of record is an **anchored joint-embedding predictive
architecture with a censored-hazard energy** (Kaveh's draft paper; being built
now). The component seams are the interpretability claim — the factorization
*is* the explanation.

| component | job | where |
|---|---|---|
| **Encoder** | positions → representation (64-token relational JEPA; frozen lc0 trunk as the incumbent) | `catspace/encoder/` |
| **Predictor** | what's coming, against whom: hazard/reach fields, atlas statistics, committor value, opponent move models, endgame ground truth (Syzygy, DTM) | `catspace/predictor/` |
| **Search** | verify and navigate: the MCTS core + goal-conditioned navigation | `catspace/search/` |
| **Planner** | choose what to steer toward; plans as chains of opponent-error structures | `catspace/planner/` |
| **Memory** | stored structures: goal banks, embedding retrieval, plan ledger — the checkpoint/atom bank lands here | `catspace/memory/` |

Support packages: `catspace/{data,train,style,harness,nn,engine,...}` (data
plumbing, training scaffold, player-style models, play harness, legacy). Old
import paths are kept as one-line aliases — pickled checkpoints and every
historical script still run.

## The repo, at a glance

| | |
|---|---|
| `catspace/` | the engine (five components above) |
| `tools/` | standalone probes & figure generators — every probe prints `VERDICT` lines and emits figures ([docs/PROBING.md](docs/PROBING.md)) |
| `scripts/` | canonical entry points (train / eval / play / launch) |
| `experiments/` | the full runnable lab notebook, chronologically honest |
| `docs/` | THESIS, COMPONENTS, TESTING, RUNBOOK, GLOSSARY, PROBING |
| `artifacts/` | run logs, checkpoints, figures (curated history) |
| `data/` | DVC-tracked datasets (pointers in git, bytes outside) |
| `ui/` | engine interface (UCI + plan-trace UI, scaffolding) |
| `JOURNAL.md` | the lab notebook — hypotheses, negative results, retractions |
| `MILESTONES.md` | the roadmap and dated design decisions |

## How research is conducted here

- **Best shot first, no A/B staging.** The complete target design is built in one
  go; attribution on failure is recovered by binary-searching components against
  known-good incumbents (the shelved M1–M5 stack).
- **Numbers are verdicts.** No number appears in docs or the journal unless a
  script printed it. Losses ship with executable invariant tests
  (`experiments/losses.py`); training runs carry collapse gates (effective rank).
- **Probing follows the field's best practice** — RankMe/LiDAR spectral health,
  frozen linear+kNN probes with group-aware splits, CKA, proper scoring rules for
  probability fields, quasimetric axioms, minimal-pair causal ablation; figures
  follow the FAIR / NVIDIA-robotics conventions (Clopper–Pearson intervals,
  green/red retrieval grids). See [docs/PROBING.md](docs/PROBING.md).
- **Interpretability is a measured endpoint:** faithfulness (remove the claimed
  structure → the decision must change), legibility (per-move traces with
  calibrated numbers), plan sparsity/stability, human-concept alignment.

## Findings so far (each number is a committed script verdict; stories in JOURNAL.md)

- **Per-player style is recoverable and exploitable — but only via retrieval**
  ("infer-then-condition": a recovered style vector overfits as a predictor,
  works as a retriever; +0.006–0.009 nats over the rating baseline).
- **You can estimate who you're playing from their moves alone** (Elo-MAE 142
  after 40 observed moves, vs 205 uninformed).
- **Blunder risk is measurable and asymmetric** (SF-refereed committor swings:
  ρ≈0.64 with realized crossings; weaker opponents cross 1.4–3× more in the same
  positions). **Where errors happen is position-driven; who errs is
  strength-driven** (crossing locations found at ~4.8× base rate, ranking nearly
  rating-invariant).
- **Reachability is a probability, not a distance** (the quasimetric field could
  not carry opponent-conditioning; the first-hit probability field can, with
  CI-separated style-lift). **Train==play matters more than data volume**
  (fine-tuning the field on our own steered games: +0.031 nats and the largest
  play-strength lever of the M5 campaign; 23× more passive data: nothing).
- **1.89M human trap checkpoints** mined from 2.39M engine-annotated lichess
  games (0.79/game) now ground the hazard energy.

![Engine vs human basins](docs/figures/engine_vs_human_basins.png)

*The object of study in one picture: the same embedding colored by game outcome.
Near-perfect play (left) separates into basins (purity 0.81); human play (right)
blurs to 0.53. That blur is fallible play crossing outcome barriers — exactly
what the planner steers with.*

## Running it

```bash
pip install -e .[nn]     # torch is the [nn] extra; lczerolens for the lc0 encoder
pytest                   # 268 tests
scripts/run_jepa_pretrain.sh   # the current build: corpus -> encoder pretraining
```

More in [docs/RUNBOOK.md](docs/RUNBOOK.md) and [scripts/README.md](scripts/README.md).

## Status (2026-07-30)

Pivoted to the anchored-JEPA architecture (Kaveh's draft): checkpoint corpus
mined (33.8M games scanned), encoder pretraining in flight. The prior stack
(M0–M5: metastability basins, style models, reach fields, chute planner, MCTS
probe — and its complete verdict ladder, 0.045→0.095 vs the 0.125 baseline) is
shelved intact as the known-good component library. History: `docs/archive/`,
`JOURNAL.md`.
