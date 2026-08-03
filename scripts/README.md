# scripts/ — the canonical entry points

Symlinks to the runnable scripts a newcomer actually needs. After the 2026-08-03
restructure each one lives with the approach that owns it (see `repo_structure.md`), which
is good for provenance and bad for typing; these symlinks are the short path back.

| script | owner | what it does |
|---|---|---|
| `mine_checkpoints.py` | `memory:checkpoint_trap_bank` | mine trap checkpoints from [%eval] lichess games (data stage 1) |
| `build_jepa_corpus.py` | `encoder:jepa_tokenizer` | transitions + Syzygy-clamped boundaries + tokenized contexts (stage 2) |
| `pretrain_jepa.py` | `encoder:jepa_tokenizer` | encoder pretraining (three losses, one clamp) |
| `run_jepa_pretrain.sh` | `encoder:jepa_tokenizer` | the full corpus→training chain, detached-run ready |
| `m5_mcts_probe.py` | `search:puct_mcts` | play the current engine stack vs Maia (VERDICT instrumented) |
| `eval_agentive_lift.py` | `planner:reach_field` | field-calibration verdict on our own games |
| `play_vs_maia.py` | `gauntlet_harness` | the shallow committor baseline (the 0.125 reference) |
| `play_traced.py` | `gauntlet_harness` | the traced engine: recognize → verify → commit, trace as the product |
| `launch.sh <name> -- <cmd>` | `tools/training_infra` | durable background launcher (log + pid + caffeinate) |

Every run prints `VERDICT` lines; no number is quoted in docs unless a script printed it.
Probing and figure tools live in `catspace/research/tools/{probes,figures}/` (see
`catspace/research/docs/PROBING.md`).

Installed console entry points, which need no path at all:

| command | what it does |
|---|---|
| `catspace-uci` | the UCI engine, for cutechess/fastchess |
| `catspace-server` | the HTTP co-analyst (`--port 8777`) |
| `catspace-banksync` | push the online mate/loss/draw banks into Qdrant |

## Stage vocabulary (field conventions, not the draft's T-labels)

pretraining (encoder) -> transcoder training (sparse atoms on frozen
activations) -> bank construction (codebook + index) -> retrieval training
(projections + predictor) -> expert iteration (planner). The draft paper's
T1–T5 map to these in order; code and docs use the functional names.
