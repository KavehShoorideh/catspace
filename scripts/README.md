# scripts/ — the canonical entry points

Symlinks into `experiments/` (the full lab notebook of runnable scripts, kept
chronologically honest). These are the ones a newcomer needs:

| script | what it does |
|---|---|
| `mine_checkpoints.py` | mine trap checkpoints from [%eval] lichess games (data stage 1) |
| `build_jepa_corpus.py` | transitions + Syzygy-clamped boundaries + tokenized contexts (stage 2) |
| `train_jepa_t1.py` | train the anchored-JEPA encoder (three losses, one clamp) |
| `run_jepa_t1.sh` | the full corpus→training chain, detached-run ready |
| `m5_mcts_probe.py` | play the current engine stack vs Maia (VERDICT instrumented) |
| `eval_agentive_lift.py` | field-calibration verdict on our own games |
| `play_vs_maia.py` | the shallow committor baseline (the 0.125 reference) |
| `launch.sh <name> -- <cmd>` | durable background launcher (log + pid + caffeinate) |

Every run prints `VERDICT` lines; no number is quoted in docs unless a script
printed it. Probing/figure tools live in `tools/` (see `docs/PROBING.md`).
