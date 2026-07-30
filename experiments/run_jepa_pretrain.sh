#!/usr/bin/env bash
# experiments/run_jepa_t1.sh -- anchored-JEPA T1, full corpus: pass-2 stream (transitions,
# syzygy-clamped boundaries, tokenized contexts) then joint three-loss encoder training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
echo "=== [1/2] build_jepa_corpus $(date) ==="
$PY -u experiments/build_jepa_corpus.py --out data/derived/checkpoints/jepa_t1_corpus.npz
echo "=== [2/2] train_jepa_t1 (30000 steps) $(date) ==="
$PY -u experiments/pretrain_jepa.py --data data/derived/checkpoints/jepa_t1_corpus.npz \
    --steps 30000 --out artifacts/experiments/jepa_t1
echo "=== DONE run_jepa_t1 $(date) ==="
