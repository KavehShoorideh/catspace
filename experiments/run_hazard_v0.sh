#!/usr/bin/env bash
# experiments/run_hazard_v0.sh -- anchored-JEPA stage 1b+2: embed the mined checkpoint
# corpus (frozen trunk, v4 region bank) then train the censored hazard head v0.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
echo "=== [1/2] embed_checkpoints $(date) ==="
$PY -u experiments/embed_checkpoints.py \
    --data data/derived/checkpoints/checkpoints_v1_full.npz \
    --out data/derived/checkpoints/checkpoints_v1_emb.npz
echo "=== [2/2] train_hazard_head (8000 steps) $(date) ==="
$PY -u experiments/train_hazard_head.py \
    --data data/derived/checkpoints/checkpoints_v1_emb.npz \
    --steps 8000 --out artifacts/experiments/hazard_v0
echo "=== DONE run_hazard_v0 $(date) ==="
