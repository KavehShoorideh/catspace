#!/usr/bin/env bash
# experiments/run_table_v4.sh -- overnight region-table v4 rebuild (Kaveh 2026-07-29:
# "do the data work to create the table"; M5 four-way verdict: signal-bound).
# Chain: gen 200k games (~1.6M positions) -> SF-label depth 12 (~5.6h, 10 workers)
#        -> maia2 aug feats (poison-guarded) -> region table v4 (1024 regions,
#        k-means fit on 150k subsample, assignments on all) -> M5 tiered+opp read
#        with the fat table (the before/after comparison vs 0.070).
# Usage: experiments/launch.sh table_v4 -- experiments/run_table_v4.sh [ngames]
set -euo pipefail
# repo root by marker walk-up, mirroring catspace/io/paths.py -- the old
# `cd "$(dirname "$0")/.."` assumed this script sat one level under the root.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -f pyproject.toml ] && [ "$PWD" != / ]; do cd ..; done; pwd)"
cd "$ROOT"
NGAMES="${1:-200000}"
PY=.venv/bin/python

echo "=== [1/5] gen_transition_data ($NGAMES games) $(date) ==="
$PY -u experiments/gen_transition_data.py --games "$NGAMES" \
    --out data/derived/transition_data_v4.npz

echo "=== [2/5] sf_label_transitions (depth 12) $(date) ==="
$PY -u experiments/sf_label_transitions.py --data data/derived/transition_data_v4.npz \
    --depth 12 --out data/derived/transition_data_v4_labeled.npz

echo "=== [3/5] build_m2a_aug_feats $(date) ==="
$PY -u experiments/build_m2a_aug_feats.py \
    --labeled data/derived/transition_data_v4_labeled.npz \
    --out data/derived/m2a_aug_feats_v4.npz

echo "=== [4/5] m3_build_region_table -> region_table_v4 $(date) ==="
$PY -u experiments/m3_build_region_table.py \
    --labeled data/derived/transition_data_v4_labeled.npz \
    --reach data/derived/reach/reach_v3.npz \
    --aug-feats data/derived/m2a_aug_feats_v4.npz \
    --aug-from data/derived/reach/reach_v3.npz \
    --regions 1024 --kmeans-sample 150000 \
    --out data/derived/reach/region_table_v4.npz

echo "=== [5/5] M5 read with v4 table (100 games, 200n, tiered+opp) $(date) ==="
$PY -u experiments/m5_mcts_probe.py --games 100 --nodes 200 \
    --table data/derived/reach/region_table_v4.npz \
    --tag m5e_v4table --save-pgn artifacts/experiments/m5_read100_v4table.pgn

echo "=== DONE run_table_v4 $(date) ==="
