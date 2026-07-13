#!/bin/bash
# Round-13 evaluation runbook (2026-07-12, see JOURNAL.md): runs the full
# comparison sequence for the freshly-trained quasimetric+ply-gap+self-play
# checkpoint against the incumbent best, SEQUENTIALLY (each stage owns MPS).
# Kept in-repo because the PI loop will re-run this every generation with
# different ckpt args: round13_eval.sh <new_ckpt> <incumbent_ckpt> <tag>
set -e
cd "$(dirname "$0")/.."
NEW=${1:-data/derived/lichess_fb_4gb_qm_gen1.pt}
INCUMBENT=${2:-data/derived/lichess_fb_4gb_qm_wpov.pt}
TAG=${3:-round13}
SHARDS=data/shards/lichess_db_standard_rated_2019-01.prefix4gb

echo "=== [1/4] node-sweep 2000-stage re-run (clean, explicit --shards) on incumbent ==="
.venv/bin/python experiments/acpl_probe.py --ckpt "$INCUMBENT" --shards "$SHARDS" \
  --n 200 --sf-depth 8 --seed 5 --max-nodes 2000 --beam 4 2>&1 | tail -2

echo "=== [2/4] ACPL n=400 on NEW ckpt ($NEW) ==="
.venv/bin/python experiments/acpl_probe.py --ckpt "$NEW" --shards "$SHARDS" \
  --n 400 --sf-depth 8 --seed 5 --max-nodes 200 --beam 4 > "/tmp/acpl_n400_${TAG}_new.log" 2>&1
tail -2 "/tmp/acpl_n400_${TAG}_new.log"

echo "=== [3/4] ACPL n=400 on INCUMBENT (same positions; reuse if already logged) ==="
.venv/bin/python experiments/acpl_probe.py --ckpt "$INCUMBENT" --shards "$SHARDS" \
  --n 400 --sf-depth 8 --seed 5 --max-nodes 200 --beam 4 > "/tmp/acpl_n400_${TAG}_incumbent.log" 2>&1
tail -2 "/tmp/acpl_n400_${TAG}_incumbent.log"

echo "=== [4/4] KRRvKBP n=60 paired arena on NEW ckpt ==="
.venv/bin/python experiments/krrkbp_arena.py --ckpt "$NEW" \
  --fixed-set artifacts/experiments/krrkbp_fixed_set_n60.json \
  --opponent sf:skill=0 --max-plies 150 --seed 0 \
  --baseline-nodes 200 --baseline-beam 4 --plan-nodes 2000 --plan-beam 4 \
  --shallow-nodes 60 --shallow-beam 3 --no-early-stop 2>&1 | tail -3

echo "ROUND13_EVAL_DONE"
