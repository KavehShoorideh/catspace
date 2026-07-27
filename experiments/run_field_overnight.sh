#!/usr/bin/env bash
# run_field_overnight.sh -- autonomous full-board field pipeline (Kaveh: "train a proper field, test it").
# Runs AFTER the full-month game-records build completes. Chains: DVC-track records -> Stage C field
# data (parallel) -> DVC-track it -> train the field (scaffold: ladders/gates/MLflow) -> test it.
# Each step logs to artifacts/experiments/. Idempotent-ish: skips DVC add if the .dvc already exists.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

REC=data/records/lichess_2019-01
FIELD=data/derived/field_fullgame_v1.npz
OUT=artifacts/experiments/field_fullgame_v1
LOG=artifacts/experiments

echo "=== [1/5] DVC-track records ($REC) ==="
[ -f "$REC.dvc" ] || dvc add "$REC" 2>&1 | tail -3

echo "=== [2/5] Stage C: records -> field data (100k games, parallel) ==="
python experiments/gen_field_data_fullgame.py --records "$REC" --out "$FIELD" \
  --games 100000 --stride 6 --per-game 8 --tail 4 2>&1 | tee "$LOG/genC_v1.log" | tail -6

echo "=== [3/5] DVC-track field data ($FIELD) ==="
[ -f "$FIELD.dvc" ] || dvc add "$FIELD" 2>&1 | tail -3

echo "=== [4/5] Train the full-board field (16k steps, scaffold) ==="
python experiments/train_field_fullgame.py --data "$FIELD" --steps 16000 \
  --eval-every 1000 --ckpt-every 2000 --w-cat 1.0 --out "$OUT" 2>&1 | tee "$LOG/train_v1.log" | grep -vE "Trial|tune|ray::|INFO|Deprecat" | tail -30

echo "=== [5/5] Test the field (calibration on held-out val) ==="
python experiments/test_field_fullgame.py --ckpt "${OUT}_latest.pt" --data "$FIELD" 2>&1 | tee "$LOG/test_v1.log" | tail -30

echo "=== FIELD PIPELINE DONE ==="
