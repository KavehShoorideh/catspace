#!/bin/bash
set -e
: "${DATA_URI:?set DATA_URI}"; : "${MODEL_URI:?set MODEL_URI}"
mkdir -p data/derived artifacts/experiments
aws s3 sync "$DATA_URI/" data/derived/ ${S3_ENDPOINT:+--endpoint-url $S3_ENDPOINT}
python -m catspace.research.components.encoder.approaches.reach_probability.experiments.train_reach_vit \
  --device cuda ${TRAIN_ARGS}
aws s3 sync artifacts/experiments/ "${MODEL_URI}/dev/" ${S3_ENDPOINT:+--endpoint-url $S3_ENDPOINT} \
  --exclude "*" --include "*_latest*.pt" --include "*_jqt.pt" --include "*.jsonl"
echo "[train] artifacts pushed to ${MODEL_URI}/dev/"
