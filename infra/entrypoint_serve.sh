#!/bin/bash
set -e
: "${MODEL_URI:?set MODEL_URI (s3://bucket/models)}"
mkdir -p /models
echo "[boot] pulling ${MODEL_URI}/${CHANNEL}/ ..."
aws s3 sync "${MODEL_URI}/${CHANNEL}/" /models/ ${S3_ENDPOINT:+--endpoint-url $S3_ENDPOINT}
CKPT=$(ls /models/*_latest.pt | head -1)
echo "[boot] serving $CKPT on :$PORT (device $DEVICE)"
exec python -m catspace.research.components.planner.approaches.quasimetric_nav.kittychess_server \
  --ckpt "$CKPT" --port "$PORT" --device "$DEVICE"
