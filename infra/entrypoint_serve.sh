#!/bin/bash
set -e
mkdir -p /models
if [ -n "${MODEL_URI:-}" ]; then                      # cloud: pull the channel from the registry
  echo "[boot] pulling ${MODEL_URI}/${CHANNEL}/ ..."
  aws s3 sync "${MODEL_URI}/${CHANNEL}/" /models/ ${S3_ENDPOINT:+--endpoint-url $S3_ENDPOINT}
else                                                  # local (docker compose): /models is a mount
  echo "[boot] MODEL_URI unset -> serving from the /models mount"
fi
CKPT=$(ls /models/*_latest.pt | head -1)
echo "[boot] serving $CKPT on :$PORT (device $DEVICE)"
exec python -m catspace.research.components.planner.approaches.quasimetric_nav.kittychess_server \
  --ckpt "$CKPT" --port "$PORT" --device "$DEVICE"
