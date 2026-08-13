#!/bin/bash
# train_cloud.sh "<TRAIN_ARGS>" -- launch a CUDA training job on the cheapest enabled vendor.
set -euo pipefail
source infra/.secrets.env
TAG=$(git rev-parse --short HEAD)
IMG=$REGISTRY/catspace-train:$TAG
docker build -f infra/Dockerfile.train -t "$IMG" --platform linux/amd64 .
docker push "$IMG"
sky launch -c catspace-train infra/sky/train.yaml --env IMAGE="$IMG" \
  --env DATA_URI="$DATA_URI" --env MODEL_URI="$MODEL_URI" \
  --env S3_ENDPOINT="${S3_ENDPOINT:-}" --env TRAIN_ARGS="${1:-}" --yes -d
echo "[train] job launched; sky logs catspace-train to follow"
