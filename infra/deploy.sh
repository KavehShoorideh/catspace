#!/bin/bash
# deploy.sh <dev|prod> -- build, push, and (re)deploy the engine service for an environment.
set -euo pipefail
ENV=${1:?usage: deploy.sh <dev|prod>}
source infra/.secrets.env    # REGISTRY, MODEL_URI, S3_ENDPOINT, cloud creds (gitignored)
TAG=$(git rev-parse --short HEAD)
IMG=$REGISTRY/catspace-serve:$ENV-$TAG
docker build -f infra/Dockerfile.serve -t "$IMG" --platform linux/amd64 .
docker push "$IMG"
sky serve up -n "catspace-$ENV" infra/sky/serve.yaml \
  --env IMAGE="$IMG" --env CHANNEL="$ENV" --env MODEL_URI="$MODEL_URI" \
  --env S3_ENDPOINT="${S3_ENDPOINT:-}" --yes
echo "[deploy] catspace-$ENV -> $IMG (channel $ENV)"
