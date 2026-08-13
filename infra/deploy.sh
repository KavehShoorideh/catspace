#!/bin/bash
# deploy.sh <dev|prod> [--lite] -- build, push, and (re)deploy the engine service.
#   default: sky serve (always-on endpoint; ~$300+/mo on a GPU -- for real traffic)
#   --lite:  sky launch + idle-autostop (VM stops itself after 20 idle min; you pay only
#            session-hours -- the right mode for a-couple-of-players-a-week)
set -euo pipefail
ENV=${1:?usage: deploy.sh <dev|prod> [--lite]}
MODE=${2:-}
source infra/.secrets.env    # REGISTRY, MODEL_URI, S3_ENDPOINT, cloud creds (gitignored)
TAG=$(git rev-parse --short HEAD)
IMG=$REGISTRY/catspace-serve:$ENV-$TAG
docker build -f infra/Dockerfile.serve -t "$IMG" --platform linux/amd64 .
docker push "$IMG"
if [ "$MODE" = "--lite" ]; then
  sky launch -c "catspace-$ENV" infra/sky/serve.yaml \
    --env IMAGE="$IMG" --env CHANNEL="$ENV" --env MODEL_URI="$MODEL_URI" \
    --env S3_ENDPOINT="${S3_ENDPOINT:-}" \
    --idle-minutes-to-autostop 20 --yes
  echo "[deploy] catspace-$ENV (lite): autostops after 20 idle min;"
  echo "         wake with: sky start catspace-$ENV   url: sky status --endpoints catspace-$ENV"
else
  sky serve up -n "catspace-$ENV" infra/sky/serve.yaml \
    --env IMAGE="$IMG" --env CHANNEL="$ENV" --env MODEL_URI="$MODEL_URI" \
    --env S3_ENDPOINT="${S3_ENDPOINT:-}" --yes
fi
echo "[deploy] catspace-$ENV -> $IMG (channel $ENV)"
