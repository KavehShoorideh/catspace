#!/bin/bash
# promote.sh -- copy the DEV channel's champion to PROD and redeploy prod.
# Promotion = data move + redeploy; the prod image only changes via deploy.sh prod.
set -euo pipefail
source infra/.secrets.env
EP=${S3_ENDPOINT:+--endpoint-url $S3_ENDPOINT}
echo "[promote] dev -> prod channel copy"
aws s3 sync "$MODEL_URI/dev/" "$MODEL_URI/prod/" $EP --delete
sky serve update -n catspace-prod infra/sky/serve.yaml --yes 2>/dev/null || \
  echo "[promote] prod service not up; run infra/deploy.sh prod"
echo "[promote] done -- prod now serves the dev champion"
