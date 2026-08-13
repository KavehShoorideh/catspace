# catspace cloud infra

Vendor-abstracted serving + training. Three portable layers:
- **App = Docker** (`Dockerfile.serve`, `Dockerfile.train`) — runs identically on any vendor.
- **Orchestration = SkyPilot** (`sky/*.yaml`) — one YAML launches on AWS / GCP / Azure /
  Lambda / RunPod, `any_of` picks the cheapest enabled GPU, or pin a vendor per component.
- **Registry = any S3-compatible bucket** (AWS S3, Cloudflare R2, GCS interop, MinIO) —
  models under `models/{dev,prod}/`, data under `data/`.

## Dev / prod
Two independent services (`catspace-dev`, `catspace-prod`) reading two registry channels.
Promotion is a data operation: `infra/promote.sh` copies dev→prod and refreshes the prod
service. Images are immutable per-commit tags; prod images change only via `deploy.sh prod`.

## One-time setup (human steps)
1. `pip install "skypilot[aws,gcp,lambda]"` then `sky check` (add credentials per vendor).
2. Create an S3-compatible bucket; copy `infra/.secrets.env.example` → `infra/.secrets.env`.
3. A docker registry login (`docker login ghcr.io` or ECR/GCR).
4. Seed the registry: `aws s3 sync artifacts/experiments/ $MODEL_URI/dev/ --exclude "*"
   --include "reach_jqt4_latest*.pt" --include "reach_jqt4_jqt.pt" ...` (champion + sidecars)
   and `aws s3 sync data/derived/ $DATA_URI/` for training data.

## Commands
    infra/deploy.sh dev            # build+push image, serve the dev channel
    infra/deploy.sh prod           # same for prod
    infra/promote.sh               # dev champion -> prod
    infra/train_cloud.sh "--steps 20000 --jqt 1 ... --out artifacts/experiments/jqt5"
    sky serve status / sky logs    # observe

## Notes
- The server image serves the full analysis board (play, load FEN/PGN, concepts, plan panel)
  on :8420 — the SkyPilot service fronts it with a stable endpoint.
- `DEVICE=auto` resolves cuda→mps→cpu; all engine/trainer code is device-parameterized.
- Local dev loop is unchanged (Mac/MPS); the cloud is for demo serving + big CUDA runs.
