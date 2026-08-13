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
2. Create an S3-compatible bucket; write `infra/.secrets.env` (gitignored, NEVER committed)
   with: `REGISTRY=` (docker registry), `MODEL_URI=` / `DATA_URI=` (s3:// paths),
   `S3_ENDPOINT=` (empty for AWS; the R2/MinIO endpoint otherwise), and the bucket's
   `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`. Secrets live ONLY in this local file and
   in your shell -- the repo carries names, never values.
3. A docker registry login (`docker login ghcr.io` or ECR/GCR).
4. Seed the registry: `aws s3 sync artifacts/experiments/ $MODEL_URI/dev/ --exclude "*"
   --include "reach_jqt4_latest*.pt" --include "reach_jqt4_jqt.pt" ...` (champion + sidecars)
   and `aws s3 sync data/derived/ $DATA_URI/` for training data.

## Cost modes (low traffic)
For a-couple-of-players-a-week, do NOT run `sky serve` (always-on GPU ≈ $300+/mo).
Ranked cheapest-first:
1. **Modal serverless** (~$0/mo): $30/mo free credits, T4 $0.59/hr billed per-second,
   scale-to-zero web endpoints that AUTO-WAKE on the visitor's request (~10-30s cold
   start; --lite needs a human `sky start` + ~2-3 min). Wrapper: `infra/modal_serve.py`
   (`pip install modal && modal setup`, create the `catspace-registry` secret per the
   file header, `modal deploy infra/modal_serve.py`). The Docker image stays the
   source of truth -- Modal builds the same Dockerfile.serve.
2. **`deploy.sh dev --lite`** (~$3-10/mo): `sky launch` + 20-min idle autostop on the
   same YAML/image -- pay only session-hours (L4 ≈ $0.40-0.70/hr across vendors);
   wake a stopped box with `sky start catspace-dev` (~2-3 min).
3. **RunPod serverless** (~$3-6/mo): L4 $0.69/hr active-seconds only, no monthly fee.
`sky serve` is for when the demo gets real traffic.

## Commands
    infra/deploy.sh dev            # build+push image, serve the dev channel
    infra/deploy.sh dev --lite     # idle-autostop VM: pay per session, not per month
    infra/deploy.sh prod           # same for prod
    infra/promote.sh               # dev champion -> prod
    infra/train_cloud.sh "--steps 20000 --jqt 1 ... --out artifacts/experiments/jqt5"
    sky serve status / sky logs    # observe

## Notes
- The server image serves the full analysis board (play, load FEN/PGN, concepts, plan panel)
  on :8420 — the SkyPilot service fronts it with a stable endpoint.
- `DEVICE=auto` resolves cuda→mps→cpu; all engine/trainer code is device-parameterized.
- Local dev loop is unchanged (Mac/MPS); the cloud is for demo serving + big CUDA runs.
