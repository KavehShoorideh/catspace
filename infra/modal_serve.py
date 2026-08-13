# modal_serve.py -- serverless GPU serving on Modal (the ~$0/mo, auto-wake path for
# a-couple-of-players-a-week; see infra/README.md "Cost modes").
#
# The Docker image stays the source of truth: this wrapper builds infra/Dockerfile.serve
# and runs the same entrypoint; only orchestration is Modal-specific.
#
# One-time setup (after `pip install modal` + `modal setup`):
#   modal secret create catspace-registry \
#       MODEL_URI=s3://<bucket>/models S3_ENDPOINT= \
#       AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
#   modal deploy infra/modal_serve.py          # prints the stable https URL
#
# Behavior: scale-to-zero when idle; a visitor's first request boots a T4 container
# (~10-30s: boot + model pull/load), then moves are full speed. CHANNEL picks the
# dev|prod registry channel. NOTE: game state lives in server memory -- max_containers=1
# keeps one live game consistent, and scaledown_window=15min means a game abandoned for
# 15+ minutes resets on return.

import os
import subprocess

import modal

CHANNEL = os.environ.get("CATSPACE_CHANNEL", "dev")
PORT = 8420

app = modal.App(f"catspace-{CHANNEL}")

image = modal.Image.from_dockerfile(
    "infra/Dockerfile.serve",
    context_dir=".",
    build_args={"TORCH": "cu121"},   # if your modal version lacks build_args: drop it and
)                                    # serve CPU-torch, or bake TORCH=cu121 as the default


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("catspace-registry")],
    max_containers=1,          # one game server; in-memory game state stays consistent
    scaledown_window=900,      # idle 15 min -> scale to zero (billing stops)
    timeout=3600,
)
@modal.web_server(port=PORT, startup_timeout=300)
def serve():
    env = dict(os.environ, CHANNEL=CHANNEL, PORT=str(PORT), DEVICE="auto")
    subprocess.Popen(["/entrypoint.sh"], env=env)
