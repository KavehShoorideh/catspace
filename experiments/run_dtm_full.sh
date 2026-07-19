#!/usr/bin/env bash
# experiments/run_dtm_full.sh -- the FULL sharp+aligned field run (Kaveh
# 2026-07-19: "do a full training run instead of fine tuning"). This is the
# proven qrl_iqe_sn_full recipe (QRL spread: IQE unit-step metric that spreads
# without collapsing) PLUS the new DTM hinge (--dtm-hinge: aligns d(F(s),MATE_W)
# to true tablebase distance-to-mate, giving the metric a mate-pointing
# gradient). QRL = sharpness, DTM hinge = alignment: the sharp+aligned field
# neither current field had.
#
# Waits for the DTM data (experiments/gen_dtm_data.py -> dtm_endgame.npz) to be
# ready before starting, so it can be launched durably before generation
# finishes. Launch via:
#   experiments/launch.sh qrl_dtm_full -- bash experiments/run_dtm_full.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DTM=data/derived/dtm_endgame.npz
GENLOG=artifacts/experiments/dtm_gen.log

# 1) wait for DTM generation to complete (VERDICT line) + file present
echo "[run_dtm_full] waiting for DTM data ($DTM) ..."
while ! grep -q "VERDICT DTM_DATA" "$GENLOG" 2>/dev/null; do sleep 30; done
[ -f "$DTM" ] || { echo "[run_dtm_full] FATAL: $DTM missing after gen VERDICT"; exit 1; }

# 2) sanity-check the dataset size before committing to a multi-hour run
N=$(.venv/bin/python -c "import numpy as np; print(len(np.load('$DTM')['dtm']))")
echo "[run_dtm_full] DTM data ready: $N positions"
[ "$N" -ge 5000 ] || { echo "[run_dtm_full] FATAL: only $N DTM positions (<5000)"; exit 1; }

# 3) ensure MPS is free (no other training job) -- the smoke should be long done
while pgrep -f "train_lichess_fb.*qrl_dtm_smoke" >/dev/null 2>&1; do
  echo "[run_dtm_full] smoke still on MPS; waiting ..."; sleep 30
done

# 4) launch the full run (foreground here; the caller's launch.sh detaches us)
echo "[run_dtm_full] starting full training @ $(date)"
exec .venv/bin/python -u experiments/train_lichess_fb.py \
  --shards data/shards/lichess_db_standard_rated_2019-01.prefix4gb \
  --ckpt data/derived/sep/qrl_dtm_full.pt --steps 40000 --ckpt-every 10000 --fresh \
  --quasimetric --iqe --iqe-components 32 --iqe-embed-scale 2.0 --iqe-leak-beta 10.0 \
  --freeze-iqe-scale --spectral-norm --omega-free-field \
  --qrl-objective --qrl-push-offset 15 --qrl-goal-pool 8192 --qrl-two-sided \
  --qrl-use-pid --qrl-pid-kp 0.5 --qrl-pid-ki 0.01 --qrl-pid-kd 0.25 --qrl-pid-eclip 3.0 \
  --qrl-lambda-max 0.0 --qrl-lambda-lr 0.01 --qrl-var-weight 1.0 --qrl-var-target 1.0 \
  --qrl-unreach-weight 8.0 --qrl-unreach-floor 30 --qrl-halt-on-collapse \
  --committor-base --phead-weight 0.1 --selfplay-frac 0.3 \
  --lr 2e-4 --batch 128 --seed 0 \
  --d 512 --dh 512 --channels 128 --blocks 10 --enc-out 512 \
  --dtm-hinge "$DTM" --dtm-weight 0.3 --dtm-batch 128
