#!/bin/bash
# Iterate the outcome-pole cost GENTLY each round (t-SNE-style small push + pull),
# cumulatively, on diverse human data, until the VALIDATED forced-mate regions
# (mate_W / mate_B / draw) clearly separate. Each round resumes the previous
# round's checkpoint (accumulating the push) and re-measures separation on the
# persisted forced-mate set. Kaveh: "every round gently push some more ... keep
# iterating until the regions clearly separate."
cd /Users/kav/code/remote/github/catspace
INC=data/derived/lichess_fb_4gb_qm_plygap_only.pt
HUMAN=data/shards/lichess_db_standard_rated_2019-01.prefix4gb
SET=artifacts/experiments/forced_mate_set.json
REC=artifacts/experiments/separation_track.jsonl
WORK=data/derived/sep
mkdir -p "$WORK"
STEP=4000; BASE=90000
ROUNDS=${1:-10}
# gentle, FIXED push per round -- accumulation over rounds does the work (avoids the
# hard-push collapse); the pole is the planner goal (pole-as-goal via embed_zgoals).
OW=0.3; TAU=1.5; RW=0.2; RM=1.5

echo "=== baseline (incumbent) forced-mate separation ==="
.venv/bin/python experiments/viz/near_mate_regions.py --ckpt "$INC" --forced-set "$SET" \
  --label sep_r0_incumbent --record "$REC" --device auto 2>&1 | grep -E "separab|corr|VALUE|forced-mate"

prev="$INC"
for r in $(seq 1 "$ROUNDS"); do
  ck="$WORK/sep_r${r}.pt"
  cp "$prev" "$ck"
  total=$((BASE + r*STEP))
  echo "=== [$(date +%H:%M)] ROUND $r : +${STEP} steps (total $total), OW=$OW TAU=$TAU RW=$RW ==="
  .venv/bin/python experiments/train_lichess_fb.py --ckpt "$ck" --shards "$HUMAN" --steps "$total" \
    --quasimetric --ply-gap-weight 0.05 \
    --outcome-poles --outcome-weight "$OW" --pole-tau "$TAU" --repel-weight "$RW" --repel-margin "$RM" \
    --device auto || { echo "TRAIN FAILED r$r"; break; }
  echo "--- round $r forced-mate separation ---"
  .venv/bin/python experiments/viz/near_mate_regions.py --ckpt "$ck" --forced-set "$SET" \
    --label "sep_r${r}" --record "$REC" --device auto 2>&1 | grep -E "separab|corr|VALUE|forced-mate"
  prev="$ck"
done
echo "=== separation loop done; trajectory in $REC ==="
