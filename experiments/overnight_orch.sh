#!/bin/bash
cd /Users/kav/code/remote/github/catspace
Q=artifacts/experiments/overnight_queue.tsv
R=artifacts/experiments/overnight_results.jsonl
INC=data/derived/lichess_fb_4gb_qm_plygap_only.pt
SHARDS=data/shards/lichess_db_standard_rated_2019-01.prefix4gb
MIX=data/selfplay/krrkbp_mix
CAP=30600   # ~8.5h wall-clock safety cap
while [ $SECONDS -lt $CAP ]; do
  progressed=0
  while IFS=$'\t' read -r name flags; do
    [ -z "$name" ] && continue
    [ "$name" = "STOP" ] && exit 0
    grep -q "\"$name\"" "$R" 2>/dev/null && continue      # already evaluated -> skip
    ckpt=data/derived/sfsf/${name}.pt
    cp "$INC" "$ckpt"
    echo "=== [$(date +%H:%M)] TRAIN $name : $flags ==="
    .venv/bin/python experiments/train_lichess_fb.py --ckpt "$ckpt" --shards "$SHARDS" \
      --steps 98000 --quasimetric --ply-gap-weight 0.05 \
      --selfplay-shards "$MIX" --selfplay-frac 0.7 $flags --device auto || echo "TRAIN FAILED $name"
    echo "=== [$(date +%H:%M)] EVAL $name ==="
    .venv/bin/python experiments/eval_variant.py --ckpt "$ckpt" --label "$name" --note "$flags" --device auto || echo "EVAL FAILED $name"
    progressed=1
  done < "$Q"
  [ $progressed -eq 0 ] && sleep 180                        # caught up; wait for appended variants
done
