#!/bin/zsh
# keep the 2 newest step-ckpts per RUN, always keep step2000 (the young-lineage
# reference) and the milestone gko3a_step8000 (the promotion candidate / battery
# subject -- deleting it would strand every printed verdict).
cd /Users/kav/code/remote/github/catspace/artifacts/experiments
while true; do
  for run in gko3a gko3b; do
    ls -t ${run}_step*.pt 2>/dev/null | tail -n +3 \
      | grep -v -E "_step2000\.pt|gko3a_step8000\.pt" \
      | while read f; do rm -f "$f"; done
  done
  sleep 600
done
