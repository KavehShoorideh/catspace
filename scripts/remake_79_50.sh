#!/bin/sh
# plan 79.50: make the records of 79.46 and 79.48 again after the one source change (not committed)
export LCP_DATA=/Users/kav/code/remote/github/catspace/data PYTHONHASHSEED=0 PYTHONPATH=/Users/kav/code/remote/github/catspace/.claude/worktrees/agent-a420062291c1de562
cd /Users/kav/code/remote/github/catspace/.claude/worktrees/agent-a420062291c1de562
export CARGO_TARGET_DIR=/private/tmp/claude-501/-Users-kav-code-remote-github-catspace/f8b8ac8d-0c97-4113-8826-c39d59eab2ef/scratchpad/target
PY=.venv/bin/python
for pair in chess:280 atomic:330 3check:300 kingofthehill:250 antichess:1500 chess960:300 horde:400; do
  g=${pair%%:*}; n=${pair##*:}
  $PY -m catspace.incubator.analyst.discovery.kernel_translated run $g --games-a-seed=$n
done
for pair in chess:57 atomic:164 3check:200 kingofthehill:100 antichess:280 chess960:57 horde:64; do
  g=${pair%%:*}; n=${pair##*:}
  $PY -m catspace.incubator.analyst.discovery.placed_translated run $g --games-a-seed=$n
done
echo ALL DONE
