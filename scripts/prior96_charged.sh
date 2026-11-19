#!/bin/bash
# plan 78.96 part three: the charged matches (the pick's reading time taken from the search's clock) of the prior against the control, sets A and B; played because the prior was ahead of the control uncharged
HERE=/Users/kav/code/remote/github/catspace/.claude/worktrees/agent-a8312bf7c6727b55f
for s in A B; do
  bash "$HERE/scripts/prior96_run.sh" prior_play match --a prior --b control --set $s --pairs 100 --tag full --charge
done
bash "$HERE/scripts/prior96_run.sh" prior_play report --tag full
echo CHARGED-DONE
