#!/bin/bash
# plan 78.96 part three: the matches, one at a time, uncharged: the prior against the control and against 78.82's player, sets A and B (the early stop and the load wait are in the module)
HERE=/Users/kav/code/remote/github/catspace/.claude/worktrees/agent-a8312bf7c6727b55f
for opp in control reasons; do
  for s in A B; do
    bash "$HERE/scripts/prior96_run.sh" prior_play match --a prior --b $opp --set $s --pairs 100 --tag full
  done
done
echo MATCHES-DONE
