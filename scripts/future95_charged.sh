#!/bin/bash
# PLAN 78.95 PART FOUR, the charged matches: only for a player that was ahead uncharged (the chooser and the joined pick, each against the control, sets A and B).
set -u
PY=.venv/bin/python
MOD=catspace.incubator.analyst.discovery.future_play
for a in chooser joined; do
  for s in A B; do
    $PY -m $MOD match --a $a --b control --set $s --pairs 100 --tag full --charge || echo "FAILED $a control $s charged"
  done
done
$PY -m $MOD report --tag full
