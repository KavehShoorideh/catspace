#!/bin/bash
# PLAN 78.95 PART FOUR: the matches, one after the other (one heavy job at a time), each through run_games with the early stop.
# Uncharged: the weighted chooser and the joined pick, each against the control and against 78.82's player, sets A and B. The charged matches are run by hand, only for a player that is ahead uncharged.
# Usage: scripts/future95_matches.sh   (from the repo root; the environment of the worktree: PYTHONPATH, LCP_DATA)
set -u
PY=.venv/bin/python
MOD=catspace.incubator.analyst.discovery.future_play
for a in chooser joined; do
  for b in control reasons; do
    for s in A B; do
      $PY -m $MOD match --a $a --b $b --set $s --pairs 100 --tag full || echo "FAILED $a $b $s"
    done
  done
done
$PY -m $MOD report --tag full
