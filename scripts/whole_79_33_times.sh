#!/bin/sh
# PLAN 79.33: usage  whole_79_33_times.sh <kind> <label>   three timed runs, one after the other (the machine quiet), uptime printed before each, OBS and WALL lines shown.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
for n in 1 2 3; do
  "$ROOT/scripts/whole_79_33_run.sh" time "$1" "$2" $n
  grep -h "^OBS 79.33\|OBS 79.33\|WALL\|load averages" "$ROOT/artifacts/experiments/whole_79_33/$2.$1.time$n.txt" | cut -c1-260
done
