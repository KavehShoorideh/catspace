#!/bin/sh
# PLAN 79.32 part C: three timed runs of layer4_goals.rs, one after the other, nothing else of ours running. Usage: whole_79_32_times.sh <label>
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
for n in 1 2 3; do
  "$ROOT/scripts/whole_79_32_run.sh" time "$1" "$n"
done
