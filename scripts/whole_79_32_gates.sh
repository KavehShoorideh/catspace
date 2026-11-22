#!/bin/sh
# PLAN 79.32: the python gates of the plan, one file at a time (scripts/gate.py engine --only), the summary line of each. Usage: whole_79_32_gates.sh [PYTHONPATH-prefix-dir-of-a-private-module]
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONHASHSEED=0 LCP_DATA=/Users/kav/code/remote/github/catspace/data
if [ -n "$1" ]; then export PYTHONPATH="$1:$ROOT"; else export PYTHONPATH="$ROOT"; fi
for t in test_plan79_32 test_plan79_31 test_plan79_30 test_plan78 test_plan79_19 test_plan79_20 test_plan79_25 test_plan79_26 test_plan79_27 test_plan79_28 test_plan79_29; do
  if [ -e "${TMPDIR:-/tmp}/catspace_gate.lock" ]; then echo "lock present"; exit 1; fi
  .venv/bin/python scripts/gate.py engine --only "tests/$t.py" 2>&1 | grep -E "^  test_|FAILED|SKIPPED|skipped"
done
