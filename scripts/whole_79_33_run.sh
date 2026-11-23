#!/bin/sh
# PLAN 79.33: usage  whole_79_33_run.sh dump <kind> <label>        writes artifacts/experiments/whole_79_33/<label>.<kind>.dump (the answers and counts of the walk of that kind) and its .out
#                    whole_79_33_run.sh time <kind> <label> <n>    runs the tests of that kind alone, one test thread; wall time and OBS lines to <label>.<kind>.time<n>.txt, with uptime first
# kinds: values (cs_base tests/value_dump.rs), toward (cs_tactics tests/toward.rs and tests/tail_end.rs' neighbours: tests/walk_values.rs), tail, frame (cs_tactics tests/walk_values.rs)
# Run from anywhere; it works in the worktree it lives in. A test build is made first and is not timed. Never staged: the outputs are under artifacts/.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/artifacts/experiments/whole_79_33"
mkdir -p "$OUT"
cd "$ROOT/rust/engine"
mode=$1
kind=$2
label=$3
case "$kind" in
  values) PKG=cs_base; TESTS="--test value_dump"; ENVV=LAYER0_VALUES ;;
  toward) PKG=cs_tactics; TESTS="--test walk_values"; ENVV=WALK_TOWARD ;;
  tail) PKG=cs_tactics; TESTS="--test walk_values"; ENVV=WALK_TAIL ;;
  frame) PKG=cs_tactics; TESTS="--test walk_values"; ENVV=WALK_FRAME ;;
  *) echo "unknown kind $kind"; exit 2 ;;
esac
cargo test --release --offline -p $PKG $TESTS --no-run > "$OUT/$label.$kind.build.txt" 2>&1
if [ "$mode" = dump ]; then
  env $ENVV="$OUT/$label.$kind.dump" cargo test --release --offline -p $PKG $TESTS -- --test-threads 1 --nocapture > "$OUT/$label.$kind.dump.out" 2>&1
else
  n=$4
  uptime > "$OUT/$label.$kind.time$n.txt"
  start=$(date +%s)
  cargo test --release --offline -p $PKG $TESTS -- --test-threads 1 --nocapture >> "$OUT/$label.$kind.time$n.txt" 2>&1
  end=$(date +%s)
  echo "WALL $((end-start)) s" >> "$OUT/$label.$kind.time$n.txt"
fi
