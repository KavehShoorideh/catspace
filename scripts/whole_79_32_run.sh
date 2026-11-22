#!/bin/sh
# PLAN 79.32: usage  whole_79_32_run.sh dump <label>   writes artifacts/experiments/whole_79_32/<label>.{goals,meaning,answers}.dump and the OBS output
#                    whole_79_32_run.sh time <label> <n> runs layer4_goals.rs alone (one test thread), wall time and OBS lines to <label>.time<n>.txt
# Run from anywhere; it works in the worktree it lives in. A test build is made first and is not timed.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/artifacts/experiments/whole_79_32"
mkdir -p "$OUT"
cd "$ROOT/rust/engine"
mode=$1
label=$2
cargo test --release --offline -p cs_tactics --test layer4_goals --test layer0_meaning --no-run > "$OUT/$label.build.txt" 2>&1
if [ "$mode" = dump ]; then
  LAYER4_BASELINE="$OUT/$label.goals.dump" cargo test --release --offline -p cs_tactics --test layer4_goals -- --test-threads 1 --nocapture the_whole_reply > "$OUT/$label.goals.dump.out" 2>&1
  LAYER0_BASELINE="$OUT/$label.meaning.dump" cargo test --release --offline -p cs_tactics --test layer0_meaning -- --test-threads 1 --nocapture > "$OUT/$label.meaning.dump.out" 2>&1
  LAYER4_DUMP="$OUT/$label.answers.dump" cargo test --release --offline -p cs_tactics --test layer4_goals -- --test-threads 1 --nocapture the_cost_of_the_second > "$OUT/$label.answers.dump.out" 2>&1
else
  n=$3
  uptime > "$OUT/$label.time$n.txt"
  start=$(date +%s)
  cargo test --release --offline -p cs_tactics --test layer4_goals -- --test-threads 1 --nocapture >> "$OUT/$label.time$n.txt" 2>&1
  end=$(date +%s)
  echo "WALL $((end-start)) s" >> "$OUT/$label.time$n.txt"
fi
