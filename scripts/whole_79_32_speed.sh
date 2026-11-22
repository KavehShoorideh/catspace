#!/bin/sh
# PLAN 79.32 part C: the single-thread test of speed (rust/engine/tactics/tests/walk_speed.rs), three runs, nothing else of ours running. Usage: whole_79_32_speed.sh <label>
# Writes artifacts/experiments/whole_79_32/<label>.speed<n>.txt (uptime, then the OBS lines). The test binary is built before the first run and is not timed.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/artifacts/experiments/whole_79_32"
cd "$ROOT/rust/engine"
cargo test --release --offline -p cs_tactics --test walk_speed --no-run > "$OUT/$1.speedbuild.txt" 2>&1
for n in 1 2 3; do
  uptime > "$OUT/$1.speed$n.txt"
  cargo test --release --offline -p cs_tactics --test walk_speed -- --nocapture >> "$OUT/$1.speed$n.txt" 2>&1
done
