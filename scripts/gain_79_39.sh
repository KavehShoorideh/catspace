#!/bin/sh
# PLAN 79.39 PART E: the gain, three runs, one thread, load printed with every run. usage  gain_79_39.sh <label> <private module folder>
#   (1) calls a decision of mover, turn, outcome, moves by the counting build (scripts/rules_share_print.py run, 40 s a run, the 60 positions of plan 79.37, 0.2 s a decision);
#   (2) the time of rust/engine/tactics/tests/walk_speed.rs (the OBS lines it prints and the wall);
#   (3) the rate of the shared search, nodes a second on the fixture (rust/engine/tactics/tests/layer4.rs, the rate test), which prints its own median of three.
# Outputs: artifacts/experiments/gain_79_39/<label>.*.txt. A build is made first and is not timed. The machine is shared: the load is printed before and after each run.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/artifacts/experiments/gain_79_39"
mkdir -p "$OUT"
label=$1
MOD=$2
cd "$ROOT/rust/engine"
cargo test --release --offline -p cs_tactics --test walk_speed --test layer4 --no-run > "$OUT/$label.build.txt" 2>&1
for n in 1 2 3; do
  uptime > "$OUT/$label.walk$n.txt"
  start=$(date +%s)
  cargo test --release --offline -p cs_tactics --test walk_speed -- --test-threads 1 --nocapture >> "$OUT/$label.walk$n.txt" 2>&1
  end=$(date +%s)
  echo "WALL $((end-start)) s" >> "$OUT/$label.walk$n.txt"
  uptime >> "$OUT/$label.walk$n.txt"
done
uptime > "$OUT/$label.rate.txt"
cargo test --release --offline -p cs_tactics --test layer4 -- --test-threads 1 --nocapture the_rate >> "$OUT/$label.rate.txt" 2>&1
uptime >> "$OUT/$label.rate.txt"
cd "$ROOT"
for n in 1 2 3; do
  uptime > "$OUT/$label.calls$n.txt"
  env PYTHONHASHSEED=0 PYTHONPATH=$ROOT:$MOD LCP_DATA=/Users/kav/code/remote/github/catspace/data .venv/bin/python scripts/rules_share_print.py run --tag t7939 --name $label$n --seconds 40 >> "$OUT/$label.calls$n.txt" 2>&1
  uptime >> "$OUT/$label.calls$n.txt"
done
echo DONE > "$OUT/$label.done"
