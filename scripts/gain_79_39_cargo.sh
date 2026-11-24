#!/bin/sh
# PLAN 79.39 PART F: the Rust tests of the nine crates, written to artifacts/experiments/gain_79_39/cargo_all.txt, with the sum of the result lines printed last.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/artifacts/experiments/gain_79_39"
mkdir -p "$OUT"
cd "$ROOT/rust/engine"
cargo test --release --offline -p cs_base -p cs_rep -p cs_memory -p cs_game -p cs_value -p cs_tactics -p cs_decision -p cs_planning -p cs_learn > "$OUT/cargo_all.txt" 2>&1
echo "exit $?" >> "$OUT/cargo_all.txt"
grep "^test result" "$OUT/cargo_all.txt" | awk '{p+=$4; f+=$6; i+=$8} END {print "passed", p, "failed", f, "ignored", i}' >> "$OUT/cargo_all.txt"
tail -2 "$OUT/cargo_all.txt"
