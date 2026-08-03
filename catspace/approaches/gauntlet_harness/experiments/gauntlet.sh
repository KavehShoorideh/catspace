#!/bin/bash
# gauntlet.sh -- fixed-TC engine gauntlet via fastchess (Kaveh: 'metrics =
# move speed + W/D/L; game timer; use the existing frameworks'). SPRT built into
# fastchess = the sequential-testing standard (fishtest-style, anytime-valid family).
# Usage: gauntlet.sh <fieldA.pt> <fieldB.pt> [tc, default 120+1] [rounds, default 40]
#
# fastchess is a third-party binary, not vendored: set FASTCHESS to its path, else we look
# on PATH. (It used to be hardcoded into a scratch job dir that does not survive a reboot.)
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
FC="${FASTCHESS:-$(command -v fastchess || true)}"
if [ -z "$FC" ]; then
  echo "gauntlet.sh: fastchess not found -- set FASTCHESS=/path/to/fastchess" >&2
  exit 1
fi
A="${1:?fieldA ckpt}"; B="${2:?fieldB ckpt}"
TC="${3:-120+1}"; ROUNDS="${4:-40}"
PY="$ROOT/.venv/bin/python"
ENG="-m catspace.deployment.server.uci_engine"
OUT="$ROOT/artifacts/experiments"
mkdir -p "$OUT"
"$FC" \
  -engine name="A_$(basename $A .pt)" cmd="$PY" args="$ENG" option.Field="$A" \
  -engine name="B_$(basename $B .pt)" cmd="$PY" args="$ENG" option.Field="$B" \
  -each tc="$TC" timemargin=3000 \
  -rounds "$ROUNDS" -repeat -concurrency 1 \
  -sprt elo0=0 elo1=20 alpha=0.05 beta=0.05 \
  -pgnout file="$OUT/gauntlet_$(basename $A .pt)_vs_$(basename $B .pt).pgn" \
  2>&1 | tee -a "$OUT/gauntlet.log"
