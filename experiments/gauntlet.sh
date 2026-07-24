#!/bin/bash
# experiments/gauntlet.sh -- fixed-TC engine gauntlet via fastchess (Kaveh: 'metrics =
# move speed + W/D/L; game timer; use the existing frameworks'). SPRT built into
# fastchess = the sequential-testing standard (fishtest-style, anytime-valid family).
# Usage: gauntlet.sh <fieldA.pt> <fieldB.pt> [tc, default 120+1] [rounds, default 40]
set -e
cd /Users/kav/code/remote/github/catspace
FC="$HOME/.claude/jobs/20b9956a/tmp/fastchess-mac-arm64/fastchess"
A="${1:?fieldA ckpt}"; B="${2:?fieldB ckpt}"
TC="${3:-120+1}"; ROUNDS="${4:-40}"
PY="$(pwd)/.venv/bin/python"; ENG="$(pwd)/experiments/uci_engine.py"
"$FC" \
  -engine name="A_$(basename $A .pt)" cmd="$PY" args="$ENG" option.Field="$A" \
  -engine name="B_$(basename $B .pt)" cmd="$PY" args="$ENG" option.Field="$B" \
  -each tc="$TC" timemargin=3000 \
  -rounds "$ROUNDS" -repeat -concurrency 1 \
  -sprt elo0=0 elo1=20 alpha=0.05 beta=0.05 \
  -pgnout artifacts/experiments/gauntlet_$(basename $A .pt)_vs_$(basename $B .pt).pgn \
  2>&1 | tee -a artifacts/experiments/gauntlet.log
