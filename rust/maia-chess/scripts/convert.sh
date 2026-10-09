#!/bin/sh
# Convert the official Maia weights (lc0 format, https://github.com/CSSLab/maia-chess) to ONNX for maia-chess.
# Usage: scripts/convert.sh <dir with maia-XXXX.pb.gz> <output dir>     (needs lc0 >= 0.30 on PATH)
set -e
IN=${1:?weights dir}; OUT=${2:?output dir}
mkdir -p "$OUT"
for w in "$IN"/maia-*.pb.gz; do
  b=$(basename "$w" .pb.gz)
  lc0 leela2onnx --input="$w" --output="$OUT/$b.onnx" >/dev/null && echo "$OUT/$b.onnx"
done
