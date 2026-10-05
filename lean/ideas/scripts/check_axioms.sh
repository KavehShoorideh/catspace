#!/bin/bash
# The axioms of Lean theorems, and a failure on anything beyond the standard three.
#   lean/ideas/scripts/check_axioms.sh Ideas.backRankMate_sound Ideas.Passer2.unstoppablePasser2_sound ...
# Prints one line per theorem: "AXIOMS <theorem> <axiom>,<axiom>,..." and exits 1 if any theorem depends on an axiom
# other than propext, Classical.choice, Quot.sound (sorryAx, native_decide's ofReduceBool, anything added).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.elan/bin:$PATH"
if [ "$#" -eq 0 ]; then echo "usage: $0 Theorem..." >&2; exit 2; fi
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
file="$tmpdir/Axioms.lean"
{ echo "import Ideas"; for t in "$@"; do echo "#print axioms $t"; done; } > "$file"
out=$(lake env lean "$file" 2>&1) || { echo "$out" >&2; exit 3; }
python3 - "$out" "$@" <<'PY'
import re, sys
out, theorems = sys.argv[1], sys.argv[2:]
standard = {"propext", "Classical.choice", "Quot.sound"}
text = " ".join(out.split())
bad = False
for t in theorems:
    m = re.search(r"'" + re.escape(t) + r"' depends on axioms: \[([^\]]*)\]", text)
    if m:
        ax = [a.strip() for a in m.group(1).split(",") if a.strip()]
    elif ("'" + t + "' does not depend on any axioms") in text:
        ax = []
    else:
        print(f"AXIOMS {t} ?MISSING"); bad = True; continue
    print(f"AXIOMS {t} {','.join(ax) if ax else '-'}")
    if set(ax) - standard:
        bad = True
sys.exit(1 if bad else 0)
PY
