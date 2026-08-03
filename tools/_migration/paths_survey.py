"""Distinct data/artifact path literals, ranked by how many files reference them."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
inv = json.loads((ROOT / "tools" / "_migration" / "inventory.json").read_text())

per_file = inv["path_literals"]
prefix = Counter()
exact = Counter()
for f, lits in per_file.items():
    for lit in set(lits):
        exact[lit] += 1
        parts = lit.split("/")
        prefix["/".join(parts[:2])] += 1

print("=== top-2-level prefixes (n files) ===")
for k, n in prefix.most_common(40):
    print(f"{n:4d}  {k}")
print("\n=== exact literals referenced by >=3 files ===")
for k, n in exact.most_common():
    if n < 3:
        break
    print(f"{n:4d}  {k}")
print(f"\ndistinct literals: {len(exact)}")
