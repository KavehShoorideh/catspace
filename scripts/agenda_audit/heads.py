"""Read-only: frequency of line heads across the guard snapshots (to craft trace signatures)."""
import collections
import glob
import os
import re
import sys

ROOT = sys.argv[1]
heads = collections.Counter()
cases = collections.defaultdict(set)
for d in ("ten_t0", "ten_t2", "wide_t0", "wide_t2"):
    for f in glob.glob(os.path.join(ROOT, d, "*.txt")):
        base = os.path.basename(f)
        if base.count(".") != 1:
            continue
        for line in open(f, errors="replace"):
            s = line.strip()
            s = re.sub(r"^\[[a-z]+ \d+\] ", "", s)
            s = re.sub(r"[A-Z]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?|O-O(-O)?[+#]?", "M", s)
            s = re.sub(r"\d+", "N", s)
            h = " ".join(s.split()[:4])
            heads[h] += 1
            cases[h].add(base)
for h, n in heads.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 150):
    print(n, len(cases[h]), h[:120])
