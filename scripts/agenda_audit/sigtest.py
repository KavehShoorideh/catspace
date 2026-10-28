"""Read-only: for each regex given, the cases (of the 109) whose trace or word file matches it, at tier 0 and tier 2, and one example line."""
import glob
import os
import re
import sys

ROOT = sys.argv[1]
pats = sys.argv[2:]


def files(d):
    out = {}
    for f in glob.glob(os.path.join(ROOT, d, "*.txt")):
        b = os.path.basename(f)
        cid = b.split(".")[0]
        out.setdefault(cid, []).append(f)
    return out


for p in pats:
    rx = re.compile(p)
    res = []
    ex = None
    for tier in ("t0", "t2"):
        n = 0
        for d in (f"ten_{tier}", f"wide_{tier}"):
            for cid, fs in files(d).items():
                hit = False
                for f in fs:
                    if f.endswith((".reads.txt", ".leaf.txt", ".why.txt")):
                        continue
                    for line in open(f, errors="replace"):
                        if rx.search(line):
                            hit = True
                            ex = ex or line.strip()[:150]
                            break
                    if hit:
                        break
                n += hit
        res.append(n)
    print(f"{res[0]:>3} {res[1]:>3}  {p!r}  e.g. {ex}")
