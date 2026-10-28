"""Read-only look at the guard snapshots: files per dir, DECISION lemma tally."""
import collections
import glob
import os
import re
import sys

ROOT = sys.argv[1]
for d in ("ten_t0", "ten_t2", "wide_t0", "wide_t2"):
    p = os.path.join(ROOT, d)
    traces = [f for f in glob.glob(os.path.join(p, "*.txt")) if f.count(".") == 1]
    words = glob.glob(os.path.join(p, "*.words.txt"))
    dec = 0
    tally = collections.Counter()
    for f in traces:
        for line in open(f):
            if line.startswith("DECISION"):
                dec += 1
                m = re.match(r"DECISION (\S+) (-?\d+) (.*?)( proved=\S+)?$", line.strip())
                tally[m.group(3) if m else line.strip()] += 1
    print(d, "traces", len(traces), "words", len(words), "decision lines", dec)
    if len(sys.argv) > 2:
        for k, v in tally.most_common(60):
            print("   ", v, k[:160])
