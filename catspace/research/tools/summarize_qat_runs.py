"""One table for every qat run, read from the run logs. No hand-copied numbers.

Regret is the headline (win probability given up per move); top-1 is kept only because
older runs were judged on it before it was retired. Reference points, from
tools/diag_regret.py over 73,728 positions: random legal move 0.266, perfect WDL ranker
0.034 (median 0.000).
"""
import glob, os, re, sys

RUNS = [("qat1  qat, 3000 steps",        "qat1.log"),
        ("qat2  qat, 8000 steps",        "qat2.log"),
        ("qat3  level1 (no attention)",  "qat3.log"),
        ("qat4  qat + regret loss",      "qat4.log")]
d = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CLAUDE_JOB_DIR", "") + "/tmp"

print(f"  {'run':30s} {'step':>6s} {'balanced':>9s} {'regret':>8s} {'top1':>7s} {'concepts':>9s}")
print(f"  {'-'*30} {'-'*6} {'-'*9} {'-'*8} {'-'*7} {'-'*9}")
for name, fn in RUNS:
    p = os.path.join(d, fn)
    if not os.path.exists(p):
        print(f"  {name:30s} {'(log not found)':>34s}")
        continue
    last = None
    for ln in open(p):
        if ln.startswith("[qat] step"):
            last = ln
    if not last:
        print(f"  {name:30s} {'(no evals)':>34s}")
        continue
    g = lambda pat, d="--": (re.search(pat, last).group(1) if re.search(pat, last) else d)
    print(f"  {name:30s} {g(r'step +(\d+)/'):>6s} {g(r'BALANCED ([0-9.]+)'):>9s} "
          f"{g(r'REGRET ([0-9.]+)'):>8s} {g(r'top1 ([0-9.]+)'):>7s} "
          f"{g(r'concepts (\d+)'):>9s}")
print(f"\n  reference   random legal move  regret 0.266 | perfect WDL ranker  regret 0.034")
print(f"  note        regret was added mid-programme; qat1-qat3 show '--' and were")
print(f"              scored afterwards with --eval-only (qat2 0.183, qat3 0.136).")
