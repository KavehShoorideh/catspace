"""LOST AND GAINED WINS BETWEEN TWO CENSUS RUNS, judged by Stockfish (2026-10-10, the regression after the dig-and-fix
round: frames' immediate tactics 676 -> 630). Usage: win_audit.py <old save tag> <new save tag> [sample]. For a sample
of the immediate wins lost and gained: Stockfish's (depth 12) loss of the move against its best; and every Stockfish
only-move tactic lost, with its old claim."""
import json
import os
import random
import sys
from multiprocessing import Pool

from catspace.io import paths
from catspace.incubator.analyst.pool_check import _sf


def main():
    old, new = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    L = lambda t: {(r["game"], r["ply"]): r for r in map(json.loads, open(paths.experiment(f"plan_check_c300_{t}.jsonl")))}
    A, B = L(old), L(new)
    lost = [k for k in A if A[k]["imm"] and not B[k]["imm"]]
    gained = [k for k in A if B[k]["imm"] and not A[k]["imm"]]
    sl = random.Random(0).sample(lost, min(n, len(lost))); sg = random.Random(0).sample(gained, min(n, len(gained)))
    with Pool(os.cpu_count()) as p:
        ll = p.map(_sf, [(A[k]["fen"], A[k]["imm"][1]) for k in sl], chunksize=4)
        lg = p.map(_sf, [(B[k]["fen"], B[k]["imm"][1]) for k in sg], chunksize=4)
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    print(f"OBS immediate wins lost {len(lost)}, gained {len(gained)}")
    for name, v in (("lost wins (the old move)", ll), ("gained wins (the new move)", lg)):
        print(f"OBS {name}: n {len(v)}, loss vs Stockfish's best median {med(v)} cp, within 50 cp {100 * sum(x <= 50 for x in v) / max(len(v), 1):.0f}%, "
              f"losing 150+ {100 * sum(x >= 150 for x in v) / max(len(v), 1):.0f}%")
    by = {}
    for k, x in zip(sl, ll):
        by.setdefault(A[k]["imm"][2], []).append(x)
    print("OBS lost wins by claimed gain (n, median loss):", {g: (len(v), med(v)) for g, v in sorted(by.items())})


if __name__ == "__main__":
    main()
