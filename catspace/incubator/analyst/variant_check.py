"""PLANNER VARIANTS ON THE STOCKFISH-TACTIC POSITIONS (2026-10-11): the census positions with a Stockfish only-move
tactic (tactic_reach.is_tactic), planned under switch settings given as env-style flags; per setting: immediate wins,
Stockfish's move found, time. Usage: variant_check.py "composed=1,strict=1" "composed=0,strict=1" ..."""
import json
import os
import sys
import time
from multiprocessing import Pool

from catspace.io import paths


def _one(args):
    fen, flags = args
    os.environ["RAYON_NUM_THREADS"] = "1"
    import analyst_core
    analyst_core.set_composed(flags.get("composed", 1) == 1)
    analyst_core.set_strict_stop(flags.get("strict", 1) == 1)
    t = time.time()
    r = analyst_core.plan_trace(fen)
    return r[5], time.time() - t


def main():
    from .tactic_census_measure import load
    from .tactic_reach import is_tactic
    rows = load("c300")
    prev = {(r["game"], r["ply"]): r["fen"] for r in rows}
    labels = {(r["game"], r["ply"]): r for r in map(json.loads, open(paths.experiment("tactic_labels_c300_d12_pv.jsonl")))}
    items = [(r["fen"], labels[(r["game"], r["ply"])]["move"]) for r in rows if (r["game"], r["ply"]) in labels and (r["game"], r["ply"] - 1) in prev
             and labels[(r["game"], r["ply"])].get("move") and is_tactic(labels[(r["game"], r["ply"])], prev[(r["game"], r["ply"] - 1)], r["fen"])]
    if sys.argv[1] == "diff":                                       # diff A B: the positions where A finds Stockfish's move and B does not
        sa, sb = sys.argv[2], sys.argv[3]
        fa = {k: int(v) for k, v in (x.split("=") for x in sa.split(","))}; fb = {k: int(v) for k, v in (x.split("=") for x in sb.split(","))}
        with Pool(os.cpu_count()) as p:
            ra = p.map(_one, [(f, fa) for f, _ in items], chunksize=2); rb = p.map(_one, [(f, fb) for f, _ in items], chunksize=2)
        for (f, mv), (a, _), (b, _) in zip(items, ra, rb):
            if a and a[1] == mv and not (b and b[1] == mv):
                print(f"DIFF {f} SF {mv}: {sa} -> {a[3]} | {sb} -> {b and b[3]}")
        return
    for spec in sys.argv[1:]:
        flags = {k: int(v) for k, v in (x.split("=") for x in spec.split(","))}
        with Pool(os.cpu_count()) as p:
            res = p.map(_one, [(f, flags) for f, _ in items], chunksize=2)
        imm = sum(1 for (r, _) in res if r); sfm = sum(1 for (r, _), (_, mv) in zip(res, items) if r and r[1] == mv)
        print(f"OBS {spec}: {len(items)} Stockfish tactics -- immediate win {imm}, Stockfish's move {sfm}; cpu {sum(t for _, t in res):.0f} s")


if __name__ == "__main__":
    main()
