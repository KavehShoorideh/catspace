"""Plan 54 (A) comparison harness: the agenda with the binder's Python path (BINDER_RUST = False) and with `analyst_core.verify_route` (True), each in a fresh
process (the memo caches must not leak between the two), diffing every decision AND every `tried` tuple list that `binder.verify_tried` produced.

  .venv/bin/python scripts/compare_binder.py                       # 07Z0P, 0LRTF, rand1_10, rand2_10, rand3_10
  .venv/bin/python scripts/compare_binder.py --targets rand1_10 --old rust --new rust,second,cache      # switches: rust | second | cache (python = all off)
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PUZZLES = {"07Z0P": dict(id="07Z0P", fen="2r2rk1/1p4bp/6p1/pP1qpp2/6PP/P1n1BQ2/2B2R2/5RK1 w - - 1 29"),
           "0LRTF": dict(id="0LRTF", fen="2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 6 62")}


def puzzles(target: str) -> list:
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    if target in PUZZLES:
        want = PUZZLES[target]
        for name in ("dev_20", "dev_300", "fresh_300"):
            for p in PA.load_set(name):
                if p["id"] == want["id"]:
                    assert p["fen"] == want["fen"], (p["fen"], want["fen"])
                    return [p]
        raise SystemExit(f"{target} not found in dev_20 / dev_300 / fresh_300")
    return PA.load_set(target)


def worker(target: str, mode: str, out: str) -> None:
    from catspace.incubator.analyst.discovery import analyse as AN
    from catspace.incubator.analyst.discovery import binder as B
    flags = set(mode.split(","))
    B.BINDER_RUST = "rust" in flags
    B.SECOND_RUST = "second" in flags
    B.VERIFY_CACHE = "cache" in flags
    B.TRIED_LOG = []
    t0 = time.time()
    res = [AN.analyse(p, quiet=True, memory=False, log=lambda *a, **k: None) for p in puzzles(target)]
    secs = time.time() - t0
    for r in res:
        r.pop("secs", None)
    pickle.dump(dict(results=res, tried=B.TRIED_LOG, seconds=secs), open(out, "wb"))


def run(target: str, mode: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "r.pkl")
        subprocess.run([sys.executable, __file__, "--worker", target, mode, out], check=True)
        return pickle.load(open(out, "rb"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["07Z0P", "0LRTF", "rand1_10", "rand2_10", "rand3_10"])
    ap.add_argument("--old", default="python", help="comma list of switches ON for the old side: rust (BINDER_RUST), second (SECOND_RUST), cache (VERIFY_CACHE); python = none")
    ap.add_argument("--new", default="rust,second,cache", help="the same for the new side")
    ap.add_argument("--worker", nargs=3, metavar=("TARGET", "MODE", "OUT"))
    a = ap.parse_args()
    if a.worker:
        worker(a.worker[0], a.worker[1], a.worker[2])
        return
    total = 0
    for t in a.targets:
        old, new = run(t, a.old), run(t, a.new)
        dec_diff = [(o["id"], o, n) for o, n in zip(old["results"], new["results"]) if o != n]
        dec_diff += [("length", len(old["results"]), len(new["results"]))] if len(old["results"]) != len(new["results"]) else []
        # the tuples, in the order verify_tried was called (the agenda is deterministic); a different count is a difference too
        to, tn = old["tried"], new["tried"]
        tdiff = [(i, x, y) for i, (x, y) in enumerate(zip(to, tn)) if x != y] + ([("count", len(to), len(tn))] if len(to) != len(tn) else [])
        total += len(dec_diff) + len(tdiff)
        print(f"OBS binder compare {t}: decisions {'identical' if not dec_diff else 'different (%d)' % len(dec_diff)}, tried tuples {'identical' if not tdiff else 'different'} "
              f"({len(tdiff)} differences of {len(to)} verify_tried calls), seconds old {old['seconds']:.1f} new {new['seconds']:.1f}")
        for d in dec_diff[:3]:
            print("  decision diff:", d)
        for d in tdiff[:5]:
            print("  tried diff:", d)
    print(f"OBS binder compare total differences: {total}")


if __name__ == "__main__":
    main()
