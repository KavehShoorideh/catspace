"""Plan 54 (B) comparison harness: the agenda with regress.py's Python path (DEFECTS_RUST = ROUTES_RUST = False) and with the Rust fix filter (the flags named by --flags),
each in a fresh process (the memo caches must not leak between the two), diffing every decision AND every route list that `regress.routes_of` / `fixes_of` /
`forcing_why` answered (regress.REGRESS_LOG).

  .venv/bin/python scripts/compare_regress.py                       # 07Z0P, 0LRTF, rand1_10, rand2_10, rand3_10
  .venv/bin/python scripts/compare_regress.py --targets rand1_10 --flags defects routes     # defects | routes | cache (the new side's switches)
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


FLAGS = {"defects": "DEFECTS_RUST", "routes": "ROUTES_RUST"}


def worker(target: str, flags: list, out: str) -> None:
    from catspace.incubator.analyst.discovery import analyse as AN
    from catspace.incubator.analyst.discovery import regress as R
    for k, attr in FLAGS.items():
        setattr(R, attr, k in flags)
    R.REGRESS_LOG = []
    import analyst_core
    analyst_core.set_caches("cache" in flags)
    t0 = time.time()
    res = [AN.analyse(p, quiet=True, memory=False, log=lambda *a, **k: None) for p in puzzles(target)]
    secs = time.time() - t0
    for r in res:
        r.pop("secs", None)
    pickle.dump(dict(results=res, log=R.REGRESS_LOG, seconds=secs), open(out, "wb"))


def run(target: str, flags: list) -> dict:
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "r.pkl")
        subprocess.run([sys.executable, __file__, "--worker", target, ",".join(flags) or "-", out], check=True)
        return pickle.load(open(out, "rb"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["07Z0P", "0LRTF", "rand1_10", "rand2_10", "rand3_10"])
    ap.add_argument("--flags", nargs="*", default=["defects", "routes"])
    ap.add_argument("--worker", nargs=3, metavar=("TARGET", "FLAGS", "OUT"))
    a = ap.parse_args()
    if a.worker:
        worker(a.worker[0], [f for f in a.worker[1].split(",") if f != "-"], a.worker[2])
        return
    total = 0
    for t in a.targets:
        old, new = run(t, []), run(t, a.flags)
        dec_diff = [(o["id"], o, n) for o, n in zip(old["results"], new["results"]) if o != n]
        dec_diff += [("length", len(old["results"]), len(new["results"]))] if len(old["results"]) != len(new["results"]) else []
        lo, ln = old["log"], new["log"]
        ldiff = [(i, x, y) for i, (x, y) in enumerate(zip(lo, ln)) if x != y] + ([("count", len(lo), len(ln))] if len(lo) != len(ln) else [])
        total += len(dec_diff) + len(ldiff)
        print(f"OBS regress compare {t}: decisions {'identical' if not dec_diff else 'different (%d)' % len(dec_diff)}, route lists {'identical' if not ldiff else 'different'} "
              f"({len(ldiff)} differences of {len(lo)} calls), seconds old {old['seconds']:.1f} new {new['seconds']:.1f}")
        for d in dec_diff[:3]:
            print("  decision diff:", d)
        for d in ldiff[:5]:
            print("  log diff:", d)
    print(f"OBS regress compare total differences: {total}")


if __name__ == "__main__":
    main()
