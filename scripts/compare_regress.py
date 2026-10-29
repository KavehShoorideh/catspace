"""Plan 54 (B) comparison harness: the agenda with regress.py's Python path (DEFECTS_RUST = ROUTES_RUST = False) and with the Rust fix filter (the flags named by --flags),
each in a fresh process (the memo caches must not leak between the two), diffing every decision AND every route list that `regress.routes_of` / `fixes_of` /
`forcing_why` answered (regress.REGRESS_LOG).

  .venv/bin/python scripts/compare_regress.py                       # 07Z0P, 0LRTF, rand1_10, rand2_10, rand3_10
  .venv/bin/python scripts/compare_regress.py --targets rand1_10 --flags routes     # routes | cache | trap | skewer | moves | threat | piece | pieces | mates (the new side's switches)
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


# switch name -> (module of catspace.incubator.analyst.discovery, attribute); the 54 (B5.3) spread: moves (regress._moves cache), threat (reasoned._threat_moves in Rust),
# piece (graph.with_piece), pieces (facts.board_piece memo), mates (graph.mates)
# (plan 76.4a: the "defects" switch left with regress.DEFECTS_RUST -- the defects and the fix filter are the regression's, no Rust path to switch)
FLAGS = {"routes": ("regress", "ROUTES_RUST"), "trap": ("regress", "TRAP_RUST"), "skewer": ("regress", "SKEWER_RUST"),
         "moves": ("regress", "MOVES_CACHE"), "threat": ("reasoned", "THREAT_RUST"), "piece": ("graph", "FAST_PIECE"), "pieces": ("facts", "PIECE_MEMO"), "mates": ("graph", "MATES_FAST"),
         "parries": ("agenda", "PARRIES"), "delayed": ("agenda", "DELAYED_THREAT"), "nosafe": ("agenda", "NO_SAFE_MOVE"), "tiers": ("agenda", "TIERS")}      # plan 61 A (agenda switches: default OFF on both sides)


def worker(target: str, flags: list, out: str) -> None:
    from catspace.incubator.analyst.discovery import analyse as AN
    from catspace.incubator.analyst.discovery import regress as R
    import importlib
    for k, (mod, attr) in FLAGS.items():
        m = importlib.import_module(f"catspace.incubator.analyst.discovery.{mod}")
        if hasattr(m, attr):
            setattr(m, attr, k in flags)
    R.REGRESS_LOG = []
    import analyst_core
    analyst_core.set_caches("cache" in flags)
    t0 = time.time()
    res = [AN.analyse(p, quiet=True, memory=False, log=lambda *a, **k: None) for p in puzzles(target)]
    secs = time.time() - t0
    for r in res:
        r.pop("secs", None)
        for t in r.get("tiers", ()):
            t.pop("secs", None)                                      # (plan 61: tiers run -> their wall seconds are not a decision)
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
    ap.add_argument("--old", nargs="*", default=[], help="the old side's switches (default none: the Python path)")
    ap.add_argument("--worker", nargs=3, metavar=("TARGET", "FLAGS", "OUT"))
    a = ap.parse_args()
    if a.worker:
        worker(a.worker[0], [f for f in a.worker[1].split(",") if f != "-"], a.worker[2])
        return
    total = 0
    for t in a.targets:
        old, new = run(t, a.old), run(t, a.flags)
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
