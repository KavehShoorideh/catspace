#!/usr/bin/env python
"""experiments/integration_tests.py -- THE REGRESSION GATE (Kaveh: 'future works have to
succeed on these specific positions, especially if we distill them into our bank/model').

Runs the bootstrap engine over tests/integration_positions.json via --fen-file, with the
IMMORTAL banks imported (distilled knowledge is part of the engine under test). PASS =
mate for must-win/clock-pressure cases. Nonzero exit on any regression -> CI-able; run
after every engine change (pairs with the stale-test enforcement)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="tests/integration_positions.json")
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--j", type=int, default=4)
    ap.add_argument("--import-banks", default="MERGE",
                    help="'MERGE' = union of all wdlr6 per-scenario banks (the immortal "
                         "distilled knowledge); a prefix path; or '' for cold")
    ap.add_argument("--tag", default="integration")
    args = ap.parse_args()
    t0 = time.time()
    cases = json.loads(Path(args.suite).read_text())
    fen_file = Path(f"artifacts/experiments/{args.tag}_fens.txt")
    fen_file.write_text("\n".join(c["fen"] for c in cases))
    res_file = Path(f"artifacts/experiments/{args.tag}_results.jsonl")
    res_file.unlink(missing_ok=True)
    for sfx in ("_bank", "_lossbank", "_drawbank"):
        Path(f"artifacts/experiments/{args.tag}{sfx}.fens").unlink(missing_ok=True)

    if args.import_banks == "MERGE":
        import glob
        pfx = f"artifacts/experiments/{args.tag}_seed"
        for sfx, pat in (("_bank", "wdlr6_bank_n5000_*.fens"),
                         ("_lossbank", "wdlr6_lossbank_n5000_*.fens"),
                         ("_drawbank", "wdlr6_drawbank_n5000_*.fens")):
            lines = set()
            for f in glob.glob(f"artifacts/experiments/{pat}"):
                lines.update(l for l in Path(f).read_text().splitlines() if l.strip())
            Path(pfx + sfx + ".fens").write_text("\n".join(sorted(lines)) + "\n")
        args.import_banks = pfx
        print(f"[integration] merged immortal banks -> {pfx}_*.fens", flush=True)

    cmd = [sys.executable, "experiments/bootstrap_mate_engine.py",
           "--fen-file", str(fen_file), "--n", str(len(cases)), "--j", str(args.j),
           "--nodes", str(args.nodes), "--max-plies", "120",
           "--bank-file", f"artifacts/experiments/{args.tag}_bank.fens",
           "--loss-bank-file", f"artifacts/experiments/{args.tag}_lossbank.fens",
           "--draw-bank-file", f"artifacts/experiments/{args.tag}_drawbank.fens",
           "--milestone-file", f"artifacts/experiments/{args.tag}_ms.jsonl",
           "--results-file", str(res_file), "--experience-db", ""]
    if args.import_banks:
        cmd += ["--import-banks", args.import_banks]
    while True:
        rc = subprocess.run(cmd).returncode
        if rc != 75:
            break
        print("[integration] relaunch on new commit", flush=True)

    seen = {}
    for ln in res_file.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            seen.setdefault(r["g"], r)
    fails = []
    for i, c in enumerate(cases):
        r = seen.get(i)
        ok = bool(r and r["mate"])
        mark = "PASS" if ok else "FAIL"
        if not ok:
            fails.append(c["name"])
        print(f"  [{mark}] {c['name']:28s} {c['kind']:14s} "
              f"{('plies=' + str(r['plies'])) if r else 'NO-RESULT'}  ({c['history']})",
              flush=True)
    print(f"VERDICT INTEGRATION {len(cases)-len(fails)}/{len(cases)} PASS "
          f"[{time.time()-t0:.0f}s]" + (f"  REGRESSIONS: {fails}" if fails else ""), flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
