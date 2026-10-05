"""THE GATE: Kaveh's decision on a round, after reading its ledger.
    continue   the round's ideas stay in the basis; the next round may run (refused when the UNSOUND control is red)
    rollback   the round's ideas are retired from the basis (append-only: retire lines; nothing is deleted, the round's
               records stay for the history); the next round may run
    swap       name a stage to replace with a research piece; the next round refuses to run until its config names a
               different module for that stage
    python -m catspace.incubator.analyst.loop.gate --round 1 --decision continue --note "..." [--by kaveh]"""
from __future__ import annotations

import argparse
import subprocess
import time

from catspace.io import paths

from . import basis
from .store import read_json, record_dir, write_json

STAGES = ("frame", "mine", "translate", "prove", "use", "measure")


class GateError(Exception):
    pass


def decide(n: int, decision: str, note: str, stage: str | None = None, by: str = "kaveh", commit: bool = True) -> dict:
    rec = record_dir(n)
    ctl = read_json(rec / "controls.json")
    if ctl is None:
        raise GateError(f"round {n} has no ledger yet (artifacts/loop/round_{n:03d}/controls.json)")
    if decision not in ("continue", "rollback", "swap"):
        raise GateError(f"unknown decision {decision}")
    if decision == "continue" and ctl.get("red"):
        raise GateError(f"round {n} has a RED control (unsound: {ctl.get('unsound')}); it cannot be gated continue -- rollback or swap")
    if decision == "swap" and stage not in STAGES:
        raise GateError(f"swap needs --stage one of {STAGES}")
    retired = []
    if decision == "rollback":
        retired = sorted({e["id"] for e in basis.read() if e["op"] == "promote" and e["round"] == n} & set(basis.active()))
        if retired:
            basis.retire(n, retired, f"rollback of round {n}: {note}")
    g = dict(round=n, decision=decision, stage=stage, note=note, by=by, time=time.strftime("%Y-%m-%d %H:%M %Z"),
             retired=retired, basis_head=basis.head())
    write_json(rec / "GATE.json", g)
    if commit:
        _commit([rec, basis.path()], f"loop round {n:03d} gate: {decision}" + (f" {stage}" if stage else "") + f" -- {note}")
    return g


def gate_of(n: int) -> dict | None:
    return read_json(record_dir(n) / "GATE.json")


def _commit(pathlist, message: str) -> None:
    root = paths.REPO_ROOT
    rel = [str(p.relative_to(root)) if str(p).startswith(str(root)) else str(p) for p in pathlist]
    subprocess.run(["git", "add", "--", *rel], cwd=root, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode != 0:
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=root, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--decision", required=True, choices=["continue", "rollback", "swap"])
    ap.add_argument("--stage", default=None, choices=list(STAGES))
    ap.add_argument("--note", required=True)
    ap.add_argument("--by", default="kaveh")
    ap.add_argument("--no-commit", action="store_true")
    a = ap.parse_args(argv)
    g = decide(a.round, a.decision, a.note, a.stage, a.by, commit=not a.no_commit)
    print(f"gate round {a.round}: {g['decision']}" + (f" ({g['stage']})" if g["stage"] else "") + (f"; retired {len(g['retired'])}" if g["retired"] else ""))


if __name__ == "__main__":
    main()
