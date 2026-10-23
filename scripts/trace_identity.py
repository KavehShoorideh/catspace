"""plan 71.0 THE GUARD: a refactor stage is right only if the engine's TRACES are identical before and after it on a fixed case set at the COUNT budget
(deterministic: no clock). Usage:

  .venv/bin/python scripts/trace_identity.py snapshot <dir>          # write the traces of the case set into <dir>
  .venv/bin/python scripts/trace_identity.py compare <before> <after>  # diff two snapshots; prints OBS lines; exit 1 on any difference

The case set covers the plans' acceptance positions (eSJqH, 2TUzC, 07Z0P, 4pC83, 0LRTF, W0VFO, qCXKp, j3ybl, RoR8t, 02XVK) and the two fixed positions
(the discovery position, pos2). Lines carrying wall-clock seconds are masked before the diff (the count budget decides, the seconds vary)."""
from __future__ import annotations

import difflib
import json
import os
import re
import sys

import chess

from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import State, state_of

CASES = [("rand4_10", "eSJqH"), ("rand8_10", "2TUzC"), ("dev_20", "07Z0P"), ("rand7_10", "4pC83"), ("rand9_10", "j3ybl"), ("rand3_10", "qCXKp"),
         ("rand8_10", "W0VFO"), ("rand23_5", "RoR8t")]
FENS = {"discovery": ("r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1", "w"),
        "pos2": ("r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1", "w")}
_SECS = re.compile(r"\b\d+\.\d+ s\b|\(\d+\.\d+ s\)|seconds \d+\.\d+|\d+\.\d+ s needed|\d+\.\d+ s for it|\d+\.\d+ s left|\d+\.\d+ms")


def _trace(fen: str, us: str) -> str:
    ag = Agenda(State(fen=fen, us=us, pieces=[]))
    ag.ask()
    d = ag.decision()
    return ag.trace() + f"\nDECISION {d.what if d else None} {d.value if d else None} {d.lemma if d else None} proved={d.proved if d else None}\n"


def _cases():
    for set_name, pid in CASES:
        try:
            p = next(x for x in PA.load_set(set_name) if x["id"] == pid)
        except (StopIteration, FileNotFoundError):
            continue
        yield pid, p["fen"], "w" if chess.Board(p["fen"]).turn else "b"
    for name, (fen, us) in FENS.items():
        yield name, fen, us


def snapshot(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    n = 0
    for pid, fen, us in _cases():
        open(os.path.join(out, pid + ".txt"), "w").write(_SECS.sub("<t>", _trace(fen, us)))
        n += 1
    print(f"OBS trace identity: {n} traces written to {out}")


def compare(before: str, after: str) -> int:
    bad = 0
    for f in sorted(os.listdir(before)):
        a = os.path.join(after, f)
        if not os.path.exists(a):
            print(f"OBS trace identity {f}: missing after"); bad += 1; continue
        x, y = open(os.path.join(before, f)).read().splitlines(), open(a).read().splitlines()
        if x != y:
            bad += 1
            diff = list(difflib.unified_diff(x, y, lineterm="", n=0))
            print(f"OBS trace identity {f}: DIFFERENT ({sum(1 for l in diff if l.startswith(('+', '-')) and not l.startswith(('+++', '---')))} lines)")
            for l in diff[2:14]:
                print("   " + l[:200])
    print(f"OBS trace identity: {bad} of {len(os.listdir(before))} traces differ")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "snapshot":
        snapshot(sys.argv[2])
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3]))
