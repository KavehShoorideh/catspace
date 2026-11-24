"""PLAN 79.39 PART E: print the gain as OBS lines from the files scripts/gain_79_39.sh wrote (artifacts/experiments/gain_79_39/<label>.*.txt). Usage: python scripts/gain_79_39_table.py <before-label> <after-label>
Every number is read from a file the runs wrote; the load averages are those of `uptime` before and after each run (the machine is shared with another builder)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from catspace.io import paths

DIR = Path(paths.experiment("gain_79_39/x")).parent
RUNS = Path(paths.experiment("rules_share_79_37/x")).parent / "t7939"


def load_of(text: str) -> str:
    return " / ".join(m.group(1) for m in re.finditer(r"load averages?: ([\d., ]+)", text))


def main(labels) -> None:
    for label in labels:
        for n in (1, 2, 3):
            t = (DIR / f"{label}.walk{n}.txt").read_text()
            wall = re.search(r"WALL (\d+) s", t)
            rates = re.findall(r"([\d.]+) nodes a second", t)
            print(f"OBS 79.39 E {label} walk_speed run {n}: wall {wall.group(1) if wall else '?'} s, nodes a second {rates}, load {load_of(t)}")
        t = (DIR / f"{label}.rate.txt").read_text()
        med = re.search(r"median of three runs ([\d.]+) nodes a second", t)
        print(f"OBS 79.39 E {label} layer4 rate: median {med.group(1) if med else '?'} nodes a second, load {load_of(t)}")
        for n in (1, 2, 3):
            t = (DIR / f"{label}.calls{n}.txt").read_text()
            m = re.search(r"done (\{.*\})", t)
            if not m:
                print(f"OBS 79.39 E {label} calls run {n}: no result")
                continue
            r = json.loads(m.group(1))
            c = r["calls_mean"]
            print(f"OBS 79.39 E {label} calls run {n}: {r['decisions']} decisions at 0.2 s, mean wall {r['wall_mean']:.4f} s; a decision: mover {c['mover']:.0f}, turn {c['turn']:.0f}, outcome {c['outcome']:.0f}, moves {c['moves']:.0f}, captured {c['captured']:.0f}, changed {c['changed']:.0f}, apply {c['apply']:.0f}; load {load_of(t)}")


if __name__ == "__main__":
    main(sys.argv[1:])
