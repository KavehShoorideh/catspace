"""PLAN 79.32 PART C: print the table of the timed runs (before and after, three runs each) from artifacts/experiments/whole_79_32/*.time<n>.txt and *.speed<n>.txt: the whole file, the plain and narrowed trees of the narrowing test (seconds summed over threads), the narrowing parts, and the single-thread speed test."""
from __future__ import annotations

import re
from pathlib import Path

from catspace.io import paths

DIR = Path(paths.experiment("whole_79_32/x")).parent


def text(label: str, kind: str, n: int) -> str:
    return (DIR / f"{label}.{kind}{n}.txt").read_text()


def main() -> None:
    print("run files: uptime printed at the start of each run")
    for label in ("before", "after"):
        for n in (1, 2, 3):
            print(f"  {label} time{n}: {text(label, 'time', n).splitlines()[0].strip()}")
        for n in (1, 2, 3):
            print(f"  {label} speed{n}: {text(label, 'speed', n).splitlines()[0].strip()}")
    print("\nwhole layer4_goals.rs (wall seconds, release, --test-threads 1)")
    for label in ("before", "after"):
        print(f"  {label}: " + ", ".join(re.search(r"WALL (\d+) s", text(label, "time", n)).group(1) for n in (1, 2, 3)))
    print("\nthe test whose seconds are summed over threads: plain tree, narrowed tree (50000 nodes)")
    for label in ("before", "after"):
        row = []
        for n in (1, 2, 3):
            m = re.search(r"seconds \(summed over threads\) plain ([\d.]+) narrowed ([\d.]+)", text(label, "time", n))
            row.append(f"{m.group(1)} / {m.group(2)}")
        print(f"  {label}: " + "; ".join(row))
    print("\nnarrowing parts at 50000 nodes, seconds summed over threads: plain, table only, order only, both")
    for label in ("before", "after"):
        row = []
        for n in (1, 2, 3):
            v = re.findall(r"narrowing parts at 50000 nodes, [a-z ]+: nodes \d+ positions made \d+ rows matched \d+ seconds \(summed over threads\) ([\d.]+)", text(label, "time", n))
            row.append(" ".join(v))
        print(f"  {label}: " + "; ".join(row))
    print("\nsingle-thread speed test (20 questions, 50000 nodes; three askings each run), seconds: plain | narrowed")
    for label in ("before", "after"):
        for n in (1, 2, 3):
            t = text(label, "speed", n)
            plain = re.findall(r"speed plain .*?, ([\d.]+) s,", t)
            nar = re.findall(r"speed narrowed .*?, ([\d.]+) s,", t)
            counts = set(re.findall(r"nodes (\d+), positions made (\d+), rows matched (\d+)", t))
            print(f"  {label} run {n}: plain {' '.join(plain)} | narrowed {' '.join(nar)} | counts {sorted(counts)}")


if __name__ == "__main__":
    main()
