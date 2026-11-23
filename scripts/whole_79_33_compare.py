"""PLAN 79.33: compare two recordings of the answers and counts of the walks moved onto layer 0's general walk, line by line, to the last number.
Usage: python scripts/whole_79_33_compare.py <before-label> <after-label> [kind ...]
The files are artifacts/experiments/whole_79_33/<label>.<kind>.dump (kinds: values, toward, tail, frame, forcing). With no kind, every kind that has a file for the before label is compared.
Prints, for each kind, the number of lines (questions), how many are identical, how many differ, and the first differences. Exit code 1 when any line differs or a file is missing. A difference is a fault to find, never a new baseline."""
from __future__ import annotations

import sys
from pathlib import Path

from catspace.io import paths

DIR = Path(paths.experiment("whole_79_33/x")).parent
KINDS = ("values", "toward", "tail", "frame", "forcing")


def compare(kind: str, a: str, b: str) -> int:
    fa, fb = DIR / f"{a}.{kind}.dump", DIR / f"{b}.{kind}.dump"
    if not fa.exists() or not fb.exists():
        print(f"OBS 79.33 compare {kind}: a file is missing ({fa.name} {fa.exists()}, {fb.name} {fb.exists()})")
        return 1
    la, lb = fa.read_text().splitlines(), fb.read_text().splitlines()
    if len(la) != len(lb):
        print(f"OBS 79.33 compare {kind}: the lengths differ, {len(la)} against {len(lb)}")
        return 1
    bad = [(x, y) for x, y in zip(la, lb) if x != y]
    for x, y in bad[:3]:
        print(f"DIFF {kind}: {x[:300]}\n  against {y[:300]}")
    print(f"OBS 79.33 compare {kind}: {len(la)} questions, identical {len(la) - len(bad)}, differ {len(bad)}")
    return len(bad)


def main() -> int:
    a, b = sys.argv[1], sys.argv[2]
    kinds = sys.argv[3:] or [k for k in KINDS if (DIR / f"{a}.{k}.dump").exists()]
    return 1 if sum(compare(k, a, b) for k in kinds) else 0


if __name__ == "__main__":
    sys.exit(main())
