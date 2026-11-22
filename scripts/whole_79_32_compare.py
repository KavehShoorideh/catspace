"""PLAN 79.32 PART B: compare two baselines (the whole reply of every question of layer4_goals.rs and layer0_meaning.rs, and the answers dump) answer by answer.
Usage: python scripts/whole_79_32_compare.py <before-label> <after-label>   (the files are artifacts/experiments/whole_79_32/<label>.{goals,meaning,answers}.dump)
Prints, for each file, the number of questions, how many are identical to the last number, and for each difference the field that differs (answers, nodes, makes, matches, calls, pairs). Exit code 1 when any question differs. A difference is a fault to find, never a new baseline."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from catspace.io import paths

DIR = Path(paths.experiment("whole_79_32/x")).parent


def load(label: str, kind: str) -> list[str]:
    return (DIR / f"{label}.{kind}.dump").read_text().splitlines()


def fields_of(payload: str) -> dict[str, str]:
    m = re.match(r"Reply \{ answers: (.*), both: (.*), nodes: (\d+), makes: (\d+), matches: (\d+), diag: (.*) \}$", payload)
    if not m:
        return {"whole": payload}
    return {"answers": m.group(1), "pairs": m.group(2), "nodes": m.group(3), "makes": m.group(4), "matches": m.group(5), "calls": m.group(6)}


def compare(kind: str, a: str, b: str) -> int:
    la, lb = load(a, kind), load(b, kind)
    differ = 0
    by_field: dict[str, int] = {}
    shown = 0
    if len(la) != len(lb):
        print(f"{kind}: the lengths differ: {len(la)} against {len(lb)}")
        return 1
    for x, y in zip(la, lb):
        if x == y:
            continue
        differ += 1
        head_x, _, pay_x = x.rpartition("\t") if kind != "answers" else (x, "", "")
        head_y, _, pay_y = y.rpartition("\t") if kind != "answers" else (y, "", "")
        if kind == "answers":
            by_field["line"] = by_field.get("line", 0) + 1
        else:
            fx, fy = fields_of(pay_x), fields_of(pay_y)
            for k in fx:
                if fx[k] != fy[k]:
                    by_field[k] = by_field.get(k, 0) + 1
        if shown < 3:
            shown += 1
            print(f"DIFF {kind}: {x[:300]}\n  against {y[:300]}")
    print(f"OBS 79.32 compare {kind}: {len(la)} questions, identical {len(la) - differ}, differ {differ}; fields that differ {by_field}")
    return differ


def main() -> int:
    a, b = sys.argv[1], sys.argv[2]
    bad = sum(compare(k, a, b) for k in ("goals", "meaning", "answers"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
