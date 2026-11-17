"""PLAN 78.76, SYSTEM_SPEC 9 (the ladder as the diagnosis) and 5 (the measured record of the kinds is the judge): THE LADDER UNDER EACH RECORD, SIDE BY SIDE. A sandbox, game-free: it reads the results files of the ladder
(ladder_measure, one per record) and the answers file ONLY where the ladder's own scoring reads it (the set of accepted moves a rung tests); it names no object of any game and prints no oracle number.

INTERFACE (what this module takes, what it returns, who calls it).
  extra_report(results, say)    the OBS lines of one record's run beyond ladder_measure.report: the division by rung over the 400 and over the scored part of stage H's split (deep_heads.split_part), the moves
                                         the search holds equal per position at the wake's clock (mean, median, share with exactly one) and at the last clock a position reached, the hit of the search's own first move
                                         against the accepted moves (all positions, and the ties of rung 2) with a 90 pct Wilson interval, and the division by the units on the board (the ladder's bands).
  rung_changes(a, b) -> {(from, to): n}  the positions whose rung differs between two runs of the same ids, as a table of (rung in a, rung in b).
  compare(names, say)                    the tables for a list of record names (the first is the control): the divisions side by side and, for each other record, the (from, to) table against the first.
  Called by: ladder_measure (run, report) and the command line (`python -m ...ladder_compare compare NAME NAME ...`); ladder_print `--records` reads the same files for the printed cases.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import deep_heads as DH
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.io import paths

RUNGS = (1, 2, 3, 0)


def load_results(name: str, tag: str = "full") -> dict:
    root = LM.out_root(tag, name)
    return {d["id"]: d for d in (json.loads(x) for x in (root / "results.jsonl").read_text().splitlines() if x.strip()) if d["result"]}


def _division(ds: list) -> str:
    c = Counter(d["result"]["rung"] for d in ds)
    return f"n {len(ds)}: rung 1 {c[1]}, rung 2 {c[2]}, rung 3 {c[3]}, open {c[0]}"


def _equal_counts(ds: list) -> list:
    return [d["result"]["equals"] for d in ds]


def _last_clock_equals(d: dict) -> "int | None":
    ks = [v["equals"] for k, v in d["result"]["chose"].items() if k.startswith("3 search")]
    return ks[-1] if ks else None


def _share(k: float, n: int) -> str:
    lo, hi = DH.wilson(k, n)
    return f"{k:.0f} of {n} ({100 * k / max(1, n):.1f} pct [{100 * lo:.1f}, {100 * hi:.1f}], 90 pct Wilson)"


def extra_report(results: list, say) -> None:
    ok = [d for d in results if d["result"]]
    scored = [d for d in ok if DH.split_part(d["id"]) == "scored"]
    say(f"DIVISION all: {_division(ok)}")
    say(f"DIVISION scored part of the stage H split: {_division(scored)}")
    for part, ds in (("all", ok), ("scored", scored)):
        eq = _equal_counts(ds)
        if eq:
            say(f"HELD EQUAL at the wake's clock, {part}: per position mean {np.mean(eq):.2f}, median {np.median(eq):.1f}, exactly one in {sum(e == 1 for e in eq)} of {len(eq)} ({100 * sum(e == 1 for e in eq) / len(eq):.1f} pct), at most {max(eq)}")
        last = [x for x in (_last_clock_equals(d) for d in ds) if x is not None]
        if last:
            say(f"HELD EQUAL at the last clock a position reached (rungs 3 and open), {part}: {len(last)} positions, mean {np.mean(last):.2f}, median {np.median(last):.1f}, exactly one in {sum(e == 1 for e in last)}")
        say(f"HIT of the search's own first move at the wake's clock against the accepted moves, {part}: {_share(sum(bool(d['result']['first_hit']) for d in ds), len(ds))}")
        ties = [d for d in ds if d["result"]["tie"]["eligible"]]
        if ties:
            say(f"HIT of the search's own first move among the ties of rung 2, {part}: {_share(sum(bool(d['result']['tie']['control_hit']) for d in ties), len(ties))}; a random pick {np.mean([d['result']['tie']['random_hit'] for d in ties]):.3f} on average")
    for b, _lo, _hi in LM.BANDS:
        ds = [d for d in ok if d["band"] == b]
        if ds:
            eq = _equal_counts(ds)
            say(f"DIVISION by units on the board, band {b}: {_division(ds)}; held equal at the wake's clock mean {np.mean(eq):.2f}, exactly one {sum(e == 1 for e in eq)}; first move accepted {sum(bool(d['result']['first_hit']) for d in ds)}")


def rung_changes(a: dict, b: dict) -> dict:
    ids = [i for i in a if i in b]
    return {k: v for k, v in Counter((a[i]["result"]["rung"], b[i]["result"]["rung"]) for i in ids if a[i]["result"]["rung"] != b[i]["result"]["rung"]).items()}


def compare(names: list, say, tag: str = "full") -> None:
    runs = {n: load_results(n, tag) for n in names}
    ids = sorted(set.intersection(*[set(r) for r in runs.values()]))
    say(f"COMPARE positions read under every record: {len(ids)} (records {names})")
    for n in names:
        say(f"COMPARE division {n}: {_division([runs[n][i] for i in ids])}")
    base = names[0]
    for n in names[1:]:
        ch = rung_changes({i: runs[base][i] for i in ids}, {i: runs[n][i] for i in ids})
        say(f"COMPARE rung changes {base} -> {n}: {sum(ch.values())} of {len(ids)} positions changed rung")
        say("COMPARE table (from, to) " + (", ".join(f"({f}->{t}) {c}" for (f, t), c in sorted(ch.items())) or "none"))
        for f in RUNGS:
            row = [sum(1 for i in ids if runs[base][i]["result"]["rung"] == f and runs[n][i]["result"]["rung"] == t) for t in RUNGS]
            say(f"COMPARE matrix {base} -> {n}: from rung {f}: to 1 {row[0]}, to 2 {row[1]}, to 3 {row[2]}, to open {row[3]}")


def main(argv: "list | None" = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) < 3 or a[0] != "compare":
        print(__doc__)
        return 2
    root = Path(paths.experiment("ladder_78_77" if any("@" in n for n in a[1:]) else "ladder_78_76", "compare"))     # (PLAN 78.77: runs named RECORD@on / RECORD@off)
    root.mkdir(parents=True, exist_ok=True)

    def say(m: str) -> None:
        line = f"OBS ladder compare: {m}"
        print(line, flush=True)
        with open(root / "obs.txt", "a") as f:
            f.write(line + "\n")
    compare(a[1:], say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
