"""PLAN 79.09 PART D: THE POSITION TEST OF THE WHOLE WAKE AGAIN (SYSTEM_SPEC 7 the stake orders goals and the budget, 8 the rule for choosing a move, 4; piece: the wake). A PRINTING SCRIPT: the only place the game's own text appears. The same sixteen positions and the same inputs as whole_79_08_print.py (it is imported and its
World, its positions and its printing are used as they are; nothing about a position is changed here), at the play clock (0.2 s) and ten times it (2.0 s), the wake and the old player each given the same clock a position, the wall time of each printed. It judges nothing: no number is written here that the engine did not print; no tuning.
Order of work (the owner's): run only after the Rust unit tests and the integration tests passed. Outputs artifacts/experiments/whole_79_09/ (through catspace.io.paths).

  python -m catspace.incubator.analyst.discovery.whole_79_09_print missed|puzzles|all [--seconds 0.2 2.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from catspace.incubator.analyst.discovery import whole_79_08_print as W8
from catspace.io import paths

OUT = ("whole_79_09",)


def out_dir() -> Path:
    p = Path(paths.experiment(*OUT))
    p.mkdir(parents=True, exist_ok=True)
    return p


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["missed", "puzzles", "all"])
    ap.add_argument("--seconds", type=float, nargs="+", default=[0.2, 2.0])
    a = ap.parse_args(argv)
    world = W8.World()
    done: list = []
    for which, pos in (("missed", W8.positions_missed), ("puzzles", W8.positions_puzzles)):
        if a.which not in (which, "all"):
            continue
        lines: list = [f"WHOLE WAKE 79.09, {which}: wake with the pieces {', '.join(W8.PIECES)}; the same inputs as 79.08; setup {world.setup_seconds:.1f} s; one run, directions only; nothing tuned", ""]
        for title, text, answer in pos():
            for sec in a.seconds:
                w, o = world.wake(text, sec), world.old(text, sec)
                done.append(W8.report(title, text, w, o, answer, sec, lines))
        (out_dir() / f"{which}_print.txt").write_text("\n".join(lines))
        print("\n".join(lines))
    (out_dir() / f"decisions_{a.which}.json").write_text(json.dumps(done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
