"""PLAN 79.18 PART D: THE TEST AGAIN, AFTER THE TWO FAULTS OF THE HELD GOAL ARE MENDED (SYSTEM_SPEC 7 the measure of progress, 8 the rule for choosing a move and the wall; piece: the wake).
A harness: it plays the cells of plan 79.17 part E again (same starts, same mined terms, same defenders and clocks) with the wake as mended, through the harness of 79.17 (`whole_79_17`: the games, the table's defender, the summary of a cell); only the folder of the outputs differs
(artifacts/experiments/whole_79_18/). Nothing is mined again, nothing is written to a real store, the endgame table is read only by the oracle module of 79.17 (the harness's defender and labels), never by the wake. This module names no object of the game.

PRINTED for each cell (OBS lines): games, wins, mean plies, the games by how they ended (repetition, the quiet-ply rule), and the fall of the held measure along the games: falls a game, the longest run of our moves with no fall, the recoveries, the goals put to sleep and woken, the decisions cut by the clock.
ALSO: `no_goal_rows(model, texts)` (the decisions of the wake with no goal given, in the words a test compares with a snapshot made on main), and `compare` (the table beside 79.17's part E).

INTERFACE (written before the bodies).
  falls18(log_dir) -> dict                              the fall of the measure along the games of a folder, as 79.17's `falls` plus recoveries, sleeps, wakes, cuts, the longest run of decisions without a line.
  no_goal_rows(model, texts, goal_kind) -> [dict]            the wake's decision on each text with no goal given at a fixed depth (what a decision is made of, without its clocks).
  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_18 cells --classes one,two --defenders table,old --clocks 0.2,2.0 --n 40 [--workers W]
  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_18 compare
Called by: the command line, tests/test_plan79_18.py.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import shutil
import time
from argparse import Namespace
from pathlib import Path

from catspace.incubator.analyst.discovery import whole_79_17 as W17
from catspace.io import paths

OUT = ("whole_79_18",)
BEFORE = ("whole_79_17", "full")                     # the cells of plan 79.17 part E, on the main checkout's artifacts (read only)
W17.OUT = OUT                                         # the cells of 79.17's harness are played into this plan's folder


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


def falls18(log_dir: Path) -> dict:
    """The fall of the held measure along the games of a folder: falls (events of ledger row 24) a game, the longest run of our moves with no fall, the recoveries played, the goals put to sleep and woken, the decisions the clock cut, the longest run of decisions in a row without a line, the moves played toward the goal."""
    games = W17.decisions_of(log_dir)
    n = falls = longest = toward = decisions = cut = recov = slept = woken = without = 0
    why: dict = {}
    for name, decs, _ in games:
        m = [d["wake"]["log"].get("measure") for d in decs if d["wake"]["log"].get("measure")]
        if not m:
            continue
        n += 1
        run = best = 0
        for x in m:
            decisions += 1
            cut += bool(x["cut"])
            toward += bool(x["played_toward"])
            if x["reached"]:
                falls += 1
                run = 0
            else:
                run += 1
                best = max(best, run)
            without = max(without, x.get("without_line", 0))
        longest = max(longest, best)
        last = m[-1]
        recov += last.get("recovered_total", 0)
        slept += last.get("slept_total", 0)
        woken += last.get("woken_total", 0)
        for x in m:
            if x["dropped"]:
                k = x["dropped"].split(":")[0][:60]
                why[k] = why.get(k, 0) + 1
                break
    return {"games": n, "falls": falls, "falls_a_game": falls / max(1, n), "longest_run": longest, "recoveries": recov, "asleep": slept, "woken": woken, "cut_decisions": cut, "decisions": decisions, "longest_without_a_line": without,
            "toward_played": toward, "dropped_for_good": why}


def cmd_cells(a, say) -> None:
    root = out_root(a.tag)
    src = Path(paths.experiment(*BEFORE, "measures.json"))
    shutil.copyfile(src, root / "measures.json")                    # the mined terms of 79.17, not mined again
    for cls in a.classes.split(","):
        before = Path(paths.experiment(*BEFORE, f"starts_{cls}.json"))
        starts = W17.starts_for(cls, a.n)
        say(f"OBS whole_79_18 starts {cls}: the same as plan 79.17's: {json.loads(before.read_text()) == starts if before.exists() else 'the old file is absent'}")
    ns = Namespace(phase="E", classes=a.classes, attackers="wake_goal", defenders=a.defenders, clocks=a.clocks, n=a.n, workers=a.workers, tag=a.tag, measures_from=a.tag)
    W17.cmd_cells(ns, say)
    for cls in a.classes.split(","):
        for sec in [float(x) for x in a.clocks.split(",")]:
            for dfn in a.defenders.split(","):
                name = f"E_{cls}_wake_goal_v_{dfn}_{sec}"
                f = falls18(root / "logs" / name)
                say(f"OBS whole_79_18 falls18 {name}: " + json.dumps(f))


def cell_numbers(root: Path, name: str) -> "dict | None":
    p = root / f"cell_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cmd_compare(a, say) -> None:
    """The cells beside 79.17's part E: class, defender, clock, games, wins before, wins after, mean plies, ended by repetition, ended by the quiet-ply rule."""
    before = Path(paths.experiment(*BEFORE))
    after = out_root(a.tag)
    for cls in ("one", "two"):
        for dfn in ("table", "old"):
            for sec in (0.2, 2.0):
                b = cell_numbers(before, f"E_{cls}_wake_goal_v_{dfn}_{sec}")
                c = cell_numbers(after, f"E_{cls}_wake_goal_v_{dfn}_{sec}")
                no = cell_numbers(before, f"A_{cls}_wake_v_{dfn}_{sec}")
                if c is None:
                    continue
                e = c["ended"]
                say(f"OBS whole_79_18 compare {cls} defender {dfn} clock {sec}: games {c['games']}, wins with no goal (79.17 A) {no['wins'] if no else '-'}, wins before (79.17 E) {b['wins'] if b else '-'}, wins after {c['wins']}, mean plies before {b['mean_plies']:.1f} after {c['mean_plies']:.1f}, "
                    f"ended by repetition before {b['ended'].get('repetition', 0) if b else '-'} after {e.get('repetition', 0)}, by the quiet-ply rule before {b['ended'].get('quiet_rule', 0) if b else '-'} after {e.get('quiet_rule', 0)}")


# ------------------------------------------------------------------------------------------------------------------------------ the wake with no goal given

NO_GOAL_ARGS = dict(seconds=60.0, depth=5, worth={"p": 475, "n": 1216, "b": 1301, "r": 1945, "q": 3583}, pieces=["goals", "threats", "direct_names"])


def decision_words(raw: str) -> dict:
    """What a decision is made of, without its clocks: the move, the value, the depth, the chain of claims, the moves held equal and named, the reason."""
    d = json.loads(raw)
    return {"move": d.get("move"), "value": d.get("value"), "depth": d.get("depth"), "chain": [{"by": c["by"], "ended": c["ended"], "line": c["line"]} for c in d.get("chain", [])], "equal": d.get("equal"),
            "named_mine": d.get("named_mine"), "named_theirs": d.get("named_theirs"), "why": d["log"].get("why"), "chosen": d["log"].get("chosen"), "measure": d["log"].get("measure")}


def no_goal_rows(model, texts: list, goal_kind: str) -> list:
    """The wake's decision on each text, no goal given, at a fixed depth and a clock the depth does not reach (so that the decision is the same run to run)."""
    a = NO_GOAL_ARGS
    out = []
    for t in texts:
        raw = model.wake_decide(t, [], a["seconds"], a["depth"], a["worth"], {}, goal_kind, a["pieces"], 0, 0, None, None, None, None)
        out.append(decision_words(raw))
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("cells", "compare"))
    ap.add_argument("--classes", default="one,two")
    ap.add_argument("--defenders", default="table,old")
    ap.add_argument("--clocks", default="0.2,2.0")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args(argv)
    root = out_root(a.tag)

    def say(line: str) -> None:
        TM.emit([line], root / "obs.txt")
    if a.stage == "cells":
        cmd_cells(a, say)
    else:
        cmd_compare(a, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
