"""PLAN 79.15 PARTS A AND E, THE MEASURE OF A LOWERED FIRST MOVE (SYSTEM_SPEC 8 the wall and the rule for choosing a move; piece: the wake). A MEASURE MODULE: it reads the decision logs of the wake (read only) and counts; it names no object of the game and calls no engine.
A decision is LOWERED when the move the search put first (`search_first`) has a null-window answer (the `value` of its entry in `wake.log.moves`) below the search's value (`wake.value`) and at least one other move's answer is not. The same count was made by hand on 79.14's logs
(about one decision in a hundred; the larger gaps in about a quarter of the games); this module makes it a tracked measure, and counts after 79.15's wall how many of those the wake PLAYED (it should be nought).

INTERFACE (written before the bodies).
  main_checkout() -> Path                                  the main checkout of the repository (where the untracked logs of earlier plans live), from a worktree or the checkout itself.
  decisions(folder) -> iterator of the decision records of every game log (`*.jsonl.gz`) in a folder, game by game.
  lowered_of(record) -> dict | None                         None where the decision has no lowered first move; else {"gap", "probe", "value", "others_hold", "named", "played_first", "depth", "first"}.
  count(folder, min_gap=400) -> dict                       {"decisions", "lowered" (any gap), "big" (gap >= min_gap), "played" (lowered and the wake played the first), "played_big", "games", "games_big", "games_played_big", "by_from" {equal_from: count}}.
  asked_of(folder) -> dict                                 the means a decision of the steps' asking, from the logs: depth, frames built, frame rows, frame questions, threat list rows (against us, against them) and its questions, and the walls by step.
  line(name, c) -> str                                     the OBS line of `count`.
  Called by: whole_79_15.py (part A's pick, part E's count), whole_79_15_print.py, tests/test_plan79_15.py; command line: .venv/bin/python -m catspace.incubator.analyst.discovery.lowered_first FOLDER [FOLDER ...]
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from catspace.io import paths


def main_checkout() -> Path:
    """The main checkout: the repository root itself, or, from a worktree under `.claude/worktrees/<name>`, the checkout the worktree hangs from."""
    r = Path(paths.REPO_ROOT)
    parts = r.parts
    if ".claude" in parts and "worktrees" in parts:
        return Path(*parts[:parts.index(".claude")])
    return r


def decisions(folder) -> "iterator":
    """Every decision record of every game log in `folder`, with the game's file stem added as `_game`."""
    for f in sorted(Path(folder).glob("*.jsonl.gz")):
        with gzip.open(f, "rt") as h:
            for line in h:
                r = json.loads(line)
                if r.get("kind") == "decision" and isinstance(r.get("wake"), dict) and "log" in r["wake"]:
                    r["_game"] = f.stem
                    yield r


def lowered_of(r: dict) -> "dict | None":
    w = r["wake"]
    lg = w["log"]
    first = r.get("search_first")
    if first is None or w.get("value") is None:
        return None
    by = {x["move"]: x["value"] for x in lg["moves"]}
    probe = by.get(first)
    if probe is None or probe >= w["value"]:
        return None
    others = any(v is not None and v >= w["value"] for k, v in by.items() if k != first)
    named = next((x["named"] for x in lg["moves"] if x["move"] == first), None)
    return {"gap": w["value"] - probe, "probe": probe, "value": w["value"], "others_hold": others, "named": named, "played_first": r["played"] == first, "depth": w.get("depth"), "first": first}


def count(folder, min_gap: int = 400) -> dict:
    c = {"decisions": 0, "lowered": 0, "big": 0, "played": 0, "played_big": 0, "games": set(), "games_big": set(), "games_played_big": set(), "by_from": Counter()}
    games = set()
    for r in decisions(folder):
        c["decisions"] += 1
        games.add(r["_game"])
        x = lowered_of(r)
        if x is None or not x["others_hold"]:
            continue
        c["lowered"] += 1
        c["by_from"][r["wake"]["log"].get("equal_from", "as_before")] += 1
        big = x["gap"] >= min_gap
        c["big"] += big
        if big:
            c["games_big"].add(r["_game"])
        if x["played_first"]:
            c["played"] += 1
            c["played_big"] += big
            if big:
                c["games_played_big"].add(r["_game"])
    return {"decisions": c["decisions"], "lowered": c["lowered"], "big": c["big"], "played": c["played"], "played_big": c["played_big"], "games": len(games), "games_big": len(c["games_big"]), "games_played_big": len(c["games_played_big"]), "by_from": dict(c["by_from"])}


def line(name: str, c: dict, min_gap: int = 400) -> str:
    n = max(1, c["decisions"])
    return (f"OBS lowered_first {name}: wake decisions {c['decisions']} in {c['games']} games; a first move whose own probe is below the search's value while another move's probe is not: {c['lowered']} ({100 * c['lowered'] / n:.2f} pct), of them {c['big']} lower by {min_gap} or more, in {c['games_big']} games; "
            f"PLAYED the lowered first move: {c['played']} ({c['played_big']} of the gap {min_gap} or more, in {c['games_played_big']} games); the candidate set made by {c['by_from']}")


def named_rate(folder) -> tuple:
    """(decisions whose first move is a candidate in the log, of them those whose first move was a named one): the base rate to put beside the share among the lowered ones."""
    n = named = 0
    for r in decisions(folder):
        by = {x["move"]: x for x in r["wake"]["log"]["moves"]}
        f = r.get("search_first")
        if f in by:
            n += 1
            named += bool(by[f]["named"])
    return n, named


def _num(note: str, word: str) -> int:
    m = re.search(r"(\d+) " + re.escape(word), note)
    return int(m.group(1)) if m else 0


def asked_of(folder) -> dict:
    """The means a decision of what the steps before the search asked (from the notes of the steps in the log) and the depth: frames built, frame rows, frame questions, the threat list's rows (against us, against them) and its questions, with the walls of the steps."""
    n = 0
    tot: dict = defaultdict(float)
    for r in decisions(folder):
        w = r["wake"]
        n += 1
        tot["depth"] += w.get("depth") or 0
        tot["wall"] += r.get("wall", 0.0)
        for s in w["steps"]:
            note, piece = s["note"], s["piece"]
            if piece == "frames and the question read":
                tot["frames"] += _num(note, "frames,")
                tot["frame_rows"] += _num(note, "rows,")
                tot["frame_questions"] += _num(note, "questions,")
            elif piece == "threat list of both sides":
                tot["list_rows_us"] += _num(note, "rows against us")
                tot["list_rows_them"] += _num(note, "against them")
                tot["list_questions"] += _num(note, "questions;")
            if s["step"] in (1, 2, 3, 4, 5, 6, 7, 0) and piece in ("frames and the question read", "threat list of both sides", "the whole decision", "search with the count of worths at its end, and the equal reading", "their goals after each move of ours"):
                tot["s_" + piece] += s["seconds"]
    d = {k: v / max(1, n) for k, v in tot.items()}
    d["decisions"] = n
    return d


def main(argv: "list | None" = None) -> int:
    for f in (argv if argv is not None else sys.argv[1:]):
        print(line(Path(f).name, count(f)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
