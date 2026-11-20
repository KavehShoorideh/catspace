"""PLAN 79.19 THE GUARD OF PART C (SYSTEM_SPEC 11 the layers, the two memories; AGENT_BRIEF section 2; piece: the memory, with the wake as its reader). The plan adds no behaviour: with the same inputs the wake's logged decisions must be equal to main's. This module makes the fixed set of
positions and runs the wake through the binding on each, as `rows`: for every decision the move, the value, the depth, the nodes, the chain, the moves held equal and named, and the whole log (the reasons) with the clocks taken out. The baseline (tests/baselines/plan79_19_main.json) was written
with the binding built from main before this plan's Rust; tests/test_plan79_19.py reads the rows again and says they are equal.
FIVE SCENARIOS (a fixed depth and a clock the depth does not reach, so that nothing depends on the clock; two runs of a scenario must be equal, `twice`):
  no_goal        ten positions, no optional input given (the guard of 79.18);
  all_inputs     six positions with every optional input given: the goal table, the chances, the reasons, the seeds, the last layer, the pieces of the whole;
  held_one       a goal given from the mined measure of the one-unit class and held across several plies, with the game's record as the history;
  held_two       the same for the two-unit class;
  carried        two seats, the open tactics carried across several plies (the carry by what the graph says was touched on, and off).
INTERFACE.  scenario_rows(world, name) -> list of rows;  all_rows(world) -> {name: rows};  strip(x) -> x without its clocks;  write_baseline(path);  main()  `python -m catspace.incubator.analyst.discovery.whole_79_19 write PATH | twice`.
Called by: tests/test_plan79_19.py, whole_79_19_print.py.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import json
import re
import sys
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import lowered_first as LF
from catspace.incubator.analyst.discovery import whole_79_15 as W15
from catspace.incubator.analyst.discovery import whole_79_16 as W16
from catspace.incubator.analyst.discovery import whole_79_18 as W18
from catspace.incubator.analyst.discovery import whole_play as WP

MEASURE_ONE = ("whole_79_17", "full", "measure_one.json")
MEASURE_TWO = ("whole_79_17", "full", "measure_two.json")
SECONDS, DEPTH = 30.0, 4
HELD_PLIES = 6
CARRIED_PLIES = 6
CHAIN = "abcd"
ALL_POSITIONS = (1, 2, 3, 5, 7, 9)           # indices into W18-style POSITIONS given by the caller


def strip(x):
    """A decision's words without the clocks (any key naming seconds or a wall)."""
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if "second" not in k and "wall" not in k and k not in ("elapsed", "time")}
    if isinstance(x, list):
        return [strip(v) for v in x]
    if isinstance(x, str):
        return re.sub(r"\d+\.\d+ s\b", "<s>", x)                  # a note that says how long a step took
    return x


def words(raw: str) -> dict:
    d = json.loads(raw)
    return strip(d)


def _call(w, text: str, history: list, pieces: list, slot, extra: dict, seconds: float = SECONDS, depth: int = DEPTH) -> dict:
    kw = dict(extra)
    if slot is not None:
        kw["carry_slot"] = slot
    raw = w.model.wake_decide(text, history, seconds, depth, dict(w.worth), {}, w.goal, pieces, 0, 0, w.keys, WP.fuses(), w.chances, w.reasons, TM.LEDGER_SHARE, **kw)
    return words(raw)


def _extra_all() -> dict:
    return {"goal_table": W15.table(), "chooser": ("gain", {}, 7), "pre_share_scale": 1.0}


def positions(texts_all: list, picks: tuple) -> list:
    return [texts_all[i] for i in picks]


def _play(w, start: str, plies: int, seats: dict, history_on: bool = True) -> list:
    """Alternate decisions from `start`: seat -> (pieces, extra, slot) for a seat the wake plays, a seat not listed plays the last of the legal moves. Returns the rows of the wake's decisions."""
    CE = WP.CE
    moves: list = []
    rows = []
    for _ in range(plies):
        pos, side, legal = CE.state(WP.OM.VARIANT, start, list(moves))[:3]
        if not legal:
            break
        seat = 0 if side == "white" else 1
        if seat in seats:
            pieces, extra, slot = seats[seat]
            hist = [CE.state(WP.OM.VARIANT, start, list(moves[:k]))[4] for k in range(len(moves))] if history_on else []
            d = _call(w, pos, hist, pieces, slot, extra)
            rows.append({"ply": len(moves), "position": pos, "words": d})
            mv = d["move"] if d["move"] in legal else legal[0]
        else:
            mv = sorted(legal)[-1]
        moves.append(mv)
    return rows


def _measure_of(path: tuple) -> dict:
    d = json.loads((LF.main_checkout() / "artifacts" / "experiments" / Path(*path)).read_text())          # the mined measures of 79.17, read from the main checkout
    return {"term": d["chosen"]["term"], "d": int(d["d"]), "samples": [s["text"] for s in d["sample_positions"][:2]], "attacker_white": [s["attacker_white"] for s in d["sample_positions"][:2]]}


def scenario_rows(w, name: str, texts: list, tag: str = "a") -> list:
    pieces = W16.CONFIGS[CHAIN]
    if name == "no_goal":
        return [{"position": t, "words": x} for t, x in zip(texts, W18.no_goal_rows(w.model, texts, w.goal))]
    if name == "all_inputs":
        return [{"position": t, "words": _call(w, t, [], pieces, None, _extra_all())} for t in positions(texts, ALL_POSITIONS)]
    if name in ("held_one", "held_two"):
        m = _measure_of(MEASURE_ONE if name == "held_one" else MEASURE_TWO)
        out = []
        for i, (text, white) in enumerate(zip(m["samples"], m["attacker_white"])):
            seat = 0 if white else 1
            extra = {"measure": (m["term"], m["d"])}
            out += [{"sample": i, **r} for r in _play(w, text, HELD_PLIES * 2, {seat: (pieces, extra, f"{tag}:{name}{i}")})]
        return out
    if name == "carried":
        out = []
        for variant, switches in (("graph_off", []), ("graph_on", ["carry_graph"])):
            seats = {s: (pieces + switches, {"goal_table": W15.table(), "pre_share_scale": 1.0}, f"{tag}:carried_{variant}_{s}") for s in (0, 1)}
            out += [{"variant": variant, **r} for r in _play(w, texts[1], CARRIED_PLIES * 2, seats)]
        return out
    raise ValueError(name)


NAMES = ("no_goal", "all_inputs", "held_one", "held_two", "carried")


def all_rows(w, texts: list, tag: str = "a") -> dict:
    """The binding keeps the open tactics of a slot between calls, so each run of the scenarios has its own slots (`tag`)."""
    return {n: scenario_rows(w, n, texts, tag) for n in NAMES}


def main(argv: "list | None" = None) -> int:
    from catspace.incubator.analyst.discovery import whole_79_14 as W14
    from catspace.incubator.analyst.discovery import whole_79_19_print as P
    a = list(argv if argv is not None else sys.argv[1:])
    w = W14.world()
    if a[0] == "write":
        t0 = time.perf_counter()
        Path(a[1]).write_text(json.dumps(all_rows(w, P.POSITIONS), sort_keys=True))
        print(f"OBS whole_79_19 baseline written {a[1]} in {time.perf_counter() - t0:.1f} s")
    elif a[0] == "twice":
        x, y = all_rows(w, P.POSITIONS, "x"), all_rows(w, P.POSITIONS, "y")
        for n in NAMES:
            print(f"OBS whole_79_19 twice {n}: {len(x[n])} decisions, equal {x[n] == y[n]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
