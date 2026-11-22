"""PLAN 79.26 PART D, WHICH DECISIONS CHANGED (SYSTEM_SPEC 8 the whole is judged, not the piece; 11 the layers; piece: tactical reasoner). A PRINTING SCRIPT: it reads puzzle positions (the game's own board, so it may use its words) and prints counts. A measure only: the puzzle
answers are never read by the search and nothing is fitted on them.
The deepening of the search stopped at an end proved with a unit given up on the way (plan 79.26 part B); this asks the wake, with no clock that decides (a fixed depth, a clock the depth does not reach), the same positions through two bindings (the build before and the build after, each
in its own process) and prints how many decisions changed and which.

  decide OUT.json --depth D [--count 300] [--workers 9]     the wake's decision on each of the puzzle positions of 79.09 (the config abcd of 79.16, the goal table of 79.14) through the binding on the path; the move, the value, the depth, the nodes
  compare A.json B.json                                     prints, as OBS lines, how many of the decisions differ and every one that does (position, the two moves, the two values and depths, the puzzle's answer)
INTERFACE: `positions(count)`, `decide_rows(items, depth)`, `compare_lines(a, b)` return what they print. Called by the command line.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing

import chess

from catspace.incubator.analyst.discovery import ledger_puzzles as LP
from catspace.incubator.analyst.discovery import whole_79_14 as W14
from catspace.incubator.analyst.discovery import whole_79_16 as W16
from catspace.incubator.analyst.discovery import whole_play as WP

_STATE: dict = {}
SECONDS = 600.0


def positions(count: int) -> list:
    rows = LP.dev_rows(14 + count, offset=11)[14:]
    items = []
    for r in rows:
        b = chess.Board(r["FEN"])
        sol = r["Moves"].split()
        b.push_uci(sol[0])
        items.append((r["PuzzleId"], b.fen(), sol[1]))
    return items


def init(depth: int) -> None:
    _STATE["world"] = W14.world()
    _STATE["table"] = W16.table()
    _STATE["depth"] = depth


def job(item: tuple) -> dict:
    pid, text, answer = item
    w = _STATE["world"]
    out = json.loads(w.model.wake_decide(text, [], SECONDS, _STATE["depth"], dict(w.worth), {}, w.goal, W16.CONFIGS["abcd"], 0, 0, w.keys, WP.fuses(), w.chances, w.reasons, TM.LEDGER_SHARE, None, goal_table=_STATE["table"]))
    return {"id": pid, "position": text, "answer": answer, "move": out["move"], "value": out["value"], "depth": out["depth"], "why": (out.get("log") or {}).get("why")}


def decide_rows(items: list, depth: int, workers: int) -> list:
    with multiprocessing.get_context("spawn").Pool(workers, initializer=init, initargs=(depth,)) as pool:
        return sorted(pool.imap_unordered(job, items), key=lambda r: r["id"])


def compare_lines(a: list, b: list) -> list:
    out = []
    da, db = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    ids = sorted(set(da) & set(db))
    moved = [i for i in ids if da[i]["move"] != db[i]["move"]]
    valued = [i for i in ids if da[i]["value"] != db[i]["value"]]
    ra = sum(1 for i in ids if da[i]["move"] == da[i]["answer"])
    rb = sum(1 for i in ids if db[i]["move"] == db[i]["answer"])
    out.append(f"OBS whole_79_26 decisions: {len(ids)} positions through the build before and the build after at a fixed depth; the move differs in {len(moved)}, the value in {len(valued)}; the puzzle's answer played: before {ra}, after {rb} (a measure only)")
    for i in moved:
        out.append(f"OBS whole_79_26 changed {i} [{da[i]['position']}]: before {da[i]['move']} (value {da[i]['value']}, depth {da[i]['depth']}), after {db[i]['move']} (value {db[i]['value']}, depth {db[i]['depth']}); the puzzle's answer {da[i]['answer']}")
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("decide", "compare"))
    ap.add_argument("files", nargs="+")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    if a.stage == "decide":
        rows = decide_rows(positions(a.count), a.depth, a.workers)
        with open(a.files[0], "w") as fh:
            json.dump(rows, fh)
        print(f"OBS whole_79_26 decide: {len(rows)} decisions at depth {a.depth} written to {a.files[0]}")
    else:
        ra, rb = (json.load(open(f)) for f in a.files[:2])
        for line in compare_lines(ra, rb):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
