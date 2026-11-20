"""PLAN 79.16 PART E, THE 300 PUZZLES OF 79.09 AT 0.2 S AND 2.0 S AGAINST THE OLD PLAYER AT EQUAL WALL, THE WAKE WITH A TO D (SYSTEM_SPEC 8 the whole is judged, not the piece; piece: the wake). A PRINTING SCRIPT: it reads puzzle positions (the game's own board, so it may use its words) and prints counts.
The same puzzles, clocks, equal-wall rule and bands as 79.15's (`whole_79_15_puzzles_print`, reused); the arms are the configs of 79.16 (`whole_79_16.CONFIGS`: abc, abcd). One run, 9 workers (a Pool from this real script file, one thread a worker), a quiet machine.

  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_16_puzzles_print [--count 300] [--configs abc,abcd] [--tag full]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing
import time

import chess

from catspace.incubator.analyst.discovery import equal_wall as EW
from catspace.incubator.analyst.discovery import ledger_puzzles as LP
from catspace.incubator.analyst.discovery import whole_79_14 as W14
from catspace.incubator.analyst.discovery import whole_79_14_puzzles_print as P14
from catspace.incubator.analyst.discovery import whole_79_16 as W16

_STATE: dict = {}


def init(configs: list) -> None:
    _STATE["world"] = W14.world()
    _STATE["table"] = W16.table()
    _STATE["configs"] = configs


def job(item: tuple) -> dict:
    pid, text, answer, rating, themes = item
    w = _STATE["world"]
    out: dict = {"id": pid, "answer": answer, "rating": rating, "themes": themes}
    for secs in P14.CLOCKS:
        o = w.old(text, secs)
        row: dict = {"old": o["move"], "old_wall": o["wall"], "old_depth": o.get("depth"), "arms": {}}
        for cfg in _STATE["configs"]:
            r = W14.Runner(w, W16.CONFIGS[cfg]).fresh(text, secs, goal_table=_STATE["table"])
            e = EW.at_wall(lambda c: w.old(text, c), r["wall"])
            lg = r["log"]
            ff = lg.get("failed_first")
            row["arms"][cfg] = {"wake": r["move"], "wall": r["wall"], "depth": r["depth"], "old_eq": e["row"]["move"], "old_eq_wall": e["wall"], "old_eq_depth": e["row"].get("depth"), "reached": e["reached"],
                                "lowered": lg.get("first_lowered") is not None, "ff": ff["outcome"] if ff else None}
        out[str(secs)] = row
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--configs", default="abc,abcd")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    root = W16.out_root(a.tag)
    configs = a.configs.split(",")
    rows = LP.dev_rows(14 + a.count, offset=11)[14:]
    items = []
    for r in rows:
        b = chess.Board(r["FEN"])
        sol = r["Moves"].split()
        b.push_uci(sol[0])
        items.append((r["PuzzleId"], b.fen(), sol[1], int(r["Rating"]), r["Themes"]))

    def say(line: str) -> None:
        TM.emit([line], root / "obs.txt")
    load0, busy = TM.wait_quiet(say)
    say(f"OBS whole_79_16 puzzles: {len(items)} puzzles of 79.09 at the clocks {P14.CLOCKS}, configs {configs}, workers {a.workers}, load average at start {load0:.2f}, cores busy {busy:.2f}")
    t0 = time.time()
    res: list = []
    with multiprocessing.get_context("spawn").Pool(a.workers, initializer=init, initargs=(configs,)) as pool:
        for k, x in enumerate(pool.imap_unordered(job, items), 1):
            res.append(x)
            if k % 50 == 0 or k == len(items):
                print(f"{k}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    (root / "puzzles.json").write_text(json.dumps(res))
    for x in P14.lines_for(sorted(res, key=lambda r: r["id"]), configs):
        say(x.replace("OBS whole_79_14", "OBS whole_79_16", 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
