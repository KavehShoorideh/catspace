"""PLAN 79.15 PART E, THE 300 PUZZLES OF 79.09 AGAINST THE OLD PLAYER AT EQUAL WALL, THE WAKE WITH THE WALL KEPT AND THE REASON FOR A LINE ON THE GOAL UNIT, AT THE THREE SHARES (SYSTEM_SPEC 8 the whole is judged, not the piece; piece: the wake). A PRINTING SCRIPT: it reads puzzle positions (the game's own text) and prints what the engine answered.
The same puzzles, clocks (0.2 s and 2.0 s), arms and equal-wall rule as 79.14's (`whole_79_14_puzzles_print`, reused); the arms are the shares s1, s2, s4 of 79.15 (`whole_79_15.SHARES`). One run, 9 workers (a Pool from this real script file, one thread a worker), a quiet machine.

  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_15_puzzles_print [--count 300] [--shares s1,s2,s4] [--tag full]
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
from catspace.incubator.analyst.discovery import whole_79_15 as W15

_STATE: dict = {}


def init(shares: list) -> None:
    _STATE["world"] = W14.world()
    _STATE["table"] = W15.table()
    _STATE["shares"] = shares


def job(item: tuple) -> dict:
    pid, text, answer, rating, themes = item
    w = _STATE["world"]
    out: dict = {"id": pid, "answer": answer, "rating": rating, "themes": themes}
    for secs in P14.CLOCKS:
        o = w.old(text, secs)
        row: dict = {"old": o["move"], "old_wall": o["wall"], "old_depth": o.get("depth"), "arms": {}}
        for sh in _STATE["shares"]:
            r = W14.Runner(w, W15.PIECES).fresh(text, secs, goal_table=_STATE["table"], pre_share_scale=W15.SHARES[sh])
            e = EW.at_wall(lambda c: w.old(text, c), r["wall"])
            row["arms"][sh] = {"wake": r["move"], "wall": r["wall"], "depth": r["depth"], "old_eq": e["row"]["move"], "old_eq_wall": e["wall"], "old_eq_depth": e["row"].get("depth"), "reached": e["reached"]}
        out[str(secs)] = row
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--shares", default="s1,s2,s4")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    root = W15.out_root(a.tag)
    shares = a.shares.split(",")
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
    say(f"OBS whole_79_15 puzzles: {len(items)} puzzles of 79.09 at the clocks {P14.CLOCKS}, shares {shares}, workers {a.workers}, load average at start {load0:.2f}, cores busy {busy:.2f}")
    t0 = time.time()
    res: list = []
    with multiprocessing.get_context("spawn").Pool(a.workers, initializer=init, initargs=(shares,)) as pool:
        for k, x in enumerate(pool.imap_unordered(job, items), 1):
            res.append(x)
            if k % 50 == 0 or k == len(items):
                print(f"{k}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    (root / "puzzles.json").write_text(json.dumps(res))
    for x in P14.lines_for(sorted(res, key=lambda r: r["id"]), shares):
        say(x.replace("OBS whole_79_14", "OBS whole_79_15", 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
