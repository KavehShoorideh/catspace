"""PLAN 78.99 PART ONE, THE ROOM UNDER EACH MEASURED RECORD OF WORTHS (SYSTEM_SPEC 4 the tactical reasoner's equal reading, 5 the value and its measured records, 12 the row "the ladder under each measured worth record"; piece: tactical reasoning).
A MEASURE and nothing else: no player is changed, the default record of the engine is not changed (the record is passed to the search as a parameter, as 78.76 and 78.98 did), nothing is written to rules/learned or data/derived; every output goes under
artifacts/experiments/record_78_99/<tag>/. This file never calls the oracle and reads none of its numbers: it reads the labels of 78.96 (prior_78_96/full/equals.jsonl, read only; the folder is never written). Game-free: a move is a word; the game is named
only through the pieces that exist.

WHAT IS READ. Every labelled position of 78.96 (the labels' equal set, the class after each equal move, the class of the position before). At each position the search of the matches (`solve_equals_batch`, ledger share 0.4, one thread, the PLAY clock 0.2 s,
the same call as room_reads.Reads) under each of the three measured records of the kinds (equal_trace.RECORDS: gen2, the odds games' and the judge in play; selfplay_78_58; human_78_59): the move it plays first and the moves it holds equal, the depth, whether the
equal reading was cut by the clock, an error. The odds record is read AGAIN here (the labels' reading is a second run of the same search: what leaves the set by running the same search again is the noise to read the other two against).

THE ROOM is 78.97's (room_reads.room_ids: the positions of the labels with two or more equal moves that differ in keeping the class). The positions NOT in the room are every other labelled position.
THE CLASS OF A MOVE is the label's: the class after it where the label read it (the equal set), the class of the oracle's own best move where the move is that one, else NO CLASS (the oracle is not called here; moves without a class are counted apart, never guessed).
A move KEEPS the class when its class is at least the class of the position before it (prior_equal.keeps).

INTERFACE (written before the bodies).
  Reads(record_names)           per process: the worth of each record by name. `.read(moves) -> {name: row}` runs the search at the play clock under each record; row {"first", "equals", "depth", "cut", "error", "seconds"} as room_reads.
  run(tag, limit, workers, say) the Pool over `workers` processes (the clock is timed: the number leaves cores free, as 78.92 and 78.97 did), a file a chunk of CHUNK positions under <tag>/reads/ (a chunk with a file is not redone), then reads.jsonl;
                                the load average at the start and the end printed. The positions are the labelled positions in the order of the hash of 78.96 (a short run is the first `limit`).
  load(root) -> [record]        {"id", "ord", "part", "band", "room", "rows": {record name: row}}.
  class_of(rec, move) -> int|None      the label's class of a move, None without one (pure; tested).
  first_keeps(rec, move) -> bool|None  whether the move keeps the class; None without a class (pure; tested).
  leaves(rec, row) -> {"lose": [bool], "keep": [bool]} | None    as room_reads.leaves (pure; tested).
  Called by: record_measure.py (the tables), the command line (`reads`).

  .venv/bin/python -m catspace.incubator.analyst.discovery.record_reads reads --tag short --limit 100 | --tag full [--workers 9]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import equal_trace as ET
from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import room_reads as RR
from catspace.io import paths

OUT = ("record_78_99",)
RECORDS = ET.RECORDS
CHUNK = 20


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS record {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


def class_of(rec: dict, move: "str | None") -> "int | None":
    """See the module doc."""
    if move is None:
        return None
    if move in rec["tied"]:
        return rec["cm"][rec["tied"].index(move)]
    if move == rec.get("best") and rec.get("best_cls") is not None:
        return rec["best_cls"]
    return None


def first_keeps(rec: dict, move: "str | None") -> "bool | None":
    c = class_of(rec, move)
    return None if c is None else PE.keeps(rec["cls"], c)


def leaves(rec: dict, row: dict) -> "dict | None":
    return RR.leaves(rec, row)


def universe_ids(recs: list) -> list:
    """The labelled positions in the order of the hash of 78.96, with whether each is in the room."""
    room = {r["id"] for r in RR.room_ids(recs)}
    return [dict(r, room=r["id"] in room) for r in sorted(recs, key=lambda r: PE.order_key(r["ord"], r["ply"]))]


class Reads:
    """See the module doc."""

    def __init__(self, names: tuple = RECORDS, seconds: float = TM.SECONDS):
        from catspace.incubator.analyst.discovery import threat_concrete as TC
        self.worth = {n: dict(TC.Record.load(n).worth) for n in names}
        self.seconds = seconds

    def one(self, moves: list, name: str) -> dict:
        import cs_engine as CE
        from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
        t0 = time.perf_counter()
        try:
            raw = CE.solve_equals_batch([(OM.VARIANT, None, list(moves))], self.seconds, worth=dict(self.worth[name]), ours_first=False, valuation="odds record", threads=1, ledger_share=TM.LEDGER_SHARE)[0]
        except Exception as e:                                         # a read the search cannot take is counted, never silently
            return {"first": None, "equals": [], "depth": None, "cut": False, "error": f"{type(e).__name__}: {str(e)[:120]}", "seconds": time.perf_counter() - t0}
        row = PM.equal_set(raw)
        return {"first": row["move"], "equals": row["equals"], "depth": row["depth"], "cut": bool(row["cut"]), "error": row["error"] or None, "seconds": time.perf_counter() - t0}

    def read(self, moves: list) -> dict:
        return {n: self.one(moves, n) for n in self.worth}


_RD = None


def _init_worker() -> None:
    global _RD
    _RD = Reads()


def _job(args: tuple) -> tuple:
    idx, items, root = args
    f = Path(root) / "reads" / f"c{idx:05d}.json"
    if f.exists():
        return idx, 0.0
    t0 = time.perf_counter()
    out = [{"id": r["id"], "ord": r["ord"], "part": r["part"], "band": r["band"], "room": r["room"], "rows": _RD.read(moves)} for r, moves in items]
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0


def load(root: Path) -> list:
    return [json.loads(x) for x in (root / "reads.jsonl").read_text().splitlines() if x.strip()]


def run(tag: str, limit: int, workers: int, say) -> None:
    root = out_root(tag)
    (root / "reads").mkdir(exist_ok=True)
    recs = universe_ids(PE.load(RR.labels_root()))
    if limit:
        recs = recs[:limit]
    games = {g["ord"]: g for g in FL.universe()}
    say(f"the labelled positions: {len(recs)} (in the room {sum(r['room'] for r in recs)}; FIND {sum(r['part'] == 'find' for r in recs)}, CONFIRM {sum(r['part'] == 'confirm' for r in recs)}); records {', '.join(RECORDS)}; the play clock {TM.SECONDS} s, "
        f"ledger share {TM.LEDGER_SHARE}; {workers} workers of {os.cpu_count()} cores")
    load0, busy = TM.wait_quiet(lambda s: say(s.replace("OBS ", "", 1)))
    jobs = [(a // CHUNK, [(r, games[r["ord"]]["moves"][:r["ply"]]) for r in recs[a:a + CHUNK]], str(root)) for a in range(0, len(recs), CHUNK)]
    say(f"load average at the start {os.getloadavg()[0]:.2f}")
    t0 = time.perf_counter()
    done = 0
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for _i, _dt in pool.imap_unordered(_job, jobs, chunksize=1):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                say(f"progress: {done} of {len(jobs)} chunks; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    out = []
    for f in sorted((root / "reads").glob("c*.json")):
        out += json.loads(f.read_text())
    (root / "reads.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    secs = {n: sum(r["rows"][n]["seconds"] for r in out) / max(1, len(out)) for n in RECORDS}
    say(f"read {len(out)} of {len(recs)} positions in {time.perf_counter() - t0:.0f} s wall; load average at the end {os.getloadavg()[0]:.2f}; one thread's seconds a position by record: " + ", ".join(f"{n} {s:.2f}" for n, s in secs.items()))


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["reads"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    run(a.tag, a.limit, a.workers, Say(out_root(a.tag), "reads"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
