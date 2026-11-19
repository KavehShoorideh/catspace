"""PLAN 79.01 PART ONE, THE SECOND ASKING (SYSTEM_SPEC 9 "the oracle labels positions and frames in sleep"; piece: the sleep's labels). A SLEEP STAGE and the only module of this plan that calls the oracle (through the wrapper ladder_oracle, as a LABELLER: its class and number after a move are
the target of a count and nothing else; no oracle number is an input of anything in play; the wall: no play path imports this module, tests/test_plan79_01_leaf.py). A sandbox: nothing is written to rules/learned or data/derived; outputs under artifacts/experiments/leaf_79_01/<tag>/.

WHAT IS ASKED. 79.00 stored, for each equal move of each position, the oracle's class after the move and its number (stack_room: `cm`, `cp`; no second number, no depth inside the class). The firm label (leaf_firm.py) needs the oracle asked AGAIN, with the same settings as 79.00 (ladder_oracle.analyse_after at depth FUTURE_DEPTH, one line,
one thread): so every equal move of every position of the ROOM of 79.00 (`leaf_firm.in_room`; the firm room is a part of the room, so no other position is asked) is asked once more, and the class and the number of the second asking are stored beside the first. The count of oracle calls and the seconds are printed.

INTERFACE (written before the bodies).
  items_of(recs, games, limit) -> [(rec, moves)]        the room positions of 79.00's labels in the order of the sample's hash (pure given the labels and the games; tested).
  ask_record(rec, classes, cps, seconds) -> dict        {"id", "ord", "tied", "cls2", "cp2", "calls", "seconds"} (pure; tested).
  run(tag, src, workers, limit, say) / load(root)       the Pool over the cores, one file a chunk of CHUNK positions under <tag>/ask/ (a chunk with a file is not redone), then ask.jsonl.
  Called by: leaf_firm (reads ask.jsonl), leaf_measure (the margin and the tables), the command line.

  .venv/bin/python -m catspace.incubator.analyst.discovery.leaf_ask ask --tag short --limit 300 | --tag full [--workers K]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import ladder_oracle as LO
from catspace.incubator.analyst.discovery import leaf_firm as LF
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import stack_room as SR
from catspace.io import paths

CHUNK = 10
SRC = "full"                                                    # 79.00's run, read only


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS leaf {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


def src_records(src: str = SRC) -> list:
    return SR.load(Path(paths.experiment(*SR.OUT, src)))


def items_of(recs: list, games: dict, limit: int = 0) -> list:
    room = sorted((r for r in recs if LF.in_room(r)), key=lambda r: PE.order_key(r["ord"], r["ply"]))
    if limit:
        room = room[:limit]
    return [(r, games[r["ord"]]["moves"][:r["ply"]]) for r in room]


def ask_record(rec: dict, classes: list, cps: list, seconds: float) -> dict:
    return {"id": rec["id"], "ord": rec["ord"], "tied": list(rec["tied"]), "cls2": classes, "cp2": cps, "calls": len(classes), "seconds": seconds}


def _job(args: tuple) -> tuple:
    idx, items, root = args
    f = Path(root) / "ask" / f"c{idx:05d}.json"
    if f.exists():
        return idx, 0.0
    t0 = time.perf_counter()
    out = []
    for rec, moves in items:
        t1 = time.perf_counter()
        cls, cps = [], []
        for m in rec["tied"]:
            a = LO.analyse_after(moves, m)
            cls.append(FL.class_of(a))
            cps.append(a["cp"] if a["forced_in"] is None else None)
        out.append(ask_record(rec, cls, cps, time.perf_counter() - t1))
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0


def load(root: Path) -> list:
    return [json.loads(x) for x in (root / "ask.jsonl").read_text().splitlines() if x.strip()]


def run(tag: str, src: str, workers: int, limit: int, say) -> None:
    root = LF.out_root(tag)
    (root / "ask").mkdir(exist_ok=True)
    recs = src_records(src)
    games = {g["ord"]: g for g in FL.universe()}
    items = items_of(recs, games, limit)
    n_calls = sum(len(r["tied"]) for r, _m in items)
    say(f"the room of 79.00's labels ({src}): {sum(1 for r in recs if LF.in_room(r))} positions of {len(recs)}; asking again {len(items)} positions, {n_calls} equal moves, oracle depth {LO.FUTURE_DEPTH}, one line, one thread (79.00's settings); {workers} workers of {os.cpu_count()} cores; load average {os.getloadavg()[0]:.2f}")
    jobs = [(a // CHUNK, items[a:a + CHUNK], str(root)) for a in range(0, len(items), CHUNK)]
    t0 = time.perf_counter()
    done = 0
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for _i, _dt in pool.imap_unordered(_job, jobs, chunksize=1):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                say(f"progress: {done} of {len(jobs)} chunks; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    out = []
    for f in sorted((root / "ask").glob("c*.json")):
        out += json.loads(f.read_text())
    (root / "ask.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    calls = sum(r["calls"] for r in out)
    say(f"asked again {len(out)} of {len(items)} positions in {time.perf_counter() - t0:.0f} s wall; ORACLE CALLS {calls}; oracle seconds in all {sum(r['seconds'] for r in out):.0f} (one thread's)")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["ask"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    run(a.tag, a.src, a.workers, a.limit, Say(LF.out_root(a.tag), a.cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
