"""PLAN 78.97 PART TWO, THE PICKS AMONG THE MOVES A RUN HOLDS EQUAL (SYSTEM_SPEC 5 the PROPOSED verified choice, 7 the prior). A MEASURE: it reads the reads of room_reads.py (the search's equal sets with the verification off, arm `r1`, and on, arm `on`) and, for each set, the pick of
the search's own move, of the pattern pick of 78.82 and of the prior fitted in 78.96 (prior.pkl, read only). It never calls the oracle (the wall of 78.96 stands: the prior module loaded here imports no oracle wrapper) and reads no score of it. Nothing is written to rules/learned or data/derived;
the output is artifacts/experiments/room_78_97/<tag>/picks.jsonl.

WHAT THE VERIFICATION CAN AND CANNOT DO TO A PICK (stated before any number was read). The verification of the choice asks the search's own move again without the prunings and, if it fails, puts the first move of the equal reading that holds in its place (SYSTEM_SPEC 5). It touches the search's
move only. A pick among the equal moves is made on the set the run handed back, as the match players do (78.93's reasons player played with the verification on); so the verification reaches a pick through the set it comes from (the replacement is in it, and the verification takes its time from the equal reading's slot)
and not through any asking of the pick itself. There is no call in the binding that asks one named move the verified question, and none was added (no Rust change).

INTERFACE (written before the bodies).
  tied_of(row) -> [move]            the run's own move first, then the moves it holds equal in their order (the form every pick of 78.96 read).
  Picker()                          per process: the pattern reader of 78.82 and the fitted prior of 78.96 (prior_play.prior_picker). `.picks(moves, row) -> {"first", "pattern", "prior"}`: the three moves; with fewer than two moves held all three are the search's own.
  run(tag, workers, say)            the Pool over all cores (untimed); reads reads.jsonl of the tag; picks.jsonl: {"id", "arms": {"r1": {...}, "on": {...}}}.
  load(root) -> {id: {arm: picks}}  for room_measure.
  Called by: room_measure.py (part two), the command line (`picks`).

  .venv/bin/python -m catspace.incubator.analyst.discovery.room_picks picks --tag short | --tag full [--workers K]
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
from catspace.incubator.analyst.discovery import prior_play as PPL
from catspace.incubator.analyst.discovery import room_reads as RR
from catspace.incubator.analyst.discovery import symbols_play as SP

ARMS = ("r1", "on")
CHUNK = 20


def tied_of(row: dict) -> list:
    first = row["first"]
    return [first] + [m for m in row["equals"] if m != first]


class Picker:
    """See the module doc."""

    def __init__(self):
        self.reader = SP.reader82()
        self.prior = PPL.prior_picker()

    def picks(self, moves: list, row: dict) -> "dict | None":
        if row.get("error") or row.get("first") is None:
            return None
        tied = tied_of(row)
        if len(tied) < 2:
            return {"first": tied[0], "pattern": tied[0], "prior": tied[0], "n": len(tied)}
        pattern, _d = self.reader.pick(list(moves), list(tied), "reasons")
        return {"first": tied[0], "pattern": pattern, "prior": self.prior.choose(list(moves), list(tied)), "n": len(tied)}


_PK = None


def _init_worker() -> None:
    global _PK
    _PK = Picker()


def _job(chunk: list) -> list:
    return [{"id": pid, "arms": {a: _PK.picks(moves, rows[a]) for a in ARMS}} for pid, moves, rows in chunk]


def load(root: Path) -> dict:
    out = {}
    for ln in (root / "picks.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[r["id"]] = r["arms"]
    return out


def run(tag: str, workers: int, say) -> None:
    root = RR.out_root(tag)
    reads = RR.load(root)
    games = {g["ord"]: g for g in FL.universe()}
    chunks, cur = [], []
    for r in reads:
        cur.append((r["id"], games[r["ord"]]["moves"][:r["ply"]], {a: r["rows"][a] for a in ARMS}))
        if len(cur) == CHUNK:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    t0 = time.perf_counter()
    out = []
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for k, part in enumerate(pool.imap_unordered(_job, chunks, chunksize=1), 1):
            out += part
            if k % 20 == 0:
                say(f"progress: {k} of {len(chunks)} chunks; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    order = {r["id"]: i for i, r in enumerate(reads)}
    out.sort(key=lambda r: order[r["id"]])
    (root / "picks.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    say(f"picked in {len(out)} positions, two arms each, in {time.perf_counter() - t0:.0f} s wall (untimed searches: no search is run here)")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["picks"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    run(a.tag, a.workers, RR.Say(RR.out_root(a.tag), "picks"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
