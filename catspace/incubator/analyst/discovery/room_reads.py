"""PLAN 78.97 PARTS ONE AND TWO, WHAT OUR OWN READING SAYS WITH MORE OF IT, AND THE VERIFIED CHOICE (SYSTEM_SPEC 5 the tactical reasoner's equal reading and the PROPOSED verified choice, 9 the oracle as a measure and labeller in sleep, 11 the layers are game-free).
A MEASURE and nothing else: no player is changed, no switch is turned on in play, nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/room_78_97/<tag>/. Game-free: a move is a word; the game is named only through the
pieces that exist. This file never calls the oracle and reads none of its numbers: it reads the labels 78.96 made (prior_78_96/full/equals.jsonl, read only; the folder is never written).

THE ROOM. The positions of the 78.96 labels with two or more equal moves where the equal moves DIFFER in keeping the class (some keep it, some do not): FIND and CONFIRM, as 78.96 kept them apart (`room_ids`). The order of the hash of 78.96 (`prior_equal.order_key`) is the
order of the sample, so a short run of N positions is the first N of the full run.

WHAT IS ASKED AT EACH POSITION (the searches of the matches: `solve_equals_batch`, ledger share 0.4, one thread, the worth record of the matches; the clock the play clock T = tiebreak_measure.SECONDS unless said):
  r1    T, both prunings on, the verification off: a second run of the search that made the labels (the noise control: what leaves the set by running the same search again);
  on    T, the verification of the choice on (mode slot, as 78.92 and 78.93 played it);
  off   T, the late-move reduction and the null move off (the prunings off; the equal reading is the same call);
  x5    5 T, both prunings on;
  x25   25 T, both prunings on.
For each: the move it plays, the moves it holds equal, the depth, whether the equal reading was cut by the clock (a cut set is smaller, never unverified), an error, the verification's state (0 off, 1 held, 2 replaced, 3 failed and kept, 4 cut), the seconds.
A move LEAVES the equal set at a setting when it is in the 78.96 set and not in the set of that setting; a setting whose equal reading was cut or failed has no answer (counted, never read as leaving). A class-losing move is one that does not keep the class (prior_equal.keeps).
STUCK (part three's set): a class-losing move of the 78.96 set that is in the set of x5, x25 and off, each read and not cut (`stuck_of`).

INTERFACE (written before the bodies).
  room_ids(recs) -> [record]        the room positions of the labels, in the order of the hash (pure; tested).
  settings(T) -> {name: (seconds, keywords)}   the five settings above.
  leaves(rec, row) -> {"lose": [bool], "keep": [bool]} | None    per class-losing and per class-keeping move of the label's set: whether it left the set of `row`; None when the row has no answer (pure; tested).
  stuck_of(rec, rows) -> [move]     the class-losing moves that stay at x5, x25 and off (pure; tested).
  Reads()                           per process: the worth record. `.read(moves) -> {name: row}` runs the five settings.
  run(tag, limit, workers, say)     the Pool over `workers` processes (the clock is timed: the number leaves cores free, as 78.92 did), one file a chunk of CHUNK positions under <tag>/reads/ (a chunk with a file is not redone), then reads.jsonl; the load average at the start and the end printed.
  load(root) -> [record]            {"id", "ord", "ply", "units", "band", "cls", "rows": {name: row}} for room_measure.
  Called by: room_measure.py (the tables), room_lines.py (the stuck moves), the command line (`reads`).

  .venv/bin/python -m catspace.incubator.analyst.discovery.room_reads reads --tag short --limit 100 | --tag full [--workers 9]
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
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.io import paths

OUT = ("room_78_97",)
LABELS = ("prior_78_96", "full")                       # read only
CHUNK = 10
SETTINGS = ("r1", "on", "off", "x5", "x25")
STATE_WORDS = ("off", "held", "replaced", "failed", "cut")


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


def labels_root() -> Path:
    return Path(paths.experiment(*LABELS))


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS room {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


def keepvec(rec: dict) -> list:
    return [PE.keeps(rec["cls"], c) for c in rec["cm"]]


def room_ids(recs: list) -> list:
    """See the module doc."""
    out = [r for r in recs if len(r["tied"]) >= 2 and any(keepvec(r)) and not all(keepvec(r))]
    return sorted(out, key=lambda r: PE.order_key(r["ord"], r["ply"]))


def settings(seconds: float) -> dict:
    return {"r1": (seconds, {}), "on": (seconds, {"verify_choice": True, "choice_mode": "slot"}), "off": (seconds, {"null_move": False, "late_reduce": False}),
            "x5": (5 * seconds, {}), "x25": (25 * seconds, {})}


def answered(row: dict) -> bool:
    return bool(row) and not row.get("error") and not row.get("cut") and row.get("first") is not None


def leaves(rec: dict, row: dict) -> "dict | None":
    if not answered(row):
        return None
    inside = set(row["equals"]) | {row["first"]}
    kp = keepvec(rec)
    return {"lose": [m not in inside for m, k in zip(rec["tied"], kp) if not k], "keep": [m not in inside for m, k in zip(rec["tied"], kp) if k]}


def stuck_of(rec: dict, rows: dict) -> list:
    if not all(answered(rows.get(n)) for n in ("x5", "x25", "off")):
        return []
    sets = {n: set(rows[n]["equals"]) | {rows[n]["first"]} for n in ("x5", "x25", "off")}
    return [m for m, k in zip(rec["tied"], keepvec(rec)) if not k and all(m in s for s in sets.values())]


class Reads:
    """See the module doc."""

    def __init__(self, seconds: float = TM.SECONDS):
        self.worth = dict(OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")["worth"])
        self.settings = settings(seconds)

    def one(self, moves: list, name: str) -> dict:
        import cs_engine as CE
        from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
        secs, kw = self.settings[name]
        t0 = time.perf_counter()
        try:
            raw = CE.solve_equals_batch([(OM.VARIANT, None, list(moves))], secs, worth=dict(self.worth), ours_first=False, valuation="odds record", threads=1, ledger_share=TM.LEDGER_SHARE, **kw)[0]
        except Exception as e:                                         # a read the search cannot take is counted, never silently
            return {"first": None, "equals": [], "depth": None, "cut": False, "error": f"{type(e).__name__}: {str(e)[:120]}", "state": 0, "seconds": time.perf_counter() - t0}
        row = PM.equal_set(raw)
        return {"first": row["move"], "equals": row["equals"], "depth": row["depth"], "cut": bool(row["cut"]), "error": row["error"] or None, "state": int(raw[12]), "seconds": time.perf_counter() - t0}

    def read(self, moves: list) -> dict:
        return {n: self.one(moves, n) for n in SETTINGS}


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
    out = [{"id": r["id"], "ord": r["ord"], "ply": r["ply"], "units": r["units"], "band": r["band"], "cls": r["cls"], "part": r["part"], "rows": _RD.read(moves)} for r, moves in items]
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0


def load(root: Path) -> list:
    return [json.loads(x) for x in (root / "reads.jsonl").read_text().splitlines() if x.strip()]


def run(tag: str, limit: int, workers: int, say) -> None:
    root = out_root(tag)
    (root / "reads").mkdir(exist_ok=True)
    recs = room_ids(PE.load(labels_root()))
    if limit:
        recs = recs[:limit]
    games = {g["ord"]: g for g in FL.universe()}
    say(f"the room: {len(recs)} positions (FIND {sum(r['part'] == 'find' for r in recs)}, CONFIRM {sum(r['part'] == 'confirm' for r in recs)}); settings {', '.join(f'{n} {s:g} s' for n, (s, _k) in settings(TM.SECONDS).items())}; ledger share {TM.LEDGER_SHARE}; {workers} workers of {os.cpu_count()} cores")
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
    secs = {n: sum(r["rows"][n]["seconds"] for r in out) / max(1, len(out)) for n in SETTINGS}
    say(f"read {len(out)} of {len(recs)} positions in {time.perf_counter() - t0:.0f} s wall; load average at the end {os.getloadavg()[0]:.2f}; one thread's seconds a position by setting: " + ", ".join(f"{n} {s:.2f}" for n, s in secs.items()))


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
