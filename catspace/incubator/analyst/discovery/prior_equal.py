"""PLAN 78.96 PART ONE, THE ROOM OF A CHOICE (SYSTEM_SPEC 7 the prior and "the future imagination layer", 9 "the oracle labels positions and frames in sleep, for the prior" and the wall). A SLEEP STAGE, and the only file of 78.96 that calls the oracle (through the
wrapper ladder_oracle, as future_labels does): for a sample of labelled positions it asks the search of the matches for the moves it holds EQUAL and asks the oracle, as a labeller, for the CLASS of the position after each of them and after its own best move.
A sandbox: nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/prior_78_96/<tag>/. Game-free: the game is named only through the pieces that exist (ladder_oracle, future_labels, odds_measure, the engine).
No player id is written (a game is an ordinal of the 78.59 file). The class of a move is the TARGET of the prior of part two and nothing else; no oracle number is an input of any fit (the score is kept beside the class for the record only).

THE SAMPLE. From the labelled games of 78.94 (labels.jsonl, FIND and CONFIRM kept apart by their part) every labelled position is a candidate; its class is the oracle's class of the side to move (stored), its band the count of units (few 0-8, mid 9-16, many 17 and more). The
sample takes, for each part, class and band, the same number of positions (fewer where the cell has fewer), at most CAP of one game, in the order of a hash (the same order every run); the cell quota is a run setting (`--cell`). A stratified sample: the printed shares are per cell and per
band, not the natural frequencies of games; coverage per band is printed.

PER POSITION. (1) The search at the play clock (0.2 s, ledger share 0.4, one thread, the search of the matches: `solve_equals_batch`): its first move and the moves it holds equal to it. (2) The oracle at the setting of 78.94 (depth FUTURE_DEPTH, one line, one thread, default hash):
the class, for the side that plays, of the position after each equal move and after the oracle's own best move (the first move of its stored line; if that move is among the equal moves its class is the one already read). A move KEEPS the class when its class is at least the class of the
position before it (the stored class); the equal moves DIFFER when some keep it and some do not (the room of a choice); they differ in class when the classes after them are not all the same.

INTERFACE (written before the bodies).
  sample(recs, cell, cap) -> [item]            `recs` the label records; item {"ord", "part", "ply", "units", "band", "cls"}; the order of the hash, the same every time.
  Equals()                                      per process: the search's worth record and clock. `.search(moves) -> {"first", "equals", "depth", "error"}`; `.label(moves, item, best) -> record`.
  label_record(item, search, classes, best_move, best_class, cps) -> record     the record as written (pure; tested).
  keeps(cls_before, cls_after) -> bool          the class is kept when it is not lower.
  differ(record) -> {"keep": bool, "class": bool}    the room of the position: some equal moves keep the class and some do not; the classes after them are not all the same.
  run(tag, cell, cap, workers, say)             the Pool over all cores, one file a chunk of CHUNK positions under <tag>/equals/ (a chunk with a file is not redone), then equals.jsonl.
  load(root) -> [record]                        for the measure.
  Called by: prior_measure.py (the stages), the command line (`label`).

  .venv/bin/python -m catspace.incubator.analyst.discovery.prior_equal label --tag short --cell 15 | --tag full --cell N [--workers K]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import ladder_oracle as LO
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.io import paths

OUT = ("prior_78_96",)
SRC = ("future_78_94", "full")
CAP = 3                                                       # positions of one game in the sample at most (the sample is spread over games)
CHUNK = 10                                                    # positions to a file: a stop loses at most one chunk a worker
SEED = "prior96"


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS prior {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


def band_of(units: int) -> str:
    return next(n for n, lo, hi in LM.BANDS if lo <= units <= hi)


def order_key(ordinal: int, ply: int) -> int:
    return int(hashlib.md5(f"{SEED}:{ordinal}:{ply}".encode()).hexdigest(), 16)


def sample(recs: list, cell: int, cap: int = CAP) -> list:
    """See the module doc."""
    cells: dict = {}
    for r in recs:
        for q in r["positions"]:
            it = {"ord": r["ord"], "part": r["part"], "ply": q["ply"], "units": q["units"], "band": band_of(q["units"]), "cls": q["cls"]}
            cells.setdefault((r["part"], q["cls"], it["band"]), []).append(it)
    out = []
    for key in sorted(cells):
        used: dict = {}
        n = 0
        for it in sorted(cells[key], key=lambda i: order_key(i["ord"], i["ply"])):
            if n >= cell:
                break
            if used.get(it["ord"], 0) >= cap:
                continue
            used[it["ord"]] = used.get(it["ord"], 0) + 1
            out.append(it)
            n += 1
    return sorted(out, key=lambda i: order_key(i["ord"], i["ply"]))


def keeps(before: int, after: int) -> bool:
    return after >= before


def differ(rec: dict) -> dict:
    ks = [keeps(rec["cls"], c) for c in rec["cm"]]
    return {"keep": any(ks) and not all(ks), "class": len(set(rec["cm"])) > 1}


def label_record(item: dict, search: dict, classes: list, best_move: "str | None", best_class: "int | None", cps: list, seconds: dict) -> dict:
    tied = [search["first"]] + [m for m in search["equals"] if m != search["first"]]
    return {"id": f"{item['ord']}_{item['ply']}", "ord": item["ord"], "part": item["part"], "ply": item["ply"], "units": item["units"], "band": item["band"], "cls": item["cls"], "first": search["first"], "tied": tied, "depth": search["depth"],
            "cm": classes, "cp": cps, "best": best_move, "best_cls": best_class, "calls": len(tied) + (0 if best_move is None or best_move in tied else 1), "seconds": seconds}


class Equals:
    """See the module doc."""

    def __init__(self):
        self.worth = dict(OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")["worth"])

    def search(self, moves: list) -> dict:
        import cs_engine as CE
        from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
        row = PM.equal_set(CE.solve_equals_batch([(OM.VARIANT, None, list(moves))], TM.SECONDS, worth=dict(self.worth), ours_first=False, valuation="odds record", threads=1, ledger_share=TM.LEDGER_SHARE)[0])
        return {"first": row["move"], "equals": row["equals"], "depth": row["depth"], "error": row["error"]}

    def label(self, moves: list, item: dict, best: "str | None") -> "dict | None":
        t0 = time.perf_counter()
        s = self.search(moves)
        t1 = time.perf_counter()
        if s["error"] or not s["first"]:
            return None
        tied = [s["first"]] + [m for m in s["equals"] if m != s["first"]]
        cls, cps = [], []
        for m in tied:
            a = LO.analyse_after(moves, m)
            cls.append(FL.class_of(a))
            cps.append(a["cp"] if a["forced_in"] is None else None)
        bc = None
        if best:
            bc = cls[tied.index(best)] if best in tied else FL.class_of(LO.analyse_after(moves, best))
        t2 = time.perf_counter()
        return label_record(item, s, cls, best, bc, cps, {"search": t1 - t0, "oracle": t2 - t1})


_EQ = None
_BEST: dict = {}


def _init_worker(best: dict) -> None:
    global _EQ, _BEST
    _EQ = Equals()
    _BEST = best


def _job(args: tuple) -> tuple:
    idx, items, root = args
    f = Path(root) / "equals" / f"c{idx:05d}.json"
    if f.exists():
        return idx, 0.0, 0
    t0 = time.perf_counter()
    out, bad = [], 0
    for it, moves in items:
        try:
            r = _EQ.label(moves, it, _BEST.get(f"{it['ord']}_{it['ply']}"))
        except Exception as e:                                      # a position the reading cannot take is counted, never silently
            r, bad = None, bad + 1
            (Path(root) / "equals" / f"c{idx:05d}.error").write_text(f"{type(e).__name__}: {str(e)[:300]}")
        if r is not None:
            out.append(r)
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0, bad


def load(root: Path) -> list:
    return [json.loads(x) for x in (root / "equals.jsonl").read_text().splitlines() if x.strip()]


def run(tag: str, cell: int, cap: int, workers: int, say) -> None:
    root = out_root(tag)
    (root / "equals").mkdir(exist_ok=True)
    recs = [json.loads(x) for x in (Path(paths.experiment(*SRC)) / "labels.jsonl").read_text().splitlines() if x.strip()]
    items = sample(recs, cell, cap)
    games = {g["ord"]: g for g in FL.universe()}
    best = {f"{r['ord']}_{q['ply']}": (q["pv"][0] if q.get("pv") else None) for r in recs for q in r["positions"]}
    for band in ("few", "mid", "many"):
        row = {p: {c: sum(1 for i in items if i["part"] == p and i["band"] == band and i["cls"] == c) for c in (1, 0, -1)} for p in ("find", "confirm")}
        say(f"sample coverage band {band}: FIND winning {row['find'][1]}, neither {row['find'][0]}, losing {row['find'][-1]}; CONFIRM winning {row['confirm'][1]}, neither {row['confirm'][0]}, losing {row['confirm'][-1]}")
    say(f"sample: {len(items)} positions from {len({i['ord'] for i in items})} games (cell quota {cell}, at most {cap} of one game, in the order of a hash); the search: {TM.SECONDS} s a position, ledger share {TM.LEDGER_SHARE}; the oracle: depth {LO.FUTURE_DEPTH}, "
        f"one line, one thread; {workers} workers; load average {os.getloadavg()[0]:.2f}")
    jobs = []
    for a in range(0, len(items), CHUNK):
        part = items[a:a + CHUNK]
        jobs.append((a // CHUNK, [(it, games[it["ord"]]["moves"][:it["ply"]]) for it in part], str(root)))
    t0 = time.perf_counter()
    done = 0
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(best,)) as pool:
        for _i, _dt, _bad in pool.imap_unordered(_job, jobs, chunksize=1):
            done += 1
            if done % 25 == 0 or done == len(jobs):
                say(f"progress: {done} of {len(jobs)} chunks; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    recs_out = []
    for f in sorted((root / "equals").glob("c*.json")):
        recs_out += json.loads(f.read_text())
    (root / "equals.jsonl").write_text("\n".join(json.dumps(r) for r in recs_out) + "\n")
    n = max(1, len(recs_out))
    calls = sum(r["calls"] for r in recs_out)
    say(f"labelled {len(recs_out)} of {len(items)} positions in {time.perf_counter() - t0:.0f} s wall (this run); oracle calls {calls}; one thread's seconds a position: search {sum(r['seconds']['search'] for r in recs_out) / n:.3f}, oracle {sum(r['seconds']['oracle'] for r in recs_out) / n:.3f}; "
        f"equal moves a position {sum(len(r['tied']) for r in recs_out) / n:.2f}; positions with two or more equal moves {sum(len(r['tied']) >= 2 for r in recs_out)}")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["label"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--cell", type=int, default=15)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    run(a.tag, a.cell, a.cap, a.workers, Say(out_root(a.tag), "label"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
