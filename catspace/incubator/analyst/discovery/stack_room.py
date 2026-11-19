"""PLAN 79.00 PART TWO, THE ROOM AGAIN UNDER THE STACK (SYSTEM_SPEC 4 the equal reading and the verified choice, 5 the measured records, 7 the pick among equal moves, 9 "the oracle labels positions and frames in sleep" and the wall). A SLEEP STAGE, and with prior_equal the only place of
this plan that calls the oracle (through the wrapper ladder_oracle, as a LABELLER: its class of the position after a move is the target of the count and nothing else; no oracle number is an input of anything, the score is kept beside the class for the record only).
A sandbox: nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/stack_79_00/<tag>/ (78.94's and 78.96's folders are read only). Game-free: the game is named only through the pieces that exist (ladder_oracle, future_labels, odds_measure, the engine).
No player id is written (a game is an ordinal of the 78.59 file).

THE SAMPLE. Plan 78.96's CONFIRM part again, FRESH: the labelled CONFIRM positions of 78.94 (labels.jsonl; the class of the side to move before the move is stored), the cells of 78.96 (class and band, the same quota a cell, at most CAP of one game in a cell), in the order of a hash with another seed than 78.96's,
and positions 78.96 did not sample FIRST (78.96 left too few unsampled for a whole disjoint sample: the stored fact is printed), so the sample has the size of 78.96's CONFIRM part and each position says whether 78.96 sampled it (`seen96`); the measure reads the fresh part apart.

PER POSITION. The search of the matches at the play clock (0.2 s, ledger share 0.4, one thread) UNDER THE STACK: the human-result record as the judge, the verification of the choice on (mode slot); both passed as parameters, nothing changed in the engine. Its first move (the choice the verification
finally made) and the moves it holds equal; 78.82's pick among them (the reasons' pick over the tied set, the first move first, the others in order of name: the player's own call, symbols_play.MoverReasons). The oracle at the setting of 78.94 (depth FUTURE_DEPTH, one line, one thread): the class, for the side that plays, of the position after each equal move
(the pick is one of them) and after the oracle's own move. A move KEEPS the class when its class is not lower than the stored class before it (prior_equal.keeps).

INTERFACE (written before the bodies).
  sample_fresh(recs, seen, cell, cap) -> [item]      as prior_equal.sample, with another hash seed and the positions in `seen` (ids of 78.96's sample) last in a cell; item gains "seen96".
  Stack()                                            per process: the human record's worths. `.search(moves) -> {"first", "equals", "depth", "error", "cut", "state"}`; `.label(moves, item, best) -> record` (the oracle is called here).
  label_record(...)                                  prior_equal.label_record plus {"pick", "pick_cls", "cut", "state", "seen96"} (pure; tested).
  run(tag, cell, cap, workers, say, limit)           the Pool over the cores, one file a chunk of CHUNK positions under <tag>/equals/ (a chunk with a file is not redone), then equals.jsonl; the count of oracle calls and the time printed.
  load(root) -> [record]                             for stack_measure.
  Called by: stack_measure.py (the tables, the cases), stack_causes.py (the room's moves), the command line (`label`).

  .venv/bin/python -m catspace.incubator.analyst.discovery.stack_room label --tag short --limit 300 | --tag full [--workers K]
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
from catspace.incubator.analyst.discovery import ladder_oracle as LO
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import record_play as RP
from catspace.io import paths

OUT = ("stack_79_00",)
SEED = "stack79"
CAP = 12                                                      # positions of one game in one cell at most (78.96's full run)
CELL = 1500                                                   # the cell quota of 78.96's full run
CHUNK = 10
RECORD = "human"                                              # the judge of the stack (record_play.RECORD_OF)
STATE_WORDS = {0: "off", 1: "held", 2: "replaced", 3: "failed", 4: "cut"}


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS stack {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


def order_key(ordinal: int, ply: int) -> int:
    return int(hashlib.md5(f"{SEED}:{ordinal}:{ply}".encode()).hexdigest(), 16)


def sample_fresh(recs: list, seen: set, cell: int = CELL, cap: int = CAP) -> list:
    """See the module doc. Only the CONFIRM part of the records is sampled."""
    cells: dict = {}
    for r in recs:
        if r["part"] != "confirm":
            continue
        for q in r["positions"]:
            it = {"ord": r["ord"], "part": r["part"], "ply": q["ply"], "units": q["units"], "band": PE.band_of(q["units"]), "cls": q["cls"], "seen96": f"{r['ord']}_{q['ply']}" in seen}
            cells.setdefault((q["cls"], it["band"]), []).append(it)
    out = []
    for key in sorted(cells):
        used: dict = {}
        n = 0
        for it in sorted(cells[key], key=lambda i: (i["seen96"], order_key(i["ord"], i["ply"]))):
            if n >= cell:
                break
            if used.get(it["ord"], 0) >= cap:
                continue
            used[it["ord"]] = used.get(it["ord"], 0) + 1
            out.append(it)
            n += 1
    return sorted(out, key=lambda i: order_key(i["ord"], i["ply"]))


def label_record(item: dict, search: dict, classes: list, best_move: "str | None", best_class: "int | None", cps: list, seconds: dict, pick: "str | None") -> dict:
    rec = PE.label_record(item, search, classes, best_move, best_class, cps, seconds)
    tied = rec["tied"]
    rec.update(pick=pick, pick_cls=(classes[tied.index(pick)] if pick in tied else None), cut=bool(search.get("cut")), state=search.get("state"), seen96=bool(item.get("seen96")))
    return rec


class Stack:
    """See the module doc."""

    def __init__(self):
        self.worth = RP.worth_of(RECORD)
        self._reader = None

    def search(self, moves: list) -> dict:
        import cs_engine as CE
        from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
        raw = CE.solve_equals_batch([(OM.VARIANT, None, list(moves))], TM.SECONDS, worth=dict(self.worth), ours_first=False, valuation="odds record", threads=1, ledger_share=TM.LEDGER_SHARE, verify_choice=True, choice_mode="slot")[0]
        row = PM.equal_set(raw)
        return {"first": row["move"], "equals": row["equals"], "depth": row["depth"], "error": row["error"], "cut": bool(row["cut"]), "state": int(raw[12] or 0)}

    def pick(self, moves: list, first: str, equals: list) -> str:
        from catspace.incubator.analyst.discovery import symbols_play as SP
        tied = [first] + sorted(m for m in equals if m != first)
        if len(tied) < 2:
            return first
        mv, _d = SP.reader82().pick(list(moves), tied, SP.MODE[SP.KIND["reasons"]])
        return mv

    def label(self, moves: list, item: dict, best: "str | None") -> "dict | None":
        t0 = time.perf_counter()
        s = self.search(moves)
        t1 = time.perf_counter()
        if s["error"] or not s["first"]:
            return None
        s = dict(s, first=s["first"], equals=s["equals"])
        pick = self.pick(moves, s["first"], s["equals"])
        tied = [s["first"]] + [m for m in s["equals"] if m != s["first"]]
        t2 = time.perf_counter()
        cls, cps = [], []
        for m in tied:
            a = LO.analyse_after(moves, m)
            cls.append(FL.class_of(a))
            cps.append(a["cp"] if a["forced_in"] is None else None)
        bc = None
        if best:
            bc = cls[tied.index(best)] if best in tied else FL.class_of(LO.analyse_after(moves, best))
        t3 = time.perf_counter()
        return label_record(item, s, cls, best, bc, cps, {"search": t1 - t0, "pick": t2 - t1, "oracle": t3 - t2}, pick)


_ST = None
_BEST: dict = {}


def _init_worker(best: dict) -> None:
    global _ST, _BEST
    _ST = Stack()
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
            r = _ST.label(moves, it, _BEST.get(f"{it['ord']}_{it['ply']}"))
        except Exception as e:                                      # a position the reading cannot take is counted, never silently
            r, bad = None, bad + 1
            (Path(root) / "equals" / f"c{idx:05d}.error").write_text(f"{type(e).__name__}: {str(e)[:300]}")
        if r is not None:
            out.append(r)
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0, bad


def load(root: Path) -> list:
    return [json.loads(x) for x in (root / "equals.jsonl").read_text().splitlines() if x.strip()]


def seen_ids() -> set:
    """The positions 78.96 sampled (read only)."""
    return {f"{r['ord']}_{r['ply']}" for r in PE.load(Path(paths.experiment(*PE.OUT, "full")))}


def run(tag: str, cell: int, cap: int, workers: int, say, limit: int = 0) -> None:
    root = out_root(tag)
    (root / "equals").mkdir(exist_ok=True)
    recs = [json.loads(x) for x in (Path(paths.experiment(*PE.SRC)) / "labels.jsonl").read_text().splitlines() if x.strip()]
    seen = seen_ids()
    items = sample_fresh(recs, seen, cell, cap)
    avail = sum(1 for r in recs if r["part"] == "confirm" for q in r["positions"] if f"{r['ord']}_{q['ply']}" not in seen)
    say(f"fresh sample: CONFIRM positions of the labels not sampled by 78.96: {avail}; the sample takes {len(items)} positions (cell quota {cell}, at most {cap} of one game in a cell, hash seed {SEED}), of them not sampled by 78.96 {sum(1 for i in items if not i['seen96'])}, sampled by it {sum(1 for i in items if i['seen96'])}")
    for band in ("few", "mid", "many"):
        say(f"sample coverage band {band}: " + ", ".join(f"class {c} {sum(1 for i in items if i['band'] == band and i['cls'] == c)}" for c in (1, 0, -1)))
    if limit:
        items = items[:limit]
    games = {g["ord"]: g for g in FL.universe()}
    best = {f"{r['ord']}_{q['ply']}": (q["pv"][0] if q.get("pv") else None) for r in recs if r["part"] == "confirm" for q in r["positions"]}
    say(f"to label: {len(items)} positions from {len({i['ord'] for i in items})} games; the search: {TM.SECONDS} s a position, ledger share {TM.LEDGER_SHARE}, record {RECORD} ({RP.RECORD_OF[RECORD]}), verify_choice on (slot); the oracle: depth {LO.FUTURE_DEPTH}, one line, one thread; {workers} workers; load average {os.getloadavg()[0]:.2f}")
    jobs = [(a // CHUNK, [(it, games[it["ord"]]["moves"][:it["ply"]]) for it in items[a:a + CHUNK]], str(root)) for a in range(0, len(items), CHUNK)]
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
    wall = time.perf_counter() - t0
    say(f"labelled {len(recs_out)} of {len(items)} positions in {wall:.0f} s wall (this run); ORACLE CALLS {calls}; one thread's seconds a position: search {sum(r['seconds']['search'] for r in recs_out) / n:.3f}, pick {sum(r['seconds']['pick'] for r in recs_out) / n:.3f}, "
        f"oracle {sum(r['seconds']['oracle'] for r in recs_out) / n:.3f}; oracle seconds in all {sum(r['seconds']['oracle'] for r in recs_out):.0f} (one thread's); equal moves a position {sum(len(r['tied']) for r in recs_out) / n:.2f}; positions with two or more equal moves {sum(len(r['tied']) >= 2 for r in recs_out)}; "
        f"the verification: " + ", ".join(f"{STATE_WORDS[k]} {sum(1 for r in recs_out if r['state'] == k)}" for k in (0, 1, 2, 3, 4)))


def cmd_cases(tag: str, say, rule) -> None:
    """The twenty judged positions as data (cases79.json): the positions `rule` picks from the stack's labels (stack_measure.pick_cases), each with the oracle's line after the search's first move and its class asked again (a sleep call, for the reader's judgement only; no engine reads it)."""
    root = out_root(tag)
    games = {g["ord"]: g for g in FL.universe()}
    chosen = rule(load(root))
    out = []
    for r in chosen:
        moves = games[r["ord"]]["moves"][:r["ply"]]
        a = LO.analyse_after(moves, r["first"])
        out.append({"id": r["id"], "moves": moves, "band": r["band"], "units": r["units"], "cls": r["cls"], "first": r["first"], "pick": r["pick"], "tied": r["tied"], "cm": r["cm"], "best": r["best"], "best_cls": r["best_cls"], "depth": r["depth"], "state": r["state"],
                    "pv_first": a["pv"][:12], "cls_again": FL.class_of(a), "forced_in": a["forced_in"]})
    (root / "cases79.json").write_text(json.dumps(out))
    say(f"{len(out)} judged positions (the room, the first move loses the class) written to {root / 'cases79.json'}; oracle calls {len(out)}")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["label"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--cell", type=int, default=CELL)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    run(a.tag, a.cell, a.cap, a.workers, Say(out_root(a.tag), "label"), a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
