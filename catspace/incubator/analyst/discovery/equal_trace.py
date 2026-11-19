"""PLAN 78.98 PART ONE, THE TRACE OF AN EQUAL MOVE AND THE COUNT OF CAUSES (SYSTEM_SPEC 4 the tactical reasoner and its equal reading, 5 the value (the judge's worths are the measured record), 12 the debt of plan 78.84; piece: tactical reasoning).
A MEASURE and nothing else: no player is changed, no switch is turned on in play, nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/equal_78_98/<tag>/. Game-free: a move is a word, a unit a kind word the engine
hands back; the game is named only through the pieces that exist (cs_engine, the 78.96 labels, the 78.97 reads). The oracle is not called: 78.96's stored class labels are used for counting only.

THE RULE OF EQUALITY AS THE CODE HAS IT (rust/engine/tactics/src/ledger.rs `Search::ties`, the binding's `EqualsLedger`): the deepening finishes a depth D with the value v of the position; every root move is then asked "at least v at depth D?" by a null window, with the table of the
deepening warm and the late-move reduction and the null move on (`probes`); a move is EQUAL iff its answer is v or more. The comparison is exact (margin zero, integers of the judge's sum); the judge is the measured record's worths (`Worth`, additive over the takings of the line), and the line
is resolved at its horizon by a run of takings (`quiet`). The value of a root move is its gain plus the value of the future after it.

WHAT IS ASKED AT EACH STUCK MOVE (the stuck moves of 78.97 whose missing statement is "all present", and the named positions of the plan):
  at D = the depth of the play clock's run (78.97's r1 row), by `cs_engine.equal_audit` with the deepening stopped at D and a node budget alone (no clock, so the result repeats): v; for the stuck move and one class-keeping equal move the move's gain, its warm answer (the rule's own
  question), its answer on a fresh table with and without the two prunings, its exact value (full window); and whether the rule holds it. At D and D+DEEP, by `cs_engine.choice_audit` naming the stuck move: the value of that move and the best, on the table with both prunings off (and with both on, for the reader).
  The line each value comes from: `cs_engine.read_lines` mode 9 (the line that starts with the move, the best replies, resolved to a settled end): its words, its ledger of takings, its value, and at its settled end the units each side has by kind (the estimator's inputs: `graph_layer1`'s position
  table). The estimator itself is not part of the equal reading: the equal reading's value is the ledger's; the inputs are printed for the reader of the trace.
THE CAUSES, decided here before any count was looked at (`cause_of`, the first that applies; a move the audit at D does not hold equal is `not held` and is not counted among them):
  (b) the rule holds the move but an exact reading does not: the move's exact value, or its answer with both prunings off, is under v, or the reading with both prunings off at the same depth D puts the move under ITS best (a bound, a window or a
      pruned reading stands in for the value, or the value v itself is too low);
  (c) the values at D are equal and the longer reading (the prunings off, D+DEEP plies) puts the move under the best: the line under the move ends before the loss is on the board;
  (a) the values at D and at D+DEEP are both at the best: the judge's value does not tell the move from the best, at any depth read;
  unread: a reading the node budget stopped, so none of the three can be told.
INTERFACE (written before the bodies).
  cause_of(v, row, deep) -> str                   the cause above from the move's audit row {"warm","plain","exact"} and the deep reading {"other","best"} (pure; tested).
  exchange_of(events, changes) -> dict            {"gave": [kind], "took": [kind], "changed": [(from, to)]} from a line read's ledger, the mover's side first (pure; tested).
  Tracer()                                        per process: the record's worths and the graph model. `.trace(moves, depth, stuck, keeper) -> dict` the whole record of one position.
  run(tag, workers, say) / load(root)             the Pool over the cores (untimed: node budgets), one file a chunk (a chunk with a file is not redone); trace.jsonl.
  run_records(tag, workers, say) / load_records(root)   STAGE `records` (a diagnostic of cause (a), not a mend: nothing in the engine reads it): the equal reading at D of each stuck position again under each of the measured records of the kinds (the odds record the search plays
                                                  with, the self-play record, the human-result record: threat_concrete.Record), the answer of the rule for every move of the label's equal set; records.jsonl.
  Called by: equal_measure.py (the counts), ladder_print --equal98 (the print), the command line.

  .venv/bin/python -m catspace.incubator.analyst.discovery.equal_trace trace|records --tag short --limit 12 | --tag full [--workers K]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import room_reads as RR
from catspace.io import paths

OUT = ("equal_78_98",)
LINES = ("room_78_97", "full", "lines")                # read only
CASES = (("room_78_97", "full", "cases97.json"), ("prior_78_96", "full", "cases96.json"))
NAMED = ((0, 5), (0, 6), (0, 9), (0, 13), (1, 1))      # (file, the case's number in it): the five positions of the plan
RUN_NODES = 40_000_000                                 # the deepening's fuse (it stops at D long before)
TIE_NODES = 30_000_000                                 # the equal reading's fuse
BIG = 30_000_000                                       # the fuse of each fresh reading
DEEP = 2                                               # the plies the longer reading adds
LINE_FUSE = 400_000
CHUNK = 4
CAUSES = ("not held", "b", "c", "a", "unread")
RECORDS = ("gen2", "selfplay_78_58", "human_78_59")      # the measured records of the kinds: the odds games' (the judge in play), the self-play results', the human results'


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS equal {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


def cause_of(v: int, row: "dict | None", deep: "dict | None") -> str:
    """See the module doc."""
    if row is None or row.get("warm") is None or row["warm"] < v:
        return "not held"
    shown = [x for x in (row.get("exact"), row.get("plain")) if x is not None]
    same_depth = deep is not None and deep.get("other_d") is not None and deep.get("best_d") is not None and deep["other_d"] < deep["best_d"]
    if any(x < v for x in shown) or same_depth:
        return "b"
    if row.get("exact") is None or row.get("plain") is None:
        return "unread"
    if deep is None or deep.get("other") is None or deep.get("best") is None:
        return "unread"
    return "c" if deep["other"] < deep["best"] else "a"


def exchange_of(events: list, changes: list) -> dict:
    """The units the mover gave and took along a read line (events: (is the mover's, kind)), and the kind changes ((is the mover's, from, to))."""
    return {"gave": sorted(k for mine, k in events if mine), "took": sorted(k for mine, k in events if not mine),
            "changed": [(a, b) for mine, a, b in changes if mine]}


def named_cases() -> list:
    """The five positions of the plan: the case records of 78.97 (5, 6, 9, 13) and 78.96 (1), each with the stuck move and the moves of the position."""
    files = [json.loads(Path(paths.experiment(*c)).read_text()) for c in CASES]
    return [dict(files[f][n - 1], source=CASES[f][0], number=n) for f, n in NAMED]


def stuck_all_present() -> list:
    """[{"id", "move"}]: the stuck moves of 78.97 whose missing statement is `all present`, in the order of the files."""
    root = Path(paths.experiment(*LINES))
    out = []
    for f in sorted(root.glob("c*.json")):
        out += [{"id": r["id"], "move": r["move"], "reply": r.get("reply")} for r in json.loads(f.read_text()) if r.get("kind") == "all present"]
    return out


class Tracer:
    """See the module doc."""

    def __init__(self):
        self.worth = dict(OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")["worth"])
        self.layer1 = GL.model(OM.VARIANT)

    def audit(self, moves: list, depth: int, max_depth: int) -> "dict | None":
        return CE.equal_audit(OM.VARIANT, list(moves), self.worth, RUN_NODES, TIE_NODES, max_depth, BIG, True)

    def line(self, moves: list, move: str, depth: int) -> dict:
        """The line that starts with `move` (mode 9, read for the side that plays it) and its settled end's units by kind."""
        res = CE.read_lines(OM.VARIANT, [(None, list(moves) + [move])], [(0, 9, 0, 1, 0, 0, 0)], self.worth, depth, LINE_FUSE, 7, 1500000, 1, True)[0]
        if not res.get("ok") or not res.get("reads"):
            return {"ok": False}
        rd = res["reads"][0]
        k = int(rd["settled"])
        leaf = list(moves) + list(rd["names"][:k])
        names, raw = self.layer1.position_table([(None, leaf)], 1)
        nk = len(names)
        import numpy as np
        t = np.frombuffer(raw, dtype=np.int16).reshape(len(leaf) + 1, 2 + 2 * nk)[-1]
        sides = (CE.state(OM.VARIANT, None, list(moves))[1], CE.state(OM.VARIANT, None, leaf)[1])               # (the side to move at the root, and at the leaf: the inputs are listed for the leaf's mover first)
        inputs = {n: [int(t[2 + i]), int(t[2 + nk + i])] for i, n in enumerate(names)}            # [the side to move at the leaf, the other side]
        return {"ok": True, "names": list(rd["names"]), "settled": k, "value": int(rd["value"]), "end": int(rd["end"]), "cut": bool(rd["cut"]), "depth": int(rd["depth"]),
                "exchange": exchange_of([tuple(e) for e in rd["events"]], [tuple(c) for c in rd["changes"]]), "root_mover": sides[0], "leaf_mover": sides[1], "inputs": inputs}

    def trace(self, moves: list, depth: int, stuck: str, keeper: "str | None") -> dict:
        t0 = time.perf_counter()
        out = {"depth": depth, "stuck": stuck, "keeper": keeper}
        a = self.audit(moves, depth, depth)
        if a is None:
            return dict(out, error="no audit")
        v = int(a["value"])
        rows = {r["name"]: r for r in a["rows"]}
        out.update(value=v, finished=int(a["depth"]), tie_cut=bool(a["tie_cut"]), equal_set=len(a["first"]), verified_set=len(a["verified"]), first_move=a.get("first"))
        names = list(rows)
        out["held"] = sorted(rows[n]["name"] for n in names if rows[n]["warm_probe"] >= v)

        def keep_row(n):
            r = rows.get(n)
            return None if r is None else {"gain": r["gain"], "warm": r["warm_probe"], "cold": r["cold_probe"], "plain": r["plain_probe"], "exact": r["exact"], "no_null": r["no_null_probe"], "no_late": r["no_late_probe"], "held": r["warm_probe"] >= v}
        out["stuck_row"], out["keeper_row"] = keep_row(stuck), keep_row(keeper) if keeper else None
        deep = None
        try:
            ca = CE.choice_audit(OM.VARIANT, list(moves), self.worth, RUN_NODES, depth, BIG, DEEP, True, stuck)
        except Exception as e:
            ca = None
            out["deep_error"] = f"{type(e).__name__}: {str(e)[:100]}"
        if ca:
            rd = {(r["depth"], r["name"]): r for r in ca["readings"]}
            at_d, at_deep = rd.get((int(ca["depth"]), "plain")), rd.get((int(ca["depth"]) + DEEP, "plain"))
            ap = {r["depth"]: r for r in ca["readings"] if r["name"] == "as_played"}
            deep = {"depth": int(ca["depth"]), "other": at_deep and at_deep["other"], "best": at_deep and at_deep["best"], "other_d": at_d and at_d["other"], "best_d": at_d and at_d["best"],
                    "pruned_other": (ap.get(int(ca["depth"]) + DEEP) or {}).get("other"), "pruned_best": (ap.get(int(ca["depth"]) + DEEP) or {}).get("best")}
        out["deep"] = deep
        out["cause"] = cause_of(v, out["stuck_row"], deep)
        out["stuck_line"] = self.line(moves, stuck, depth)
        if keeper:
            out["keeper_line"] = self.line(moves, keeper, depth)
        out["seconds"] = time.perf_counter() - t0
        return out


_TR = None


def _init_worker() -> None:
    global _TR
    _TR = Tracer()


def _job(args: tuple) -> tuple:
    idx, items, root = args
    f = Path(root) / "trace" / f"c{idx:05d}.json"
    if f.exists():
        return idx, 0.0
    t0 = time.perf_counter()
    out = []
    for it in items:
        try:
            r = _TR.trace(it["moves"], it["depth"], it["stuck"], it["keeper"])
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        out.append(dict(r, id=it["id"], band=it["band"], tag=it.get("tag")))
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0


def items_of(limit: int = 0, named_only: bool = False) -> list:
    """The positions to trace: the named five first, then the stuck moves of kind all present (one item per stuck move; the keeper is the first class-keeping equal move of the labels that the move's position holds at the clock)."""
    labels = {r["id"]: r for r in PE.load(RR.labels_root())}
    reads = {r["id"]: r for r in RR.load(RR.out_root("full"))}
    games = {g["ord"]: g for g in FL.universe()}
    items = []
    for c in named_cases():
        keeper = next((m for m, k in zip(c["tied"], c["cm"]) if PE.keeps(c["cls"], k)), None)
        if c["source"] == "prior_78_96":
            keeper = c["prior"]
        items.append({"id": c["id"], "moves": c["moves"], "depth": reads[c["id"]]["rows"]["r1"]["depth"], "stuck": c["move"] if "move" in c else c["first"], "keeper": keeper, "band": c["band"], "tag": f"named {c['source']} {c['number']}"})
    if named_only:
        return items
    for s in stuck_all_present():
        r, rd = labels.get(s["id"]), reads.get(s["id"])
        if r is None or rd is None or rd["rows"]["r1"].get("depth") is None:
            continue
        held = set(rd["rows"]["r1"]["equals"]) | {rd["rows"]["r1"]["first"]}
        keepers = [m for m, k in zip(r["tied"], RR.keepvec(r)) if k]
        keeper = next((m for m in keepers if m in held), keepers[0] if keepers else None)
        items.append({"id": s["id"], "moves": games[r["ord"]]["moves"][:r["ply"]], "depth": rd["rows"]["r1"]["depth"], "stuck": s["move"], "keeper": keeper, "band": r["band"], "tag": "all present"})
    return items[:limit] if limit else items


def run(tag: str, limit: int, workers: int, say, named_only: bool = False) -> None:
    root = out_root(tag)
    (root / "trace").mkdir(exist_ok=True)
    items = items_of(limit, named_only)
    say(f"{len(items)} stuck moves to trace ({sum(1 for i in items if i['tag'].startswith('named'))} named); D the play clock's depth of 78.97's r1 row; node budgets alone (no clock); {workers} workers of {os.cpu_count()} cores; load average {os.getloadavg()[0]:.2f}")
    jobs = [(a // CHUNK, items[a:a + CHUNK], str(root)) for a in range(0, len(items), CHUNK)]
    t0 = time.perf_counter()
    done = 0
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for _i, _dt in pool.imap_unordered(_job, jobs, chunksize=1):
            done += 1
            if done % 5 == 0 or done == len(jobs):
                say(f"progress: {done} of {len(jobs)} chunks; {time.perf_counter() - t0:.0f} s")
    out = []
    for f in sorted((root / "trace").glob("c*.json")):
        out += json.loads(f.read_text())
    (root / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    say(f"traced {len(out)} of {len(items)} in {time.perf_counter() - t0:.0f} s wall")


def load(root: Path) -> list:
    return [json.loads(x) for x in (root / "trace.jsonl").read_text().splitlines() if x.strip()]


def _records_job(args: tuple) -> tuple:
    idx, items, root = args
    f = Path(root) / "records" / f"c{idx:05d}.json"
    if f.exists():
        return idx, 0.0
    from catspace.incubator.analyst.discovery import threat_concrete as TC
    t0 = time.perf_counter()
    out = []
    for it in items:
        row = {"id": it["id"], "band": it["band"], "tied": it["tied"], "keep": it["keep"], "depth": it["depth"], "records": {}}
        for name in RECORDS:
            try:
                w = dict(TC.Record.load(name).worth)
                a = CE.equal_audit(OM.VARIANT, it["moves"], w, RUN_NODES, TIE_NODES, it["depth"], BIG, True)
                if a is None:
                    row["records"][name] = {"error": "no audit"}
                    continue
                v = int(a["value"])
                warm = {r["name"]: r["warm_probe"] for r in a["rows"]}
                row["records"][name] = {"value": v, "tie_cut": bool(a["tie_cut"]), "held": [m for m in it["tied"] if warm.get(m) is not None and warm[m] >= v], "set": len(a["first"]), "worth": w}
            except Exception as e:
                row["records"][name] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
        out.append(row)
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0


def run_records(tag: str, workers: int, say) -> None:
    """STAGE `records`: for each position that has a stuck move of kind all present, the rule's answer at D for every move of the labels' equal set under each record. Positions, not moves, so a position is read once."""
    root = out_root(tag)
    (root / "records").mkdir(exist_ok=True)
    labels = {r["id"]: r for r in PE.load(RR.labels_root())}
    reads = {r["id"]: r for r in RR.load(RR.out_root("full"))}
    games = {g["ord"]: g for g in FL.universe()}
    ids = []
    for s in stuck_all_present():
        if s["id"] not in ids and s["id"] in labels and s["id"] in reads and reads[s["id"]]["rows"]["r1"].get("depth") is not None:
            ids.append(s["id"])
    items = [{"id": i, "band": labels[i]["band"], "tied": labels[i]["tied"], "keep": RR.keepvec(labels[i]), "depth": reads[i]["rows"]["r1"]["depth"], "moves": games[labels[i]["ord"]]["moves"][:labels[i]["ply"]]} for i in ids]
    say(f"{len(items)} positions with a stuck move of kind all present, read under {len(RECORDS)} records of the kinds; {workers} workers; load average {os.getloadavg()[0]:.2f}")
    jobs = [(a // CHUNK, items[a:a + CHUNK], str(root)) for a in range(0, len(items), CHUNK)]
    t0 = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for _i, _dt in pool.imap_unordered(_records_job, jobs, chunksize=1):
            pass
    out = []
    for f in sorted((root / "records").glob("c*.json")):
        out += json.loads(f.read_text())
    (root / "records.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    say(f"read {len(out)} positions in {time.perf_counter() - t0:.0f} s wall")


def load_records(root: Path) -> list:
    return [json.loads(x) for x in (root / "records.jsonl").read_text().splitlines() if x.strip()]


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["trace", "records"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--named", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "records":
        run_records(a.tag, a.workers, Say(out_root(a.tag), "records"))
        return 0
    run(a.tag, a.limit, a.workers, Say(out_root(a.tag), "trace"), a.named)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
