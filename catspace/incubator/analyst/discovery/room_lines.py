"""PLAN 78.97 PARTS TWO AND THREE, THE ORACLE'S ANSWERS THE MEASURE NEEDS, AND WHAT KIND OF LOSS A STUCK MOVE IS (SYSTEM_SPEC 5 the line reading and the threat list, 9 the oracle as a measure and labeller in sleep, and the wall). A SLEEP STAGE, and the only file of 78.97 that calls the
oracle (through the wrapper ladder_oracle, as prior_equal.py of 78.96 does). A sandbox: nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/room_78_97/<tag>/. Game-free: a move or a unit is a word or a number read from the engine at run time; the game
is named only through the pieces that exist. No number of the oracle is an input of anything: its class is a label for counting and its line is a list of move words read by OUR line reading; the wake never reaches this file.

STAGE `classes` (part two). A pick of room_picks.py can be a move the 78.96 labels do not hold (the run that made the labels held another set). The oracle's class after each such move, for the side that plays it, at the setting of 78.94 (the same call as 78.96: `ladder_oracle.analyse_after`), so that every move a
pick plays has a class. Only picked moves are asked. The number of calls and the seconds are printed.
STAGE `lines` (part three). For every STUCK move (room_reads.stuck_of: a class-losing move of the labels that stays equal at five times the clock, at twenty-five times it and with the prunings off) the oracle's best line after the move (at most PV_PLIES moves, the first of them the other side's reply) and its class again
(the agreement with the stored class is printed). THE SPLIT, decided here before any count was looked at: the line is cut at the first move the engine's own rules refuse; the boards after each ply are read; the move is TACTICAL when, after the longest even number of plies of the line (the mover's own move
completes an exchange), the mover has lost more of the units than the other side, each unit in the measured record's worth of its kind (the judge's own table, no worth assumed: future_frames.material_net), or when the oracle gives a forced end against the mover inside the line; otherwise it is SLOW (a loss
the value's business, no unit lost within the six plies). For a tactical move, the reading of OUR side at the position after the move, for the other side's first reply of the line (the same three questions as 78.95 part one (b), asked of the reply):
    list     does the threat list of the position name the reply (a taking, or the quiet first move of a threat in two moves, or a forced end: future_labels.Labeller.sees)?
    line     does our line reading of the reply give the verdict CONFIRMED for the side that plays it (symbols_play.LineCut)?
    search   does our search at the play clock, run for the other side at that position, play the reply or hold it equal to its best?
THE KIND OF THE MISSING STATEMENT, the first that applies: `absent` (the list does not name the reply), `unconfirmed` (it names it and the line reading does not confirm the gain), `unheld` (named and confirmed and the search at the play clock does not hold the reply), `all present` (all three hold: our reading of the
other side's reply is there, and the move was held equal all the same).

INTERFACE (written before the bodies).
  is_tactical(net, forced_in, pv_n) -> bool      the split above on the net loss and the forced end (pure; tested).
  kind_of(seen, verdict, held) -> str             the kind of the missing statement (pure; tested).
  Reader()                                       per process: the labeller (reader, lines, record), the line cut, the search's worth, the unit names.
    .read_stuck(moves, move) -> record           {"move", "cls_again", "forced_in", "pv", "net", "tactical", "reply", "seen", "verdict", "held", "kind"} for one stuck move at the position `moves`.
  run_classes(tag, workers, say) / run_lines(tag, workers, say)    the Pool over all cores; one file a chunk (a chunk with a file is not redone); classes.json {"<id> <move>": class}, lines.jsonl.
  load_classes(root) -> {key: class}; load_lines(root) -> [record with "id"].
  Called by: room_measure.py (parts two and three, the cases), the command line.

  .venv/bin/python -m catspace.incubator.analyst.discovery.room_lines classes|lines --tag short | --tag full [--workers K]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import future_frames as FF
from catspace.incubator.analyst.discovery import future_goals as FG
from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import ladder_oracle as LO
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import room_picks as RP
from catspace.incubator.analyst.discovery import room_reads as RR
from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
from catspace.incubator.analyst.discovery import symbols_play as SP

PLIES = 6                                                  # the line read after the move, in plies (the oracle's stored line is at most LO.PV_PLIES)
CHUNK = 10
KINDS = ("absent", "unconfirmed", "unheld", "all present")


def key_of(pid: str, move: str) -> str:
    return f"{pid} {move}"


def is_tactical(net: int, forced_in: "int | None", pv_n: int) -> bool:
    """See the module doc: a net loss in the record's worths after the longest even number of plies, or a forced end against the mover inside the line (forced_in negative: the mover gets the end in that many of its own moves)."""
    if forced_in is not None and forced_in < 0 and 2 * (-forced_in) - 1 <= PLIES:
        return True
    return net > 0


def kind_of(seen: bool, verdict: "str | None", held: bool) -> str:
    if not seen:
        return "absent"
    if verdict != "CONFIRMED":
        return "unconfirmed"
    if not held:
        return "unheld"
    return "all present"


class Reader:
    """See the module doc."""

    def __init__(self):
        self.lab = FL.Labeller()
        self.cut = SP.LineCut(self.lab.record)
        self.sym = OM.kernel_names()
        self.worth = dict(OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")["worth"])
        self.names = None

    def worth_of(self, idx: int) -> int:
        return int(self.lab.record.worth.get(self.sym.get(self.names["kinds"][idx], ""), 0))

    def boards(self, wires: list) -> list:
        items = self.lab._items(wires, [len(w) for w in wires])
        self.names = self.lab.names
        return items

    def held(self, moves: list, reply: str) -> bool:
        import cs_engine as CE
        raw = CE.solve_equals_batch([(OM.VARIANT, None, list(moves))], TM.SECONDS, worth=dict(self.worth), ours_first=False, valuation="odds record", threads=1, ledger_share=TM.LEDGER_SHARE)[0]
        row = PM.equal_set(raw)
        return row["move"] == reply or reply in row["equals"]

    def read_stuck(self, moves: list, move: str) -> dict:
        after = list(moves) + [move]
        a = LO.analyse_after(moves, move)
        cls = FL.class_of(a)
        pv = self.lab.legal_prefix(after, a["pv"])[:PLIES]
        out = {"move": move, "cls_again": cls, "forced_in": a["forced_in"], "pv": pv, "net": 0, "tactical": False, "reply": pv[0] if pv else None, "seen": None, "verdict": None, "held": None, "kind": None}
        if not pv and a["forced_in"] is None:
            return out
        cut = len(pv) - (len(pv) % 2) if len(pv) >= 2 else len(pv)
        wires = [after + pv[:j] for j in (0, cut)] if cut else [after]
        items = self.boards(wires)
        if all(it is not None for it in items) and cut:
            root, last = items[0]["raw"], items[1]["raw"]
            us = 1 - root["mover"]                                          # the side that played the move; the side to move after it is the other
            out["net"] = FF.material_net(root["board"], last["board"], 1 - us, us, self.worth_of)
        out["tactical"] = is_tactical(out["net"], a["forced_in"], len(pv))
        if out["tactical"] and pv and items[0] is not None:
            lst = FG.goals_of([items[0]], self.lab.lines, self.lab.record)[0]["list"]
            out["seen"] = bool(FL.Labeller.sees(lst, pv[0], self.names["places"])["any"])
            v = self.cut.reads(after, [pv[0]]).get(pv[0])
            out["verdict"] = v["verdict"] if v else None
            out["held"] = bool(self.held(after, pv[0]))
            out["kind"] = kind_of(out["seen"], out["verdict"], out["held"])
        return out


_RD = None


def _init_worker() -> None:
    global _RD
    _RD = Reader()


def _class_job(args: tuple) -> tuple:
    idx, items, root = args
    f = Path(root) / "classes" / f"c{idx:05d}.json"
    if f.exists():
        return idx, 0.0, 0
    t0 = time.perf_counter()
    out = {}
    for pid, moves, m in items:
        out[key_of(pid, m)] = FL.class_of(LO.analyse_after(moves, m))
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0, len(items)


def _line_job(args: tuple) -> tuple:
    idx, items, root = args
    f = Path(root) / "lines" / f"c{idx:05d}.json"
    if f.exists():
        return idx, 0.0, 0
    t0 = time.perf_counter()
    out = []
    for pid, moves, m in items:
        try:
            r = _RD.read_stuck(moves, m)
        except Exception as e:                                          # a move the reading cannot take is counted, never silently
            r = {"move": m, "error": f"{type(e).__name__}: {str(e)[:150]}"}
        r["id"] = pid
        out.append(r)
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0, len(items)


def load_classes(root: Path) -> dict:
    out = {}
    for f in sorted((root / "classes").glob("c*.json")):
        out.update(json.loads(f.read_text()))
    return out


def load_lines(root: Path) -> list:
    out = []
    for f in sorted((root / "lines").glob("c*.json")):
        out += json.loads(f.read_text())
    return out


def needed_classes(reads: list, picks: dict, labels: dict) -> list:
    """The (id, move) pairs a pick plays that the labels hold no class for."""
    need, seen = [], set()
    for r in reads:
        have = set(labels[r["id"]]["tied"])
        for arm in RP.ARMS:
            p = picks.get(r["id"], {}).get(arm)
            if not p:
                continue
            for m in {p["first"], p["pattern"], p["prior"]}:
                if m not in have and (r["id"], m) not in seen:
                    seen.add((r["id"], m))
                    need.append((r["id"], m))
    return need


def _run(tag: str, workers: int, say, stage: str) -> None:
    root = RR.out_root(tag)
    (root / ("classes" if stage == "classes" else "lines")).mkdir(exist_ok=True)
    reads = RR.load(root)
    labels = {r["id"]: r for r in PE.load(RR.labels_root())}
    games = {g["ord"]: g for g in FL.universe()}
    if stage == "classes":
        need = needed_classes(reads, RP.load(root), labels)
        items = [(pid, games[labels[pid]["ord"]]["moves"][:labels[pid]["ply"]], m) for pid, m in need]
        job, init = _class_job, None
    else:
        items = []
        for r in reads:
            lab = labels[r["id"]]
            for m in RR.stuck_of(lab, r["rows"]):
                items.append((r["id"], games[lab["ord"]]["moves"][:lab["ply"]], m))
        job, init = _line_job, _init_worker
    jobs = [(a // CHUNK, items[a:a + CHUNK], str(root)) for a in range(0, len(items), CHUNK)]
    say(f"{len(items)} oracle calls asked (depth {LO.FUTURE_DEPTH}, one line, one thread); {workers} workers; load average {os.getloadavg()[0]:.2f}")
    t0 = time.perf_counter()
    done = 0
    with multiprocessing.get_context("spawn").Pool(workers, initializer=init) as pool:
        for _i, _dt, _n in pool.imap_unordered(job, jobs, chunksize=1):
            done += 1
            if done % 25 == 0 or done == len(jobs):
                say(f"progress: {done} of {len(jobs)} chunks; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    say(f"{stage}: {len(items)} oracle calls in {time.perf_counter() - t0:.0f} s wall on {workers} workers")


def run_classes(tag: str, workers: int, say) -> None:
    _run(tag, workers, say, "classes")


def run_lines(tag: str, workers: int, say) -> None:
    _run(tag, workers, say, "lines")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["classes", "lines"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    (run_classes if a.cmd == "classes" else run_lines)(a.tag, a.workers, RR.Say(RR.out_root(a.tag), a.cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
