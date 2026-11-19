"""PLAN 78.99 PART ONE, THE TABLES (SYSTEM_SPEC 4 the equal reading, 5 the value and its measured records; piece: tactical reasoning). Reads reads.jsonl of record_reads.py (artifacts/experiments/record_78_99/<tag>/) and the labels of 78.96 (read only) and prints OBS lines;
writes nothing else. No oracle, no engine call, no number put in by hand.

STAGE `table`, per record (gen2 read again = the noise control; selfplay_78_58; human_78_59), for FIND and CONFIRM apart and by band (few, mid, many units), each with a 90 pct band by games resampled whole:
  IN THE ROOM   the share of class-losing moves of the labels' equal set that leave the equal set of the record's reading, the share of class-keeping ones that leave, the share of positions where the search's first move keeps the class (positions whose first
                move has no class are counted apart and left out of the share), the share of positions where the first move differs from the odds record's first move of the same run.
  OUTSIDE THE ROOM (the price)   of the positions not in the room where the odds record's first move keeps the class: the share where the record's first move keeps it, loses it, has no class (counted apart).
  A reading that was cut by the clock, or failed, has no answer and is counted apart, never read as a move leaving.
INTERFACE. cmd_table(tag, say); the counting functions are pure and tested: counts(recs, name, ref) -> dict of lists per band and part. Called by the command line only.

  .venv/bin/python -m catspace.incubator.analyst.discovery.record_measure table --tag short | full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json

import numpy as np

from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import prior_measure as PMS
from catspace.incubator.analyst.discovery import record_reads as RDS
from catspace.incubator.analyst.discovery import room_reads as RR

BANDS = ("few", "mid", "many")
PARTS = ("find", "confirm")
REF = "gen2"
SEED = 9999


def share(vals, game) -> tuple:
    return PMS.share(np.asarray(vals, float), np.asarray(game), SEED)


def fmt(t: tuple) -> str:
    return PMS.fmt(t)


def answered(row: "dict | None") -> bool:
    return RR.answered(row or {})


def room_counts(reads: list, labels: dict, name: str) -> dict:
    """Per room position that answered under `name`: the losing and keeping moves that left, the first move's keeping (None without a class), and whether the first move differs from the reference record's. -> {"lose": [(game, band, part, left)], "keep": [...],
    "first": [(game, band, part, keeps)], "differs": [(game, band, part, differs)], "cut": n, "noclass": n}."""
    out = {"lose": [], "keep": [], "first": [], "differs": [], "cut": 0, "noclass": 0}
    for r in reads:
        if not r["room"]:
            continue
        row, ref, lab = r["rows"][name], r["rows"][REF], labels[r["id"]]
        if not answered(row):
            out["cut"] += 1
            continue
        key = (r["ord"], r["band"], r["part"])
        lv = RDS.leaves(lab, row)
        out["lose"] += [key + (x,) for x in lv["lose"]]
        out["keep"] += [key + (x,) for x in lv["keep"]]
        k = RDS.first_keeps(lab, row["first"])
        if k is None:
            out["noclass"] += 1
        else:
            out["first"].append(key + (k,))
        if answered(ref):
            out["differs"].append(key + (row["first"] != ref["first"],))
    return out


def price_counts(reads: list, labels: dict, name: str) -> dict:
    """Outside the room, among the positions where the reference record's first move keeps the class: the record's first move keeps it ("keeps"), loses it ("loses") or has no class ("none"); positions that did not answer are "cut"."""
    out = {"keeps": [], "loses": [], "none": [], "cut": 0, "base": 0}
    for r in reads:
        if r["room"]:
            continue
        row, ref, lab = r["rows"][name], r["rows"][REF], labels[r["id"]]
        if not (answered(row) and answered(ref)):
            out["cut"] += 1
            continue
        if RDS.first_keeps(lab, ref["first"]) is not True:
            continue
        out["base"] += 1
        key = (r["ord"], r["band"], r["part"])
        k = RDS.first_keeps(lab, row["first"])
        out["none"].append(key + (k is None,))
        out["keeps"].append(key + (k is True,))
        out["loses"].append(key + (k is False,))
    return out


def _sel(rows: list, band: "str | None", part: "str | None") -> tuple:
    use = [x for x in rows if (band is None or x[1] == band) and (part is None or x[2] == part)]
    return np.array([x[3] for x in use], float), np.array([x[0] for x in use])


def cell(rows: list, band: "str | None" = None, part: "str | None" = None) -> str:
    v, g = _sel(rows, band, part)
    return f"{fmt(share(v, g))} ({int(v.sum())} of {len(v)})" if len(v) else "n/a (0)"


def cmd_table(tag: str, say) -> None:
    root = RDS.out_root(tag)
    reads = RDS.load(root)
    labels = {r["id"]: r for r in PE.load(RR.labels_root())}
    n_room = sum(r["room"] for r in reads)
    say(f"read {len(reads)} labelled positions (in the room {n_room}, outside {len(reads) - n_room}); play clock {TM.SECONDS} s, ledger share {TM.LEDGER_SHARE}; records {', '.join(RDS.RECORDS)}; shares are percent with a 90 pct band by games (the label's game) resampled whole")
    for name in RDS.RECORDS:
        errs = sum(1 for r in reads if (r["rows"][name].get("error")))
        cuts = sum(1 for r in reads if r["rows"][name].get("cut"))
        say(f"{name}: readings with an error {errs}, cut by the clock {cuts} (counted apart, never read as a move leaving); mean depth {np.mean([r['rows'][name]['depth'] for r in reads if r['rows'][name].get('depth') is not None]):.2f}")
    for name in RDS.RECORDS:
        c = room_counts(reads, labels, name)
        say(f"ROOM under {name}: positions cut or failed {c['cut']}; first move without a class {c['noclass']} of {n_room - c['cut']}")
        for what, label in (("lose", "class-losing equal moves that leave the equal set"), ("keep", "class-keeping equal moves that leave the equal set"), ("first", "positions where the first move keeps the class"),
                            ("differs", "positions where the first move differs from the odds record's first move of the same run")):
            lines = [f"all {cell(c[what])}"]
            lines += [f"{b} {cell(c[what], b)}" for b in BANDS]
            lines += [f"{p} {cell(c[what], None, p)}" for p in PARTS]
            lines += [f"{p}/{b} {cell(c[what], b, p)}" for p in PARTS for b in BANDS]
            say(f"ROOM under {name}: {label}: " + "; ".join(lines))
    for name in RDS.RECORDS:
        c = price_counts(reads, labels, name)
        say(f"OUTSIDE THE ROOM under {name}: positions where the odds record's first move keeps the class: {c['base']} (cut or failed: {c['cut']})")
        for what, label in (("keeps", "the record's first move keeps the class"), ("loses", "the record's first move loses the class"), ("none", "the record's first move has no class (the oracle was not called)")):
            lines = [f"all {cell(c[what])}"]
            lines += [f"{b} {cell(c[what], b)}" for b in BANDS]
            lines += [f"{p} {cell(c[what], None, p)}" for p in PARTS]
            say(f"OUTSIDE THE ROOM under {name}: {label}: " + "; ".join(lines))


CASES_ROOM = 10


def room_cases(reads: list, n: int = CASES_ROOM) -> list:
    """THE RULE FOR THE TEN ROOM POSITIONS (fixed before any was read): in the order of the hash of 78.96, the room positions where the first move under the human-result record differs from the first move under the odds record of the same run, both readings answered; the first `n`."""
    out = []
    for r in reads:
        h, o = r["rows"].get("human_78_59"), r["rows"].get(REF)
        if r["room"] and answered(h) and answered(o) and h["first"] != o["first"]:
            out.append(r)
        if len(out) == n:
            break
    return out


def cmd_cases(tag: str, say) -> None:
    """STAGE `cases`: the judged positions as data (cases99.json): the five positions of 78.98 and the ten room positions above, with the first move, the equal set and the depth under each record, the label's class of each first move and the oracle's own move as the label has it
    (read, not called). ladder_print `--record99 TAG` writes them out with the boards. The search is read here at the play clock, one thread, in this process."""
    from catspace.incubator.analyst.discovery import equal_trace as ET
    from catspace.incubator.analyst.discovery import future_labels as FL
    root = RDS.out_root(tag)
    labels = {r["id"]: r for r in PE.load(RR.labels_root())}
    games = {g["ord"]: g for g in FL.universe()}
    reads = RDS.load(root)
    rd = RDS.Reads()
    items = [(c["id"], f"named {c['source']} {c['number']}") for c in ET.named_cases()] + [(r["id"], "room, the first move changes") for r in room_cases(reads)]
    out = []
    for pid, tg in items:
        lab = labels[pid]
        moves = games[lab["ord"]]["moves"][:lab["ply"]]
        rows = rd.read(moves)
        out.append({"id": pid, "tag": tg, "moves": moves, "band": lab["band"], "cls": lab["cls"], "label_first": lab["first"], "tied": lab["tied"], "cm": lab["cm"], "best": lab.get("best"), "best_cls": lab.get("best_cls"),
                    "rows": rows, "class_of_first": {n: RDS.class_of(lab, rows[n]["first"]) for n in rows}})
    (root / "cases99.json").write_text(json.dumps(out))
    say(f"{len(out)} judged positions (five of 78.98, {len(out) - 5} room positions where the first move changes between the odds record and the human-result record) written to {root / 'cases99.json'}")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["table", "cases"])
    ap.add_argument("--tag", default="short")
    a = ap.parse_args(argv)
    {"table": cmd_table, "cases": cmd_cases}[a.stage](a.tag, RDS.Say(RDS.out_root(a.tag), a.stage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
