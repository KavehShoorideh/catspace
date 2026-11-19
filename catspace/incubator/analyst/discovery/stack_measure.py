"""PLAN 79.00 PART TWO, THE TABLES (SYSTEM_SPEC 4 the equal reading and the verified choice, 5 the measured records, 7 the pick among equal moves; piece: tactical reasoning). Reads the labels of stack_room.py (equals.jsonl under artifacts/experiments/stack_79_00/<tag>/) and 78.96's
(prior_78_96/full, read only: equals.jsonl, picks.json) and prints OBS lines; writes nothing but the cases file. It reads no oracle number beyond the stored classes and makes no engine call; the one oracle call (the line after a first move, for the twenty judged positions) is stack_room.cmd_cases.

THE SHARES, all over positions (a stratified sample: the shares are those of the sample, not of games), by band, with a 90 pct band by games resampled whole:
  two or more equal moves; the equal moves DIFFER IN CLASS (the classes after them are not all the same); the equal moves differ in the class KEPT (the room: some keep it and some do not); NO EQUAL MOVE KEEPS it; the search's first move KEEPS it; the pick keeps it
  (the stack's pick is 78.82's, over the set the verified search hands back; 78.96's is the same pick over the unverified odds-record set: picks.json `pattern`; with one equal move the pick is that move). A move KEEPS the class when its class is not lower than the class before it.
  Also in the room alone: the first move and the pick keep it; and the search's part (no equal move keeps it, the oracle's move does).
THE DATASETS: the stack (fresh sample), the stack on the part 78.96 did not sample (`seen96` false), 78.96's CONFIRM part, and 78.96 / the stack each on the positions the two samples share (the paired comparison).
THE CASES: twenty positions of the new room where the first move loses the class (RULE, fixed before any was read: in the order of the sample's hash, the room positions of the stack's labels where the first move does not keep the class, the first twenty), written to cases79.json by stack_room `cases`.

INTERFACE.
  metrics(recs, picks) -> dict of arrays   the per-position vectors above (pure; tested).
  cmd_table(tag, say)                      the OBS lines.  cmd_pick_cases(tag) -> [record]   the rule for the twenty.
  Called by: the command line, ladder_print --stack79 (reads cases79.json).

  .venv/bin/python -m catspace.incubator.analyst.discovery.stack_measure table|cases --tag short|full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import prior_measure as PMS
from catspace.incubator.analyst.discovery import stack_room as SR
from catspace.io import paths

BANDS = ("few", "mid", "many")
OLD = ("prior_78_96", "full")


def metrics(recs: list, picks: dict) -> dict:
    """Per-position vectors over `recs` (label records). `picks`: {id: pick} for the records without a `pick` of their own (78.96's pattern picks); a record with one equal move picks it."""
    keep = [PMS.keepvec(r) for r in recs]
    out = {"game": np.array([r["ord"] for r in recs]), "band": np.array([r["band"] for r in recs]), "ids": [r["id"] for r in recs]}
    out["ge2"] = np.array([len(r["tied"]) >= 2 for r in recs], float)
    out["differ_class"] = np.array([len(set(r["cm"])) > 1 for r in recs], float)
    out["room"] = np.array([bool(k.any() and not k.all()) for k in keep], float)
    out["nokeep"] = np.array([not k.any() for k in keep], float)
    out["first"] = np.array([bool(k[0]) for k in keep], float)

    def pick_of(r):
        p = r.get("pick") or picks.get(r["id"])
        return p if p in r["tied"] else r["tied"][0]
    out["pick"] = np.array([float(k[r["tied"].index(pick_of(r))]) for r, k in zip(recs, keep)])
    out["searchs"] = np.array([float((not k.any()) and r["best_cls"] is not None and PE.keeps(r["cls"], r["best_cls"])) for r, k in zip(recs, keep)])
    out["size"] = np.array([len(r["tied"]) for r in recs], float)
    return out


def cell(m: dict, key: str, sel: np.ndarray) -> str:
    if int(sel.sum()) < 30:
        return "too few"
    return PMS.fmt(PMS.share(m[key][sel], m["game"][sel]))


def table_lines(name: str, recs: list, picks: dict) -> list:
    if not recs:
        return [f"TABLE {name}: no positions"]
    m = metrics(recs, picks)
    out = []
    for b in ("all",) + BANDS:
        sel = np.ones(len(recs), bool) if b == "all" else (m["band"] == b)
        room = sel & (m["room"] > 0)
        out.append(f"TABLE {name} {b}: positions {int(sel.sum())}; two or more equal moves {cell(m, 'ge2', sel)}; the equal moves differ in CLASS {cell(m, 'differ_class', sel)}; differ in the class KEPT (the room) {cell(m, 'room', sel)}; NO EQUAL MOVE KEEPS the class {cell(m, 'nokeep', sel)}; "
                   f"the search's first move keeps it {cell(m, 'first', sel)}; the pick keeps it {cell(m, 'pick', sel)}; the search's part (no equal move keeps it, the oracle's move does) {cell(m, 'searchs', sel)}; in the room alone ({int(room.sum())} positions): "
                   f"first {cell(m, 'first', room)}, pick {cell(m, 'pick', room)} (pct, 90 pct bands, games resampled); mean equal moves {m['size'][sel].mean():.2f}")
    return out


def old_picks() -> dict:
    return {k: v["pattern"] for k, v in json.loads((Path(paths.experiment(*OLD)) / "picks.json").read_text()).items()}


def cmd_table(tag: str, say) -> None:
    root = SR.out_root(tag)
    new = SR.load(root)
    old = [r for r in PE.load(Path(paths.experiment(*OLD))) if r["part"] == "confirm"]
    picks = old_picks()
    say(f"TABLES: the stack's labels {len(new)} positions (fresh sample of the CONFIRM part: human record, verified choice, 78.82's pick); 78.96's CONFIRM labels {len(old)} positions (odds record, no verification, the pattern pick of 78.82 over its equal set); "
        f"the stack's positions 78.96 did not sample {sum(1 for r in new if not r['seen96'])}, sampled {sum(1 for r in new if r['seen96'])}")
    st = {k: sum(1 for r in new if r["state"] == k) for k in (1, 2, 3, 4, 0)}
    say(f"the verification over the stack's {len(new)} positions: held {st[1]} ({100 * st[1] / max(1, len(new)):.1f} pct), REPLACED the choice {st[2]} ({100 * st[2] / max(1, len(new)):.1f} pct), failed {st[3]}, cut {st[4]} ({100 * st[4] / max(1, len(new)):.1f} pct), none {st[0]}; the equal reading cut by the clock in {sum(1 for r in new if r['cut'])} ({100 * sum(1 for r in new if r['cut']) / max(1, len(new)):.1f} pct); "
        f"mean ledger depth {np.mean([r['depth'] for r in new if r['depth'] is not None]):.2f}")
    for name, recs in (("STACK all", new), ("STACK fresh part", [r for r in new if not r["seen96"]]), ("78.96 CONFIRM all", old)):
        for x in table_lines(name, recs, picks):
            say(x)
    shared = {r["id"] for r in new if r["seen96"]}
    old_ids = {r["id"] for r in old}
    both = shared & old_ids
    for name, recs in (("STACK on the shared positions", [r for r in new if r["id"] in both]), ("78.96 on the shared positions", [r for r in old if r["id"] in both])):
        for x in table_lines(name, recs, picks):
            say(x)
    m_new = metrics([r for r in new if r["id"] in both], picks)
    old_by = {r["id"]: r for r in old}
    m_old = metrics([old_by[i] for i in m_new["ids"]], picks)
    for key in ("first", "pick", "room", "nokeep", "differ_class"):
        d = m_new[key] - m_old[key]
        say(f"PAIRED on the {len(d)} shared positions, {key}: stack less 78.96 {PMS.fmt(PMS.share(d, m_new['game']), plus=True)} points (the first move of the stack's verified human-record search against the first move of the unverified odds-record search, both 0.2 s)")


def pick_cases(recs: list, n: int = 20) -> list:
    """THE RULE FOR THE TWENTY POSITIONS (fixed before any was read): in the order of the sample (the hash), the positions of the room (some equal move keeps the class, some does not) where the search's first move does not keep it; the first `n`."""
    out = []
    for r in recs:
        k = PMS.keepvec(r)
        if k.any() and not k.all() and not k[0]:
            out.append(r)
        if len(out) == n:
            break
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["table", "cases"])
    ap.add_argument("--tag", default="short")
    a = ap.parse_args(argv)
    if a.stage == "table":
        cmd_table(a.tag, SR.Say(SR.out_root(a.tag), "table"))
    else:
        SR.cmd_cases(a.tag, SR.Say(SR.out_root(a.tag), "cases"), pick_cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
