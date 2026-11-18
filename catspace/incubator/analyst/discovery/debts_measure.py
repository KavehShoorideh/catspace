"""PLAN 78.85 THE TWO DEBTS OF THE REASONER. SYSTEM_SPEC 5 (the tactical reasoner: its equal reading and its line verdict), 11, 12. Game-free: the game is named only through the pieces that exist.

INTERFACE (what each stage takes, what it returns, who calls it; written before the bodies).
  testset    takes the stored reads of plan 78.84 (read only) and returns testset.json: the decisions whose cause there was BELOW BEST (the move's value at the search's own depth under the search's value: the
             equal reading holds it at a best it is not at), each as the moves played, the depth and value the search had, and the pair of moves.
  audit      takes testset.json (or decisions.json with --all) and, per decision, asks cs_engine.equal_audit (rust/engine/tactics equal_audit: a fresh table and a node limit instead of a clock, so the result repeats) for
             every root move's answer to "at least the best at depth D?" in four readings and its value at depth D in two (see equal_audit.rs); writes audit.jsonl (a rerun skips the rows done).
  report     takes audit.jsonl and prints OBS lines: how many of the test set are still wrong (a tied move whose exact value is under the best), which reading removes the fault, the tie set sizes.
  ties       (before / after) takes decisions.json, plays each position again through solve_equals_batch at a node limit alone (no clock) and prints the mean size of the tie set, the nodes and the depth.
  verdict    (part two) takes the stored flagged decisions of 78.84 and reads each bad move's line by mode 9 with the verdict as it was and as it is now; prints how often events and value disagree.
Called by: the command line only.

  PYTHONHASHSEED=0 .venv/bin/python -m catspace.incubator.analyst.discovery.debts_measure testset | audit | report | ties | verdict [--tag full]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

import cs_engine as CE

from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import tie_cut_measure as TCM
from catspace.io import paths

OUT = ("reasoner_debts_78_85",)
SOURCE = ("symbols_play_78_84", "full")                                 # read only
RUN_NODES = 150_000                                                   # the deepening's node limit (about a 0.2 s decision at the match's share: the stored searches spent 110 to 130 thousand)
TIE_NODES = 30_000_000                                                 # the equal reading's node limit
BIG = 8_000_000                                                     # the fuse of the fresh readings
MAX_DEPTH = 64


def out_root(tag: str = "full") -> Path:
    return Path(paths.experiment(*OUT, tag))


def worth() -> dict:
    return dict(OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")["worth"])


def source_rows() -> tuple:
    root = Path(paths.experiment(*SOURCE))
    dec = {d["key"]: d for d in json.loads((root / "decisions.json").read_text())}
    rows = [json.loads(x) for x in (root / "reads.jsonl").read_text().splitlines() if x.strip()]
    return dec, rows


# ------------------------------------------------------------------------------------------------------------------------------ testset

def testset(say) -> list:
    dec, rows = source_rows()
    out = []
    for r in rows:
        if not (r.get("tied") and r.get("bad") and r.get("good")):
            continue
        if TCM.cause_of(r) != "BELOW BEST":
            continue
        d = dec[r["key"]]
        out.append({"key": r["key"], "band": r["band"], "units": r["units"], "ply": r["ply"], "moves": d["moves"], "depth84": r["search"]["depth"], "value84": r["search"]["value"],
                    "tied84": r["tied"], "bad": r["pair"]["bad"], "good": r["pair"]["good"], "search_move": r["search"]["move"]})
    out.sort(key=lambda x: x["key"])
    say(f"TESTSET {len(out)} decisions whose cause in plan 78.84 was BELOW BEST (of {len(rows)} read); by band " + ", ".join(f"{b} {sum(1 for x in out if x['band'] == b)}" for b in ("few", "mid", "many")))
    return out


# ------------------------------------------------------------------------------------------------------------------------------ audit

_W: dict = {}


def audit_one(d: dict) -> dict:
    if "worth" not in _W:
        _W["worth"] = worth()
    t0 = time.perf_counter()
    a = CE.equal_audit(OM.VARIANT, list(d["moves"]), _W["worth"], RUN_NODES, TIE_NODES, MAX_DEPTH, BIG)
    return {"key": d["key"], "band": d["band"], "units": d["units"], "bad": d.get("bad"), "good": d.get("good"), "audit": a, "wall": time.perf_counter() - t0}


def audit_chunk(chunk: list) -> list:
    return [audit_one(d) for d in chunk]


def run_audit(rows: list, workers: int, out_file: Path, say) -> None:
    done = set()
    if out_file.exists():
        done = {json.loads(x)["key"] for x in out_file.read_text().splitlines() if x.strip()}
    todo = [d for d in rows if d["key"] not in done]
    chunks = [todo[i:i + 4] for i in range(0, len(todo), 4)]
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    n = len(done)
    with open(out_file, "a") as f, ctx.Pool(workers) as pool:
        for res in pool.imap_unordered(audit_chunk, chunks):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n += len(res)
    say(f"AUDIT done: {len(rows)} decisions in {time.perf_counter() - t0:.0f} s, {workers} workers")


# ------------------------------------------------------------------------------------------------------------------------------ report

READINGS = ("warm_probe", "cold_probe", "no_null_probe", "no_late_probe")


def tied_of(a: dict, key: str) -> list:
    """The moves a reading holds equal: its probe is the best (the search's value) or more."""
    v = a["value"]
    return [r for r in a["rows"] if r[key] is not None and r[key] >= v]


def faults_of(a: dict, key: str) -> list:
    """The moves a reading holds equal that the reference (plain probe: alpha-beta and the table alone, a fresh table, no forward pruning) does not: its probe is under the best. Moves the reference did not finish are not counted."""
    v = a["value"]
    return [r["name"] for r in tied_of(a, key) if r["plain_probe"] is not None and r["plain_probe"] < v]


def report(tag: str, say, name: str = "audit.jsonl") -> None:
    root = out_root(tag)
    rows = [json.loads(x) for x in (root / name).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["audit"] is not None]
    n = len(rows)
    say(f"REPORT {n} decisions audited; depth D mean {np.mean([r['audit']['depth'] for r in rows]):.1f}; nodes of the deepening mean {np.mean([r['audit']['run_nodes'] for r in rows]):.0f}; "
        f"nodes of the equal reading (warm) mean {np.mean([r['audit']['tie_nodes'] for r in rows]):.0f}; nodes of the reference mean {np.mean([r['audit']['plain_nodes'] for r in rows]):.0f}; equal reading cut by the fuse {sum(1 for r in rows if r['audit']['tie_cut'])}")
    unfinished = sum(1 for r in rows if any(x["plain_probe"] is None for x in r["audit"]["rows"]))
    say(f"REPORT the reference (plain probe) did not finish within {BIG} nodes for some root move in {unfinished} of {n} decisions (those moves are not counted)")
    for k in READINGS:
        bad = [r for r in rows if faults_of(r["audit"], k)]
        moves = sum(len(faults_of(r["audit"], k)) for r in rows)
        say(f"REPORT {k}: tie set holds a move the reference puts under the best in {len(bad)} of {n} decisions ({moves} moves); mean tie set size {np.mean([len(tied_of(r['audit'], k)) for r in rows]):.2f}")
    sizes = [len([x for x in r["audit"]["rows"] if x["plain_probe"] is not None and x["plain_probe"] >= r["audit"]["value"]]) for r in rows]
    say(f"REPORT the reference's own tie set: mean size {np.mean(sizes):.2f}")
    # the new `ties` (the first reading, then each candidate asked again on a fresh table with the late-move reduction off) and the same with the late-move reduction off and the null move on (the first, lighter variant)
    for label, key, nodes in (("verified (the new ties: late-move reduction and null move off)", "verified", "verify_nodes"), ("verified with the late-move reduction off only", "verified_late", "verify_late_nodes")):
        bad = 0
        moves = 0
        sizes_v = []
        for r in rows:
            a = r["audit"]
            ref = {x["index"]: x["plain_probe"] for x in a["rows"]}
            wrong = [i for i in a[key] if ref.get(i) is not None and ref[i] < a["value"]]
            bad += 1 if wrong else 0
            moves += len(wrong)
            sizes_v.append(len(a[key]))
        lost = sum(1 for r in rows for x in r["audit"]["rows"] if x["plain_probe"] is not None and x["plain_probe"] >= r["audit"]["value"] and x["index"] not in r["audit"][key])
        say(f"REPORT {label}: tie set holds a move the reference puts under the best in {bad} of {n} decisions ({moves} moves); mean tie set size {np.mean(sizes_v):.2f} (the first reading {np.mean([len(r['audit']['first']) for r in rows]):.2f}); "
            f"moves the reference holds at the best that this set leaves out {lost}; stopped by the node limit in {sum(1 for r in rows if r['audit'].get('verify_cut')) if key == 'verified' else 0} decisions; nodes of this reading mean {np.mean([r['audit'][nodes] for r in rows]):.0f} (the first reading {np.mean([r['audit']['tie_nodes'] for r in rows]):.0f})")
    warm_bad = [r for r in rows if faults_of(r["audit"], "warm_probe")]
    gone_null = [r for r in warm_bad if not faults_of(r["audit"], "no_null_probe")]
    gone_late = [r for r in warm_bad if not faults_of(r["audit"], "no_late_probe")]
    say(f"REPORT of the {len(warm_bad)} decisions with a fault in the warm reading: the fault is gone with the null move off in {len(gone_null)}, gone with the late reduction off in {len(gone_late)}")


# ------------------------------------------------------------------------------------------------------------------------------ ties before and after

TIES_EVERY = 3                                                       # the fixed set: every third of the 3600 decisions of plan 78.84 (400 a band, hash order inside a band)


def ties_chunk(chunk: list) -> list:
    out = []
    for d in chunk:
        r = TCM.search_row(d["moves"], TM.SECONDS, TM.LEDGER_SHARE)
        out.append({"key": d["key"], "band": d["band"], "move": r["move"], "equals": r["equals"], "depth": r["depth"], "value": r["value"], "nodes": r["nodes"], "seconds": r["seconds"], "ties_seconds": r["ties_seconds"],
                    "cut": r["cut"], "legal": r["legal"]})
    return out


def run_ties(rows: list, workers: int, out_file: Path, say) -> None:
    chunks = [rows[i:i + 6] for i in range(0, len(rows), 6)]
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with open(out_file, "w") as f, ctx.Pool(workers) as pool:
        for res in pool.imap_unordered(ties_chunk, chunks):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
    say(f"TIES done: {len(rows)} positions in {time.perf_counter() - t0:.0f} s, {workers} workers")


def report_ties(tag: str, say) -> None:
    """Before (the extension of main: the equal reading as it was) and after (this build) on the same positions, each played at the match's clock (0.2 s, ledger share 0.4): the mean size of the equal set, the nodes and depth of the
    search, the seconds and the clock-cut share of the equal reading; and how many of the moves the old set held are not in the new one."""
    root = out_root(tag)
    b = {r["key"]: r for r in (json.loads(x) for x in (root / "ties_before.jsonl").read_text().splitlines() if x.strip())}
    a = {r["key"]: r for r in (json.loads(x) for x in (root / "ties_after.jsonl").read_text().splitlines() if x.strip())}
    keys = sorted(set(a) & set(b))
    for band in ("few", "mid", "many", "all"):
        ks = [k for k in keys if band == "all" or a[k]["band"] == band]
        if not ks:
            continue

        def mean(rows: dict, f) -> float:
            return float(np.mean([f(rows[k]) for k in ks]))
        dropped = sum(len(set(b[k]["equals"]) - set(a[k]["equals"])) for k in ks)
        added = sum(len(set(a[k]["equals"]) - set(b[k]["equals"])) for k in ks)
        say(f"TIES {band}: {len(ks)} positions; mean equal set {mean(b, lambda r: len(r['equals'])):.2f} before, {mean(a, lambda r: len(r['equals'])):.2f} after; moves of the old set not in the new {dropped}, new not in old {added}; "
            f"depth of the ledger {mean(b, lambda r: r['depth'] or 0):.2f} before, {mean(a, lambda r: r['depth'] or 0):.2f} after; nodes of the decision {mean(b, lambda r: r['nodes']):.0f} before, {mean(a, lambda r: r['nodes']):.0f} after; "
            f"seconds of the equal reading {mean(b, lambda r: r['ties_seconds']):.3f} before, {mean(a, lambda r: r['ties_seconds']):.3f} after; equal reading cut by the clock {sum(1 for k in ks if b[k]['cut'])} before, {sum(1 for k in ks if a[k]['cut'])} after; "
            f"equal set of one move {sum(1 for k in ks if len(b[k]['equals']) == 1)} before, {sum(1 for k in ks if len(a[k]['equals']) == 1)} after")


# ------------------------------------------------------------------------------------------------------------------------------ part two: the line verdict

def verdict_chunk(chunk: list) -> list:
    """The lines of the bad and the good move of each flagged decision of 78.84, read again by the line reading as it is now (mode 9, the settings of the cut); the verdict as verdict_of gives it."""
    from catspace.incubator.analyst.discovery import symbols_play as SP
    out = []
    cut = SP.cutter()
    for d in chunk:
        tied = [d["bad"], d["good"]]
        v = cut.reads(d["moves"], tied)
        out.append({"key": d["key"], "band": d["band"], "bad": d["bad"], "good": d["good"], "after": {m: (None if v[m] is None else {k: v[m][k] for k in ("verdict", "end", "left_lost", "left_won", "net", "net_search", "resolution", "pv_disagrees", "kind_net", "unsettled", "depth", "full_names")}) for m in tied}})
    return out


def run_verdict(rows: list, workers: int, out_file: Path, say) -> None:
    chunks = [rows[i:i + 8] for i in range(0, len(rows), 8)]
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with open(out_file, "w") as f, ctx.Pool(workers) as pool:
        for res in pool.imap_unordered(verdict_chunk, chunks):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
    say(f"VERDICT done: {len(rows)} flagged decisions in {time.perf_counter() - t0:.0f} s, {workers} workers")


def flagged_rows() -> list:
    dec, rows = source_rows()
    out = []
    for r in rows:
        if r.get("tied") and r.get("bad") and r.get("good"):
            out.append({"key": r["key"], "band": r["band"], "moves": dec[r["key"]]["moves"], "bad": r["pair"]["bad"], "good": r["pair"]["good"], "before": r["lines"]})
    return sorted(out, key=lambda x: x["key"])


def report_verdict(tag: str, say) -> None:
    """Before (the stored reads of 78.84: events off the printed line, the net without a kind change) and after (the line read on through its tail, the net with the kind change), on the flagged decisions: how often the events
    of the bad move's line disagree with the line search's own value (`pv_disagrees`), how often the bad move still gives up a unit by the literal rule (events alone) and by the rule that also asks the line's own value."""
    from catspace.incubator.analyst.discovery import symbols_play as SP
    root = out_root(tag)
    fl = {r["key"]: r for r in flagged_rows()}
    after = [json.loads(x) for x in (root / "verdict.jsonl").read_text().splitlines() if x.strip()]
    for band in ("few", "mid", "many", "all"):
        rs = [a for a in after if band == "all" or a["band"] == band]
        if not rs:
            continue
        n = len(rs)
        b_dis = sum(1 for a in rs if fl[a["key"]]["before"][a["bad"]]["pv_dis"])
        a_dis = sum(1 for a in rs if a["after"][a["bad"]] and a["after"][a["bad"]]["pv_disagrees"])

        def give_before(a: dict, own: bool) -> bool:
            c = fl[a["key"]]["before"][a["bad"]]
            return SP.gives_up({"verdict": c["verdict"], "end": c["end"], "left_lost": c["lost"], "left_won": c["won"], "net_search": c["net_search"], "resolution": c["res"]}, own)
        b_lit = sum(1 for a in rs if give_before(a, False))
        b_own = sum(1 for a in rs if give_before(a, True))
        a_lit = sum(1 for a in rs if SP.gives_up(a["after"][a["bad"]], False))
        a_own = sum(1 for a in rs if SP.gives_up(a["after"][a["bad"]], True))
        uns = sum(1 for a in rs if a["after"][a["bad"]] and a["after"][a["bad"]]["unsettled"])
        kn = sum(1 for a in rs if a["after"][a["bad"]] and a["after"][a["bad"]]["kind_net"] != 0)
        same_v = sum(1 for a in rs if a["after"][a["bad"]] and a["after"][a["bad"]]["net_search"] == fl[a["key"]]["before"][a["bad"]]["net_search"])
        say(f"VERDICT {band}: {n} flagged decisions; the bad move's line: events and the line's own value disagree before {b_dis}, after {a_dis}; gives up a unit by the events alone before {b_lit}, after {a_lit}; "
            f"by the events and the line's own value before {b_own}, after {a_own}; line search's own number the same in {same_v} of {n}; kind change in the net {kn}; unsettled {uns}")
        bad_after = [a for a in rs if SP.gives_up(a["after"][a["bad"]], False)]
        good_after = [a for a in rs if a["after"][a["good"]] and SP.gives_up(a["after"][a["good"]], False)]
        say(f"VERDICT {band}: the good move's line gives up a unit by the events alone after: {len(good_after)}; decisions where the bad move still gives up and the good does not (still flagged) after: {sum(1 for a in bad_after if a not in good_after)}")


# ------------------------------------------------------------------------------------------------------------------------------ the run

def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("testset", "audit", "report", "ties", "ties_report", "verdict", "verdict_report"))
    ap.add_argument("--label", default="after")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--file", default="audit.jsonl")
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)
    obs = root / "obs.txt"

    def say(line: str) -> None:
        TM.emit([f"OBS debts {line}"], obs)
    if a.stage == "testset":
        rows = testset(say)
        (root / "testset.json").write_text(json.dumps(rows))
    elif a.stage == "audit":
        rows = json.loads((root / "testset.json").read_text())
        if a.n:
            rows = rows[:a.n]
        load, busy = TM.wait_quiet(lambda s: TM.emit([s], obs))
        say(f"AUDIT start: load average {load:.2f}, cores busy {busy:.2f}; deepening limit {RUN_NODES} nodes, equal reading {TIE_NODES} nodes, fresh readings {BIG} nodes")
        run_audit(rows, a.workers, root / a.file, say)
    elif a.stage == "report":
        report(a.tag, say, a.file)
    elif a.stage == "ties":
        dec = json.loads((Path(paths.experiment(*SOURCE)) / "decisions.json").read_text())
        rows = dec[::TIES_EVERY]
        load, busy = TM.wait_quiet(lambda s: TM.emit([s], obs))
        say(f"TIES {a.label} start: load average {load:.2f}, cores busy {busy:.2f}; {len(rows)} positions (every {TIES_EVERY}rd of the 3600 decisions of plan 78.84), {TM.SECONDS} s, ledger share {TM.LEDGER_SHARE}")
        run_ties(rows, a.workers, root / f"ties_{a.label}.jsonl", say)
    elif a.stage == "ties_report":
        report_ties(a.tag, say)
    elif a.stage == "verdict":
        rows = flagged_rows()
        load, busy = TM.wait_quiet(lambda s: TM.emit([s], obs))
        say(f"VERDICT start: load average {load:.2f}, cores busy {busy:.2f}; {len(rows)} flagged decisions of plan 78.84")
        run_verdict(rows, a.workers, root / "verdict.jsonl", say)
        report_verdict(a.tag, say)
    elif a.stage == "verdict_report":
        report_verdict(a.tag, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
