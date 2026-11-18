"""PLAN 78.92 DOES THE SEARCH'S OWN CHOICE CARRY THE PRUNING FAULT. SYSTEM_SPEC 5 (the tactical reasoner: the ledger search, its two forward prunings) and 12 (the row "the search's equal reading and the line
reading's verdict", the open debt). Game-free: the game is named only through the pieces that exist (the engine's variant name, the record's worth table, the stored positions); the board and the words of a case are
printed by ladder_print only. A sandbox: no player is changed, nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/reasoner_choice_78_92/<tag>/. The reference is the
engine's own search with the prunings off; no outside engine, oracle or labeller number; no worth of any unit is assumed (the numbers are in the judge's own units, the record's table); no player id is written.

INTERFACE (what each stage takes and writes; who calls it; written before the bodies). Each stage is a command and prints OBS lines (also to obs.txt).
  select   takes the 600 positions of plan 78.91 (order_78_91/full/positions.json, read only) and plan 78.90's stored count (symbols_78_90/full: left_s1.npy, left_s2.npy, with the samples of 78.80 and 78.89), and
           writes positions.json: set "p600" (the 600, as they are) and set "flag" (the human CONFIRM positions where the first move of the tie set is flagged "leaves a unit of the mover's borne on and
           unguarded that was not so before", FLAG_PER_BAND = 100 a band, evenly spaced in the pooled order of the two samples; each carries `other`, that first move).
  audit    takes positions.json and asks cs_engine.choice_audit (rust/engine/tactics choice_audit.rs: a node limit and a fresh table, so the result repeats) for each position: the search as it plays (both prunings on,
           RUN_NODES nodes), and the readings plain (both off, THE REFERENCE), no_null, no_late, as_played at the depth D it finished and at D + 2: best value, best move, the played move's value, the value of `other`, the
           nodes; and the played move asked again without the two prunings (the verification a repair would do). Writes audit.jsonl (a rerun skips the rows done). A Pool over all cores, one thread a worker.
  report   takes audit.jsonl and prints OBS lines. (a) the search as it plays: depth, nodes; (b) by band and pooled, for each reference reading and for depth D and D + 2: how often the played move's value is under the
           reference's best (the SHARE, a 90 pct bootstrap band over positions), the size of the shortfall in the judge's units (mean and median among those), how often the reference two plies deeper agrees (the played
           move is under the best there too), the node cost of each reading, and what the verification would cost and reject; (c) for the flagged first moves: how many are under the reference's best (a real loss) and
           how many are level (an exchange). Writes the condition (below) to condition_<audit file>.json.
  paired   takes two audit files of the same positions, without and with the verification of the choice (`audit --repair`), and prints how many positions leave or enter the state "played move under the plain best at D".
  cases    JUDGE: twenty positions of the p600 where the played move is under the plain reference's best at depth D, by a fixed rule (CASES_RULE), cases92.json for ladder_print `--choice92`.
  Matches (the PLAY part) are choice_play.py's.
Called by: the command line only; ladder_print.py reads cases92.json (`--choice92`).

  .venv/bin/python -m catspace.incubator.analyst.discovery.choice_measure select | audit | report | cases --tag short|full

HOW THE CONDITION FOR THE VERIFICATION SWITCH IS READ (written before any result was looked at). The plan builds `verify_choice` ONLY IF (b) shows the played move under the reference's best in a share clearly larger than
78.84's share for the equal reading, AND play does not punish the repair. 78.84 printed 81 BELOW BEST decisions of 3600 (2.3 pct). Read here as numbers, all on the p600 set pooled over its three bands, plain reading at
the depth D the search finished, positions where the reading finished only:
  (1) the share under is at least 0.046 (twice 78.84's 2.3 pct, "clearly larger") and the lower end of its 90 pct bootstrap band is above 0.023;
  (2) play does not punish the repair: NOT (in the match "as now" against "both off", pooled sets A and B, the lower end of as-now's band is above 0.5) -- i.e. the prunings do not clearly buy strength at the match's clock --
      OR the median node cost of the verification of the played move alone is under 0.10 of the nodes of the search as it plays (a repair that cheap cannot be told from noise in a depth).
The switch is built if (1) and (2) both hold; otherwise it is not, and the row in section 12 is closed as measured or reopened by what (1) shows. If (b) is cut by the reference's fuse in more than a fifth of the p600,
that is printed and (1) is read on the positions that finished, said so.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

import cs_engine as CE

from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import symbols_own_measure as SO
from catspace.incubator.analyst.discovery import symbols_sample_measure as SS
from catspace.io import paths

OUT = ("reasoner_choice_78_92",)
SRC91 = ("order_78_91", "full")                                   # read only
SRC90 = ("symbols_78_90", "full")                                 # read only
RUN_NODES = 150_000                                               # the deepening's node limit (about a 0.2 s decision at the match's share; 78.85 used the same)
BIG = 4_000_000                                                   # the fuse of each fresh reference reading
EXTRA = 2                                                         # plies the deeper reference adds
MAX_DEPTH = 64
FLAG_PER_BAND = 100
BANDS = ("few", "mid", "many")
READINGS = ("plain", "no_null", "no_late", "as_played")
SEED = 9292
DECIDED = 1 << 39                                                 # a shortfall this large means a reading sees the game end (the ledger's WIN is 2^40): not a count of units
SHARE_FLOOR = 0.046                                               # twice 78.84's 81 of 3600
SHARE_BASE = 0.023
VERIFY_CHEAP = 0.10
CASES_N = 20
CASES_RULE = ("THE RULE (fixed before any position was read): of the p600, the positions where the played move is under the plain reference's best at the depth the search finished (both finished), "
              "in each band the ones at evenly spaced places of the band's list in position order, 7 few, 7 mid, 6 many, twenty in all.")


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS choice92 {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


def worth() -> dict:
    return dict(OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")["worth"])


# ------------------------------------------------------------------------------------------------------------------------------ select

def flagged_positions() -> list:
    """The human CONFIRM positions of 78.80's and 78.89's samples whose first tied move leaves a unit loose (78.90's stored flag), in the pooled order, with that first move as `other`."""
    out = []
    for no in (1, 2):
        root = SS.src80() if no == 1 else SO.root_of(2)
        sample = SS.load_sample(root)
        left = np.load(Path(paths.experiment(*SRC90)) / f"left_s{no}.npy")
        start = 0
        for r in sample:
            n = len(r["tied"])
            if r["part"] == "confirm" and r.get("source", "human") == "human" and n and left[start]:
                out.append({"sample": no, "band": LM.band_of(int(r["units"])), "units": int(r["units"]), "game": r["game"], "ply": r["ply"], "id": r["id"], "moves": list(r["moves"]), "other": r["tied"][0]})
            start += n
        assert start == len(left)
    return out


def stage_select(root: Path, say, limit: int = 0) -> None:
    p600 = json.loads((Path(paths.experiment(*SRC91)) / "positions.json").read_text())
    pos = []
    for p in p600:
        pos.append({"set": "p600", "pid": p["pid"], "band": p["band"], "units": p["units"], "id": p["id"], "moves": p["moves"], "other": None})
    flag = flagged_positions()
    for b in BANDS:
        rows = [r for r in flag if r["band"] == b]
        n = limit or FLAG_PER_BAND
        ix = np.unique(np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int))
        for i in ix:
            r = rows[int(i)]
            pos.append({"set": "flag", "pid": len([x for x in pos if x["set"] == "flag"]), "band": b, "units": r["units"], "id": r["id"], "moves": r["moves"], "other": r["other"]})
        say(f"SELECT flag band {b}: {len(rows)} human CONFIRM positions whose first tied move leaves a unit loose (78.90's flag), {len(ix)} taken evenly spaced in the pooled order")
    if limit:
        pos = [x for x in pos if x["set"] == "flag" or x["pid"] % max(1, 600 // (3 * limit)) == 0][:6 * limit]
    (root / "positions.json").write_text(json.dumps(pos))
    say(f"SELECT written: {len(pos)} positions ({sum(1 for x in pos if x['set'] == 'p600')} of the 600, {sum(1 for x in pos if x['set'] == 'flag')} flagged)")


# ------------------------------------------------------------------------------------------------------------------------------ audit

_W: dict = {}


def audit_one(p: dict) -> dict:
    if "worth" not in _W:
        _W["worth"] = worth()
    t0 = time.perf_counter()
    a = CE.choice_audit(OM.VARIANT, list(p["moves"]), _W["worth"], p["run_nodes"], MAX_DEPTH, p["big"], EXTRA, True, p["other"], p["repair"])
    return {"set": p["set"], "pid": p["pid"], "band": p["band"], "units": p["units"], "id": p["id"], "other": p["other"], "audit": a, "wall": time.perf_counter() - t0}


def audit_chunk(chunk: list) -> list:
    return [audit_one(p) for p in chunk]


def stage_audit(root: Path, say, limit: int = 0, run_nodes: int = RUN_NODES, big: int = BIG, every: int = 1, name: str = "audit.jsonl", repair: bool = False) -> None:
    """`every` > 1 takes every that-many-th of the 600 (by position number) and no flagged position (the second setting: the play's own node limit)."""
    pos = [dict(p, run_nodes=run_nodes, big=big, repair=repair) for p in json.loads((root / "positions.json").read_text()) if every == 1 or (p["set"] == "p600" and p["pid"] % every == 0)]
    out_file = root / name
    done = set()
    if out_file.exists():
        done = {(r["set"], r["pid"]) for r in map(json.loads, filter(str.strip, out_file.read_text().splitlines()))}
    todo = [p for p in pos if (p["set"], p["pid"]) not in done]
    chunks = [todo[i:i + 3] for i in range(0, len(todo), 3)]
    t0 = time.perf_counter()
    n = len(done)
    with open(out_file, "a") as f, mp.get_context("spawn").Pool(min(9, os.cpu_count() or 1)) as pool:
        for res in pool.imap_unordered(audit_chunk, chunks):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n += len(res)
            if n % 60 < 3:
                say(f"progress {n} of {len(pos)} positions, {time.perf_counter() - t0:.0f} s")
    say(f"AUDIT done: {len(todo)} positions in {time.perf_counter() - t0:.0f} s; run nodes {run_nodes}, reference fuse {big}, deeper reading +{EXTRA} plies; load average {os.getloadavg()[0]:.2f}")


# ------------------------------------------------------------------------------------------------------------------------------ report

def load_rows(root: Path, name: str = "audit.jsonl") -> list:
    return [r for r in map(json.loads, filter(str.strip, (root / name).read_text().splitlines())) if r["audit"] is not None]


def reading(a: dict, name: str, deep: bool) -> dict:
    d = a["depth"] + (EXTRA if deep else 0)
    return next(x for x in a["readings"] if x["name"] == name and x["depth"] == d)


def short_of(a: dict, name: str, deep: bool) -> "int | None":
    """The reference's best less the played move's value at that depth (judge's units; 0 or more), None where the fuse cut the reading."""
    r = reading(a, name, deep)
    if r["best"] is None or r["played"] is None:
        return None
    return int(r["best"] - r["played"])


def boot(x: np.ndarray, draws: int = 1000, seed: int = SEED) -> "tuple[float, float, float]":
    """The mean of a 0/1 vector and its 90 pct bootstrap band over positions."""
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(draws)])
    return float(x.mean()), float(np.percentile(m, 5)), float(np.percentile(m, 95))


def cell(rows: list, band: str) -> list:
    return rows if band == "all" else [r for r in rows if r["band"] == band]


def share_line(rows: list, name: str, deep: bool) -> "tuple[str, dict]":
    sh = [(r, short_of(r["audit"], name, deep)) for r in rows]
    fin = [(r, s) for r, s in sh if s is not None]
    under = np.array([s > 0 for _r, s in fin], float)
    m, lo, hi = boot(under)
    sizes = np.array([s for _r, s in fin if 0 < s < DECIDED], float)
    ends = sum(1 for _r, s in fin if s >= DECIDED)
    txt = (f"finished {len(fin)} of {len(rows)}; played move under the best: {int(under.sum())} = {100 * m:.1f} pct [{100 * lo:.1f}, {100 * hi:.1f}]; of them the shortfall is a game's end (a reading sees it end) in {ends} and units in {len(sizes)}"
           + (f", units: shortfall in the judge's units mean {sizes.mean():.0f}, median {np.median(sizes):.0f}, largest {sizes.max():.0f}" if len(sizes) else ""))
    return txt, {"n": len(fin), "k": int(under.sum()), "share": m, "lo": lo, "hi": hi, "ends": ends, "units": len(sizes)}


def stage_report(root: Path, say, limit: int = 0, audit_file: str = "audit.jsonl", run_nodes: int = RUN_NODES, big: int = BIG) -> None:
    rows = load_rows(root, audit_file)
    p600 = [r for r in rows if r["set"] == "p600"]
    flag = [r for r in rows if r["set"] == "flag"]
    say(f"REPORT {len(rows)} positions audited ({len(p600)} of the 600, {len(flag)} flagged); file {audit_file}; run nodes limit {run_nodes}, reference fuse {big} nodes a reading, deeper reading +{EXTRA} plies; fresh table every reading; no clock")
    # (a) the search as it plays
    for band in BANDS + ("all",):
        c = cell(p600, band)
        if not c:
            continue
        say(f"(a) the search as it plays, p600 {band}: {len(c)} positions; depth D mean {np.mean([r['audit']['depth'] for r in c]):.2f} (min {min(r['audit']['depth'] for r in c)}, max {max(r['audit']['depth'] for r in c)}); "
            f"nodes mean {np.mean([r['audit']['run_nodes'] for r in c]):.0f}; value agrees with its own fresh full-window reading (as_played) in "
            f"{sum(1 for r in c if reading(r['audit'], 'as_played', False)['best'] == r['audit']['value'])} of {len(c)}")
    # (b) by band and reading
    cond = {}
    for deep in (False, True):
        for name in ("plain", "no_null", "no_late", "as_played"):                          # (as_played is the control: the search's own pruning, a fresh table and the full window; it should be near zero)
            for band in BANDS + ("all",):
                c = cell(p600, band)
                if not c:
                    continue
                txt, st = share_line(c, name, deep)
                say(f"(b) p600 {band}, reference '{name}' at depth D{'+' + str(EXTRA) if deep else ''}: {txt}")
                cond[(name, deep, band)] = st
    # what the shortfalls are made of, in the judge's own table: a game's end, exactly one unit's worth, less than the smallest worth (an unequal exchange: the difference of two worths), more than that and not one worth
    table = sorted(set(worth().values()))
    for band in BANDS + ("all",):
        sh = [s for s in (short_of(r["audit"], "plain", False) for r in cell(p600, band)) if s is not None and s > 0]
        ends = sum(1 for s in sh if s >= DECIDED)
        one = sum(1 for s in sh if s < DECIDED and s in table)
        below = sum(1 for s in sh if s < DECIDED and s not in table and s < table[0])
        other = sum(1 for s in sh if s < DECIDED and s not in table and s >= table[0])
        say(f"(b) what the shortfalls at D are made of, p600 {band} ({len(sh)} positions under the plain best; the table's worths {table}): a game's end {ends}; exactly one unit's worth {one}; under the smallest worth (the difference of unequal worths, an unequal "
            f"exchange) {below}; over the smallest worth and not one worth {other}")
    # the deeper reference agreeing: among positions under at D (plain), under also at D + 2
    for band in BANDS + ("all",):
        c = cell(p600, band)
        both = [(short_of(r["audit"], "plain", False), short_of(r["audit"], "plain", True)) for r in c]
        und = [(a, b) for a, b in both if a is not None and a > 0 and b is not None]
        agree = [b > 0 for a, b in und]
        unfinished_deep = sum(1 for r in c if short_of(r["audit"], "plain", False) is not None and short_of(r["audit"], "plain", False) > 0 and short_of(r["audit"], "plain", True) is None)
        say(f"(b) p600 {band}, the reference two plies deeper: of the {sum(1 for a, _b in both if a is not None and a > 0)} positions under the plain best at D, the deeper reading finished in {len(und)} and holds the played move under the best there too in "
            f"{sum(agree)} ({100 * np.mean(agree) if agree else float('nan'):.1f} pct); cut by the fuse {unfinished_deep}")
        over = [(a, b) for a, b in both if a is not None and a == 0 and b is not None]
        say(f"(b) p600 {band}, the other way: of the {len(over)} positions level with the plain best at D (both readings finished), the played move is under the best two plies deeper in {sum(1 for _a, b in over if b > 0)} ({100 * np.mean([b > 0 for _a, b in over]) if over else float('nan'):.1f} pct)")
    # node costs
    for band in BANDS + ("all",):
        c = cell(p600, band)
        if not c:
            continue
        parts = []
        for name in READINGS:
            for deep in (False, True):
                ns = [reading(r["audit"], name, deep)["nodes"] for r in c if short_of(r["audit"], name, deep) is not None]
                cut = len(c) - len(ns)
                parts.append(f"{name}{'+' + str(EXTRA) if deep else ''} mean {np.mean(ns) if ns else float('nan'):.0f} (cut {cut})")
        say(f"(b) node cost p600 {band}: search as played {np.mean([r['audit']['run_nodes'] for r in c]):.0f}; " + "; ".join(parts))
    # the verification of the played move alone
    ver = []
    for r in p600:
        a = r["audit"]
        ver.append((a["verify_nodes"], a["run_nodes"], a["verify_probe"], a["value"], a["depth"]))
    for band in BANDS + ("all",):
        c = cell(p600, band)
        v = [(r["audit"]["verify_nodes"], r["audit"]["run_nodes"], r["audit"]["verify_probe"], r["audit"]["value"]) for r in c]
        ok = [x for x in v if x[2] is not None]
        ratio = np.array([n / max(1, run) for n, run, _p, _v in ok])
        say(f"(b) verification of the played move alone (null window at its value, depth D, fresh table, both prunings off), p600 {band}: finished {len(ok)} of {len(c)}; the played move fails it in "
            f"{sum(1 for _n, _r, p, val in ok if p < val)} ({100 * np.mean([p < val for _n, _r, p, val in ok]) if ok else float('nan'):.1f} pct); nodes mean {np.mean([n for n, *_ in ok]) if ok else float('nan'):.0f}, "
            f"median {np.median([n for n, *_ in ok]) if ok else float('nan'):.0f}; ratio to the search's nodes mean {ratio.mean() if len(ratio) else float('nan'):.3f}, median {np.median(ratio) if len(ratio) else float('nan'):.3f}")
    # what the verification of the played move alone would reach: the positions under the plain best at D, split by whether the played move fails its own claim (the verification sees it) or holds it (a better move is missed, the verification does not)
    for band in BANDS + ("all",):
        under = [r for r in cell(p600, band) if (short_of(r["audit"], "plain", False) or 0) > 0]
        seen = [r for r in under if r["audit"]["verify_probe"] is not None and r["audit"]["verify_probe"] < r["audit"]["value"]]
        held = [r for r in under if r["audit"]["verify_probe"] is not None and r["audit"]["verify_probe"] >= r["audit"]["value"]]
        say(f"(b) what the verification of the played move alone reaches, p600 {band}: of the {len(under)} positions with the played move under the plain best at D, the played move fails its own claim (the verification sees it) in {len(seen)}, "
            f"holds it while a better move exists (the verification cannot see it) in {len(held)}")
    # with the verification of the choice applied (a run with --repair): what it did and what it cost
    rep = [r for r in rows if r["audit"].get("repair_state", "off") != "off"]
    if rep:
        for band in BANDS + ("all",):
            c = [r for r in cell(rows, band) if r["set"] == "p600"]
            sts = {s: sum(1 for r in c if r["audit"]["repair_state"] == s) for s in ("held", "replaced", "failed, kept", "cut")}
            ns = np.array([r["audit"]["repair_nodes"] for r in c], float)
            say(f"(repair) the verification of the choice applied, p600 {band}: {len(c)} positions; " + ", ".join(f"{k} {v}" for k, v in sts.items()) + f"; nodes of the verification mean {ns.mean():.0f}, median {np.median(ns):.0f}, "
                f"ratio to the search's nodes mean {np.mean([r['audit']['repair_nodes'] / max(1, r['audit']['run_nodes']) for r in c]):.3f}")
    # (c) the flagged first moves
    for band in BANDS + ("all",):
        c = cell(flag, band)
        if not c:
            continue
        res = []
        for r in c:
            rd = reading(r["audit"], "plain", False)
            res.append(None if rd["best"] is None or rd["other"] is None else int(rd["best"] - rd["other"]))
        fin = [x for x in res if x is not None]
        loss = [x for x in fin if x > 0]
        same = sum(1 for r in c if r["audit"]["played_name"] == r["other"])
        say(f"(c) flagged first moves, {band}: {len(c)} positions; the reference finished for {len(fin)}; the first move is UNDER the plain best at the search's depth (a real loss) in {len(loss)} ({100 * len(loss) / max(1, len(fin)):.1f} pct), "
            f"LEVEL with it (an exchange or a harmless move) in {len(fin) - len(loss)} ({100 * (len(fin) - len(loss)) / max(1, len(fin)):.1f} pct); shortfall of the real losses mean {np.mean(loss) if loss else float('nan'):.0f}, median "
            f"{np.median(loss) if loss else float('nan'):.0f}; the rerun search plays that move itself in {same}")
        txt, _ = share_line(c, "plain", False)
        say(f"(c) the move the rerun search plays on the flagged positions, {band}, reference 'plain' at D: {txt}")
    # the condition
    st = cond[("plain", False, "all")]
    met_share = st["share"] >= SHARE_FLOOR and st["lo"] > SHARE_BASE
    dump = {"share_all": st, "met_share": met_share, "median_verify_ratio": float(np.median([n / max(1, run) for n, run, p, _v, _d in ver if p is not None])) if ver else None}
    (root / ("condition_" + audit_file.replace(".jsonl", "") + ".json")).write_text(json.dumps(dump))
    say(f"CONDITION part (1): plain reading at D, p600 pooled: share under {100 * st['share']:.1f} pct [{100 * st['lo']:.1f}, {100 * st['hi']:.1f}] on {st['n']} positions; needs at least {100 * SHARE_FLOOR:.1f} and the lower end above {100 * SHARE_BASE:.1f}: "
        + ("MET" if met_share else "NOT MET") + f"; part (2) is read from the matches (choice_play.py report) and the median verification ratio {dump['median_verify_ratio']:.3f} (needs under {VERIFY_CHEAP} if the prunings clearly win)")


# ------------------------------------------------------------------------------------------------------------------------------ paired: before and after the verification of the choice

def stage_paired(root: Path, say, limit: int = 0, before: str = "audit20k.jsonl", after: str = "audit20k_repair.jsonl") -> None:
    """The same positions, the same deepening (a node limit, so the search is the same), the move read before and after the verification of the choice: how many positions leave the 'under the plain best at D' state,
    how many enter it, the shares, and what the verification cost."""
    a = {(r["set"], r["pid"]): r for r in load_rows(root, before)}
    b = {(r["set"], r["pid"]): r for r in load_rows(root, after)}
    keys = sorted(k for k in a if k in b and k[0] == "p600")
    for band in BANDS + ("all",):
        ks = [k for k in keys if band == "all" or a[k]["band"] == band]
        pairs = [(short_of(a[k]["audit"], "plain", False), short_of(b[k]["audit"], "plain", False)) for k in ks]
        fin = [(x, y) for x, y in pairs if x is not None and y is not None]
        same_run = sum(1 for k in ks if (a[k]["audit"]["depth"], a[k]["audit"]["value"], a[k]["audit"]["run_nodes"]) == (b[k]["audit"]["depth"], b[k]["audit"]["value"], b[k]["audit"]["run_nodes"]))
        out_of = sum(1 for x, y in fin if x > 0 and y == 0)
        into = sum(1 for x, y in fin if x == 0 and y > 0)
        still = sum(1 for x, y in fin if x > 0 and y > 0)
        bu, au = np.array([x > 0 for x, _y in fin], float), np.array([y > 0 for _x, y in fin], float)
        mb, lb, hb = boot(bu)
        ma, la, ha = boot(au)
        diff = np.array([y - x for x, y in fin], float)
        say(f"PAIRED p600 {band}: {len(ks)} positions with both runs, the same deepening in {same_run}, the reference finished for both in {len(fin)}; under the plain best at D before {int(bu.sum())} = {100 * mb:.1f} pct [{100 * lb:.1f}, {100 * hb:.1f}], "
            f"after {int(au.sum())} = {100 * ma:.1f} pct [{100 * la:.1f}, {100 * ha:.1f}]; left the state {out_of}, entered it {into}, stayed in it {still}")
        sts = [b[k]["audit"]["repair_state"] for k in ks]
        say(f"PAIRED p600 {band}: the verification held {sts.count('held')}, replaced the choice {sts.count('replaced')}, failed and kept {sts.count('failed, kept')}, was cut {sts.count('cut')}; nodes mean "
            f"{np.mean([b[k]['audit']['repair_nodes'] for k in ks]):.0f} (the search's {np.mean([b[k]['audit']['run_nodes'] for k in ks]):.0f})")


# ------------------------------------------------------------------------------------------------------------------------------ cases

def stage_cases(root: Path, say, limit: int = 0, audit_file: str = "audit.jsonl") -> None:
    rows = [r for r in load_rows(root, audit_file) if r["set"] == "p600"]
    out = []
    quota = {"few": 7, "mid": 7, "many": 6}
    for band in BANDS:
        c = sorted([r for r in rows if r["band"] == band and (short_of(r["audit"], "plain", False) or 0) > 0], key=lambda r: r["pid"])
        if not c:
            continue
        ix = np.unique(np.linspace(0, len(c) - 1, min(quota[band], len(c))).astype(int))
        for i in ix:
            r = c[int(i)]
            a = r["audit"]
            rd0, rd2 = reading(a, "plain", False), reading(a, "plain", True)
            pos = next(p for p in json.loads((root / "positions.json").read_text()) if p["set"] == "p600" and p["pid"] == r["pid"])
            out.append({"pid": r["pid"], "id": r["id"], "band": band, "units": r["units"], "moves": pos["moves"], "depth": a["depth"], "value": a["value"], "played": a["played_name"],
                        "best0": rd0["best"], "best_move0": rd0["best_name"], "played0": rd0["played"], "best2": rd2["best"], "best_move2": rd2["best_name"], "played2": rd2["played"],
                        "no_null0": [reading(a, "no_null", False)["best"], reading(a, "no_null", False)["played"]], "no_late0": [reading(a, "no_late", False)["best"], reading(a, "no_late", False)["played"]],
                        "verify_probe": a["verify_probe"]})
    (root / "cases92.json").write_text(json.dumps({"rule": CASES_RULE, "cases": out}))
    say(f"CASES: {CASES_RULE} {len(out)} positions written to cases92.json")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["select", "audit", "report", "cases", "paired"])
    ap.add_argument("--tag", default="full")
    ap.add_argument("--limit", type=int, default=0, help="a short run: this many flagged positions a band (select)")
    ap.add_argument("--run-nodes", type=int, default=RUN_NODES, help="the deepening's node limit (audit, report)")
    ap.add_argument("--big", type=int, default=BIG, help="the fuse of each fresh reference reading (audit, report)")
    ap.add_argument("--every", type=int, default=1, help="audit: only every this-many-th of the 600, no flagged position (the second setting)")
    ap.add_argument("--file", default="audit.jsonl", help="the audit file of this setting (audit, report, cases)")
    ap.add_argument("--file2", default="audit20k_repair.jsonl", help="paired: the run with the verification of the choice applied")
    ap.add_argument("--repair", action="store_true", help="audit: the verification of the choice applied first (the move read is the one it leaves); report: the file is such a run")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    say = Say(root, a.stage)
    if a.stage == "audit":
        stage_audit(root, say, a.limit, a.run_nodes, a.big, a.every, a.file, a.repair)
    elif a.stage == "report":
        stage_report(root, say, a.limit, a.file, a.run_nodes, a.big)
    elif a.stage == "paired":
        stage_paired(root, say, a.limit, a.file, a.file2)
    else:
        stage_select(root, say, a.limit) if a.stage == "select" else stage_cases(root, say, a.limit, a.file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
