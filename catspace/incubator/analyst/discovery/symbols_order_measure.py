"""PLAN 78.91 (SYSTEM_SPEC 7 "subgoals enter through the prior", 5 the search's reading order, 9 the score): THE HINT IN THE SEARCH'S ORDER.
A sandbox: no player is changed; nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/order_78_91/<tag>/ (OUT). Game-free: the game is named only through the pieces that exist
(symbols_order_play, which holds the orders and the reader; symbols_sample_measure's samples; ladder_measure's bands); the board and the words of a case are printed by ladder_print only. The outputs of 78.80 to 78.90 are READ,
never rebuilt, never written. THE REFERENCE MOVE IS THE ENGINE'S OWN SEARCH AT TEN TIMES THE PLAY CLOCK: no outside engine, oracle or labeller number is anywhere. No worth of any unit is assumed (the judge is the odds record the
control plays with). No player id is written. Nothing is fitted here: the weights are 78.81's and 78.90's fit A, as they were. CONFIRM positions only.

WHAT IS MEASURED (part one, a print; part two, only on the condition below; part three is NOT built: the cost print says what its port would have to beat).
  select   N_CELL positions (100) of each band (few, mid, many; ladder_measure.BANDS) of each CONFIRM part (78.80's, 78.89's), evenly spaced in the part's order (human positions only). The same positions for every order.
  search   the engine's own search on each position at the long clock (LONG = ten times the play clock 0.2 s, ledger share 0.4, the odds record's Worth judge, solve_equals_batch as every player of 78.67): the move it returns is
           THE REFERENCE; and at the play clock: the move it returns and the moves it holds equal (the short reading). Eight threads in one batch; the load is printed first.
           A SECOND REFERENCE (added after a 18-position pilot of this module showed the confound, before the full run): the search keeps the FIRST of the moves of equal best value in its own reading order, so the move
           it returns at the long clock sits early in the static order by construction. The NEUTRAL reference is the same search at the same clock with the root moves in a seeded random order (root order "strict", plan 78.51):
           the first of the best-valued moves in a random order, so no order of the table is favoured. Both references are tabled; THE PART TWO CONDITION AND THE CASES READ THE NEUTRAL ONE.
  orders   for each position, the legal moves, their gains, and the scores of the quiet moves under the two weight sets: "reasons" (78.81's weights) and "own" (78.90's fit A); Pool over all cores.
  cost     the time of the hint on 60 positions, one process, one at a time (idle machine): replay of the legal moves, the pattern count, the cells, the whole pick, per position and per quiet move, beside the play clock; and the
           profile of the pick by function (where the time goes). Print only.
  report   the place of the reference move (neutral and own, both tabled) in each order, by band: (i) the static order (the search's own before any reading: the moves that gain by the gain, then the rest as generated); (ii) the same with the QUIET moves
           ordered by the reasons' score ((ii-reasons) 78.81's weights, (ii-own) fit A); (iii) the order a short reading leaves (the move chosen first, then the others it held equal, then the static order). Mean place,
           the share with the reference first, the share with it in the first three: among quiet moves only (positions where the reference move is quiet; places counted among the quiet moves) and over all moves (every position).
           Paired against (i) with a 90 pct bootstrap band over games. THE PART TWO CONDITION (fixed here, before any number was read): (ii) is better than (i) among quiet moves, with a paired band clear of zero, in at
           least one of the bands few, mid, many (both CONFIRM parts pooled): the mean quiet place of (i) less that of (ii) has its 5th percentile above zero. If both (ii) variants meet it, "own" is the one played; else the one
           that meets it. condition.json says which.
  cases    (JUDGE) thirty positions: twenty where the reasons' order ((ii-own)) puts the reference move first among the quiet moves and the static order does not, and ten of the opposite. THE RULE (fixed before any position was
           read): in each band, the positions of the two CONFIRM parts taken alternately (part 1, part 2, ...) each in the order of select, the first k of each kind: k = 7, 7, 6 (few, mid, many) for the first kind and 4, 3, 3 for
           the second. cases91.json for ladder_print `--order91`.
  nodes    (part two, only if the condition holds) the same positions at a fixed depth D (the median depth the short reading reached, printed first; DEPTH FIXED FROM THAT LINE ALONE) with and without the root order (ii) in mode
           "ties", a clock long enough that the depth is read in full: nodes to depth D, and whether the move and the value are the same.
  match    (part two) the player `or_own` / `or_reasons` v control, `--set A|B --pairs 100 [--charge]`, 0.2 s a move, ledger share 0.4, ply cap 120, 9 workers, bootstrap bands over opening pairs, the load wait first.
  playreport  the lines of every match found.

INTERFACE (what each stage takes and writes; who calls it). Each stage is a command and prints OBS lines (also to obs.txt). Calls symbols_order_play (orders, reader, player) and the engine's `solve_equals_batch`; called by the
command line only; ladder_print.py reads cases91.json and groups_A.json of 78.90 (`--order91`).

  .venv/bin/python -m catspace.incubator.analyst.discovery.symbols_order_measure select|search|orders|cost|report|cases|nodes|match|playreport --tag short|full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import cProfile
import io
import json
import multiprocessing
import os
import pstats
import random
import re
import time
from pathlib import Path

import numpy as np

import cs_engine as CE

from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
from catspace.incubator.analyst.discovery import symbols_groups_measure as SGM
from catspace.incubator.analyst.discovery import symbols_measure as SM
from catspace.incubator.analyst.discovery import symbols_order_play as OO
from catspace.incubator.analyst.discovery import symbols_sample_measure as SS
from catspace.io import paths

OUT = ("order_78_91",)
N_CELL = 100                                                      # positions a band of each CONFIRM part
PLAY_CLOCK = TM.SECONDS                                           # 0.2 s
LONG = 10 * PLAY_CLOCK                                            # the reference's clock
LEDGER_SHARE = TM.LEDGER_SHARE
SEED = 9191
BANDS = SGM.BANDS
VARIANTS = (("ii-reasons", "reasons"), ("ii-own", "own"))
ORDERS = ("i static", "ii-reasons", "ii-own", "iii short")
CASE_K = {"reasons_first": {"few": 7, "mid": 7, "many": 6}, "static_first": {"few": 4, "mid": 3, "many": 3}}
COST_N = 60
THREADS = 8
MATCH_KINDS = dict(TM.KIND, or_own="or_own", or_reasons="or_reasons")


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS order91 {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


def load_json(root: Path, name: str):
    return json.loads((root / name).read_text())


# ------------------------------------------------------------------------------------------------------------------------------ select

def confirm_positions(no: int) -> list:
    root = SS.src80() if no == 1 else Path(paths.experiment(*SS.OUT, "full"))
    return [r for r in SS.load_sample(root) if r["part"] == "confirm" and r.get("source", "human") == "human"]


def stage_select(root: Path, say, limit: int = 0) -> None:
    out = []
    for no in (1, 2):
        conf = confirm_positions(no)
        for b, _lo, _hi in BANDS:
            rows = [r for r in conf if LM.band_of(int(r["units"])) == b]
            n = limit or N_CELL
            ix = np.unique(np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int))
            for i in ix:
                r = rows[int(i)]
                out.append({"pid": len(out), "sample": no, "band": b, "units": int(r["units"]), "game": r["game"], "ply": r["ply"], "id": r["id"], "moves": list(r["moves"])})
            say(f"SELECT CONFIRM {no}, band {b}: {len(rows)} human positions, {len(ix)} taken, evenly spaced in the part's order; {len(set(r['game'] for r in rows))} games in the band")
    (root / "positions.json").write_text(json.dumps(out))
    say(f"SELECT written: {len(out)} positions to {root / 'positions.json'}; games {len(set((p['sample'], p['game']) for p in out))}")


# ------------------------------------------------------------------------------------------------------------------------------ search

def solve_all(positions: list, seconds: float, threads: int, **kw) -> list:
    items = [(OM.VARIANT, None, list(p["moves"])) for p in positions]
    base = OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")
    rows = CE.solve_equals_batch(items, seconds, worth=dict(base["worth"]), ours_first=False, valuation="odds record", threads=threads, ledger_share=LEDGER_SHARE, **kw)
    return [PM.equal_set(r) for r in rows]


def stage_search(root: Path, say, limit: int = 0) -> None:
    pos = load_json(root, "positions.json")
    if limit:
        pos = pos[:limit]
    TM.wait_quiet(say)
    out = {}
    rng_orders = []
    for p in pos:
        n = len(CE.state(OM.VARIANT, None, p["moves"])[2])
        o = list(range(n))
        random.Random(f"{SEED}:{p['pid']}").shuffle(o)
        rng_orders.append(o)
    for tag, secs in (("short", PLAY_CLOCK), ("ref", LONG), ("refn", LONG)):
        t0 = time.perf_counter()
        rows = solve_all(pos, secs, THREADS, **({"root_orders": rng_orders, "root_order_mode": "strict"} if tag == "refn" else {}))
        say(f"SEARCH {tag}: {len(pos)} positions at {secs} s a move, ledger share {LEDGER_SHARE}, {THREADS} threads in one batch, {time.perf_counter() - t0:.0f} s; "
            f"errors {sum(1 for r in rows if r['error'])}; mean ledger depth {np.mean([r['depth'] or 0 for r in rows]):.2f}, median {np.median([r['depth'] or 0 for r in rows]):.0f}; "
            f"mean moves held equal {np.mean([len(r['equals']) for r in rows]):.2f}")
        out[tag] = rows
    res = {}
    for p, s, r, rn in zip(pos, out["short"], out["ref"], out["refn"]):
        legal = CE.state(OM.VARIANT, None, p["moves"])[2]
        res[str(p["pid"])] = {"refn": legal.index(rn["move"]) if rn["move"] in legal else None, "refn_move": rn["move"], "refn_depth": rn["depth"], "refn_equals": rn["equal_indices"], "ref_equals": r["equal_indices"], "legal": len(legal), "ref": legal.index(r["move"]) if r["move"] in legal else None, "ref_move": r["move"], "ref_depth": r["depth"], "ref_cut": r["cut"],
                              "short": legal.index(s["move"]) if s["move"] in legal else None, "short_move": s["move"], "short_depth": s["depth"], "short_equals": s["equal_indices"],
                              "short_nodes": s["nodes"], "equals_agree_names": [legal[i] for i in s["equal_indices"]] == s["equals"], "error": s["error"] or r["error"] or rn["error"]}
    (root / "search.json").write_text(json.dumps(res))
    bad = [k for k, v in res.items() if v["error"] or v["ref"] is None or v["refn"] is None or v["short"] is None or not v["equals_agree_names"]]
    same = sum(1 for v in res.values() if v["ref"] == v["short"])
    samen = sum(1 for v in res.values() if v["refn"] == v["short"])
    say(f"SEARCH written: {len(res)} positions; unusable {len(bad)}; the short reading's move is the own reference in {same} ({100 * same / max(1, len(res)):.1f} pct) and the neutral reference in {samen} ({100 * samen / max(1, len(res)):.1f} pct); the own and the neutral reference are the same move in {sum(1 for v in res.values() if v['ref'] == v['refn'])}; the equal indices name the same moves as the names in {sum(v['equals_agree_names'] for v in res.values())}")


# ------------------------------------------------------------------------------------------------------------------------------ orders

def _orders_job(chunk: list) -> list:
    rd = OO.hint_order_reader()
    out = []
    for p in chunk:
        t0 = time.perf_counter()
        read = rd.read(p["moves"])
        t1 = time.perf_counter()
        sc = {w: rd.scores(p["moves"], read, w) for _n, w in VARIANTS}
        t2 = time.perf_counter()
        out.append({"pid": p["pid"], "legal": read["legal"], "gains": read["gains"], "scores": {w: {str(i): v for i, v in s.items()} for w, s in sc.items()}, "read_s": t1 - t0, "score_s": t2 - t1})
    return out


def stage_orders(root: Path, say, limit: int = 0) -> None:
    pos = load_json(root, "positions.json")
    if limit:
        pos = pos[:limit]
    chunks = [pos[i::36] for i in range(36)]
    t0 = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(os.cpu_count() or 1) as pool:
        rows = [r for part in pool.imap_unordered(_orders_job, chunks) for r in part]
    rows.sort(key=lambda r: r["pid"])
    (root / "orders.json").write_text(json.dumps(rows))
    say(f"ORDERS: {len(rows)} positions, {sum(len(r['legal']) for r in rows)} legal moves, quiet {sum(sum(1 for g in r['gains'] if g == 0) for r in rows)}; {time.perf_counter() - t0:.0f} s on {os.cpu_count()} cores; "
        f"legal moves a position mean {np.mean([len(r['legal']) for r in rows]):.1f}; positions with a gaining move {sum(1 for r in rows if any(g != 0 for g in r['gains']))}")


# ------------------------------------------------------------------------------------------------------------------------------ the cost

def stage_cost(root: Path, say, limit: int = 0) -> None:
    from catspace.incubator.analyst.discovery import symbols as SY
    from catspace.incubator.analyst.discovery import symbols_own_measure as O
    pos = load_json(root, "positions.json")
    ix = np.unique(np.linspace(0, len(pos) - 1, limit or COST_N).astype(int))
    sel = [pos[int(i)] for i in ix]
    rd = OO.hint_order_reader()
    load = os.getloadavg()[0]
    say(f"COST: load average (1 min) at the start {load:.2f}; cores busy at the instant of start {OM.cpu_busy():.2f}; {len(sel)} positions one at a time, one process; the play clock is {1000 * PLAY_CLOCK:.0f} ms a move")
    rd.read(sel[0]["moves"])                                         # warm: the readers and the engine's tables are built before the clock starts
    rd.scores(sel[0]["moves"], rd.read(sel[0]["moves"]), "own")
    rd.scores(sel[0]["moves"], rd.read(sel[0]["moves"]), "reasons")
    by: dict = {}
    for p in sel:
        t0 = time.perf_counter()
        read = rd.read(p["moves"])
        t1 = time.perf_counter()
        quiet = [read["legal"][i] for i in read["quiet"]]
        if len(quiet) < 2:
            continue
        table = SY.deltas_of({"moves": list(p["moves"])}, quiet, rd.r81.counter)
        t2 = time.perf_counter()
        own = rd.own
        O._own_job(([(list(p["moves"]), quiet)], own.raws))
        t3 = time.perf_counter()
        from catspace.incubator.analyst.discovery import symbols_groups_measure as G
        G._split_job(([(list(p["moves"]), quiet)], own.raws))
        t4 = time.perf_counter()
        rd.scores(p["moves"], read, "reasons")
        t5 = time.perf_counter()
        rd.scores(p["moves"], read, "own")
        t6 = time.perf_counter()
        by.setdefault(p["band"], []).append({"quiet": len(quiet), "legal": len(read["legal"]), "replay": t1 - t0, "count": t2 - t1, "own_cells": t3 - t2, "mover_cells": t4 - t3, "pick_reasons": t5 - t4, "pick_own": t6 - t5})
    keys = ("replay", "count", "mover_cells", "own_cells", "pick_reasons", "pick_own")
    allrows = [r for v in by.values() for r in v]
    for b, rows in list(by.items()) + [("all", allrows)]:
        q = np.mean([r["quiet"] for r in rows])
        say(f"COST band {b} ({len(rows)} positions, {np.mean([r['legal'] for r in rows]):.1f} legal moves, {q:.1f} quiet): ms a position "
            + ", ".join(f"{k} {1000 * np.mean([r[k] for r in rows]):.1f}" for k in keys) + "; per quiet move "
            + ", ".join(f"{k} {1000 * np.mean([r[k] / r['quiet'] for r in rows]):.2f}" for k in keys)
            + f"; the whole hint (replay + the own pick) {1000 * np.mean([r['replay'] + r['pick_own'] for r in rows]):.1f} ms = {100 * np.mean([r['replay'] + r['pick_own'] for r in rows]) / PLAY_CLOCK:.0f} pct of the play clock")
    prof = cProfile.Profile()
    prof.enable()
    for p in sel[:20]:
        read = rd.read(p["moves"])
        rd.scores(p["moves"], read, "own")
    prof.disable()
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(12)
    tot = pstats.Stats(prof).total_tt
    say(f"COST profile of reading and the own pick on {len(sel[:20])} positions: total {tot:.2f} s; by own time of each function (top 12):")
    for line in s.getvalue().splitlines():
        if line.strip() and ("{" in line or ".py" in line or "(" in line) and not line.startswith("Ordered") and "ncalls" not in line:
            say("  " + re.sub(r"/\S+/(\w+\.py)", r"\1", " ".join(line.split())))


# ------------------------------------------------------------------------------------------------------------------------------ the report

def table_rows(root: Path, key: str = "refn") -> list:
    pos = load_json(root, "positions.json")
    sr = load_json(root, "search.json")
    od = {r["pid"]: r for r in load_json(root, "orders.json")}
    rows = []
    for p in pos:
        s, o = sr.get(str(p["pid"])), od.get(p["pid"])
        if s is None or o is None or s["error"] or s[key] is None or s["short"] is None:
            continue
        gains = o["gains"]
        ref = s[key]
        quiet = {i for i, g in enumerate(gains) if g == 0}
        orders = {"i static": OO.static_order(gains), "iii short": OO.short_order(gains, s["short"], s["short_equals"])}
        for name, w in VARIANTS:
            orders[name] = OO.hint_order(gains, {int(i): v for i, v in o["scores"][w].items()})
        row = {"pid": p["pid"], "sample": p["sample"], "band": p["band"], "game": f"{p['sample']}:{p['game']}", "legal": len(gains), "n_quiet": len(quiet), "ref_quiet": ref in quiet, "orders": orders, "ref": ref}
        row["place_all"] = {k: OO.place_of(v, ref) for k, v in orders.items()}
        row["place_quiet"] = {k: (OO.place_of(v, ref, quiet) if ref in quiet else None) for k, v in orders.items()}
        rows.append(row)
    return rows


def place_line(vals: np.ndarray) -> str:
    if len(vals) == 0:
        return "no positions"
    return f"mean place {vals.mean():.2f}, first {100 * (vals == 1).mean():.1f} pct, first three {100 * (vals <= 3).mean():.1f} pct"


def cells(rows: list) -> list:
    out = [("all bands", lambda r: True)]
    for b, _lo, _hi in BANDS:
        out.append((f"band {b}", lambda r, b=b: r["band"] == b))
    return out


def condition_of(verdicts: dict) -> dict:
    """The part two condition of the module doc: `verdicts` {ii variant: {band: the 5th percentile of the paired mean place of i less ii among quiet moves}}; met when some band of some variant is above zero; "own" is the one played when both meet it."""
    met = {n: [b for b, lo in v.items() if lo > 0] for n, v in verdicts.items()}
    which = "ii-own" if met.get("ii-own") else ("ii-reasons" if met.get("ii-reasons") else None)
    return {"met": which is not None, "which": None if which is None else dict(VARIANTS)[which], "bands": met}


def stage_report(root: Path, say, limit: int = 0) -> None:
    verdicts = _report(root, say, "refn", "NEUTRAL reference (the search at the long clock, root moves in a seeded random order)")
    _report(root, say, "ref", "OWN reference (the search at the long clock in its own order)")
    cond = condition_of(verdicts)
    met = cond["bands"]
    (root / "condition.json").write_text(json.dumps(cond))
    say(f"CONDITION (fixed in the module doc before the full run; read on the NEUTRAL reference): bands where the paired band of i less ii among quiet moves is above zero: ii-reasons {met.get('ii-reasons', [])}, ii-own {met.get('ii-own', [])}; "
        f"part two {'RUNS with ' + cond['which'] if cond['met'] else 'does NOT run'}")


def _report(root: Path, say, key: str, what: str) -> dict:
    rows = table_rows(root, key)
    t = key.upper()
    say(f"REPORT {t}: {what}: {len(rows)} positions; the reference move is quiet in {sum(r['ref_quiet'] for r in rows)}; places are 1-based; 'quiet' = among quiet moves only, in the positions where the "
        "reference move is quiet; 'all' = among all legal moves, every position; the order (i) static is the search's own before any reading")
    for label, sel in cells(rows):
        use = [r for r in rows if sel(r)]
        qr = [r for r in use if r["ref_quiet"]]
        chance_q = np.mean([(r["n_quiet"] + 1) / 2 for r in qr]) if qr else float("nan")
        chance_a = np.mean([(r["legal"] + 1) / 2 for r in use]) if use else float("nan")
        say(f"TABLE {t} {label}: {len(use)} positions, {len(qr)} with a quiet reference move; legal moves a position {np.mean([r['legal'] for r in use]):.1f}, quiet {np.mean([r['n_quiet'] for r in use]):.1f}; a random order would give a mean place "
            f"{chance_q:.2f} among quiet moves and {chance_a:.2f} among all")
        for name in ORDERS:
            say(f"TABLE {t} {label}, order {name}: quiet only ({len(qr)}): " + place_line(np.array([r["place_quiet"][name] for r in qr], float)) + "; all moves (" + str(len(use)) + "): "
                + place_line(np.array([r["place_all"][name] for r in use], float)))
    verdicts: dict = {}
    for label, sel in cells(rows):
        use = [r for r in rows if sel(r)]
        qr = [r for r in use if r["ref_quiet"]]
        g_q = np.array([r["game"] for r in qr])
        g_a = np.array([r["game"] for r in use])
        for name in ("ii-reasons", "ii-own", "iii short"):
            pq = {k: np.array([r["place_quiet"][k] for r in qr], float) for k in ORDERS}
            pa = {k: np.array([r["place_all"][k] for r in use], float) for k in ORDERS}
            if len(qr) < 5:
                say(f"PAIRED {t} {label}, {name} against i: too few positions ({len(qr)})")
                continue
            dq = SM.paired(pq["i static"], pq[name], g_q, SEED)
            da = SM.paired(pa["i static"], pa[name], g_a, SEED + 1)
            f_q = SM.paired((pq[name] == 1).astype(float), (pq["i static"] == 1).astype(float), g_q, SEED + 2)
            f_a = SM.paired((pa[name] == 1).astype(float), (pa["i static"] == 1).astype(float), g_a, SEED + 3)
            say(f"PAIRED {t} {label}, {name} against i static: quiet ({len(qr)} positions) mean place i less {name} {dq[0]:+.3f} [{dq[1]:+.3f}, {dq[2]:+.3f}] (positive = {name} puts the reference earlier); "
                f"first-place share {name} less i {100 * f_q[0]:+.2f} [{100 * f_q[1]:+.2f}, {100 * f_q[2]:+.2f}] points; all moves ({len(use)}) mean place i less {name} {da[0]:+.3f} [{da[1]:+.3f}, {da[2]:+.3f}]; "
                f"first-place share {name} less i {100 * f_a[0]:+.2f} [{100 * f_a[1]:+.2f}, {100 * f_a[2]:+.2f}] points")
            if label.startswith("band") and name != "iii short":
                verdicts.setdefault(name, {})[label] = dq[1]
    return verdicts


# ------------------------------------------------------------------------------------------------------------------------------ cases (JUDGE)

def stage_cases(root: Path, say, limit: int = 0) -> None:
    rows = {r["pid"]: r for r in table_rows(root)}
    pos = {p["pid"]: p for p in load_json(root, "positions.json")}
    rd = OO.hint_order_reader()
    picked = []
    for kind, ks in CASE_K.items():
        for b, _lo, _hi in BANDS:
            def ok(r):
                if not r["ref_quiet"] or r["band"] != b:
                    return False
                a, s = r["place_quiet"]["ii-own"], r["place_quiet"]["i static"]
                return (a == 1 and s != 1) if kind == "reasons_first" else (s == 1 and a != 1)
            per = {no: [r for r in sorted(rows.values(), key=lambda r: r["pid"]) if ok(r) and r["sample"] == no] for no in (1, 2)}
            inter = []
            for k in range(max(len(per[1]), len(per[2]))):
                for no in (1, 2):
                    if k < len(per[no]):
                        inter.append(per[no][k])
            got = inter[:ks[b]]
            say(f"CASES {kind}, band {b}: {len(per[1])} + {len(per[2])} positions qualify (part 1 + part 2); {len(got)} taken (rule: parts alternately, in select's order, first {ks[b]})")
            picked += [(kind, r) for r in got]
    out = []
    for kind, r in picked:
        p = pos[r["pid"]]
        read = rd.read(p["moves"])
        quiet = read["quiet"]
        names = [read["legal"][i] for i in quiet]
        detail = rd.own.pick(list(p["moves"]), names)[1]
        top = lambda i: sorted(detail["carried"].get(read["legal"][i], []), key=lambda gw: -abs(gw[1]))[:3]
        st, hint = r["orders"]["i static"], r["orders"]["ii-own"]
        qs = [i for i in st if i in set(quiet)]
        qh = [i for i in hint if i in set(quiet)]
        out.append({"pid": r["pid"], "kind": kind, "band": r["band"], "sample": r["sample"], "units": p["units"], "id": p["id"], "ply": p["ply"], "moves": p["moves"], "legal": read["legal"], "gains": read["gains"],
                    "ref": r["ref"], "static_quiet": qs[:5], "hint_quiet": qh[:5], "place_static": r["place_quiet"]["i static"], "place_hint": r["place_quiet"]["ii-own"], "place_reasons": r["place_quiet"]["ii-reasons"],
                    "place_short": r["place_quiet"]["iii short"], "groups": {str(i): top(i) for i in set([r["ref"], qs[0], qh[0]])}, "n_quiet": len(quiet)})
    (root / "cases91.json").write_text(json.dumps({"groups_file": "groups_A.json", "groups_exp": "symbols_78_90", "cases": out}))
    say(f"CASES written: {len(out)} positions ({sum(c['kind'] == 'reasons_first' for c in out)} where the reasons' order puts the reference first among the quiet moves and the static order does not, "
        f"{sum(c['kind'] == 'static_first' for c in out)} of the opposite) to {root / 'cases91.json'} (print with ladder_print --order91)")


# ------------------------------------------------------------------------------------------------------------------------------ part two

def stage_nodes(root: Path, say, limit: int = 0) -> None:
    cond = load_json(root, "condition.json")
    if not cond["met"]:
        say("NODES: the condition of part two is not met (condition.json): not run")
        return
    pos = load_json(root, "positions.json")
    sr = load_json(root, "search.json")
    if limit:
        pos = pos[:limit]
    depths = [sr[str(p["pid"])]["short_depth"] for p in pos if sr[str(p["pid"])]["short_depth"]]
    depth = int(np.median(depths))
    say(f"NODES: depth fixed at the median ledger depth of the short readings, {depth} (positions {len(pos)}, short depths {np.percentile(depths, [10, 50, 90]).tolist()} at the 10th, 50th, 90th percentile)")
    TM.wait_quiet(say)
    rd = OO.hint_order_reader()
    orders = [rd.order(p["moves"], cond["which"])[0] for p in pos]
    secs = 60.0
    t0 = time.perf_counter()
    base = solve_all(pos, secs, THREADS, ledger_depth=depth)
    t1 = time.perf_counter()
    ordd = solve_all(pos, secs, THREADS, ledger_depth=depth, root_orders=orders, root_order_mode="ties")
    t2 = time.perf_counter()
    rows = []
    for p, a, b in zip(pos, base, ordd):
        rows.append({"pid": p["pid"], "band": p["band"], "nodes_none": a["nodes"], "nodes_order": b["nodes"], "depth_none": a["depth"], "depth_order": b["depth"], "same_move": a["move"] == b["move"], "same_value": a["value"] == b["value"],
                     "cut_none": a["cut"], "cut_order": b["cut"], "value_none": a["value"], "value_order": b["value"], "err": bool(a["error"] or b["error"])})
    (root / "nodes.json").write_text(json.dumps(rows))
    say(f"NODES: {len(rows)} positions, no order {t1 - t0:.0f} s, order {t2 - t1:.0f} s; ledger depth read in full: none {np.mean([r['depth_none'] == depth for r in rows]):.3f}, order {np.mean([r['depth_order'] == depth for r in rows]):.3f} of positions")
    for label in ["all"] + [b for b, _l, _h in BANDS]:
        use = [r for r in rows if (label == "all" or r["band"] == label) and not r["err"]]
        if not use:
            continue
        a, b = np.array([r["nodes_none"] for r in use], float), np.array([r["nodes_order"] for r in use], float)
        g = np.array([f"{r['pid']}" for r in use])
        d = SM.paired(b / a, np.ones(len(use)), g, SEED + 4)
        say(f"NODES {label} ({len(use)} positions): total nodes none {a.sum():.0f}, with the order {b.sum():.0f} (ratio {b.sum() / a.sum():.3f}); per position median ratio {np.median(b / a):.3f}, mean ratio less one {d[0]:+.3f} "
            f"[{d[1]:+.3f}, {d[2]:+.3f}]; the same move {100 * np.mean([r['same_move'] for r in use]):.1f} pct, the same value {100 * np.mean([r['same_value'] for r in use]):.1f} pct; cut by the clock or fuse: none "
            f"{sum(r['cut_none'] for r in use)}, order {sum(r['cut_order'] for r in use)}")


def games_path(root: Path, a: str, b: str, set_name: str, charge: bool) -> Path:
    return root / "matches" / f"{a}_v_{b}_{set_name}{'_charged' if charge else ''}.jsonl"


def stage_match(root: Path, say, args) -> None:
    cond = load_json(root, "condition.json")
    if not cond["met"]:
        say("MATCH: the condition of part two is not met (condition.json): not run")
        return
    a = args.a or ("or_own" if cond["which"] == "own" else "or_reasons")
    specs = TM.match_specs(a, args.b, args.set, args.pairs, PLAY_CLOCK, root / "scratch", args.charge, MATCH_KINDS)
    load, busy = TM.wait_quiet(say)
    t0 = time.perf_counter()
    out = games_path(root, a, args.b, args.set, args.charge)
    games = TM.run_games(specs, args.workers, out, say=lambda m: say(m), play=OO.play91)
    wall = time.perf_counter() - t0
    lines = TM.match_lines(a, args.b, args.set + ("+charge" if args.charge else ""), games, load, busy, wall, PLAY_CLOCK, args.workers)
    for ln in lines:
        say(ln.replace("OBS tiebreak match", "MATCH"))
    (out.with_suffix(".meta.json")).write_text(json.dumps({"load": load, "busy": busy, "wall": wall}))


def stage_playreport(root: Path, say) -> None:
    for f in sorted((root / "matches").glob("*.jsonl")):
        name = f.stem
        games = TM.read_games(f)
        a, rest = name.split("_v_")
        b, s = rest.rsplit("_", 1) if not rest.endswith("_charged") else (rest[:-len("_charged")].rsplit("_", 1)[0], rest[:-len("_charged")].rsplit("_", 1)[1] + "+charge")
        meta = json.loads(f.with_suffix(".meta.json").read_text()) if f.with_suffix(".meta.json").exists() else {"load": float("nan"), "busy": float("nan"), "wall": float("nan")}
        for ln in TM.match_lines(a, b, s, games, meta["load"], meta["busy"], meta["wall"], PLAY_CLOCK, 9):
            say(ln.replace("OBS tiebreak match", "MATCH"))


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "search", "orders", "cost", "report", "cases", "nodes", "match", "playreport"])
    ap.add_argument("--tag", default="full")
    ap.add_argument("--limit", type=int, default=0, help="a short run: this many positions a cell (select) or in all (other stages)")
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default="control")
    ap.add_argument("--set", default="A")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--charge", action="store_true")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    say = Say(root, a.stage)
    if a.stage == "match":
        stage_match(root, say, a)
    elif a.stage == "playreport":
        stage_playreport(root, say)
    else:
        {"select": stage_select, "search": stage_search, "orders": stage_orders, "cost": stage_cost, "report": stage_report, "cases": stage_cases, "nodes": stage_nodes}[a.stage](root, say, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
