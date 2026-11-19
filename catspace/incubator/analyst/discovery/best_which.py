"""PLAN 79.03 PART ONE, SECOND HALF: WHICH PRUNING THE BEST MOVE'S READING PASSED THROUGH (SYSTEM_SPEC 4 the tactical reasoner's deepening and its prunings, 12 the debt of plan 78.84; piece: tactical reasoning, a MEASURE). No engine change, no oracle call; node budgets, no clock; outputs under artifacts/experiments/best_79_03/<tag>/.

WHAT IS READ. For every position with a gap (best_count: the best exact value above v, of any size, the game's own end included) and for the same number of positions with no gap (the control: the first ones, in the sample's hash order), the engine's choice audit (cs_engine.choice_audit; plan 78.92's instrument, no change) at the depth D the deepening finished: four deepenings on fresh tables, one per
reading, each stopped at D, differing in one thing:
  plain      both the late-move reduction and the null move off (alpha-beta and the table alone: the reference)
  no_null    the null move off, the reduction on        no_late    the reduction off, the null move on        as_played   both on (the search itself)
each gives the best value of the position at D and the value of the move NAMED as `other` (here: the best exact move of best_count, the move the gap is about), read on the same table with the full window. "The line the search plays passes through" a pruning when switching that pruning off lifts the reading to the best exact value:
  late      the reading with the reduction off reaches the best exact value (the null move on), and the reading with the null move off does not
  null      the reading with the null move off reaches it, and the one with the reduction off does not           either   each of the two reaches it           joint   only the reading with both off reaches it
  neither   not even the plain reading reaches the best exact value (the `exact` of the audit is not the plain value; a table or window effect)       unread   a reading was cut by the fuse
The control gives the rate at which these readings differ from v where best_count found no gap.

INTERFACE (written before the bodies).
  cause_of(best_exact, readings) -> str               "late" | "null" | "either" | "joint" | "neither" | "unread" (pure; tested). `readings` is {name: best value or None}.
  items_of(tag, src) -> [(position row, moves)]       the gap positions and the control (pure choice, reads best_count's gap_rows.jsonl).
  Which()                                             per process: the judge's worths; `.read(moves, depth, other) -> {"depth", "value", "readings": {name: {"best", "other", "played"}}}`.
  run(tag, workers, limit, say) / load(root) / cmd_count(tag, say)
  Called by: the command line, best_mend (the positions with a gap), ladder_print --best03.

  .venv/bin/python -m catspace.incubator.analyst.discovery.best_which run --tag short --limit 40 | run --tag full | count --tag full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import best_count as BC
from catspace.incubator.analyst.discovery import equal_trace as ET
from catspace.incubator.analyst.discovery import leaf_ask as LA
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import prior_measure as PMS
from catspace.incubator.analyst.discovery import record_play as RP
from catspace.incubator.analyst.discovery import stack_room as SR
from catspace.incubator.analyst.discovery.best_out import Say, out_root

CHUNK = 8
MIN_CELL = 30
NAMES = ("plain", "no_null", "no_late", "as_played")
CAUSES = ("late", "null", "either", "joint", "neither", "unread")


# ------------------------------------------------------------------------------------------------------------------------------ pure parts

def cause_of(best_exact: int, readings: dict) -> str:
    nn, nl, pl = readings.get("other_no_null"), readings.get("other_no_late"), readings.get("other_plain")
    if nn is None or nl is None or pl is None:
        return "unread"
    a, b, c = nn >= best_exact, nl >= best_exact, pl >= best_exact
    if a and b:
        return "either"
    if b:
        return "late"
    if a:
        return "null"
    return "joint" if c else "neither"


def items_of(tag_gap: str, src: str = LA.SRC, limit: int = 0) -> list:
    """The gap positions (kind less, unit, ends), then the same number of positions with no gap, in the sample's hash order (the control), each with its moves."""
    from catspace.incubator.analyst.discovery import cut_count as CC
    rows = BC.load_rows(tag_gap)
    gap = [r for r in rows if r["kind"] in ("less", "unit", "ends")]
    none = sorted((r for r in rows if r["kind"] == "none"), key=lambda r: PE.order_key(r["game"], int(r["id"].rsplit("_", 1)[1])))[:len(gap)]
    pick = {r["id"]: ("gap" if r["kind"] != "none" else "control") for r in gap + none}
    byid = {r["id"]: r for r in gap + none}
    items = [(rec, moves) for rec, moves in CC.items_of(src, 0) if rec["id"] in pick]
    out = [(dict(byid[rec["id"]], role=pick[rec["id"]], depth=rec["depth"]), moves) for rec, moves in items]
    out.sort(key=lambda it: (it[0]["role"] != "gap", PE.order_key(it[0]["game"], int(it[0]["id"].rsplit("_", 1)[1]))))
    if limit:
        g = [x for x in out if x[0]["role"] == "gap"][:limit // 2]
        c = [x for x in out if x[0]["role"] == "control"][:limit // 2]
        out = g + c
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the reads

class Which:
    def __init__(self):
        self.worth = RP.worth_of(SR.RECORD)

    def read(self, moves: list, depth: int, other: str) -> dict:
        import cs_engine as CE
        ca = CE.choice_audit(OM.VARIANT, list(moves), self.worth, ET.RUN_NODES, depth, ET.BIG, 0, True, other, False)
        if ca is None:
            return {"error": "no audit"}
        rd = {r["name"]: r for r in ca["readings"] if r["depth"] == int(ca["depth"])}
        return {"depth": int(ca["depth"]), "value": int(ca["value"]), "played_name": ca["played_name"],
                "readings": {n: {"best": (rd[n]["best"] if n in rd else None), "other": (rd[n]["other"] if n in rd else None), "played": (rd[n]["played"] if n in rd else None), "best_name": (rd[n]["best_name"] if n in rd else None), "nodes": (rd[n]["nodes"] if n in rd else None)} for n in NAMES}}


_WH = None


def _init_worker() -> None:
    global _WH
    _WH = Which()


def _job(args: tuple) -> tuple:
    idx, items, root = args
    f = Path(root) / "which" / f"c{idx:05d}.json"
    if f.exists() and "\"error\"" not in f.read_text():
        return idx, 0.0
    t0 = time.perf_counter()
    out = []
    for row, moves in items:
        try:
            out.append(dict(_WH.read(moves, row["depth"], row["best_move"]), id=row["id"]))
        except Exception as e:
            out.append({"id": row["id"], "error": f"{type(e).__name__}: {str(e)[:150]}"})
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0


def load(root: Path) -> list:
    return [json.loads(x) for x in (root / "which.jsonl").read_text().splitlines() if x.strip()]


def run(tag: str, tag_gap: str, workers: int, limit: int, say) -> None:
    root = out_root(tag)
    (root / "which").mkdir(exist_ok=True)
    items = items_of(tag_gap, LA.SRC, limit)
    say(f"which pruning: {sum(1 for r, _ in items if r['role'] == 'gap')} positions with a gap and {sum(1 for r, _ in items if r['role'] == 'control')} control positions; four deepenings each at the depth D the position finished (node budgets, no clock; the human-result record); {workers} workers of {os.cpu_count()} cores; load average at the start {os.getloadavg()[0]:.2f}")
    jobs = [(a // CHUNK, items[a:a + CHUNK], str(root)) for a in range(0, len(items), CHUNK)]
    t0 = time.perf_counter()
    done = 0
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for _i, _dt in pool.imap_unordered(_job, jobs, chunksize=1):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                say(f"progress: {done} of {len(jobs)} chunks; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    out = []
    for f in sorted((root / "which").glob("c*.json")):
        out += json.loads(f.read_text())
    (root / "which.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    say(f"read {len(out)} of {len(items)} positions in {time.perf_counter() - t0:.0f} s wall ({sum(1 for r in out if 'error' in r)} errors); load average at the end {os.getloadavg()[0]:.2f}")


# ------------------------------------------------------------------------------------------------------------------------------ the counts

def cell(vals: np.ndarray, game: np.ndarray, sel: np.ndarray) -> str:
    n = int(sel.sum())
    if n < MIN_CELL:
        return f"{100.0 * float(vals[sel].mean()):.1f} (n {n}, too few for a band)" if n else "none (n 0)"
    return PMS.fmt(PMS.share(vals[sel], game[sel])) + f" (n {n})"


def flat(r: dict) -> dict:
    rd = r["readings"]
    return {"other_" + n: rd[n]["other"] for n in NAMES} | {"best_" + n: rd[n]["best"] for n in NAMES}


def cmd_count(tag: str, tag_gap: str, say) -> None:
    root = out_root(tag)
    rows = {r["id"]: r for r in BC.load_rows(tag_gap)}
    reads = [r for r in load(root) if "error" not in r]
    rs = []
    for r in reads:
        g = rows[r["id"]]
        fl = flat(r)
        rs.append({"id": r["id"], "game": g["game"], "band": g["band"], "role": "gap" if g["kind"] != "none" else "control", "kind": g["kind"], "v": g["v"], "best_exact": g["best_exact"], "fl": fl, "same_v": (r["value"] == g["v"]),
                   "cause": cause_of(g["best_exact"], fl), "plain_above_v": (fl["best_plain"] is not None and fl["best_plain"] > g["v"]), "plain_differs": (fl["best_plain"] is not None and fl["best_plain"] != g["v"]),
                   "plain_reaches": (fl["other_plain"] is not None and fl["other_plain"] >= g["best_exact"]), "unit": None})
    (root / "which_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rs) + "\n")
    unit = BC.weakest_unit(RP.worth_of(SR.RECORD))
    for r in rs:
        r["plain_gap"] = (r["fl"]["best_plain"] - r["v"]) if r["fl"]["best_plain"] is not None else None
    for role in ("gap", "control"):
        up = [r["plain_gap"] / unit for r in rs if r["role"] == role and r["plain_gap"] is not None and r["plain_gap"] > 0 and r["kind"] != "ends"]
        ends = sum(1 for r in rs if r["role"] == role and r["plain_gap"] is not None and r["plain_gap"] > 0 and r["kind"] == "ends")
        if up:
            u = np.array(up)
            say(f"[{role}] the plain deepening's best less v, where it is above v and the position is not one of the game's end (n {len(up)}), in units of the weakest kind: under one {int((u < 1).sum())}, one {int((u == 1).sum())}, over one to three {int(((u > 1) & (u <= 3)).sum())}, over three {int((u > 3).sum())}; median {np.median(u):.2f}; positions of the game's end by best_count's kind {ends}")
    allrows = BC.load_rows(tag_gap)
    n_all = len(allrows)
    n_gap = sum(1 for r in allrows if r["kind"] in ("less", "unit", "ends"))
    g_rate = float(np.mean([r["plain_above_v"] for r in rs if r["role"] == "gap"]))
    c_rate = float(np.mean([r["plain_above_v"] for r in rs if r["role"] == "control"]))
    say(f"POINT ESTIMATE over all {n_all} positions (no band: the control is a sample of the positions with no gap; the {n_all - n_gap} with none or a reverse gap take its rate): the unpruned deepening's best at D is above v in about {100.0 * (n_gap * g_rate + (n_all - n_gap) * c_rate) / n_all:.1f} pct of positions (gap positions {100.0 * g_rate:.1f}, no-gap {100.0 * c_rate:.1f}); best_count's exact gap {100.0 * n_gap / n_all:.1f} pct")
    say(f"positions read {len(rs)} of {len(load(root))} audited (errors {len(load(root)) - len(rs)}); the audit's deepening gives v again in {sum(r['same_v'] for r in rs)} of them")
    for role in ("gap", "control"):
        for kind in (("any",) if role == "control" else ("any", "unit", "less", "ends")):
            sub = [r for r in rs if r["role"] == role and (kind == "any" or r["kind"] == kind)]
            if not sub:
                continue
            g = np.array([r["game"] for r in sub])
            al = np.ones(len(sub), bool)
            f = lambda fn: np.array([float(fn(r)) for r in sub])
            say(f"[{role} {kind}] positions {len(sub)}; the plain deepening's best at D is above v {cell(f(lambda r: r['plain_above_v']), g, al)} pct, differs from v {cell(f(lambda r: r['plain_differs']), g, al)} pct; the best exact move's value under the plain reading reaches the best exact value {cell(f(lambda r: r['plain_reaches']), g, al)} pct")
            if role == "gap":
                say(f"[{role} {kind}] pruning the best move's reading passed through: " + ", ".join(f"{c} {cell(f(lambda r, c=c: r['cause'] == c), g, al)} pct" for c in CAUSES))
                for band in BC.LF.BANDS:
                    sb = np.array([r["band"] == band for r in sub])
                    if sb.sum():
                        say(f"[{role} {kind} {band}] " + ", ".join(f"{c} {cell(f(lambda r, c=c: r['cause'] == c), g, sb)}" for c in CAUSES))


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "count"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--tag-gap", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    say = Say(out_root(a.tag), "which-" + a.cmd)
    gap_tag = a.tag_gap or a.tag
    if a.cmd == "run":
        run(a.tag, gap_tag, a.workers, a.limit, say)
    else:
        cmd_count(a.tag, gap_tag, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
