"""PLAN 78.77 (b), SYSTEM_SPEC 4 ("how a line ends": the settled end against the search's horizon) and 6 (the judge at the line's end): THE BEST LINE AND THE SEARCH AT THE ROOT, a diagnosis as a measurement. A sandbox: no player is
changed, nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/ladder_78_77/fault_b/. Game-free: the game is named only through odds_measure and the readers that exist.

QUESTION. The best line the line read prints (read_lines mode 8) often starts with a move the search (solve_equals_batch) does not hold among its equal best moves, under the same record. Why: depth or fuse, what counts as the end,
the order among equals, the side's view, the game's record of positions, or a bug?
INTERFACE (what it takes, what it returns, who calls it).
  probe(job) -> dict   one position at one clock (the ladder's clocks: 1 s is rung 3's first, scale index 0): the search's equal set at that clock; the line of mode 8 at the ladder's depth and fuse (the print before 78.77); the line of
                       mode 9 (the line that follows the search's own move, the print after); and, where the mode-8 line's first move is outside the equal set, three re-reads that each change ONE thing: the line read to the
                       search's finished depth with a large fuse (depth and fuse), the search read without the game's record of positions (the record), and both. Returns the booleans and the values.
  run / report         the Pool and the OBS lines. Called by: the command line only.
  .venv/bin/python -m catspace.incubator.analyst.discovery.line_root_measure run|report [--workers W] [--limit N] [--clocks 1,4]
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import multiprocessing
import time
from collections import Counter
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import ladder as LD
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import tiebreak_measure as TM
from catspace.io import paths

OUT = ("ladder_78_77", "fault_b")
BIG_FUSE = 4_000_000                                                   # the fuse of the matched re-read (a stated cap of this measurement)
_REC = None


def _equals(moves: list, seconds: float, worth: dict, record: bool) -> dict:
    r = CE.solve_equals_batch([(OM.VARIANT, None, moves)], seconds, worth=dict(worth), ours_first=False, valuation="odds record", threads=1, ledger_share=TM.LEDGER_SHARE, record=record)[0]
    return {"move": r[0], "equals": list(r[5]), "depth": r[7], "value": r[8], "cut": r[11]}


def _line(moves: list, mode: int, depth: int, fuse: int, worth: dict) -> dict:
    r = CE.read_lines(OM.VARIANT, [(None, moves)], [(0, mode, 0, 1, 0, 0, 0)], dict(worth), depth, fuse, TC.END_PLIES, TC.END_FUSE, 1)[0]
    rd = (r.get("reads") or [{}])[0] if r.get("ok") else {}
    return {"names": rd.get("names", []), "value": rd.get("value"), "depth": rd.get("depth"), "cut": rd.get("cut")}


def probe(job: tuple) -> dict:
    global _REC
    pid, moves, clock = job
    if _REC is None:
        _REC = TC.Record.load(None)
    w = _REC.worth
    i = list(LD.LIMITS["ladder"]).index(clock)
    t0 = time.perf_counter()
    s = _equals(moves, clock, w, True)
    old = _line(moves, 8, TC.DEPTH + 2 * i, TC.FUSE * 4 ** i, w)
    new = _line(moves + [s["move"]], 9, TC.DEPTH + 2 * i + 1, TC.FUSE * 4 ** i, w) if s["move"] else {"names": [], "value": None, "depth": None, "cut": None}
    first = old["names"][0] if old["names"] else None
    out = {"id": pid, "clock": clock, "search_move": s["move"], "equals": s["equals"], "search_depth": s["depth"], "search_value": s["value"], "search_cut": s["cut"],
           "old_first": first, "old_value": old["value"], "old_depth": old["depth"], "old_cut": old["cut"],
           "new_first": new["names"][0] if new["names"] else None, "new_value": new["value"], "new_depth": new["depth"], "new_cut": new["cut"],
           "old_in": first in s["equals"], "new_in": bool(new["names"]) and new["names"][0] in s["equals"]}
    if not out["old_in"] and s["depth"]:
        deep = _line(moves, 8, int(s["depth"]) - 1, BIG_FUSE, w)                      # depth and fuse matched to the search's finished depth
        out["deep_first"], out["deep_in"], out["deep_depth"], out["deep_cut"] = (deep["names"][0] if deep["names"] else None), bool(deep["names"]) and deep["names"][0] in s["equals"], deep["depth"], deep["cut"]
        nh = _equals(moves, clock, w, False)                                          # the search without the game's record of positions
        out["nohist_equals"], out["nohist_in"] = nh["equals"], first in nh["equals"]
        out["deep_nohist_in"] = bool(deep["names"]) and deep["names"][0] in nh["equals"]
    out["wall"] = time.perf_counter() - t0
    return out


def run(rows: list, clocks: list, workers: int, root: Path, say) -> list:
    out_file = root / "probe.jsonl"
    done = {}
    if out_file.exists():
        for ln in out_file.read_text().splitlines():
            if ln.strip():
                d = json.loads(ln)
                done[(d["id"], d["clock"])] = d
    jobs = [(r["id"], r["moves"], c) for r in rows for c in clocks if (r["id"], c) not in done]
    say(f"load average {os.getloadavg()[0]:.2f}; {len(jobs)} probes ({len(rows)} positions, clocks {clocks}); {workers} workers")
    t0 = time.perf_counter()
    with open(out_file, "a") as f, multiprocessing.get_context("spawn").Pool(workers) as pool:
        for n, d in enumerate(pool.imap_unordered(probe, jobs, chunksize=1)):
            done[(d["id"], d["clock"])] = d
            f.write(json.dumps(d) + "\n")
            f.flush()
            if (n + 1) % 20 == 0:
                say(f"progress: {n + 1} of {len(jobs)}, {time.perf_counter() - t0:.0f} s")
    return list(done.values())


def report(res: list, say) -> None:
    for clock in sorted({d["clock"] for d in res}):
        sub = [d for d in res if d["clock"] == clock]
        n = len(sub)
        old_in = sum(d["old_in"] for d in sub)
        new_in = sum(d["new_in"] for d in sub)
        say(f"BEFORE at {clock:g} s: the best line's first move (mode 8, the ladder's depth and fuse) is inside the search's equal set in {old_in} of {n} positions ({100 * old_in / n:.1f} pct)")
        say(f"AFTER at {clock:g} s: the line that follows the search's own move (mode 9) starts inside the equal set in {new_in} of {n} ({100 * new_in / n:.1f} pct)")
        same_v = sum(1 for d in sub if d["old_value"] == d["search_value"])
        same_v9 = sum(1 for d in sub if d["new_value"] == d["search_value"])
        say(f"VALUE at {clock:g} s: the line's value equals the search's ledger value for mode 8 in {same_v} of {n}, for mode 9 in {same_v9} of {n} (the two readings differ in depth, so a difference is not an error)")
        out = [d for d in sub if not d["old_in"]]
        if out:
            say(f"OUTSIDE at {clock:g} s ({len(out)}): mode 8 read cut by its fuse {sum(bool(d['old_cut']) for d in out)}; the search's finished depth {sum(d['search_depth'] or 0 for d in out) / len(out):.1f} on average against the line's {sum(d['old_depth'] or 0 for d in out) / len(out):.1f}")
            c = Counter((bool(d.get("deep_in")), bool(d.get("nohist_in")), bool(d.get("deep_nohist_in"))) for d in out)
            say(f"OUTSIDE at {clock:g} s, re-read changing one thing: with the depth and the fuse matched the first move is inside the equal set in {sum(1 for d in out if d.get('deep_in'))} of {len(out)}; "
                f"with the search read without the game's record of positions {sum(1 for d in out if d.get('nohist_in'))}; with both {sum(1 for d in out if d.get('deep_nohist_in'))}; none of the three in {c[(False, False, False)]}")
            forced = [d for d in out if d["search_depth"] is None]
            rest = [d for d in out if d["search_depth"] is not None and not d.get("deep_in") and not d.get("nohist_in")]
            say(f"OUTSIDE at {clock:g} s, by cause: the line read to the search's finished depth with a large fuse is inside the equal set in {sum(1 for d in out if d.get('deep_in'))} (the line stopped at a shallower depth than the search's); "
                f"the search decided by the forcing read and has no ledger depth, so its set is the one forced move, in {len(forced)} (the forcing read is a separate operator that looks for the end of the game on its own; the line read ends in a win by its own first move in {sum(1 for d in forced if (d['old_value'] or 0) >= 1 << 39)} of them and does not see the end in the rest); "
                f"neither in {len(rest)}")
            say(f"OUTSIDE at {clock:g} s: the matched re-read was itself cut by the fuse in {sum(1 for d in out if d.get('deep_cut'))}; the search's equal set was cut by the clock in {sum(1 for d in out if d['search_cut'])}")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("run", "report"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clocks", default="1,4")
    ap.add_argument("--open", action="store_true", help="only the positions the 78.76 run of the generation-2 record left at rung 3 or open (the ones that reached the longer clocks)")
    a = ap.parse_args(argv)
    root = Path(paths.experiment(*OUT))
    root.mkdir(parents=True, exist_ok=True)

    def say(m: str) -> None:
        line = f"OBS line_root {a.stage}: {m}"
        print(line, flush=True)
        with open(root / "obs.txt", "a") as f:
            f.write(line + "\n")
    rows = [json.loads(x) for x in Path(paths.experiment("ladder_78_68", "full", "positions.jsonl")).read_text().splitlines() if x.strip()]
    if a.open:
        reached = {d["id"] for d in (json.loads(x) for x in Path(paths.experiment("ladder_78_76", "gen2", "results.jsonl")).read_text().splitlines() if x.strip()) if d["result"] and d["result"]["rung"] in (0, 3)}
        rows = [r for r in rows if r["id"] in reached]
    if a.limit:
        rows = rows[:a.limit]
    clocks = [float(c) for c in a.clocks.split(",")]
    if a.stage == "run":
        res = run(rows, clocks, a.workers, root, say)
    else:
        res = [json.loads(x) for x in (root / "probe.jsonl").read_text().splitlines() if x.strip()]
    keep = {r["id"] for r in rows}
    report([d for d in res if d["id"] in keep and d["clock"] in clocks], say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
