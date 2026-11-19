"""PLAN 79.01 PART THREE, THE DEBT OF PLAN 78.84 IN THE FIRM ROOM (SYSTEM_SPEC 4 the equal reading and the verified choice, 12 the debt of plan 78.84: the equal reading holds a move that the same reading with the prunings off puts under its own best; piece: tactical reasoning, a MEASURE). No engine change, no player
changed, no switch turned on, nothing written to rules/learned or data/derived; outputs under artifacts/experiments/leaf_79_01/<tag>/ (debt/, debt.jsonl). It never calls the oracle: it reads the firm labels (leaf_firm, leaf_ask's second asking) and 79.00's trace (cause of each class-losing equal move).

THE QUESTION. For the cause-(b) moves of 79.00 in the firm room (the class-losing equal moves whose cause was the pruned value too low: stack_causes), does asking the equal question of each candidate with the prunings off, at the same depth, inside the time the verified choice now spends, put them out, and at what price in
keeping moves? Two ways of asking, both under the stack's judge (the human-result record) and both read for EVERY equal move of every firm-room position (so the price is counted on the firmly keeping moves):
  AUDIT   at the depth D of the position's run (stored by 79.00): equal_trace.Tracer.audit (cs_engine.equal_audit: the deepening finishes D by node budgets alone; every root move is asked "at least the value of the position at depth D?" on a fresh table; `plain` is that answer with the late-move reduction and the null move OFF, `exact` the
          full window). A move is PUT OUT by the audit when its prunings-off answer (or its exact value) is under the value of the position: the rule 78.98 used to name cause (b). The audit's seconds are printed beside the seconds 79.00's search spent on the same position (`seconds.search` of its labels: the verified choice's time), and the share of positions where the audit fits in that time.
  CLOCK   the search of the matches at the play clock with the verified choice on (mode slot) and the prunings OFF (the late-move reduction and the null move off, the equal reading the same call), at the clock the verified choice spends (TM.SECONDS, ledger share TM.LEDGER_SHARE, one thread): the moves it holds equal; a move PUT OUT is an equal move of 79.00's set that
          is not in this set. A read cut by the clock or failed has no answer (counted, never read as putting out). The CONTROL is the same call with the prunings ON (79.00's own search again): the moves that leave by noise.
THE COUNTS: of the cause-(b) moves of the firm room, of the firmly losing moves in all, and of the firmly keeping moves (the price): the share put out, by each way; the positions where the question puts out every firmly keeping move (nothing is left to play); 90 pct bands by games resampled whole.

INTERFACE (written before the bodies).
  items_of(tag, src) -> [(rec, moves)]                   leaf_line.items_of (the firm room).
  put_out_audit(row, v) / put_out_exact(row, v)          pure; tested.
  put_out_clock(tied, answer) -> [bool | None]            per equal move: left the set of the answer; None when the answer is cut, failed or missing (pure; tested).
  Debt()                                                   per process: the stack's tracer and the record's worths. `.read(moves, depth, tied) -> dict`.
  run(tag, src, workers, limit, say) / load(root) / cmd_count(tag, say)   the Pool (the clocked reads wait for a quiet machine as 78.97 did), one file a chunk; debt.jsonl; the counts.
  Called by: the command line.

  .venv/bin/python -m catspace.incubator.analyst.discovery.leaf_debt run --tag short --limit 40 | run --tag full | count --tag full
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

from catspace.incubator.analyst.discovery import leaf_ask as LA
from catspace.incubator.analyst.discovery import leaf_firm as LF
from catspace.incubator.analyst.discovery import leaf_line as LL
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import prior_measure as PMS
from catspace.incubator.analyst.discovery import record_play as RP
from catspace.incubator.analyst.discovery import stack_causes as SC
from catspace.incubator.analyst.discovery import stack_room as SR
from catspace.io import paths

CHUNK = 6
MIN_CELL = 30


def put_out_audit(row: "dict | None", v: int) -> "bool | None":
    if row is None or row.get("plain") is None:
        return None
    return bool(row["plain"] < v)


def put_out_exact(row: "dict | None", v: int) -> "bool | None":
    if row is None or row.get("exact") is None:
        return None
    return bool(row["exact"] < v)


def put_out_clock(tied: list, answer: "dict | None") -> list:
    if not answer or answer.get("error") or answer.get("cut") or answer.get("first") is None:
        return [None] * len(tied)
    inside = set(answer["equals"]) | {answer["first"]}
    return [m not in inside for m in tied]


class Debt(SC.StackTracer):
    """See the module doc."""

    def ask(self, moves: list, off: bool) -> dict:
        import cs_engine as CE
        from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
        kw = {"null_move": False, "late_reduce": False} if off else {}
        t0 = time.perf_counter()
        try:
            raw = CE.solve_equals_batch([(OM.VARIANT, None, list(moves))], TM.SECONDS, worth=dict(self.worth), ours_first=False, valuation="odds record", threads=1, ledger_share=TM.LEDGER_SHARE, verify_choice=True, choice_mode="slot", **kw)[0]
        except Exception as e:                                         # a read the search cannot take is counted, never silently
            return {"first": None, "equals": [], "depth": None, "cut": False, "error": f"{type(e).__name__}: {str(e)[:120]}", "seconds": time.perf_counter() - t0}
        row = PM.equal_set(raw)
        return {"first": row["move"], "equals": row["equals"], "depth": row["depth"], "cut": bool(row["cut"]), "error": row["error"] or None, "state": int(raw[12] or 0), "seconds": time.perf_counter() - t0}

    def read(self, moves: list, depth: int, tied: list) -> dict:
        t0 = time.perf_counter()
        a = self.audit(moves, depth, depth)
        secs = time.perf_counter() - t0
        out = {"audit_seconds": secs}
        if a is None:
            return dict(out, audit=None)
        rows = {r["name"]: r for r in a["rows"]}
        out["audit"] = {"value": int(a["value"]), "finished": int(a["depth"]), "rows": {m: ({"warm": rows[m]["warm_probe"], "plain": rows[m]["plain_probe"], "exact": rows[m]["exact"]} if m in rows else None) for m in tied}}
        out["off"] = self.ask(moves, True)
        out["on"] = self.ask(moves, False)
        return out


_DB = None


def _init_worker() -> None:
    global _DB
    _DB = Debt()


def _job(args: tuple) -> tuple:
    idx, items, root = args
    f = Path(root) / "debt" / f"c{idx:05d}.json"
    if f.exists() and "\"error\"" not in f.read_text():
        return idx, 0.0
    t0 = time.perf_counter()
    out = []
    for rec, moves in items:
        try:
            out.append(dict(_DB.read(moves, rec["depth"], rec["tied"]), id=rec["id"]))
        except Exception as e:
            out.append({"id": rec["id"], "error": f"{type(e).__name__}: {str(e)[:150]}"})
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0


def load(root: Path) -> list:
    return [json.loads(x) for x in (root / "debt.jsonl").read_text().splitlines() if x.strip()]


def run(tag: str, src: str, workers: int, limit: int, say) -> None:
    root = LF.out_root(tag)
    (root / "debt").mkdir(exist_ok=True)
    items = LL.items_of(tag, src)
    if limit:
        items = items[:limit]
    say(f"the firm room: {len(items)} positions (margin {LF.load_margin(tag):g}); every equal move asked by the AUDIT at D (node budgets) and by the CLOCK reads at {TM.SECONDS} s, ledger share {TM.LEDGER_SHARE}, verify on, prunings off and (control) on; judge the human-result record {RP.worth_of(SR.RECORD)}; {workers} workers of {os.cpu_count()} cores")
    load0, busy = TM.wait_quiet(lambda s: say(s.replace("OBS ", "", 1)))
    say(f"load average at the start {os.getloadavg()[0]:.2f}")
    jobs = [(a // CHUNK, items[a:a + CHUNK], str(root)) for a in range(0, len(items), CHUNK)]
    t0 = time.perf_counter()
    done = 0
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for _i, _dt in pool.imap_unordered(_job, jobs, chunksize=1):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                say(f"progress: {done} of {len(jobs)} chunks; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    out = []
    for f in sorted((root / "debt").glob("c*.json")):
        out += json.loads(f.read_text())
    (root / "debt.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    say(f"read {len(out)} of {len(items)} positions in {time.perf_counter() - t0:.0f} s wall ({sum(1 for r in out if 'error' in r)} errors); load average at the end {os.getloadavg()[0]:.2f}")


def cmd_count(tag: str, say) -> None:
    root = LF.out_root(tag)
    margin = LF.load_margin(tag)
    recs = {r["id"]: r for r in LA.src_records()}
    asks = {a["id"]: a for a in LA.load(root)}
    debt = {r["id"]: r for r in load(root) if "error" not in r and r.get("audit")}
    trace = {}
    for r in SC.load(Path(paths.experiment(*SR.OUT, LA.SRC))):
        if "error" not in r:
            trace.setdefault(r["id"], {})[r["stuck"]] = r["cause"]
    moves = []                                                       # one row per equal move of the firm room
    for i, d in debt.items():
        r, a = recs[i], asks[i]
        vec = LF.firm_vec(r, a, margin)
        off, on = put_out_clock(r["tied"], d.get("off")), put_out_clock(r["tied"], d.get("on"))
        for j, m in enumerate(r["tied"]):
            row = d["audit"]["rows"].get(m)
            moves.append({"id": i, "game": r["ord"], "band": r["band"], "part": LF.part_of(r["ord"]), "label": vec[j], "cause": trace.get(i, {}).get(m), "audit": put_out_audit(row, d["audit"]["value"]), "exact": put_out_exact(row, d["audit"]["value"]), "off": off[j], "on": on[j]})
    say(f"MARGIN {margin:g}; firm-room positions read {len(debt)} of the firm room; equal moves {len(moves)}: firmly losing {sum(1 for m in moves if m['label'] == 'lose')}, firmly keeping {sum(1 for m in moves if m['label'] == 'keep')}, neither {sum(1 for m in moves if m['label'] is None)}; of the firmly losing, cause (b) in 79.00's trace {sum(1 for m in moves if m['label'] == 'lose' and m['cause'] == 'b')}")
    game = np.array([m["game"] for m in moves])
    sets = (("cause-(b) firmly losing moves", lambda m: m["label"] == "lose" and m["cause"] == "b"), ("firmly losing moves (all)", lambda m: m["label"] == "lose"), ("firmly keeping moves (the price)", lambda m: m["label"] == "keep"))
    for part in ("find", "confirm", "both"):
        for name, f in sets:
            sel = np.array([f(m) and (part == "both" or m["part"] == part) for m in moves])
            n = int(sel.sum())
            if n == 0:
                say(f"[{part}] {name}: none")
                continue
            parts = []
            for way, label in (("audit", "the audit (prunings off, same depth)"), ("exact", "the exact value at that depth"), ("off", "the clocked read, prunings off"), ("on", "the clocked read, prunings on (noise control)")):
                ans = np.array([m[way] is not None for m in moves])
                v = np.array([bool(m[way]) for m in moves], float)
                s = sel & ans
                parts.append(f"{label}: put out {PMS.fmt(PMS.share(v[s], game[s])) if int(s.sum()) >= MIN_CELL else (f'{100.0 * float(v[s].mean()):.1f} (n {int(s.sum())}, too few for a band)' if int(s.sum()) else 'no answer')} pct of {int(s.sum())} answered")
            say(f"[{part}] {name} ({n}): " + "; ".join(parts))
    allout = {}
    for way in ("audit", "off"):
        k = tot = 0
        for i in debt:
            keeps = [m for m in moves if m["id"] == i and m["label"] == "keep" and m[way] is not None]
            if not keeps:
                continue
            tot += 1
            k += all(m[way] for m in keeps)
        allout[way] = (k, tot)
    say(f"positions where the question puts out EVERY firmly keeping move (nothing is left to play that keeps the class): the audit {allout['audit'][0]} of {allout['audit'][1]}, the clocked read with the prunings off {allout['off'][0]} of {allout['off'][1]}")
    a_s = np.array([debt[i]["audit_seconds"] for i in debt])
    s_s = np.array([recs[i]["seconds"]["search"] for i in debt])
    say(f"time (one thread's seconds a position): the audit {a_s.mean():.3f}, the search of 79.00 on the same positions (the verified choice's time) {s_s.mean():.3f}; the audit fits inside that time in {int((a_s <= s_s).sum())} of {len(a_s)} positions; the clocked reads (prunings off) {np.mean([debt[i]['off']['seconds'] for i in debt]):.3f}, depth {np.mean([debt[i]['off']['depth'] for i in debt if debt[i]['off'].get('depth') is not None]):.2f} against the control (prunings on) {np.mean([debt[i]['on']['depth'] for i in debt if debt[i]['on'].get('depth') is not None]):.2f}; "
        f"cut by the clock: off {sum(1 for i in debt if debt[i]['off'].get('cut'))}, on {sum(1 for i in debt if debt[i]['on'].get('cut'))}")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "count"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--src", default=LA.SRC)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args(argv)
    say = LA.Say(LF.out_root(a.tag), "debt " + a.cmd)
    if a.cmd == "run":
        run(a.tag, a.src, a.workers, a.limit, say)
    else:
        cmd_count(a.tag, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
