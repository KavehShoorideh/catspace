"""PLAN 79.02 PART ONE, THE COUNT OF WHAT THE EQUAL SET HOLDS WHEN THE READING IS CUT OR PRUNED (SYSTEM_SPEC 4 the equal reading and the verified choice, 12 the debt of plan 78.84; piece: tactical reasoning, a MEASURE). No engine change, no player changed, no switch turned on, nothing written to rules/learned or data/derived; outputs
under artifacts/experiments/cut_79_02/<tag>/. It never calls the oracle: it reads 79.00's labels (stack_room's equals.jsonl, the class after each equal move) and 79.01's second asking (leaf_ask's ask.jsonl, the firm label of leaf_firm), and it asks the engine's audit (node budgets, no clock) once for every position.

WHAT THE CODE DOES WITH A ROOT MOVE WHEN THE EQUAL READING IS CUT BY THE CLOCK, AND BY WHICH KIND OF ANSWER A MOVE IS HELD (read from rust/engine/tactics/src/ledger.rs `Search::ties`, `probes`, and the binding's `EqualsLedger`/`equals_one`, before any number was looked at):
  The deepening finishes a depth D with the value v of the root. Then every root move is asked "at least v at depth D?" by a null window (`probes`), one after the other, in a fixed reading order (the moves that gain first, the best gain first, then the rest by their number in the game's list: `Search::ordered`, which is NOT
  the deepening's own order of moves), on the table the deepening left warm, WITH the late-move reduction and the null move on. A move is EQUAL iff its answer is v or more. The reading has a slot of the clock (`ledger_share`); when the clock stops it, `probes` stops at once: the moves it had answered are the answers, the move being asked is
  dropped, the moves not yet asked are not asked. The moves in the set are therefore only moves that were asked and held; a move not asked is NOT in the set (the set is too small, not unsound), with one exception: the binding adds the move the search plays to the list whatever the reading did (`equals_one`: `idx.push(k)`),
  so the first move is in the set whether it was asked before the cut or not; its own value is the deepening's, v by construction. A cut is told apart by the part named "ties (cut)" (the 12th element of the row). With the switch verify_ties on, the held moves are asked once more on a fresh table with both prunings off and a cut leaves only
  the moves verified so far (78.85; off by default). So a move is HELD, in play, by one kind of answer: the null-window answer of the search's own pruned reading, a bound that the table stores as if it were full (the late-move reduction reads a late quiet reply two plies shallower; the null move reads a pass shallow); the move the
  search plays is held by the deepening's own value (and, with the verified choice on, by the same question asked once without the prunings).

THE COUNT. Of every equal move of a position (the set of 79.00's run at the play clock under the stack: the human-result record, the verified choice on), and of the moves the same reading holds when it is not cut (the audit):
  FIRST    the move the search plays; ASKED (the position is not cut, or the move's place in the reading order is not after the last other member of the set) or UNSHOWN (the position is cut and nothing shows the reading reached it: it is in the set because it is the move played). A bound, not a count: the reading's place at the cut is not
           stored; the audit gives the order and the set gives the last move certainly asked.
  OTHER    a move of the set that is not the first: asked and held. Held by an answer that the reading with both prunings off confirms (PLAIN), or held ONLY BY A REDUCED READING (its answer with both prunings off is under v: LATE if the late-move reduction alone puts it out, NULL if the null move alone does, BOTH if either does, JOINT if only the two together); UNREAD if the audit could not answer.
  DROPPED  a move the untimed reading holds (warm answer at least v) that is not in the set: the cut's cost in a cut position, the noise of the clock in a position that is not cut (the control).
  Each class is counted on the firmly losing and the firmly keeping equal moves (leaf_firm: the firm room of 79.01, FIND to choose and CONFIRM to report, split by game), by band, with 90 pct bands by resampling games; and on ALL sampled positions of 79.00 with the first asking's class (keeps or not: the second asking exists only for the room).

INTERFACE (written before the bodies).
  hold_kind(row, v) -> str                    "none" | "plain" | "late" | "null" | "both" | "joint" | "unread": how the move's answer in the set is held, from an audit row {warm, no_late, no_null, plain, exact} (pure; tested).
  first_state(order, members, first, cut) -> str   "asked" | "unshown" | "unknown" (pure; tested).
  dropped(rows, v, members) -> [name]         the moves the audit holds that are not in the set (pure; tested).
  equal_moves(rec, aud, vec) -> [dict]        one row per equal move of a position: its class, firm label, the hold kind and the state (pure; tested).
  CutTracer()                                 per process: the stack's tracer. `.read(moves, depth) -> dict` the audit of the position at D (the warm, the no-late, the no-null, the plain answer, the exact value of every root move, in the reading order).
  run(tag, src, workers, limit, say) / load(root) / cmd_count(tag, say)   the Pool over the cores (node budgets, untimed), one file a chunk; audit.jsonl; the counts.
  Called by: the command line, ladder_print --cut02 (reads the files), the mend's measure (cut_mend.py) for the sets of the unmended reading.

  .venv/bin/python -m catspace.incubator.analyst.discovery.cut_count run --tag short --limit 60 | run --tag full | count --tag full
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

from catspace.incubator.analyst.discovery.cut_out import OUT, Say, out_root                  # (the output folder and the OBS lines: a helper that loads nothing of the oracle)
from catspace.incubator.analyst.discovery import leaf_ask as LA
from catspace.incubator.analyst.discovery import leaf_firm as LF
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import prior_measure as PMS
from catspace.incubator.analyst.discovery import stack_causes as SC
from catspace.incubator.analyst.discovery import stack_room as SR

CHUNK = 8
MIN_CELL = 30
HOLDS = ("plain", "late", "null", "both", "joint", "unread")
REDUCED = ("late", "null", "both", "joint")


STATEMENT = ("WHAT THE CODE DOES ON A CUT, in plain words (from ledger.rs Search::ties/probes and the binding's equals_one, before any number): the equal reading asks the root moves one after the other, in a fixed order (the moves that gain first, then the rest by their number), each \"at least the value v at the depth D the "
             "deepening finished?\" by a null window on the warm table with the late-move reduction and the null move on; a move is equal iff the answer is v or more. When the clock of the slot stops it, `probes` stops: the moves answered stay, the one being asked and every later one are dropped, so a move not asked is NOT in the set (the set is too small, "
             "not unsound); the only exception is the move the search plays, which the binding adds to the set whatever the reading did. A move is held by ONE kind of answer: the pruned null-window answer (a reduced bound the table stores as a full one); the played move by the deepening's own value (and by one unpruned asking when the verified choice is on).")


# ------------------------------------------------------------------------------------------------------------------------------ pure parts

def hold_kind(row: "dict | None", v: int) -> str:
    """How a move that the search's own reading holds is held, by the audit row: PLAIN when the answer with both prunings off is v or more too; else the pruning that puts it out."""
    if row is None or row.get("warm") is None:
        return "unread"
    if row["warm"] < v:
        return "none"
    if row.get("plain") is None:
        return "unread"
    if row["plain"] >= v:
        return "plain"
    nl, nn = row.get("no_late"), row.get("no_null")
    if nl is None or nn is None:
        return "both"
    late, null = nl < v, nn < v
    if late and null:
        return "both"
    if late:
        return "late"
    if null:
        return "null"
    return "joint"


def first_state(order: list, members: list, first: str, cut: bool) -> str:
    """Was the played move asked before the clock stopped the reading? Not cut: asked. Cut: asked when some other member of the set comes after it in the reading order (the reading reached that member, so it had passed the first); else unshown."""
    if not cut:
        return "asked"
    if first not in order:
        return "unknown"
    at = order.index(first)
    later = [order.index(m) for m in members if m != first and m in order]
    return "asked" if later and max(later) > at else "unshown"


def dropped(rows: list, v: int, members: list) -> list:
    return [r["name"] for r in rows if r["warm"] is not None and r["warm"] >= v and r["name"] not in members]


def equal_moves(rec: dict, aud: dict, vec: "list | None") -> list:
    """One row per equal move of the position's set. `vec` is the firm label of each (None where the position has none: the second asking is the room's): `label` is the firm one (keep, lose, None), `l1` the first asking's (keep or lose)."""
    rows = {r["name"]: r for r in aud["rows"]}
    order = [r["name"] for r in aud["rows"]]
    v = aud["value"]
    fs = first_state(order, rec["tied"], rec["first"], bool(rec["cut"]))
    out = []
    for j, m in enumerate(rec["tied"]):
        is_first = m == rec["first"]
        out.append({"id": rec["id"], "game": rec["ord"], "band": rec["band"], "part": LF.part_of(rec["ord"]), "move": m, "first": is_first, "cut": bool(rec["cut"]),
                    "label": (vec[j] if vec else None), "l1": ("keep" if PE.keeps(rec["cls"], rec["cm"][j]) else "lose"),
                    "kind": ("first-" + fs) if is_first else ("other-" + hold_kind(rows.get(m), v)), "hold": hold_kind(rows.get(m), v)})
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the audit

class CutTracer(SC.StackTracer):
    """See the module doc."""

    def read(self, moves: list, depth: int) -> dict:
        a = self.audit(moves, depth, depth)
        if a is None:
            return {"audit": None}
        rows = [{"name": r["name"], "gain": r["gain"], "warm": r["warm_probe"], "cold": r["cold_probe"], "no_null": r["no_null_probe"], "no_late": r["no_late_probe"], "plain": r["plain_probe"], "exact": r["exact"]} for r in a["rows"]]
        return {"audit": {"value": int(a["value"]), "finished": int(a["depth"]), "tie_cut": bool(a["tie_cut"]), "rows": rows}}


_TR = None


def _init_worker() -> None:
    global _TR
    _TR = CutTracer()


def _job(args: tuple) -> tuple:
    idx, items, root = args
    f = Path(root) / "audit" / f"c{idx:05d}.json"
    if f.exists() and "\"error\"" not in f.read_text():
        return idx, 0.0
    t0 = time.perf_counter()
    out = []
    for rec, moves in items:
        try:
            out.append(dict(_TR.read(moves, rec["depth"]), id=rec["id"]))
        except Exception as e:                                         # a position the audit cannot take is counted, never silently
            out.append({"id": rec["id"], "error": f"{type(e).__name__}: {str(e)[:150]}"})
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0


def load(root: Path) -> list:
    return [json.loads(x) for x in (root / "audit.jsonl").read_text().splitlines() if x.strip()]


def items_of(src: str, limit: int = 0) -> list:
    """Every position of 79.00's sample that has a depth (the equal reading ran), in the order of the sample's hash; the firm room first when a limit is given so that a short run holds the room."""
    from catspace.incubator.analyst.discovery import future_labels as FL
    games = {g["ord"]: g for g in FL.universe()}
    recs = [r for r in LA.src_records(src) if r.get("depth") is not None]
    recs.sort(key=lambda r: (not LF.in_room(r), PE.order_key(r["ord"], r["ply"])) if limit else PE.order_key(r["ord"], r["ply"]))
    if limit:
        recs = recs[:limit]
    return [(r, games[r["ord"]]["moves"][:r["ply"]]) for r in recs]


def run(tag: str, src: str, workers: int, limit: int, say) -> None:
    root = out_root(tag)
    (root / "audit").mkdir(exist_ok=True)
    items = items_of(src, limit)
    say(f"audit of {len(items)} positions of 79.00's sample (those with a depth; node budgets, no clock; the stack's judge, the human-result record); {workers} workers of {os.cpu_count()} cores; load average at the start {os.getloadavg()[0]:.2f}")
    jobs = [(a // CHUNK, items[a:a + CHUNK], str(root)) for a in range(0, len(items), CHUNK)]
    t0 = time.perf_counter()
    done = 0
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for _i, _dt in pool.imap_unordered(_job, jobs, chunksize=1):
            done += 1
            if done % 25 == 0 or done == len(jobs):
                say(f"progress: {done} of {len(jobs)} chunks; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    out = []
    for f in sorted((root / "audit").glob("c*.json")):
        out += json.loads(f.read_text())
    (root / "audit.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    say(f"read {len(out)} of {len(items)} positions in {time.perf_counter() - t0:.0f} s wall ({sum(1 for r in out if 'error' in r)} errors, {sum(1 for r in out if r.get('audit') is None and 'error' not in r)} with no audit); load average at the end {os.getloadavg()[0]:.2f}")


# ------------------------------------------------------------------------------------------------------------------------------ the counts

def cell(vals: np.ndarray, game: np.ndarray, sel: np.ndarray) -> str:
    n = int(sel.sum())
    if n < MIN_CELL:
        return f"{100.0 * float(vals[sel].mean()):.1f} (n {n}, too few for a band)" if n else "none (n 0)"
    return PMS.fmt(PMS.share(vals[sel], game[sel])) + f" (n {n})"


def gather(tag: str, src: str) -> dict:
    """Rows of every position and every equal move: {"moves": [...], "positions": [...]}; the firm label where the position is in the firm room."""
    root = out_root(tag)
    margin = LF.load_margin("full")
    recs = {r["id"]: r for r in LA.src_records(src)}
    asks = {a["id"]: a for a in LA.load(LF.out_root("full"))}
    aud = {r["id"]: r for r in load(root) if "error" not in r and r.get("audit")}
    moves, positions = [], []
    for i, a in aud.items():
        r = recs[i]
        vec = LF.firm_vec(r, asks[i], margin) if i in asks else None
        firm = bool(vec) and LF.is_firm(vec)
        ms = equal_moves(r, a["audit"], vec)
        drop = dropped(a["audit"]["rows"], a["audit"]["value"], r["tied"])
        rows = {x["name"]: x for x in a["audit"]["rows"]}
        positions.append({"id": i, "game": r["ord"], "band": r["band"], "part": LF.part_of(r["ord"]), "cut": bool(r["cut"]), "firm": firm, "room": LF.in_room(r), "n": len(r["tied"]), "dropped": len(drop),
                          "dropped_reduced": sum(1 for m in drop if hold_kind(rows[m], a["audit"]["value"]) not in ("plain", "none")), "first_state": next(x["kind"] for x in ms if x["first"]) if any(x["first"] for x in ms) else "unknown",
                          "reduced": sum(1 for x in ms if x["hold"] in REDUCED and not x["first"]), "seconds_ties": None})
        ex = [x["exact"] for x in a["audit"]["rows"] if x["exact"] is not None]
        best_ex = max(ex) if ex else None
        positions[-1]["ref_low"] = bool(best_ex is not None and best_ex > a["audit"]["value"])
        for x in ms:
            x["firm"] = firm
            x["room"] = LF.in_room(r)
            e = rows.get(x["move"], {}).get("exact")
            x["under_best"] = bool(e is not None and best_ex is not None and e < best_ex)
            x["under_v"] = bool(e is not None and e < a["audit"]["value"])
            moves.append(x)
    return {"moves": moves, "positions": positions, "margin": margin}


def _arr(rows: list, key: str, fn=None) -> np.ndarray:
    return np.array([(fn(r[key]) if fn else r[key]) for r in rows])


def count_block(say, name: str, moves: list, positions: list, label_key: str) -> None:
    """One population: positions cut, the played move unshown, a reduced hold, the dropped moves; then each class of equal move: its share of the losing moves, of the keeping moves, and its own rate of losing."""
    pg = np.array([p["game"] for p in positions])
    pn = len(positions)
    say(f"[{name}] positions {pn}; cut by the clock {cell(np.array([float(p['cut']) for p in positions]), pg, np.ones(pn, bool))} pct; played move UNSHOWN (cut, nothing shows the reading reached it) {cell(np.array([float(p['first_state'] == 'first-unshown') for p in positions]), pg, np.ones(pn, bool))} pct; "
        f"a non-first equal move held only by a reduced reading in {cell(np.array([float(p['reduced'] > 0) for p in positions]), pg, np.ones(pn, bool))} pct")
    cutp = np.array([p["cut"] for p in positions])
    dr = np.array([float(p["dropped"] > 0) for p in positions])
    say(f"[{name}] positions where the untimed reading holds a move that is not in the set: cut positions {cell(dr, pg, cutp)} pct, positions not cut (the noise control) {cell(dr, pg, ~cutp)} pct; moves dropped a position (cut) {float(np.mean([p['dropped'] for p in positions if p['cut']] or [float('nan')])):.2f}, (not cut) {float(np.mean([p['dropped'] for p in positions if not p['cut']] or [float('nan')])):.2f}; "
        f"of the moves dropped in cut positions, held by a reduced reading (not by the plain one) {sum(p['dropped_reduced'] for p in positions if p['cut'])} of {sum(p['dropped'] for p in positions if p['cut'])}")
    lab = [m for m in moves if m[label_key] in ("keep", "lose")]
    if not lab:
        say(f"[{name}] no labelled equal moves")
        return
    g = _arr(lab, "game")
    lose = _arr(lab, label_key, lambda x: x == "lose")
    keep = ~lose
    say(f"[{name}] labelled equal moves {len(lab)}: losing {int(lose.sum())}, keeping {int(keep.sum())}; the first move loses {cell(lose.astype(float), g, _arr(lab, 'first'))} pct of {int(_arr(lab, 'first').sum())}; the other equal moves lose {cell(lose.astype(float), g, ~_arr(lab, 'first'))} pct")
    say(f"[{name}] THE REFERENCE (a reading of the value the set is held against, not of the cut): positions where some root move's exact value at D is above the value v of the position (the deepening's value is under the exact best) {cell(np.array([float(p['ref_low']) for p in positions]), pg, np.ones(pn, bool))} pct; "
        f"of the LOSING equal moves, exact value under the exact best of the position {cell(_arr(lab, 'under_best', lambda x: float(x)), g, lose)} pct, under v itself {cell(_arr(lab, 'under_v', lambda x: float(x)), g, lose)} pct; of the KEEPING equal moves, under the exact best {cell(_arr(lab, 'under_best', lambda x: float(x)), g, keep)} pct, under v {cell(_arr(lab, 'under_v', lambda x: float(x)), g, keep)} pct")
    kinds = ("first-asked", "first-unshown", "other-plain", "other-late", "other-null", "other-both", "other-joint", "other-unread")
    for k in kinds:
        inn = _arr(lab, "kind", lambda x, k=k: x == k)
        if not inn.any():
            continue
        say(f"[{name}] class {k}: {int(inn.sum())} labelled moves; of the LOSING moves in it {cell(inn.astype(float), g, lose)} pct, of the KEEPING moves {cell(inn.astype(float), g, keep)} pct; the moves in it lose {cell(lose.astype(float), g, inn)} pct")
    red = _arr(lab, "kind", lambda x: x in ("other-late", "other-null", "other-both", "other-joint"))
    oth = ~_arr(lab, "first")
    say(f"[{name}] all reduced-only holds (late, null, both, joint): of the LOSING moves {cell(red.astype(float), g, lose)} pct, of the KEEPING moves {cell(red.astype(float), g, keep)} pct; of the non-first LOSING moves {cell(red.astype(float), g, lose & oth)} pct")
    uns = _arr(lab, "kind", lambda x: x == "first-unshown")
    say(f"[{name}] played move unshown or reduced-only held, together: of the LOSING moves {cell((red | uns).astype(float), g, lose)} pct, of the KEEPING moves {cell((red | uns).astype(float), g, keep)} pct")


def cmd_count(tag: str, src: str, say) -> None:
    d = gather(tag, src)
    moves, positions = d["moves"], d["positions"]
    say(f"MARGIN {d['margin']:g} (79.01's margin.json); positions audited {len(positions)}; equal moves {len(moves)}; in the firm room {sum(1 for p in positions if p['firm'])} positions; the audit's own untimed reading cut in {sum(1 for r in load(out_root(tag)) if r.get('audit') and r['audit']['tie_cut'])}")
    for part in ("find", "confirm"):
        for band in ("all",) + LF.BANDS:
            sel = lambda r: r["part"] == part and (band == "all" or r["band"] == band)
            fm = [m for m in moves if m["firm"] and sel(m)]
            fp = [p for p in positions if p["firm"] and sel(p)]
            if fp:
                count_block(say, f"{part} {band} FIRM ROOM", fm, fp, "label")
            am = [m for m in moves if sel(m)]
            ap = [p for p in positions if sel(p)]
            if ap:
                count_block(say, f"{part} {band} ALL POSITIONS (first asking's class)", am, ap, "l1")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "count"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--src", default=LA.SRC)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    say = Say(out_root(a.tag), a.cmd)
    if a.cmd == "run":
        say(STATEMENT)
        run(a.tag, a.src, a.workers, a.limit, say)
    else:
        cmd_count(a.tag, a.src, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
