"""PLAN 79.03 PART THREE, THE MEASURE OF THE MEND (SYSTEM_SPEC 4 the tactical reasoner's deepening and its prunings, 12 the debt of plan 78.84; piece: tactical reasoning, a MEASURE of the engine's switch `root_full`, OFF by default: rust/engine/tactics/src/ledger.rs `Search::root_full`, no late-move reduction at the root).
Nothing is turned on in any default; the player of 79.00 (the human-result record, the verified choice on, 78.82's pick) is read with and without the switch passed as a parameter of the call. Nothing is written to rules/learned or data/derived; outputs under artifacts/experiments/best_79_03/<tag>/. This module NEVER calls the oracle: the labels are
79.00's class after each equal move (a move that is not in 79.00's equal set has no label and is counted as UNLABELLED).

WHAT IS READ. For every position of 79.00's sample that has a depth (9669), twice at a FIXED DEPTH (the depth 79.00 finished; node budgets, no clock) with the switch off and on (the engine's gap audit: the depth, the value v, the move chosen, the nodes, and the best root move's own full-window value at that depth), and three times at the play clock (0.2 s,
ledger share 0.4, one thread, verify_choice on, the human-result record) in this order: A (switch off), M (switch on), N (switch off again: the clock's noise, the control), each then audited at the depth it reached with its own switch. For each clocked arm: the move the search plays, the equal set, the depth, the seconds, the nodes, the pick of 78.82.
THE FAULT of a position under an arm is a gap (best root move's own value above v) of one unit of the weakest kind or more, or the game's own end (best_count.gap_kind); the share of such positions is what the mend is to lower. The class of a move is 79.00's stored label (keeps or loses the oracle's class after it); both the move played and the pick are read.

THE RULE FOR PLAYING (written by `rule` BEFORE CONFIRM is read; `count --part confirm` refuses without it): the mend is played against the unmended player iff on CONFIRM (all positions) (1) the pick keeps the class, M less A over the positions where both picks have a label, has a 90 pct band (games resampled whole) wholly above zero; (2) the share of positions
with the fault at the clock, M less A, has a band wholly below zero; (3) the mean depth reached under M is not more than one ply under A's (a named reading: the mend may not buy the root's moves with the horizon); (4) the mean seconds of a decision under M stay within the play clock; and FIND agrees with (1) and (2) in sign.

INTERFACE (written before the bodies).
  class_of(move, labels) -> "keep" | "lose" | "unlabelled"                   (pure; tested). labels_of(rec) -> {move: bool keeps}.
  fault_of(audit, unit) -> str                                                  the best_count kind of a gap audit {"value", "best_exact"} (pure; tested).
  conditions(rows) -> dict                                                      the four conditions of the rule on a set of positions (pure; tested).
  Best()                                                                        per process: cut_mend.Mend's search and pick with the switch as a parameter; `.read(moves, depth) -> {"a", "m", "n", "fixed_off", "fixed_on"}`.
  run(tag, workers, limit, say) / load(root) / cmd_rule(tag, say) / cmd_count(tag, part, say)
  Called by: the command line, ladder_print --best03 (reads mend.jsonl), the match (best_play.py) reads the rule.

  .venv/bin/python -m catspace.incubator.analyst.discovery.best_mend run --tag short --limit 80 | run --tag full | rule --tag full | count --tag full --part find|confirm
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
from catspace.incubator.analyst.discovery import cut_count as CC
from catspace.incubator.analyst.discovery import cut_mend as CM
from catspace.incubator.analyst.discovery import equal_trace as ET
from catspace.incubator.analyst.discovery import leaf_ask as LA
from catspace.incubator.analyst.discovery import leaf_firm as LF
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import prior_measure as PMS
from catspace.incubator.analyst.discovery import record_play as RP
from catspace.incubator.analyst.discovery import stack_room as SR
from catspace.incubator.analyst.discovery.best_out import Say, out_root

CHUNK = 6
MIN_CELL = 30
CLOCK = float(os.environ.get("CUT_CLOCK", TM.SECONDS))
DEPTH_LOSS_MAX = 1.0
ARMS = ("a", "m", "n")
SWITCH = {"a": False, "m": True, "n": False}


# ------------------------------------------------------------------------------------------------------------------------------ pure parts

def labels_of(rec: dict) -> dict:
    return {m: PE.keeps(rec["cls"], c) for m, c in zip(rec["tied"], rec["cm"])}


def class_of(move: "str | None", labels: dict) -> str:
    if move is None or move not in labels:
        return "unlabelled"
    return "keep" if labels[move] else "lose"


def fault_of(audit: "dict | None", unit: int) -> str:
    """The best_count kind of a gap audit {"value", "best_exact"}: none, reverse, less, unit, ends; "unread" where the audit has no exact value."""
    if not audit or audit.get("best_exact") is None:
        return "unread"
    v, b = int(audit["value"]), int(audit["best_exact"])
    return BC.gap_kind(b - v, unit, abs(b) >= BC.WIN - 1000 or abs(v) >= BC.WIN - 1000)


def is_fault(kind: str) -> bool:
    return kind in ("unit", "ends")


def conditions(rows: list) -> dict:
    """The four conditions of the rule on a set of positions (rows of `rows_of`): each a number and whether it passes (a band is the 90 pct band by games resampled whole)."""
    g = np.array([r["game"] for r in rows])
    both = [r for r in rows if r["pick"]["a"] != "unlabelled" and r["pick"]["m"] != "unlabelled"]
    gb = np.array([r["game"] for r in both])
    pick = PMS.share(np.array([float(r["pick"]["m"] == "keep") - float(r["pick"]["a"] == "keep") for r in both]), gb) if both else (float("nan"),) * 3
    fault = PMS.share(np.array([float(is_fault(r["fault"]["m"])) - float(is_fault(r["fault"]["a"])) for r in rows]), g)
    dd = float(np.mean([r["reads"]["a"]["depth"] - r["reads"]["m"]["depth"] for r in rows]))
    sec = float(np.mean([r["reads"]["m"]["seconds"] for r in rows]))
    return {"pick_gain": pick, "pick_pass": bool(pick[1] > 0), "fault_change": fault, "fault_pass": bool(fault[2] < 0), "depth_loss": dd, "depth_pass": bool(dd <= DEPTH_LOSS_MAX), "seconds": sec, "time_pass": bool(sec <= CLOCK)}


# ------------------------------------------------------------------------------------------------------------------------------ the reads

class Best(CM.Mend):
    """cut_mend.Mend's search and pick, with root_full as the parameter."""

    def gap(self, moves: list, depth: int, flag: bool) -> "dict | None":
        import cs_engine as CE
        a = CE.gap_audit(OM.VARIANT, list(moves), self.worth, ET.RUN_NODES, int(depth), ET.BIG, True, flag)
        if a is None:
            return None
        ex = [(r["exact"], r["name"]) for r in a["rows"] if r["exact"] is not None]
        best = max((e for e, _ in ex), default=None)
        return {"depth": int(a["depth"]), "value": int(a["value"]), "first": a["first"], "nodes": int(a["run_nodes"]), "best_exact": best, "best_names": [n for e, n in ex if e == best][:3] if best is not None else [], "exact_cut": bool(a["exact_cut"])}

    def read(self, moves: list, depth: "int | None" = None) -> dict:
        out = {}
        for arm in ARMS:
            r = self.search2(moves, False, {"root_full": True} if SWITCH[arm] else None)
            if r.get("first") and not r.get("error"):
                r["pick"] = self.pick(moves, r["first"], r["equals"])
            out[arm] = r
        for arm in ARMS:
            d = out[arm].get("depth")
            out[arm]["gap"] = self.gap(moves, d, SWITCH[arm]) if d else None
        if depth:
            out["fixed_off"] = self.gap(moves, depth, False)
            out["fixed_on"] = self.gap(moves, depth, True)
        return out


_BS = None


def _init_worker() -> None:
    global _BS
    _BS = Best()


def _job(args: tuple) -> tuple:
    idx, items, root = args
    f = Path(root) / "mend" / f"c{idx:05d}.json"
    if f.exists() and "\"error\"" not in f.read_text()[:200]:
        return idx, 0.0
    t0 = time.perf_counter()
    out = []
    for rec, moves in items:
        try:
            out.append(dict(_BS.read(moves, rec["depth"]), id=rec["id"]))
        except Exception as e:
            out.append({"id": rec["id"], "error": f"{type(e).__name__}: {str(e)[:150]}"})
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0


def load(root: Path) -> list:
    return [json.loads(x) for x in (root / "mend.jsonl").read_text().splitlines() if x.strip()]


def run(tag: str, workers: int, limit: int, say) -> None:
    root = out_root(tag)
    (root / "mend").mkdir(exist_ok=True)
    items = CC.items_of(LA.SRC, limit)
    say(f"positions {len(items)} (79.00's sample with a depth; the first {limit} in the firm-room-first order if a limit is given); each read at the fixed depth with the switch off and on (node budgets), and three times at {CLOCK} s, ledger share {TM.LEDGER_SHARE}, verify on (slot), the human-result record {RP.worth_of(SR.RECORD)}: "
        f"A (switch off), M (switch on), N (switch off again, the clock's noise), each audited at its own depth; the pick of 78.82 inside each set; {workers} workers of {os.cpu_count()} cores")
    load0, busy = TM.wait_quiet(lambda s: say(s.replace("OBS ", "", 1)))
    say(f"load average at the start {os.getloadavg()[0]:.2f}")
    jobs = [(a // CHUNK, items[a:a + CHUNK], str(root)) for a in range(0, len(items), CHUNK)]
    t0 = time.perf_counter()
    done = 0
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for _i, _dt in pool.imap_unordered(_job, jobs, chunksize=1):
            done += 1
            if done % 25 == 0 or done == len(jobs):
                say(f"progress: {done} of {len(jobs)} chunks; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    out = []
    for f in sorted((root / "mend").glob("c*.json")):
        out += json.loads(f.read_text())
    (root / "mend.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    say(f"read {len(out)} of {len(items)} positions in {time.perf_counter() - t0:.0f} s wall ({sum(1 for r in out if 'error' in r)} errors); load average at the end {os.getloadavg()[0]:.2f}")


# ------------------------------------------------------------------------------------------------------------------------------ the counts

def cell(vals: np.ndarray, game: np.ndarray, plus: bool = False) -> str:
    n = len(vals)
    if n < MIN_CELL:
        return f"{100.0 * float(np.mean(vals)):{'+' if plus else ''}.1f} (n {n}, too few for a band)" if n else "none (n 0)"
    return PMS.fmt(PMS.share(vals, game), plus=plus) + f" (n {n})"


def rows_of(tag: str) -> list:
    """One row per position read without error in all three arms: the labels, the reads, the classes of the move played and the pick, the fault of each arm."""
    root = out_root(tag)
    unit = BC.weakest_unit(RP.worth_of(SR.RECORD))
    recs = {r["id"]: r for r in LA.src_records()}
    out = []
    for r in load(root):
        if "error" in r or any((r[a].get("error") or not r[a].get("first") or r[a].get("depth") is None) for a in ARMS):
            continue
        rec = recs[r["id"]]
        lab = labels_of(rec)
        out.append({"id": r["id"], "game": rec["ord"], "band": rec["band"], "part": LF.part_of(rec["ord"]), "labels": lab, "reads": {a: r[a] for a in ARMS},
                    "first": {a: class_of(r[a]["first"], lab) for a in ARMS}, "pick": {a: class_of(r[a].get("pick"), lab) for a in ARMS},
                    "fault": {a: fault_of(r[a].get("gap"), unit) for a in ARMS},
                    "fixed": {k: r.get(k) for k in ("fixed_off", "fixed_on")}, "fixed_fault": {k: fault_of(r.get(k), unit) for k in ("fixed_off", "fixed_on")}})
    return out


def cmd_rule(tag: str, say) -> None:
    rule = {"pick_keep_gain_band_above_zero": True, "fault_share_change_band_below_zero": True, "depth_loss_max": DEPTH_LOSS_MAX, "time_within_clock_seconds": TM.SECONDS, "find_agrees_in_sign": True}
    (out_root(tag) / "rule.json").write_text(json.dumps(rule))
    say("THE RULE FOR PLAYING, written before CONFIRM is read: play the mend against the unmended player iff on CONFIRM (all positions) (1) the pick keeps the class, M less A over the positions where both picks have a label, has a 90 pct band (games resampled whole) wholly above zero; (2) the share of positions with the fault (a gap of a unit or more, or the game's end) "
        f"at the clock, M less A, has a band wholly below zero; (3) the mean depth reached under M is not more than {DEPTH_LOSS_MAX:g} ply under A's; (4) the mean seconds of a decision under M stay within the play clock {TM.SECONDS} s; and FIND agrees with (1) and (2) in sign.")


def block(say, name: str, rows: list) -> None:
    g = np.array([r["game"] for r in rows])
    n = len(rows)
    f = lambda fn: np.array([float(fn(r)) for r in rows])
    say(f"[{name}] positions {n}")
    for arm, lab in (("a", "A (switch off)"), ("m", "M (switch on)"), ("n", "N (switch off again)")):
        rd = [r["reads"][arm] for r in rows]
        say(f"[{name}] {lab}: depth {np.mean([x['depth'] for x in rd]):.3f}; seconds a decision {np.mean([x['seconds'] for x in rd]):.4f}; nodes a decision {np.mean([x.get('nodes') or 0 for x in rd]):.0f}; set size {np.mean([len(x['equals']) for x in rd]):.2f}; cut by the clock {100 * np.mean([float(x['cut']) for x in rd]):.1f} pct; "
            f"fault at the clock (gap of a unit or more or the game's end) {cell(f(lambda r: is_fault(r['fault'][arm])), g)} pct, any gap {cell(f(lambda r: r['fault'][arm] in ('less', 'unit', 'ends')), g)} pct; "
            f"the move played keeps the class {cell(f(lambda r: r['first'][arm] == 'keep'), g)}, loses it {cell(f(lambda r: r['first'][arm] == 'lose'), g)}, unlabelled {cell(f(lambda r: r['first'][arm] == 'unlabelled'), g)} pct; the pick keeps {cell(f(lambda r: r['pick'][arm] == 'keep'), g)}, loses {cell(f(lambda r: r['pick'][arm] == 'lose'), g)}, unlabelled {cell(f(lambda r: r['pick'][arm] == 'unlabelled'), g)} pct")
    for arm in ("m", "n"):
        lab = "M less A" if arm == "m" else "N (the noise) less A"
        d = f(lambda r: float(r["fault"][arm] in ("unit", "ends")) - float(r["fault"]["a"] in ("unit", "ends")))
        say(f"[{name}] fault share at the clock, {lab} (points): {cell(d, g, plus=True)}; depth {lab}: {np.mean([r['reads'][arm]['depth'] - r['reads']['a']['depth'] for r in rows]):+.3f}; seconds {np.mean([r['reads'][arm]['seconds'] - r['reads']['a']['seconds'] for r in rows]):+.4f}; the move played differs from A's in {100 * np.mean([float(r['reads'][arm]['first'] != r['reads']['a']['first']) for r in rows]):.1f} pct")
        for key in ("first", "pick"):
            sub = [r for r in rows if r[key]["a"] != "unlabelled" and r[key][arm] != "unlabelled"]
            if sub:
                gs = np.array([r["game"] for r in sub])
                dd = np.array([float(r[key][arm] == "keep") - float(r[key]["a"] == "keep") for r in sub])
                say(f"[{name}] the {'move played' if key == 'first' else 'pick'} keeps the class, {lab}, over positions where both have a label (points): {cell(dd, gs, plus=True)}; positions where either has no label {n - len(sub)} of {n}")
    # the positions where A has the fault at the clock
    sub = [r for r in rows if is_fault(r["fault"]["a"])]
    if sub:
        gs = np.array([r["game"] for r in sub])
        say(f"[{name}] positions where A has the fault at the clock (n {len(sub)}): the fault is gone under M in {100 * np.mean([float(not is_fault(r['fault']['m'])) for r in sub]):.1f} pct, under N in {100 * np.mean([float(not is_fault(r['fault']['n'])) for r in sub]):.1f} pct; the move played differs from A's under M in {100 * np.mean([float(r['reads']['m']['first'] != r['reads']['a']['first']) for r in sub]):.1f} pct, under N in {100 * np.mean([float(r['reads']['n']['first'] != r['reads']['a']['first']) for r in sub]):.1f} pct")
        for key in ("first", "pick"):
            s2 = [r for r in sub if r[key]["a"] != "unlabelled" and r[key]["m"] != "unlabelled"]
            if s2:
                gsub = np.array([r["game"] for r in s2])
                say(f"[{name}] in them the {'move played' if key == 'first' else 'pick'} keeps the class: A {cell(np.array([float(r[key]['a'] == 'keep') for r in s2]), gsub)}, M {cell(np.array([float(r[key]['m'] == 'keep') for r in s2]), gsub)}; M less A {cell(np.array([float(r[key]['m'] == 'keep') - float(r[key]['a'] == 'keep') for r in s2]), gsub, plus=True)}")
    # the fixed depth
    ok = [r for r in rows if r["fixed"]["fixed_off"] and r["fixed"]["fixed_on"]]
    if ok:
        go = np.array([r["game"] for r in ok])
        fo = lambda k: np.array([float(is_fault(r["fixed_fault"][k])) for r in ok])
        ao = lambda k: np.array([float(r["fixed_fault"][k] in ("less", "unit", "ends")) for r in ok])
        ratio = np.array([r["fixed"]["fixed_on"]["nodes"] / max(1, r["fixed"]["fixed_off"]["nodes"]) for r in ok])
        say(f"[{name}] AT THE FIXED DEPTH (79.00's, node budgets, n {len(ok)}): fault share off {cell(fo('fixed_off'), go)} pct, on {cell(fo('fixed_on'), go)} pct, on less off {cell(fo('fixed_on') - fo('fixed_off'), go, plus=True)} points; any gap off {cell(ao('fixed_off'), go)}, on {cell(ao('fixed_on'), go)} pct; "
            f"the deepening's nodes on over off: median {np.median(ratio):.2f}, mean {ratio.mean():.2f}, quartiles {np.percentile(ratio, 25):.2f}/{np.percentile(ratio, 75):.2f}; the move chosen differs in {100 * np.mean([float(r['fixed']['fixed_on']['first'] != r['fixed']['fixed_off']['first']) for r in ok]):.1f} pct")


def say_conditions(say, name: str, c: dict) -> None:
    say(f"[{name}] THE RULE'S CONDITIONS: (1) the pick keeps the class, M less A: {PMS.fmt(c['pick_gain'], plus=True)} points, band wholly above zero: {c['pick_pass']}; (2) the fault share at the clock, M less A: {PMS.fmt(c['fault_change'], plus=True)} points, band wholly below zero: {c['fault_pass']}; "
        f"(3) depth A less M {c['depth_loss']:+.3f} ply, not over {DEPTH_LOSS_MAX:g}: {c['depth_pass']}; (4) seconds a decision under M {c['seconds']:.4f} within the clock {CLOCK} s: {c['time_pass']}")


def cmd_count(tag: str, part: str, say) -> None:
    if part == "confirm" and not (out_root(tag) / "rule.json").exists():
        raise RuntimeError("rule.json is missing: the rule for playing is written (best_mend rule) before CONFIRM is read")
    rows = [r for r in rows_of(tag) if r["part"] == part]
    say(f"{part.upper()}: positions read in all three arms {len(rows)} of {len(load(out_root(tag)))}")
    for band in ("all",) + LF.BANDS:
        sub = rows if band == "all" else [r for r in rows if r["band"] == band]
        if sub:
            block(say, f"{part} {band}", sub)
            if band == "all":
                say_conditions(say, f"{part} all", conditions(sub))


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "rule", "count"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--part", default="find", choices=["find", "confirm"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    say = Say(out_root(a.tag), "mend " + a.cmd)
    if a.cmd == "run":
        run(a.tag, a.workers, a.limit, say)
    elif a.cmd == "rule":
        cmd_rule(a.tag, say)
    else:
        cmd_count(a.tag, a.part, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
