"""PLAN 79.02 PART THREE, THE MEASURE OF THE MEND (SYSTEM_SPEC 4 the equal reading, 12 the debt of plan 78.84; piece: tactical reasoning, a MEASURE of the engine's switch `reask`, OFF by default: rust/engine/tactics/src/ledger.rs `Search::ties_reasked`). Nothing is turned on in any default; the player of 79.00 (the human-result record, the
verified choice on, 78.82's pick) is the one that is read, with and without the switch passed as a parameter of the call. Nothing is written to rules/learned or data/derived; outputs under artifacts/experiments/cut_79_02/<tag>/. This module NEVER calls the oracle: the labels are 79.00's class after each equal move and 79.01's second asking (the firm label of
leaf_firm); a move that is in a set read here and was not in 79.00's set has no label and is counted as UNLABELLED (cut_ask.py asks the oracle for those, in sleep).

WHAT IS READ. For each position of the firm room (79.01: some equal move firmly keeps the class and some firmly loses it), the engine's search at the play clock (0.2 s, ledger share 0.4, one thread) three times, in this order: A (the switch off: the equal set as 79.00 read it), M (the switch on), N (the switch off again: the noise of the clock, the control). For each: the move the search plays, the equal set, the depth, whether the reading was cut,
the verified choice's state, the seconds, and the counts of the second asking. The pick of 78.82 (the reasons' pick over the set, the first move first) is taken inside each set.
  PUT OUT: a move of A's set (labelled) that is not in the other arm's set. M against A is the mend; N against A is what the clock does alone.
  PRICE: the same on A's firmly KEEPING moves.
  The first move is the move the search plays; it is the same in all arms unless the clock moved the deepening (counted).

THE RULE FOR PLAYING (written by `rule` BEFORE CONFIRM is read; `count --part confirm` refuses without it): the mend is played against the unmended player iff on CONFIRM (the firm room), (1) the pick's firm-keeps share under M less under A has a 90 pct band (games resampled whole) wholly above zero; (2) the firmly losing moves M puts out
less those N puts out (the noise) has a band wholly above zero; (3) the firmly keeping moves M puts out are under a tenth of A's firmly keeping moves; (4) the mean seconds of a decision under M stay within the play clock (the slot the equal reading already has is what is left of the clock; M's reading takes the part of it that A's leaves unused, so M's decisions are longer than A's by that part, which is PRINTED and is not a condition: the clock bounds both sides alike); and FIND agrees with (1) and (2) in sign. Written before any number of CONFIRM (condition (4) first read "no more than the clock's noise plus 0.01 s over A's": changed after the short run of 14 positions showed M spends the unused slot by design, before any FIND or CONFIRM number of the full run).

INTERFACE (written before the bodies).
  labels_of(rec, ask, margin) -> {move: "keep" | "lose" | None}     the firm label of every equal move of 79.00's set (pure; tested).
  put_out(base, other, labels) -> {"lose": (out, n), "keep": (out, n), "unlabelled": n}     of the labelled moves of `base`'s set, how many are not in `other`'s set (pure; tested).
  pick_state(pick, labels) -> "keep" | "lose" | "neither" | "unlabelled"     (pure; tested).
  Mend()                       per process: the stack's search and pick. `.read(moves) -> {"a", "m", "n"}` each a dict {first, equals, depth, cut, state, error, seconds, nodes, reask, pick}.
  run(tag, workers, limit, say) / load(root) / cmd_rule(tag, say) / cmd_count(tag, part, say)   the Pool over the cores (clocked: the machine is waited quiet as 79.01's clocked reads did), one file a chunk; mend.jsonl; the rule; the counts.
  Called by: the command line, ladder_print --cut02 (reads mend.jsonl for the judged positions), the match (cut_play.py) reads the rule.

  .venv/bin/python -m catspace.incubator.analyst.discovery.cut_mend run --tag short --limit 40 | run --tag full | rule --tag full | count --tag full --part find|confirm
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

from catspace.incubator.analyst.discovery import cut_count as CC
from catspace.incubator.analyst.discovery import leaf_ask as LA
from catspace.incubator.analyst.discovery import leaf_firm as LF
from catspace.incubator.analyst.discovery import leaf_line as LL
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import prior_measure as PMS
from catspace.incubator.analyst.discovery import record_play as RP
from catspace.incubator.analyst.discovery import stack_room as SR

CHUNK = 6
MIN_CELL = 30
CLOCK = float(os.environ.get("CUT_CLOCK", TM.SECONDS))     # the play clock; another clock only for a SUPPLEMENT (the environment variable CUT_CLOCK, read by the workers too), never for the plan's reading
PRICE_MAX = 0.10                                       # the rule's ceiling on the share of firmly keeping moves put out
ARMS = ("a", "m", "n")


# ------------------------------------------------------------------------------------------------------------------------------ pure parts

def labels_of(rec: dict, ask: dict, margin: float) -> dict:
    return dict(zip(rec["tied"], LF.firm_vec(rec, ask, margin)))


def put_out(base: list, other: list, labels: dict) -> dict:
    out = {"lose": [0, 0], "keep": [0, 0], "unlabelled": 0}
    inside = set(other)
    for m in base:
        lab = labels.get(m)
        if lab is None:
            out["unlabelled"] += 1
            continue
        out[lab][1] += 1
        out[lab][0] += int(m not in inside)
    return {"lose": tuple(out["lose"]), "keep": tuple(out["keep"]), "unlabelled": out["unlabelled"]}


def pick_state(pick: "str | None", labels: dict) -> str:
    if pick is None or pick not in labels:
        return "unlabelled"
    return labels[pick] or "neither"


# ------------------------------------------------------------------------------------------------------------------------------ the reads

class Mend(SR.Stack):
    """The stack's search with the switch as a parameter, and the stack's pick."""

    def search2(self, moves: list, reask: bool) -> dict:
        import cs_engine as CE
        from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
        t0 = time.perf_counter()
        kw = {"reask": True} if reask else {}
        try:
            raw = CE.solve_equals_batch([(OM.VARIANT, None, list(moves))], CLOCK, worth=dict(self.worth), ours_first=False, valuation="odds record", threads=1, ledger_share=TM.LEDGER_SHARE, verify_choice=True, choice_mode="slot", **kw)[0]
        except Exception as e:
            return {"first": None, "equals": [], "depth": None, "cut": False, "error": f"{type(e).__name__}: {str(e)[:120]}", "seconds": time.perf_counter() - t0}
        row = PM.equal_set(raw)
        return {"first": row["move"], "equals": row["equals"], "depth": row["depth"], "cut": bool(row["cut"]), "error": row["error"] or None, "state": int(raw[12] or 0), "seconds": time.perf_counter() - t0, "nodes": int(raw[2] or 0),
                "reask": list(raw[15]) if len(raw) > 15 else [], "ties_seconds": float(raw[9] or 0.0)}

    def read(self, moves: list) -> dict:
        out = {}
        for arm, reask in (("a", False), ("m", True), ("n", False)):
            r = self.search2(moves, reask)
            if r.get("first") and not r.get("error"):
                r["pick"] = self.pick(moves, r["first"], r["equals"])
            out[arm] = r
        return out


_MD = None


def _init_worker() -> None:
    global _MD
    _MD = Mend()


def _job(args: tuple) -> tuple:
    idx, items, root = args
    f = Path(root) / "mend" / f"c{idx:05d}.json"
    if f.exists() and "\"error\"" not in f.read_text()[:200]:
        return idx, 0.0
    t0 = time.perf_counter()
    out = []
    for rec, moves in items:
        try:
            out.append(dict(_MD.read(moves), id=rec["id"]))
        except Exception as e:
            out.append({"id": rec["id"], "error": f"{type(e).__name__}: {str(e)[:150]}"})
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0


def load(root: Path) -> list:
    return [json.loads(x) for x in (root / "mend.jsonl").read_text().splitlines() if x.strip()]


def run(tag: str, workers: int, limit: int, say) -> None:
    root = CC.out_root(tag)
    (root / "mend").mkdir(exist_ok=True)
    items = LL.items_of("full", LA.SRC)
    if limit:
        items = items[:limit]
    say(f"the firm room: {len(items)} positions (margin {LF.load_margin('full'):g}); each read three times at {CLOCK} s, ledger share {TM.LEDGER_SHARE}, verify on (slot), the judge the human-result record {RP.worth_of(SR.RECORD)}: A (switch off), M (switch on), N (switch off again, the clock's noise); the pick of 78.82 inside each set; {workers} workers of {os.cpu_count()} cores")
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
    for f in sorted((root / "mend").glob("c*.json")):
        out += json.loads(f.read_text())
    (root / "mend.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    say(f"read {len(out)} of {len(items)} positions in {time.perf_counter() - t0:.0f} s wall ({sum(1 for r in out if 'error' in r)} errors); load average at the end {os.getloadavg()[0]:.2f}")


# ------------------------------------------------------------------------------------------------------------------------------ the counts

def cell(vals: np.ndarray, game: np.ndarray, sel: np.ndarray, plus: bool = False) -> str:
    n = int(sel.sum())
    if n < MIN_CELL:
        return f"{100.0 * float(vals[sel].mean()):{'+' if plus else ''}.1f} (n {n}, too few for a band)" if n else "none (n 0)"
    return PMS.fmt(PMS.share(vals[sel], game[sel]), plus=plus) + f" (n {n})"


def rows_of(tag: str) -> list:
    """One row per firm-room position read without error in all three arms: the labels, the three reads, the picks' states."""
    root = CC.out_root(tag)
    margin = LF.load_margin("full")
    recs = {r["id"]: r for r in LA.src_records()}
    asks = {a["id"]: a for a in LA.load(LF.out_root("full"))}
    out = []
    for r in load(root):
        if "error" in r or any((r[a].get("error") or not r[a].get("first")) for a in ARMS):
            continue
        rec = recs[r["id"]]
        lab = labels_of(rec, asks[r["id"]], margin)
        out.append({"id": r["id"], "game": rec["ord"], "band": rec["band"], "part": LF.part_of(rec["ord"]), "labels": lab, "reads": {a: r[a] for a in ARMS},
                    "pick": {a: pick_state(r[a].get("pick"), lab) for a in ARMS}, "first": {a: pick_state(r[a]["first"], lab) for a in ARMS},
                    "sets": {a: sorted(set(r[a]["equals"]) | {r[a]["first"]}) for a in ARMS}})
    return out


def cmd_rule(tag: str, say) -> None:
    rule = {"pick_gain_band_above_zero": True, "put_out_over_noise_band_above_zero": True, "price_max": PRICE_MAX, "time_within_clock_seconds": TM.SECONDS, "find_agrees_in_sign": True}
    (CC.out_root(tag) / "rule.json").write_text(json.dumps(rule))
    say("THE RULE FOR PLAYING, written before CONFIRM is read: play the mend against the unmended player iff on CONFIRM (the firm room) (1) the pick's firm-keeps share under M less under A has a 90 pct band (games resampled whole) wholly above zero; (2) the firmly losing moves M puts out, less those N puts out (the clock's noise), has a band wholly above zero; "
        f"(3) the firmly keeping moves M puts out are under {PRICE_MAX:.2f} of A's firmly keeping moves; (4) the mean seconds of a decision under M stay within the play clock {TM.SECONDS} s (M's reading takes the part of the slot that A's leaves unused: the excess over A is printed, not a condition); and FIND agrees with (1) and (2) in sign. "
        "(Condition (4) first read: no more than the clock's noise plus 0.01 s over A's; changed after the short run of 14 positions showed M spends the unused slot by design, before any FIND or CONFIRM number of the full run.)")


def block(say, name: str, rows: list) -> dict:
    g = np.array([r["game"] for r in rows])
    n = len(rows)
    allp = np.ones(n, bool)
    f = lambda fn: np.array([fn(r) for r in rows], float)
    res = {}
    say(f"[{name}] firm-room positions {n}")
    for arm, lab in (("a", "A (switch off)"), ("m", "M (switch on)"), ("n", "N (switch off again)")):
        rd = [r["reads"][arm] for r in rows]
        say(f"[{name}] {lab}: set size {np.mean([len(r['sets'][arm]) for r in rows]):.2f}; depth {np.mean([x['depth'] for x in rd if x['depth'] is not None] or [float('nan')]):.2f} ({sum(1 for x in rd if x['depth'] is None)} with none); cut by the clock {cell(f(lambda r: float(r['reads'][arm]['cut'])), g, allp)} pct; seconds a decision {np.mean([x['seconds'] for x in rd]):.4f}; ties seconds {np.mean([x.get('ties_seconds', 0.0) for x in rd]):.4f}; "
            f"the move played firmly keeps {cell(f(lambda r: float(r['first'][arm] == 'keep')), g, allp)}, firmly loses {cell(f(lambda r: float(r['first'][arm] == 'lose')), g, allp)} pct; the pick firmly keeps {cell(f(lambda r: float(r['pick'][arm] == 'keep')), g, allp)}, firmly loses {cell(f(lambda r: float(r['pick'][arm] == 'lose')), g, allp)}, unlabelled {cell(f(lambda r: float(r['pick'][arm] == 'unlabelled')), g, allp)} pct")
    changed = f(lambda r: float(r["reads"]["m"]["first"] != r["reads"]["a"]["first"]))
    changed2 = f(lambda r: float(r["reads"]["n"]["first"] != r["reads"]["a"]["first"]))
    say(f"[{name}] the move played differs from A's: M {cell(changed, g, allp)} pct, N (noise) {cell(changed2, g, allp)} pct")
    rk = [r["reads"]["m"]["reask"] for r in rows if r["reads"]["m"].get("reask")]
    if rk:
        k = np.array(rk, float)
        say(f"[{name}] M's second asking a position (mean): moves asked in the first reading {k[:, 0].mean():.2f}, held {k[:, 1].mean():.2f}, asked again {k[:, 2].mean():.2f}, put out {k[:, 3].mean():.2f}, held and not reached {k[:, 4].mean():.2f}; first reading cut in {100 * k[:, 5].mean():.1f} pct of positions; positions with a move not reached {100 * float((k[:, 4] > 0).mean()):.1f} pct")
    for arm, lab in (("m", "M (the mend)"), ("n", "N (noise)")):
        po = [put_out(r["sets"]["a"], r["sets"][arm], r["labels"]) for r in rows]
        for who in ("lose", "keep"):
            out = np.array([p[who][0] for p in po], float)
            tot = np.array([p[who][1] for p in po], float)
            pg = np.repeat(g, tot.astype(int))
            vals = np.concatenate([[1.0] * int(o) + [0.0] * int(t - o) for o, t in zip(out, tot)]) if tot.sum() else np.array([])
            say(f"[{name}] {lab}: of A's firmly {'LOSING' if who == 'lose' else 'KEEPING'} equal moves ({int(tot.sum())}) put out {PMS.fmt(PMS.share(vals, pg)) if tot.sum() >= MIN_CELL else (f'{100 * vals.mean():.1f} (n {int(tot.sum())}, too few for a band)' if tot.sum() else 'none')} pct")
        say(f"[{name}] {lab}: moves of A's set with no label (not in 79.00's set) {sum(p['unlabelled'] for p in po)}; moves of the other set not in A's set {sum(len(set(r['sets'][arm]) - set(r['sets']['a'])) for r in rows)}")
    # the difference in the pick's keep share, and the put-out rates over the noise, paired by positions
    for arm in ("m", "n"):
        d = f(lambda r: float(r["pick"][arm] == "keep") - float(r["pick"]["a"] == "keep"))
        say(f"[{name}] the pick firmly keeps, {arm.upper()} less A (points): {PMS.fmt(PMS.share(d, g), plus=True) if n >= MIN_CELL else f'{100 * d.mean():+.1f} (n {n}, too few for a band)'}")
    dl = f(lambda r: float(r["pick"]["m"] == "lose") - float(r["pick"]["a"] == "lose"))
    say(f"[{name}] the pick firmly loses, M less A (points): {PMS.fmt(PMS.share(dl, g), plus=True) if n >= MIN_CELL else f'{100 * dl.mean():+.1f} (n {n}, too few for a band)'}")
    sec = {a: np.mean([r["reads"][a]["seconds"] for r in rows]) for a in ARMS}
    say(f"[{name}] seconds a decision: A {sec['a']:.4f}, M {sec['m']:.4f}, N {sec['n']:.4f}; M less A {sec['m'] - sec['a']:+.4f}, N less A {sec['n'] - sec['a']:+.4f}")
    return res


def conditions(rows: list) -> dict:
    """The four conditions of the rule on a set of firm-room positions: each a number and whether it passes (a band is the 90 pct band by games resampled whole)."""
    g = np.array([r["game"] for r in rows])
    d = np.array([float(r["pick"]["m"] == "keep") - float(r["pick"]["a"] == "keep") for r in rows])
    pick = PMS.share(d, g)
    mv, nv, gv = [], [], []
    keep_out = keep_n = 0
    for r in rows:
        for m in r["sets"]["a"]:
            lab = r["labels"].get(m)
            if lab == "lose":
                mv.append(float(m not in r["sets"]["m"]))
                nv.append(float(m not in r["sets"]["n"]))
                gv.append(r["game"])
            elif lab == "keep":
                keep_n += 1
                keep_out += int(m not in r["sets"]["m"])
    over = PMS.share(np.array(mv) - np.array(nv), np.array(gv)) if mv else (float("nan"),) * 3
    sec = {a: float(np.mean([r["reads"][a]["seconds"] for r in rows])) for a in ARMS}
    price = keep_out / max(1, keep_n)
    return {"pick_gain": pick, "pick_pass": bool(pick[1] > 0), "over_noise": over, "over_pass": bool(over[1] > 0), "price": price, "price_pass": bool(price < PRICE_MAX),
            "dt": sec["m"] - sec["a"], "noise_dt": sec["n"] - sec["a"], "time_pass": bool(sec["m"] <= CLOCK)}


def say_conditions(say, name: str, c: dict) -> None:
    say(f"[{name}] THE RULE'S CONDITIONS: (1) the pick firmly keeps, M less A: {PMS.fmt(c['pick_gain'], plus=True)} points, band wholly above zero: {c['pick_pass']}; (2) firmly losing moves put out, M less the noise: {PMS.fmt(c['over_noise'], plus=True)} points, band wholly above zero: {c['over_pass']}; "
        f"(3) firmly keeping moves put out by M {100 * c['price']:.1f} pct, under {100 * PRICE_MAX:.0f}: {c['price_pass']}; (4) seconds a decision under M within the clock {CLOCK} s: {c['time_pass']} (M less A {c['dt']:+.4f} s, N less A {c['noise_dt']:+.4f} s: printed)")


def put_out_reasons(rows: list, aud: dict, arm: str = "m") -> dict:
    """Of the firmly LOSING moves of A's set that M puts out: by the audit of the position, the move's unpruned answer is under v (REASKED: the reason the mend names), or it is held by the plain answer too (DROPPED: the set M read lacks it because its first reading was cut or read in another order, which is the clock and the order, not the unpruned answer). A position with no audit is counted apart."""
    out = {"reasked": 0, "dropped": 0, "no_audit": 0}
    for r in rows:
        a = aud.get(r["id"])
        for m in r["sets"]["a"]:
            if r["labels"].get(m) != "lose" or m in r["sets"][arm]:
                continue
            if not a:
                out["no_audit"] += 1
                continue
            row = {x["name"]: x for x in a["audit"]["rows"]}.get(m)
            out["reasked" if CC.hold_kind(row, a["audit"]["value"]) in CC.REDUCED else "dropped"] += 1
    return out


def cmd_count(tag: str, part: str, say) -> None:
    if part == "confirm" and not (CC.out_root(tag) / "rule.json").exists():
        raise RuntimeError("rule.json is missing: the rule for playing is written (cut_mend rule) before CONFIRM is read")
    rows = rows_of(tag)
    sel = [r for r in rows if r["part"] == part]
    say(f"{part.upper()}: firm-room positions read in all three arms {len(sel)} of {len(load(CC.out_root(tag)))}")
    for band in ("all",) + LF.BANDS:
        sub = sel if band == "all" else [r for r in sel if r["band"] == band]
        if sub:
            block(say, f"{part} {band}", sub)
            if band == "all":
                say_conditions(say, f"{part} all", conditions(sub))
                aud = {x["id"]: x for x in CC.load(CC.out_root("full")) if x.get("audit")} if (CC.out_root("full") / "audit.jsonl").exists() else {}
                for arm, lab in (("m", "M"), ("n", "N (the noise)")):
                    pr = put_out_reasons(sub, aud, arm)
                    say(f"[{part} all] the firmly LOSING moves {lab} puts out ({sum(pr.values())}): by the unpruned answer (under v by the audit) {pr['reasked']}; held by the plain answer too, so dropped by the cut or the order of the reading and not by the mend's question {pr['dropped']}; no audit {pr['no_audit']}")


JUDGED_NEW = 12
FROM_79_01 = (2, 3, 5)                                 # the positions of 79.01's fifteen (numbered from 1) that the plan names


def cmd_cases(tag: str, say) -> None:
    """The judged positions as data (cases_cut.json): positions 2, 3 and 5 of 79.01's fifteen, and the first JUDGED_NEW positions (in the sample's hash order, firm room, both parts, not among 79.01's three) where the mend puts out a firmly losing move of A's set (the rule, fixed here before any was read)."""
    from catspace.incubator.analyst.discovery import future_labels as FL
    margin = LF.load_margin("full")
    recs = {r["id"]: r for r in LA.src_records()}
    asks = {a["id"]: a for a in LA.load(LF.out_root("full"))}
    lines = {r["id"]: r for r in LL.load(LF.out_root("full"))}
    games = {g["ord"]: g for g in FL.universe()}
    aud = {r["id"]: r for r in CC.load(CC.out_root(tag)) if r.get("audit")} if (CC.out_root(tag) / "audit.jsonl").exists() else {}
    old = json.loads((Path(LF.out_root("full")) / "cases_leaf.json").read_text())
    named = [old[n - 1]["id"] for n in FROM_79_01]
    read = {r["id"]: r for r in load(CC.out_root(tag)) if "error" not in r and not any(r[a].get("error") or not r[a].get("first") for a in ARMS)}
    chosen = list(named)
    for r in rows_of(tag):
        if len(chosen) >= len(named) + JUDGED_NEW:
            break
        po = put_out(r["sets"]["a"], r["sets"]["m"], r["labels"])
        if r["id"] not in named and po["lose"][0] > 0:
            chosen.append(r["id"])
    out = []
    for i in chosen:
        rec, ask = recs[i], asks[i]
        rd = read.get(i)
        a = aud.get(i)
        ln = lines.get(i)
        tied = list(rec["tied"])
        rows = {x["name"]: x for x in a["audit"]["rows"]} if a else {}
        vec = LF.firm_vec(rec, ask, margin)
        out.append({"id": i, "named": i in named, "part": LF.part_of(rec["ord"]), "band": rec["band"], "units": rec["units"], "cls": rec["cls"], "moves": games[rec["ord"]]["moves"][:rec["ply"]], "depth": rec["depth"], "first": rec["first"], "pick": rec["pick"], "tied": tied, "cm": rec["cm"], "cp": rec["cp"], "cls2": ask["cls2"], "cp2": ask["cp2"], "vec": vec,
                    "value": (a["audit"]["value"] if a else None), "rows": [rows.get(m) for m in tied], "hold": [CC.hold_kind(rows.get(m), a["audit"]["value"]) if a else None for m in tied],
                    "reads": ({x: {k: rd[x].get(k) for k in ("first", "equals", "depth", "cut", "state", "reask", "pick", "seconds")} for x in ARMS} if rd else None),
                    "lines": ([{k: x[k] for k in ("names", "settled", "value", "cut")} if x.get("ok") else None for x in ln["rows"]] if ln and "rows" in ln else None)})
    (CC.out_root(tag) / "cases_cut.json").write_text(json.dumps(out))
    made = sum(1 for c in out if c["reads"] and not c["named"])
    say(f"{len(out)} judged positions written to {CC.out_root(tag) / 'cases_cut.json'}: 79.01's positions {list(FROM_79_01)} and {made} positions where the mend (M) puts out a firmly losing move of A's set (the first {JUDGED_NEW} of {sum(1 for r in rows_of(tag) if put_out(r['sets']['a'], r['sets']['m'], r['labels'])['lose'][0] > 0)} such positions in the sample's hash order)")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "rule", "count", "cases"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--part", default="find", choices=["find", "confirm"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    say = CC.Say(CC.out_root(a.tag), "mend " + a.cmd)
    if a.cmd == "run":
        run(a.tag, a.workers, a.limit, say)
    elif a.cmd == "rule":
        cmd_rule(a.tag, say)
    elif a.cmd == "cases":
        cmd_cases(a.tag, say)
    else:
        cmd_count(a.tag, a.part, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
