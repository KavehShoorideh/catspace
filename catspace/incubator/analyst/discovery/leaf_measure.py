"""PLAN 79.01 PARTS ONE AND TWO, THE MARGIN, THE FIRM ROOM AND THE TABLES (SYSTEM_SPEC 4 the equal reading, 5 and 6 the value fitted on results read where the reasoning ends, 9 the oracle as labeller in sleep; piece: the estimator at the leaf, a MEASURE). Reads the files of leaf_ask.py (the second asking), leaf_line.py (the line and the inputs of
every equal move of a firm-room position) and 79.00's labels; makes no engine call and no oracle call; prints OBS lines (obs.txt beside the files) and writes the judged positions (cases_leaf.json).

STAGES, in this order (the order is enforced: `table` and `firm` refuse to run without the margin of `margin`).
  margin   FIND only: the margin of the firm label by the rule of leaf_firm.choose_margin; written to margin.json, printed; no CONFIRM row is read.
  firm     PART ONE: the room's positions (79.00's, by the first asking), the firm room (leaf_firm), its share of the room by band, and in the firm room the first move and the pick (firmly keeps, firmly loses, neither); FIND and CONFIRM apart, 90 pct bands by games resampled whole.
  table    PART TWO: in the firm room, the share of positions where the move each rule takes FIRMLY KEEPS the class: the search's first move, 78.82's pick, a random equal move, the equal move with the highest estimator chance at the END OF THE LINE, and the highest chance ONE MOVE AFTER THE ROOT (the same
           estimator). A tie for the highest chance gives each tied move its share (the expectation over a random tie-break; the share of positions with a unique highest and of positions with no difference at all are printed). The same positions for every rule (positions with a line not read are left out, counted).
           Paired differences of the end-of-line rule against each other rule.
  cases    the judged positions (rule fixed here, before any was read): firm-room positions with a unique highest end-of-line chance on a move that is not the first move, the first JUDGED of them in the order of the sample's hash (both parts), written to cases_leaf.json for ladder_print --leaf01.

INTERFACE (written before the bodies).
  firm_table(recs, asks, margin) -> dict of arrays            per room position: game, band, part, firm room, the first move's and the pick's firm label (pure; tested).
  rule_table(recs, asks, lines, chance, margin) -> dict       per firm-room position with every line read: the keep and lose weights of each rule, the ties, the end-of-line and one-ply chances of each move (pure given the files and the chance function; tested).
  pick_cases(table, n) -> [index]                              the rule of the judged positions (pure; tested).
  Called by: the command line, ladder_print --leaf01 (reads cases_leaf.json).

  .venv/bin/python -m catspace.incubator.analyst.discovery.leaf_measure margin|firm|table|cases --tag short|full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import leaf_ask as LA
from catspace.incubator.analyst.discovery import leaf_firm as LF
from catspace.incubator.analyst.discovery import leaf_line as LL
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import prior_measure as PMS

JUDGED = 15
MIN_CELL = 30


def firm_table(recs: dict, asks: list, margin: float) -> dict:
    cols = {k: [] for k in ("id", "game", "band", "part", "firm", "first", "pick", "nmoves")}
    for a in asks:
        r = recs[a["id"]]
        v = LF.firm_vec(r, a, margin)
        pi = r["tied"].index(r["pick"]) if r.get("pick") in r["tied"] else 0
        for k, x in (("id", r["id"]), ("game", r["ord"]), ("band", r["band"]), ("part", LF.part_of(r["ord"])), ("firm", LF.is_firm(v)), ("first", v[0] or "neither"), ("pick", v[pi] or "neither"), ("nmoves", len(v))):
            cols[k].append(x)
    return {k: (np.array(v) if k not in ("firm",) else np.array(v, bool)) for k, v in cols.items()}


def rule_table(recs: dict, asks: dict, lines: dict, chance: "LL.Chance", margin: float) -> dict:
    rows, skipped = [], 0
    for i, ln in lines.items():
        r, a = recs[i], asks[i]
        if "rows" not in ln or not all(x["ok"] for x in ln["rows"]):
            skipped += 1
            continue
        vec = LF.firm_vec(r, a, margin)
        pi = r["tied"].index(r["pick"]) if r.get("pick") in r["tied"] else 0
        v_end = [chance.of(x["end"], x["end_is_root"]) for x in ln["rows"]]
        v_one = [chance.of(x["one"], x["one_is_root"]) for x in ln["rows"]]
        v_sym = [(chance.sym(x["end"]) if x["end_is_root"] else 1.0 - chance.sym(x["end"])) for x in ln["rows"]]
        sc = LF.rule_scores(vec, 0, pi, v_end, v_one, v_sym)
        we, wo = LF.weights(v_end), LF.weights(v_one)
        rows.append({"id": i, "game": r["ord"], "band": r["band"], "part": LF.part_of(r["ord"]), "vec": vec, "scores": sc, "v_end": v_end, "v_one": v_one, "v_sym": v_sym, "pick_i": pi,
                     "end_unique": sum(1 for w in we if w > 0) == 1, "one_unique": sum(1 for w in wo if w > 0) == 1, "end_flat": max(v_end) - min(v_end) <= LF.EPS, "one_flat": max(v_one) - min(v_one) <= LF.EPS,
                     "end_argmax": [j for j, w in enumerate(we) if w > 0], "one_argmax": [j for j, w in enumerate(wo) if w > 0]})
    return {"rows": rows, "skipped": skipped}


def pick_cases(rows: list, n: int = JUDGED) -> list:
    """The rule of the judged positions: a unique highest end-of-line chance on a move that is not the first move; the first n in the order of the sample's hash (rows are already in that order when `rows` comes from the lines file)."""
    hit = [k for k, r in enumerate(rows) if r["end_unique"] and r["end_argmax"][0] != 0]
    return hit[:n]


def cell(vals: np.ndarray, game: np.ndarray, sel: np.ndarray) -> str:
    if int(sel.sum()) < MIN_CELL:
        return (f"{100.0 * float(vals[sel].mean()):.1f} (n {int(sel.sum())}, too few for a band)" if int(sel.sum()) else "none")
    return PMS.fmt(PMS.share(vals[sel], game[sel]))


def cmd_margin(tag: str, say) -> None:
    root = LF.out_root(tag)
    recs = {r["id"]: r for r in LA.src_records()}
    pairs, rows = [], 0
    for a in LA.load(root):
        r = recs[a["id"]]
        if LF.part_of(r["ord"]) != "find":
            continue
        rows += 1
        for i in range(len(r["tied"])):
            pairs.append((LF.depth_inside(r["cm"][i], r["cp"][i]), r["cm"][i], a["cls2"][i]))
    m, table = LF.choose_margin(pairs)
    LF.save_margin(tag, m, table, rows)
    say(f"MARGIN chosen on FIND only ({rows} room positions, {len(pairs)} equal moves; the rule: the smallest of {list(LF.GRID)} at which at least {LF.AGREE:g} of the moves whose first number is that deep inside their class keep the class in the second asking, and at least {LF.MIN_MOVES} such moves exist): {m:g}")
    say("margin table (margin: moves at least that deep inside their class, of them the second asking agrees): " + "; ".join(f"{g}: {n}, {k}" + (f" ({100.0 * k / n:.1f} pct)" if n else "") for g, n, k in table))


def data(tag: str) -> tuple:
    root = LF.out_root(tag)
    margin = LF.load_margin(tag)
    recs = {r["id"]: r for r in LA.src_records()}
    asks = LA.load(root)
    return margin, recs, asks


def cmd_firm(tag: str, say) -> None:
    margin, recs, asks = data(tag)
    t = firm_table(recs, asks, margin)
    say(f"MARGIN {margin:g} (margin.json, chosen on FIND); the room of 79.00 asked again: {len(asks)} positions; the oracle's second asking agrees with the first on the class of every equal move in {sum(1 for a in asks for i in range(len(a['tied'])) if a['cls2'][i] == recs[a['id']]['cm'][i])} of {sum(len(a['tied']) for a in asks)} moves")
    every = list(recs.values())
    firm_ids = {i for i, f in zip(t["id"], t["firm"]) if f}
    for part in ("find", "confirm"):
        for band in ("all",) + LF.BANDS:
            sel = [r for r in every if LF.part_of(r["ord"]) == part and (band == "all" or r["band"] == band)]
            vals = np.array([float(r["id"] in firm_ids) for r in sel])
            rm = np.array([float(LF.in_room(r)) for r in sel])
            g = np.array([r["ord"] for r in sel])
            say(f"[{part} {band}] of all {len(sel)} positions of 79.00's sample: the room {cell(rm, g, np.ones(len(sel), bool))} pct; the firm room {cell(vals, g, np.ones(len(sel), bool))} pct")
    for part in ("find", "confirm"):
        sp = t["part"] == part
        for band in ("all",) + LF.BANDS:
            sel = sp if band == "all" else sp & (t["band"] == band)
            n, f = int(sel.sum()), int((sel & t["firm"]).sum())
            say(f"[{part} {band}] room positions {n}; the firm room {f}; share of the room that is firm {cell(t['firm'].astype(float), t['game'], sel)} pct")
            fr = sel & t["firm"]
            if int(fr.sum()) >= 1:
                for who in ("first", "pick"):
                    say(f"[{part} {band}] firm room ({int(fr.sum())}): the {who} move firmly keeps {cell((t[who] == 'keep').astype(float), t['game'], fr)}; firmly loses {cell((t[who] == 'lose').astype(float), t['game'], fr)}; neither {cell((t[who] == 'neither').astype(float), t['game'], fr)} pct")


def cmd_table(tag: str, say) -> None:
    margin, recs, asks = data(tag)
    lines = {r["id"]: r for r in LL.load(LF.out_root(tag))}
    ch = LL.Chance()
    asks_by = {a["id"]: a for a in asks}
    tab = rule_table(recs, asks_by, lines, ch, margin)
    rows = tab["rows"]
    say(f"the estimator: {ch.source}")
    say(f"MARGIN {margin:g}; firm-room positions with a line file {len(lines)}; every line read: {len(rows)}; left out (a line not read or an error): {tab['skipped']}")
    game = np.array([r["game"] for r in rows])
    band = np.array([r["band"] for r in rows])
    part = np.array([r["part"] for r in rows])
    for p in ("find", "confirm"):
        for b in ("all",) + LF.BANDS:
            sel = (part == p) if b == "all" else (part == p) & (band == b)
            n = int(sel.sum())
            if n == 0:
                continue
            head = f"[{p} {b}] firm room {n} positions"
            for rule in LF.RULES:
                kv = np.array([r["scores"][rule][0] for r in rows])
                lv = np.array([r["scores"][rule][1] for r in rows])
                say(f"{head}: {rule}: firmly keeps {cell(kv, game, sel)}; firmly loses {cell(lv, game, sel)} pct")
            e = np.array([r["scores"]["end"][0] for r in rows])
            for other in ("first", "pick", "random", "one", "end_sym"):
                o = np.array([r["scores"][other][0] for r in rows])
                say(f"{head}: end-of-line less {other}, firmly keeps (points): " + (PMS.fmt(PMS.share((e - o)[sel], game[sel]), plus=True) if n >= MIN_CELL else f"{100.0 * float((e - o)[sel].mean()):+.1f} (n {n}, too few for a band)"))
            for k, nm in (("end_unique", "end-of-line"), ("one_unique", "one-ply")):
                say(f"{head}: {nm} chance: a unique highest move in {cell(np.array([float(r[k]) for r in rows]), game, sel)}, no difference between the equal moves in " + cell(np.array([float(r[k.replace('unique', 'flat')]) for r in rows]), game, sel) + " pct")
            for k, nm in (("v_end", "end-of-line"), ("v_sym", "end-of-line with the side to move's bonus out")):
                flat = np.array([float(max(r[k]) - min(r[k]) <= LF.EPS) for r in rows])
                uni = np.array([float(sum(1 for w in LF.weights(r[k]) if w > 0) == 1) for r in rows])
                say(f"{head}: {nm} chance: the equal moves all alike in {cell(flat, game, sel)}, a unique highest in {cell(uni, game, sel)} pct")
            fm = np.array([float(r["end_argmax"][0] == 0 and r["end_unique"]) for r in rows])
            say(f"{head}: the end-of-line highest is the first move alone in {cell(fm, game, sel)} pct")


def cmd_cases(tag: str, say) -> None:
    margin, recs, asks = data(tag)
    lines = {r["id"]: r for r in LL.load(LF.out_root(tag))}
    asks_by = {a["id"]: a for a in asks}
    ch = LL.Chance()
    tab = rule_table(recs, asks_by, lines, ch, margin)
    rows = tab["rows"]
    idx = pick_cases(rows)
    out = []
    from catspace.incubator.analyst.discovery import future_labels as FL
    games = {g["ord"]: g for g in FL.universe()}
    for k in idx:
        r, row = recs[rows[k]["id"]], rows[k]
        a, ln = asks_by[r["id"]], lines[r["id"]]["rows"]
        out.append({"id": r["id"], "part": row["part"], "band": r["band"], "units": r["units"], "cls": r["cls"], "moves": games[r["ord"]]["moves"][:r["ply"]], "depth": r["depth"], "first": r["first"], "pick": r["pick"], "tied": r["tied"], "cm": r["cm"], "cp": r["cp"], "cls2": a["cls2"], "cp2": a["cp2"],
                    "vec": row["vec"], "v_end": row["v_end"], "v_one": row["v_one"], "v_sym": row["v_sym"], "end_argmax": row["end_argmax"], "lines": [{kk: x[kk] for kk in ("names", "settled", "value", "cut", "end", "end_is_root", "one")} for x in ln]})
    (LF.out_root(tag) / "cases_leaf.json").write_text(json.dumps(out))
    alike = sum(1 for c in out if max(c["v_sym"]) - min(c["v_sym"]) <= LF.EPS)
    say(f"of the {len(out)} judged positions the equal moves are all alike once the side to move's bonus is taken out of the chance in {alike}; the highest end-of-line chance firmly keeps the class in {sum(1 for c in out if all(c['vec'][j] == 'keep' for j in c['end_argmax']))}, firmly loses it in {sum(1 for c in out if all(c['vec'][j] == 'lose' for j in c['end_argmax']))}, neither in the rest; the first move firmly keeps it in {sum(1 for c in out if c['vec'][0] == 'keep')}")
    say(f"{len(out)} judged positions (firm room, a unique highest end-of-line chance on a move that is not the first move; {sum(1 for r in rows if r['end_unique'] and r['end_argmax'][0] != 0)} such positions in all, of {len(rows)}; the first {JUDGED} in the sample's hash order) written to {LF.out_root(tag) / 'cases_leaf.json'}")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["margin", "firm", "table", "cases"])
    ap.add_argument("--tag", default="short")
    a = ap.parse_args(argv)
    say = LA.Say(LF.out_root(a.tag), a.cmd)
    {"margin": cmd_margin, "firm": cmd_firm, "table": cmd_table, "cases": cmd_cases}[a.cmd](a.tag, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
