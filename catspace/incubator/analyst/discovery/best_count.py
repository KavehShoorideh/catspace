"""PLAN 79.03 PART ONE, THE COUNT OF THE SEARCH'S VALUE UNDER THE BEST MOVE IT COULD READ AT ITS OWN DEPTH (SYSTEM_SPEC 4 the tactical reasoner's deepening and its prunings, 12 the debt of plan 78.84; piece: tactical reasoning, a MEASURE). No engine change, no switch turned on, no oracle call (79.00's stored class labels are read), nothing
written to rules/learned or data/derived; outputs under artifacts/experiments/best_79_03/<tag>/.

WHAT THE CODE DOES AT THE ROOT WHEN A REDUCED OR NARROW-WINDOW ANSWER COMES BACK ABOVE THE WINDOW, AND WHAT THE NULL MOVE DOES AT AND JUST UNDER THE ROOT (read from rust/engine/tactics/src/ledger.rs `Search::value`, `run`, `probes_in`, before any number was looked at): see STATEMENT.

THE COUNT. The audit of 79.02 (artifacts/experiments/cut_79_02/full/audit.jsonl: every position of 79.00's sample that has a depth, 9669; node budgets, no clock; the stack's judge, the human-result record) holds for each position the value v the deepening reports at the depth D it finished and, for every root move, `exact`: the move's value
at D read with the FULL window on a fresh table (the prunings on below the root, the late-move reduction not applied at the root itself). THE GAP of a position is the best exact value minus v. It is read here on every position, not the room only:
  none      the gap is zero            reverse   the gap is negative (the deepening's value is ABOVE every exact value; the reverse fault, counted for the record)
  less      0 < gap < one unit of the weakest kind (the smallest positive worth of the record)         unit   the gap is one such unit or more
  ends      the best exact value or v is a value of the game's own end (WIN): reported apart, in the judge's integers a gap of that kind has no size
Each by band and by part (FIND to choose, CONFIRM to report: 79.01's split by game), 90 pct bands by resampling games whole. And from 79.00's stored labels (the class after each equal move; a move KEEPS the class when it is not lower than the class before it): in a position with a gap, does the move the search plays keep the class, and does the best exact move
keep it (the stored class exists only for the moves in 79.00's equal set; the best exact move is in it nearly always, because it holds at least v; where it is not, it is counted as unlabelled).

INTERFACE (written before the bodies).
  weakest_unit(worth) -> int                          the smallest positive worth (pure; tested).
  gap_kind(gap, unit, ends) -> str                    "none" | "reverse" | "less" | "unit" | "ends" (pure; tested).
  position_row(rec, aud, unit) -> dict | None         one row for a position: v, D, best exact, the gap, its kind, how many moves are above v, the played move's exact, whether the best exact move is in the set and keeps the class (pure; tested).
  gather(tag_cut, src) -> {"rows", "unit"}            rows of every audited position with the stored labels.
  cmd_count(tag, tag_cut, src, say)                   the counts, OBS lines.
  Called by: the command line, best_which (the positions to read), best_mend (the positions with a gap), ladder_print --best03.

  .venv/bin/python -m catspace.incubator.analyst.discovery.best_count count --tag full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import cut_count as CC
from catspace.incubator.analyst.discovery import leaf_ask as LA
from catspace.incubator.analyst.discovery import leaf_firm as LF
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import prior_measure as PMS
from catspace.incubator.analyst.discovery import record_play as RP
from catspace.incubator.analyst.discovery import stack_room as SR
from catspace.incubator.analyst.discovery.best_out import Say, out_root

WIN = 1 << 40                  # the game's own end in the judge's integers (ledger.rs WIN)
MIN_CELL = 30
KINDS = ("none", "reverse", "less", "unit", "ends")

STATEMENT = (
    "WHAT THE CODE DOES (ledger.rs Search::value / run / probes_in, read before any number). "
    "AT THE ROOT: the deepening reads every root move with the window (alpha, beta) at the depth D, the first move in the table's order (the previous best first, then the moves that gain, then killers and history). A root move that gains nothing (a quiet one) and comes fourth or later, other than the table's best, "
    "is first read TWO PLIES SHALLOWER than the rest (the late-move reduction, depth D-3 under the root instead of D-1). If that reduced answer is ABOVE alpha it is read again at the full depth with the same window, which at the root is open above (beta is infinite), so the answer is exact and a reduced answer above the window is always asked again unreduced: the repair the plan names as standard is already in the code. "
    "If the reduced answer is NOT above alpha it is kept as it is, the move is a fail-low and is never read at the full depth: a late quiet root move whose win lies deeper than the reduced reading is dropped for good, and the child's table entry carries the reduced depth, so a later full-depth reading of it is not served by that entry. "
    "A narrow window at the root exists only in the equal reading (probes_in, window (v-1, v)), not in the deepening, whose root window is open. "
    "THE NULL MOVE: it is never tried at the root (it needs ply above 0), and it is NOT tried one ply under the root either, in the deepening: it needs the node's beta to be finite (below WIN) and the root's beta is infinite, so the reply node has an open beta. It is first tried two plies under the root (the side that moved asks for a pass read three plies shallower; "
    "a pass that already holds the bound cuts the node), only when the mover keeps more than three units and the depth is 3 or more. In the equal reading's null window (v-1, v) both bounds are finite, so there the null move is already tried one ply under the root. "
    "SO, in the deepening: a move can come out under its full value at D by (a) the late-move reduction at the root (a fail-low at the reduced depth, never re-read) or (b) the late-move reduction or the null move below the root (a bound proved on a shallower or a passed reading, stored as if full). "
    "The `exact` value of the audit differs from the deepening in exactly this: the root's reduction is not applied and every root move gets the open window; the prunings below the root stay on. A GAP (exact best above v) is therefore, by the code, explained by the root's reduction or by a pruning below the root met in another order; part one's second half (best_which) says which.")


# ------------------------------------------------------------------------------------------------------------------------------ pure parts

def weakest_unit(worth: dict) -> int:
    pos = [int(w) for w in worth.values() if w and int(w) > 0]
    return min(pos)


def gap_kind(gap: "int | None", unit: int, ends: bool) -> str:
    if gap is None:
        return "none"
    if ends and gap != 0:
        return "ends"
    if gap == 0:
        return "none"
    if gap < 0:
        return "reverse"
    return "unit" if gap >= unit else "less"


def position_row(rec: dict, aud: dict, unit: int) -> "dict | None":
    """See the module doc. `rec` is 79.00's record (tied, cm, cls, first, ...), `aud` the audit of 79.02 ({"value", "finished", "rows": [{"name", "exact", ...}]})."""
    rows = aud["rows"]
    v = int(aud["value"])
    ex = [(int(r["exact"]), r["name"]) for r in rows if r.get("exact") is not None]
    if not ex:
        return None
    best_ex = max(e for e, _ in ex)
    best_names = [n for e, n in ex if e == best_ex]
    gap = best_ex - v
    ends = abs(best_ex) >= WIN - 1000 or abs(v) >= WIN - 1000
    cm = dict(zip(rec["tied"], rec["cm"]))
    keeps = lambda m: (PE.keeps(rec["cls"], cm[m]) if m in cm else None)
    played = rec["first"]
    played_ex = next((int(r["exact"]) for r in rows if r["name"] == played and r.get("exact") is not None), None)
    best_in = [n for n in best_names if n in cm]
    best_move = best_in[0] if best_in else best_names[0]
    return {"id": rec["id"], "game": rec["ord"], "band": rec["band"], "part": LF.part_of(rec["ord"]), "v": v, "depth": int(aud["finished"]), "best_exact": best_ex, "gap": gap, "kind": gap_kind(gap, unit, ends), "above": sum(1 for e, _ in ex if e > v),
            "played": played, "played_exact": played_ex, "played_under_best": (played_ex is not None and played_ex < best_ex), "played_keeps": keeps(played), "best_move": best_move, "best_in_set": bool(best_in), "best_keeps": keeps(best_move), "n_set": len(rec["tied"]),
            "cut": bool(rec.get("cut"))}


# ------------------------------------------------------------------------------------------------------------------------------ the rows

def gather(tag_cut: str = "full", src: str = LA.SRC) -> dict:
    from catspace.incubator.analyst.discovery.cut_out import out_root as cut_root
    recs = {r["id"]: r for r in LA.src_records(src)}
    aud = [a for a in CC.load(cut_root(tag_cut)) if "error" not in a and a.get("audit")]
    unit = weakest_unit(RP.worth_of(SR.RECORD))
    rows = []
    for a in aud:
        r = position_row(recs[a["id"]], a["audit"], unit)
        if r is not None:
            rows.append(r)
    return {"rows": rows, "unit": unit}


# ------------------------------------------------------------------------------------------------------------------------------ the counts

def cell(vals: np.ndarray, game: np.ndarray, sel: np.ndarray) -> str:
    n = int(sel.sum())
    if n < MIN_CELL:
        return f"{100.0 * float(vals[sel].mean()):.1f} (n {n}, too few for a band)" if n else "none (n 0)"
    return PMS.fmt(PMS.share(vals[sel], game[sel])) + f" (n {n})"


def quartiles(x: list) -> str:
    if not x:
        return "none"
    a = np.array(x, float)
    return "quartiles " + "/".join(f"{np.percentile(a, p):.2f}" for p in (25, 50, 75)) + f", 90th {np.percentile(a, 90):.2f}, mean {a.mean():.2f}"


def block(say, name: str, rows: list, unit: int) -> None:
    n = len(rows)
    g = np.array([r["game"] for r in rows])
    allp = np.ones(n, bool)
    f = lambda fn: np.array([float(fn(r)) for r in rows])
    say(f"[{name}] positions {n}; gap above zero (some root move's full-window value at D above v) {cell(f(lambda r: r['gap'] > 0), g, allp)} pct; a unit or more {cell(f(lambda r: r['kind'] == 'unit'), g, allp)} pct; less than a unit {cell(f(lambda r: r['kind'] == 'less'), g, allp)} pct; the game's own end in v or the best {cell(f(lambda r: r['kind'] == 'ends'), g, allp)} pct; "
        f"reverse (every exact value under v) {cell(f(lambda r: r['kind'] == 'reverse'), g, allp)} pct")
    pos = [r for r in rows if r["kind"] in ("less", "unit")]
    say(f"[{name}] size of the gap where there is one and the game does not end in it (n {len(pos)}), in units of the weakest kind: {quartiles([r['gap'] / unit for r in pos])}; moves above v a position: mean {float(np.mean([r['above'] for r in pos])) if pos else float('nan'):.2f}")
    if pos:
        u = np.array([r["gap"] / unit for r in pos])
        say(f"[{name}] the same gaps in buckets of the weakest unit: under one {int((u < 1).sum())}, exactly one {int((u == 1).sum())}, over one to three {int(((u > 1) & (u <= 3)).sum())}, over three to nine {int(((u > 3) & (u <= 9)).sum())}, over nine {int((u > 9).sum())}")
    isg = f(lambda r: r["kind"] in ("less", "unit", "ends"))
    cg = f(lambda r: r["kind"] == "unit")
    lost = lambda r: r["played_keeps"] is False
    labelled = lambda r: r["played_keeps"] is not None
    sel_all = np.array([labelled(r) for r in rows])
    for lab, sel in (("gap of any size (less, unit, ends)", isg > 0), ("gap of a unit or more", cg > 0), ("no gap (none)", f(lambda r: r["kind"] == "none") > 0)):
        s2 = sel & sel_all
        if s2.sum():
            say(f"[{name}] the move the search plays LOSES the oracle's class (stored label), positions with {lab}: {cell(f(lost), g, s2)} pct")
    # the best exact move against the played one, in the same positions
    for lab, kinds in (("gap of a unit or more", ("unit",)), ("gap of any size (less, unit)", ("less", "unit"))):
        sub = [r for r in rows if r["kind"] in kinds and r["played_keeps"] is not None and r["best_keeps"] is not None]
        if sub:
            gg = np.array([r["game"] for r in sub])
            al = np.ones(len(sub), bool)
            pk = np.array([float(r["played_keeps"]) for r in sub])
            bk = np.array([float(r["best_keeps"]) for r in sub])
            say(f"[{name}] positions with {lab} where both the played move and the best exact move have a stored label (n {len(sub)}): the played move keeps the class {cell(pk, gg, al)} pct, the best exact move keeps it {cell(bk, gg, al)} pct, best less played {cell(bk - pk, gg, al).replace('(n', '(n')} points-of-share")
        nolab = sum(1 for r in rows if r["kind"] in kinds and r["best_keeps"] is None)
        say(f"[{name}] positions with {lab} whose best exact move has no stored label (not in 79.00's equal set): {nolab} of {sum(1 for r in rows if r['kind'] in kinds)}")
    sub = [r for r in rows if r["kind"] in ("less", "unit")]
    if sub:
        gg = np.array([r["game"] for r in sub])
        al = np.ones(len(sub), bool)
        say(f"[{name}] gap positions (less, unit): the played move's own exact value is under the best exact {cell(np.array([float(r['played_under_best']) for r in sub]), gg, al)} pct; the best exact move is in 79.00's equal set {cell(np.array([float(r['best_in_set']) for r in sub]), gg, al)} pct; the position was read to the clock's cut {cell(np.array([float(r['cut']) for r in sub]), gg, al)} pct")


def cmd_count(tag: str, tag_cut: str, src: str, say) -> None:
    d = gather(tag_cut, src)
    rows, unit = d["rows"], d["unit"]
    say(f"the weakest unit of the human-result record {RP.worth_of(SR.RECORD)} is {unit} in the judge's integers; positions with an exact reading {len(rows)} (the audit of 79.02, all sampled positions with a depth)")
    for part in ("all", "find", "confirm"):
        for band in ("all",) + LF.BANDS:
            sel = [r for r in rows if (part == "all" or r["part"] == part) and (band == "all" or r["band"] == band)]
            if sel:
                block(say, f"{part} {band}", sel, unit)
    (out_root(tag) / "gap_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def load_rows(tag: str) -> list:
    return [json.loads(x) for x in (out_root(tag) / "gap_rows.jsonl").read_text().splitlines() if x.strip()]


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["count"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--tag-cut", default="full")
    ap.add_argument("--src", default=LA.SRC)
    a = ap.parse_args(argv)
    say = Say(out_root(a.tag), a.cmd)
    obs = out_root(a.tag) / "obs.txt"
    if not obs.exists() or not obs.read_text().strip():
        Say(out_root(a.tag), "statement")(STATEMENT)
    cmd_count(a.tag, a.tag_cut, a.src, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
