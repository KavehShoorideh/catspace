"""PLAN 79.04 PARTS ONE AND TWO, THE TABLES, THE COUNT AND THE JUDGED POSITIONS (SYSTEM_SPEC 5 and 6 the value fitted on results and its heads, 4 the leaf, 9 the oracle as labeller in sleep, 12 the debt of plan 78.84; piece: the estimator's heads, a MEASURE). Reads heads_read's blocks, 79.01's firm labels and 79.00's labels; no engine call, no oracle call, no fit, no new head; OBS lines to obs.txt beside the files.
The firm room, the firm label, the split into FIND and CONFIRM, the first move, the pick and the tie rule are 79.01's (leaf_firm): a move FIRMLY KEEPS or FIRMLY LOSES the oracle's class; a tie for the highest value gives each tied move its share (the expectation over a random tie-break). FIND is to choose, CONFIRM is to report; 90 pct bands resample games whole.

STAGES, in this order (`table`, `count` and `cases` refuse to run before `choose` has written best.json, which reads FIND only).
  choose   FIND only. Every head in two readings (one move after the root; the end of the search's line), each from both sides' view (heads_read), is scored by the share of FIND firm-room positions where the move it takes firmly keeps the class. THE BEST HEAD IS THE ONE WITH THE HIGHEST SHARE OVER BOTH READINGS (the kinds head of 79.01 is among the candidates); written to best.json before any CONFIRM
           number is read. The ranking of all of them on FIND is printed.
  table    PART ONE. For every head and reading, FIND and CONFIRM, all and by band: the share of positions where the highest FIRMLY KEEPS the class (and firmly loses), the share where the head gives the equal moves different values at all, a unique highest; the first move, the pick and a random equal move beside; paired differences against the first move, the pick, a random equal move and the kinds head.
           The raw reading (the side to move's bonus left in) of the kinds head and of the best head are printed beside, as a reproduction of 79.01 and for comparison. THE CONDITION OF PART TWO: the best head's reading is CLEARLY AHEAD on CONFIRM (all) when its paired difference against the first move AND against the pick have a band wholly above zero; (how many other head readings are, is printed: with forty readings some will be by chance).
  fitrows  the share of the fit rows on which each statement is not zero (what the heads could learn).
  count    PART TWO, only if the condition is not met (a COUNT, not a fit). On FIND's firm room, for each of the 26 statements read in the two readings: the within-position difference between the firmly keeping and the firmly losing equal moves (root mover's view, ours less theirs), in units of the statement's within-position spread, and the share of positions where the keepers are above;
           ranked by size on FIND; CONFIRM printed for the eight largest of FIND; and for those statements which heads read them and with which sign.
  cases    the judged positions, the rule fixed here before any was read: firm-room positions (both parts) where the best head's reading has a unique highest value on a move that is NOT the first move, the first fifteen in the order of the sample's hash; cases_heads.json for ladder_print --heads04. Each carries, for the best head, the columns that carried the difference between its pick and the first move (the weight of a column times its value, root mover's view).

INTERFACE (written before the bodies).
  position_rows(rec, recs, asks, positions, margin) -> (rows, left)     per firm-room position with all blocks defined: firm labels, first and pick indices, every head's value per equal move in each reading, sym and raw (pure given the record; tested).
  scores_of(vec, values) -> {"keep","lose","sep","uniq"}   a rule's weight on firmly keeping and losing moves (pure; tested).
  carriers(rec, name, blocks_a, root_a, blocks_b, root_b, ply, top) -> [(column, delta)]   the columns that carried the difference of two moves for a head (pure; tested).
  effects(rows_S, vec) ...                the count of part two (pure; tested).
  Called by: the command line, ladder_print --heads04.

  .venv/bin/python -m catspace.incubator.analyst.discovery.heads_measure choose|table|count|cases --tag short|full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import heads_read as HRD
from catspace.incubator.analyst.discovery import leaf_ask as LA
from catspace.incubator.analyst.discovery import leaf_firm as LF
from catspace.incubator.analyst.discovery import leaf_line as LL
from catspace.incubator.analyst.discovery import leaf_measure as LM
from catspace.incubator.analyst.discovery import prior_measure as PMS
from catspace.io import paths

READS = ("one", "end")
JUDGED = 15
TOP_STATEMENTS = 8
EPS = LF.EPS
META = ("estimator_78_59", "full", "rows.json")


# ------------------------------------------------------------------------------------------------------------------------------ pure parts

def scores_of(vec: list, values: list) -> dict:
    """The weight a rule that takes the highest of `values` (a tie shares its weight) puts on firmly keeping and on firmly losing moves; whether the values differ at all and whether the highest is unique; the tied moves."""
    w = LF.weights(list(values))
    return {"keep": sum(x for x, v in zip(w, vec) if v == "keep"), "lose": sum(x for x, v in zip(w, vec) if v == "lose"), "sep": float(max(values) - min(values) > EPS), "uniq": float(sum(1 for x in w if x > 0) == 1), "argmax": [j for j, x in enumerate(w) if x > 0]}


def fixed_scores(vec: list, i: "int | None") -> dict:
    n = len(vec)
    w = [1.0 / n] * n if i is None else [1.0 if j == i else 0.0 for j in range(n)]
    return {"keep": sum(x for x, v in zip(w, vec) if v == "keep"), "lose": sum(x for x, v in zip(w, vec) if v == "lose")}


def blocks_of(pos: dict, reading: str) -> "tuple[dict, np.ndarray]":
    """(the input blocks of every equal move at the reading, is_root per move) from a position's stored blocks."""
    rows = pos["rows"]
    blk = {"K": np.array([r[reading]["K"] for r in rows], float), "S": np.array([r[reading]["S"] for r in rows], float)}
    if rows[0][reading]["C"] is not None:
        blk["C"] = np.array([r[reading]["C"] for r in rows], float)
    is_root = np.array([bool(r["end_is_root"]) if reading == "end" else False for r in rows])
    return blk, is_root


def position_rows(rec: HRD.Record, recs: dict, asks: dict, positions: list, margin: float) -> "tuple[list, dict]":
    out, left = [], {"no blocks": 0, "a statement not defined": 0}
    for pos in positions:
        r, a = recs.get(pos["id"]), asks.get(pos["id"])
        if r is None or a is None or not pos["rows"]:
            left["no blocks"] += 1
            continue
        if not all(x["one"]["ok"] and x["end"]["ok"] for x in pos["rows"]):
            left["a statement not defined"] += 1
            continue
        vec = LF.firm_vec(r, a, margin)
        pi = r["tied"].index(r["pick"]) if r.get("pick") in r["tied"] else 0
        vals = {}
        for rd in READS:
            blk, is_root = blocks_of(pos, rd)
            for name in rec.heads:
                need = set(rec.letters(name)) - {"R"}
                sub = {k: v for k, v in blk.items() if k in need}
                vals.setdefault(name, {})[rd] = [float(v) for v in rec.view(name, sub, pos["ply"], is_root, True)]
                vals[name][rd + "_raw"] = [float(v) for v in rec.view(name, sub, pos["ply"], is_root, False)]
        out.append({"id": pos["id"], "game": r["ord"], "band": r["band"], "part": LF.part_of(r["ord"]), "vec": vec, "pick_i": pi, "n": len(vec), "vals": vals})
    return out, left


def carriers(rec: HRD.Record, name: str, blocks_a: dict, root_a: bool, blocks_b: dict, root_b: bool, ply: float, names_of: dict, top: int = 3) -> list:
    """The columns that carried the difference between move a and move b for head `name`: the weight of a column (the log-odds of a win against a loss per raw unit, or the arrival value per raw unit) times the column's value for the ROOT mover (the leaf's own inputs, signed by whose turn it is there); the `top` largest in absolute size."""
    cn, w = rec.weights(name)
    cols_a, cols_b = rec.columns(name, {k: np.asarray(v, float).reshape(1, -1) for k, v in blocks_a.items()}, ply)[0], rec.columns(name, {k: np.asarray(v, float).reshape(1, -1) for k, v in blocks_b.items()}, ply)[0]
    xa, xb = cols_a * (1.0 if root_a else -1.0), cols_b * (1.0 if root_b else -1.0)
    delta = w * (xa - xb)
    label = []
    for ch in rec.letters(name):
        if ch == "R":
            label += ["rating mean", "rating difference", "ply"]
        else:
            label += [names_of[ch][i] for i in range(HRD.WIDTH[ch])]
    order = np.argsort(-np.abs(delta))[:top]
    return [(label[i], float(delta[i])) for i in order if abs(delta[i]) > 0]


def within(values: np.ndarray, vec: list) -> "tuple[np.ndarray, np.ndarray] | None":
    """For one position (values: moves x statements, the root mover's view): the difference between the mean over firmly keeping and over firmly losing moves, and the within-position variance of each statement over all the equal moves."""
    k = np.array([v == "keep" for v in vec])
    l = np.array([v == "lose" for v in vec])
    if not k.any() or not l.any():
        return None
    return values[k].mean(axis=0) - values[l].mean(axis=0), values.var(axis=0)


def eff_band(t: tuple) -> str:
    return f"[{t[1]:+.3f}, {t[2]:+.3f}]"


def effects(diffs: np.ndarray, variances: np.ndarray, games: np.ndarray) -> "tuple[np.ndarray, list, np.ndarray, np.ndarray]":
    """(the effect of each statement: the mean within-position difference over the pooled within-position spread; its 90 pct band by games; the share of positions with the keepers above among those where they differ; the share of positions where it differs at all)."""
    sw = np.sqrt(np.maximum(variances.mean(axis=0), 1e-12))
    scaled = diffs / sw
    eff = scaled.mean(axis=0)
    bands = [PMS.share(scaled[:, j], games) for j in range(scaled.shape[1])]
    nz = np.abs(diffs) > EPS
    above = np.array([float((diffs[nz[:, j], j] > 0).mean()) if nz[:, j].any() else float("nan") for j in range(diffs.shape[1])])
    return eff, bands, above, nz.mean(axis=0)


# ------------------------------------------------------------------------------------------------------------------------------ the command line

class Out:
    def __init__(self, tag: str):
        self.root = HRD.out_root(tag)

    def __call__(self, stage: str, msg: str) -> None:
        line = f"OBS heads {stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


def load_all(tag: str) -> tuple:
    margin = LF.load_margin(HRD.SRC_TAG)
    recs = {r["id"]: r for r in LA.src_records()}
    asks = {a["id"]: a for a in LA.load(LF.out_root(HRD.SRC_TAG))}
    positions = HRD.load(HRD.out_root(tag))
    rec = HRD.Record()
    rows, left = position_rows(rec, recs, asks, positions, margin)
    return margin, recs, asks, positions, rec, rows, left


def columns_of(rows: list) -> dict:
    """{(head, reading): arrays over rows of keep, lose, sep, uniq}, with the first move, the pick and a random equal move under their own names."""
    cols: dict = {}
    for name in HRD.NAMES:
        for rd in READS + tuple(r + "_raw" for r in READS):
            sc = [scores_of(r["vec"], r["vals"][name][rd]) for r in rows]
            cols[(name, rd)] = {k: np.array([s[k] for s in sc], float) for k in ("keep", "lose", "sep", "uniq")}
    for tag, pick in (("first", lambda r: 0), ("pick", lambda r: r["pick_i"]), ("random", lambda r: None)):
        sc = [fixed_scores(r["vec"], pick(r)) for r in rows]
        cols[(tag, "")] = {k: np.array([s[k] for s in sc], float) for k in ("keep", "lose")}
    return cols


def masks_of(rows: list) -> tuple:
    part = np.array([r["part"] for r in rows])
    band = np.array([r["band"] for r in rows])
    game = np.array([r["game"] for r in rows])
    return part, band, game


def choose(tag: str, out: Out) -> None:
    margin, _recs, _asks, _pos, _rec, rows, left = load_all(tag)
    cols = columns_of(rows)
    part, _band, game = masks_of(rows)
    sel = part == "find"
    out("choose", f"MARGIN {margin:g} (79.01's); firm-room positions with every block read {len(rows)} (left out: {left}); FIND positions {int(sel.sum())}; heads {len(HRD.NAMES)}, readings {list(READS)} each from both sides' view; the best head is chosen on FIND ONLY by the share of positions where the move it takes firmly keeps the class")
    rank = sorted(((float(cols[(n, rd)]["keep"][sel].mean()), n, rd) for n in HRD.NAMES for rd in READS), reverse=True)
    for v, n, rd in rank:
        out("choose", f"[find] {n}, read {rd}: firmly keeps {100.0 * v:.1f} pct (first move {100.0 * float(cols[('first', '')]['keep'][sel].mean()):.1f}, pick {100.0 * float(cols[('pick', '')]['keep'][sel].mean()):.1f}, random {100.0 * float(cols[('random', '')]['keep'][sel].mean()):.1f})")
    v, n, rd = rank[0]
    (HRD.out_root(tag) / "best.json").write_text(json.dumps({"head": n, "reading": rd, "find_keep": v, "chosen_on": "find", "kinds_head": HRD.BASELINE}))
    out("choose", f"BEST (written to best.json before CONFIRM is read): {n}, read {rd}, {100.0 * v:.1f} pct on FIND")


def load_best(tag: str) -> dict:
    p = HRD.out_root(tag) / "best.json"
    if not p.exists():
        raise RuntimeError("best.json is missing: the best head is chosen on FIND first (heads_measure choose) before CONFIRM is read")
    return json.loads(p.read_text())


def table(tag: str, out: Out) -> None:
    best = load_best(tag)
    _m, _r, _a, _p, _rec, rows, left = load_all(tag)
    cols = columns_of(rows)
    part, band, game = masks_of(rows)
    out("table", f"firm-room positions {len(rows)} (left out {left}); best head {best['head']} read {best['reading']} (chosen on FIND); the kinds head {best['kinds_head']}; every head from both sides' view ('sym') unless marked raw")
    clear = []
    for p in ("find", "confirm"):
        for b in ("all",) + LF.BANDS:
            sel = (part == p) if b == "all" else (part == p) & (band == b)
            n = int(sel.sum())
            if n == 0:
                continue
            head = f"[{p} {b}] firm room {n} positions"
            for tg in ("first", "pick", "random"):
                out("table", f"{head}: {tg}: firmly keeps {LM.cell(cols[(tg, '')]['keep'], game, sel)}; firmly loses {LM.cell(cols[(tg, '')]['lose'], game, sel)} pct")
            for name in HRD.NAMES:
                for rd in READS + (("end_raw", "one_raw") if name in (best["head"], best["kinds_head"]) else ()):
                    c = cols[(name, rd)]
                    out("table", f"{head}: {name}, read {rd}: firmly keeps {LM.cell(c['keep'], game, sel)}; firmly loses {LM.cell(c['lose'], game, sel)}; separates the equal moves in {LM.cell(c['sep'], game, sel)}; a unique highest in {LM.cell(c['uniq'], game, sel)} pct")
                    if rd in READS and n >= LM.MIN_CELL:
                        diffs = []
                        for other, key in (("first", ("first", "")), ("pick", ("pick", "")), ("random", ("random", "")), ("kinds head", (best["kinds_head"], rd))):
                            d = c["keep"] - cols[key]["keep"]
                            diffs.append((other, PMS.share(d[sel], game[sel])))
                        out("table", f"{head}: {name}, read {rd}, firmly keeps less (points): " + "; ".join(f"{o} {PMS.fmt(t, plus=True)}" for o, t in diffs))
                        if p == "confirm" and b == "all":
                            lo = {o: t[1] for o, t in diffs}
                            if lo["first"] > 0 and lo["pick"] > 0:
                                clear.append((name, rd))
    bn, br = best["head"], best["reading"]
    isc = (bn, br) in clear
    out("table", f"CONDITION OF PART TWO: the best head on FIND ({bn}, read {br}) is clearly ahead of the first move and of the pick on CONFIRM (a band wholly above zero against both): {'YES' if isc else 'NO'}; head readings (of {2 * len(HRD.NAMES)}, chosen after the fact, so some by chance) clearly ahead of both on CONFIRM: {len(clear)} {clear}")
    (HRD.out_root(tag) / "ahead.json").write_text(json.dumps({"best_clear": isc, "any": [list(x) for x in clear]}))


def statement_names() -> dict:
    meta = json.loads(Path(paths.experiment(*META)).read_text())
    return {"K": list(meta["kinds"]), "S": list(meta["statements"]), "C": [f"concept {i}" for i in range(len(meta["concepts"]))]}


def count(tag: str, out: Out) -> None:
    load_best(tag)
    ahead = json.loads((HRD.out_root(tag) / "ahead.json").read_text())
    if ahead["best_clear"]:
        out("count", "PART TWO NOT RUN: the condition is not met (the best head is clearly ahead of the first move and the pick on CONFIRM)")
        return
    _m, _r, _a, positions, rec, rows, _left = load_all(tag)
    ids = {r["id"]: r for r in rows}
    names = statement_names()["S"]
    out("count", f"PART TWO (a count, no fit): the condition holds (no best head clearly ahead); statements {len(names)}; the positions are the firm room with every block read ({len(rows)}); the effect of a statement is the mean over positions of (keepers' mean less losers' mean of the root mover's view, ours less theirs) in units of the statement's within-position spread; positive: the keepers are higher")
    res: dict = {}
    for rd in READS:
        d, v, g, pt = [], [], [], []
        for pos in positions:
            r = ids.get(pos["id"])
            if r is None:
                continue
            blk, is_root = blocks_of(pos, rd)
            x = blk["S"] * np.where(is_root, 1.0, -1.0)[:, None]
            w = within(x, r["vec"])
            if w is None:
                continue
            d.append(w[0]); v.append(w[1]); g.append(r["game"]); pt.append(r["part"])
        d, v, g, pt = np.array(d), np.array(v), np.array(g), np.array(pt)
        fi = pt == "find"
        eff_f, bands_f, above_f, nz_f = effects(d[fi], v[fi], g[fi])
        order = list(np.argsort(-np.abs(eff_f)))
        out("count", f"[find, read {rd}] {int(fi.sum())} positions; statements ranked by the size of the effect: " + "; ".join(f"{names[j]} {eff_f[j]:+.3f} {eff_band(bands_f[j])} (keepers above in {100.0 * above_f[j]:.0f} pct where they differ; differ in {100.0 * nz_f[j]:.0f} pct of positions)" for j in order))
        top = order[:TOP_STATEMENTS]
        ci = ~fi
        eff_c, bands_c, above_c, nz_c = effects(d[ci], v[ci], g[ci])
        for j in top:
            out("count", f"[confirm, read {rd}] ({int(ci.sum())} positions) {names[j]} (one of the {TOP_STATEMENTS} largest on FIND): effect {bands_c[j][0]:+.3f} {eff_band(bands_c[j])}; keepers above in {100.0 * above_c[j]:.0f} pct where they differ; differ in {100.0 * nz_c[j]:.0f} pct of positions; FIND {eff_f[j]:+.3f}")
        sw = np.sqrt(np.maximum(v[fi].mean(axis=0), 1e-12))
        res[rd] = (top, eff_f, sw)
        for j in top:
            reading = []
            agree = 0
            readers = 0
            for name in HRD.NAMES:
                if "S" not in rec.letters(name):
                    continue
                cn, wt = rec.weights(name)
                off = rec.offset(name, "S")
                wj = float(wt[off + j])
                readers += int(abs(wj) > 1e-12)                    # a head whose weight on the column is exactly zero (the sparse fits) does not read it
                agree += int(wj * eff_f[j] > 0)
                if name in ("result_human_RKS", "arrival_human_KS"):
                    reading.append(f"{name}: {wj * sw[j]:+.3f} per spread")
            out("count", f"[read {rd}] {names[j]}: given a nonzero weight by {readers} of the heads that take the statements, {agree} of them weigh it in the direction the keepers differ ({'; '.join(reading)})")


def fitrows(tag: str, out: Out) -> None:
    """What the heads could learn: the share of the fit rows (the quiet positions of 78.59's human games) on which each statement is not zero, and its spread."""
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    rows = HD.load_rows(Path(paths.experiment("estimator_78_59", "full", "rows.npz")))
    sd = rows["sd"]
    names = statement_names()["S"]
    out("fitrows", f"the fit rows of the statement heads: {len(sd)} quiet rows; per statement the share of rows on which ours less theirs is not zero: " + "; ".join(f"{n} {100.0 * float((sd[:, j] != 0).mean()):.1f}" for j, n in enumerate(names)))
    zero = [n for j, n in enumerate(names) if not (sd[:, j] != 0).any()]
    out("fitrows", f"statements that are zero on EVERY fit row (no head can have learned a weight for them): {zero}")


def cases(tag: str, out: Out) -> None:
    best = load_best(tag)
    _m, recs, asks, positions, rec, rows, _left = load_all(tag)
    nm = statement_names()
    ids = {r["id"]: r for r in rows}
    bn, br = best["head"], best["reading"]
    pick = []
    for pos in positions:
        r = ids.get(pos["id"])
        if r is None:
            continue
        w = LF.weights(r["vals"][bn][br])
        top = [j for j, x in enumerate(w) if x > 0]
        if len(top) == 1 and top[0] != 0:
            pick.append((pos, r, top[0]))
    chosen = pick[:JUDGED]
    ln = {x["id"]: x for x in LL.load(LF.out_root(HRD.SRC_TAG))}
    games = {g["ord"]: g for g in FL.universe()}
    outp = []
    for pos, r, j in chosen:
        rr, a = recs[pos["id"]], asks[pos["id"]]
        blk, is_root = blocks_of(pos, br)
        need = set(rec.letters(bn)) - {"R"}

        def bl(k):
            return {c: blk[c][k] for c in need}
        car = carriers(rec, bn, bl(j), bool(is_root[j]), bl(0), bool(is_root[0]), pos["ply"], nm)
        outp.append({"id": pos["id"], "part": r["part"], "band": rr["band"], "units": rr["units"], "cls": rr["cls"], "moves": games[rr["ord"]]["moves"][:rr["ply"]], "depth": rr["depth"], "first": rr["first"], "pick": rr["pick"], "tied": rr["tied"], "cm": rr["cm"], "cp": rr["cp"], "cls2": a["cls2"], "cp2": a["cp2"],
                     "vec": r["vec"], "head": bn, "reading": br, "best_values": r["vals"][bn], "kinds_values": r["vals"][best["kinds_head"]], "chosen": j,
                     "carriers": car, "lines": [{kk: x[kk] for kk in ("names", "settled", "value", "cut")} for x in ln[pos["id"]]["rows"]],
                     "quiet": [[x["one"]["quiet"], x["end"]["quiet"]] for x in pos["rows"]], "end_is_root": [x["end_is_root"] for x in pos["rows"]]})
    (HRD.out_root(tag) / "cases_heads.json").write_text(json.dumps(outp))
    keep = sum(1 for c in outp if c["vec"][c["chosen"]] == "keep")
    lose = sum(1 for c in outp if c["vec"][c["chosen"]] == "lose")
    first_keep = sum(1 for c in outp if c["vec"][0] == "keep")
    out("cases", f"{len(outp)} judged positions (firm room, the best head {bn} read {br} has a unique highest on a move that is not the first move; {len(pick)} such positions in all, of {len(rows)}; the first {JUDGED} in the sample's hash order) written to {HRD.out_root(tag) / 'cases_heads.json'}; the head's move firmly keeps the class in {keep}, firmly loses it in {lose}, neither in {len(outp) - keep - lose}; the first move firmly keeps it in {first_keep}")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["choose", "table", "count", "cases", "fitrows"])
    ap.add_argument("--tag", default="short")
    a = ap.parse_args(argv)
    out = Out(a.tag)
    {"choose": choose, "table": table, "count": count, "cases": cases, "fitrows": fitrows}[a.cmd](a.tag, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
