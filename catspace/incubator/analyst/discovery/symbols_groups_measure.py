"""PLAN 78.88 (SYSTEM_SPEC 7 "subgoals as symbols that go with winning", the form of the model "a combination, mixture of experts kind of thing", 9 the score): ONE REASON ONE PURPOSE, AND WEIGHTS BY PART OF THE GAME.
A sandbox: no player is changed; nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/symbols_78_88/<tag>/ (OUT below, set before any run). Game-free: the game is named only
through the pieces that exist (ladder_measure's bands and unit counts, human_heads_data, graph_layer1/2, symbols, symbols_reasons_measure); the board and the words of a case are printed by ladder_print only.
The 78.80 and 78.81 outputs (sample, events, keys, controls, reasons, fits) are READ, never rebuilt, and never written.

INTERFACE (what each stage takes and writes; who calls it). Each stage is a command and prints OBS lines (also to obs.txt).
  first   (the print, no new method) 78.81's agreement with the move played on the CONFIRM positions by band of units on the board (few 0-8, mid 9-16, many 17 up) and by quiet or not (no tied move takes a unit / some does), against the
          search's first move and against 78.80's pick, paired (bootstrap over games, symbols_measure.paired). Reads fits.pkl, reasons.json. CONFIRM is read here for a score only.
  groups  (A) the groups of the 136 reasons: symbols.groups_of on the FIND rows (its rule is in symbols.py's doc comment, stated before the run); one group one feature (symbols.group_matrix), the group's words. Writes groups.json.
  split   (B) the cells of the reasons of the form "a unit of the mover's bears on a unit of theirs" (symbols.split_borne, symbols.cell_counts): per tied move of the sample, which cells' instance count rose or fell. Writes cells.npz.
  fit     (C) the fits on FIND only: (i) groups, one fit; (ii) groups with the split, one fit; (iii) as (ii), one fit per band of units on the board (fit_choice(..., band=)). Writes fits.pkl, groups.json.
  score   the variants on CONFIRM against 78.81's single fit, paired, by band and overall, quiet or not; FIND counts per band; the endgame coverage of the human sample.
  cases   (JUDGE) thirty decisions, ten a band, by the rule in `cases_rule` (fixed before looking); ladder_print `--groups88` prints them.
  play    only when the condition in the plan holds (see stage_score's last line); uses symbols_play with the reader of this module.
Called by: the command line only.

  .venv/bin/python -m catspace.incubator.analyst.discovery.symbols_groups_measure first|groups|split|fit|score|cases --tag short|full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import dataclasses
import json
import multiprocessing
import os
import pickle
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2 as L2
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import symbols as SY
from catspace.incubator.analyst.discovery import symbols_measure as SM
from catspace.incubator.analyst.discovery import symbols_reasons_measure as SR
from catspace.io import paths

OUT = ("symbols_78_88",)
SRC81 = ("symbols_78_81", "full")                                 # 78.81's run: reasons.json, fits.pkl, controls.npz (read only)
SEED = 8888
BANDS = LM.BANDS                                                  # few 0-8, mid 9-16, many 17 up (the ladder's)


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


def src81() -> Path:
    return Path(paths.experiment(*SRC81))


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS groups88 {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


# ------------------------------------------------------------------------------------------------------------------------------ FIRST: 78.81's agreement by band and by taking

def stage_first(root: Path, say, limit: int = 0) -> None:
    """The print of the plan's FIRST step: the SECOND reading of CONFIRM that symbols_reasons_measure.score_confirm_reasons does (its hits, not printed again), crossed by band and by taking, paired."""
    inp = SR.load_inputs(limit)
    ctl = SR.load_controls(src81(), limit)
    quiet_say = lambda m: None                                     # 78.81's own lines are in its obs.txt; here only the crossing is printed
    res = SR.score_confirm_reasons(src81(), inp, ctl, quiet_say)
    rows, hits, game = res["rows"], res["hits"], res["game"]
    sample_c = [r for r in inp["sample"] if r["part"] == "confirm"]
    P = rows.n_positions()
    units = np.array([r["units"] for r in sample_c])
    band = np.array([LM.band_of(int(u)) for u in units])
    sizes = np.bincount(rows.pos)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    some_take = np.array([bool(rows.C[start[p]:start[p] + sizes[p], 0].any()) for p in range(P)])
    say(f"FIRST: agreement with the move played on the {P} CONFIRM positions of the 78.80 sample (78.81's reasons: the sum of the weights of the reasons carried, no controls in the pick; 'with controls' adds the two controls' weights); "
        f"bands by units on the board: " + ", ".join(f"{b} {int((band == b).sum())}" for b, _l, _h in BANDS))
    cells = [("all", np.ones(P, bool))]
    for b, _l, _h in BANDS:
        cells += [(f"{b}", band == b), (f"{b}, quiet (no tied move takes a unit)", (band == b) & ~some_take), (f"{b}, some tied move takes a unit", (band == b) & some_take)]
    cells += [("all, quiet", ~some_take), ("all, some tied move takes a unit", some_take)]
    for label, sel in cells:
        m = int(sel.sum())
        if m < 5:
            say(f"FIRST {label}: {m} positions (too few for a band)")
            continue
        parts = [f"{k} {SM.wilson_line(hits[k][sel].sum(), m)}" for k in ("new", "new with the controls in the pick", "78.80 pick", "search first")]
        pd = []
        for a, b in (("new", "search first"), ("new", "78.80 pick"), ("new with the controls in the pick", "search first")):
            d = SM.paired(hits[a][sel], hits[b][sel], game[sel], SEED + 1)
            pd.append(f"{a} less {b} {100 * d[0]:+.2f} [{100 * d[1]:+.2f}, {100 * d[2]:+.2f}]")
        say(f"FIRST {label}: {m} positions; " + "; ".join(parts) + "; paired (points, 90 pct bootstrap over games): " + "; ".join(pd))


# ------------------------------------------------------------------------------------------------------------------------------ the inputs, the columns of every variant

def load_all(limit: int = 0) -> dict:
    """78.80's sample, events and 78.81's controls and reasons (read only). `reasons` in 78.81's order."""
    inp = SR.load_inputs(limit)
    ctl = SR.load_controls(src81(), limit)
    reasons = json.loads((src81() / "reasons.json").read_text())
    return {"inp": inp, "ctl": ctl, "reasons": reasons, "n": inp["n"]}


def rows_of(data: dict, E, part: str) -> SY.Choice:
    """The choice rows of one part for a matrix E whose columns are the features (row order = the sample's tied moves), with the controls of 78.81 and the units on the board at each position."""
    sample = data["inp"]["sample"]
    rows = SR.choice_rows(sample, E, list(range(E.shape[1])), data["ctl"], part)
    units = np.array([r["units"] for r in sample if r["part"] == part for _m in r["tied"]])
    return dataclasses.replace(rows, units=units)


def reason_columns(data: dict, split: bool, cells: "dict | None") -> "tuple[list, object]":
    """The reasons (78.81's, or with the split of the unit borne on) and their event matrix over every tied move of the sample: a reason's column is its column of 78.80's event matrix, a cell's is the cell matrix's."""
    R = data["reasons"]
    if not split:
        return [dict(r, cell=None, parent=i) for i, r in enumerate(R)], data["inp"]["M"].tocsc()[:, [r["event"] for r in R]].tocsr()
    S = SY.split_borne(R, [])
    M = data["inp"]["M"].tocsc()
    C = cells["M"].tocsc()
    pat = {k: j for j, k in enumerate(cells["keys"])}
    cols = []
    for r in S:
        if r["cell"] is None:
            cols.append(M[:, r["event"]])
        else:
            cols.append(C[:, (pat[int(r["key"])] * 4 + r["cell"]) * 2 + (0 if r["dir"] == "up" else 1)])
    return S, sparse.hstack(cols, format="csr")


def group_words(R: list, groups: list, rows: SY.Choice) -> list:
    """The groups as records for the print: the members' words, sides, directions, weights and cells, and how many FIND tied moves carry the group."""
    E = rows.E.tocsc()
    out = []
    for gi, g in enumerate(groups):
        members = [{"index": i, "words": R[i]["words"], "side": R[i]["side"], "dir": R[i]["dir"], "weight": float(R[i]["weight"]), "size": R[i]["size"], "level": R[i]["level"], "id": R[i]["id"], "cell": R[i]["cell"]} for i in g]
        car = np.asarray((E[:, g].sum(axis=1) > 0)).ravel()
        out.append({"group": gi, "members": members, "carriers": int(car.sum()), "side": R[g[0]]["side"], "dir": R[g[0]]["dir"], "sign": 1 if R[g[0]]["weight"] > 0 else -1})
    return out


# ------------------------------------------------------------------------------------------------------------------------------ (A) the groups

def stage_groups(root: Path, say, limit: int = 0) -> None:
    data = load_all(limit)
    R, E = reason_columns(data, False, None)
    rows = rows_of(data, E, "find")
    t0 = time.perf_counter()
    groups = SY.groups_of(R, rows)
    sizes = sorted((len(g) for g in groups), reverse=True)
    say(f"GROUPS (the rule in symbols.py: average linkage on the Jaccard index of the carrier sets on the FIND rows, same side, direction and sign of weight, merged while the best pair is at or above {SY.GROUP_SIM}): "
        f"{len(R)} reasons, {len(groups)} groups ({sum(1 for g in groups if len(g) > 1)} with two or more members, {sum(1 for g in groups if len(g) == 1)} alone); sizes largest first: {sizes[:20]}; "
        f"FIND rows {len(rows.pos)} tied moves in {rows.n_positions()} positions; {time.perf_counter() - t0:.1f} s")
    say("GROUPS, for information only (the rule's threshold is not changed): the number of groups at other thresholds: " + ", ".join(f"{t}: {len(SY.groups_of(R, rows, t))}" for t in (0.3, 0.4, 0.6, 0.7)))
    gw = group_words(R, groups, rows)
    (root / "groups_i.json").write_text(json.dumps(gw))
    for g in sorted(gw, key=lambda g: -len(g["members"]))[:20]:
        say(f"GROUP {g['group']}: {len(g['members'])} members, {g['side']} {g['dir']}, weights {min(m['weight'] for m in g['members']):+.2f} to {max(m['weight'] for m in g['members']):+.2f}, carried by {g['carriers']} FIND tied moves; sizes of the members' patterns {sorted({m['size'] for m in g['members']})}")


# ------------------------------------------------------------------------------------------------------------------------------ (B) the cells

_SPLIT_ENV: dict = {}


def _split_env(raws: list) -> dict:
    if not _SPLIT_ENV:
        model = GL.model(HD.GAME, HD.VARIANT)
        kinds = [n[len("kind:"):] for ty, n in model.static_part()["nodes"] if ty == "kind"]
        _SPLIT_ENV.update(env=SY.Reach(model, HD.VARIANT, G2M.goal_kind_of(HD.VARIANT), kinds), pats=[L2.decode(bytes.fromhex(r)) for r in raws])
    return _SPLIT_ENV


def _split_job(job: tuple) -> list:
    """For a chunk of (moves, tied moves): per tied move the (pattern, cell, direction) triples that hold, as column numbers of the cell matrix (see stage_split)."""
    import cs_engine as CE
    chunk, raws = job
    e = _split_env(raws)
    env, pats = e["env"], e["pats"]
    out = []
    for moves, tied in chunk:
        text = env.text(moves)
        items, units = env.items(text)
        before = [SY.cell_counts(items, units, p, env.kinds)[0] for p in pats]
        for m in tied:
            t2 = CE.state(env.variant, text, [m])[0]
            items2, units2 = env.items(t2)
            cols = []
            for pi, p in enumerate(pats):
                after = SY.cell_counts(items2, units2, p, env.kinds)[1]
                up, down = SY.cell_events(before[pi], after)
                cols += [(pi * 4 + c) * 2 for c in up] + [(pi * 4 + c) * 2 + 1 for c in down]
            out.append(cols)
    return out


def split_patterns(reasons: list) -> "tuple[list, list]":
    """The distinct patterns (keys, encodings) of the reasons that split_borne splits, in the reasons' order."""
    keys, raws = [], []
    for r in reasons:
        if r["side"] == "mover" and SY.bear_form(L2.decode(bytes.fromhex(r["bytes"]))) and int(r["key"]) not in keys:
            keys.append(int(r["key"]))
            raws.append(r["bytes"])
    return keys, raws


def stage_split(root: Path, say, workers: int, limit: int = 0) -> None:
    data = load_all(limit)
    sample = data["inp"]["sample"]
    keys, raws = split_patterns(data["reasons"])
    say(f"SPLIT: {len(keys)} patterns of the form 'a unit of the mover's bears on a unit of theirs' ({sum(1 for r in data['reasons'] if r['side'] == 'mover' and int(r['key']) in keys)} reasons, one for each direction); "
        f"cells = 2 x (the unit of theirs is guarded) + (the bearing unit is borne on); {len(sample)} positions, {sum(len(r['tied']) for r in sample)} tied moves; Python, from the graph's own bear items (see symbols.py)")
    chunks = [[(r["moves"], r["tied"]) for r in sample[i:i + 25]] for i in range(0, len(sample), 25)]
    t0 = time.perf_counter()
    res = []
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for i, part in enumerate(pool.imap(_split_job, [(c, raws) for c in chunks], chunksize=1), 1):
            res += part
            if i % 40 == 0:
                say(f"progress: {i} of {len(chunks)} chunks, {time.perf_counter() - t0:.0f} s, load average {os.getloadavg()[0]:.2f}")
    r_i = [i for i, cols in enumerate(res) for _c in cols]
    c_i = [c for cols in res for c in cols]
    M = sparse.csr_matrix((np.ones(len(r_i), np.float32), (r_i, c_i)), shape=(len(res), 8 * len(keys)))
    sparse.save_npz(root / "cells.npz", M)
    (root / "cells_keys.json").write_text(json.dumps({"keys": keys}))
    for pi in range(len(keys)):
        say(f"CELLS pattern {pi} (key {keys[pi]:x}): tied moves on which the count of the cell rose / fell: " + "; ".join(f"cell {c} ({SY.CELL_WORDS[c]}) {int(M[:, (pi * 4 + c) * 2].sum())} / {int(M[:, (pi * 4 + c) * 2 + 1].sum())}" for c in range(4)))
    say(f"SPLIT done: {len(res)} tied moves in {time.perf_counter() - t0:.0f} s, load average {os.getloadavg()[0]:.2f}")


def load_cells(root: Path) -> dict:
    return {"M": sparse.load_npz(root / "cells.npz").tocsr(), "keys": json.loads((root / "cells_keys.json").read_text())["keys"]}


# ------------------------------------------------------------------------------------------------------------------------------ (C) the fits

VARIANTS = (("i", "groups, one fit", False, False), ("ii", "groups with the split of the unit borne on, one fit", True, False), ("iii", "groups with the split, one fit for each band of units", True, True))


def stage_fit(root: Path, say, limit: int = 0) -> None:
    data = load_all(limit)
    cells = load_cells(root)
    out: dict = {}
    names = ()
    for tag, label, split, banded in VARIANTS:
        if tag == "iii":
            out["iii"] = {"groups": out["ii"]["groups"], "R": out["ii"]["R"], "fits": {}, "label": label}
            Eg = out["ii"].pop("Eg")
        else:
            R, E = reason_columns(data, split, cells if split else None)
            rows = rows_of(data, E, "find")
            groups = SY.groups_of(R, rows)
            Eg = SY.group_matrix(E, groups)
            out[tag] = {"groups": groups, "R": R, "label": label, "fits": {}, "Eg": Eg}
            (root / f"groups_{tag}.json").write_text(json.dumps(group_words(R, groups, rows)))
            say(f"VARIANT {tag} ({label}): {len(R)} reasons in {len(groups)} groups ({sum(1 for g in groups if len(g) > 1)} with two or more members, largest {max(len(g) for g in groups)})")
        rows_g = rows_of(data, Eg, "find")
        names = tuple(rows_g.control_names)
        t0 = time.perf_counter()
        if not banded:
            f = SY.fit_choice(rows_g, controls=names, events=None, seed=SEED)
            out[tag]["fits"]["all"] = f
            cv = f["cv"][f["pick"]]
            say(f"FIT {tag} (all positions, FIND only): {len(f['events'])} of {f['n_events']} groups with a weight; penalty {f['lambda']:.5f}; held out inside FIND {cv[1]:.4f} (standard error {cv[2]:.4f}); on all FIND rows {f['loglik']:.4f}; {time.perf_counter() - t0:.0f} s")
        else:
            for b, lo, hi in BANDS:
                f = SY.fit_choice(rows_g, controls=names, events=None, seed=SEED, band=(lo, hi))
                f["band"] = [lo, hi]
                out[tag]["fits"][b] = f
                cv = f["cv"][f["pick"]] if f.get("cv") else None
                say(f"FIT {tag} band {b} ({lo}-{hi} units, FIND only): {f['positions']} positions; {len(f['events'])} of {f['n_events']} groups with a weight; penalty {f['lambda']:.5f}; "
                    + (f"held out inside FIND {cv[1]:.4f} (standard error {cv[2]:.4f}); " if cv else "no path (too few positions); ") + f"{time.perf_counter() - t0:.0f} s")
    for tag in out:
        out[tag].pop("Eg", None)
    pickle.dump({"variants": out, "controls": list(names)}, open(root / "fits.pkl", "wb"))


# ------------------------------------------------------------------------------------------------------------------------------ the scores

def hit_vector(rows: SY.Choice, pick: np.ndarray) -> np.ndarray:
    """1 where the pick (an index inside the position) is the move played."""
    sizes = np.bincount(rows.pos)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    at = np.zeros(rows.n_positions(), int)
    pr = np.flatnonzero(rows.played)
    at[rows.pos[pr]] = pr - start[rows.pos[pr]]
    return (pick == at).astype(float)


def picks_of(rows: SY.Choice, fits: dict, controls: bool) -> np.ndarray:
    if "all" in fits:
        return SY.score_choice(rows, fits["all"], controls)
    return SY.score_choice_by_band(rows, fits, BANDS, controls)


def stage_score(root: Path, say, limit: int = 0) -> dict:
    data = load_all(limit)
    cells = load_cells(root)
    saved = pickle.load(open(root / "fits.pkl", "rb"))
    base = SR.score_confirm_reasons(src81(), data["inp"], data["ctl"], lambda m: None)
    game, hits_b = base["game"], base["hits"]
    sample_c = [r for r in data["inp"]["sample"] if r["part"] == "confirm"]
    sample_f = [r for r in data["inp"]["sample"] if r["part"] == "find"]
    P = len(sample_c)
    band = np.array([LM.band_of(int(r["units"])) for r in sample_c])
    rows_b = base["rows"]
    sizes = np.bincount(rows_b.pos)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    some_take = np.array([bool(rows_b.C[start[p]:start[p] + sizes[p], 0].any()) for p in range(P)])
    say("FIND positions per band (what the fits read): " + ", ".join(f"{b} {sum(1 for r in sample_f if LM.band_of(int(r['units'])) == b)} positions in {len({r['game'] for r in sample_f if LM.band_of(int(r['units'])) == b})} games" for b, _l, _h in BANDS)
        + "; CONFIRM positions per band: " + ", ".join(f"{b} {int((band == b).sum())}" for b, _l, _h in BANDS))
    cell_defs = [("all", np.ones(P, bool))]
    for b, _l, _h in BANDS:
        cell_defs += [(b, band == b), (f"{b}, quiet", (band == b) & ~some_take), (f"{b}, some tied move takes a unit", (band == b) & some_take)]
    cell_defs += [("all, quiet", ~some_take), ("all, some tied move takes a unit", some_take)]
    result: dict = {"hits": {}, "picks": {}, "game": game, "hits_base": hits_b, "band": band, "some_take": some_take}
    for tag, _label, split, _banded in VARIANTS:
        v = saved["variants"][tag]
        _R, E = reason_columns(data, split, cells if split else None)
        rows = rows_of(data, SY.group_matrix(E, v["groups"]), "confirm")
        for ctl_in in (False, True):
            pk = picks_of(rows, v["fits"], ctl_in)
            result["picks"][(tag, ctl_in)] = pk
            result["hits"][(tag, ctl_in)] = hit_vector(rows, pk)
    for label, sel in cell_defs:
        m = int(sel.sum())
        if m < 5:
            say(f"SCORE {label}: {m} positions (too few)")
            continue
        for ctl_in, kind in ((False, "new"), (True, "new with the controls in the pick")):
            parts = [f"78.81 single fit {SM.wilson_line(hits_b[kind][sel].sum(), m)}", f"search first {SM.wilson_line(hits_b['search first'][sel].sum(), m)}"]
            pd = []
            for tag, _lab, _s, _b in VARIANTS:
                h = result["hits"][(tag, ctl_in)]
                parts.append(f"{tag} {SM.wilson_line(h[sel].sum(), m)}")
                dbase = SM.paired(h[sel], hits_b[kind][sel], game[sel], SEED + 2)
                dfirst = SM.paired(h[sel], hits_b["search first"][sel], game[sel], SEED + 3)
                pd.append(f"{tag} less 78.81 {100 * dbase[0]:+.2f} [{100 * dbase[1]:+.2f}, {100 * dbase[2]:+.2f}], less search first {100 * dfirst[0]:+.2f} [{100 * dfirst[1]:+.2f}, {100 * dfirst[2]:+.2f}]")
            say(f"SCORE {label} ({m} positions), pick {'with the controls' if ctl_in else 'events only'}: " + "; ".join(parts) + "; paired (points, 90 pct bootstrap over games): " + "; ".join(pd))
    return result


def stage_report(root: Path, say, limit: int = 0) -> None:
    """The fits in numbers: for each variant and fit the groups with a weight, the strongest five, and the weights of the groups that hold a cell of the split."""
    saved = pickle.load(open(root / "fits.pkl", "rb"))
    for tag, label, _split, _banded in VARIANTS:
        v = saved["variants"][tag]
        gw = json.loads((root / f"groups_{'i' if tag == 'i' else 'ii'}.json").read_text())
        for name, f in v["fits"].items():
            w = dict(zip(f["events"], f["weights"]))
            top = sorted(w.items(), key=lambda x: -abs(x[1]))[:5]
            cellw = [(g["group"], w[g["group"]], [m["cell"] for m in g["members"] if m["cell"] is not None]) for g in gw if any(m["cell"] is not None for m in g["members"]) and g["group"] in w]
            say(f"REPORT variant {tag} fit '{name}': {len(w)} groups with a weight of {f['n_events']}; controls {', '.join(f'{k} {x:+.2f}' for k, x in f['controls'].items() if k == 'takes')}; strongest five: " + ", ".join(f"group {g} {x:+.3f}" for g, x in top)
                + ("; groups holding a cell of the split: " + ", ".join(f"group {g} cells {c} weight {x:+.3f}" for g, x, c in cellw) if cellw else ""))
    gw = json.loads((root / "groups_ii.json").read_text())
    say("REPORT the cell groups of variant ii: " + "; ".join(f"group {g['group']} ({len(g['members'])} members, {g['side']} {g['dir']}, cells {[m['cell'] for m in g['members'] if m['cell'] is not None]}, carried by {g['carriers']} FIND tied moves)" for g in gw if any(m["cell"] is not None for m in g["members"])))


def stage_coverage(root: Path, say, limit: int = 0) -> None:
    """The endgame coverage of the human sample as 78.80 printed it (positions tried / kept as ties, by units left on the board), and the kept positions by part and band; nothing is filled with an engine's continuation."""
    src = Path(paths.experiment(*SM.OUT, "full"))
    tries = SM.jsonl(src / "tries.jsonl")
    sample = SM.jsonl(src / "sample.jsonl")
    ok = [r for r in tries if r.get("ok")]
    bt = {SM.bin_of(r["units"]): 0 for r in ok}
    for r in ok:
        bt[SM.bin_of(r["units"])] += 1
    bs = {b: 0 for b in bt}
    for r in sample:
        bs[SM.bin_of(r["units"])] += 1
    order = [SM.bin_of(a) for a, _h in SM.BINS]
    say("ENDGAME COVERAGE of the human sample by units left on the board (tried / kept as ties): " + ", ".join(f"{b}: {bt.get(b, 0)} / {bs.get(b, 0)}" for b in order))
    for part in ("find", "confirm"):
        rs = [r for r in sample if r["part"] == part]
        say(f"ENDGAME COVERAGE kept ties, {part}: " + ", ".join(f"{b} {sum(1 for r in rs if LM.band_of(int(r['units'])) == b)} positions in {len({r['game'] for r in rs if LM.band_of(int(r['units'])) == b})} games" for b, _l, _h in BANDS)
            + f" (of {len(rs)} positions in {len({r['game'] for r in rs})} games)")
    few = [r for r in ok if LM.band_of(int(r["units"])) == "few"]
    say(f"ENDGAME COVERAGE the few band (8 units or fewer): {len(few)} of {len(ok)} playable positions tried ({100 * len(few) / max(1, len(ok)):.1f} pct), {sum(1 for r in sample if r['units'] <= 8)} of {len(sample)} kept as ties ({100 * sum(1 for r in sample if r['units'] <= 8) / max(1, len(sample)):.1f} pct); "
        f"a few-band position is kept as a tie in {100 * sum(1 for r in sample if r['units'] <= 8) / max(1, len(few)):.1f} pct of those tried; no generated position enters any fit or score of this plan")


# ------------------------------------------------------------------------------------------------------------------------------ (JUDGE) the thirty decisions

CASES_PER_BAND = 10
TOP_GROUPS = 3


def cases_rule() -> str:
    return ("THE RULE (fixed before any position was read): the best variant is the one of i, ii, iii whose events-only pick agrees most with the move played on CONFIRM in the mid and few bands pooled; "
            "in each band, the CONFIRM positions in the sample's order where that pick differs from the search's first move, at most one a game, the first ten.")


def best_variant(result: dict) -> str:
    sel = result["band"] != "many"
    scores = {tag: float(result["hits"][(tag, False)][sel].mean()) for tag, *_r in VARIANTS}
    return max(scores, key=lambda t: scores[t])


def carried_top(rows: SY.Choice, row: int, fit: dict, k: int = TOP_GROUPS) -> list:
    """The groups the tied move (a row of rows) carries that have a weight in the fit, the strongest k by |weight|: [[group, weight]]."""
    E = rows.E.tocsr()
    cols = E.indices[E.indptr[row]:E.indptr[row + 1]]
    w = dict(zip(fit["events"], fit["weights"]))
    mine = [(int(c), float(w[int(c)])) for c in cols if int(c) in w]
    return [list(x) for x in sorted(mine, key=lambda x: -abs(x[1]))[:k]]


def stage_cases(root: Path, say, limit: int = 0) -> None:
    data = load_all(limit)
    cells = load_cells(root)
    saved = pickle.load(open(root / "fits.pkl", "rb"))
    result = stage_score(root, lambda m: None, limit)
    tag = best_variant(result)
    v = saved["variants"][tag]
    split = tag != "i"
    _R, E = reason_columns(data, split, cells if split else None)
    rows = rows_of(data, SY.group_matrix(E, v["groups"]), "confirm")
    pk = result["picks"][(tag, False)]
    sample_c = [r for r in data["inp"]["sample"] if r["part"] == "confirm"]
    sizes = np.bincount(rows.pos)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    say(f"CASES: best variant {tag} ({v['label']}); " + cases_rule())
    out, seen = [], {b: set() for b, _l, _h in BANDS}
    n_diff = {b: 0 for b, _l, _h in BANDS}
    for p, r in enumerate(sample_c):
        b = LM.band_of(int(r["units"]))
        if pk[p] == 0:
            continue
        n_diff[b] += 1
        if len([c for c in out if c["band"] == b]) >= CASES_PER_BAND or r["game"] in seen[b]:
            continue
        seen[b].add(r["game"])
        fit = v["fits"]["all"] if "all" in v["fits"] else v["fits"][b]
        out.append({"id": r["id"], "game": r["game"], "ply": r["ply"], "band": b, "units": r["units"], "moves": r["moves"], "tied": r["tied"], "played": r["played"], "first": r["tied"][0], "pick": r["tied"][int(pk[p])],
                    "pick_groups": carried_top(rows, start[p] + int(pk[p]), fit), "first_groups": carried_top(rows, start[p], fit)})
    (root / "cases88.json").write_text(json.dumps({"variant": tag, "groups_file": f"groups_{tag}.json", "cases": out}))
    say(f"CASES written: {len(out)} decisions, " + ", ".join(f"{b} {sum(1 for c in out if c['band'] == b)} (of {n_diff[b]} CONFIRM positions where the pick differs from the first move)" for b, _l, _h in BANDS)
        + f"; the pick equals the move played in {sum(1 for c in out if c['pick'] == c['played'])} of them, the first move does in {sum(1 for c in out if c['first'] == c['played'])}")


# ------------------------------------------------------------------------------------------------------------------------------ main

def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("first", "groups", "split", "fit", "score", "coverage", "report", "cases", "play"))
    ap.add_argument("--tag", default="short")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    say = Say(root, a.stage)
    t0 = time.perf_counter()
    say(f"stage start; tag {a.tag}; limit {a.limit}; load average {os.getloadavg()[0]:.2f}")
    if a.stage == "first":
        stage_first(root, say, a.limit)
    elif a.stage == "groups":
        stage_groups(root, say, a.limit)
    elif a.stage == "split":
        stage_split(root, say, a.workers, a.limit)
    elif a.stage == "fit":
        stage_fit(root, say, a.limit)
    elif a.stage == "score":
        stage_score(root, say, a.limit)
    elif a.stage == "report":
        stage_report(root, say, a.limit)
    elif a.stage == "coverage":
        stage_coverage(root, say, a.limit)
    elif a.stage == "cases":
        stage_cases(root, say, a.limit)
    say(f"stage done in {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
