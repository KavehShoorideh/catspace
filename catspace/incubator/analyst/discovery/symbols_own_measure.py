"""PLAN 78.90 (SYSTEM_SPEC 7 "subgoals as symbols that go with winning" -- the opportunity policy's hint among equal moves --, 9 the score): THE MOVER'S OWN LOOSE UNIT, AND ONE FIT ON BOTH SAMPLES.
A sandbox: no player is changed; nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/symbols_78_90/<tag>/ (OUT below, set before any run). Game-free: the game is named only through
the pieces that exist (ladder_measure's bands and unit counts, human_heads_data, graph_layer1/2, subgoal_goals' replay, symbols, symbols_groups_measure, symbols_sample_measure); the board and the words of a case are printed by
ladder_print only. The outputs of 78.80, 78.81, 78.88 and 78.89 are READ, never rebuilt, never written. No player id is written (a game is an ordinal of its run). No oracle, engine value or labeller number is in any fit; no worth
of any unit is assumed; every fit reads FIND rows of human games only (Choice.fill_ok stays False; the second sample holds no generated row and this module asserts it).

INTERFACE (what each stage takes and writes; who calls it). Each stage is a command and prints OBS lines (also to obs.txt).
  cells    (A, the facts)  for each of the two samples (78.80's, 78.89's): per tied move the cells of the OTHER side's bearing count (symbols.cell_counts, second of the position before, first of the position after: "a unit of theirs
           bears on a unit of the mover's", cell = 2 x (the unit of the mover's is guarded) + (the bearing unit is borne on)) that rose and that fell, for the bear-form patterns of the 136 reasons of 78.81 (both sides), and whether
           the move LEAVES a unit of the mover's borne on and unguarded that was not so before (symbols.loose_units, symbols.left_loose, the moved unit followed by the engine's replay). Writes own_cells_s1.npz, own_cells_s2.npz
           (rows = the sample's tied moves; columns (pattern x 4 + cell) x 2 + 0 up / 1 down), own_keys.json, left_s1.npy, left_s2.npy.
  fit      (B)  FIND = the union of the two samples' FIND parts (human rows, games of the two samples are disjoint: asserted). The reasons are grouped again on that FIND by the rule of 78.88 (symbols.groups_of, one fixed
           threshold, no tuning) and the joint choice fit of 78.81 (symbols.fit_choice, the controls of 78.81, the penalty chosen on FIND alone, seed fixed) is run twice: variant "ii" the 145 reasons of 78.88's variant ii (the split of
           the mover's own bearing into cells), variant "A" those with the cells of the other side's bearing (symbols.split_borne side="own": the existing reason of that form is replaced by its cells, and the up and down cells of
           every bear-form pattern are added as candidates with no parent reason, weight 0). Writes fits.pkl, groups_ii.json, groups_A.json.
  report   the cells of A in words, with their weights in the fit of A.
  score    on each CONFIRM part (78.80's, 78.89's) and on both pooled: agreement with the move played of the pick of the old weights (78.88's fit, as it was), of "ii" and of "A", against the search's first move and against the old
           weights, and A against ii; by band of units on the board (few 0-8, mid 9-16, many 17 up) and by quiet (no tied move takes a unit) or not; paired, 90 pct bootstrap over games. THE PLAY CONDITION (fixed here, before any
           score was read): A less ii, the pick without the controls, has a paired band whose lower end is above zero in at least one of the cells few, mid, many on either CONFIRM part. Part C of the plan (count): on the tie
           positions of both CONFIRM parts, by band, the tie sets with a move that leaves a unit of the mover's borne on and unguarded that was not so before, and how often the first move, the old pick, the ii pick, the A pick
           and the move played is such a move.
  cases    (JUDGE) thirty decisions: ten a band (few, mid, many), five from each CONFIRM part, in the part's order, one a game, where A's pick (without the controls) differs from the old weights' pick; cases90.json for
           ladder_print `--own90`.
  agreement, match, playreport   (PLAY, only if the condition holds, which it did: see the OBS line of `score`)  agreement: the player (symbols_own_play) against the offline pick of A on 200 positions of each
           CONFIRM part; match: `--b control|reasons --set A|B --pairs 100`: the player `ownfit` (the fit of A, no window, the full player) against the control or against 78.82's reasons player, the settings of 78.89
           (0.2 s, ledger share 0.4, ply cap 120, 9 workers, uncharged, bootstrap bands over opening pairs, the load wait first); playreport: the pooled lines.
Called by: the command line only; ladder_print.py reads cases90.json and groups_A.json (`--own90`).

  .venv/bin/python -m catspace.incubator.analyst.discovery.symbols_own_measure cells|fit|report|score|cases --tag short|full
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

from catspace.incubator.analyst.discovery import graph_layer2 as L2
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import symbols as SY
from catspace.incubator.analyst.discovery import symbols_groups_measure as SGM
from catspace.incubator.analyst.discovery import symbols_measure as SM
from catspace.incubator.analyst.discovery import symbols_sample_measure as SS
from catspace.io import paths

OUT = ("symbols_78_90",)
SEED = 9090
BANDS = SGM.BANDS
SAMPLES = (1, 2)                                                  # 1: the 78.80 sample (the early game), 2: the 78.89 sample (the later game)
PICKS = (False, True)                                             # the pick without / with the controls' weights


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


def root_of(sample_no: int) -> Path:
    """Where a sample's own files (sample, events, cells, controls) are: 78.80's run or 78.89's run (read only)."""
    return SS.src80() if sample_no == 1 else Path(paths.experiment(*SS.OUT, "full"))


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS own90 {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


# ------------------------------------------------------------------------------------------------------------------------------ the patterns and the candidate reasons

def bear_patterns(reasons: list) -> "tuple[list, list]":
    """The distinct patterns (keys, encodings in hexadecimal) of the reasons of either side that are of the bear form, in the reasons' order."""
    keys, raws = [], []
    for r in reasons:
        if SY.bear_form(L2.decode(bytes.fromhex(r["bytes"]))) and int(r["key"]) not in keys:
            keys.append(int(r["key"]))
            raws.append(r["bytes"])
    return keys, raws


def with_candidates(reasons: list) -> list:
    """78.81's reasons and, for every bear-form pattern and each direction of the OTHER side that no reason holds, a candidate with no parent reason: weight 0 (it has no sign of its own), the pattern's words from a reason
    of that pattern. Appended after the reasons, in the order of the patterns, up before down."""
    keys, _raws = bear_patterns(reasons)
    out = list(reasons)
    for k in keys:
        base = next(r for r in reasons if int(r["key"]) == k and SY.bear_form(L2.decode(bytes.fromhex(r["bytes"]))))
        for d in SY.DIRS:
            if not any(r["side"] == "other" and int(r["key"]) == k and r["dir"] == d for r in reasons):
                out.append(dict(base, side="other", dir=d, column=None, event=None, weight=0.0, weight_events_alone=0.0))
    return out


def reasons_a(reasons: list) -> list:
    """The reasons of variant A: both splits applied to the reasons with the candidates (see the module doc)."""
    return SY.split_borne(SY.split_borne(with_candidates(reasons), [], side="own"), [], side="theirs")


# ------------------------------------------------------------------------------------------------------------------------------ cells: the other side's bearing, and the unit left loose

_FACTS: dict = {}


def _own_job(job: tuple) -> list:
    """For a chunk of (moves, tied moves): per tied move (the columns of the own-cell matrix that hold, 1 if the move leaves a unit of the mover's borne on and unguarded that was not so before, 1 if the moved unit could
    not be followed). The other side's count of a pattern is the second of the position before the move and the first of the position after it."""
    import cs_engine as CE
    chunk, raws = job
    e = SGM._split_env(raws)
    env, pats = e["env"], e["pats"]
    if "facts" not in _FACTS:
        _FACTS["facts"] = SG.Facts(HD.VARIANT)
    facts = _FACTS["facts"]
    out = []
    for moves, tied in chunk:
        text = env.text(moves)
        items, units = env.items(text)
        before = [SY.cell_counts(items, units, p, env.kinds)[1] for p in pats]
        loose0 = SY.loose_units(items, units, 0)
        for m in tied:
            t2 = CE.state(env.variant, text, [m])[0]
            items2, units2 = env.items(t2)
            cols = []
            for pi, p in enumerate(pats):
                after = SY.cell_counts(items2, units2, p, env.kinds)[0]
                up, down = SY.cell_events(before[pi], after)
                cols += [(pi * 4 + c) * 2 for c in up] + [(pi * 4 + c) * 2 + 1 for c in down]
            pl = facts.replay(text, [m])
            followed = bool(pl and pl[0]["from"] and pl[0]["to"])
            frm, to = (pl[0]["from"], pl[0]["to"]) if followed else ("", "")
            left = SY.left_loose(loose0, SY.loose_units(items2, units2, 1), frm, to)
            out.append((cols, int(bool(left)), int(not followed)))
    return out


def load_sample_of(sample_no: int) -> list:
    return SS.load_sample(root_of(sample_no))


def stage_cells(root: Path, say, workers: int, limit: int = 0) -> None:
    reasons = json.loads((SGM.src81() / "reasons.json").read_text())
    keys, raws = bear_patterns(reasons)
    (root / "own_keys.json").write_text(json.dumps({"keys": keys, "bytes": raws}))
    say(f"CELLS: {len(keys)} bear-form patterns among the {len(reasons)} reasons (both sides); for each, the four cells of the OTHER side's count (a unit of theirs bears on a unit of the mover's) that rose and that fell across "
        f"each tied move, and whether the move leaves a unit of the mover's borne on and unguarded that was not so before; cell = 2 x (the unit of the mover's is guarded) + (the bearing unit of theirs is borne on)")
    for no in SAMPLES:
        sample = load_sample_of(no)
        if limit:
            sample = sample[:limit]
        t0 = time.perf_counter()
        chunks = [[(r["moves"], r["tied"]) for r in sample[i:i + 25]] for i in range(0, len(sample), 25)]
        res = []
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for i, part in enumerate(pool.imap(_own_job, [(c, raws) for c in chunks], chunksize=1), 1):
                res += part
                if i % 100 == 0:
                    say(f"progress sample {no}: {i} of {len(chunks)} chunks, {time.perf_counter() - t0:.0f} s, load average {os.getloadavg()[0]:.2f}")
        r_i = [i for i, (cols, _l, _u) in enumerate(res) for _c in cols]
        c_i = [c for cols, _l, _u in res for c in cols]
        M = sparse.csr_matrix((np.ones(len(r_i), np.float32), (r_i, c_i)), shape=(len(res), 8 * len(keys)))
        sparse.save_npz(root / f"own_cells_s{no}.npz", M)
        np.save(root / f"left_s{no}.npy", np.array([l for _c, l, _u in res], np.uint8))
        left = np.array([l for _c, l, _u in res])
        unf = np.array([u for _c, _l, u in res])
        say(f"CELLS sample {no}: {len(sample)} positions, {len(res)} tied moves in {time.perf_counter() - t0:.0f} s; moves that leave a unit of the mover's borne on and unguarded that was not so before: {int(left.sum())} "
            f"({100 * left.mean():.1f} pct); moves whose moved unit could not be followed by the replay: {int(unf.sum())}")
        for pi in range(len(keys)):
            say(f"CELLS sample {no} pattern {pi} (key {keys[pi]:x}): tied moves on which the cell's count rose / fell: "
                + "; ".join(f"cell {c} ({SY.CELL_WORDS_OWN[c]}) {int(M[:, (pi * 4 + c) * 2].sum())} / {int(M[:, (pi * 4 + c) * 2 + 1].sum())}" for c in range(4)))


def load_own(root: Path, no: int, n_rows: int) -> dict:
    M = sparse.load_npz(root / f"own_cells_s{no}.npz").tocsr()
    left = np.load(root / f"left_s{no}.npy")
    if M.shape[0] != n_rows or len(left) != n_rows:
        raise ValueError(f"sample {no}: the own cells hold {M.shape[0]} rows, the sample {n_rows}")
    return {"M": M, "left": left, "keys": json.loads((root / "own_keys.json").read_text())["keys"]}


# ------------------------------------------------------------------------------------------------------------------------------ the two samples as one

def load_sample_data(no: int) -> dict:
    """A `data` dict (inp: sample, M, n; ctl; reasons) and the mover's cells (78.88's reader) of a sample."""
    if no == 1:
        data = SGM.load_all(0)
    else:
        data = SS.load_data(root_of(2))
    data["cells"] = SGM.load_cells(root_of(2) if no == 2 else SS.src88())
    return data


def columns(data: dict, own: dict, variant: str) -> "tuple[list, object]":
    """The reasons of a variant and their event matrix over every tied move of a sample: 'ii' = 78.88's variant ii (symbols_groups_measure.reason_columns); 'A' = reasons_a with the columns of the own cells."""
    if variant == "ii":
        return SGM.reason_columns(data, True, data["cells"])
    R = reasons_a(data["reasons"])
    M = data["inp"]["M"].tocsc()
    mover = data["cells"]
    mpat = {k: j for j, k in enumerate(mover["keys"])}
    mC = mover["M"].tocsc()
    opat = {k: j for j, k in enumerate(own["keys"])}
    oC = own["M"].tocsc()
    cols = []
    for r in R:
        if r["cell"] is None:
            cols.append(M[:, r["event"]])
        elif r["side"] == "mover":
            cols.append(mC[:, (mpat[int(r["key"])] * 4 + r["cell"]) * 2 + (0 if r["dir"] == "up" else 1)])
        else:
            cols.append(oC[:, (opat[int(r["key"])] * 4 + r["cell"]) * 2 + (0 if r["dir"] == "up" else 1)])
    return R, sparse.hstack(cols, format="csr")


def rows_part(data: dict, E, part: str) -> SY.Choice:
    """symbols_groups_measure.rows_of, the rows asserted to be human (Choice.fill_ok stays False)."""
    if {r.get("source", "human") for r in data["inp"]["sample"]} != {"human"}:
        raise ValueError("a generated position in the sample")
    rows = SGM.rows_of(data, E, part)
    assert not rows.fill_ok
    return rows


def concat_choice(a: SY.Choice, b: SY.Choice) -> SY.Choice:
    """Two sets of rows as one: positions renumbered, the games of the two disjoint (raises if one game is in both)."""
    if set(np.unique(a.game).tolist()) & set(np.unique(b.game).tolist()):
        raise ValueError("a game is in both samples")
    if a.control_names != b.control_names:
        raise ValueError("the controls differ")
    return SY.Choice(sparse.vstack([a.E, b.E], format="csr"), np.vstack([a.C, b.C]), a.control_names, np.concatenate([a.pos, b.pos + a.n_positions()]), np.concatenate([a.played, b.played]),
                     np.concatenate([a.game, b.game]), np.concatenate([a.part, b.part]), np.concatenate([a.source, b.source]), np.concatenate([a.units, b.units]), False)


def load_both(root: Path) -> dict:
    out = {}
    for no in SAMPLES:
        data = load_sample_data(no)
        n_rows = int(sum(len(r["tied"]) for r in data["inp"]["sample"]))
        out[no] = {"data": data, "own": load_own(root, no, n_rows)}
    return out


# ------------------------------------------------------------------------------------------------------------------------------ B: the fits

def stage_fit(root: Path, say, limit: int = 0) -> None:
    both = load_both(root)
    out: dict = {"variants": {}}
    names: tuple = ()
    old = pickle.load(open(Path(paths.experiment(*SS.SRC88)) / "fits.pkl", "rb"))
    for variant in ("ii", "A"):
        finds, R = [], None
        for no in SAMPLES:
            R, E = columns(both[no]["data"], both[no]["own"], variant)
            finds.append(rows_part(both[no]["data"], E, "find"))
        rows = concat_choice(finds[0], finds[1])
        t0 = time.perf_counter()
        groups = SY.groups_of(R, rows)
        Eg = SY.group_matrix(rows.E, groups)
        rows_g = dataclasses.replace(rows, E=Eg)
        names = tuple(rows_g.control_names)
        say(f"VARIANT {variant}: {len(R)} reasons in {len(groups)} groups ({sum(1 for g in groups if len(g) > 1)} with two or more members, largest {max(len(g) for g in groups)}); FIND = the union: "
            f"{rows.n_positions()} positions ({finds[0].n_positions()} of sample 1, {finds[1].n_positions()} of sample 2), {len(rows.pos)} tied moves, {len(np.unique(rows.game))} games; grouping {time.perf_counter() - t0:.0f} s")
        (root / f"groups_{variant}.json").write_text(json.dumps(SGM.group_words(R, groups, rows)))
        t1 = time.perf_counter()
        f = SY.fit_choice(rows_g, controls=names, events=None, seed=SEED)
        cv = f["cv"][f["pick"]]
        say(f"FIT {variant} (all FIND positions): {len(f['events'])} of {f['n_events']} groups with a weight; penalty {f['lambda']:.5f}; held out inside FIND {cv[1]:.4f} (standard error {cv[2]:.4f}); on all FIND rows {f['loglik']:.4f}; {time.perf_counter() - t1:.0f} s")
        out["variants"][variant] = {"groups": groups, "R": R, "fit": f}
    out["controls"] = list(names)
    out["old_groups"] = old["variants"]["ii"]["groups"]
    pickle.dump(out, open(root / "fits.pkl", "wb"))
    po = old["variants"]["ii"]["fits"]["all"]
    say(f"FIT old weights for reference (78.88's variant ii, fitted on 78.80's FIND, {po['positions']} positions): {len(po['events'])} of {po['n_events']} groups with a weight, penalty {po['lambda']:.5f}")


def stage_report(root: Path, say, limit: int = 0) -> None:
    """The cells of the other side in words, with their weights in the fit of A (and the mover's cells beside, from the same fit)."""
    saved = pickle.load(open(root / "fits.pkl", "rb"))
    v = saved["variants"]["A"]
    gw = json.loads((root / "groups_A.json").read_text())
    w = dict(zip(v["fit"]["events"], v["fit"]["weights"]))
    say(f"REPORT fit A: {len(w)} of {v['fit']['n_events']} groups with a weight; controls " + ", ".join(f"{k} {x:+.2f}" for k, x in v["fit"]["controls"].items() if k == "takes"))
    for g in gw:
        cells = [m for m in g["members"] if m["cell"] is not None]
        if not cells:
            continue
        side = g["side"]
        own = side == "other"
        words = "; ".join(f"{m['dir']}: {m['words'].split('; ')[-1]}" for m in cells[:2])
        say(f"REPORT cell group {g['group']} ({'the other side bears on the mover' if own else 'the mover bears on theirs'}, {g['dir']}, {len(g['members'])} members, carried by {g['carriers']} FIND tied moves): "
            f"weight {w.get(g['group'], 0.0):+.3f}{'' if g['group'] in w else ' (no weight)'}; {words}")
    top = sorted(w.items(), key=lambda x: -abs(x[1]))[:8]
    say("REPORT the eight strongest groups of A: " + "; ".join(f"group {g} {x:+.3f} ({gw[g]['side']} {gw[g]['dir']}, {len(gw[g]['members'])} members, cells {[m['cell'] for m in gw[g]['members'] if m['cell'] is not None]})" for g, x in top))


# ------------------------------------------------------------------------------------------------------------------------------ the scores

def first_hits(sample_c: list) -> np.ndarray:
    return np.array([r["tied"][0] == r["played"] for r in sample_c], float)


def left_of(rows: SY.Choice, left_all: np.ndarray, data: dict) -> np.ndarray:
    """The 'leaves a unit loose' flag of every row of a part (the rows of the part's CONFIRM positions, in the sample's order)."""
    sample = data["inp"]["sample"]
    sizes = np.array([len(r["tied"]) for r in sample])
    start = np.concatenate([[0], np.cumsum(sizes)])
    keep = [i for i, r in enumerate(sample) if r["part"] == "confirm"]
    return np.concatenate([left_all[start[i]:start[i + 1]] for i in keep])


def part_picks(root: Path, no: int, both: dict, saved: dict, old: dict) -> dict:
    """On one CONFIRM part: the rows, the band, quiet or not, the first move's hit, each variant's pick and hit (without and with the controls), the flags."""
    data, own = both[no]["data"], both[no]["own"]
    sample_c = [r for r in data["inp"]["sample"] if r["part"] == "confirm"]
    assert all(r.get("source", "human") == "human" for r in sample_c)
    P = len(sample_c)
    res = {"no": no, "sample": sample_c, "P": P, "game": np.array([r["game"] for r in sample_c]), "units": np.array([r["units"] for r in sample_c]),
           "first": first_hits(sample_c), "picks": {}, "hits": {}, "rows": {}}
    res["band"] = np.array([LM.band_of(int(u)) for u in res["units"]])
    for variant in ("old", "ii", "A"):
        if variant == "old":
            R, E = columns(data, own, "ii")
            groups, fits = saved["old_groups"], old["variants"]["ii"]["fits"]
        else:
            R, E = columns(data, own, variant)
            groups, fits = saved["variants"][variant]["groups"], {"all": saved["variants"][variant]["fit"]}
        rows = rows_part(data, SY.group_matrix(E, groups), "confirm")
        res["rows"][variant] = rows
        for ctl_in in PICKS:
            pk = SGM.picks_of(rows, fits, ctl_in)
            res["picks"][(variant, ctl_in)] = pk
            res["hits"][(variant, ctl_in)] = SGM.hit_vector(rows, pk)
    rows0 = res["rows"]["old"]
    res["some_take"] = SS._some_take(rows0, P)
    res["left_rows"] = left_of(rows0, own["left"], data)
    sizes = np.bincount(rows0.pos)
    res["start"] = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    res["sizes"] = sizes
    return res


def cell_defs(res: dict) -> list:
    P, band, some = res["P"], res["band"], res["some_take"]
    out = [("all", np.ones(P, bool))]
    for b, _l, _h in BANDS:
        out += [(b, band == b), (f"{b}, quiet", (band == b) & ~some), (f"{b}, some tied move takes a unit", (band == b) & some)]
    out += [("all, quiet", ~some), ("all, some tied move takes a unit", some)]
    return out


def pooled(parts: list) -> dict:
    """Both CONFIRM parts as one set of positions (games are disjoint between the samples)."""
    keys = ("game", "units", "first", "band", "some_take")
    out = {"no": 0, "P": sum(p["P"] for p in parts), "picks": {}, "hits": {}}
    for k in keys:
        out[k] = np.concatenate([p[k] for p in parts])
    for k in parts[0]["hits"]:
        out["hits"][k] = np.concatenate([p["hits"][k] for p in parts])
    return out


def stage_score(root: Path, say, limit: int = 0) -> dict:
    both = load_both(root)
    saved = pickle.load(open(root / "fits.pkl", "rb"))
    old = pickle.load(open(Path(paths.experiment(*SS.SRC88)) / "fits.pkl", "rb"))
    parts = [part_picks(root, no, both, saved, old) for no in SAMPLES]
    allp = pooled(parts)
    say("SCORE on the CONFIRM parts (human positions only; the pick is the highest sum of group weights among the moves the search holds equal, the search's first move on a tie; 'old' = 78.88's weights as they were, 'ii' = "
        "the groups of the union FIND without the other side's cells, 'A' = with them); positions: " + "; ".join(f"sample {p['no']} {p['P']} in {len(set(p['game'].tolist()))} games (" + ", ".join(f"{b} {int((p['band'] == b).sum())}" for b, _l, _h in BANDS) + ")" for p in parts))
    cond = []
    for part in parts + [allp]:
        label_part = "both parts" if part["no"] == 0 else f"CONFIRM {part['no']}"
        for ctl_in in PICKS:
            kind = "with the controls" if ctl_in else "events only"
            for label, sel in cell_defs(part):
                m = int(sel.sum())
                if m < 5:
                    say(f"SCORE {label_part} {label} ({kind}): {m} positions (too few)")
                    continue
                g = part["game"][sel]
                h = {v: part["hits"][(v, ctl_in)][sel] for v in ("old", "ii", "A")}
                first = part["first"][sel]
                parts_txt = [f"search first {SM.wilson_line(first.sum(), m)}"] + [f"{v} {SM.wilson_line(h[v].sum(), m)}" for v in h]
                pd = []
                for a, b, nm in (("old", None, "old less first"), ("ii", None, "ii less first"), ("A", None, "A less first")):
                    d = SM.paired(h[a], first, g, SEED + 1)
                    pd.append(f"{nm} {100 * d[0]:+.2f} [{100 * d[1]:+.2f}, {100 * d[2]:+.2f}]")
                for a, b in (("ii", "old"), ("A", "old"), ("A", "ii")):
                    d = SM.paired(h[a], h[b], g, SEED + 2)
                    pd.append(f"{a} less {b} {100 * d[0]:+.2f} [{100 * d[1]:+.2f}, {100 * d[2]:+.2f}]")
                    if a == "A" and b == "ii" and not ctl_in and part["no"] != 0 and label in ("few", "mid", "many") and d[1] > 0:
                        cond.append((part["no"], label, d))
                say(f"SCORE {label_part} {label} ({kind}, {m} positions): " + "; ".join(parts_txt) + "; paired (points, 90 pct bootstrap over games): " + "; ".join(pd))
    say("PLAY CONDITION (A less ii, events-only pick, a paired band with the lower end above zero in a cell few, mid or many of either CONFIRM part): "
        + (f"MET in {[(no, b, f'{100 * d[0]:+.2f} [{100 * d[1]:+.2f}, {100 * d[2]:+.2f}]') for no, b, d in cond]}" if cond else "NOT MET in any of the six cells"))
    count_c(parts, say)
    return {"parts": parts, "cond": cond}


# ------------------------------------------------------------------------------------------------------------------------------ C: the count

def count_c(parts: list, say) -> None:
    """PART C (a count): by band and part, the tie sets where at least one tied move leaves a unit of the mover's borne on and unguarded that was not so before, the tie sets where that holds for every tied move, and how
    often the first move, the old pick, the ii pick, the A pick and the move played is such a move (shares of all positions of the band, and of the positions where some but not every tied move does)."""
    for part in parts + [None]:
        use = [part] if part is not None else parts
        name = f"CONFIRM {part['no']}" if part is not None else "both parts"
        flag_first, flag_old, flag_ii, flag_a, flag_pl, some_f, all_f, band, units = [], [], [], [], [], [], [], [], []
        for p in use:
            rows0 = p["rows"]["old"]
            fl = p["left_rows"]
            played = rows0.played
            for i in range(p["P"]):
                s0, n = p["start"][i], p["sizes"][i]
                seg = fl[s0:s0 + n]
                flag_first.append(seg[0])
                flag_old.append(seg[int(p["picks"][("old", False)][i])])
                flag_ii.append(seg[int(p["picks"][("ii", False)][i])])
                flag_a.append(seg[int(p["picks"][("A", False)][i])])
                flag_pl.append(seg[int(np.flatnonzero(played[s0:s0 + n])[0])])
                some_f.append(bool(seg.any()))
                all_f.append(bool(seg.all()))
            band += p["band"].tolist()
        band = np.array(band)
        ff, fo, fi, fa, fp, sf, af = (np.array(x, bool) for x in (flag_first, flag_old, flag_ii, flag_a, flag_pl, some_f, all_f))
        for b in [x[0] for x in BANDS] + ["all"]:
            sel = np.ones(len(band), bool) if b == "all" else band == b
            m = int(sel.sum())
            mix = sel & sf & ~af
            say(f"COUNT {name} {b}: {m} tie positions; tie sets with a tied move that leaves a unit of the mover's borne on and unguarded that was not so before: {int((sel & sf).sum())} ({100 * (sel & sf).sum() / max(1, m):.1f} pct), "
                f"of them every tied move does in {int((sel & af).sum())}, some but not every in {int(mix.sum())}; the move is such a move for the first move {int(ff[sel].sum())}, the old pick {int(fo[sel].sum())}, the ii pick {int(fi[sel].sum())}, "
                f"the A pick {int(fa[sel].sum())}, the move played {int(fp[sel].sum())} (of {m} positions); among the {int(mix.sum())} where some but not every tied move does: first move {int(ff[mix].sum())}, old {int(fo[mix].sum())}, "
                f"ii {int(fi[mix].sum())}, A {int(fa[mix].sum())}, played {int(fp[mix].sum())}")


# ------------------------------------------------------------------------------------------------------------------------------ JUDGE: the thirty decisions

CASES_PER_BAND = 10
CASES_PER_PART = 5
TOP_GROUPS = 3


def cases_rule() -> str:
    return ("THE RULE (fixed before any position was read): in each band (few, mid, many), from each CONFIRM part five decisions, the first five in the part's own order, at most one a game, where the pick of A (fit with the other side's cells, "
            "events only) differs from the pick of the old weights (78.88's, events only); the thirty are printed beside the search's first move, the move played and the moves that leave a unit of the mover's borne on and unguarded.")


def stage_cases(root: Path, say, limit: int = 0) -> None:
    both = load_both(root)
    saved = pickle.load(open(root / "fits.pkl", "rb"))
    old = pickle.load(open(Path(paths.experiment(*SS.SRC88)) / "fits.pkl", "rb"))
    parts = [part_picks(root, no, both, saved, old) for no in SAMPLES]
    fitA = saved["variants"]["A"]["fit"]
    say("CASES: " + cases_rule())
    out = []
    n_diff = {}
    for p in parts:
        rowsA = p["rows"]["A"]
        seen = {b: set() for b, _l, _h in BANDS}
        for i, r in enumerate(p["sample"]):
            b = p["band"][i]
            if p["picks"][("A", False)][i] == p["picks"][("old", False)][i]:
                continue
            n_diff[(p["no"], b)] = n_diff.get((p["no"], b), 0) + 1
            if len([c for c in out if c["band"] == b and c["sample"] == p["no"]]) >= CASES_PER_PART or r["game"] in seen[b]:
                continue
            seen[b].add(r["game"])
            s0 = int(p["start"][i])
            pa, po = int(p["picks"][("A", False)][i]), int(p["picks"][("old", False)][i])
            tops = lambda j, rowsA=rowsA, s0=s0: SGM.carried_top(rowsA, s0 + j, fitA)
            out.append({"id": r["id"], "game": r["game"], "ply": r["ply"], "band": b, "sample": p["no"], "units": r["units"], "moves": r["moves"], "tied": r["tied"], "played": r["played"], "first": r["tied"][0],
                        "old": r["tied"][po], "new": r["tied"][pa], "new_groups": tops(pa), "old_groups": tops(po), "first_groups": tops(0),
                        "left": [int(x) for x in p["left_rows"][s0:s0 + int(p["sizes"][i])]]})
    order = {b: k for k, (b, _l, _h) in enumerate(BANDS)}
    out.sort(key=lambda c: (order[c["band"]], c["sample"]))
    (root / "cases90.json").write_text(json.dumps({"variant": "A", "groups_file": "groups_A.json", "cases": out}))
    say(f"CASES written: {len(out)} decisions, " + ", ".join(f"{b} {sum(1 for c in out if c['band'] == b)}" for b, _l, _h in BANDS)
        + "; positions where the pick of A differs from the old weights' (sample 1 / sample 2): " + ", ".join(f"{b} {n_diff.get((1, b), 0)} / {n_diff.get((2, b), 0)}" for b, _l, _h in BANDS)
        + f"; the A pick equals the move played in {sum(1 for c in out if c['new'] == c['played'])} of them, the old pick in {sum(1 for c in out if c['old'] == c['played'])}, the first move in {sum(1 for c in out if c['first'] == c['played'])}")


# ------------------------------------------------------------------------------------------------------------------------------ PLAY (only on the condition of `score`)

def stage_agreement(root: Path, say, n: int = 200) -> None:
    """The player's pick (symbols_own_play.OwnReader, one position at a time) against the offline pick of A for the same tie set, on `n` positions evenly spaced in each CONFIRM part."""
    from catspace.incubator.analyst.discovery import symbols_own_play as OP
    both = load_both(root)
    saved = pickle.load(open(root / "fits.pkl", "rb"))
    old = pickle.load(open(Path(paths.experiment(*SS.SRC88)) / "fits.pkl", "rb"))
    rd = OP.own_reader()
    for no in SAMPLES:
        p = part_picks(root, no, both, saved, old)
        chosen = np.unique(np.linspace(0, p["P"] - 1, n).astype(int))
        ok = sum(1 for i in chosen if rd.pick(p["sample"][i]["moves"], p["sample"][i]["tied"])[0] == p["sample"][i]["tied"][int(p["picks"][("A", False)][i])])
        say(f"AGREEMENT the player's pick against the offline pick of A for the same tie set, CONFIRM {no}: {ok} of {len(chosen)} positions (evenly spaced)")


def match_specs90(a: str, b: str, set_name: str, pairs: int, seconds: float, root: Path) -> list:
    """78.89's match specs (tiebreak_measure.match_specs, both colours, the ply cap and the ledger share of 78.67); the player `ownfit` is the full reasons player with the spec's fit set to "own"."""
    from catspace.incubator.analyst.discovery import symbols_play as SP
    kinds = dict(SP.KIND, ownfit="sy_reasons")
    specs = TM.match_specs(a, b, set_name, pairs, seconds, root / "scratch", False, kinds)
    for s in specs:
        s["players"][s["colour"]]["fit"] = "own"                      # a is the first player of the pair: it sits at index `colour` of the spec's list
    return specs


def games_file(root: Path, a: str, b: str, set_name: str) -> Path:
    return root / "matches" / f"{a}_v_{b}_{set_name}.jsonl"


def stage_match(root: Path, a: str, b: str, set_name: str, pairs: int, workers: int, seconds: float, say) -> None:
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import symbols_own_play as OP
    from catspace.incubator.analyst.discovery import symbols_play as SP
    load, busy = TM.wait_quiet(lambda s: TM.emit([s], root / "obs.txt"))
    t0 = time.perf_counter()
    gf = games_file(root, a, b, set_name)
    gf.parent.mkdir(parents=True, exist_ok=True)
    games = TM.run_games(match_specs90(a, b, set_name, pairs, seconds, root), workers, gf, lambda s: TM.emit([s], root / "obs.txt"), play=OP.play90)
    wall = time.perf_counter() - t0
    gf.with_suffix(".meta.json").write_text(json.dumps({"load": load, "busy": busy, "wall": wall, "workers": workers, "seconds": seconds}))
    TM.emit(SP.lines_of(a, b, set_name, games, load, busy, wall, seconds, workers, False), root / "obs.txt")
    TM.emit(SP.decision_lines(a, b, set_name, games, GL.model(HD.GAME, HD.VARIANT)), root / "obs.txt")


def stage_playreport(root: Path) -> None:
    from catspace.incubator.analyst.discovery import symbols_play as SP
    found: dict = {}
    for p in sorted((root / "matches").glob("*.jsonl")):
        x, rest = p.stem.split("_v_")
        b, s = rest.rsplit("_", 1)
        found.setdefault((x, b), {})[s] = TM.read_games(p)
    for (x, b), sets in sorted(found.items()):
        if len(sets) > 1:
            TM.emit(SP.pooled_lines(x, b, sets, False), root / "obs.txt")


# ------------------------------------------------------------------------------------------------------------------------------ main

def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("cells", "fit", "report", "score", "cases", "agreement", "match", "playreport"))
    ap.add_argument("--b", default="control")
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    ap.add_argument("--tag", default="short")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    say = Say(root, a.stage)
    t0 = time.perf_counter()
    say(f"stage start; tag {a.tag}; limit {a.limit}; load average {os.getloadavg()[0]:.2f}")
    if a.stage == "cells":
        stage_cells(root, say, a.workers, a.limit)
    elif a.stage == "fit":
        stage_fit(root, say, a.limit)
    elif a.stage == "report":
        stage_report(root, say, a.limit)
    elif a.stage == "score":
        stage_score(root, say, a.limit)
    elif a.stage == "cases":
        stage_cases(root, say, a.limit)
    elif a.stage == "agreement":
        stage_agreement(root, say)
    elif a.stage == "match":
        stage_match(root, "ownfit", a.b, a.set, a.pairs, a.workers, a.seconds, say)
    elif a.stage == "playreport":
        stage_playreport(root)
    say(f"stage done in {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
