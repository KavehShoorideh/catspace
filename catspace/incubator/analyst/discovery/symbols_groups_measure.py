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
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

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


# ------------------------------------------------------------------------------------------------------------------------------ main

def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("first", "groups", "split", "fit", "score", "cases", "play"))
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
    say(f"stage done in {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
