"""Plan 78.88: twins grouped, the unit borne on, weights by band (catspace/incubator/analyst/discovery/symbols.py and symbols_groups_measure.py). SYSTEM_SPEC 7, 9.

The grouping merges reasons whose carrier sets are alike and only those of one side, direction and sign of weight; a group is the OR of its members; the split replaces a reason of the form 'a unit of the mover's bears on a unit
of theirs' by four cells and leaves every other reason alone; the cell counts agree with the engine's counts in total; a fit by band reads the positions of that band only and finds a weight planted in that band; the mixture
picks with the fit of the position's band; the measure's sources name no object of any game."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from catspace.incubator.analyst.discovery import graph_layer2 as L2
from catspace.incubator.analyst.discovery import symbols as SY
from catspace.incubator.analyst.discovery import symbols_groups_measure as G

HAVE = (G.src81() / "reasons.json").exists() and (G.src81() / "controls.npz").exists()
needs_run = pytest.mark.skipif(not HAVE, reason="needs the 78.81 run's files")


def choice(E, pos, played, units=None, game=None):
    n = len(pos)
    return SY.Choice(sparse.csr_matrix(E), np.zeros((n, 1)), ["takes"], np.asarray(pos), np.asarray(played, bool), np.asarray(game if game is not None else np.asarray(pos) // 2), np.array(["find"] * n), np.array(["human"] * n),
                     None if units is None else np.asarray(units))


def test_twins_merge_and_only_within_one_side_direction_and_sign():
    # columns 0 and 1 fire together; column 2 fires together with them but is of the other side; column 3 of the other sign; column 4 is alone
    E = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 0], [0, 1, 0, 0, 1]] * 4, float)
    rows = choice(E, np.arange(len(E)), [True] * len(E))
    reasons = [{"side": "mover", "dir": "up", "weight": 0.5}, {"side": "mover", "dir": "up", "weight": 0.2}, {"side": "other", "dir": "up", "weight": 0.3}, {"side": "mover", "dir": "up", "weight": -0.3},
               {"side": "mover", "dir": "up", "weight": 0.1}]
    assert SY.groups_of(reasons, rows) == [[0, 1], [2], [3], [4]]


def test_the_threshold_is_the_one_fixed_value_and_dissimilar_columns_stay_apart():
    assert SY.GROUP_SIM == 0.5
    E = np.zeros((20, 3))
    E[:10, 0] = 1
    E[5:15, 1] = 1                      # Jaccard 5 / 15, below one half
    E[10:20, 2] = 1
    rows = choice(E, np.arange(20), [True] * 20)
    reasons = [{"side": "mover", "dir": "up", "weight": 1.0}] * 3
    assert SY.groups_of(reasons, rows) == [[0], [1], [2]]


def test_groups_of_reads_find_rows_only():
    E = np.ones((4, 2))
    rows = choice(E, np.arange(4), [True] * 4)
    rows.part = np.array(["find", "find", "confirm", "find"])
    with pytest.raises(ValueError):
        SY.groups_of([{"side": "mover", "dir": "up", "weight": 1.0}] * 2, rows)


def test_a_group_is_true_when_any_member_fires():
    E = sparse.csr_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], float))
    M = SY.group_matrix(E, [[0, 1], [2]]).toarray()
    assert M.tolist() == [[1, 0], [1, 0], [0, 1], [0, 0]]


def planted_pattern() -> L2.Pattern:
    return L2.Pattern(0, [("bear", 0, "u", 1, None, None)], [True, False])


def unit(side: str, mine: bool) -> dict:
    return dict(mine=mine, side=side, goal=False, edge=False, kind="k", place="p")


def test_the_cells_count_guarded_and_borne_on_by_the_graph_s_own_bears():
    units = {"a": unit("x", True), "b": unit("y", False), "c": unit("y", False), "d": unit("y", False), "e": unit("x", True), "f": unit("y", False)}
    bear = lambda u, t: ("bear", u, ("u", t), None, None)
    items = [bear("a", "b"), bear("c", "b"), bear("d", "a"), bear("e", "f")]
    c0, c1 = SY.cell_counts(items, units, planted_pattern(), [])
    assert c0 == [1, 0, 0, 1]                      # (e, f): f loose, e free: cell 0; (a, b): b guarded by c, a borne on by d: cell 3
    assert c1 == [1, 0, 0, 0]                      # read for the other side: the one bear of y onto x is (d, a); a is not guarded, d is not borne on
    assert SY.cell_events([1, 0, 0, 1], [2, 0, 0, 0]) == ([0], [3])


def test_the_pattern_form_and_the_split():
    assert SY.bear_form(planted_pattern())
    assert not SY.bear_form(L2.Pattern(0, [("bear", 0, "u", 1, None, None)], [True, True]))
    assert not SY.bear_form(L2.Pattern(0, [("move", 0, "u", 1, None, None)], [True, False]))
    raw = lambda p: bytes([p.level, 1, 0, 0, 0, 1, 255, 255, 0, 1]).hex()
    bear_reason = {"side": "mover", "dir": "up", "weight": 0.7, "bytes": raw(planted_pattern()), "words": "w"}
    other = dict(bear_reason, side="other")
    out = SY.split_borne([bear_reason, other], [])
    assert [(r["parent"], r["cell"]) for r in out] == [(0, 0), (0, 1), (0, 2), (0, 3), (1, None)]
    assert all(r["weight"] == 0.7 for r in out[:4])


@needs_run
def test_the_split_of_78_81_s_reasons_names_three_patterns_and_keeps_the_rest():
    reasons = json.loads((G.src81() / "reasons.json").read_text())
    keys, raws = G.split_patterns(reasons)
    out = SY.split_borne(reasons, [])
    assert len(keys) == len(set(keys))
    n_split = sum(1 for r in reasons if r["side"] == "mover" and int(r["key"]) in keys)
    assert len(out) == len(reasons) - n_split + 4 * n_split
    assert sum(1 for r in out if r["cell"] is None) == len(reasons) - n_split


@needs_run
def test_the_cell_counts_agree_in_total_with_the_engine_s_event_on_real_positions():
    import cs_engine as CE
    data = G.load_all(40)
    keys, raws = G.split_patterns(data["reasons"])
    env = G._split_env(raws)
    ev = [r for r in data["reasons"] if r["side"] == "mover" and int(r["key"]) == keys[0] and r["dir"] == "up"][0]["event"]
    M = data["inp"]["M"].tocsc()
    row, bad = 0, 0
    for r in data["inp"]["sample"][:12]:
        text = env["env"].text(r["moves"])
        items, units = env["env"].items(text)
        before = SY.cell_counts(items, units, env["pats"][0], env["env"].kinds)[0]
        for m in r["tied"]:
            i2, u2 = env["env"].items(CE.state("chess", text, [m])[0])
            after = SY.cell_counts(i2, u2, env["pats"][0], env["env"].kinds)[1]
            bad += int((sum(after) > sum(before)) != bool(M[row, ev]))
            row += 1
    assert row > 0 and bad == 0


def test_a_fit_by_band_reads_that_band_only_and_finds_the_weight_planted_there():
    rng = np.random.default_rng(0)
    pos, played, E, units = [], [], [], []
    for p in range(240):
        low = p < 120
        w = rng.integers(0, 3)
        for t in range(3):
            pos.append(p)
            played.append(t == w)
            f0 = 1.0 if (low and t == w) else float(rng.random() < 0.3)         # in the low band the played move carries feature 0
            f1 = 1.0 if ((not low) and t == w) else float(rng.random() < 0.3)   # in the high band it carries feature 1
            E.append([f0, f1])
            units.append(4 if low else 30)
    rows = choice(E, pos, played, units, game=np.array(pos) // 2)
    f_lo = SY.fit_choice(rows, controls=(), events=None, band=(0, 8))
    f_hi = SY.fit_choice(rows, controls=(), events=None, band=(17, 99))
    assert f_lo["positions"] == 120 and f_hi["positions"] == 120
    wl, wh = dict(zip(f_lo["events"], f_lo["weights"])), dict(zip(f_hi["events"], f_hi["weights"]))
    assert wl.get(0, 0) > 0.5 and wh.get(1, 0) > 0.5
    assert wl.get(1, 0) < wl.get(0, 0) and wh.get(0, 0) < wh.get(1, 0)


def test_the_mixture_picks_with_the_fit_of_the_position_s_band():
    # two positions, three tied moves each; the low-band fit prefers the feature of move 1, the high-band fit the feature of move 2
    E = np.array([[0, 0], [1, 0], [0, 1], [0, 0], [1, 0], [0, 1]], float)
    rows = choice(E, [0, 0, 0, 1, 1, 1], [True, False, False, False, False, True], units=[5, 5, 5, 30, 30, 30])
    fits = {"few": {"events": [0], "weights": [1.0], "controls": {}}, "many": {"events": [1], "weights": [1.0], "controls": {}}}
    bands = [("few", 0, 8), ("mid", 9, 16), ("many", 17, 99)]
    assert SY.score_choice_by_band(rows, fits, bands).tolist() == [1, 2]          # a band with no fit (mid) would be silent: not used here
    assert SY.score_choice_by_band(rows, {"few": fits["few"]}, bands).tolist() == [1, 0]


def test_the_measure_and_the_symbols_name_no_object_of_any_game():
    import re
    from pathlib import Path
    for f in ("symbols.py", "symbols_groups_measure.py"):
        text = (Path(G.__file__).parent / f).read_text().lower()
        assert not re.search(r"\b[a-h][1-8]\b", text), f
        for w in ("fen", "san", "uci", "check", "mate", "promotion", "pawn", "bishop", "knight", "rook", "queen"):
            if f == "symbols.py":
                continue                      # symbols.py is older and has its own scan in tests/test_plan78.py
            assert not re.search(rf"\b{w}\b", text), (f, w)
