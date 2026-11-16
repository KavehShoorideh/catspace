"""Plan 78.69: the two readings and what their difference tells (catspace/incubator/analyst/discovery/two_readings.py). Synthetic rows only; the real rows are measured by the module's commands."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import two_readings as TR

CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"


def test_the_module_names_no_game_object_and_no_relative_data_path():
    src = Path(TR.__file__).read_text()
    hit = re.search(CHESS_WORDS, src, flags=re.I)
    assert hit is None, hit.group(0)
    assert not re.search(r"[\"']data/", src)
    assert "sys.path" not in src
    assert "experiments." not in src                                  # no import from an experiments directory


def test_the_module_writes_the_interface_before_the_body():
    doc = TR.__doc__
    for word in ("INTERFACE", "takes", "returns", "Called by", "THE STAKE"):
        assert word in doc


def test_bands_have_equal_counts_and_strata_bands_hold_the_stratum_fixed():
    rng = np.random.default_rng(0)
    v = rng.normal(size=1000)
    b = TR.quantile_bands(v)
    assert [int((b == k).sum()) for k in range(TR.BANDS)] == [200] * 5
    assert v[b == 0].max() <= v[b == 1].min()
    strat = np.repeat([0, 1], 500)
    v2 = v + 100.0 * strat                                            # the stratum shifts everything; bands made inside each stratum still have equal counts in each
    sb = TR.strata_bands(v2, strat)
    for s in (0, 1):
        assert [int(((sb == k) & (strat == s)).sum()) for k in range(TR.BANDS)] == [100] * 5


def test_the_bootstrap_resamples_games_not_rows():
    # two games; all the rows of a game carry the same value: a resample of rows would give a narrow band, a resample of games a wide one
    game = np.repeat([0, 1], 200)
    num = np.where(game == 0, 1.0, 0.0)
    den = np.ones(400)
    gb = TR.GameBoot(game, draws=400, seed=1)
    pt, dr = gb.ratio(num, den, np.zeros(400, np.int64), 1)
    assert abs(pt[0] - 0.5) < 1e-9
    lo, hi = TR.band(pt[0], dr[:, 0])
    assert lo == 0.0 and hi == 1.0                                    # with two games the draws are 0, 0.5 or 1
    assert set(np.round(np.unique(dr[:, 0]), 6)) <= {0.0, 0.5, 1.0}


def test_the_ratio_of_sums_is_per_group():
    game = np.arange(100) // 5
    grp = np.arange(100) % 2
    num = np.where(grp == 1, 2.0, 1.0)
    pt, _ = TR.GameBoot(game, draws=10).ratio(num, np.ones(100), grp, 2)
    assert np.allclose(pt, [1.0, 2.0])


def test_a_ratio_with_the_stratum_held_fixed_is_one_when_only_the_stratum_matters():
    rng = np.random.default_rng(3)
    n = 6000
    strat = rng.integers(0, 5, n)
    # the label is more common in the high strata, and the event rate depends on the stratum only
    grp = np.where(rng.random(n) < 0.1 + 0.15 * strat, 2, 0)
    rate = 0.05 + 0.05 * strat
    obs = (rng.random(n) < rate).astype(float)
    expo = np.ones(n)
    game = np.arange(n) // 3
    gb = TR.GameBoot(game, draws=50, seed=2)
    raw, _ = gb.ratio(obs, expo, grp // 2, 2)
    assert raw[1] / raw[0] > 1.3                                      # unmatched: the stratum's effect looks like the label's
    matched, dm = TR.stratified_ratio(gb, obs, expo, grp, strat, 5)
    assert 0.85 < matched < 1.15
    assert dm.shape == (50,)
    shuffled, _ = TR.stratified_ratio(gb, obs, expo, grp, strat, 5, np.random.default_rng(1))
    assert 0.85 < shuffled < 1.15


def test_a_label_that_matters_inside_every_stratum_stays_above_one():
    rng = np.random.default_rng(4)
    n = 8000
    strat = rng.integers(0, 5, n)
    grp = np.where(rng.random(n) < 0.3, 2, 0)
    obs = (rng.random(n) < np.where(grp == 2, 0.20, 0.10)).astype(float)
    gb = TR.GameBoot(np.arange(n) // 4, draws=50, seed=2)
    matched, _ = TR.stratified_ratio(gb, obs, np.ones(n), grp, strat, 5)
    assert matched > 1.5


def test_event_times_count_the_events_step_and_leave_the_end_of_the_game_out():
    res = [{"n": 10, "events": {("taken", "a"): [3, 7], ("went", "b"): [5], (TR.END, "a"): [9]}}]
    rows = {"g": np.array([0, 0, 0, 0]), "ply": np.array([0, 3, 4, 8])}
    d = {"rows": rows, "sample": {"res": res}}
    te = np.arange(4)
    expo, obs, nxt = TR.event_times(d, te, None)
    assert list(nxt[:3]) == [3, 3, 5]
    assert list(obs) == [True, True, True, False]                      # from ply 8 only the end of the game is left, which is not an event here
    assert list(expo) == [4, 1, 2, 2]                                  # the event's own step counts; with none, the steps to the end of the record
    expo2, obs2, _ = TR.event_times(d, te, "taken")
    assert list(obs2) == [True, True, True, False] and list(expo2) == [4, 1, 4, 2]


def test_the_held_out_rows_are_those_the_split_did_not_fit():
    rows = {"g": np.repeat(np.arange(6), 3), "fit": np.repeat(np.array([True, True, False, True, False, False]), 3)}
    d = {"rows": rows}
    te = TR.test_rows(d, None)
    assert not rows["fit"][te].any()
    assert set(rows["g"][te]) == {2, 4, 5}
    assert set(rows["g"][TR.test_rows(d, 2)]) == {2, 4}


def test_next_rows_stay_inside_the_game():
    d = {"rows": {"g": np.array([0, 0, 1, 1, 1])}}
    te = np.arange(5)
    assert list(TR.next_rows(d, te)) == [1, -1, 3, 4, -1]
    assert list(TR.next_rows(d, te, 2)) == [-1, -1, 4, -1, -1]


def test_claims_are_counted_by_owner_and_kind_and_only_in_the_named_classes():
    from catspace.incubator.analyst.discovery import tactical_state as TS
    lost, safe, contested = (TS.CLASS_WORDS.index(w) for w in ("lost", "safe", "contested"))
    units = np.array([[1, 0, lost, 0, 0], [0, 1, lost, 0, 0], [1, 1, safe, 0, 0], [0, 0, contested, 255, 0], [1, 0, lost, 2, 0]], np.uint8)
    st = {"units": units, "uoff": np.array([0, 3, 5], np.int64)}
    d = {"st": st, "ix": np.array([0, 1]), "state_kinds": ["x", "y"]}
    ca = TR.claim_arrays(d, np.array([0, 1]), ("lost",))
    assert ca["owner_kind"][0, 1, 0] == 1 and ca["owner_kind"][0, 0, 1] == 1 and ca["owner_kind"][0].sum() == 2
    assert ca["owner_kind"][1, 1, 0] == 1 and ca["owner_kind"][1].sum() == 1
    both = TR.claim_arrays(d, np.array([0, 1]), ("lost", "contested"))
    assert both["owner_kind"][1].sum() == 2
