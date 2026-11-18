"""Plan 78.89: the many band split, and a sample that covers the later game (catspace/incubator/analyst/discovery/symbols_sample_measure.py, symbols_measure.plies_stratified, symbols.Choice.fill_ok). SYSTEM_SPEC 7, 9, 10.

The stratified draw takes plies evenly over bins of unit count, at most a fixed number a game in a bin; the cuts of the first print come from counts alone; a generated row is refused by the fit unless the rows' builder says
fill_ok, and is never of the CONFIRM part; the second sample leaves out every game of the first and the ladder's; the window of a match player is set on the reasons player only; the measure's source names no game object."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from catspace.incubator.analyst.discovery import symbols as SY
from catspace.incubator.analyst.discovery import symbols_measure as SM
from catspace.incubator.analyst.discovery import symbols_sample_measure as S

HAVE = (S.src80() / "sample.jsonl").exists() and (S.SGM.src81() / "reasons.json").exists() and (S.src88() / "fits.pkl").exists()
needs_run = pytest.mark.skipif(not HAVE, reason="needs the 78.80, 78.81 and 78.88 runs' files")


def test_the_stratified_draw_is_evenly_over_bins_and_capped_per_game_and_bin():
    counts = [32] * 6 + [30 - p // 2 for p in range(60)]            # a game that falls from 30 units to 1
    bins = ((0, 4), (5, 8), (9, 12), (13, 16), (17, 28))
    got = SM.plies_stratified(7, counts, bins, 3, len(counts) - 1)
    assert set(got) <= set(range(len(bins)))
    for bi, plies in got.items():
        assert 1 <= len(plies) <= 3
        assert len(set(plies)) == len(plies)
        assert all(SM.MIN_PLY <= p < len(counts) and bins[bi][0] <= counts[p] <= bins[bi][1] for p in plies)
    assert got == SM.plies_stratified(7, counts, bins, 3, len(counts) - 1)          # seeded by the ordinal: the same draw again
    assert got != SM.plies_stratified(8, counts, bins, 3, len(counts) - 1)
    assert SM.plies_stratified(7, [32] * 40, bins, 3, 39) == {}                      # a game that never leaves the opening count has no position in these bins
    short = SM.plies_stratified(7, [10] * 12 + [3, 3], bins, 3, 13)
    assert len(short[0]) == 2 and len(short[2]) == 2 and set(short) == {0, 2}


def test_the_cuts_of_the_first_print_come_from_counts_alone():
    sample = [{"part": "confirm", "units": 20 + i % 13, "ply": 10 + i, "game": i} for i in range(400)] + [{"part": "find", "units": 17 + i % 16, "ply": 3, "game": 1000 + i} for i in range(300)] + \
             [{"part": "confirm", "units": 5, "ply": 99, "game": 5000}]
    c = S.scale_cuts(sample)
    assert c["ply_edges"] == sorted(c["ply_edges"]) and len(c["ply_edges"]) == 3
    plies = np.array([r["ply"] for r in sample if r["part"] == "confirm" and r["units"] >= 17])
    below = [(plies < e).sum() for e in c["ply_edges"]]
    assert abs(below[0] - len(plies) / 4) <= 1 and abs(below[1] - len(plies) / 2) <= 1 and abs(below[2] - 3 * len(plies) / 4) <= 1
    units = np.array([r["units"] for r in sample if r["units"] >= 17])
    assert c["n_all"] == len(units) and c["unit_edges"][0] <= c["unit_edges"][1]
    assert abs((units <= c["unit_edges"][0]).mean() - 1 / 3) < 0.1


def test_the_bins_index_by_the_edges():
    assert S._bin_index(np.array([1, 5, 9, 20]), [5, 9]).tolist() == [0, 1, 2, 2]


def choice(source, part="find"):
    n = 12
    E = sparse.csr_matrix(np.tile([[1, 0], [0, 1], [0, 0]], (4, 1)).astype(float))
    pos = np.repeat(np.arange(4), 3)
    played = np.array([True, False, False] * 4)
    return SY.Choice(E, np.zeros((n, 1)), ["takes"], pos, played, pos.copy(), np.array([part] * n), np.array(source), None)


def test_a_generated_row_is_refused_by_the_fit_unless_the_builder_says_so():
    rows = choice(["human"] * 6 + ["generated"] * 6)
    with pytest.raises(ValueError):
        SY.require_find(rows)
    rows.fill_ok = True
    SY.require_find(rows)                                                              # now read
    assert rows.subset(np.array([True, False, True, False])).fill_ok
    with pytest.raises(ValueError):
        SY.require_find(choice(["human"] * 12, part="confirm"))                        # a CONFIRM row is refused whatever the flag says
    c = choice(["human"] * 6 + ["generated"] * 6, part="confirm")
    c.fill_ok = True
    with pytest.raises(ValueError):
        SY.require_find(c)


def test_a_kept_generated_position_is_of_the_find_part_whatever_the_hash_says():
    rows = [{"game": g, "ply": 12, "ok": True, "n_eq": 3, "tie": True, "tied": ["x", "y"], "move": "x", "played": "y"} for g in range(40)]
    meta = {(g, 12): {"units": 9, "bin": "9-12", "origin": "generated"} for g in range(40)}
    games = {g: {"moves": ["m"] * 30} for g in range(40)}
    kept, tries = S._keep(rows, meta, games, "generated", "find")
    assert {r["part"] for r in kept} == {"find"} and {r["source"] for r in kept} == {"generated"} and len(kept) == 40
    kept_h, _t = S._keep(rows, meta, games, "human", None)
    assert {r["part"] for r in kept_h} == {"find", "confirm"}                          # a human game's part is the 78.80 hash of its ordinal
    assert all(r["part"] == SM.part_of(r["game"]) for r in kept_h)


def test_the_window_is_set_on_the_reasons_player_only(monkeypatch):
    import types
    cut = {"cut": 28, "lower": [17, 28], "upper": [29, 99]}
    assert S.window_of("manylo", cut) == [17, 28] and S.window_of("manyhi", cut) == [29, 99]
    assert S.KIND89["manylo"] == S.KIND89["manyhi"] == "sy_reasons"
    fake = lambda a, b, s, pairs, sec, scratch, charge, kinds: [{"players": [{"kind": kinds[a]}, {"kind": "tb_control"}]}]
    monkeypatch.setattr(S.TM, "match_specs", fake)
    specs = S.match_specs89("manylo", "A", 1, 0.2, Path("."), cut)
    assert specs[0]["players"][0]["window"] == [17, 28] and "window" not in specs[0]["players"][1]
    assert S.bins_of(cut)[-1] == (17, 28) and S.bins_of(cut)[0] == (0, 4)


@needs_run
def test_the_second_sample_leaves_out_every_game_of_the_first_and_of_the_ladder():
    left = S.held_left()
    used = {r["game"] for r in SM.jsonl(S.src80() / "sample.jsonl")}
    assert left and not ({g["ord"] for g in left} & used)
    assert not ({g["ord"] for g in left} & SM.ladder_ordinals())
    assert all(not g["fit"] and not g["arrival"] for g in left)


@needs_run
def test_the_cases_rule_never_takes_the_first_decision_of_a_game():
    root = S.out_root("full")
    f = root / "cases89.json"
    if not f.exists():
        pytest.skip("the cases stage was not run")
    cases = json.loads(f.read_text())
    cut = S.load_cut(root)
    assert S.CASE_NTH == 3 and len(cases) == 2 * S.CASES_PER_PART
    for c in cases:
        assert c["decisions_in_part"] >= S.CASE_NTH and c["pick"] != c["first"]
        lo, hi = (cut["lower"] if c["band"] == "lower many" else (9, 16))
        assert lo <= c["units"] <= hi
    assert len({c["gid"] for c in cases if c["band"] == "mid"}) == S.CASES_PER_PART


def test_the_measure_names_no_object_of_any_game():
    for f in ("symbols_sample_measure.py",):
        text = (Path(S.__file__).parent / f).read_text().lower()
        assert not re.search(r"\b[a-h][1-8]\b", text), f
        for w in ("fen", "san", "uci", "check", "mate", "promotion", "pawn", "bishop", "knight", "rook", "queen"):
            assert not re.search(rf"\b{w}\b", text), (f, w)


def test_no_player_id_is_written_by_the_sample():
    text = (Path(S.__file__).parent / "symbols_sample_measure.py").read_text()
    assert "game_id" not in text
    assert 'g["id"]' not in text                                                         # the source's game number is dropped, only the ordinal of this run is kept
