"""Plan 78.83: the window of the reasons' hint, in units on the board (catspace/incubator/analyst/discovery/symbols_play.py). SYSTEM_SPEC 7 (the hint among equal moves), 8, 9.

The window kinds name the bands of ladder_measure; outside the window a tie decision is the search's first move with no pick (as the control plays it), inside it the pick of 78.82; the unit count and the kept state
agree with the layer-1 table read from the first position; a short match with a window player logs picks inside the window only."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

import cs_engine as CE

from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import symbols_play as SP
from catspace.incubator.analyst.discovery import tiebreak_measure as TM

HAVE = (SP.src_root() / "reasons.json").exists() and (SP.src_root() / "fits.pkl").exists()
needs_run = pytest.mark.skipif(not HAVE, reason="needs the 78.81 run's reasons.json and fits.pkl")


class StubReader:
    def __init__(self, units: int):
        self.units, self.picked = units, 0

    def units_of(self, moves):
        return self.units

    def pick(self, moves, tied, mode):
        self.picked += 1
        return tied[-1], {"scores": {m: 0.0 for m in tied}, "carried": {m: [] for m in tied}, "takes": None, "kinds": None}


def mover(kind: str, monkeypatch, units: int) -> "tuple[SP.MoverReasons, StubReader]":
    rd = StubReader(units)
    monkeypatch.setattr(SP, "_READER", rd)
    monkeypatch.setattr(SP.CE, "solve_equals_batch", lambda *a, **k: ["row"])
    monkeypatch.setattr(SP.PM, "equal_set", lambda row: {"move": "m0", "equals": ["m0", "m1", "m2"], "depth": [1, 1]})
    return SP.MoverReasons({"kind": kind, "worth": {}}, 0.01, "g", "/nonexistent/trace"), rd


def test_the_window_kinds_are_the_bands_of_the_ladder_and_the_default_is_none():
    band = {n: (lo, hi) for n, lo, hi in LM.BANDS}
    assert SP.WINDOW[SP.KIND["many"]] == band["many"] and SP.WINDOW[SP.KIND["mid"]] == band["mid"] and SP.WINDOW[SP.KIND["few"]] == band["few"]
    assert SP.WINDOW[SP.KIND["midfew"]] == (band["few"][0], band["mid"][1])
    assert set(SP.WINDOW) <= set(SP.MODE) and all(SP.MODE[k] == "reasons" for k in SP.WINDOW)
    for p in ("many", "mid", "few", "midfew"):
        assert p in SP.PLAYERS and p in SP.MODE_PLAYERS
    for p in ("reasons", "quiet", "withctl"):
        assert SP.MoverReasons({"kind": SP.KIND[p], "worth": {}}, 0.01, "g", "/x").window is None
    assert SP.MoverReasons({"kind": SP.KIND["reasons"], "worth": {}, "window": [3, 5]}, 0.01, "g", "/x").window == (3, 5)


@pytest.mark.parametrize("kind,units,inside", [("many", 17, True), ("many", 16, False), ("mid", 9, True), ("mid", 16, True), ("mid", 17, False), ("mid", 8, False), ("few", 0, True), ("few", 8, True), ("few", 9, False),
                                               ("midfew", 3, True), ("midfew", 16, True), ("midfew", 17, False), ("reasons", 500, True)])
def test_inside_the_window_the_pick_is_used_and_outside_it_the_search_s_first_move_is_played(monkeypatch, kind, units, inside):
    m, rd = mover(SP.KIND[kind], monkeypatch, units)
    pick, _depth = m.choose(None, ["x"], [])
    assert m.in_window(["x"]) == inside
    assert (pick, rd.picked, m.stats["ties_in"]) == (("m2", 1, 1) if inside else ("m0", 0, 0))
    assert m.stats["ties"] == 1 and m.stats["decisions"] == 1                 # a tie is counted whether or not the hint reads it
    assert len(m.stats["dlog"]) == m.stats["ties_in"] == len(m.stats["cases"])
    assert m.stats["changed"] == (1 if inside else 0)


def random_game(n: int, seed: int) -> list:
    rng, moves = random.Random(seed), []
    for _ in range(n):
        legal = CE.state(HD.VARIANT, None, moves)[2]
        if not legal:
            break
        moves.append(rng.choice(sorted(legal)))
    return moves


@needs_run
def test_the_kept_state_and_the_unit_count_equal_the_reading_from_the_first_position():
    rd = SP.ReasonsReader()
    game = random_game(40, 3)
    for n in (6, 7, 9, 20, 21, 40, 12, 5):                                    # growing, then a position that does not extend the kept one
        assert rd.state_of(game[:n]) == CE.state(HD.VARIANT, None, game[:n])[0]
        assert rd.units_of(game[:n]) == LM.unit_counts(rd.counter.model, game[:n])[n]


@needs_run
def test_a_two_pair_match_with_a_window_player_picks_only_inside_the_window(tmp_path):
    specs = TM.match_specs("many", "control", "A", 2, 0.05, tmp_path / "scratch", False, SP.KIND)
    for s in specs:
        s["max_plies"] = 40
    games = TM.run_games(specs, 1, tmp_path / "g.jsonl", lambda s: None, play=SP.play82)
    rd = SP.reader82()
    n_in = n_all = 0
    for g in games:
        st = g["stats"][TM.seat_of(g, 0)]
        n_in += st["ties_in"]
        n_all += st["ties"]
        assert st["ties_in"] == len(st["dlog"]) <= st["ties"]
        for c in st["cases"]:
            assert rd.units_of(c["moves"]) >= 17 and c["pick"] in c["tied"] and c["pick"] != c["first"]
    assert n_all >= n_in >= 0
    assert all(g["stats"][TM.seat_of(g, 1)]["changed"] == 0 for g in games)


def test_the_match_stage_prints_the_load_it_saw(tmp_path):
    src = (Path(SP.__file__)).read_text()
    assert ".meta.json" in src and 'meta.get("load"' in src


def test_the_thirty_rule_takes_the_first_decision_per_game_in_each_band_and_ten_games_a_band(monkeypatch):
    counts = {4: 30, 9: 30, 20: 12, 30: 5, 40: 5}
    monkeypatch.setattr(SP.LM, "unit_counts", lambda layer1, moves: [counts.get(i, 20) for i in range(len(moves) + 1)])
    games = []
    for k in range(14):
        cs = [{"ply": p, "moves": ["x"] * p, "tied": ["a", "b"], "first": "a", "pick": "b"} for p in (30, 4, 9, 20, 30, 40)]
        games.append({"gid": f"g{13 - k:02d}", "moves": ["x"] * 45, "colour": 0, "stats": [{"cases": cs}, {"cases": []}]})
    games.append({"gid": "g99", "moves": ["x"] * 45, "colour": 0, "stats": [{"cases": []}, {"cases": []}]})
    out = SP.cases_by_band(games, "reasons", None)
    assert len(out) == 30 and [c["band"] for c in out] == ["few"] * 10 + ["mid"] * 10 + ["many"] * 10
    assert [c["gid"] for c in out[:10]] == [f"g{k:02d}" for k in range(10)]           # games in the order of their ids
    for band, ply in (("few", 30), ("mid", 20), ("many", 4)):                         # the first decision of the game in that band
        assert all(c["ply"] == ply for c in out if c["band"] == band)
