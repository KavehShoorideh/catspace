"""PLAN 79.13 A CHANCE OF EACH GOAL COUNTED FROM PLAY, AS THE CHOOSER'S INPUT (SYSTEM_SPEC 7 "Calculation and estimate, decided together" point 2 the estimate per goal, 6, 8, 9 sleep; piece: the estimate per goal).
Rust holds the unit tests of the table, the key, the value of the chains, their reading after each move and the gain rule (cargo test --release --offline -p cs_decision --test goal_gain). THIS file holds what belongs to Python: the wall for the sources (no game object named, no oracle, counting is sleep and never in
the wake), the event on constructed games (a taking and a taking that costs more than it wins, the end of the game for the goal kind, a window past a game cut by the ply cap), the plan's rule for K, the table counted on a planted signal, the key of a goal as the wake builds it against the key as the sleep builds it, the binding. No match is played here."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import goal_reach as GR
from catspace.incubator.analyst.discovery import goal_table as GT
from catspace.incubator.analyst.discovery import whole_79_13 as W13
from catspace.incubator.analyst.discovery import whole_play as WP

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")
WORTH = {"pawn": 475, "knight": 1216, "bishop": 1301, "rook": 1945, "queen": 3583}


@pytest.mark.parametrize("name", ["goal_reach.py", "goal_table.py", "whole_79_13.py"])
def test_the_wall_the_measure_scripts_name_no_game_object_and_reach_for_no_oracle(name):
    low = (DIR / name).read_text().lower()
    assert not re.search(r"\b[a-h][1-8]\b", low), re.search(r"\b[a-h][1-8]\b", low).group(0)
    assert "sys.path.insert" not in low
    for w in BANNED:
        assert not re.search(rf"\b{w}\b", low), (name, w)
    assert not re.search(r"ladder_oracle|leaf_ask|stockfish|analyse_pos|cs_learn|anthropic|\blo\.", low)


def test_counting_the_table_is_sleep_and_nothing_of_it_is_in_the_wake_or_the_binding():
    rust = "".join(p.read_text() for p in (ROOT / "rust" / "engine" / "decision" / "src").glob("*.rs"))
    assert "sklearn" not in rust and "np.unique" not in rust and "bootstrap" not in rust.lower()
    assert "cs_learn" not in (ROOT / "rust" / "engine" / "decision" / "Cargo.toml").read_text()
    for name in ("goal_reach.py", "goal_table.py"):
        assert ".wake_decide(" not in (DIR / name).read_text(), "the sleep's counting never calls the wake"
    play = (DIR / "whole_play.py").read_text()
    assert "goal_table import" not in play and "sklearn" not in play


# ------------------------------------------------------------------------------------------------------------------------------ A: the event

def _model():
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    return GL.model(HD.GAME, HD.VARIANT)


def _goal(side: str, target: str, unit: bool = False) -> dict:
    return {"side": side, "target": target, "source": "no_bearer_no_move", "depth": 0, "parent": None, "via": None, "unmet": ["bears", "bears"], "stoppers": [], "fixes": [{"kind": "Remove", "moves": ["x"]}],
            "in_the_way": 1, "chance": {"chance": 0.1, "n": 50, "source": "key"}, "chosen": False, "goal_unit": unit}


def _dec(ply: int, goals: list) -> dict:
    return {"ply": ply, "wake": {"log": {"goals": goals}}}


def _row_at(rows: dict, i: int) -> list:
    return [int(x) for x in rows["reached"][i]]


def test_a_taking_is_reached_while_the_side_is_not_behind_and_not_reached_when_the_taking_costs_more_than_it_wins():
    moves = ["e2e4", "e7e5", "d1h5", "b8c6", "h5e5", "c6e5"]               # the queen takes a pawn and is taken: the balance of worths falls under where it was
    res = {"moves": moves, "unfinished": False, "outcome": 0, "gid": "g"}
    rows = GR.game_rows(_model(), [_dec(4, [_goal("white", "black pawn on e5")])], res, WORTH, "king")
    r = dict(zip(GR.KS, _row_at(rows, 0)))
    assert r[1] == 1, "taken at once and the side is ahead by a pawn"
    assert r[2] == 0 and r[60] == 0, "taken, then the queen is lost: behind by the count of worths"
    assert rows["rank"][0] == GR.rank_of(WORTH, "pawn", "king") == 0 and rows["fix"][0] == 1 and rows["unmet"][0] == "bears" and rows["mine"][0] == 1


def test_a_window_past_the_end_of_a_game_cut_by_the_ply_cap_is_censored_and_past_a_decided_game_it_is_known():
    moves = ["e2e4", "e7e5", "d1h5", "b8c6", "h5e5", "c6e5"]
    cut = GR.game_rows(_model(), [_dec(4, [_goal("white", "black pawn on e5")])], {"moves": moves, "unfinished": True, "outcome": None, "gid": "g"}, WORTH, "king")
    r = dict(zip(GR.KS, _row_at(cut, 0)))
    assert (r[1], r[2], r[3], r[60]) == (1, 0, -1, -1), "inside the game the window is exact, past the cut the game was not decided"


def test_the_goal_kind_is_reached_when_the_game_ends_for_the_side_within_the_window():
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
    res = {"moves": moves, "unfinished": False, "outcome": 0, "gid": "g"}                 # white's score 0: the side that moves second won
    rows = GR.game_rows(_model(), [_dec(2, [_goal("black", "white king on e1", True), _goal("white", "black king on e8", True)])], res, WORTH, "king")
    black, white = (dict(zip(GR.KS, _row_at(rows, i))) for i in (0, 1))
    assert (black[1], black[2], black[60]) == (0, 1, 1)
    assert (white[2], white[60]) == (0, 0), "the game did not end for the side that lost it"
    assert rows["rank"][0] == len(set(WORTH.values())) and rows["role"][0] == 1 and rows["role"][1] == 0


def test_a_target_the_tracker_does_not_find_is_left_out():
    rows = GR.game_rows(_model(), [_dec(2, [_goal("white", "black pawn on e5")])], {"moves": ["e2e4", "e7e5", "d1h5"], "unfinished": False, "outcome": 50, "gid": "g"}, WORTH, "king")
    assert len(rows["ply"]) == 1
    rows = GR.game_rows(_model(), [_dec(2, [_goal("white", "black pawn on h5")])], {"moves": ["e2e4", "e7e5", "d1h5"], "unfinished": False, "outcome": 50, "gid": "g"}, WORTH, "king")
    assert len(rows["ply"]) == 0


def test_the_rank_of_a_stake_is_the_count_of_distinct_worths_under_it():
    assert [GR.rank_of(WORTH, k, "king") for k in ("pawn", "knight", "bishop", "rook", "queen", "king")] == [0, 1, 2, 3, 4, 5]
    assert GR.rank_of({"a": 1, "b": 1, "c": 2}, "c", "z") == 1 and GR.rank_of({"a": 1}, "unknown", "z") == 0
    assert GR.kind_of("white pawn on e5", "white") == "pawn"


def test_the_rule_for_k_is_the_smallest_k_at_which_nine_in_ten_of_the_goals_ever_reached_are_reached():
    n = 1000
    r = np.zeros((n, len(GR.KS)), np.int8)
    t = np.r_[np.full(500, 3), np.full(450, 10), np.full(50, 50), np.full(n - 1000, 1)]       # 500 reached by 3, 450 by 10, 50 by 50 plies
    for j, K in enumerate(GR.KS):
        r[:, j] = (t <= K)
    k, table = GR.choose_k(r, np.ones(n, bool))
    assert k == 10, "950 of 1000 are reached by 10: nine in ten"
    assert [x[0] for x in table] == list(GR.KS)
    r[:100, 3] = -1                                                           # a censored window is left out of that K
    assert GR.choose_k(r, np.ones(n, bool))[1][3][1] == n - 100


# ------------------------------------------------------------------------------------------------------------------------------ B: the table

def _planted(seed: int = 3, games: int = 200, per: int = 30) -> dict:
    """A table of goals where the chance depends on the things in the way and nothing else."""
    rng = np.random.default_rng(seed)
    n = games * per
    in_way = rng.integers(0, 4, n)
    p = np.array([0.8, 0.5, 0.2, 0.05])[in_way]
    hit = (rng.random(n) < p).astype(np.int8)
    d = {"game": np.repeat(np.arange(games), per).astype(np.int32), "in_way": in_way.astype(np.int32), "depth": rng.integers(0, 2, n).astype(np.int32), "rank": rng.integers(0, 3, n).astype(np.int32),
         "fix": rng.integers(0, 2, n).astype(np.int32), "mine": rng.integers(0, 2, n).astype(np.int32), "source": rng.integers(0, 2, n).astype(np.int32), "unmet": rng.integers(0, 3, n).astype(np.int32),
         "source_vocab": np.array(["a_source", "b_source"]), "unmet_vocab": np.array(["bears", "bears+kept", "unguarded"]), "reached": np.tile(hit[:, None], (1, len(GR.KS)))}
    return d


def test_the_table_is_counted_on_the_fit_rows_backs_off_and_predicts_a_planted_signal_on_held_out_rows():
    d = _planted()
    j = GR.KS.index(8)
    side = d["game"] < 100
    table = GT.fit_table(d, j, side)
    assert table["k"] == 8 and table["floor"] == GT.FLOOR and len(table["levels"]) == len(GT.LEVELS)
    assert table["total"][0] == int(side.sum())
    assert all(c[1] >= GT.FLOOR for lv in table["levels"] for c in lv["cells"]), "a cell under the floor is left out"
    held = np.where(~side)[0]
    p, lv = GT.predict(table, d, held, j)
    y = d["reached"][held, j].astype(float)
    base = (table["total"][1] + 1.0) / (table["total"][0] + 2.0)
    assert GT.logloss_rows(p, y).mean() < GT.logloss_rows(np.full(len(held), base), y).mean() - 0.05, "the table predicts what the things in the way decide"
    assert (lv <= len(table["levels"])).all()
    top = {iw: p[d["in_way"][held] == iw].mean() for iw in range(4)}
    assert top[0] > top[1] > top[2] > top[3]
    m, lo, hi = GT.boot_diff(GT.logloss_rows(np.full(len(held), base), y), GT.logloss_rows(p, y), d["game"][held], draws=60)
    assert lo < m < hi and lo > 0


def test_the_key_text_is_the_values_in_the_order_of_the_fields_and_the_binding_form_is_the_levels_with_their_cells():
    d = _planted(games=10, per=5)
    ks = GT.key_strings(d, ("source", "unmet", "in_way", "depth", "rank", "fix", "mine"), np.array([0, 1]))
    assert ks[0].count("|") == 6 and ks[0].split("|")[0] in ("a_source", "b_source") and ks[0].split("|")[1] in ("bears", "bears+kept", "unguarded")
    tab = {"k": 60, "floor": 20, "levels": [{"fields": ["mine"], "cells": [["1", 30, 10]]}], "total": [100, 25]}
    assert GT.binding_form(tab) == (60, 20, [(["mine"], [("1", 30, 10)])], (100, 25))
    gl = _goal("white", "black pawn on e5")
    assert GT.goal_key(gl, True, WORTH, "king", ("source", "unmet", "in_way", "depth", "rank", "fix", "mine"), "black") == "no_bearer_no_move|bears|1|0|0|1|1"


# ------------------------------------------------------------------------------------------------------------------------------ C: the player and the binding

def test_the_player_gives_the_wake_a_table_only_where_its_spec_has_one():
    spec = {"kind": WP.KIND_WAKE, "worth": {}, "chooser": {"mode": "gain", "weights": {}, "seed": 1}}
    plain = WP.MoverWhole(dict(spec), 0.1, "game:7:0", "/tmp/none")
    assert "goal_table" not in plain.chooser_kw(10)
    tab = (60, 20, [(["mine"], [("1", 30, 10)])], (100, 25))
    with_table = WP.MoverWhole({**spec, "goal_table": tab}, 0.1, "game:7:0", "/tmp/none")
    assert with_table.chooser_kw(10)["goal_table"] == tab and with_table.chooser_kw(10)["chooser"][0] == "gain"
    ps = WP.player_spec("wake", None, None, tab)
    assert ps["goal_table"] == tab and "goal_table" not in WP.player_spec("old", None, None, tab)
    ms = WP.match_specs("wake", "old", "A", 1, 0.2, Path("/tmp/s"), Path("/tmp/l"), table_a=tab, max_plies=200)
    assert ms[0]["players"][0].get("goal_table") == tab and all(m["max_plies"] == 200 for m in ms)
    assert WP.match_specs("wake", "old", "A", 1, 0.2, Path("/tmp/s"), Path("/tmp/l"))[0]["max_plies"] == WP.TM.PLY_CAP


def _decide(table, chooser, moves=(), seconds=1.5):
    import cs_engine as CE
    from catspace.incubator.analyst.discovery import odds_measure as OM
    from catspace.incubator.analyst.discovery import record_play as RP
    w = WP.world()
    pos = CE.state(OM.VARIANT, None, list(moves))[0]
    history = [CE.state(OM.VARIANT, None, list(moves[:k]))[4] for k in range(len(moves))]
    kw = {}
    if table is not None:
        kw["goal_table"] = table
    out = json.loads(w.model.wake_decide(pos, history, seconds, 64, dict(RP.worth_of("human")), {}, w.goal, WP.PIECES, 0, 0, w.keys, WP.fuses(), w.chances, w.reasons, WP.TM.LEDGER_SHARE, None, chooser, **kw))
    return out, CE.state(OM.VARIANT, None, list(moves))[1], w


FULL = ("source", "unmet", "in_way", "depth", "rank", "fix", "mine")
OPENING = ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6")


def test_the_key_of_a_goal_as_the_wake_builds_it_is_the_key_as_the_sleep_builds_it():
    probe = (60, 1, [(list(FULL), [])], (100, 10))
    first, mover, w = _decide(probe, None, OPENING)
    goals = first["log"]["goals"]
    assert goals and all(g["chance"]["source"] == "total" and abs(g["chance"]["chance"] - 11.0 / 102.0) < 1e-9 for g in goals), "a table with no cell gives the whole's rate"
    from catspace.incubator.analyst.discovery import record_play as RP
    worth = GR.worth_by_kind(dict(RP.worth_of("human")))
    cells = {}
    for i, g in enumerate(goals):
        cells.setdefault(GT.goal_key(g, g["side"] == mover, worth, w.goal, FULL, "black" if g["side"] == "white" else "white"), (40 + i, 7 + i % 5))
    table = (60, 1, [(list(FULL), [(k, n, h) for k, (n, h) in cells.items()])], (100, 10))
    second, _m, _w = _decide(table, None, OPENING)
    seen = 0
    for g in second["log"]["goals"]:
        key = GT.goal_key(g, g["side"] == mover, worth, w.goal, FULL, "black" if g["side"] == "white" else "white")
        if key in cells:
            n, h = cells[key]
            assert g["chance"]["source"] == "level0" and g["chance"]["n"] == n and abs(g["chance"]["chance"] - (h + 1.0) / (n + 2.0)) < 1e-9, (key, g["chance"])
            seen += 1
    assert seen >= max(1, len(second["log"]["goals"]) // 2), "the keys of the wake are the keys of the sleep"


def test_the_binding_takes_the_table_and_the_gain_mode_and_refuses_a_field_that_is_none():
    table = (60, 20, [(["in_way"], [("0", 100, 90), ("1", 100, 50), ("2", 100, 20)])], (1000, 150))
    out, _m, _w = _decide(table, ("gain", {}, 0), OPENING)
    c = out["log"]["chooser"]
    if c is not None:
        assert c["mode"] == "Gain" and "our_gain" in c["inputs"] and all(len(x["row"]) == len(c["inputs"]) for x in c["candidates"])
        assert set(x["move"] for x in c["candidates"]) <= set(out["equal"]) and out["move"] in out["equal"]
    plain, _m, _w = _decide(None, ("fixed", {}, 0), OPENING)
    assert plain["log"]["chooser"] is None or all(x["row"][-4:] == [0.0, 0.0, 0.0, 0.0] for x in plain["log"]["chooser"]["candidates"]), "no table: the gain inputs are nought"
    with pytest.raises(Exception, match="field"):
        _decide((60, 20, [(["no_such_field"], [])], (1, 0)), None, OPENING)
