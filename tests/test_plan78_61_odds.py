"""Plan 78.61: the judge that counts nothing, the odds starts, the effect of a unit, the record round trip, and the measure code naming no object of any game."""
import io
import json
import re
import tokenize
from pathlib import Path

import numpy as np
import pytest

import cs_engine as CE

from catspace.incubator.analyst.discovery import odds_measure as O

SRC = Path(O.__file__)
WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
FREE = "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1"          # a loose unit of theirs can be taken for nothing
ENDS = "7k/8/6K1/8/8/8/8/R7 w - - 0 1"             # the game ends in one move


def pick(text, judge=None, order=None, worth=None):
    kw = {}
    if judge:
        kw["judge"] = judge
    if worth:
        kw["worth"] = worth
    if order is not None:
        kw["root_orders"] = [order]
        kw["root_order_mode"] = "strict"
    return CE.solve_batch([("chess", text, [])], 0.2, threads=1, labels=["t"], **kw)[0][0]


def test_the_judge_that_counts_nothing_prefers_nothing_but_a_forced_end():
    legal = CE.state("chess", FREE, [])[2]
    n = len(legal)
    assert pick(FREE) == "d1d5"                                                 # the default judge takes the loose unit
    picks = {pick(FREE, judge="nothing", order=[(i + k) % n for k in range(n)]) for i in range(n)}
    assert len(picks) > 3 and picks <= set(legal)                                # counting nothing, any of the moves that do not lose is as good: the order decides
    ends = CE.state("chess", ENDS, [])[2]
    for i in range(len(ends)):
        order = [(i + k) % len(ends) for k in range(len(ends))]
        assert pick(ENDS, judge="nothing", order=order) == "a1a8"               # a forced end is preferred whatever the order


def test_an_unknown_judge_name_is_an_error_and_no_name_is_what_it_was():
    with pytest.raises(ValueError):
        pick(FREE, judge="count")
    assert pick(FREE, judge=None) == pick(FREE)


def test_odds_starts_are_legal_positions_that_differ_from_the_even_start_by_exactly_the_units_named():
    made = O.make_starts(2)
    even = made["starts"]["even"]
    base = O.units_of(even["text"])
    assert base == O.units_of(None)                                             # the control lacks nothing
    assert O.playable(even["text"])
    assert "king" in made["refused"] or any(k not in {s["kind"] for s in made["starts"].values()} for k in {u[1] for u in base})
    seen_kinds = set()
    for sid, s in made["starts"].items():
        if s["kind"] is None:
            continue
        assert O.playable(s["text"]), sid
        units = O.units_of(s["text"])
        gone = sorted(set(base) - set(units))
        assert set(units) <= set(base), sid
        assert [g[2] for g in gone] == sorted(s["places"]), sid
        assert len(gone) == s["n"] and all(g[1] == s["kind"] for g in gone), sid
        seen_kinds.add(s["kind"])
    assert len(seen_kinds) >= 4
    assert any(s["n"] == 2 for s in made["starts"].values()) and any(s["side"] == 0 for s in made["starts"].values()) and any(s["side"] == 1 for s in made["starts"].values())


def test_the_goal_unit_cannot_be_removed():
    made = O.make_starts(2)
    assert len(made["refused"]) == 1 and not any(s["kind"] in made["refused"] for s in made["starts"].values())


def test_effect_of_a_unit_has_the_sign_of_taking_it_away():
    starts = {"even": {"kind": None, "side": None, "n": 0}, "a:0:1": {"kind": "a", "side": 0, "n": 1}, "a:1:1": {"kind": "a", "side": 1, "n": 1}, "b:0:1": {"kind": "b", "side": 0, "n": 1}, "b:1:1": {"kind": "b", "side": 1, "n": 1}}
    even = np.array([1.0] * 30 + [0.5] * 40 + [0.0] * 30)
    lose = np.array([1.0] * 10 + [0.5] * 40 + [0.0] * 50)           # white short loses more
    win = np.array([1.0] * 50 + [0.5] * 40 + [0.0] * 10)            # black short: white wins more
    flat = even.copy()
    cells = {"even": even, "a:0:1": lose, "a:1:1": win, "b:0:1": flat, "b:1:1": flat}
    e = O.effects(cells, starts)
    assert e[("a", 1)] > 0.5 and abs(e[("b", 1)]) < 1e-9
    s = O.summarise(cells, starts, draws=100)
    assert s["order"] == ["a", "b"] and s["per_kind"]["a"]["one"]["lo"] > 0


def test_the_record_round_trip_feeds_the_next_judge_and_nothing_else(tmp_path):
    kinds = ["queen", "pawn"]
    starts = {"even": {"kind": None, "side": None, "n": 0}}
    cells = {"even": np.array([1.0, 0.5, 0.0, 0.5])}
    for k, strength in zip(kinds, (1, 0)):
        for side in (0, 1):
            sid = f"{k}:{side}:1"
            starts[sid] = {"kind": k, "side": side, "n": 1}
            win_white = (side == 1) == bool(strength)
            cells[sid] = np.array([1.0, 1.0, 0.5, 0.0]) if win_white else np.array([0.0, 0.0, 0.5, 1.0])
    s = O.summarise(cells, starts, draws=50)
    rec = O.record_of_generation(0, s, {"seconds": 0.05}, [], None)
    (tmp_path / "gen0").mkdir()
    (tmp_path / "gen0" / "record.json").write_text(json.dumps(rec))
    assert O.players_of(0, tmp_path) == {"kind": "nothing"}
    p = O.players_of(1, tmp_path)
    assert p["kind"] == "worth" and p["record_kinds"] == rec["heads"]["odds_effect"]["kind_values"]
    keys = O.kernel_names()
    assert p["worth"] == {keys[k]: v for k, v in p["record_kinds"].items()} and set(p["worth"]) <= set(keys.values())
    assert rec["heads"]["odds_effect"]["source"] == "odds games, generation 0"
    assert p["record_kinds"]["queen"] == int(round(O.SCALE * s["per_kind"]["queen"]["one"]["term"]))


def test_a_worth_judge_fed_from_a_record_changes_the_choice_and_a_table_keyed_by_the_graphs_word_would_not(tmp_path):
    """The scar of the first run: the engine names a captured unit by its own symbol, so a table keyed by the graph's kind word is worth nothing to it."""
    keys = O.kernel_names()
    legal = CE.state("chess", FREE, [])[2]
    zero = CE.solve_batch([("chess", FREE, [])], 0.2, threads=1, worth={"x": 0})[0][0]
    wrong = CE.solve_batch([("chess", FREE, [])], 0.2, threads=1, worth={k: 1000 for k in keys})[0][0]
    right = CE.solve_batch([("chess", FREE, [])], 0.2, threads=1, worth={v: 1000 for v in keys.values()})[0][0]
    assert wrong == zero and right == "d1d5" and zero != "d1d5" and zero in legal


def test_a_match_score_counts_each_colour_of_an_opening_once():
    games = [{"start": "1", "colour": 0, "outcome": 100}, {"start": "1", "colour": 1, "outcome": 100}, {"start": "2", "colour": 0, "outcome": None}, {"start": "2", "colour": 1, "outcome": 50}]
    pairs = O.pair_scores(games)
    assert pairs == {"1": 1.0, "2": 1.0}                                         # white wins both: a win and a loss for a; unfinished and a draw are halves
    sc, lo, hi = O.score_band(pairs, draws=50)
    assert sc == 0.5 and lo <= sc <= hi


def test_the_measure_code_names_no_object_of_any_game():
    toks = tokenize.generate_tokens(io.StringIO(SRC.read_text()).readline)
    names = [t.string for t in toks if t.type == tokenize.NAME]
    hits = [n for n in names if re.fullmatch(WORDS, n, flags=re.I)]
    assert hits == []
