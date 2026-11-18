"""Plan 78.67: the event heads as the choice among the moves the search holds equal (catspace/incubator/analyst/discovery/tiebreak_number.py and tiebreak_measure.py)."""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pytest

import cs_engine as CE

from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import tiebreak_measure as TM
from catspace.incubator.analyst.discovery import tiebreak_number as TN

CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
MODULES = [Path(TM.__file__), Path(TN.__file__)]


def _position(seed: int, plies: int) -> list:
    rng = random.Random(seed)
    moves: list = []
    for _ in range(plies):
        legal = CE.state(OM.VARIANT, None, moves)[2]
        moves.append(rng.choice(sorted(legal)))
    return moves


def test_the_scripts_name_no_game_object():
    for p in MODULES:
        hit = re.search(CHESS_WORDS, p.read_text(), flags=re.I)
        assert hit is None, (p.name, hit.group(0))


def test_the_two_sets_of_openings_share_no_opening():
    a, b = TM.openings("A", 100), TM.openings("B", 100)
    assert len(a) == len(b) == 100
    assert {tuple(o) for _s, o in a}.isdisjoint({tuple(o) for _s, o in b})
    assert {s for s, _o in a}.isdisjoint({s for s, _o in b})
    assert len({tuple(o) for _s, o in b}) == 100                      # no opening twice inside the set either


def test_an_order_over_the_root_moves_changes_neither_the_value_nor_the_moves_held_best():
    """The mechanism of 78.51 (strict mode) at a fixed depth: the value and the set of equal moves are the search's, the move kept is the first of that set in the order given."""
    for seed in (1, 2, 3):
        moves = _position(seed, 8)
        legal = CE.state(OM.VARIANT, None, moves)[2]
        plain = CE.solve_equals_batch([(OM.VARIANT, None, moves)], 5.0, threads=1, ledger_depth=3)[0]
        eq_plain, val_plain = set(plain[6]), plain[8]
        rng = random.Random(seed)
        for _ in range(3):
            order = list(range(len(legal)))
            rng.shuffle(order)
            row = CE.solve_equals_batch([(OM.VARIANT, None, moves)], 5.0, threads=1, ledger_depth=3, root_orders=[order], root_order_mode="strict")[0]
            assert row[8] == val_plain
            assert set(row[6]) == eq_plain
            first = next(i for i in order if i in eq_plain)
            assert row[0] == legal[first]


def test_the_choice_among_equals_is_one_of_them_and_leaves_the_search_alone():
    nums = {"x": 0.2, "y": 0.5, "z": 0.5}
    assert TM.pick_best("z", nums) == "z"                              # a tie at the top goes to the search's own move
    assert TM.pick_best("x", nums) == "y"                              # else the first by name
    assert TM.pick_best("x", {}) == "x"                                # no number: the search's own move
    assert TM.pick_best("x", {"x": 0.1, "y": 0.2}) == "y"


def test_the_number_is_antisymmetric_when_the_two_sides_are_swapped():
    rng = np.random.default_rng(0)
    a, b = rng.random(50), rng.random(50)
    assert np.allclose(TN.first_difference(a, b), -TN.first_difference(b, a))
    assert np.allclose(TN.first_difference(a, a), 0.0)


def test_the_reading_time_is_taken_from_the_reading_players_clock():
    assert TM.reading_clock(0.2, []) == 0.2
    assert TM.reading_clock(0.2, [0.05, 0.05]) == pytest.approx(0.15)
    assert TM.reading_clock(0.2, [5.0]) == pytest.approx(0.1)          # never under half of it
    base = {"kind": "worth", "worth": {}}
    for name, charge, expect in (("trees", True, True), ("trees", False, False), ("control", True, False), ("random", True, False), ("arrival", True, True)):
        m = TM.Mover67(dict(base, kind=TM.KIND[name], charge=charge), 0.2, "s", "/tmp/none")
        m.costs = [0.05]
        assert (m.search_clock() < 0.2) == expect, (name, charge)
    TM.REG.clear()


def test_the_statements_of_the_children_are_those_of_the_game_read_whole():
    """The one-hot reading of the root's children gives, with its sign turned, the statements the fit used at that child position (the side to move there less the other side)."""
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.incubator.analyst.discovery import sheet_rust as SR
    sheet = SR.model(QS.Env(OM.VARIANT), {})
    n = len(sheet.subgoal_names())
    moves = _position(5, 12)
    legal = CE.state(OM.VARIANT, None, moves)[2]
    got = TN.statements_children(sheet, n, moves, legal[0])
    for k in (0, len(legal) // 2, len(legal) - 1):
        full = np.array(sheet.subgoal_games([(None, moves + [legal[k]])], 1)[0][-1], np.float32)
        assert np.allclose(got[k], full[:n] - full[n:])


def test_every_player_is_named_and_the_default_player_is_untouched():
    assert set(TM.KIND) == set(TM.PLAYERS)
    assert all(k.startswith("tb_") for k in TM.KIND.values())
    assert OM.Mover.choose is not TM.Mover67.choose                     # the 78.61 mover is subclassed, not edited


def _fake_specs(n_pairs: int) -> list:
    return [{"gid": f"x:{i}:{c}", "start": f"S{i:03d}", "colour": c} for i in range(n_pairs) for c in (0, 1)]


def test_a_match_that_is_clear_stops_early_on_whole_leading_pairs_and_a_close_one_plays_on(tmp_path):
    specs = _fake_specs(100)
    # player a wins every game it moves first in and draws the rest: far from a half
    one_sided = lambda s: dict(s, outcome=100 if s["colour"] == 0 else 50)
    said: list = []
    got = TM.run_games(specs, 1, tmp_path / "a.jsonl", say=said.append, play=one_sided, stop=TM.settled)
    assert 60 <= len(got) < 200 and len(got) % 20 == 0
    assert [g["gid"] for g in got] == [s["gid"] for s in specs[:len(got)]]
    assert any("early stop" in line for line in said)
    # a match of alternating whole points is neither clear nor level: every game is played
    mixed = lambda s: dict(s, outcome=100 if (int(s["start"][1:]) + s["colour"]) % 2 == 0 else 0)
    rng = random.Random(5)
    noisy = lambda s: dict(s, outcome=rng.choice([0, 50, 100, 100, 0, 50, 0, 100]))
    assert len(TM.run_games(specs, 1, tmp_path / "b.jsonl", say=lambda _: None, play=noisy, stop=lambda games: None)) == 200
    assert TM.settled([one_sided(s) for s in specs[:40]]) is None            # under the least number of pairs
    assert TM.settled([dict(s, outcome=50) for s in specs]) == "level"
    assert TM.settled([one_sided(s) for s in specs]) == "clear"
    assert len(TM.run_games(specs, 1, tmp_path / "c.jsonl", say=lambda _: None, play=mixed)) == 200      # no stop given: as before
