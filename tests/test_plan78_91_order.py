"""Plan 78.91: the hint in the search's order (catspace/incubator/analyst/discovery/symbols_order_play.py, symbols_order_measure.py). SYSTEM_SPEC 5, 7, 9.

The static order puts the moves that gain first (best gain first, ties as generated) and the rest as generated; the hint's order keeps the gaining moves where they are and reorders only the quiet moves by the score, a tie in the
static order, an unscored quiet move after the scored; every order is a permutation; the order of a short reading puts its move first; the place among a set counts that set only; the gains the reader finds are zero exactly where no
unit is taken (on positions of a seeded walk); the order handed to the engine's root order argument is accepted and the move comes back from the legal moves; the paired bootstrap and the condition read what the doc says; the
measure names no game object and writes under the experiment folder only."""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pytest

import cs_engine as CE

from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import symbols_order_measure as M
from catspace.incubator.analyst.discovery import symbols_order_play as P

HAVE = (Path(M.paths.experiment("symbols_78_90", "full", "fits.pkl")).exists() and Path(OM.out_root() / "full" / "gen2" / "record.json").exists())
needs_run = pytest.mark.skipif(not HAVE, reason="needs the 78.90 run and the odds record")


def test_the_static_order_is_the_moves_that_gain_best_first_then_the_rest_as_generated():
    gains = [0, 3, 0, 9, -2, 0, 3]
    assert P.static_order(gains) == [3, 1, 6, 4, 0, 2, 5]
    assert P.static_order([0, 0, 0]) == [0, 1, 2]


def test_the_hint_order_reorders_only_the_quiet_moves_and_keeps_the_gaining_ones_in_place():
    gains = [0, 3, 0, 9, 0, 0]
    scores = {0: 0.1, 2: 0.9, 4: 0.1, 5: -0.3}
    o = P.hint_order(gains, scores)
    assert o == [3, 1, 2, 0, 4, 5]                          # gains first as the static order has them; then 0.9, then the tie of 0.1 in the static order, then -0.3
    assert o[:2] == P.static_order(gains)[:2]


def test_a_quiet_move_without_a_score_comes_after_the_scored_ones_in_the_static_order():
    gains = [0, 0, 0, 0]
    assert P.hint_order(gains, {1: 0.5, 3: 0.4}) == [1, 3, 0, 2]
    assert P.hint_order(gains, {}) == [0, 1, 2, 3]


def test_scores_that_agree_to_the_tie_digits_are_a_tie():
    gains = [0, 0, 0]
    assert P.hint_order(gains, {0: 0.3, 1: 0.3 + 1e-12, 2: 0.1}) == [0, 1, 2]
    assert P.hint_order(gains, {0: 0.3, 1: 0.3 + 1e-3, 2: 0.1}) == [1, 0, 2]


def test_every_order_is_a_permutation_of_the_moves():
    rng = random.Random(3)
    for _ in range(50):
        n = rng.randrange(1, 40)
        gains = [rng.choice([0, 0, 0, 5, -2, 9]) for _ in range(n)]
        quiet = [i for i, g in enumerate(gains) if g == 0]
        scores = {i: rng.random() for i in quiet if rng.random() < 0.8}
        first = rng.randrange(n)
        eq = rng.sample(range(n), min(n, rng.randrange(1, 5)))
        for o in (P.static_order(gains), P.hint_order(gains, scores), P.short_order(gains, first, eq)):
            assert sorted(o) == list(range(n))


def test_the_order_of_a_short_reading_puts_its_move_first_then_the_others_it_held_equal_then_the_static_order():
    gains = [0, 3, 0, 0, 9, 0]
    assert P.short_order(gains, 2, [5, 2, 0]) == [2, 0, 5, 4, 1, 3]


def test_the_place_is_one_based_and_counts_only_the_moves_of_the_set_asked():
    o = [4, 1, 3, 0, 2]
    assert P.place_of(o, 4) == 1 and P.place_of(o, 2) == 5
    assert P.place_of(o, 0, {0, 2, 3}) == 2 and P.place_of(o, 3, {3}) == 1


def test_the_paired_band_is_clear_of_zero_for_a_shift_and_not_for_noise():
    from catspace.incubator.analyst.discovery import symbols_measure as SM
    g = np.repeat(np.arange(60), 2)
    rng = np.random.default_rng(1)
    a = rng.integers(1, 20, 120).astype(float)
    m, lo, hi = SM.paired(a + 3.0, a, g, 5)
    assert lo > 0 and abs(m - 3.0) < 1e-9
    m, lo, hi = SM.paired(a, rng.permutation(a), g, 5)
    assert lo < 0 < hi


def random_walk(n: int, seed: int) -> list:
    rng = random.Random(seed)
    moves: list = []
    for _ in range(n):
        legal = CE.state(OM.VARIANT, None, moves)[2]
        if not legal:
            break
        moves.append(rng.choice(sorted(legal)))
    return moves


@needs_run
def test_the_gain_is_nonzero_exactly_where_a_unit_is_taken():
    rd = P.hint_order_reader()
    seen_take = 0
    for seed in range(6):
        walk = random_walk(36, seed)
        for k in (12, 20, 30, len(walk)):
            moves = walk[:k]
            read = rd.read(moves)
            text, mover = read["text"], CE.state(OM.VARIANT, None, moves)[1]
            before = rd.facts.board(text)
            for m, g in zip(read["legal"], read["gains"]):
                after = rd.facts.board(CE.state(OM.VARIANT, text, [m])[0])
                lost = sum(1 for o, _k, _p in (before - after) if o != mover)
                assert (g != 0) == (lost > 0), (moves, m, g)
                seen_take += lost > 0
            assert read["quiet"] == [i for i, g in enumerate(read["gains"]) if g == 0]
    assert seen_take > 0


@needs_run
def test_the_hint_order_is_accepted_as_the_root_order_and_the_move_comes_back_from_the_legal_moves():
    rd = P.hint_order_reader()
    base = OM.players_of(M.TM.WORTH_GEN, OM.out_root() / "full")
    for moves in (random_walk(8, 1), random_walk(24, 2)):
        for which in ("own", "reasons"):
            order, read, scores = rd.order(moves, which)
            assert sorted(order) == list(range(len(read["legal"])))
            assert set(scores) <= set(read["quiet"])
            row = PM_row(moves, base, order)
            assert row["error"] is None and row["move"] in read["legal"]


def PM_row(moves: list, base: dict, order: list) -> dict:
    from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
    return PM.equal_set(CE.solve_equals_batch([(OM.VARIANT, None, list(moves))], 0.1, worth=dict(base["worth"]), ours_first=False, valuation="odds record", threads=1, ledger_share=0.4,
                                              root_orders=[order], root_order_mode="ties")[0])


@needs_run
def test_a_strict_order_returns_the_first_of_the_best_valued_moves_in_that_order():
    base = OM.players_of(M.TM.WORTH_GEN, OM.out_root() / "full")
    moves = random_walk(10, 4)
    n = len(CE.state(OM.VARIANT, None, moves)[2])
    o = list(range(n))
    random.Random(7).shuffle(o)
    from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
    row = PM.equal_set(CE.solve_equals_batch([(OM.VARIANT, None, list(moves))], 0.3, worth=dict(base["worth"]), ours_first=False, valuation="odds record", threads=1, ledger_share=0.4,
                                             root_orders=[o], root_order_mode="strict")[0])
    legal = CE.state(OM.VARIANT, None, moves)[2]
    assert row["error"] is None
    held = set(row["equal_indices"])
    chosen = legal.index(row["move"])
    assert chosen in held and chosen == min(held, key=o.index)       # the move kept is the first of the equal moves in the given order


def test_the_condition_prefers_the_own_fit_and_needs_a_band_clear_of_zero():
    verdicts = {"ii-own": {"band few": 0.4, "band mid": -0.1, "band many": -0.3}, "ii-reasons": {"band few": -0.2, "band mid": 0.3, "band many": -0.1}}
    c = M.condition_of(verdicts)
    assert c == {"met": True, "which": "own", "bands": {"ii-own": ["band few"], "ii-reasons": ["band mid"]}}
    c = M.condition_of({"ii-own": {"band few": -0.4, "band mid": 0.0}, "ii-reasons": {"band few": 0.2}})
    assert c["met"] and c["which"] == "reasons"
    c = M.condition_of({"ii-own": {"band few": -0.4, "band mid": 0.0}, "ii-reasons": {"band few": -0.2}})
    assert not c["met"] and c["which"] is None


def test_the_reference_is_the_engines_own_search_at_ten_times_the_play_clock_and_the_settings_are_the_matches_of_78_89():
    assert abs(M.LONG - 10 * M.PLAY_CLOCK) < 1e-12 and M.PLAY_CLOCK == 0.2 and M.LEDGER_SHARE == 0.4 and M.N_CELL == 100
    assert sum(M.CASE_K["reasons_first"].values()) == 20 and sum(M.CASE_K["static_first"].values()) == 10
    assert set(M.MATCH_KINDS) >= {"or_own", "or_reasons", "control"}


def test_the_measure_names_no_object_of_any_game():
    for f in ("symbols_order_measure.py", "symbols_order_play.py"):
        text = (Path(M.__file__).parent / f).read_text().lower()
        assert not re.search(r"\b[a-h][1-8]\b", text), f
        for w in ("fen", "san", "uci", "check", "mate", "promotion", "pawn", "bishop", "knight", "rook", "queen", "king", "castle"):
            assert not re.search(rf"\b{w}\b", text), (f, w)


def test_no_player_id_is_written_and_the_outputs_go_to_the_experiment_folder_only():
    text = (Path(M.__file__).parent / "symbols_order_measure.py").read_text()
    assert "game_id" not in text and 'g["id"]' not in text
    assert "paths.experiment" in text and "paths.derived" not in text and "write_text" in text
    assert "rules/learned" not in text.replace("rules/learned or data/derived", "")
    assert M.OUT == ("order_78_91",)
