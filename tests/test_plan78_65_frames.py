"""PLAN 78.65: the frame inputs are relative to the mover and swap with the side; the fit and the test games are disjoint; the tree head and the linear head are read on the same test rows by the same
likelihood; a column under the stated count is dropped; the fate of a unit (the row's future) is never an input; no chess word or event name is a literal of the new code."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import events_hazard as EH
from catspace.incubator.analyst.discovery import frames_heads as FH
from catspace.incubator.analyst.discovery import frames_state_rows as FS
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import tactical_state as TS
from catspace.io import paths

CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
CODE = ["frames_heads.py", "frames_state_rows.py"]
DISC = Path(__file__).resolve().parents[1] / "catspace" / "incubator" / "analyst" / "discovery"


def _state(units_per_row, frames):
    """A made-up state dict: units are (owner, kind, class, basis, trap) tuples per row."""
    n = len(units_per_row)
    flat = [u for row in units_per_row for u in row]
    return {"ply": np.arange(n, dtype=np.int16), "mover": np.zeros(n, np.int8), "frames": np.array(frames, np.int16), "unread": np.zeros(n, np.int32), "game": np.zeros(n, np.int32),
            "units": np.array(flat, np.uint8).reshape(-1, 5), "uoff": np.concatenate([[0], np.cumsum([len(r) for r in units_per_row])]).astype(np.int64)}


def test_the_frame_inputs_are_relative_to_the_mover_and_swap_with_the_side():
    lost, safe = TS.CLASS_WORDS.index("lost"), TS.CLASS_WORDS.index("safe")
    est, forced = TS.BASIS_WORDS.index("estimate"), TS.BASIS_WORDS.index("forced")
    row = [(1, 2, lost, est, 1), (1, 3, safe, forced, 0), (0, 4, safe, forced, 2), (0, 0, lost, est, 0), (0, 0, lost, est, 0)]
    flipped = [(1 - o, k, c, b, t) for (o, k, c, b, t) in row]                 # the same position seen from the other side to move
    st = _state([row, flipped], [3, 3])
    blocks = FH.frame_blocks(st, 6, ["a", "b", "c", "d", "e", "f"])
    for g, (X, names) in blocks.items():
        assert X.shape[0] == 2 and len(names) == X.shape[1], g
        half = X.shape[1] // 2
        if g != "frames":
            assert all(nm.startswith("us:") for nm in names[:half]) and all(nm.startswith("them:") for nm in names[half:]), g
            assert np.array_equal(X[0, :half], X[1, half:]) and np.array_equal(X[0, half:], X[1, :half]), g          # the sides swap, nothing else moves
    cls = blocks["classes"][0]
    nb = len(TS.BASIS_WORDS) + 1
    assert cls[0, lost * nb + est] == 1 and cls[0, cls.shape[1] // 2 + lost * nb + est] == 2                           # one of ours, two of theirs
    lk = blocks["lost kinds"][0]
    assert lk[0, 2] == 1 and lk[0, 6 + 0] == 2 and lk[0].sum() == 3
    fr = blocks["frames"][0]
    assert fr[0, 2] == 2 and fr[0, 3] == 3 and fr[0, 0] == 3 and fr[0, 4] == pytest.approx(5 / 3)


def test_the_fate_of_a_unit_is_not_stored_and_not_an_input():
    assert FS.KEEP == [0, 1, 2, 3, 4]                                         # the unit record's sixth column, what became of the unit, is the row's future
    st = _state([[(1, 0, 0, 2, 0)]], [1])
    assert st["units"].shape[1] == 5


def test_a_column_under_the_stated_count_is_dropped_and_counted():
    X = np.zeros((100, 3), np.float32)
    X[:FH.MIN_ROWS, 0] = 1.0                                                  # exactly the count: kept
    X[:FH.MIN_ROWS - 1, 1] = 2.0                                              # one under: dropped
    fit = np.ones(100, bool)
    X2, names, dropped = FH.drop_sparse(X, ["a", "b", "c"], fit)
    assert names == ["a"] and X2.shape == (100, 1)
    assert dropped == [("b", FH.MIN_ROWS - 1), ("c", 0)]
    only_fit = np.zeros(100, bool)
    only_fit[50:] = True                                                      # the count is read on the FIT rows only
    _x, names2, _d = FH.drop_sparse(X, ["a", "b", "c"], only_fit)
    assert names2 == []


def test_the_state_row_of_each_hazard_row_is_found_by_game_and_ply():
    st = {"game": np.array([0, 0, 1, 1, 1], np.int32), "ply": np.array([10, 12, 10, 11, 15], np.int16)}
    rows = {"g": np.array([0, 1, 0]), "ply": np.array([11, 12, 15])}
    ix = FH.align_state(rows, ["x", "y"], st, ["y", "x"])                      # the sample's game 0 is `x`, which the state numbers 1; its game 1 is `y`, the state's 0
    assert ix.tolist() == [3, 1, 4]
    # the sample's `y` has no row at ply 11: the lookup must refuse rather than guess
    with pytest.raises(AssertionError):
        FH.align_state({"g": np.array([1]), "ply": np.array([11])}, ["x", "y"], st, ["y", "x"])


def test_the_fit_and_the_test_games_are_disjoint_in_the_rows_the_heads_are_read_on():
    d = Path(paths.experiment(*TS.HEADS_DIR))
    if not (d / "rows.npz").exists():
        pytest.skip("the 78.59 rows are absent")
    rows = HD.load_rows(d / "rows.npz")
    fit_games, test_games = set(rows["game"][rows["fit"]].tolist()), set(rows["game"][~rows["fit"]].tolist())
    assert fit_games and test_games and not (fit_games & test_games)


def _head_with_a_pair_effect(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    rate = np.where((x[:, 0] > 0) == (x[:, 1] > 0), 0.30, 0.04)                 # the chance depends on the pair, and on neither input alone
    T = rng.integers(20, 40, n)
    k = rng.geometric(rate)
    d_ev = 2 * k - 1
    has = d_ev <= T
    D = np.where(has, d_ev - 1, T).astype(np.int64)
    h = EH.Head(("seen",), "us", D, has, np.zeros(n, bool))
    g = np.repeat(np.arange(n // 10), 10)
    rows = {"g": g, "ply": np.zeros(n, int), "T": T}
    return h, x, rows


def test_the_tree_head_and_the_linear_head_are_read_on_the_same_test_rows_by_the_same_likelihood():
    h, x, rows = _head_with_a_pair_effect()
    fit = np.flatnonzero(rows["g"] < 400)
    test = np.flatnonzero(rows["g"] >= 400)
    h.activate(np.ones(len(x), bool))
    lin = EH.fit_best(h, x, rows, fit, 1e-3)
    tree = FH.fit_tree(h, x, rows, fit, 60)
    P_lin, P_tree = FH.table_linear(lin, x, test), FH.table_tree(tree, x, test)
    assert P_lin.shape == P_tree.shape == (len(test), EH.NB, 2)
    tot_lin, _ = FH.nll_table(h, P_lin, test)
    tot_tree, _ = FH.nll_table(h, P_tree, test)
    assert len(tot_lin) == len(tot_tree) == len(test)                           # one total per test row, the same rows, the same denominator
    den = h.steps()[test].sum()
    ref, _ = EH.eval_head(h, x, lin, test)                                       # the linear head's own likelihood agrees with the table's
    assert np.allclose(ref, tot_lin, rtol=1e-4, atol=1e-4)
    # the base hazard from the table equals the closed form's
    b = h.base_logit(fit)
    base, _ = FH.nll_table(h, FH.table_linear(None, None, test, b), test)
    ref_b, _ = EH.eval_head(h, None, None, test, b)
    assert np.allclose(base, ref_b, rtol=1e-4, atol=1e-4)
    # a pair effect: the trees see it, the linear head cannot
    assert tot_lin.sum() / den > tot_tree.sum() / den + 0.01


def test_no_chess_word_or_event_name_is_a_literal_of_the_new_code():
    for name in CODE:
        src = (DISC / name).read_text()
        tree = ast.parse(src)
        docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr)
                and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str)}
        bad = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and re.search(CHESS_WORDS, n.value, flags=re.I):
                bad.append(n.value)
            if isinstance(n, ast.Name) and re.search(CHESS_WORDS, n.id, flags=re.I):
                bad.append(n.id)
        assert not bad, (name, bad)
        # comments and docstrings too: the whole text
        assert not re.search(CHESS_WORDS, src, flags=re.I), (name, re.search(CHESS_WORDS, src, flags=re.I).group(0))
