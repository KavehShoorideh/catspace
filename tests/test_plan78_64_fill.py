"""PLAN 78.64 stage E: the key of a small set of units is the same for either order of the sides and does not count the goal units; the rows of the human games and of the fill are joined with the
fill games kept out of the test rows; no event word is a literal of the fill code."""
from __future__ import annotations

import numpy as np

from catspace.incubator.analyst.discovery import events_fill as EF


def test_the_key_of_a_set_ignores_the_goal_units_and_the_order_of_the_sides():
    kinds = ["goal", "disc", "ring"]
    a = np.zeros((2, 2, 3), np.int8)                                  # two positions, two sides, three kinds
    a[0, 0] = [1, 2, 0]                                               # side 0: the goal unit and two discs
    a[0, 1] = [1, 0, 1]                                               # side 1: the goal unit and a ring
    a[1, 0] = [1, 0, 1]                                               # the same position with the sides swapped
    a[1, 1] = [1, 2, 0]
    non, key = EF.set_keys(a, 0)
    assert non.tolist() == [3, 3]
    assert key[0] == key[1]
    assert EF.decode(int(key[0]), kinds, "goal") in ("disc+disc | ring", "ring | disc+disc")
    b = a.copy()
    b[0, 0] = [1, 1, 0]                                               # one disc fewer: another set
    _n, kb = EF.set_keys(b, 0)
    assert kb[0] != key[0]


def test_the_fill_rows_are_joined_after_the_human_ones_and_never_enter_the_test_rows():
    h = {"g": np.array([0, 0, 1]), "ply": np.array([10, 11, 10]), "fit": np.array([True, True, False]), "K": np.zeros((3, 2)), "C": np.zeros((3, 4)), "state": None, "meta": {"k": 1}}
    f = {"g": np.array([0, 0]), "ply": np.array([2, 3]), "fit": np.array([True, True]), "K": np.zeros((2, 2)), "C": np.zeros((2, 0)), "state": None, "meta": {"k": 2}}
    c = EF.concat(h, f, 2)
    assert c["g"].tolist() == [0, 0, 1, 2, 2] and c["fit"].tolist() == [True, True, False, True, True]
    assert c["C"] is None
    assert not (set(c["g"][~c["fit"]].tolist()) & set(c["g"][2 * 0 + 3:].tolist()))          # a test row is never a fill row
