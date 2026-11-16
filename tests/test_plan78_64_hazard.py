"""PLAN 78.64 stage B: the hazard reproduces a known rate on made-up rows, a game that ends first is cut (not counted as no event), a taking with no settled position after it is cut, the fit and the
test games are disjoint, the step table counts the steps by bin and parity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import events_hazard as EH
from catspace.io import paths


def _rows(g, ply, mover):
    return {"g": np.array(g), "ply": np.array(ply), "mover": np.array(mover, np.int8)}


def _res(n, events=None, kept=None, cut=None):
    return {"n": n, "events": events or {}, "kept": kept or {}, "cut": cut or {}}


ROLES = ["north", "south"]


def test_the_step_table_counts_the_steps_by_bin_and_parity():
    t = EH.TAB
    assert t[1].sum() == 1 and t[1][0][0] == 1                       # d = 1: the first bin, odd
    assert t[4][0].tolist() == [1.0, 1.0] and t[4][1].tolist() == [1.0, 1.0]            # d = 1..4: pairs 1 and 2
    assert t[EH.MAXD].sum() == EH.MAXD
    ds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 24, 25, 40, 41, 80, 81, 200]
    assert EH.bin_of_d(np.array(ds)).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]


def test_a_game_that_ends_first_is_cut_and_a_taking_with_no_settled_position_is_cut_not_an_event():
    rows = _rows([0, 0, 0], [10, 11, 12], [0, 1, 0])
    # game of 20 steps; the north side takes at step 14 (the event); nothing else happens
    res = [_res(20, events={("seen", "north", "disc"): [14]})]
    D, has, cut = EH.targets(("seen", "disc"), "us", ROLES, rows, res)
    # row 0 (ply 10, north to move): us = north: the event at step 14 -> 4 steps before it
    assert D[0] == 4 and has[0] and not cut[0]
    # row 2 (ply 12, north to move): 2 steps before it
    assert D[2] == 2 and has[2]
    # row 1 (ply 11, south to move): `us` is south, which never takes: the record ends first, all 9 steps are non-event and it is NOT an event
    assert D[1] == 20 - 11 and not has[1] and not cut[1]
    # `them` at row 1 is north: the event at step 14
    D2, has2, _c = EH.targets(("seen", "disc"), "them", ROLES, rows, res)
    assert D2[1] == 3 and has2[1] and not has2[0]
    # a taking with no settled position after it: cut at that step, not an event, and the steps before it are non-event steps
    res2 = [_res(20, kept={}, cut={("seen", "north", "disc"): [14]})]
    D3, has3, cut3 = EH.targets(("seen", "disc"), "us", ROLES, rows, res2)
    assert D3[0] == 4 and not has3[0] and cut3[0]


def _geometric_rows(n, rate, rng, own_odd=True):
    """Rows whose first event comes at each of the head's own steps (d = 1, 3, 5, ...) with chance `rate`, or is cut at T - ply."""
    T = rng.integers(30, 60, n)
    k = rng.geometric(rate, n)                                         # the number of own steps up to and including the event
    d_ev = 2 * k - 1
    has = d_ev <= T
    D = np.where(has, d_ev - 1, T)
    return D.astype(np.int64), has, np.zeros(n, bool)


def test_the_hazard_reproduces_a_known_rate_on_made_up_rows():
    rng = np.random.default_rng(1)
    D, has, cut = _geometric_rows(40000, 0.2, rng)
    h = EH.Head(("seen",), "us", D, has, cut)
    h.activate(np.ones(len(D), bool))
    assert h.active.tolist() == [True, False]                           # every event is on the side's own step
    b = h.base_logit(np.arange(len(D)))
    p = 1 / (1 + np.exp(-b[:, 0]))
    assert np.allclose(p[:4], 0.2, atol=0.02), p                       # the early bins hold many steps
    # a feature that doubles the odds is recovered with its sign, and the log loss falls
    x = rng.normal(size=(len(D), 1)).astype(np.float32)
    odds = 0.25 * np.exp(1.0 * x[:, 0])
    pr = odds / (1 + odds)
    T = rng.integers(30, 60, len(D))
    k = rng.geometric(np.clip(pr, 1e-3, 0.99))
    d_ev = 2 * k - 1
    has2 = d_ev <= T
    D2 = np.where(has2, d_ev - 1, T).astype(np.int64)
    h2 = EH.Head(("seen",), "us", D2, has2, np.zeros(len(D2), bool))
    h2.activate(np.ones(len(D2), bool))
    sel = np.arange(len(D2))
    mean, scale = EH.standardise(x, sel)
    m = EH.fit_hazard(h2, x, sel, 1e-6, mean, scale, iters=100)
    assert m["w"][0] > 0.2
    tot_h, _ = EH.eval_head(h2, x, m, sel)
    tot_b, _ = EH.eval_head(h2, None, None, sel, h2.base_logit(sel))
    assert tot_h.sum() < tot_b.sum()


def test_the_flag_without_events_is_out_of_the_risk_set_and_the_event_step_is_counted_once():
    D = np.array([2, 4, 0]); has = np.array([True, False, True]); cut = np.zeros(3, bool)
    h = EH.Head(("seen",), "us", D, has, cut)
    h.activate(np.ones(3, bool))
    assert h.steps().tolist() == [1 + 1, 2, 0 + 1]                      # own steps among d = 1..D (odd d) plus the event step
    assert h.active.tolist() == [True, False]


def test_fit_and_test_games_are_disjoint_in_the_rows_of_the_sample():
    p = Path(paths.experiment("estimator_78_59", "full", "rows.npz"))
    if not p.exists():
        pytest.skip("the 78.59 rows are not in this checkout")
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    rows = HD.load_rows(p)
    fit_games = set(rows["game"][rows["fit"]].tolist())
    test_games = set(rows["game"][~rows["fit"]].tolist())
    assert fit_games and test_games and not (fit_games & test_games)
