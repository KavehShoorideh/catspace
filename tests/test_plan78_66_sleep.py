"""Plan 78.66: the sleep's yardstick made of what the player reads. The yardstick is a function of the worths alone (two held-out sets scored by the record turned into a result prediction), the own-player
held-out games are fresh and disjoint from every game an action adds and never fitted on, an action that cannot change the record has computed gain zero and is never chosen, an action's observed gain enters
the chooser's record only after three slices, and the module names no object of any game."""
import io
import json
import re
import shutil
import tokenize
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import odds_measure as O
from catspace.incubator.analyst.discovery import sleep_loop as SL
from catspace.incubator.analyst.discovery import sleep_loop66 as S6
from catspace.io import paths

SRC = Path(S6.__file__)
WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
TAG = "test_78_66"
NEEDS = [Path(paths.experiment("estimator_78_59", "full", "rows.npz")), Path(paths.experiment("odds_78_61", "full", "gen2", "record.json"))]
CFG = {"sleep_seconds": 30.0, "slice_seconds": 3.0, "seconds": 0.01, "max_plies": 8, "workers": 1, "pairs": 2, "own_games": 12}


def quiet(*_a, **_k):
    return None


@pytest.fixture(scope="module")
def fixed():
    if not all(p.exists() for p in NEEDS):
        pytest.skip("the 78.59 rows and the 78.61 generation 2 are not present")
    return SL.Fixed()


def fresh(fixed):
    root = S6.sandbox(TAG)
    if root.exists():
        shutil.rmtree(root)
    sl = S6.Sleep66(TAG, 1, dict(CFG), fixed, None, quiet, None)
    sl.prepare()
    return sl


def test_the_yardstick_is_a_function_of_the_worths_alone(fixed):
    sl = fresh(fixed)
    y = sl.yard
    w = sl.measure()["worth"]
    a, b = y.y_of(w), y.y_of(w.copy())
    assert a == b and abs(a["Y"] - 0.5 * (a["own"] / y.base["own"] + a["human"] / y.base["human"])) < 1e-12
    big = np.array(w, dtype=float)
    big[0] += 500
    assert y.y_of(big)["Y"] != a["Y"]                                                          # the worths move it
    # nothing the player does not read: fitting the human result head, or any further human fit game, leaves it as it was
    before = sl.measure()["Y"]
    S6.SL.HumanFit().run(sl, 5.0)
    sl.measure_cache = None
    assert sl.measure()["Y"] == before


def test_the_curvature_is_not_negative_and_a_kind_nobody_holds_has_none(fixed):
    sl = fresh(fixed)
    w = sl.measure()["worth"]
    curv = sl.measure()["curv"]
    assert all(c >= 0 for c in curv) and any(c > 0 for c in curv)
    y = sl.yard
    z = zero_column(y, 0)
    assert z.curvature(0, w) == 0.0


def zero_column(y, k):
    """A yardstick whose held-out counts of kind k are all zero: moving that kind's worth cannot move it."""
    own_x, hum_x = y.own_x.copy(), y.hum_x.copy()
    own_x[:, k] = 0
    hum_x[:, k] = 0
    z = S6.Yardstick.__new__(S6.Yardstick)
    z.own_x, z.own_y, z.hum_x, z.hum_y = own_x, y.own_y, hum_x, y.hum_y
    z.model_own, z.model_hum, z.base = y.model_own, y.model_hum, y.base
    return z


def test_own_player_held_out_games_are_fresh_disjoint_and_never_fitted_on(fixed):
    sl = fresh(fixed)
    held = {g["gid"] for g in sl.own_set}
    assert len(held) == CFG["own_games"] and all(g.startswith("h1:") for g in held)
    assert held.isdisjoint({g["gid"] for g in sl.memory.games})
    assert set(fixed.test_games).isdisjoint(fixed.fit_order)
    digest = sl.write_yardstick()
    coef_own, coef_hum = sl.yard.model_own.coef_.copy(), sl.yard.model_hum.coef_.copy()
    sl.pool = None
    S6.SL.OddsKind(sl.kinds[0]).run(sl, 1.0)
    sl.measure_cache = None
    sl.measure()
    added = {g["gid"] for g in sl.memory.games if g["gid"].startswith("s1:")}
    assert added and added.isdisjoint(held) and held.isdisjoint({g["gid"] for g in sl.memory.games})
    assert sl.write_yardstick()["own_held_out_digest"] == digest["own_held_out_digest"]
    assert np.array_equal(coef_own, sl.yard.model_own.coef_) and np.array_equal(coef_hum, sl.yard.model_hum.coef_)       # the frozen maps did not move
    # the own-player games are the sleep's own player, as seen in the file
    lines = [json.loads(x) for x in (sl.dir / "heldout_own.jsonl").read_text().splitlines()]
    assert {g["gid"] for g in lines} == held


def test_an_action_that_cannot_change_the_record_has_gain_zero_and_is_never_chosen(fixed, monkeypatch):
    sl = fresh(fixed)
    m = sl.measure()
    assert not sl.changes_record(S6.SL.HumanFit()) and all(sl.changes_record(a) for a in sl.candidates(m) if a.type != "human_fit")
    g = sl.computed_gain(S6.SL.HumanFit(), 60.0, m)
    assert g["total"] == 0.0 and g["per_second"] == 0.0
    assert any(isinstance(a, S6.SL.HumanFit) for a in sl.candidates(m))                           # it is printed as a candidate with gain zero
    monkeypatch.setattr(SL, "EXPLORE", 1.0)                                                      # every choice an exploration of the type tried longest ago: still never it
    for step in range(30):
        a, info = sl.choose(60.0, step)
        assert a is not None and a.type != "human_fit"
        sl.last_tried[a.type] = step


def test_an_observed_gain_enters_the_chooser_only_after_three_slices(fixed, monkeypatch):
    for stop_after, want_tries in ((2, 0), (3, 1)):
        shutil.rmtree(S6.sandbox(TAG), ignore_errors=True)
        real = S6.Sleep66.stop_reason

        def stop(self, elapsed, info, _s=stop_after):
            return "test stop" if len(self.history) >= _s else None
        monkeypatch.setattr(S6.Sleep66, "stop_reason", stop)
        sl2 = S6.Sleep66(TAG, 1, {**CFG, "sleep_seconds": 60.0, "slice_seconds": 2.0}, fixed, None, quiet, None)
        res = sl2.run()
        monkeypatch.setattr(S6.Sleep66, "stop_reason", real)
        assert res["actions"] == stop_after
        tries = sum(t["tries"] for t in sl2.record["types"].values())
        assert tries == want_tries and len(sl2.bouts) == want_tries
        if want_tries:
            assert all(b["slices"] == S6.SLICES_PER_BOUT for b in sl2.bouts)
        shutil.rmtree(S6.sandbox(TAG), ignore_errors=True)


def test_the_module_names_no_object_of_any_game():
    text = SRC.read_text()
    toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    names = [t.string for t in toks if t.type == tokenize.NAME]
    assert [n for n in names if re.fullmatch(WORDS, n, flags=re.I)] == []
    for t in toks:
        if t.type in (tokenize.COMMENT, tokenize.STRING):
            assert re.search(WORDS, t.string, flags=re.I) is None, (t.start, t.string[:60])
