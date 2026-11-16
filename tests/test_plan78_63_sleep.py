"""Plan 78.63: the sleep that chooses its actions and wakes by a stop rule. The action interface (cut at its limit, resumable), the held-out sets disjoint from everything fitted, the computed gain a number
before the action runs, the stop rule on each of its reasons, the store kept only when the wake test says not worse, and the module naming no object of any game."""
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
from catspace.io import paths

SRC = Path(SL.__file__)
WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
TAG = "test_78_63"
NEEDS = [Path(paths.experiment("estimator_78_59", "full", "rows.npz")), Path(paths.experiment("odds_78_61", "full", "gen2", "record.json"))]
CFG = {"sleep_seconds": 30.0, "slice_seconds": 5.0, "seconds": 0.01, "max_plies": 8, "workers": 1, "pairs": 2}


def quiet(*_a, **_k):
    return None


@pytest.fixture(scope="module")
def fixed():
    if not all(p.exists() for p in NEEDS):
        pytest.skip("the 78.59 rows and the 78.61 generation 2 are not present")
    return SL.Fixed()


def fresh(fixed, index=1, store=None):
    root = SL.sandbox(TAG)
    if index == 1 and store is None and root.exists():
        shutil.rmtree(root)
    return SL.Sleep(TAG, index, dict(CFG), fixed, None, quiet, store)


def test_the_stop_rule_fires_on_each_of_its_reasons():
    slice_s, budget = 60.0, 600.0
    ok = dict(elapsed=100.0, budget=budget, slice_s=slice_s, free=50.0, sandbox_gb=0.1, best_corrected=1e-4, recent=[(0.0, 1.0), (60.0, 0.9), (100.0, 0.8)])
    assert SL.stop_rule(**ok) is None
    assert "time budget" in SL.stop_rule(**{**ok, "elapsed": 590.0})
    assert "disk limit" in SL.stop_rule(**{**ok, "free": 5.0})
    assert "sandbox" in SL.stop_rule(**{**ok, "sandbox_gb": 9.0})
    assert "expected to pay" in SL.stop_rule(**{**ok, "best_corrected": 1e-9})
    flat = [(0.0, 1.0)] + [(t, 0.8) for t in range(60, 500, 60)]
    assert "no recent gain" in SL.stop_rule(**{**ok, "elapsed": 480.0, "recent": flat})
    still = [(0.0, 1.0)] + [(t, 1.0 - 0.01 * t / 60) for t in range(60, 500, 60)]
    assert SL.stop_rule(**{**ok, "elapsed": 480.0, "recent": still}) is None                 # falling by far more than theta a second


def test_helpers_are_monotone_and_the_curve_reads_a_falling_loss():
    assert SL.p_ordered(1.0, 0.2) > SL.p_ordered(1.0, 0.8) > SL.p_ordered(0.0, 0.8)
    assert SL.p_ordered(0.0, 1.0) < 0.1 and SL.p_ordered(5.0, 0.1) > 0.99
    assert SL.soft_unordered(0.5) == 1.0 and SL.soft_unordered(0.02) == 0.0 and SL.soft_unordered(0.98) == 0.0 and 0 < SL.soft_unordered(0.2) < 1
    pts = [(100, 0.5 + 10 / 100), (200, 0.5 + 10 / 200), (400, 0.5 + 10 / 400)]
    assert abs(SL.curve_slope(pts) - 10.0) < 1e-6
    assert SL.curve_slope([(100, 0.7)]) == 0.0 and SL.curve_slope([(100, 0.6), (200, 0.7)]) == 0.0      # a rising loss has no slope to spend


def test_the_odds_head_terms_have_the_sign_of_what_a_unit_does():
    rng = np.random.default_rng(0)
    x = rng.integers(-1, 2, size=(600, 2)).astype(float)
    p = 1 / (1 + np.exp(-(2.0 * x[:, 0])))                          # only the first kind matters
    y = np.where(rng.random(600) < p, 0, 2)
    t = SL.kind_terms_of(SL.fit_odds(x, y), 2)
    assert t[0] > 0.5 and abs(t[1]) < 0.5
    draws = SL.bootstrap_terms(x, y, [str(i % 5) for i in range(600)], 40, 1)
    pairs = SL.pair_table(draws, ["a", "b"])
    assert len(pairs) == 1 and not pairs[0]["unordered"] and pairs[0]["share"] > 0.95


def test_the_yardstick_sets_are_disjoint_from_everything_fitted(fixed):
    sl = fresh(fixed)
    sl.prepare()
    y0 = sl.write_yardstick()
    assert set(fixed.test_games).isdisjoint(fixed.fit_order)
    assert fixed.rows["fit"][fixed.fit_rows(len(fixed.fit_order))].all() and not fixed.rows["fit"][fixed.test_rows].any()
    kept = {g["gid"] for g in sl.memory.games}
    assert set(sl.held_gids).isdisjoint(kept) and all(g.startswith("g2:") for g in sl.held_gids)
    assert all(SL.held_out_odds(g) for g in sl.held_gids) and not any(SL.held_out_odds(g) for g in kept if g.startswith("g2:"))
    # an action that plays and an action that fits leave the held-out sets as they were, and nothing it writes is in them
    sl.pool = None
    SL.OddsKind(sl.kinds[0]).run(sl, 1.0)
    SL.HumanFit().run(sl, 5.0)
    assert sl.human_n == 1000 and set(fixed.fit_order[:sl.human_n]).isdisjoint(fixed.test_games)
    assert {g["gid"] for g in sl.memory.games if g["gid"].startswith("s1:")}.isdisjoint(sl.held_gids)
    y1 = sl.write_yardstick()
    assert y0["human_held_out_digest"] == y1["human_held_out_digest"] and y0["odds_held_out_digest"] == y1["odds_held_out_digest"]


def test_an_action_is_cut_at_its_limit_keeps_what_it_wrote_and_resumes(fixed):
    import time
    sl = fresh(fixed)
    sl.prepare()
    t0 = time.time()
    res = SL.OddsKind(sl.kinds[0]).run(sl, 1.5)
    el = time.time() - t0
    lines = [json.loads(x) for x in (sl.dir / "games.jsonl").read_text().splitlines()]
    assert res["games"] == len(lines) > 0 and el < 1.5 + 1.0                                    # cut near the limit; every game it played is on the log
    gids = [g["gid"] for g in lines]
    assert len(set(gids)) == len(gids)
    again = SL.Sleep(TAG, 1, dict(CFG), fixed, None, quiet, None)                              # a new process reads the log back
    assert again.new_games == len(lines) and sum(again.counts.values()) == len(lines)
    again.prepare()
    SL.OddsKind(sl.kinds[0]).run(again, 1.0)
    gids2 = [json.loads(x)["gid"] for x in (again.dir / "games.jsonl").read_text().splitlines()]
    assert len(set(gids2)) == len(gids2) and set(gids) <= set(gids2) and len(gids2) > len(gids)


def test_a_later_start_differs_from_its_control_by_exactly_one_unit_of_the_kind(fixed):
    sl = fresh(fixed)
    sl.prepare()
    kind = sl.kinds[-1]
    ids = sl.sample_later(kind)
    assert len(ids) == 3 and ids[0].endswith(":even")
    base = O.units_of(sl.memory.registry[ids[0]]["text"])
    for sid in ids[1:]:
        s = sl.memory.registry[sid]
        gone = sorted(set(base) - set(O.units_of(s["text"])))
        assert len(gone) == 1 and gone[0][1] == kind and O.playable(s["text"])
    assert all(g in set(fixed.fit_order) for g in {int(i.split(":")[1]) for i in ids})            # sampled from fit games only


def test_the_computed_gain_is_a_number_before_the_action_runs(fixed):
    sl = fresh(fixed)
    sl.prepare()
    m = sl.measure()
    cands = sl.candidates(m)
    assert any(isinstance(c, SL.HumanFit) for c in cands) and {c.type for c in cands} >= {"odds_kind", "odds_later", "human_fit"}
    for c in cands:
        g = sl.computed_gain(c, CFG["slice_seconds"], m)
        assert all(np.isfinite(g[k]) for k in ("total", "seconds", "per_second")) and g["seconds"] > 0 and set(g["parts"])
    action, info = sl.choose(CFG["slice_seconds"], 0)
    assert action is not None and np.isfinite(info["computed"]["per_second"]) and np.isfinite(info["corrected"]) and info["ranked"]
    assert sl.history == []                                                                      # nothing had run when the numbers were made


def test_the_record_of_computed_against_observed_corrects_the_chooser(fixed):
    sl = fresh(fixed)
    assert sl.correction("odds_kind") == 1.0
    sl.record["types"]["odds_kind"] = {"tries": 3, "computed": 1.0, "observed": 0.2, "seconds": 10.0}
    assert abs(sl.correction("odds_kind") - 0.2) < 1e-9
    sl.record["types"]["odds_kind"] = {"tries": 3, "computed": 1.0, "observed": 100.0, "seconds": 10.0}
    assert sl.correction("odds_kind") == 10.0                                                    # bounded
    sl.record["types"]["odds_kind"] = {"tries": 1, "computed": 1.0, "observed": 100.0, "seconds": 10.0}
    assert sl.correction("odds_kind") == 1.0                                                     # one try is not a record


def test_the_new_store_is_kept_only_when_the_wake_test_says_not_worse(fixed, monkeypatch):
    for score, lo, hi, want in ((0.62, 0.55, 0.70, True), (0.50, 0.42, 0.58, True), (0.46, 0.40, 0.49, False), (0.30, 0.2, 0.4, False)):
        keep, why = SL.wake_decision({"score": score, "lo": lo, "hi": hi})
        assert keep is want and "upper end" in why
    for hi, want in ((0.7, True), (0.4, False)):
        sl = fresh(fixed)
        sl.prepare()
        monkeypatch.setattr(SL, "match", lambda *a, _hi=hi, **k: {"label": "x", "score": _hi - 0.05, "lo": _hi - 0.1, "hi": _hi, "w": 1, "d": 1, "l": 1})
        out = sl.finish({"final": sl.measure()}, 2)
        assert out["kept"] is want
        if want:
            assert out["store"]["from"].startswith("sleep 1") and out["store"]["memory"] and out["store"]["player"] == out["new_player"]
        else:
            assert out["store"] is not None and out["store"]["from"] == sl.store["from"] and out["store"]["memory"] == sl.store["memory"] and out["store"]["player"] == sl.player


def test_a_whole_short_sleep_runs_wakes_for_a_stated_reason_and_prints_computed_beside_observed(fixed, monkeypatch):
    lines = []
    root = SL.sandbox(TAG)
    if root.exists():
        shutil.rmtree(root)
    sl = SL.Sleep(TAG, 1, {**CFG, "sleep_seconds": 12.0, "slice_seconds": 3.0}, fixed, None, lines.append, None)
    res = sl.run()
    assert res["reason"] and res["actions"] >= 1
    acts = [json.loads(x) for x in (sl.dir / "log.jsonl").read_text().splitlines()]
    assert all(np.isfinite(a["computed_per_second"]) and "observed_per_second" in a for a in acts)
    assert any("computed" in x and "observed" in x and "action" in x for x in lines) and any("wakes after" in x for x in lines)
    assert (sl.root / "store" / "chooser_record.json").exists()


def test_the_module_names_no_object_of_any_game():
    text = SRC.read_text()
    toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    names = [t.string for t in toks if t.type == tokenize.NAME]
    assert [n for n in names if re.fullmatch(WORDS, n, flags=re.I)] == []
    for t in toks:
        if t.type in (tokenize.COMMENT, tokenize.STRING):
            assert re.search(WORDS, t.string, flags=re.I) is None, (t.start, t.string[:60])
