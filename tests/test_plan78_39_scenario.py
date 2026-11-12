"""Plan 78.39 stage 1 through the Python surface: the goal's target read from the description, the holes, the supply and the scenarios of a position (a report, not a decision step)."""
import json

import cs_engine as CE
from catspace.incubator.analyst.discovery import game_store as GS

GAME = "chess"


def _args():
    raw = GS.vocabulary_file(GAME, "scenario")
    vocab = {**GS.vocabulary(GAME, "scenario"), "scenario": raw["scenario"]}
    return GS.spec_texts(GAME, "scenario")[GAME], json.dumps(vocab)


def _read(fen=None, **kw):
    spec, vocab = _args()
    return CE.scenario(GAME, spec, vocab, fen=fen, **kw)


def test_a_cornered_target_with_two_holes_and_one_unit_that_closes_both():
    r = _read("7k/5K2/8/8/8/8/8/R7 w - - 0 1")
    assert not r["unreadable"] and not r["goal_holds"]
    assert len(r["holes"]) == 2
    assert r["scenarios"] and r["safe"] is False
    assert min(len(s["placements"]) if "placements" in s else 1 for s in r["scenarios"]) == 1
    print("OBS 78.39 report seconds", round(r["seconds"], 4), "holes", len(r["holes"]), "scenarios", len(r["scenarios"]))


def test_no_scenario_reads_as_safe():
    r = _read("7k/5K2/8/8/8/8/p7/8 w - - 0 1", k=2)          # our side holds nothing that could close the hole on the target's own location
    assert r["holes"] and r["scenarios"] == [] and r["safe"] is True


def test_the_other_side_is_read_with_role():
    r = _read("7k/5K2/8/8/8/8/8/R7 w - - 0 1", role="other")
    assert r["role"] != _read("7k/5K2/8/8/8/8/8/R7 w - - 0 1")["role"]
    assert r["safe"] is True          # the other side has no unit to supply


def test_the_report_is_the_same_twice():
    a, b = _read("7k/5K2/8/8/8/8/8/R7 w - - 0 1"), _read("7k/5K2/8/8/8/8/8/R7 w - - 0 1")
    for k in ("holes", "covered", "scenarios", "safe"):
        assert a[k] == b[k]
