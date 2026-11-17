"""Plan 78.73: the verdict and the responses of the concrete threat list (catspace/incubator/analyst/discovery/threat_concrete.py; the checks in threat_concrete_measure.py). SYSTEM_SPEC 8, the threat list.
A: units of one kind taken and lost cancel exactly before the net and the resolution are computed. B: a response carries its own end, the ones that lose nothing come first, one that loses more than the threat
would take is dropped, and a move the sweep's small fuse cuts is read again (the plain retreat of the attacked unit was cut and dropped in 78.72). The synthetic lines are built here; the cause of B is shown on
the real position it was found in."""
from __future__ import annotations

import cs_engine as CE

from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import threat_concrete_measure as TCM

WORTH = {"q": 4343, "b": 1878, "n": 1661, "r": 1605, "p": 1045}
HALF = {"q": 120.0, "b": 606.0, "n": 628.0, "r": 534.0, "p": 508.0}
REC = TC.Record(WORTH, HALF, "synthetic")
PLACES = sorted(f + r for f in "abcdefgh" for r in "12345678")


def _line(events, value, end=0, cut=False, depth=6):
    return {"events": events, "value": value, "end": end, "cut": cut, "depth": depth, "names": ["m1", "m2", "m3"], "settled": 3, "nodes": 10, "first": (1, 2), "lead": None, "prefix": []}


def test_like_for_like_cancels_exactly_and_the_rest_is_kept():
    assert TC.cancel_like(["n"], ["n"]) == ([], [], 1)
    assert TC.cancel_like(["p", "p"], ["p", "p", "n"]) == ([], ["n"], 2)
    assert TC.cancel_like(["q", "p"], ["p"]) == (["q"], [], 1)
    assert TC.cancel_like([], []) == ([], [], 0)


def test_the_resolution_is_built_only_from_the_kinds_left():
    # the taker takes three of one kind and loses two of it: one is left; the resolution is that kind's alone (before 78.73 it grew with the length of the line)
    v = TC.verdict_of(_line([(False, "p"), (False, "p"), (False, "p"), (True, "p"), (True, "p")], 1045), REC)
    assert v["left_won"] == ["p"] and v["left_lost"] == [] and v["cancelled"] == 2
    assert v["resolution"] == HALF["p"] and v["net"] == WORTH["p"] and v["verdict"] == "CONFIRMED"
    old_resolution = REC.resolution(["p"] * 5)
    assert old_resolution > v["net"] > v["resolution"]                                        # the 78.72 resolution would have called it an exchange


def test_a_trade_of_equal_kinds_is_an_exchange_with_nothing_left_to_tell_apart():
    v = TC.verdict_of(_line([(False, "n"), (True, "n"), (False, "p"), (True, "p")], 0), REC)
    assert v["verdict"] == "EXCHANGE" and v["resolution"] == 0.0 and v["net"] == 0 and v["left_won"] == [] and v["left_lost"] == []


def test_the_end_of_the_game_still_decides_whatever_cancels():
    assert TC.verdict_of(_line([(False, "p"), (True, "p")], 1 << 40, end=1), REC)["verdict"] == "CONFIRMED"
    assert TC.verdict_of(_line([(False, "p")], 1045, cut=True), REC)["verdict"] == "UNREAD"


def _after(events, value, takes, cut=False):
    return {"whole": dict(_line(events, value), cut=cut), "takes": takes}


def test_a_response_carries_its_own_end_and_the_ones_that_lose_nothing_sort_first():
    passed = _line([(False, "q")], 4343)
    free = TC.response_of(_after([(False, "p"), (True, "p")], 0, []), passed, REC, "the unit moves", "t", 4343)
    cheap = TC.response_of(_after([(False, "p")], 1045, []), passed, REC, "a unit is put between", "t", 4343)
    assert free["state"] == "KEPT" and free["loses_nothing"] and free["resp_loses"] == [] and free["cancelled"] == 1
    assert cheap["state"] == "KEPT" and not cheap["loses_nothing"] and cheap["resp_loses"] == ["p"] and cheap["resp_takes"] == []
    order = sorted([cheap, free], key=lambda x: (not x["loses_nothing"], x["net"], x["names"]))
    assert order[0] is free


def test_a_response_that_loses_more_than_the_threat_would_take_is_dropped():
    passed = _line([(False, "q")], 4343)                                                       # the taker could count on a unit worth 4343 if the owner passed
    dear = TC.response_of(_after([(False, "n")], 1661, []), passed, REC, "a counter threat", "t", 1045)       # but the threat on this unit is worth 1045
    assert dear["state"] == "COSTS MORE THAN THE THREAT"
    fair = TC.response_of(_after([(False, "p")], 1045, []), passed, REC, "a counter threat", "t", 1045)       # losing what the threat would have taken is not more
    assert fair["state"] == "KEPT"
    none = TC.response_of(_after([(False, "n")], 1661, []), passed, REC, "a counter threat", "t")             # no row net given: the old rule only
    assert none["state"] == "KEPT"


def test_the_sweeps_fuse_and_depth_are_the_lines_own_to_set():
    lines = TC.Lines(REC, threads=1, sweep_depth=6, sweep_fuse=40000)
    assert (lines.sweep_depth, lines.sweep_fuse) == (6, 40000)
    d = TC.Lines(REC, threads=1)
    assert (d.sweep_depth, d.sweep_fuse) == (TC.SWEEP_DEPTH, TC.SWEEP_FUSE)


def test_the_plain_retreat_was_cut_by_the_sweeps_small_fuse_and_is_read_with_the_fuse_of_a_line():
    """The cause found in B (position 15 of the 78.72 print): the quiet retreat of the attacked unit costs the most nodes (no capture to prune on); with the sweep's fuse the line is CUT, the response is UNREAD
    and was dropped; with the fuse of a line it is read and kept."""
    moves = ("e2e4 e7e5 f2f4 f8c5 g1f3 e5f4 f1c4 d8e7 b1c3 b8c6 a2a3 c6d4 b2b4 d4f3 d1f3 c5d6 e1g1 d6e5 c1b2 d7d6 d2d3 g8f6 a1b1 c8g4").split()
    q = [(0, 7, 0, 0, 0, 0, PLACES.index("f3"))]
    small = CE.read_lines("chess", [(None, moves)], q, WORTH, 4, TC.SWEEP_FUSE, 7, 1500000, 1)[0]["afters"]
    big = CE.read_lines("chess", [(None, moves)], q, WORTH, 4, TC.FUSE, 7, 1500000, 1)[0]["afters"]
    retreat_small = [a["whole"] for a in small if a["whole"]["names"][0] == "f3f2"]
    retreat_big = [a["whole"] for a in big if a["whole"]["names"][0] == "f3f2"]
    assert retreat_small and retreat_small[0]["cut"]
    assert retreat_big and not retreat_big[0]["cut"] and retreat_big[0]["value"] == 0
    v = TC.response_of(next(a for a in big if a["whole"]["names"][0] == "f3f2"), CE.read_lines("chess", [(None, moves)], [(0, 8, 1, 1, 0, 0, 0)], WORTH, 4, TC.SWEEP_FUSE * 4, 7, 1500000, 1)[0]["reads"][0], REC, "x", "t", 2465)
    assert v["state"] == "KEPT" and v["loses_nothing"]


def test_the_measure_commands_exist():
    assert callable(TCM.cmd_verdicts) and callable(TCM.cmd_replies)
