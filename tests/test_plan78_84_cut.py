"""Plan 78.84: the tie set cut by the line reading (catspace/incubator/analyst/discovery/symbols_play.py, tie_cut_measure.py). SYSTEM_SPEC 5 (the reasoner decides, the hint orders only what it holds equal), 7, 8.

The cut drops a tied move whose line gives up a unit for nothing while another tied move's line does not, and never returns an empty set; the players with the cut charge its time with the pick; part one's measure writes under
symbols_play_78_84 and names no game object."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import symbols_play as SP
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import tie_cut_measure as TCM
from catspace.incubator.analyst.discovery import tiebreak_measure as TM

HAVE = (SP.src_root() / "reasons.json").exists() and (SP.src_root() / "fits.pkl").exists()
needs_run = pytest.mark.skipif(not HAVE, reason="needs the 78.81 run's reasons.json and fits.pkl")


def verdict(kind: str = "give", **kw) -> dict:
    base = {"give": dict(verdict="REFUTED", end=0, left_lost=["x"], left_won=[], net_search=-1000, resolution=500.0),
            "even": dict(verdict="EXCHANGE", end=0, left_lost=[], left_won=[], net_search=0, resolution=0.0)}[kind]
    return dict(base, **kw)


def test_the_rule_for_a_line_that_gives_up_a_unit_for_nothing():
    assert SP.gives_up(verdict("give"))
    assert not SP.gives_up(None)
    assert not SP.gives_up(verdict("even"))
    assert not SP.gives_up(verdict("give", verdict="UNREAD"))                        # a cut line claims nothing
    assert not SP.gives_up(verdict("give", end=2))                                   # the game's end is not a unit
    assert not SP.gives_up(verdict("give", left_won=["y"]))                          # something taken in return: not for nothing
    assert not SP.gives_up(verdict("give", left_lost=[]))


def test_the_line_s_own_value_can_be_asked_to_agree():
    far = verdict("give", net_search=0)                                              # the events show a loss, the line search's own number does not
    assert SP.gives_up(far) and not SP.gives_up(far, own_value=True)
    assert SP.gives_up(verdict("give"), own_value=True)
    assert not SP.gives_up(verdict("give", net_search=-400), own_value=True)         # under the resolution: not a unit


def test_the_cut_drops_the_moves_that_give_up_a_unit_and_keeps_the_order():
    tied = ["a", "b", "c", "d"]
    v = {"a": verdict("give"), "b": verdict("even"), "c": verdict("give"), "d": None}
    assert SP.cut_of(v, tied) == ["b", "d"]
    assert SP.cut_of({m: verdict("give") for m in tied}, tied) == tied               # every line gives up a unit: nothing to choose by, the set stays whole
    assert SP.cut_of({m: verdict("even") for m in tied}, tied) == tied
    assert SP.cut_of({}, tied) == tied                                               # no line read at all
    assert SP.cut_of(v, tied, own_value=True) == ["b", "d"]


class StubCutter:
    def __init__(self, bad):
        self.bad, self.asked = set(bad), []

    def reads(self, moves, tied):
        self.asked.append((list(moves), list(tied)))
        return {m: verdict("give" if m in self.bad else "even") for m in tied}


def test_sound_ties_is_a_subset_never_empty_and_reads_nothing_for_one_move(monkeypatch):
    st = StubCutter({"b"})
    monkeypatch.setattr(SP, "_CUTTER", st)
    assert SP.sound_ties({"moves": ["x"]}, ["a", "b", "c"]) == ["a", "c"]
    assert st.asked == [(["x"], ["a", "b", "c"])]
    assert SP.sound_ties({"moves": ["x"]}, ["a"]) == ["a"] and len(st.asked) == 1
    monkeypatch.setattr(SP, "_CUTTER", StubCutter({"a", "b", "c"}))
    assert SP.sound_ties({"moves": []}, ["a", "b", "c"]) == ["a", "b", "c"]


def test_the_cut_players_are_registered_and_their_windows_are_the_bands():
    for p in ("cutreasons", "cutonly", "cutmidfew"):
        assert p in SP.PLAYERS and p in SP.MODE_PLAYERS and SP.KIND[p] in SP.CUT_KINDS and SP.KIND[p] in SP.MODE
    band = {n: (lo, hi) for n, lo, hi in LM.BANDS}
    assert SP.WINDOW["sy_cutmidfew"] == (band["few"][0], band["mid"][1]) and "sy_cutreasons" not in SP.WINDOW and "sy_cutonly" not in SP.WINDOW
    assert SP.MODE["sy_cutonly"] == "first" and SP.MODE["sy_cutreasons"] == "reasons"
    assert not SP.MoverReasons({"kind": SP.KIND["reasons"], "worth": {}}, 0.01, "g", "/x").cut
    assert SP.MoverReasons({"kind": SP.KIND["cutonly"], "worth": {}}, 0.01, "g", "/x").cut
    assert SP.MoverReasons({"kind": SP.KIND["reasons"], "worth": {}, "cut": True}, 0.01, "g", "/x").cut


def test_the_literal_cut_kind_reads_the_events_alone_and_the_others_ask_the_line_s_value_to_agree(monkeypatch):
    assert "evreasons" in SP.PLAYERS and SP.KIND["evreasons"] in SP.CUT_KINDS and SP.EVENTS_ONLY == (SP.KIND["evreasons"],)
    assert not SP.MoverReasons({"kind": SP.KIND["evreasons"], "worth": {}}, 0.01, "g", "/x").cut_value
    assert all(SP.MoverReasons({"kind": SP.KIND[p], "worth": {}}, 0.01, "g", "/x").cut_value for p in ("cutreasons", "cutonly", "cutmidfew"))
    far = verdict("give", net_search=0)                                              # events show a loss, the line's own number does not
    seen = []

    class Far(StubCutter):
        def reads(self, moves, tied):
            return {m: (far if m == "m1" else verdict("even")) for m in tied}
    for kind, dropped in (("evreasons", 1), ("cutreasons", 0)):
        m, _rd = mover(SP.KIND[kind], monkeypatch, set())
        monkeypatch.setattr(SP, "_CUTTER", Far(set()))
        m.choose(None, ["x"], [])
        seen.append(m.stats["cut_log"][0][2])
    assert seen == [1, 0]


class StubReader:
    def __init__(self, units: int = 20):
        self.units, self.picked = units, []

    def units_of(self, moves):
        return self.units

    def pick(self, moves, tied, mode):
        self.picked.append(list(tied))
        return tied[-1], {"scores": {m: 0.0 for m in tied}, "carried": {m: [] for m in tied}, "takes": None, "kinds": None}


def mover(kind: str, monkeypatch, bad, units: int = 20, charge: bool = False):
    rd = StubReader(units)
    monkeypatch.setattr(SP, "_READER", rd)
    monkeypatch.setattr(SP, "_CUTTER", StubCutter(bad))
    monkeypatch.setattr(SP.CE, "solve_equals_batch", lambda *a, **k: ["row"])
    monkeypatch.setattr(SP.PM, "equal_set", lambda row: {"move": "m0", "equals": ["m0", "m1", "m2", "m3"], "depth": [1, 1]})
    return SP.MoverReasons({"kind": kind, "worth": {}, "charge": charge}, 0.2, "g", "/nonexistent/trace"), rd


def test_the_cut_alone_plays_the_search_s_first_move_among_the_rest(monkeypatch):
    m, rd = mover(SP.KIND["cutonly"], monkeypatch, {"m1"})
    pick, _d = m.choose(None, ["x"], [])
    assert pick == "m0" and rd.picked == []                                          # the first move is sound: it stays, the reasons are not asked
    assert m.stats["cut_log"][0][1:4] == [4, 1, 0]
    m, rd = mover(SP.KIND["cutonly"], monkeypatch, {"m0", "m2"})
    pick, _d = m.choose(None, ["x"], [])
    assert pick == "m1"                                                              # the first move is dropped: the first of the rest in the tie order
    assert m.stats["cut_log"][0][1:4] == [4, 2, 1] and m.stats["changed"] == 1
    assert m.stats["cases"][0]["pick"] == "m1" and m.stats["cases"][0]["first"] == "m0"


def test_cut_and_reasons_pick_among_the_moves_left_and_log_what_was_dropped(monkeypatch):
    m, rd = mover(SP.KIND["cutreasons"], monkeypatch, {"m3", "m0"})
    pick, _d = m.choose(None, ["x"], [])
    assert rd.picked == [["m1", "m2"]] and pick == "m2"                              # the reasons see only the moves that remain
    assert m.stats["cut_log"][0][1:4] == [4, 2, 1] and len(m.stats["dlog"]) == 1
    m, rd = mover(SP.KIND["cutreasons"], monkeypatch, {"m1", "m2", "m3"})            # one move left: it is played, nothing is asked of the reasons
    pick, _d = m.choose(None, ["x"], [])
    assert pick == "m0" and rd.picked == []


def test_the_window_of_the_cut_and_pick_player_leaves_the_control_outside(monkeypatch):
    m, rd = mover(SP.KIND["cutmidfew"], monkeypatch, {"m0"}, units=17)               # many units: the control's first move, no cut, no pick
    pick, _d = m.choose(None, ["x"], [])
    assert pick == "m0" and rd.picked == [] and m.stats["cut_log"] == [] and m.stats["ties"] == 1 and m.stats["ties_in"] == 0
    m, rd = mover(SP.KIND["cutmidfew"], monkeypatch, {"m0"}, units=16)
    pick, _d = m.choose(None, ["x"], [])
    assert pick in ("m1", "m2", "m3") and rd.picked == [["m1", "m2", "m3"]] and len(m.stats["cut_log"]) == 1


def test_the_time_of_the_cut_is_charged_with_the_pick(monkeypatch):
    m, _rd = mover(SP.KIND["cutreasons"], monkeypatch, {"m1"}, charge=True)
    assert m.search_clock() == 0.2                                                   # nothing read yet
    m.choose(None, ["x"], [])
    assert len(m.costs) == 1 and m.costs[0] >= m.stats["cut_log"][0][4] / 1000 - 1e-6   # the cost holds the cut's time
    assert m.search_clock() == TM.reading_clock(0.2, m.costs)
    m2, _ = mover(SP.KIND["cutreasons"], monkeypatch, {"m1"}, charge=False)
    m2.costs.append(0.1)
    assert m2.search_clock() == 0.2


def test_the_output_folder_is_this_plans_and_nothing_is_written_to_the_store():
    assert SP.OUT == ("symbols_play_78_84",) and TCM.OUT == ("symbols_play_78_84",)
    assert "78_84" in str(SP.out_root("full")) and "78_84" in str(TCM.out_root("full"))
    src = Path(TCM.__file__).read_text()
    assert "rules/learned" not in src.replace("rules/learned or data/derived", "") and "data/derived" not in src.replace("rules/learned or data/derived", "")


def test_the_measure_names_no_game_object():
    words = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
    for mod in (TCM,):
        hit = re.search(words, Path(mod.__file__).read_text(), flags=re.I)
        assert hit is None, (mod.__name__, hit.group(0) if hit else None)


def test_the_decision_rule_orders_by_the_hash_of_the_moves():
    a, b = TCM.key_of(["p", "q"]), TCM.key_of(["p", "r"])
    assert a != b and a == TCM.key_of(["p", "q"]) and len(a) == 40


@needs_run
def test_the_line_reading_reads_every_tied_move_and_agrees_with_the_matches_record():
    from catspace.incubator.analyst.discovery import odds_measure as OM
    cut = SP.LineCut()
    assert cut.record.worth == OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")["worth"]       # the record the matches judge by
    tied = ["a2a3", "b2b4", "g1f3"]
    v = cut.reads(["e2e4", "e7e5"], tied)
    assert set(v) == set(tied) and all(x is not None and x["verdict"] in TC.VERDICTS for x in v.values())
    assert SP.sound_ties({"moves": ["e2e4", "e7e5"]}, tied) and set(SP.sound_ties({"moves": ["e2e4", "e7e5"]}, tied)) <= set(tied)


@needs_run
def test_a_two_pair_match_with_the_cut_player_logs_each_cut(tmp_path):
    specs = TM.match_specs("cutonly", "control", "A", 2, 0.05, tmp_path / "scratch", True, SP.KIND)
    for s in specs:
        s["max_plies"] = 40
    games = TM.run_games(specs, 1, tmp_path / "g.jsonl", lambda s: None, play=SP.play82)
    n_cut = 0
    for g in games:
        st = g["stats"][TM.seat_of(g, 0)]
        assert st["ties_in"] == len(st["dlog"]) == len(st["cut_log"])
        for ply, n_tied, dropped, first, ms in st["cut_log"]:
            assert 0 <= dropped < n_tied and first in (0, 1) and ms >= 0
            n_cut += dropped > 0
        for c in st["cases"]:
            assert c["pick"] in c["tied"]
        assert g["stats"][TM.seat_of(g, 1)]["cut_log"] == []
    assert len(games) == 4
