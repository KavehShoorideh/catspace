"""Plan 78.42 USE: the sheet player (catspace.incubator.analyst.discovery.sheet_player), with a stub search, a stub sheet and a stub game: the acceptance rule, the fallback to the search's first
choice, `max_tries` and the budget, no override under a forced win or with one legal move, the log row, the flags, and the generic test (the module names no object of any one game)."""
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import question_sheet as QS
from catspace.incubator.analyst.discovery import sheet_player as SP

SOURCE = Path(__file__).resolve().parents[1] / "catspace" / "incubator" / "analyst" / "discovery" / "sheet_player.py"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class StubSearch:
    def __init__(self, first="m1", rest=("m2", "m3", "m4", "m5", "m6", "m7"), forced=False):
        self.first, self.rest, self.forced = first, list(rest), forced
        self.decided = self.ranked = 0

    def decide(self, start, moves, seconds):
        self.decided += 1
        return {"move": self.first, "proved": self.forced, "value": 0, "forced": self.forced, "error": None, "seconds": 0.01}

    def alternatives(self, start, moves, legal, exclude, seconds, threads, limit):
        self.ranked += 1
        return {"moves": [m for m in self.rest if m not in exclude][:limit], "seconds": 0.02}


class StubSheet:
    """`table`: move -> an Assessment (or a verdict word); every assessment costs `cost` seconds of the fake clock."""

    def __init__(self, table, clock=None, cost=0.0, standing=()):
        self.table, self.clock, self.cost, self._standing = table, clock, cost, list(standing)
        self.asked, self.opened = [], 0

    def open(self, position):
        self.opened += 1
        return object()

    def standing(self, r):
        return self._standing

    def assess(self, r, move, standing, use_question=True, use_standing=True):
        self.asked.append(move)
        if self.clock is not None:
            self.clock.t += self.cost
        a = self.table.get(move, SP.Assessment(move))
        if isinstance(a, str):
            a = SP.Assessment(move, verdict=a)
        a.seconds = self.cost
        return a


def state(legal=("m1", "m2", "m3")):
    return lambda start, moves: ("pos", "us", list(legal), None)


def lose(move, stake=3.0):
    return SP.Assessment(move, verdict=QS.LOSES_V, stake=stake, lines=[f"{move} loses a unit worth {stake:g}"])


def left(move, excess=4.0):
    return SP.Assessment(move, unanswered=[{"unit": "my unit", "after": excess, "excess": excess}], stake=excess, lines=["a loss that stood is left"])


def player(search, sheet, cfg=None, clock=None, legal=("m1", "m2", "m3")):
    return SP.SheetPlayer(search, sheet, state(legal), cfg or SP.PlayerConfig(), clock=clock or Clock())


# ---------------------------------------------------------------------------------------------------------------------------------- the acceptance rule

def test_acceptance_rule():
    assert SP.accepts(SP.Assessment("a", verdict=QS.SAFE_V))
    assert not SP.accepts(SP.Assessment("a", verdict=QS.LOSES_V))
    assert not SP.accepts(SP.Assessment("a", verdict=QS.MOVE_UNSETTLED))
    assert not SP.accepts(left("a"))                                                   # SAFE by the question, a standing loss left unanswered
    assert not SP.accepts(SP.Assessment("a", error="the kernel does not accept the move"))
    # the flags switch each part off
    assert SP.accepts(left("a"), use_standing=False)
    assert SP.accepts(SP.Assessment("a", verdict=QS.LOSES_V), use_question=False)
    assert not SP.accepts(SP.Assessment("a", verdict=QS.LOSES_V, unanswered=[{"unit": "u", "after": 1, "excess": 1}]), use_question=False)


def test_a_taking_that_pays_more_than_it_loses_is_accepted():
    after = [("my unit", 3.0)]
    assert SP.unanswered_losses(after, took=5.0) == []                                  # it took more than the exchange wins back
    assert SP.unanswered_losses(after, took=3.0) == []                                  # an even balance is not worse
    left_ = SP.unanswered_losses(after, took=1.0)
    assert left_ == [{"unit": "my unit", "after": 3.0, "excess": 2.0}]
    a = SP.Assessment("a", took=5.0, unanswered=SP.unanswered_losses(after, 5.0))
    assert SP.accepts(a)
    assert not SP.accepts(SP.Assessment("a", took=1.0, unanswered=left_))
    assert SP.unanswered_losses([], 0.0) == []


# ---------------------------------------------------------------------------------------------------------------------------------- the player

def test_safe_first_choice_is_played_and_logged():
    se, sh = StubSearch(), StubSheet({"m1": "SAFE"})
    mv, row = player(se, sh).choose(None, [])
    assert mv == "m1" and sh.asked == ["m1"] and se.ranked == 0                       # the next choices are not even ordered
    assert row["played"] == row["search_move"] == "m1" and not row["overridden"] and row["why"] == "first_accepted"
    assert [t["move"] for t in row["tries"]] == ["m1"] and row["tries"][0]["accepted"]
    for key in ("ply", "proved", "value", "standing", "rejected_stake", "reason", "seconds"):
        assert key in row
    assert set(row["seconds"]) == {"search", "read", "standing", "first", "tries", "rank", "total"}


def test_loses_is_skipped_and_the_next_safe_choice_is_played():
    se, sh = StubSearch(), StubSheet({"m1": lose("m1", 3.0), "m2": "LOSES", "m3": "SAFE"})
    mv, row = player(se, sh).choose(None, [4, 5])
    assert mv == "m3" and row["overridden"] and row["why"] == "overridden" and row["search_move"] == "m1" and row["played"] == "m3" and row["ply"] == 2
    assert [t["move"] for t in row["tries"]] == ["m1", "m2", "m3"] and [t["accepted"] for t in row["tries"]] == [False, False, True]
    assert row["rejected_stake"] == 3.0 and "loses a unit worth 3" in " ".join(row["reason"])        # the words of the search's choice
    assert "reason_played" in row


def test_a_standing_loss_left_unanswered_is_skipped():
    se = StubSearch()
    sh = StubSheet({"m1": left("m1", 4.0), "m2": "SAFE"}, standing=[(QS.Unit(QS.SELF, "kind_x", ("p", "q")), 4.0)])
    mv, row = player(se, sh).choose(None, [])
    assert mv == "m2" and row["overridden"] and row["rejected_stake"] == 4.0
    assert row["standing"] == [["my kind_x on pq", 4.0]]


def test_none_accepted_plays_the_first_choice():
    se, sh = StubSearch(), StubSheet({m: lose(m) for m in ("m1", "m2", "m3", "m4", "m5", "m6", "m7")})
    mv, row = player(se, sh).choose(None, [])
    assert mv == "m1" and not row["overridden"] and row["why"] == "none_accepted"
    assert len(row["tries"]) == 5 and row["rejected_stake"] == 3.0                     # max_tries 5, the first included


def test_max_tries_is_honoured():
    se, sh = StubSearch(), StubSheet({m: lose(m) for m in ("m1", "m2", "m3", "m4", "m5", "m6", "m7")})
    _mv, row = player(se, sh, SP.PlayerConfig(max_tries=3)).choose(None, [])
    assert sh.asked == ["m1", "m2", "m3"]
    mv, row = player(se, StubSheet({"m1": "LOSES"}), SP.PlayerConfig(max_tries=1)).choose(None, [])
    assert mv == "m1" and len(row["tries"]) == 1 and row["why"] == "none_accepted"


def test_the_budget_is_honoured():
    clock = Clock()
    se = StubSearch()
    sh = StubSheet({m: lose(m) for m in ("m1", "m2", "m3", "m4", "m5", "m6", "m7")}, clock=clock, cost=0.1)
    # the first try is always made; tries after it go on while the time spent on them is under the budget
    mv, row = player(se, sh, SP.PlayerConfig(budget=0.15, max_tries=10), clock=clock).choose(None, [])
    assert sh.asked == ["m1", "m2", "m3"] and mv == "m1"                                # m2 (0.1 < 0.15) and m3 (0.1 < 0.15) are made, then 0.2 is over
    sh = StubSheet({m: lose(m) for m in ("m1", "m2", "m3", "m4")}, clock=clock, cost=0.1)
    player(se, sh, SP.PlayerConfig(budget=0.0, max_tries=10), clock=clock).choose(None, [])
    assert sh.asked == ["m1"]                                                           # no budget for the tries after the first
    assert SP.PlayerConfig(seconds=0.2).sheet_budget() == 0.2 and SP.PlayerConfig(seconds=0.2, budget=0.05).sheet_budget() == 0.05


def test_no_override_under_a_forced_win_or_with_one_legal_move():
    se, sh = StubSearch(forced=True), StubSheet({"m1": lose("m1")})
    mv, row = player(se, sh).choose(None, [])
    assert mv == "m1" and row["why"] == "forced" and sh.asked == [] and sh.opened == 0 and not row["overridden"]
    se, sh = StubSearch(), StubSheet({"m1": lose("m1")})
    mv, row = player(se, sh, legal=("m1",)).choose(None, [])
    assert mv == "m1" and row["why"] == "one_legal" and sh.asked == [] and not row["overridden"]


def test_flags_switch_the_parts_off():
    se, sh = StubSearch(), StubSheet({"m1": lose("m1")})
    mv, row = player(se, sh, SP.PlayerConfig(use_standing=False, use_question=False)).choose(None, [])
    assert mv == "m1" and row["why"] == "off" and sh.asked == [] and sh.opened == 0       # the control: the plain search
    cfg = SP.PlayerConfig(use_standing=False)
    player(se, sh, cfg).choose(None, [])
    assert sh.asked and sh._standing == []


def test_the_search_that_has_no_move_and_the_unreadable_position():
    class Dead(StubSearch):
        def decide(self, start, moves, seconds):
            return {"move": None, "proved": False, "value": None, "forced": False, "error": "x", "seconds": 0.0}
    mv, row = player(Dead(), StubSheet({})).choose(None, [])
    assert mv is None and row["why"] == "no_move"

    class Unreadable(StubSheet):
        def open(self, position):
            raise ValueError("the position is over or has no legal moves")
    mv, row = player(StubSearch(), Unreadable({})).choose(None, [])
    assert mv == "m1" and row["why"] == "unreadable"


# ---------------------------------------------------------------------------------------------------------------------------------- the real sheet, a small board

WORTH = {"k": 0, "q": 14, "r": 8, "b": 5, "n": 4, "p": 2}
STAND = "4k3/8/8/8/3q4/8/8/R3K2R w - - 0 1"         # their queen is far from my rooks and strikes none of them


def test_the_real_engine_reads_a_position_and_a_move():
    env = QS.Env("chess")
    eng = SP.SheetEngine(env, WORTH)
    r = eng.open(STAND)
    st = eng.standing(r)
    assert isinstance(st, list)
    mv = r.legal[0]
    a = eng.assess(r, mv, st)
    assert a.move == mv and a.verdict in (QS.SAFE_V, QS.LOSES_V, QS.MOVE_UNSETTLED) and a.seconds > 0 and a.error is None
    assert SP.accepts(a) == (a.verdict == QS.SAFE_V and not a.unanswered)


def test_the_module_names_no_object_of_any_one_game():
    text = SOURCE.read_text()
    words = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check\w*|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
    assert not re.findall(words, text, flags=re.I)


# ---------------------------------------------------------------------------------------------------------------------------------- 78.42 equal time

class ClockSearch(StubSearch):
    """A stub search that records the clock it is given and spends it on the fake clock."""

    def __init__(self, clock, **kw):
        super().__init__(**kw)
        self.clock, self.given = clock, []

    def decide(self, start, moves, seconds):
        self.given.append(seconds)
        self.clock.t += seconds
        return {"move": self.first, "proved": False, "value": 0, "forced": False, "error": None, "seconds": seconds}


class OrderedSheet(StubSheet):
    def order(self, r, moves):
        return list(moves)


EQUAL = dict(seconds=0.2, equal=True, filtered=True, use_standing=False, floor=0.05, prior_cost=0.01)


def test_under_equal_the_search_gets_the_clock_less_the_estimate_and_the_estimate_follows_the_moves():
    clock = Clock()
    se, sh = ClockSearch(clock), OrderedSheet({}, clock=clock, cost=0.03)
    p = player(se, sh, SP.PlayerConfig(**EQUAL), clock=clock)
    p.choose(None, [])                                   # the prior estimate 0.01
    p.choose(None, [])                                   # the measured 0.03
    p.choose(None, [])
    assert se.given == pytest.approx([0.2 - 0.01, 0.2 - 0.03, 0.2 - 0.03]) and abs(p.sheet_estimate() - 0.03) < 1e-9
    assert abs(sum(se.given[1:]) / 2 + p.sheet_estimate() - 0.2) < 1e-9                      # the whole move comes to the clock of the plain player


def test_the_search_clock_never_falls_under_the_floor():
    clock = Clock()
    se, sh = ClockSearch(clock), OrderedSheet({}, clock=clock, cost=0.5)
    p = player(se, sh, SP.PlayerConfig(**EQUAL), clock=clock)
    p.choose(None, [])
    p.choose(None, [])
    p.choose(None, [])
    assert se.given[1:] == pytest.approx([0.05, 0.05]) and min(se.given) >= 0.05


def test_without_equal_the_search_keeps_its_whole_clock():
    clock = Clock()
    se, sh = ClockSearch(clock), OrderedSheet({}, clock=clock, cost=0.1)
    p = player(se, sh, SP.PlayerConfig(seconds=0.2), clock=clock)
    p.choose(None, [])
    p.choose(None, [])
    assert se.given == [0.2, 0.2] and p.costs == []


def test_a_move_the_sheet_never_reads_costs_nothing_and_lowers_the_estimate():
    clock = Clock()
    se, sh = ClockSearch(clock, forced=True), OrderedSheet({}, clock=clock, cost=0.1)
    se.decide = lambda start, moves, seconds: (se.given.append(seconds), {"move": "m1", "proved": True, "value": 0, "forced": True, "error": None, "seconds": seconds})[1]
    p = player(se, sh, SP.PlayerConfig(**EQUAL), clock=clock)
    p.choose(None, [])
    assert p.costs == [0.0] and sh.asked == []


def test_filtered_asks_the_others_in_the_sheets_order_and_the_search_orders_only_the_accepted():
    clock = Clock()
    se = ClockSearch(clock, rest=("m4", "m3", "m2"))
    sh = OrderedSheet({"m1": lose("m1"), "m2": "SAFE", "m3": lose("m3"), "m4": "SAFE"}, clock=clock, cost=0.001)
    mv, row = player(se, sh, SP.PlayerConfig(**EQUAL), clock=clock, legal=("m1", "m2", "m3", "m4")).choose(None, [])
    assert sh.asked == ["m1", "m2", "m3", "m4"] and mv == "m4" and row["overridden"] and row["why"] == "overridden"      # the search puts m4 before m2 among the accepted
    assert se.ranked == 1 and "reason_played" in row


def test_filtered_stops_at_the_number_of_accepted_moves_and_at_the_budget():
    clock = Clock()
    se = ClockSearch(clock)
    sh = OrderedSheet({"m1": lose("m1")}, clock=clock, cost=0.001)
    cfg = SP.PlayerConfig(**{**EQUAL, "max_alternatives": 2})
    player(se, sh, cfg, clock=clock, legal=("m1", "m2", "m3", "m4", "m5")).choose(None, [])
    assert sh.asked == ["m1", "m2", "m3"]                                                   # two accepted, then no more asked
    sh = OrderedSheet({m: lose(m) for m in ("m1", "m2", "m3", "m4", "m5")}, clock=clock, cost=0.03)
    mv, row = player(se, sh, SP.PlayerConfig(**{**EQUAL, "budget": 0.05}), clock=clock, legal=("m1", "m2", "m3", "m4", "m5")).choose(None, [])
    assert sh.asked == ["m1", "m2", "m3"] and mv == "m1" and row["why"] == "none_accepted"      # the first, then 0.03 and 0.06 spent: over the budget


def test_the_equal_configs_are_the_cheap_reading_with_the_clock_shared():
    for name in ("equal", "equal_both", "equal_wide"):
        cfg = SP.PlayerConfig(**SP.CONFIGS[name])
        assert cfg.equal and cfg.light and cfg.filtered and cfg.use_question
    assert not SP.PlayerConfig(**SP.CONFIGS["equal"]).use_standing and SP.PlayerConfig(**SP.CONFIGS["equal_both"]).use_standing
    assert SP.PlayerConfig(**SP.CONFIGS["equal_wide"]).max_alternatives == 8 and SP.PlayerConfig(**SP.CONFIGS["equal"]).max_alternatives == 4


def test_the_light_engine_reads_a_position_and_the_standing_follows_a_unit_that_changes_kind():
    env = QS.Env("chess")
    eng = SP.SheetEngine(env, WORTH, light=True)
    r = eng.open("2k5/P7/8/8/8/8/r7/4K3 w - - 0 1")
    assert r.light
    a = eng.assess(r, "a7a8q", [(next(u for u in r.units if u.kind == "pawn" and u.side == QS.SELF), 2.0)], use_question=False)
    assert a.error is None and a.took == 12.0                                              # the change itself counts for the move
    assert eng.order(r, r.legal)[0] in r.legal and sorted(eng.order(r, r.legal)) == sorted(r.legal)
