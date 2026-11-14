"""Plan 78.42, the whole sheet on the light reading: the codec's move tables against the fact view's, every level of the sheet on the light reading against the full one on small made-up positions, the
goal sign from the kernel's legal moves (no scenario read), the light scenario report against the kernel's, and the classification helpers of "what the sheet sees that the search does not" with a stub
search."""
import re
from pathlib import Path

import pytest

import cs_engine as CE

from catspace.incubator.analyst.discovery import question_sheet as Q

WORTH = {"k": 0, "q": 14, "r": 8, "b": 5, "n": 4, "p": 2}
HANGING = "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"
PAWNS = "4k3/p7/8/3q4/8/8/P2R4/4K3 w - - 0 1"
NO_PLACE = "7k/8/1p6/3q4/8/8/2p5/R6K w - - 0 1"
NO_GAIN = "7k/8/4p3/3p4/8/8/8/R3K3 w - - 0 1"
KING_ROUTE = "8/6Np/8/1P6/2p1P2K/8/8/4k3 w - - 0 1"
OWN_IN_WAY = "4k3/8/8/3b4/8/8/2P5/N3K3 w - - 0 1"
PAWN_TAKES = "7k/8/8/3b4/8/3p4/8/N3K3 w - - 0 1"
THEIR_IN_WAY = "7k/8/8/7n/1p6/p7/8/R3K3 w - - 0 1"
CONFINED_FAR = "7k/6pp/8/8/8/8/8/K7 w - - 0 1"
CONFINED_NEAR = "7k/6pp/8/8/8/8/8/R3K3 w - - 0 1"
IN_CHECK = "4k3/8/8/4q3/8/8/3R4/4K3 w - - 0 1"
ALL = [HANGING, PAWNS, NO_PLACE, NO_GAIN, KING_ROUTE, OWN_IN_WAY, PAWN_TAKES, THEIR_IN_WAY, CONFINED_FAR, CONFINED_NEAR, IN_CHECK]
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "question_sheet.py"


@pytest.fixture(scope="module")
def env():
    return Q.Env("chess")


def _sheet(env, fen, light):
    return Q.run(fen, WORTH, env=env, max_steps=10000, light=light)


def _digest(sheet):
    out = {}
    for q in sheet.questions:
        if q.quiet:
            continue
        levels = {f.level: f for f in q.findings}
        l1 = levels.get(1)
        out[(q.target.side, q.target.kind, q.target.at)] = (levels[0].status, None if l1 is None else (l1.status, l1.detail.get("failed_at")), Q.verdict_kind(q))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- the codec's tables

def test_the_codec_move_tables_of_any_placement_equal_the_fact_views(env):
    codec = env.codec
    for fen in (HANGING, PAWNS, KING_ROUTE, THEIR_IN_WAY, "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"):
        r = Q.Reading(env, fen, WORTH)
        other = codec.opposite(r.mover)
        for (kind, side, at), to in r.go[0].items():
            assert set(codec.reach_from(fen, r.mover if side == Q.SELF else other, kind, "".join(at))) == set(to)
        for (kind, side, at), to in r.acts.items():
            assert set(codec.bearing_from(fen, r.mover if side == Q.SELF else other, kind, "".join(at))) == set(to)


def test_the_light_move_tables_within_two_and_three_equal_the_fact_views(env):
    for fen in (HANGING, KING_ROUTE):
        full, light = Q.Reading(env, fen, WORTH), Q.Reading(env, fen, WORTH, light=True)
        for j in range(3):
            for key, to in list(full.go[j].items())[:80]:
                assert set(light.go[j].get(key)) == set(to)


def test_a_light_reading_reads_no_fact_view_for_its_tables(env, monkeypatch):
    calls = []
    real = Q.CE.facts
    monkeypatch.setattr(Q.CE, "facts", lambda *a, **k: calls.append(1) or real(*a, **k))
    r = Q.Reading(env, KING_ROUTE, WORTH, light=True)
    king = next(u for u in r.units if u.side == Q.SELF and u.kind == "king")
    assert r.exits_of(king) and r.go[1].get((king.kind, king.side, king.at)) and not calls


# ---------------------------------------------------------------------------------------------------------------------------------- level 0, level 1 and its three links, the frames, the routes, waiting

@pytest.mark.parametrize("fen", [HANGING, PAWNS, NO_PLACE, NO_GAIN, CONFINED_FAR, CONFINED_NEAR])
def test_levels_0_and_1_and_the_deep_verdicts_agree_with_the_full_reading(env, fen):
    assert _digest(_sheet(env, fen, True)) == _digest(_sheet(env, fen, False))


@pytest.mark.parametrize("fen", [KING_ROUTE, OWN_IN_WAY, PAWN_TAKES, THEIR_IN_WAY])
def test_the_routes_followed_whole_and_the_waiting_verdicts_agree(env, fen):
    full, light = _sheet(env, fen, False), _sheet(env, fen, True)
    assert _digest(full) == _digest(light)

    def ways(sheet):
        out = {}
        for q in sheet.questions:
            deep = next((f for f in reversed(q.findings) if f.level >= 2 and f.detail.get("results")), None)
            for x in (deep.detail["results"] if deep else ()):
                if x.get("way") is not None:
                    out[(q.target.kind, q.target.at, x["way"][1].kind, x["way"][1].at, x["way"][2])] = (x["status"], x.get("value"), x.get("move"))
        return out
    assert ways(full) == ways(light) and ways(light)


def test_the_exchange_and_cornering_frames_agree(env):
    for fen in (HANGING, "3rk3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", IN_CHECK):
        full, light = Q.Reading(env, fen, WORTH), Q.Reading(env, fen, WORTH, light=True)
        for side in (Q.SELF, Q.OTHER):
            for uf, ul in zip([u for u in full.units if u.side == Q.OTHER], [u for u in light.units if u.side == Q.OTHER]):
                a, b = Q.exchange_stands(full, uf, Q.SELF), Q.exchange_stands(light, ul, Q.SELF)
                assert (a["value"], a["line"], a["source"]) == (b["value"], b["line"], b["source"])
            break
    r = Q.Reading(env, "7k/6pp/8/8/6R1/8/8/K7 w - - 0 1", WORTH, light=True)
    king = next(u for u in r.units if u.side == Q.OTHER and u.kind == "king")
    rf = Q.Reading(env, "7k/6pp/8/8/6R1/8/8/K7 w - - 0 1", WORTH)
    assert Q.cornering(r, king, Q.SELF) == Q.cornering(rf, next(u for u in rf.units if u.side == Q.OTHER and u.kind == "king"), Q.SELF)


# ---------------------------------------------------------------------------------------------------------------------------------- the goal sign from legal moves

def _goal_level1(env, fen, light, params=None):
    r = Q.Reading(env, fen, WORTH, light=light)
    q = next(x for x in Q.first_order(r) if x.goal and x.target.side == Q.OTHER)
    Q.step(q, r, params or Q.Params())
    return Q.step(q, r, params or Q.Params()), r


def test_the_goal_sign_is_read_from_legal_moves_and_needs_no_scenario(env, monkeypatch):
    Q.Reading(env, CONFINED_FAR, WORTH, light=True).goal_kind()                   # the goal's kind is the game's: read once, then kept on the env
    calls = []
    real = Q.CE.scenario
    monkeypatch.setattr(Q.CE, "scenario", lambda *a, **k: calls.append(1) or real(*a, **k))
    far, _ = _goal_level1(env, CONFINED_FAR, True)
    assert far.status == Q.IDLE and far.detail["open_exits"] == 1 and far.detail["near"] is False
    near, r = _goal_level1(env, CONFINED_NEAR, True)
    assert near.status == Q.LIVE and near.detail["near"] and near.detail["open_exits"] == 1
    assert not calls
    goal = r.goal_unit(Q.OTHER)
    exits, source = Q.goal_exits(r, goal)
    assert source == "moves" and exits == {("g", "8")}


def test_the_goal_unit_exits_are_the_legal_moves_of_the_side_to_move_or_of_the_pass(env):
    r = Q.Reading(env, "4k3/8/8/8/8/8/8/R3K3 w - - 0 1", WORTH, light=True)
    mine = r.goal_unit(Q.SELF)
    exits, src = Q.goal_exits(r, mine)
    assert src == "moves" and {"".join(p) for p in exits} == {m[2:4] for m in r.legal if m[:2] == "e1"}
    theirs = r.goal_unit(Q.OTHER)
    ex2, src2 = Q.goal_exits(r, theirs)
    passed = CE.state("chess", r.passed_position(), [])[2]
    assert src2 == "moves" and {"".join(p) for p in ex2} == {m[2:4] for m in passed if m[:2] == "e8"}


def test_the_zone_is_the_fallback_when_the_pass_is_refused(env):
    r = Q.Reading(env, IN_CHECK, WORTH, light=True)                               # the mover's goal unit is struck: no pass
    theirs = r.goal_unit(Q.OTHER)
    exits, src = Q.goal_exits(r, theirs)
    assert src == "zone" and exits == set(r.exits_of(theirs))


@pytest.mark.parametrize("fen", [CONFINED_FAR, CONFINED_NEAR, IN_CHECK, HANGING, PAWNS, "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "8/8/4k3/1b6/8/1K6/8/2q1b3 w - - 4 60"])
def test_the_light_scenario_report_equals_the_kernels(env, fen):
    full, light = Q.Reading(env, fen, WORTH), Q.Reading(env, fen, WORTH, light=True)
    for role in ("mover", "other"):
        a, b = full.scenario(role), light.scenario(role)
        norm = lambda x: ({(h["at"], h["kind"]) for h in x["holes"]}, sorted((c["at"], c["own"], tuple(sorted((u["kind"], u["at"]) for u in c["by"]))) for c in x["covered"]), x["safe"],
                          [(s["moves"], tuple(sorted((p["kind"], p["from"], p["to"], p["moves"]) for p in s["placements"]))) for s in x["scenarios"]])
        assert norm(a) == norm(b)


def test_a_goal_unit_newly_under_the_sign_is_read_on_the_light_reading(env):
    r = Q.Reading(env, "7k/6pp/8/8/8/8/8/R3K3 w - - 0 1", WORTH, light=True)
    r.goal_kind()
    assert Q.goal_sign(r, Q.OTHER, Q.Params()) is True
    far = Q.Reading(env, CONFINED_FAR, WORTH, light=True)
    assert Q.goal_sign(far, Q.OTHER, Q.Params()) is False


def test_the_move_question_agrees_on_the_light_reading(env):
    for fen in (HANGING, PAWNS, KING_ROUTE):
        full, light = Q.Reading(env, fen, WORTH), Q.Reading(env, fen, WORTH, light=True)
        for mv in full.legal[:6]:
            a, b = Q.move_question(env, full, mv), Q.move_question(env, light, mv)
            assert (a["verdict"], round(a["low"], 6)) == (b["verdict"], round(b["low"], 6))


def test_only_asks_the_questions_of_the_units_it_names(env):
    sheet = Q.run(HANGING, WORTH, env=env, max_steps=10000, light=True, only=lambda u: u.kind == "queen")
    assert [q.target.kind for q in sheet.questions if not q.quiet] == ["queen"]


# ---------------------------------------------------------------------------------------------------------------------------------- what the sheet sees that the search does not

def rec(owner="theirs", first="a1a2", own=("a1a2",), low=4.0, status="confirmed"):
    return {"owner": owner, "status": status, "first": first, "own_moves": own, "low": low}


def test_classify_a_way_of_the_mover_plays_contains_or_holds():
    assert Q.classify_way(rec(), {"move": "a1a2", "line": [], "value": None}) == {"plays": True, "contains": False, "holds": False, "seen": True}
    got = Q.classify_way(rec(), {"move": "h2h3", "line": ["h2h3", "g8g7", "a1a2", "g7g6"], "value": 0})
    assert got["plays"] is False and got["contains"] is True and got["seen"]
    got = Q.classify_way(rec(), {"move": "h2h3", "line": ["h2h3", "a1a2"], "value": 0})                  # a move of theirs in the line is not the way's
    assert got["contains"] is False and got["seen"] is False
    assert Q.classify_way(rec(), {"move": "h2h3", "line": [], "value": 4})["holds"] is True
    assert Q.classify_way(rec(), {"move": "h2h3", "line": [], "value": 3})["seen"] is False
    assert Q.classify_way(rec(), {"move": None, "line": [], "value": None})["seen"] is False
    assert Q.classify_way(rec(), {"move": "h2h3", "line": [], "value": None, "proved": True})["holds"] is True            # a proved forced win holds any gain


def test_classify_a_way_of_the_other_side_answers_shows_the_loss_or_is_in_the_line():
    mine = rec(owner="mine", first="g8g7", own=("g8g7",))
    assert Q.classify_way(mine, {"move": "h2h3", "line": [], "value": 0}, answered=True) == {"answers": True, "shows_loss": False, "contains": False, "seen": True}
    assert Q.classify_way(mine, {"move": "h2h3", "line": [], "value": -4}, answered=False)["shows_loss"] is True
    assert Q.classify_way(mine, {"move": "h2h3", "line": [], "value": -3}, answered=False)["seen"] is False
    got = Q.classify_way(mine, {"move": "h2h3", "line": ["h2h3", "g8g7"], "value": 0}, answered=False)
    assert got["contains"] is True and got["seen"] and not got["answers"]
    assert Q.classify_way(mine, {"move": "h2h3", "line": [], "value": 0}, answered=None)["seen"] is False


def test_a_waiting_record_is_judged_by_its_high_bound_and_has_no_value_flag_without_one():
    w = dict(rec(status="waiting", low=0.0), high=3.0)
    assert Q.classify_way(w, {"move": "x", "line": [], "value": 3})["holds"] is True
    assert Q.classify_way(w, {"move": "x", "line": [], "value": 2})["holds"] is False
    none = dict(rec(status="waiting", low=0.0), high=0.0)
    assert Q.classify_way(none, {"move": "x", "line": [], "value": 9})["holds"] is False


def test_the_condition_a_record_waits_on_is_read_from_the_game():
    by_ply = [{"c2": 1, "a1": 2}, {"c2": 1, "a1": 2}, {"a1": 2}, {"a1": 2, "c3": 1}]
    assert Q.condition_came_true(by_ply, 0, [("vacated", ("c2",), "w")], 8) is True
    assert Q.condition_came_true(by_ply, 0, [("vacated", ("c2",), "w")], 1) is False
    assert Q.condition_came_true(by_ply, 0, [("duty", ("a1",), "w")], 8) is None


def test_multi_move_records_list_the_confirmed_arrivals_and_every_waiting_record(env):
    sheet = _sheet(env, OWN_IN_WAY, True)
    got = Q.multi_move_records(sheet)
    assert got and all(x["hops"] >= 1 and x["status"] in ("confirmed", "waiting") for x in got)
    w = next(x for x in got if x["status"] == "waiting")
    assert w["owner"] == "theirs" and w["unit"][0] == "knight" and ("vacated", ("c2",)) == w["waits_on"][0][:2] and w["chain"]
    sheet = _sheet(env, KING_ROUTE, True)
    c = [x for x in Q.multi_move_records(sheet) if x["status"] == "confirmed"]
    assert c and c[0]["first"] and c[0]["low"] > 0 and c[0]["hops"] >= 1


def test_still_confirmed_reads_the_child_sheet(env):
    sheet = _sheet(env, HANGING, True)                                          # the standing loss: my rook strikes their unguarded queen, so on their move it is lost no more... read as a sheet
    assert Q.still_confirmed(sheet, {"unit": ("rook", "d2")}) in (True, False)


def test_the_module_still_names_no_object_of_one_game():
    words = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check\w*|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
    text = SOURCE.read_text()
    hit = re.search(words, text, flags=re.I)
    assert hit is None, hit and (hit.group(0), text[max(0, hit.start() - 40):hit.end() + 20])


def test_the_full_reading_stays_reachable_by_flags(env):
    base = _digest(_sheet(env, CONFINED_NEAR, False))
    assert _digest(Q.run(CONFINED_NEAR, WORTH, env=env, max_steps=10000, light=True, full_go=True, full_goal=True)) == base
    r = Q.Reading(env, HANGING, WORTH, light=True)
    r.full_go = True
    assert isinstance(r.go[0], dict)
    assert not isinstance(Q.Reading(env, HANGING, WORTH).go[0], Q._Lazy)
