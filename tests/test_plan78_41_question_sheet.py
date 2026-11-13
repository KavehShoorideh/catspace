"""Plan 78.41 stage 1 THE QUESTION SHEET (catspace.incubator.analyst.discovery.question_sheet): the first order, the levels, the schedule, the quiet question, the plain-words
explanation, the measures on a hand-made pair of game files, and the generic test (the module names no object of any one game)."""
import json
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import question_sheet as Q

WORTH = {"k": 0, "q": 14, "r": 8, "b": 5, "n": 4, "p": 2}
HANGING = "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"            # my rook strikes their unguarded queen
SPARSE = "7k/p7/8/8/8/8/8/K7 w - - 0 1"                  # my king is far from their pawn
PAWNS = "4k3/p7/8/3q4/8/8/P2R4/4K3 w - - 0 1"
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "question_sheet.py"


@pytest.fixture(scope="module")
def env():
    return Q.Env("chess")


def _first_order(env, fen):
    r = Q.Reading(env, fen, WORTH)
    return r, Q.first_order(r)


def _find(qs, side, kind):
    return next(q for q in qs if q.target.side == side and q.target.kind == kind)


def test_first_order_alternates_sides_down_the_stakes(env):
    _r, qs = _first_order(env, PAWNS)
    assert [(q.target.side, q.target.kind) for q in qs] == [("us", "king"), ("them", "king"), ("them", "queen"), ("us", "rook"), ("us", "pawn"), ("them", "pawn")]
    assert [q.rank for q in qs] == [1, 2, 3, 4, 5, 6]
    assert qs[0].goal and qs[1].goal and not qs[2].goal
    assert qs[0].stake == qs[1].stake > qs[2].stake > qs[3].stake > qs[4].stake
    assert [q.asker for q in qs] == ["them", "us", "us", "them", "them", "us"]          # a question is asked by the side that does not own the target


def test_the_worth_table_decides_the_order_below_the_goal_unit(env):
    r = Q.Reading(env, PAWNS, {**WORTH, "p": 20})                                       # a table that puts the pawn above the rook and the queen
    qs = Q.first_order(r)
    assert [(q.target.side, q.target.kind) for q in qs][2:4] == [("them", "pawn"), ("us", "pawn")]


def test_a_target_nothing_can_reach_is_closed_with_its_reason(env):
    r, qs = _first_order(env, SPARSE)
    q = _find(qs, "them", "pawn")
    assert Q.step(q, r).status == Q.IDLE and q.state == "open"                          # level 0: nothing strikes it, still open
    f = Q.step(q, r)                                                                    # level 1: nothing can come
    assert f.level == 1 and f.status == Q.CLOSED and q.state == "closed"
    assert re.search(r"no unit of mine can reach it within 2 moves", f.text)
    assert q.reason == f.text


def test_a_target_struck_and_unguarded_is_live_at_level_0(env):
    r, qs = _first_order(env, HANGING)
    q = _find(qs, "them", "queen")
    f = Q.step(q, r)
    assert f.level == 0 and f.status == Q.LIVE and f.new
    assert "unguarded" in f.text and "my rook" in f.text


def test_the_goal_unit_is_read_through_the_scenario_report(env):
    r, qs = _first_order(env, "7k/5K2/8/8/8/8/8/R7 w - - 0 1")
    q = _find(qs, "them", "king")
    f0 = Q.step(q, r)
    assert q.goal and f0.detail["open_exits"] == 1
    f1 = Q.step(q, r)
    assert f1.status == Q.LIVE and "plan" in f1.text


def test_a_deeper_step_finds_the_best_line_and_its_value(env):
    r, qs = _first_order(env, HANGING)
    q = _find(qs, "them", "queen")
    Q.step(q, r), Q.step(q, r)
    f = Q.step(q, r)
    assert f.level == 2 and f.status == Q.LIVE and f.value == 14 and f.line[0] == "d2d5"          # by the exchange frame (frames on is the default)


def test_the_schedule_goes_to_the_higher_stake_and_moves_on_when_nothing_is_found(env):
    p = Q.Params()
    costs = Q._Costs(p)
    hi = Q.Question(Q.Unit("them", "queen", ("d", "5")), "us", 14.0, 1, level=1, urgency=1.0)
    lo = Q.Question(Q.Unit("them", "rook", ("a", "8")), "us", 8.0, 2, level=1, urgency=1.0)
    assert Q.priority(hi, costs) > Q.priority(lo, costs)                                # equal urgency and cost: the higher stake
    sheet = Q.Sheet("x", [hi, lo])
    nothing = lambda q: Q.Finding(q, q.level, Q.IDLE, "nothing", 0.001)
    order = []
    for _ in range(6):
        q = max((x for x in sheet.questions), key=lambda x: Q.priority(x, costs))
        order.append(q.rank)
        Q._record(sheet, nothing(q), costs, p)
    assert order[0] == 1 and 2 in order and order.index(2) <= 3                         # urgency halves with every empty step, so the lower stake gets its turn
    q = Q.Question(Q.Unit("them", "rook", ("a", "8")), "us", 8.0, 3, urgency=1.0)
    Q._record(sheet, Q.Finding(q, 0, Q.LIVE, "plan", 0.001, new=True), costs, p)
    assert q.urgency == p.urgency_start * p.urgency_up


def test_run_takes_every_level_0_step_first_in_the_first_order(env):
    sheet = Q.run(HANGING, WORTH, env=env, max_steps=12)
    n = len(sheet.reading and Q.first_order(sheet.reading))
    assert [s.level for s in sheet.steps[:n]] == [0] * n
    assert [s.question.rank for s in sheet.steps[:n]] == list(range(1, n + 1))
    assert max(s.level for s in sheet.steps) >= 1


def test_the_quiet_question_appears_only_when_the_rest_are_closed_or_idle(env):
    early = Q.run(HANGING, WORTH, env=env, max_steps=6)                                  # the queen question is live: nothing quiet yet
    assert not any(q.quiet for q in early.questions)
    late = Q.run(HANGING, WORTH, env=env, max_steps=60)
    quiet = [q for q in late.questions if q.quiet]
    assert len(quiet) == 1 and quiet[0].state == "closed"
    assert all(q.state == "closed" or q.status == Q.IDLE for q in late.questions if not q.quiet)
    assert "does least" in quiet[0].reason
    still = Q.run("7k/5K2/8/8/8/8/8/R7 w - - 0 1", WORTH, env=env, max_steps=3)         # three level 0 steps are not enough to offer it either way
    assert len([q for q in still.questions if q.quiet]) <= 1


def test_explain_is_plain_words_with_no_internal_label(env):
    sheet = Q.run(PAWNS, WORTH, env=env, max_steps=40)
    text = "\n".join(Q.explain(sheet))
    assert "step 1:" in text and "where the questions stand" in text and "my king" in text
    for label in ("go_within", "census", "scenario", "perspective", "acts_on", "can_go", "supply"):
        assert label not in text
    assert not re.search(r"\b(us|them)\b", text)


def _games_dir(tmp_path):
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "d7d6"]
    ours = {str(p): {"value": 0, "proved": False, "claim": None, "balance": b} for p, b in enumerate([0, 0, 0, 0, 0, -2, 2, 0])}
    (tmp_path / "games.jsonl").write_text(json.dumps({"variant": "chess", "game": 0, "start": None, "moves": moves, "ours": ours}) + "\n")
    rows = [{"game": 0, "ply": 3, "kind": "MISSED", "judge": {"jump": -0.5}}, {"game": 0, "ply": 4, "kind": "MISSED", "judge": {"jump": 0.2}}, {"game": 0, "ply": 5, "kind": "FALSE", "judge": {"jump": -0.9}}]
    (tmp_path / "surprises.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(tmp_path)


def test_load_positions_reads_bad_moves_and_the_balance_fall(tmp_path):
    items = Q.load_positions(_games_dir(tmp_path), "chess")
    assert [i[0] for i in items] == [(0, p) for p in range(8)]
    bad = [i[0][1] for i in items if i[2]]
    assert bad == [3]                                                                    # only a MISSED surprise with a judge jump below 0
    fell = {i[0][1]: i[3] for i in items}
    # the mover's balance at a ply against the next 4 plies, the odd ones read from the other side (negated): [0, 0, 0, 0, 0, -2, 2, 0]
    assert fell[3] is True                      # ply 5 (two plies on, the same mover) holds -2, below 0
    assert fell[4] is False                     # 0, then 2, 2, 0
    assert fell[5] is False                     # -2, then -2 and 0
    assert fell[6] is True                      # 2, then ply 7 negated is 0, below 2
    assert fell[7] is None                      # nothing after it
    assert Q.load_positions(_games_dir(tmp_path), "chess", limit=3).__len__() == 3


def _fake_sheet(live, closed1):
    q = Q.Question(Q.Unit("us", "king", ("e", "1")), "them", 1.0, 1)
    q.live_by_level1 = live
    q.closed_at = 1 if closed1 else None
    q.state = "closed" if closed1 else "open"
    q2 = Q.Question(Q.Unit("them", "king", ("e", "8")), "us", 1.0, 2)
    return Q.Sheet("x", [q, q2], steps=[Q.Step(q, 0, Q.IDLE, "t", 0.002), Q.Step(q, 1, Q.IDLE, "t", 0.004)])


def test_the_measures_count_what_a_hand_made_pair_of_files_holds(tmp_path):
    items = Q.load_positions(_games_dir(tmp_path), "chess")
    # open (a step found it live) at plies 3 and 4 only; level 1 closed on my unit at plies 4, 5 and 6
    open_at, closed_at = {3, 4}, {4, 5, 6}
    records, text, unreadable = Q.measure(items, lambda variant, start, moves: _fake_sheet(len(moves) in open_at, len(moves) in closed_at))
    assert unreadable == 0 and len(records) == 8
    lines = Q.format_obs(records)
    a = next(ln for ln in lines if "[open = a step of level 0 or 1 found it live]" in ln)
    assert "no question open: 6 of 8 positions (75.0%)" in a
    assert "bad move follows 50.0% when at least one is open (2 positions)" in a          # ply 3 is bad, ply 4 is not
    assert "0.0% when none is (6)" in a
    assert "the none-open positions hold 0.0% of the bad moves" in a
    b = next(ln for ln in lines if ln.startswith("OBS (b) level 1 closed"))
    assert "in 3 positions" in b and "fell within 4 plies in 1 (33.3%)" in b             # of plies 4, 5, 6 only ply 6 is followed by a fall
    c = [ln for ln in lines if ln.startswith("OBS (c) level")]
    assert "level 0: 8 steps, mean 2.00 ms" in c[0] and "level 1: 8 steps, mean 4.00 ms" in c[1]


def test_the_module_names_no_object_of_one_game():
    words = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check\w*|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
    text = SOURCE.read_text()
    hit = re.search(words, text, flags=re.I)
    assert hit is None, hit and (hit.group(0), text[max(0, hit.start() - 40):hit.end() + 20])


# ---------------------------------------------------------------------------------------------------------------------------------------------- level 1 as a chain of three links

NO_PLACE = "7k/8/1p6/3q4/8/8/2p5/R6K w - - 0 1"        # every place from which my rook would strike their queen is struck by a cheaper unit of theirs
NO_GAIN = "7k/8/4p3/3p4/8/8/8/R3K3 w - - 0 1"          # their pawn is guarded and worth less than my rook


def _level1(env, fen, side, kind, params=None):
    r, qs = _first_order(env, fen)
    q = _find(qs, side, kind)
    params = params or Q.Params()
    Q.step(q, r, params)
    return Q.step(q, r, params), q


def test_link_1_reach_fails_when_no_unit_can_come(env):
    f, q = _level1(env, SPARSE, "them", "pawn")
    assert f.status == Q.CLOSED and f.detail["failed_at"] == 1 and "can reach it within 2 moves" in f.text


def test_link_2_a_place_to_stand_fails_where_a_cheaper_unit_strikes_the_arrival_place(env):
    f, q = _level1(env, NO_PLACE, "them", "queen")
    assert f.status == Q.CLOSED and f.detail["failed_at"] == 2
    assert "none passes all three links" in f.text and "a place to stand" in f.text and "worth less" in f.text
    old, _ = _level1(env, NO_PLACE, "them", "queen", Q.Params(links=False))                 # reach alone calls it live
    assert old.status == Q.LIVE


def test_link_3_a_strike_that_gains_fails_on_a_guarded_target_worth_less(env):
    f, q = _level1(env, NO_GAIN, "them", "pawn")
    assert f.status == Q.CLOSED and f.detail["failed_at"] == 3
    assert "a strike that gains" in f.text and "guarded by their pawn" in f.text
    assert _level1(env, NO_GAIN, "them", "pawn", Q.Params(links=False))[0].status == Q.LIVE


def test_all_three_links_hold_for_an_unguarded_target(env):
    f, q = _level1(env, HANGING.replace("3R4", "R7").replace("w - - 0 1", "w - - 0 1"), "them", "queen")
    assert f.status == Q.LIVE and f.detail["links"][2] >= 1 and "pass reach, a place to stand and a strike that gains" in f.text


def test_explain_names_the_link_that_failed(env):
    sheet = Q.run(NO_PLACE, WORTH, env=env, max_steps=30)
    text = "\n".join(Q.explain(sheet))
    assert "fails at a place to stand" in text


# ---------------------------------------------------------------------------------------------------------------------------------------------- measures (b) by unit and (d) the quiet question

def test_load_positions_reads_the_unit_moved_and_the_weight(tmp_path):
    moves = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"]
    ours = {str(p): {"balance": 0} for p in range(6)}
    (tmp_path / "games.jsonl").write_text(json.dumps({"variant": "chess", "game": 0, "start": None, "moves": moves, "ours": ours}) + "\n")
    (tmp_path / "surprises.jsonl").write_text(json.dumps({"game": 0, "ply": 2, "kind": "MISSED", "judge": {"jump": -0.3}, "weight": 2.5}) + "\n")
    items = {i.key[1]: i for i in Q.load_positions(str(tmp_path), "chess", game="chess")}
    assert (items[0].moved, items[0].moved_to, items[1].moved, items[2].moved_to) == ("e2", "e4", "d7", "d5")
    assert items[2].bad and items[2].weight == 2.5 and items[0].weight == 0.0 and items[3].played == "d8d5"


# ---------------------------------------------------------------------------------------------------------------------------------------------- level 2 as a search of the question's own ways

def _to_level_2(env, fen, side, kind, params=None):
    r, qs = _first_order(env, fen)
    q = _find(qs, side, kind)
    params = params or Q.Params(frames=False)
    Q.step(q, r, params), Q.step(q, r, params)
    return Q.step(q, r, params), q, r


def test_a_confirmed_way_is_played_and_valued_after_the_reply(env):
    f, q, r = _to_level_2(env, HANGING, "them", "queen")
    assert f.detail["mode"] == "search" and f.status == Q.LIVE and f.detail["confirmed"] >= 1
    assert f.value == 14 and f.line == ("d2d5",)
    assert Q.step(q, r).level == 3                                                       # asked again: one ply deeper, still standing


def test_a_refuted_way_names_its_reply_and_closes_the_question(env):
    # my rook can take their queen, but the queen is guarded: the way is met by the recapture
    f, q, r = _to_level_2(env, "4k3/8/8/3q4/8/8/3R4/3rK3 w - - 0 1", "them", "queen")
    refuted = [x for x in f.detail["results"] if x["status"] == "refuted"]
    assert refuted and refuted[0]["reply"]
    # a position whose only way is answered closes with the reply named
    g, q2, r2 = _to_level_2(env, "4k3/8/8/3q4/8/8/3R4/3rK3 w - - 0 1", "them", "rook")
    assert g.detail["mode"] == "search"


def test_every_way_answered_closes_with_the_first_move_and_the_reply(env):
    r, qs = _first_order(env, "4k3/3q4/8/8/8/8/3R4/3K4 w - - 0 1")
    q = _find(qs, "them", "queen")
    Q.step(q, r), Q.step(q, r)
    f = Q.step(q, r)
    if f.status == Q.CLOSED:
        assert f.text.startswith("every way is answered: ") and " is met by " in f.text
    else:
        assert f.detail["refuted"] + f.detail["open"] + f.detail["confirmed"] == len(f.detail["results"])


def test_when_the_asker_is_not_to_move_the_census_is_kept_and_said(env):
    f, q, r = _to_level_2(env, HANGING, "us", "rook")
    assert f.detail["mode"].startswith("census") and "as if it were their turn" in f.text
    old, _q, _r = _to_level_2(env, HANGING, "them", "queen", Q.Params(search=False, frames=False))
    assert old.detail["mode"] == "census" and old.value == 14


def test_the_codec_finds_a_move_and_follows_a_unit():
    from catspace.incubator.analyst.discovery import curriculum as CU
    codec = CU.codec_of("chess")
    assert codec.find_move("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", "d2", "d5") == "d2d5"
    assert codec.find_move("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", "d2", "a8") is None
    moves = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"]
    assert codec.taken_within(None, moves, 0, "e2", 4) is True                           # the pawn moves to e4 and is taken on d5 at ply 3
    assert codec.taken_within(None, moves, 0, "a2", 4) is False
    assert codec.taken_within(None, moves, 6, "a2", 4) is None


def _rec(bad, rows, quiet_at=None, moved=None, **kw):
    return Q.Record((0, 0), bad, rows, None, [("0", 0.001)], quiet_at=quiet_at, moved=moved, **kw)


def test_measure_b_by_the_exact_unit_and_measure_d_on_hand_made_records():
    row = lambda kind, taken, **kw: Q.Row("us", kw.pop("live", False), kw.pop("closed1", False), kind=kind, taken=taken, **kw)
    records = [_rec(False, [row("rook", True, live0=True, weighty=True), row("pawn", False, closed1=True, failed_at=2), row("pawn", True, closed1=True, failed_at=3),
                            row("rook", False, closed1=True, failed_at=3, weighty=True)]),
               _rec(False, [row("rook", True, live=True, live1=True, weighty=True), row("pawn", False, closed1=True, failed_at=1)])]
    lines = Q.format_obs(records)
    get = lambda s: next(ln for ln in lines if s in ln)
    assert "above the lowest worth 1 of 1 (100.0%); the lowest 0 of 0" in get("live at level 0:")
    assert "above the lowest worth 1 of 1 (100.0%)" in get("live at level 1 (a way")
    assert "by a place to stand: taken within 4 plies, above the lowest worth 0 of 0 (n/a); the lowest 0 of 1 (0.0%)" in get("by a place to stand")
    assert "by no gain: taken within 4 plies, above the lowest worth 0 of 1 (0.0%); the lowest 1 of 1 (100.0%)" in get("by no gain")
    assert "by reach: taken within 4 plies, above the lowest worth 0 of 0 (n/a); the lowest 0 of 1 (0.0%)" in get("by reach")
    assert "all questions: taken 3 of 6" in get("all questions")
    # (d): two positions with nothing live, the quiet unit moved in one of them
    rows0 = [row("pawn", False, closed1=True)]
    recs = [_rec(True, rows0, quiet_at="a2", moved="a2"), _rec(False, rows0, quiet_at="a2", moved="b1"), _rec(True, rows0, quiet_at="a2", moved="b1"),
            _rec(False, [row("pawn", False, live=True)], quiet_at="a2", moved="a2")]
    d = next(ln for ln in Q.format_obs(recs) if ln.startswith("OBS (d)"))
    assert "nothing live at level 0 or 1: 3 " in d and "move read: 3" in d and "a move of the unit it names in 1 (33.3%)" in d
    assert "bad move follows 100.0% when it was (1), 50.0% when it was not (2)" in d


def test_measure_e_counts_the_objections_on_hand_made_records():
    recs = [_rec(True, [], obj0=True, obj2=False, weight=3.0), _rec(True, [], obj0=False, obj2=True, weight=1.0), _rec(True, [], obj0=False, obj2=False, weight=2.0),
            _rec(False, [], obj0=True, obj2=False), _rec(False, [], obj0=False, obj2=False), _rec(False, [], obj0=False, obj2=False, alt_share=0.5)]
    recs[0].alt_share, recs[1].alt_share = 0.25, 0.75
    lines = Q.format_obs(recs)
    e0 = next(ln for ln in lines if "objects at level 0 to" in ln)
    assert "to 1 of 3 bad moves (33.3%) and to 1 of 3 other moves (33.3%)" in e0
    assert "when it objects 50.0% (2), when it does not 50.0% (4)" in e0 and "holds 50.0% of the bad moves' weight (total 6.0)" in e0
    e2 = next(ln for ln in lines if "objects at level 0 or 2 to" in ln)
    assert "to 2 of 3 bad moves (66.7%) and to 1 of 3 other moves" in e2 and "holds 66.7% of the bad moves' weight" in e2
    alt = next(ln for ln in lines if "alternatives" in ln)
    assert "in 3 bad-move positions" in alt and "mean 50.0%" in alt


def test_the_objection_function_reads_a_move_that_leaves_a_unit_struck_and_unguarded(env):
    params = Q.Params()
    # white's rook stands on d2; after the move rook to d3 the black queen on d5 strikes it (the file is open) and nothing guards it
    after = "4k3/8/8/3q4/8/3R4/8/4K3 b - - 1 1"
    # the new mover is black; the original mover's rook is "theirs" in that reading
    obj0, obj2 = Q.objection(env, WORTH, params, after, set(), ("d2", "d3"))
    assert obj0 is True
    # the rook was already struck and unguarded before the move: not an objection to the move
    obj0b, _ = Q.objection(env, WORTH, params, after, {"d2"}, ("d2", "d3"))
    assert obj0b is False
    share = Q.alternatives_share(env, WORTH, params, "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", set(), "d2d5")
    assert 0.0 <= share <= 1.0


def test_the_module_names_no_object_of_one_game():
    words = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check\w*|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
    text = SOURCE.read_text()
    hit = re.search(words, text, flags=re.I)
    assert hit is None, hit and (hit.group(0), text[max(0, hit.start() - 40):hit.end() + 20])


# ---------------------------------------------------------------------------------------------------------------------------------------------- F: the goal unit against how games ended; G: the search on flagged units

def test_f_reads_how_a_short_game_ended_and_the_goal_unit_of_the_loser(tmp_path):
    (tmp_path / "games.jsonl").write_text(json.dumps({"variant": "chess", "game": 0, "start": None, "moves": ["f2f3", "e7e5", "g2g4", "d8h4"], "outcome": [0, 100], "unfinished": False}) + "\n")
    rows, endings, n = Q.goal_verdict([str(tmp_path)], Q.Env("chess"), WORTH, Q.Params())
    assert n == 4 and list(endings.values()) == [("decisive", 1)]                          # the second role (black) won
    loser = {(x["to_end"], x["read"]): x for x in rows if x["group"] == "loser"}
    assert loser[(1, "theirs")]["live"] and loser[(1, "theirs")]["scenario"] and loser[(1, "theirs")]["total_moves"] == 1     # the attacker to move, one move from the end
    assert loser[(2, "mine")]["live"] and not loser[(3, "theirs")]["live"]
    lines = Q.format_goal_verdict(rows, endings, n)
    assert "decisive 1" in lines[0]
    assert any("LOST" in ln and "read as theirs" in ln and "sign live" in ln and "1-2: 1/1 (100.0%)" in ln for ln in lines)
    assert any("sign live (level 0 or 1): live from the end back to 2 plies before the end (median; quartiles 2 and 2) in 1 games" in ln for ln in lines)
    assert Q.game_ending("chess", {"outcome": None, "unfinished": True, "moves": []}, ["white", "black"]) == ("ply cap", None)


def test_f_bands_cover_every_distance():
    assert [Q.band_of(x) for x in (1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 500)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_g_follows_a_unit_of_either_side_and_reads_the_level_2_result_of_the_flagged(env):
    from catspace.incubator.analyst.discovery import curriculum as CU
    codec = CU.codec_of("chess")
    moves = ["e2e4", "d7d5", "e4d5", "d8d5"]
    assert codec.taken_within(None, moves, 2, "d5", 4) is True              # the unit is not the mover of ply 2 and is taken at once
    assert codec.taken_within(None, moves, 2, "a7", 4) is False
    sheet = Q.run(HANGING, WORTH, env=env, max_steps=0, params=Q.Params(max_level=2, search_depth_cap=2, frames=False))
    Q.finish_levels(sheet, 1, Q.Params(max_level=2, search_depth_cap=2, frames=False))
    Q.settle_flagged(sheet, Q.Params(max_level=2, search_depth_cap=2, frames=False))
    queen = next(q for q in sheet.questions if not q.quiet and q.target.kind == "queen")
    assert len(queen.findings) == 3 and queen.findings[2].detail["mode"] == "search" and queen.findings[2].detail["depth"] == 2
    rec = Q.summarize_sheet((0, 0), False, None, sheet, taken_of=lambda place: place == "d5", follow_all=True)
    assert ("them", "flagged", "confirmed", True) in rec.units


def test_g_lines_count_taken_by_side_and_result():
    units = [("us", "flagged", "confirmed", True), ("us", "flagged", "refuted", False), ("us", "closed1", "", False), ("them", "flagged", "confirmed", True),
             ("them", "flagged", "not settled", False), ("them", "other", "", True)]
    lines = Q.format_flagged([Q.Record((0, 0), False, [], None, [], units=units)])
    get = lambda s: next(ln for ln in lines if s in ln)
    assert "mine (level 2 read as if it were their turn): units read 3; flagged (open after level 1) 2: taken 1 of 2 (50.0%)" in get("OBS (G) mine (")
    assert "theirs, flagged, level 2 confirmed: taken 1 of 1 (100.0%)" in get("theirs, flagged, level 2 confirmed")
    assert "mine, closed at level 1: taken 0 of 1 (0.0%)" in get("mine, closed at level 1")
    assert "theirs, neither flagged nor closed at level 1: taken 1 of 1" in get("theirs, neither")


# ---------------------------------------------------------------------------------------------------------------------------------------------- frames: the exchange on a unit's place, cornering, the goal unit's live sign

def _unit(env, fen, side, kind):
    r = Q.Reading(env, fen, WORTH)
    return r, next(u for u in r.units if u.side == side and u.kind == kind)


def test_the_exchange_frame_the_starter_wins(env):
    r, u = _unit(env, "3rk3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", "them", "queen")           # the queen is guarded, but it is worth more than my rook
    x = Q.exchange_stands(r, u, "us")
    assert x["source"] == "real" and x["value"] == 6 and x["line"] == ("d2d5", "d8d5")
    r2, u2 = _unit(env, HANGING, "them", "queen")
    assert Q.exchange_stands(r2, u2, "us")["value"] == 14


def test_the_exchange_frame_the_starter_loses_and_so_stops(env):
    r, u = _unit(env, "7k/8/4p3/3p4/8/8/3R4/4K3 w - - 0 1", "them", "pawn")           # a guarded pawn: taking it loses the rook for it, so the starter stops
    x = Q.exchange_stands(r, u, "us")
    assert x["value"] == 0 and x["line"][:2] == ("d2d5", "e6d5")                         # the line it would lose by is shown; the value is the exit's
    f = _level2(env, "7k/8/4p3/3p4/8/8/3R4/4K3 w - - 0 1", "them", "pawn")
    assert f.status == Q.CLOSED and "stopped" in f.text and f.detail["lost_where_it_stands"] is False


def _level2(env, fen, side, kind, params=None):
    r, qs = _first_order(env, fen)
    q = _find(qs, side, kind)
    params = params or Q.Params()
    Q.step(q, r, params), Q.step(q, r, params)
    return Q.step(q, r, params)


def test_a_unit_behind_another_on_the_same_line_decides_the_exchange_by_real_moves(env):
    behind = "3rk3/8/8/3p4/8/3R4/3R4/4K3 w - - 0 1"
    alone = "3rk3/8/8/3p4/8/3R4/8/4K3 w - - 0 1"
    r, u = _unit(env, behind, "them", "pawn")
    real = Q.exchange_stands(r, u, "us")
    assert real["source"] == "real" and real["value"] == 2                              # the second rook, behind the first, recaptures: the kernel's own moves see it
    assert Q.exchange_static(r, u, "us", Q.Params())["value"] == 0                       # the fact view does not see a unit behind another
    r1, u1 = _unit(env, alone, "them", "pawn")
    assert Q.exchange_stands(r1, u1, "us")["value"] == 0


def test_the_static_fallback_is_marked_when_the_asker_is_not_to_move(env):
    r, u = _unit(env, HANGING, "us", "rook")                                             # their queen strikes my rook, my king guards it
    x = Q.exchange_stands(r, u, "them")
    assert x["source"] == "static" and x["note"] == "read from the board as it stands, lines behind not read" and x["value"] == 0
    f = _level2(env, HANGING, "us", "rook")
    assert f.detail["mode"] == "frame" and "read from the board as it stands, lines behind not read" in f.text
    r2, u2 = _unit(env, "7k/8/8/8/8/8/q7/N3K3 w - - 0 1", "us", "knight")
    assert Q.exchange_stands(r2, u2, "them")["value"] == 4                                # unguarded and struck: lost where it stands


def test_a_cornered_unit_and_a_unit_with_one_safe_place(env):
    r, u = _unit(env, "7k/8/8/8/8/8/q7/N3K3 w - - 0 1", "us", "knight")                  # struck, unguarded, and both places it can go to are struck
    c = Q.cornering(r, u, "them")
    assert c["cornered"] and c["lost_now"] and c["open_exits"] == 0 and len(c["places"]) == 2
    r2, u2 = _unit(env, "7k/8/8/8/8/8/q7/N2K4 w - - 0 1", "us", "knight")                # my king guards c2: one place is safe
    c2 = Q.cornering(r2, u2, "them")
    assert c2["lost_now"] and not c2["cornered"] and c2["open_exits"] == 1
    r3, u3 = _unit(env, "4k3/8/8/3p4/8/8/3R4/4K3 w - - 0 1", "them", "pawn")             # the owner is not to move: its places are read from the fact view
    c3 = Q.cornering(r3, u3, "us")
    assert c3["cornered"] and c3["places"] == [("d4", True)]


def test_a_level_2_of_a_cornered_unit_says_so(env):
    f = _level2(env, "7k/8/8/8/8/8/q7/N3K3 w - - 0 1", "us", "knight")
    assert f.status == Q.LIVE and f.detail["cornered"] and "cornered" in f.text


def test_the_goal_unit_is_idle_when_no_unit_of_the_asker_is_near_and_live_when_confined_with_a_striker_near(env):
    def level1(fen, params=None):
        r, qs = _first_order(env, fen)
        q = next(x for x in qs if x.goal and x.target.side == "them")
        Q.step(q, r, params or Q.Params())
        return Q.step(q, r, params or Q.Params())
    far = level1("7k/6pp/8/8/8/8/8/K7 w - - 0 1")                                        # confined, but my only unit is far away
    assert far.status == Q.IDLE and far.detail["open_exits"] == 1 and far.detail["near"] is False
    assert level1("7k/6pp/8/8/8/8/8/K7 w - - 0 1", Q.Params(goal_near=False)).status == Q.CLOSED       # the earlier rule closed it
    near = level1("7k/6pp/8/8/8/8/8/R3K3 w - - 0 1")                                     # my rook reaches the line in one move
    assert near.status == Q.LIVE and near.detail["near"] and near.detail["open_exits"] == 1
    assert "scenarios" in near.detail                                                    # a plan is a recorded fact, not the status
