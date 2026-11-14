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
    assert re.search(r"hops away: too far in time", f.text)
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
    assert f.status == Q.CLOSED and f.detail["failed_at"] == 1 and "hops away: too far in time" in f.text


def test_link_2_a_place_to_stand_fails_where_a_cheaper_unit_strikes_the_arrival_place(env):
    f, q = _level1(env, NO_PLACE, "them", "queen")
    assert f.status == Q.CLOSED and f.detail["failed_at"] == 2
    assert "none passes all three links" in f.text and "a place to stand" in f.text and ("worth less" in f.text or "not guarded" in f.text)
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


IN_CHECK = "4k3/8/8/4q3/8/8/3R4/4K3 w - - 0 1"           # the mover's goal unit is struck: the kernel refuses the pass


def test_the_static_fallback_is_marked_when_the_pass_is_refused(env):
    r, u = _unit(env, IN_CHECK, "us", "rook")
    assert r.passed_position() is None
    x = Q.exchange_stands(r, u, "them")
    assert x["source"] == "static" and x["note"] == "read from the board as it stands, lines behind not read"
    f = _level2(env, IN_CHECK, "us", "rook")
    assert f.detail["mode"] == "frame" and "read from the board as it stands, lines behind not read" in f.text
    r1, u1 = _unit(env, HANGING, "us", "rook")                                           # a pass is given: the same question by real moves
    x1 = Q.exchange_stands(r1, u1, "them")
    assert r1.passed_position() is not None and x1["source"] == "real" and x1["value"] == 0
    r1b, u1b = _unit(env, HANGING, "us", "rook")
    r1b.passes = False                                                                   # switched off: the static form, marked
    assert Q.exchange_stands(r1b, u1b, "them")["source"] == "static"
    r2, u2 = _unit(env, "7k/8/8/8/8/8/q7/N3K3 w - - 0 1", "us", "knight")
    assert Q.exchange_stands(r2, u2, "them")["value"] == 4                                # unguarded and struck: lost where it stands


def test_passed_gives_the_same_placement_with_the_other_side_to_move_and_is_refused_where_the_kernel_refuses():
    import cs_engine as CE
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    p = CE.passed("chess", fen, [])
    assert p[0].split()[0] == fen.split()[0] and p[1] == "white" and CE.state("chess", fen, [])[1] == "black"
    assert CE.state("chess", p[0], [])[:3] == p[:3]                                       # the string is a start for the other calls
    assert CE.passed("chess", None, ["e2e4"])[0] == p[0]                                  # a start and a move list as `state` takes it
    assert CE.passed("chess", IN_CHECK, []) is None                                       # the mover is in check
    assert CE.passed("chess", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", []) is None              # the game is over


def test_a_question_on_the_movers_own_unit_is_read_by_real_moves_refuted_when_answered_confirmed_when_not(env):
    answered = _level2(env, "1r4k1/8/8/8/8/4K3/8/N7 w - - 0 1", "us", "knight")        # the knight can leave the file the rook comes to
    assert answered.status == Q.CLOSED and set(answered.detail["sources"]) >= {"given", "real"} and answered.detail["confirmed"] == 0
    lost = _level2(env, "1r4k1/8/8/5b2/2p5/8/8/N3K3 w - - 0 1", "us", "knight")          # every place the knight can go to is struck
    assert lost.status == Q.LIVE and lost.value == 4 and "real moves" in lost.text and set(lost.detail["sources"]) >= {"given", "real"}
    off = _level2(env, "1r4k1/8/8/5b2/2p5/8/8/N3K3 w - - 0 1", "us", "knight", Q.Params(passes=False))
    assert off.detail["sources"] == ["static"]                                            # --no-pass: the earlier form


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


# ---------------------------------------------------------------------------------------------------------------------------------------------- 78.41 routes: layer one (the medium), layer two (obstacles), the whole route followed, waiting, the pool, the graph

KING_ROUTE = "8/6Np/8/1P6/2p1P2K/8/8/4k3 w - - 0 1"          # my king h4 comes to g5 or h5, then h6, from where it strikes their pawn on h7: nothing stops it
OWN_IN_WAY = "4k3/8/8/3b4/8/8/2P5/N3K3 w - - 0 1"            # my knight a1 must pass c2 to strike their bishop, and my own pawn stands on c2
PAWN_TAKES = "7k/8/8/3b4/8/3p4/8/N3K3 w - - 0 1"             # the same, but their pawn on d3 takes the knight when it lands on c2
THEIR_IN_WAY = "7k/8/8/7n/1p6/p7/8/R3K3 w - - 0 1"           # my rook a1 -> a5 passes a3, where their guarded pawn stands
STRONG_PARAMS = Q.Params()


def _deep(env, fen, side, kind, place=None, params=None):
    params = params or Q.Params()
    r = Q.Reading(env, fen, WORTH)
    q = next(x for x in Q.first_order(r, params) if x.target.side == side and x.target.kind == kind and (place is None or Q.place_text(x.target.at) == place))
    Q.step(q, r, params)
    Q.step(q, r, params)
    f = Q.step(q, r, params) if q.state == "open" else None
    return r, q, f


def _ways(f, kind=None, to=None):
    return [x for x in f.detail["results"] if x.get("way") is not None and (kind is None or x["way"][1].kind == kind) and (to is None or Q.place_text(x["way"][2]) == to)]


def test_the_hop_table_gives_the_known_counts_on_the_empty_medium(env):
    m = env.medium()
    for side in ("white", "black"):
        assert m.distance("knight", side, "a1", "h8") == 6            # the far pair
        assert m.distance("rook", side, "a1", "h8") == 2
        assert m.distance("bishop", side, "a1", "h8") == 1
        assert m.distance("bishop", side, "a1", "a2") is None         # the other colour of square: never
        assert m.distance("king", side, "a1", "h8") == 7
        assert m.distance("queen", side, "a1", "b3") == 2
    assert m.distance("pawn", "white", "e2", "e5") == 3 and m.distance("pawn", "white", "e5", "e2") is None
    assert m.distance("pawn", "black", "e5", "e2") == 3
    assert m.over("rook", "white", "a1", "a5") == ("a2", "a3", "a4")
    assert m.over("knight", "white", "a1", "b3") == ()
    assert ("c2", "e3") in m.routes("knight", "white", "a1", "e3") and all(len(x) == 2 for x in m.routes("knight", "white", "a1", "e3"))
    assert m.strike_places("pawn", "white", "e5") == {"d4", "f4"}


def test_the_medium_is_read_once_per_game_and_the_move_tables_do_not_carry_it(env):
    assert env.medium() is env.medium()
    assert not any(x.startswith("hop") for x in env.vocabulary["derived"]) and env.medium_names == ["hop", "hop_take", "hop_over"]


def test_a_two_move_way_has_its_route_printed_step_by_step(env):
    r, q, f = _deep(env, KING_ROUTE, "them", "pawn", "h7")
    f1 = q.findings[1]
    assert "my king on h4 to g5, then to h6, from where it strikes their pawn on h7" in f1.text
    way = next(w for w in f1.detail["supply"] if w[0] == 2 and w[1].kind == "king")
    route = f1.detail["routes"][way[:4]][0]
    assert [Q.place_text(p) for p in route.places] == ["g5", "h6"] and not route.obstacles and len(route.stands) == 2
    sheet = Q.run(KING_ROUTE, WORTH, env=env, max_steps=60)
    Q.finish_levels(sheet, 1)
    Q.settle_flagged(sheet)
    text = "\n".join(Q.explain(sheet))
    assert re.search(r"way: my king on h4 to \w\d, then to h6, from where it strikes their pawn on h7: CONFIRMED, worth 2 against every answer", text)


def test_a_route_confirmed_against_every_answer(env):
    r, q, f = _deep(env, KING_ROUTE, "them", "pawn", "h7")
    got = _ways(f, "king", "h6")[0]
    assert got["status"] == Q.CONFIRMED_V and got["value"] > 0 and not got["breaks"] and got["nodes"] > 20 and got["line"][0] == "h4g5" or got["line"][0] == "h4h5"
    assert f.status == Q.LIVE and "the whole route followed" in f.text


def test_a_route_refuted_by_an_answer_that_takes_the_travelling_unit_names_the_answer(env):
    r, q, f = _deep(env, PAWN_TAKES, "them", "bishop")
    got = _ways(f, "knight", "b4")[0]
    assert got["status"] == Q.REFUTED_V
    assert "their pawn takes on c2 after the first step" in got["answer"] and "nothing is taken back" in got["answer"]
    assert got["breaks"][0].reason == "unit_taken" and got["breaks"][0].mover == ("d3", "c2")
    assert q.state == "closed"                                       # every way is refuted: the question closes


def test_a_route_whose_in_between_place_is_taken_is_waiting_and_says_what_it_waits_on(env):
    r, q, f = _deep(env, OWN_IN_WAY, "them", "bishop")
    got = _ways(f, "knight", "b4")[0]
    assert got["status"] == Q.WAITING_V and "the place c2 is taken" in got["answer"]
    assert ("vacated", ("c2",)) == got["alts"][0][:2] and "the place c2 is vacated" in got["alts"][0][2]
    assert got["breaks"] and got["breaks"][0].reason == "place_taken" and got["breaks"][0].kind == "delay"
    assert "c2" in got["reads"] and "a1" in got["reads"] and "d5" in got["reads"] and "b4" in got["reads"]


def test_a_question_with_only_waiting_ways_stays_open_idle_with_the_record_and_a_lowered_urgency(env):
    r, q, f = _deep(env, OWN_IN_WAY, "them", "bishop")
    assert f.status == Q.IDLE and q.state == "open" and len(q.waiting) == 2 and f.detail["confirmed"] == 0 and f.detail["refuted"] == 1
    assert all(x["status"] == Q.WAITING_V and x["alts"] and x["breaks"] for x in q.waiting)
    costs = Q._Costs(Q.Params())
    before = q.urgency
    Q._record(Q.Sheet(OWN_IN_WAY, [q]), f, costs, Q.Params())
    assert q.urgency == before * Q.Params().urgency_down * Q.Params().waiting_urgency
    r2, q2, f2 = _deep(env, PAWN_TAKES, "them", "bishop")
    assert f2.status == Q.CLOSED and q2.state == "closed" and not q2.waiting


def test_the_node_cap_gives_not_settled_never_refuted(env):
    r, q, f = _deep(env, KING_ROUTE, "them", "pawn", "h7", Q.Params(route_nodes=3))
    got = _ways(f, "king", "h6")[0]
    assert got["status"] == Q.UNSETTLED_V and got["open"] and f.status == Q.IDLE and q.state == "open"
    assert not any(x["status"] == Q.REFUTED_V for x in f.detail["results"][1:] if x["nodes"] >= 3 and x["status"] != Q.UNSETTLED_V and x["way"][1].kind == "king")


def test_a_route_with_an_own_unit_in_the_way_asks_whether_it_can_move_away_and_graphs_it(env):
    r, q, f = _deep(env, OWN_IN_WAY, "them", "bishop")
    g = r.graph
    ifs = [(e, g.nodes[e.dst]) for e in g.edges if e.kind == "if_removed"]
    assert ifs and any("it can move away, one move more" in n.answer for _e, n in ifs)
    e, n = next((e, n) for e, n in ifs if "pawn" in n.answer)
    assert e.gone == "my pawn on c2" and "3 moves with the one that clears it" in e.then and n.verdict == "movable"
    assert any(e.kind == "blocked_by" for e in g.edges)


def test_a_route_blocked_by_the_other_sides_unit_that_cannot_be_taken_is_dead_with_that_unit_named(env):
    r, q, f = _deep(env, THEIR_IN_WAY, "them", "knight")
    got = _ways(f, "rook", "a5")[0]
    assert got["status"] == Q.REFUTED_V and "their pawn on a3 will not move for us and cannot be taken" in got["answer"]
    assert any(n.verdict == "dead" and n.kind == "follow_up" for n in r.graph.nodes)
    clear = _ways(f, "rook", "h1")                                     # the first rank is shut by my own goal unit: the kernel will not read the board without it, so it is moved aside instead
    assert clear and clear[0]["status"] != Q.REFUTED_V and "no single unit accounts" not in clear[0]["answer"]


def test_explain_chain_prints_because_and_if_gone(env):
    sheet = Q.run(OWN_IN_WAY, WORTH, env=env, max_steps=60)
    Q.finish_levels(sheet, 1)
    Q.settle_flagged(sheet)
    q = next(x for x in sheet.questions if x.target and x.target.kind == "bishop" and x.target.side == "them")
    lines = Q.explain_chain(sheet, q)
    text = "\n".join(lines)
    assert lines[0].startswith("their bishop on d5, asked by mine") and "because" in text
    assert "if my pawn on c2 were gone then the route would take 2 hops" in text
    assert "needs:" in text and "blocked_by:" in text and "if_removed:" in text and "waits_on:" in text and "breaks:" in text
    sheet2 = Q.run(PAWN_TAKES, WORTH, env=env, max_steps=60)
    Q.finish_levels(sheet2, 1)
    Q.settle_flagged(sheet2)
    q2 = next(x for x in sheet2.questions if x.target and x.target.kind == "bishop" and x.target.side == "them")
    assert "stopped_by:" in "\n".join(Q.explain_chain(sheet2, q2))


def test_the_medium_sees_a_unit_the_frozen_board_hid_and_the_old_form_stays_behind_a_flag(env):
    new_f, _ = _level1(env, OWN_IN_WAY, "them", "bishop")
    old_f, _ = _level1(env, OWN_IN_WAY, "them", "bishop", Q.Params(geometry=False))
    assert new_f.status == Q.LIVE and ("knight", "a1") in new_f.detail["reach_geo"] and ("knight", "a1") not in new_f.detail["reach_old"]
    assert old_f.status == Q.CLOSED and "none passes all three links" in old_f.text             # the board frozen: the pawn on c2 makes the knight's way vanish, only the pawn's own ways are read


def test_a_unit_farther_than_reach_k_hops_is_closed_as_too_far_in_time_with_the_count(env):
    f, q = _level1(env, "7k/p7/8/8/8/8/8/K7 w - - 0 1", "them", "pawn")
    assert f.status == Q.CLOSED and f.detail["too_far"] == 5 and "my king on a1 is 5 hops away: too far in time" in f.text
    k3, _ = _level1(env, "7k/p7/8/8/8/8/8/K7 w - - 0 1", "them", "pawn", Q.Params(reach_k=5))
    assert k3.status == Q.LIVE


def test_reach_three_works_when_set(env):
    fen = "k7/6q1/8/8/8/8/8/N3K3 w - - 0 1"
    r, q, f = _deep(env, fen, "them", "queen", params=Q.Params(reach_k=3))
    ways3 = [w for w in q.findings[1].detail["supply"] if w[0] == 3]
    assert q.findings[1].status == Q.LIVE and ways3 and all(len(q.findings[1].detail["routes"][w[:4]][0].places) == 3 for w in ways3)
    assert f is not None and any(x["way"] is not None and x["way"][0] == 3 for x in f.detail["results"])
    assert _level1(env, fen, "them", "queen")[0].status == Q.CLOSED                        # at reach 2 it is too far in time


def test_the_weak_places_around_the_goal_unit_are_counted(env):
    pos = "q6k/R7/8/8/8/8/8/R3K3 w - - 0 1"
    r = Q.Reading(env, pos, WORTH)
    q = next(x for x in Q.first_order(r) if x.goal and x.target.side == "them")
    weak = Q.weak_places(q, r, Q.Params())
    assert [(w["place"], len(w["guards"]), len(w["strikers"]), w["need"]) for w in weak] == [("h8", 1, 0, 2), ("g8", 1, 0, 2)]
    assert all(len(w["candidates"]) >= 1 for w in weak)
    Q.step(q, r)
    Q.step(q, r)
    nodes = [n for n in r.graph.nodes if n.kind == "weak_place"]
    assert len(nodes) == 2 and any(e.kind == "needs" and r.graph.nodes[e.dst].kind == "weak_place" for e in r.graph.edges)
    assert "bring 2 more units to strike h8" in nodes[0].asks


# ---- the pool across moves: the fact store as the sheet's memory

def _read(env, pool, fen, params=None):
    pool.advance(fen)
    sheet = Q.run(fen, WORTH, env=env, max_steps=60, pool=pool, params=params or Q.Params())
    Q.finish_levels(sheet, 1, params or Q.Params())
    Q.settle_flagged(sheet, params or Q.Params())
    return sheet


def _way_fact(pool, verdict, unit_place=None):
    return next(f for f in pool.ways() if f.payload["verdict"]["verdict"] == verdict and (unit_place is None or f.key[2] == unit_place))


def test_the_pool_is_the_fact_store_and_files_claims_kills_and_gaps(env):
    from catspace.incubator.analyst.discovery.facts import FactStore
    pool = Q.Pool(env.codec)
    assert isinstance(pool.store, FactStore)
    _read(env, pool, KING_ROUTE)
    assert _way_fact(pool, Q.CONFIRMED_V).kind == "claim" and _way_fact(pool, Q.REFUTED_V).kind == "kill"
    pool2 = Q.Pool(env.codec)
    _read(env, pool2, OWN_IN_WAY)
    gap = _way_fact(pool2, Q.WAITING_V)
    assert gap.kind == "gap" and gap.resume["route"] == ["c2", "b4"] or gap.resume["route"] == ["c2", "e3"]
    assert gap.resume["node_cap"] == Q.Params().route_nodes and gap.payload["waits_on"] == ["the place c2 is vacated"] and gap.payload["breaks"]


def test_the_pool_serves_a_fact_a_move_elsewhere_did_not_flag(env):
    pool = Q.Pool(env.codec)
    _read(env, pool, KING_ROUTE)
    assert pool.new >= 1 and pool.kept == 0
    rec = _way_fact(pool, Q.CONFIRMED_V)
    assert {"h7", "h4", "h6"} <= rec.reads and rec.key[0] == "h7" and rec.sig
    after = "8/6Np/1P6/8/2p1P2K/8/8/4k3 b - - 0 1"                      # my pawn b5 -> b6
    assert not (env.codec.changed_places(KING_ROUTE, after) & rec.reads)
    flagged = pool.advance(after)
    assert rec.id not in flagged and rec.id not in pool.store.stale
    sheet = Q.run(after, WORTH, env=env, max_steps=60, pool=pool)
    Q.finish_levels(sheet, 1)
    Q.settle_flagged(sheet)
    assert pool.kept >= 1 and pool.reread == 0 and not rec.superseded      # served as it is, not searched again


def test_the_pool_flags_what_a_move_touched_and_rereads_it_when_a_question_asks(env):
    pool = Q.Pool(env.codec)
    _read(env, pool, OWN_IN_WAY)
    rec = _way_fact(pool, Q.WAITING_V, "a1")
    assert "c2" in rec.reads
    after = "4k3/8/8/3b4/8/2P5/8/N3K3 b - - 0 1"                        # the pawn on c2 steps to c3: the place the way waited on is vacated
    assert env.codec.changed_places(OWN_IN_WAY, after) & rec.reads
    flagged = pool.advance(after)
    assert rec.id in flagged and rec.id in pool.store.stale and pool.queue()[0].id in {f for f in pool.store.stale}
    sheet = Q.run(after, WORTH, env=env, max_steps=60, pool=pool)
    Q.finish_levels(sheet, 1)
    Q.settle_flagged(sheet)
    assert pool.reread >= 1
    q = next(x for x in sheet.questions if x.target and x.target.kind == "bishop")
    got = _ways([x for x in q.findings if x.level >= 2][0], "knight", "b4")[0]
    assert got["status"] != Q.WAITING_V and got.get("pool") == "reread"


def test_the_pool_audit_compares_a_served_fact_with_a_fresh_reading(env):
    pool = Q.Pool(env.codec)
    pool.audit_every = 1
    _read(env, pool, KING_ROUTE)
    _read(env, pool, "8/6Np/1P6/8/2p1P2K/8/8/4k3 b - - 0 1")
    assert pool.audited >= 1 and pool.audit_differed == 0


def test_the_graph_is_filed_as_facts_with_edges_in_just_and_the_bounds_in_the_payload(env):
    pool = Q.Pool(env.codec)
    sheet = _read(env, pool, OWN_IN_WAY)
    n = pool.file_graph(sheet, legacy=True)                                 # the first form (every node a fact); the default files one fact a question
    assert n == len(sheet.graph.nodes)
    waits = [f for f in pool.store.facts.values() if f.payload.get("node") == "way" and f.payload["verdict"] == Q.WAITING_V]
    assert waits and all(f.kind == "gap" and f.just and ("needs" in [k for k, _s in f.payload["edges"]]) and f.stake == round(f.payload["stake_high"]) for f in waits)
    assert any(f.payload["node"] == "question" for f in pool.store.facts.values() if f.id in {j for w in waits for j in w.just})


def test_the_store_is_imported_lazily_and_facts_py_is_not_edited():
    text = SOURCE.read_text()
    assert "import facts" in text and "from catspace.incubator.analyst.discovery import facts" not in text.split("class Pool")[0]


# ---------------------------------------------------------------------------------------------------------------------------------------------- the counterfactual obstacles, the cost of leaving, the bounds

def _route_of(env, fen, side, kind, unit_kind, to, place=None, hops=None):
    r = Q.Reading(env, fen, WORTH)
    q = next(x for x in Q.first_order(r) if x.target.side == side and x.target.kind == kind and (place is None or Q.place_text(x.target.at) == place))
    Q.step(q, r)
    f1 = Q.step(q, r)
    way = next(w for w in f1.detail["supply"] if w[1].kind == unit_kind and Q.place_text(w[2]) == to and (hops is None or w[0] == hops))
    return r, q, way, f1.detail["routes"][way[:4]][0]


def test_a_step_blocked_by_a_unit_on_the_line_is_explained_by_that_units_removal(env):
    r, q, way, route = _route_of(env, THEIR_IN_WAY, "them", "knight", "rook", "a5")
    obs, cut = Q.counterfactual_obstacles(r, q, way, route, Q.Params(), r.position)
    assert not cut and [(o.unit.kind, Q.place_text(o.unit.at), o.kind) for o in obs] == [("pawn", "a3", "stands on a place passed over")]
    assert obs[0].note == "without their pawn on a3 the step from a1 to a5 can be made, read on the board without it" and obs[0].hypothetical and not obs[0].dead
    assert r.cf_log and r.cf_log[0]["found"] == 1 and r.cf_log[0]["calls"] > 5


def test_a_step_the_unit_may_not_make_because_it_would_uncover_its_goal_unit_is_explained_by_the_other_sides_unit(env):
    fen = "4r1k1/8/8/n7/8/8/4R3/4K3 w - - 0 1"                # my rook on e2 is tied to my king on e1 by their rook on e8
    r = Q.Reading(env, fen, WORTH)
    q = next(x for x in Q.first_order(r) if x.target.side == "them" and x.target.kind == "knight")
    Q.step(q, r)
    f1 = Q.step(q, r)
    way = next(w for w in f1.detail["supply"] if w[1].kind == "rook" and Q.place_text(w[2]) == "a2")
    route = f1.detail["routes"][way[:4]][0]
    assert "e2a2" not in r.legal                                # the kernel does not offer the step
    obs, _ = Q.counterfactual_obstacles(r, q, way, route, Q.Params(), r.position)
    assert {(o.unit.side, o.unit.kind, Q.place_text(o.unit.at)) for o in obs} == {("them", "rook", "e8"), ("us", "king", "e1")}               # the tying rook; and my goal unit, which can step aside (the kernel will not read the board without it)
    rook = next(o for o in obs if o.unit.kind == "rook")
    assert "without their rook on e8 the step from e2 to a2 can be made" in rook.note and rook.kind in ("other", "strikes the asker's goal unit")
    assert any("my king on e1 moved aside to" in o.note for o in obs if o.unit.kind == "king")


def test_a_step_that_no_single_removal_explains_is_recorded_as_such(env):
    r, q, way, route = _route_of(env, "7k/8/8/7n/1p6/p7/8/R3K3 w - - 0 1", "them", "knight", "rook", "h1")
    off = Q.Params(aside_places=0)                                    # my own goal unit stands on the line: the board without it is no position, and with moving aside off nothing explains it
    obs, _ = Q.counterfactual_obstacles(r, q, way, route, off, r.position)
    assert len(obs) == 1 and obs[0].kind == "none" and obs[0].dead and obs[0].unit is None
    assert "no single unit accounts for the step from a1 to h1" in obs[0].note and "rejected by the kernel as not a position" in obs[0].note
    assert r.cf_log[-1]["words"] == ["none"] and r.cf_log[-1]["rejected"] >= 1


def test_a_refuted_route_whose_stopping_unit_once_removed_leaves_it_confirmed_gets_the_if_removed_edge_with_the_value(env):
    r, q, f = _deep(env, "7k/3b2n1/8/8/4K3/8/2Q5/5R2 w - - 0 1", "us", "queen")
    got = _ways(f, "bishop", "f5")[0]
    assert got["status"] == Q.REFUTED_V and got["if_removed_value"] == 14.0 and got["without_stopper"]["unit"].kind == "rook"
    edges = [(e, r.graph.nodes[e.dst]) for e in r.graph.edges if e.kind == "if_removed" and "without my rook on f1 this is worth 14" in e.then]
    assert {n.asks for _e, n in edges} == {"can it be taken", "can it be driven off", "can it be drawn away"}
    assert {n.asks: n.verdict for _e, n in edges}["can it be driven off"] == Q.UNREAD and {n.asks: n.verdict for _e, n in edges}["can it be drawn away"] == Q.UNREAD


def test_a_route_blocked_by_the_askers_own_goal_unit_is_explained_by_moving_it_aside_and_asks_whether_it_can_move_away(env):
    r, q, way, route = _route_of(env, THEIR_IN_WAY, "them", "knight", "rook", "h1")
    obs, _ = Q.counterfactual_obstacles(r, q, way, route, Q.Params(), r.position)
    assert [(o.unit.kind, o.unit.side, Q.place_text(o.unit.at), o.dead, o.hypothetical) for o in obs] == [("king", "us", "e1", False, True)]
    assert re.search(r"read on the board with my king on e1 moved aside to [a-h]\d$", obs[0].note) and "e1 moved aside to d1" not in obs[0].note and "moved aside to f1" not in obs[0].note   # not a place of the route
    assert r.cf_log[0]["rejected"] >= 1 and r.cf_log[0]["aside"] == 1 and r.cf_log[0]["found"] == 1
    ans = Q.ask_obstacle(r, q, way, route, obs[0], Q.Params(), r.position)
    assert ans["verdict"] == "movable" and "my king on e1: it can move away, one move more" in ans["text"] and ans["more"] == 1
    r2, q2, f = _deep(env, THEIR_IN_WAY, "them", "knight")
    assert any(n.kind == "follow_up" and n.asks == "can it move away" and "my king on e1" in n.answer for n in r2.graph.nodes)


def test_a_route_blocked_by_the_other_sides_goal_unit_is_dead_with_that_unit_named(env):
    fen = "8/8/3r4/4k3/4P3/8/8/K7 w - - 0 1"                   # my pawn on e4 may not step to e5: their goal unit stands there
    r, q, way, route = _route_of(env, fen, "them", "rook", "pawn", "e5")
    obs, _ = Q.counterfactual_obstacles(r, q, way, route, Q.Params(), r.position)
    assert [(o.unit.side, o.unit.kind, o.kind, o.dead) for o in obs] == [("them", "king", "stands on the landing place", False)]
    assert "their king on e5 moved aside to" in obs[0].note
    ans = Q.ask_obstacle(r, q, way, route, obs[0], Q.Params(), r.position)
    assert ans["verdict"] == "dead" and "their king on e5 will not move for us and cannot be taken" in ans["text"] and ans["more"] == 0
    assert Q.counterfactual_obstacles(r, q, way, route, Q.Params(aside_places=0), r.position)[0][0].dead


def _leaving_case(env, fen, kind, mv, full=False):
    r = Q.Reading(env, fen, WORTH)
    q = next(x for x in Q.first_order(r) if x.target.side == "them" and x.target.kind == "pawn")
    unit = next(u for u in r.units if u.side == "us" and u.kind == kind)
    return Q.cost_of_leaving(r, q, unit, mv, r.position, Q.Params(leaving_full=full)), r, q, unit


def _duty_set(cost):
    return sorted((d["unit"].kind, d["stake"], d["kind"], d["value"]) for d in cost["duties"])


def test_the_narrow_cost_of_leaving_finds_a_dropped_guard_and_a_dropped_shield_and_equals_the_full_reread(env):
    for fen, kind, mv, held in ((DUTY, "knight", "b3a5", "rook"), ("3r2k1/8/2p5/8/3B4/8/8/3QK3 w - - 0 1", "bishop", "d4e5", "queen")):
        narrow, r, q, unit = _leaving_case(env, fen, kind, mv)
        full, *_ = _leaving_case(env, fen, kind, mv, full=True)
        assert [d["unit"].kind for d in narrow["duties"]] == [held] and _duty_set(narrow) == _duty_set(full), fen
        cands, _sign = Q._leaving_candidates(r, q, unit, mv)
        assert held in {c.kind for c in cands}
    assert _duty_set(_leaving_case(env, DUTY_GUARDED, "knight", "b3a5")[0]) == _duty_set(_leaving_case(env, DUTY_GUARDED, "knight", "b3a5", full=True)[0]) == []


def test_the_narrow_cost_of_leaving_reads_fewer_units_and_the_audit_logs_a_comparison(env):
    r = Q.Reading(env, DUTY, WORTH)
    q = next(x for x in Q.first_order(r) if x.target.side == "them" and x.target.kind == "pawn")
    knight = next(u for u in r.units if u.kind == "knight")
    cands, sign = Q._leaving_candidates(r, q, knight, "b3a5")
    assert {c.kind for c in cands} == {"rook"}                   # the knight guards the rook; nothing else of mine is touched
    Q.cost_of_leaving(r, q, knight, "b3a5", r.position, Q.Params(leaving_audit_every=1))
    assert len(r.leaving_audit) == 1 and r.leaving_audit[0]["differ"] is False and r.leaving_audit[0]["missed"] == []


def test_the_pairs_are_tried_only_when_asked_for(env):
    assert Q.Params().pair_cap == 0


DUTY = "7k/3r4/8/6p1/3R4/1N6/8/6K1 w - - 0 1"           # my knight on b3 guards my rook on d4, which their rook on d7 strikes; the knight's way to the pawn is clear
DUTY_GUARDED = "7k/3r4/8/6p1/3R4/1N6/5B2/6K1 w - - 0 1"  # the same with my bishop on f2 guarding the rook too


def test_a_unit_whose_leaving_lets_a_heavier_unit_be_taken_is_waiting_with_the_duty_named(env):
    r, q, f = _deep(env, DUTY, "them", "pawn")
    got = _ways(f, "knight", "f3")[0]
    assert got["status"] == Q.WAITING_V and got["duties"][0]["unit"].kind == "rook" and got["duties"][0]["stake"] == 8
    assert "my knight on b3 holds my rook on d4" in got["answer"] and ("duty", ("d4",), "something else holds my rook on d4") in got["alts"]
    assert any(e.kind == "holds" and "my knight on b3" in r.graph.nodes[e.src].asks and "it holds my rook on d4" in r.graph.nodes[e.dst].asks for e in r.graph.edges)
    assert any(n.asks == "can another unit of mine take over the duty" and "not read" in n.answer for n in r.graph.nodes)
    assert any(n.asks == "what obstacles does it meet on its way" and n.verdict == Q.UNREAD for n in r.graph.nodes)          # the route questions are not read: leaving drops more than the attack could win


def test_a_confirmed_way_is_held_back_by_a_duty_and_not_by_a_second_guard(env):
    fake = lambda: {"way": None, "status": Q.CONFIRMED_V, "verdict": Q.CONFIRMED_V, "value": 2.0, "line": (), "alts": [], "breaks": [], "route": None}
    for fen, held in ((DUTY, True), (DUTY_GUARDED, False)):
        r = Q.Reading(env, fen, WORTH)
        q = next(x for x in Q.first_order(r) if x.target.side == "them" and x.target.kind == "pawn")
        Q.step(q, r)
        f1 = Q.step(q, r)
        way = next(w for w in f1.detail["supply"] if w[1].kind == "knight" and Q.place_text(w[2]) == "f3")
        entry = fake()
        entry["route"] = f1.detail["routes"][way[:4]][0]
        Q.apply_duties(r, q, way, entry, Q.Params())
        assert (entry["status"] == Q.WAITING_V) == held and bool(entry["duties"]) == held, fen
        assert entry["status"] == (Q.WAITING_V if held else Q.CONFIRMED_V)


def test_a_unit_shielding_a_heavier_unit_on_a_line_holds_it_with_no_special_case(env):
    fen = "3r2k1/8/2p5/8/3B4/8/8/3QK3 w - - 0 1"             # my bishop on d4 stands between their rook on d8 and my queen on d1; every step of the bishop is legal
    r = Q.Reading(env, fen, WORTH)
    q = next(x for x in Q.first_order(r) if x.target.side == "them" and x.target.kind == "pawn")
    bishop = next(u for u in r.units if u.kind == "bishop")
    cost = Q.cost_of_leaving(r, q, bishop, "d4e5", r.position, Q.Params())
    assert cost["read"] and cost["mark"] == "read after the step"
    assert [(d["unit"].kind, d["stake"], d["kind"]) for d in cost["duties"]] == [("queen", 14.0, "newly lost")]
    wo = Q.cost_of_leaving(r, q, bishop, None, r.position, Q.Params())                 # a step that is not legal: the board without the unit, marked
    assert wo["mark"].startswith("read on the board without it") and [d["unit"].kind for d in wo["duties"]] == ["queen"]


def test_every_node_carries_a_stake_bound_that_closes_as_answers_arrive_and_explain_prints_it(env):
    r, q, f = _deep(env, KING_ROUTE, "them", "pawn", "h7")
    g = r.graph
    assert all(n.stake_high is not None and 0 <= n.stake_low <= n.stake_high and n.chance is None for n in g.nodes)
    qn = g.nodes[q.node]
    assert qn.stake_low == qn.stake_high == f.value                                      # confirmed: both ends at the value
    way = next(n for n in g.nodes if n.kind == "way" and n.verdict == Q.CONFIRMED_V)
    assert way.stake_low == way.stake_high == 2.0 and way.tried > 20 and way.cut == 0
    refuted = next(n for n in g.nodes if n.kind == "way" and n.verdict == Q.REFUTED_V)
    assert refuted.stake_low == refuted.stake_high == 0.0
    r2, q2, f2 = _deep(env, OWN_IN_WAY, "them", "bishop")
    qn2 = r2.graph.nodes[q2.node]
    assert qn2.stake_low == 0.0 and qn2.stake_high == q2.stake                           # nothing settled: 0 to the target's stake
    sheet = Q.run(OWN_IN_WAY, WORTH, env=env, max_steps=60)
    Q.finish_levels(sheet, 1)
    Q.settle_flagged(sheet)
    text = "\n".join(Q.explain_chain(sheet, next(x for x in sheet.questions if x.target and x.target.kind == "bishop" and x.target.side == "them")))
    assert "worth between 0 and 5" in text
    r3, q3, f3 = _deep(env, KING_ROUTE, "them", "pawn", "h7", Q.Params(route_nodes=3))
    capped = next(n for n in r3.graph.nodes if n.kind == "way" and n.cut == 1)
    assert capped.stake_low == 0.0 and capped.stake_high == q3.stake and capped.tried >= 3     # cut by the cap: the bounds stay as they were, the tries are on the node


# ---------------------------------------------------------------------------------------------------------------------------------------------- 78.42 third form: a step blocked by several units, the chance of a node, the clock

TWO_IN_LINE = "7k/8/8/7n/1p6/p7/p7/R3K3 w - - 0 1"       # my rook a1 -> a5: their pawns on a2 and a3 both stand on the line


def _two_in_line(env, params=None):
    r = Q.Reading(env, TWO_IN_LINE, WORTH)
    q = next(x for x in Q.first_order(r) if x.target.side == "them" and x.target.kind == "knight")
    Q.step(q, r)
    f1 = Q.step(q, r)
    way = next(w for w in f1.detail["supply"] if w[1].kind == "rook" and Q.place_text(w[2]) == "a5")
    return r, q, way, f1.detail["routes"][way[:4]][0]


def test_a_step_blocked_by_two_units_is_explained_by_removing_both_and_each_must_go(env):
    r, q, way, route = _two_in_line(env)
    obs, _ = Q.counterfactual_obstacles(r, q, way, route, Q.Params(), r.position)
    assert [(Q.place_text(o.unit.at), o.kind, o.joint, o.dead) for o in obs] == [("a2", "one of several units, each of which must go", ("a2", "a3"), False), ("a3", "one of several units, each of which must go", ("a2", "a3"), False)]
    assert "only with all 2 of their pawn on a2, their pawn on a3 out of the way" in obs[0].note and obs[0].hypothetical
    assert r.cf_log[-1]["found"] == 0 and r.cf_log[-1]["joint"] == 2 and r.cf_log[-1]["joint_state"] == "joint"


def test_the_earlier_none_is_reachable_by_a_flag(env):
    r, q, way, route = _two_in_line(env)
    obs, _ = Q.counterfactual_obstacles(r, q, way, route, Q.Params(joint=False), r.position)
    assert len(obs) == 1 and obs[0].kind == "none" and obs[0].dead and r.cf_log[-1]["joint_state"] == ""


def test_more_units_than_the_cap_is_a_refutation_by_cost_said_in_words(env):
    r, q, way, route = _two_in_line(env)
    obs, _ = Q.counterfactual_obstacles(r, q, way, route, Q.Params(max_joint=1), r.position)
    assert len(obs) == 1 and obs[0].dead and obs[0].unit is None and "more than 1 units in the way" in obs[0].note and "refuted by cost" in obs[0].note
    assert r.cf_log[-1]["joint_state"] == "cap"
    assert Q.Params().max_joint == 3


def test_the_set_is_asked_cheapest_first_and_a_dead_member_kills_the_route_and_is_named(env):
    r, q, f = _deep(env, TWO_IN_LINE, "them", "knight")
    got = _ways(f, "rook", "a5")[0]
    assert got["status"] == Q.REFUTED_V and "their pawn on a3 will not move for us and cannot be taken" in got["answer"]
    assert r.joint_log and all(x["dead"] and x["members"] == 2 for x in r.joint_log)
    assert any(n.kind == "obstacle" and n.asks.startswith("several units in the way, each of which must go") for n in r.graph.nodes)
    ask = [n for n in r.graph.nodes if n.kind == "follow_up" and n.target is not None and Q.place_text(n.target.at) in ("a2", "a3")]
    assert {Q.place_text(n.target.at): n.verdict for n in ask} == {"a2": "takeable", "a3": "dead"}


def test_the_moves_to_clear_a_set_add_up(env):
    fen = "7k/8/8/7n/8/N7/N7/R3K3 w - - 0 1"                  # my own knights on a2 and a3 stand on the rook's line: each must move away
    r = Q.Reading(env, fen, WORTH)
    q = next(x for x in Q.first_order(r) if x.target.side == "them" and x.target.kind == "knight")
    Q.step(q, r)
    f1 = Q.step(q, r)
    way = next(w for w in f1.detail["supply"] if w[1].kind == "rook" and Q.place_text(w[2]) == "a5")
    route = f1.detail["routes"][way[:4]][0]
    obs, _ = Q.counterfactual_obstacles(r, q, way, route, Q.Params(), r.position)
    assert [o.joint for o in obs] == [("a2", "a3")] * 2 and not any(o.dead for o in obs)
    r2, q2, f = _deep(env, fen, "them", "knight")
    assert r2.joint_log and any(x["moves_more"] == 2 and not x["dead"] for x in r2.joint_log)
    assert any("the moves add up" in e.then for e in r2.graph.edges)


def test_the_format_counts_the_former_none_steps_by_what_explains_them(env):
    r, q, way, route = _two_in_line(env)
    Q.counterfactual_obstacles(r, q, way, route, Q.Params(), r.position)
    rec = Q.Record(("g", 0), False, [], None, [], cf=list(r.cf_log), joint=list(r.joint_log))
    text = "\n".join(Q.format_counterfactual([rec]))
    assert "(the former \"none\"): 1; explained by removing the set of units the first layer names together 1 (100.0%, by size 2 units 1)" in text


# ---- the key, the fit, the merge, the smoothing

def _fake_question(owner_side, stake=10.0, rank=1, kind="rook", place=("a", "1")):
    return Q.Question(Q.Unit(owner_side, kind, place), Q.OTHER if owner_side == Q.SELF else Q.SELF, stake, rank)


def _finding(q, level, status, detail=None):
    f = Q.Finding(q, level, status, "", 0.0, detail=detail or {})
    q.findings.append(f)
    return f


def test_the_key_is_owner_verdict_moves_route_and_reads_the_deepest_level():
    q = _fake_question(Q.SELF)
    assert Q.question_key(q) == ("mine", "unread", "-", "-")
    _finding(q, 0, Q.LIVE)
    assert Q.question_key(q)[:2] == ("mine", "live0")
    shut = _finding(q, 1, Q.CLOSED, {"failed_at": 2, "supply": []})
    assert Q.question_key(q, 1) == ("mine", "live0", "-", "-") and Q.question_key(q, -1)[1] == "unread"
    q2 = _fake_question(Q.OTHER)
    _finding(q2, 0, Q.IDLE)
    _finding(q2, 1, Q.CLOSED, {"failed_at": 3, "supply": []})
    assert Q.question_key(q2) == ("theirs", "closed3", "-", "-")
    way = (2, Q.Unit(Q.SELF, "rook", ("a", "1")), ("a", "5"), 0)
    q3 = _fake_question(Q.OTHER)
    _finding(q3, 0, Q.IDLE)
    _finding(q3, 1, Q.LIVE, {"supply": [way], "routes": {way[:4]: []}})
    assert Q.question_key(q3) == ("theirs", "live1", "2", "clear")
    results = [{"way": None, "status": Q.REFUTED_V}, {"way": way, "status": Q.WAITING_V}]
    _finding(q3, 2, Q.IDLE, {"results": results})
    assert Q.question_key(q3) == ("theirs", "waiting", "2", "clear") and Q.question_key(q3, 1)[1] == "live1"
    q3.findings[-1].detail["results"] = [{"way": None, "status": Q.CONFIRMED_V}, {"way": way, "status": Q.REFUTED_V}]
    assert Q.question_key(q3)[1:3] == ("lost", "0")
    q3.findings[-1].detail["results"] = [{"way": None, "status": Q.REFUTED_V}, {"way": way, "status": Q.CONFIRMED_V}]
    assert Q.question_key(q3)[1:3] == ("confirmed", "2")
    q3.findings[-1].detail["results"] = [{"way": None, "status": Q.REFUTED_V}, {"way": way, "status": Q.REFUTED_V}]
    assert Q.question_key(q3)[1] == "refuted"


def test_the_route_word_is_clear_own_or_theirs():
    way = (1, Q.Unit(Q.SELF, "rook", ("a", "1")), ("a", "5"), 0)
    mine, theirs = Q.Obstacle("other", Q.Unit(Q.SELF, "pawn", ("a", "2")), "a2", 0), Q.Obstacle("other", Q.Unit(Q.OTHER, "pawn", ("a", "3")), "a3", 0)
    for obstacles, word in (((), "clear"), ((mine,), "own"), ((mine, theirs), "theirs")):
        q = _fake_question(Q.OTHER)
        _finding(q, 1, Q.LIVE, {"supply": [way], "routes": {way[:4]: [Q.Route(way[1], (("a", "5"),), ((True, ""),), "strike", (), obstacles)]}})
        assert Q.question_key(q)[3] == word
    q = _fake_question(Q.OTHER)
    _finding(q, 1, Q.LIVE, {"supply": [way], "routes": {way[:4]: [Q.Route(way[1], (("a", "5"),), ((True, ""),), "strike", (), (Q.Obstacle("no unit", None, "a5", 0, dead=True),))]}})
    assert Q.question_key(q)[3] == "theirs"


RECORDS = [{"keys": [["mine", "unread", "-", "-"], ["mine", "waiting", "2", "clear"]], "out": {4: True, 8: True}},
           {"keys": [["mine", "unread", "-", "-"], ["mine", "waiting", "2", "clear"]], "out": {4: False, 8: True}},
           {"keys": [["mine", "unread", "-", "-"], ["mine", "lost", "0", "clear"]], "out": {4: True, 8: None}},
           {"keys": [["theirs", "unread", "-", "-"]], "out": {"4": False, "8": False}}]


def test_the_fit_counts_taken_and_total_per_key_and_horizon_and_skips_what_the_game_did_not_say():
    cells = Q.cells_from_records(RECORDS)
    assert cells[("mine", "unread", "-", "-")] == {4: [2, 3], 8: [2, 2]}
    assert cells[("mine", "waiting", "2", "clear")] == {4: [1, 2], 8: [2, 2]}
    assert cells[("mine", "lost", "0", "clear")] == {4: [1, 1], 8: [0, 0]}
    assert cells[("theirs", "unread", "-", "-")] == {4: [0, 1], 8: [0, 1]}


def test_the_merge_of_two_fits_is_the_sum_and_equals_the_fit_of_both():
    a, b = Q.cells_from_records(RECORDS[:2]), Q.cells_from_records(RECORDS[2:])
    assert Q.merge_cells([a, b]) == Q.cells_from_records(RECORDS)
    assert Q.merge_cells([]) == {}
    h = Q.merge_headers([{"positions": 3, "unreadable": 1, "records": 5, "seconds": 2.0, "shards": [[0, 2]]}, {"positions": 4, "unreadable": 0, "records": 6, "seconds": 3.0, "shards": [[1, 2]]}])
    assert (h["positions"], h["unreadable"], h["records"], h["seconds"], h["shards"]) == (7, 1, 11, 5.0, [[0, 2], [1, 2]])


def test_the_chance_is_smoothed_toward_the_next_coarser_key_and_an_unseen_key_takes_the_coarser_chance():
    t = Q.Chances(Q.cells_from_records(RECORDS))
    root = (4 + 1) / (7 + 2)                                                       # taken 4 of the 7 states counted at horizon 4
    assert t.counts((), 4) == (4, 7) and abs(t.chance((), 4) - root) < 1e-12
    mine = (4 + 2.0 * root) / (6 + 2.0)
    assert abs(t.chance(("mine",), 4) - mine) < 1e-12
    unread = (2 + 2.0 * mine) / (3 + 2.0)
    assert abs(t.chance(("mine", "unread"), 4) - unread) < 1e-12
    third = (2 + 2.0 * unread) / (3 + 2.0)                                                   # each finer key with the same counts is smoothed once more toward the one above
    assert abs(t.chance(("mine", "unread", "-"), 4) - third) < 1e-12 and abs(t.chance(("mine", "unread", "-", "-"), 4) - (2 + 2.0 * third) / (3 + 2.0)) < 1e-12
    assert t.chance(("mine", "refuted", "1", "own"), 4) == t.chance(("mine",), 4) or abs(t.chance(("mine", "refuted", "1", "own"), 4) - mine) < 1e-12   # never seen: the coarser key's
    assert 0.0 < t.chance(("theirs", "unread", "-", "-"), 4) < root                  # none of three taken: below the base rate, above 0


def test_a_table_is_saved_with_its_header_and_loaded_and_a_node_gets_its_chance(env, tmp_path):
    cells = Q.cells_from_records(RECORDS)
    path = str(tmp_path / "t.json")
    Q.save_chances(path, cells, {"directories": ["x"], "positions": 3})
    blob = json.loads(Path(path).read_text())
    assert blob["format"] == Q.CHANCE_FORMAT and blob["header"]["positions"] == 3 and "mine|unread|-|-" in blob["cells"]
    t = Q.load_chances(path)
    assert t.counts(("mine", "unread", "-", "-"), 4) == (2, 3)
    sheet = Q.run(HANGING, WORTH, env=env, max_steps=12, params=Q.Params(chances=path))
    qn = [n for n in sheet.graph.nodes if n.kind == "question"]
    assert qn and all(n.chance is not None for n in qn)
    q = sheet.questions[0]
    assert abs(sheet.graph.nodes[q.node].chance - t.chance(Q.question_key(q), 4)) < 1e-12
    below = [n for n in sheet.graph.nodes if n.kind not in ("question",)]
    assert all(n.chance is not None for n in below if any(e.dst == n.id for e in sheet.graph.edges))        # a node with a parent takes the parent's
    plain = Q.run(HANGING, WORTH, env=env, max_steps=12)
    assert all(n.chance is None for n in plain.graph.nodes)                                              # no table: as before


def test_the_calibration_bands_compare_predicted_with_observed():
    t = Q.Chances(Q.cells_from_records(RECORDS))
    recs = [{"final": ["mine", "waiting", "2", "clear"], "out": {4: True}}, {"final": ["mine", "waiting", "2", "clear"], "out": {4: False}}, {"final": ["theirs", "unread", "-", "-"], "out": {4: None}}]
    lines = Q.format_calibration(recs, t, 4, "x")
    assert lines[0].startswith("OBS calibration x: 2 questions, horizon 4, taken 1 (50.0%)")
    assert any("1 taken (50.0%)" in ln and "2 questions" in ln for ln in lines[1:])
    assert any("Brier" in ln for ln in lines)


def _game_dir(tmp_path, moves):
    d = tmp_path / "games"
    d.mkdir()
    (d / "games.jsonl").write_text(json.dumps({"variant": "chess", "game": 0, "start": None, "moves": moves}) + "\n")
    return str(d)


def test_the_outcome_follows_the_exact_unit_and_is_none_when_the_game_ended_too_soon(env, tmp_path):
    d = _game_dir(tmp_path, ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3"])
    g = Q.Games([d], env)
    assert g.outcomes(0, 0, 2, "d5") == {4: True, 8: True}              # their pawn on d5 is taken at once
    assert g.outcomes(0, 0, 0, "e2") == {4: True, 8: True}              # my pawn is taken at the fourth ply: followed through its moves
    assert g.outcomes(0, 0, 4, "b1") == {4: None, 8: None}              # not taken, and the game ends before the horizon
    assert g.outcomes(0, 0, 0, "h5") is None                           # nothing stands there
    assert Q.pick_positions([d], 2) == [(0, 0, 0), (0, 0, 2), (0, 0, 4)] and len(Q.pick_positions([d], 1, 2)) == 2


def test_the_fit_records_the_keys_a_question_passed_through_and_never_reads_the_judge_files(env, tmp_path):
    d = _game_dir(tmp_path, ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "f1b5", "c7c6", "b5c6", "b8c6"])
    (tmp_path / "games" / "surprises.jsonl").write_text("not json at all")             # the judge's file: opening it would fail
    cells, header = Q.fit_chances([d], env, WORTH, Q.Params(), stride=3, say=None)
    assert header["positions"] == 4 and header["records"] > 0 and header["horizons"] == [4, 8] and "no teacher number" in header["source"]
    assert any(k[1] == "unread" for k in cells) and sum(c[4][1] for c in cells.values()) > 0
    again, _ = Q.fit_chances([d], env, WORTH, Q.Params(), stride=3, shard=(0, 2), say=None)
    other, _ = Q.fit_chances([d], env, WORTH, Q.Params(), stride=3, shard=(1, 2), say=None)
    assert Q.merge_cells([again, other]) == cells


# ---- the clock

def _table(tmp_path, cells):
    path = str(tmp_path / "c.json")
    Q.save_chances(path, cells, {})
    return Q.load_chances(path)


def test_the_chance_clock_puts_a_likely_small_stake_before_an_unlikely_large_one(tmp_path):
    t = _table(tmp_path, {("theirs", "unread", "-", "-"): {4: [30, 40], 8: [30, 40]}, ("mine", "unread", "-", "-"): {4: [1, 40], 8: [1, 40]}})
    p = Q.Params(clock="chance")
    costs = Q._Costs(p)
    likely_small, unlikely_large = _fake_question(Q.OTHER, stake=4.0), _fake_question(Q.SELF, stake=14.0)
    assert Q.chance_priority(likely_small, costs, p, t, None) > Q.chance_priority(unlikely_large, costs, Q.Params(clock="chance"), t, None)
    flat = Q.Params(clock="present")
    assert Q.priority(unlikely_large, costs) > Q.priority(likely_small, costs)               # the present order goes by stake alone here


def test_the_risk_dial_moves_the_stake_of_my_own_unit_between_the_worst_case_and_the_midpoint(tmp_path):
    mine, theirs = _fake_question(Q.SELF, stake=12.0), _fake_question(Q.OTHER, stake=12.0)
    assert Q.stake_at_risk(mine, None, Q.Params(risk=1.0)) == 12.0 and Q.stake_at_risk(mine, None, Q.Params(risk=0.0)) == 6.0 and Q.stake_at_risk(mine, None, Q.Params(risk=0.5)) == 9.0
    assert Q.stake_at_risk(theirs, None, Q.Params(risk=1.0)) == 6.0 == Q.stake_at_risk(theirs, None, Q.Params(risk=0.0))        # theirs: always the expected size
    t = _table(tmp_path, {("theirs", "unread", "-", "-"): {4: [20, 40], 8: [20, 40]}, ("mine", "unread", "-", "-"): {4: [20, 40], 8: [20, 40]}})
    costs = Q._Costs(Q.Params())
    other = _fake_question(Q.OTHER, stake=8.0)                                  # midpoint 4
    worst, expected = Q.Params(clock="chance", risk=1.0), Q.Params(clock="chance", risk=0.0)
    own = _fake_question(Q.SELF, stake=7.0)                                    # worst case 7, expected 3.5
    assert Q.chance_priority(own, costs, worst, t, None) > Q.chance_priority(other, costs, worst, t, None)
    assert Q.chance_priority(own, costs, expected, t, None) < Q.chance_priority(other, costs, expected, t, None)


def test_a_question_that_waits_sinks_and_one_lost_where_it_stands_rises(tmp_path):
    t = _table(tmp_path, {("mine", "waiting", "2", "clear"): {4: [1, 50], 8: [1, 50]}, ("mine", "lost", "0", "clear"): {4: [30, 40], 8: [30, 40]}})
    p, costs = Q.Params(clock="chance"), Q._Costs(Q.Params())
    waiting, lost = _fake_question(Q.SELF, stake=10.0), _fake_question(Q.SELF, stake=10.0)
    way = (2, Q.Unit(Q.OTHER, "rook", ("a", "1")), ("a", "5"), 0)
    for q, kind in ((waiting, Q.WAITING_V), (lost, Q.CONFIRMED_V)):
        _finding(q, 0, Q.IDLE)
        _finding(q, 1, Q.LIVE, {"supply": [way], "routes": {way[:4]: []}})
        _finding(q, 2, Q.IDLE, {"results": [{"way": None, "status": Q.CONFIRMED_V if q is lost else Q.REFUTED_V}, {"way": way, "status": kind}]})
        q.level = 3
    assert Q.question_key(waiting)[1] == "waiting" and Q.question_key(lost)[1] == "lost"
    assert Q.chance_priority(lost, costs, p, t, None) > 10 * Q.chance_priority(waiting, costs, p, t, None)


def test_run_under_the_chance_clock_takes_first_the_question_the_table_favours(env, tmp_path):
    cells = {("theirs", "unread", "-", "-"): {4: [39, 40], 8: [39, 40]}, ("mine", "unread", "-", "-"): {4: [0, 40], 8: [0, 40]}}
    path = str(tmp_path / "c.json")
    Q.save_chances(path, cells, {})
    sheet = Q.run(PAWNS, WORTH, env=env, max_steps=1, params=Q.Params(clock="chance", chances=path))
    assert len(sheet.steps) == 1 and sheet.steps[0].question.owner == Q.OTHER
    flipped = {("theirs", "unread", "-", "-"): {4: [0, 40], 8: [0, 40]}, ("mine", "unread", "-", "-"): {4: [39, 40], 8: [39, 40]}}
    Q.save_chances(path, flipped, {})
    Q._TABLES.pop(path, None)
    sheet = Q.run(PAWNS, WORTH, env=env, max_steps=1, params=Q.Params(clock="chance", chances=path))
    assert sheet.steps[0].question.owner == Q.SELF


def test_the_plain_order_takes_a_question_to_its_full_depth_before_the_next(env):
    sheet = Q.run(PAWNS, WORTH, env=env, max_steps=8, params=Q.Params(clock="plain"))
    firsts = [s.question.rank for s in sheet.steps]
    assert firsts == sorted(firsts) and sheet.steps[0].question.rank == 1 and firsts.count(1) >= 2


def test_the_budget_is_honoured_in_steps_in_summed_step_cost_and_in_time_and_the_order_decides_what_is_read(env, tmp_path):
    assert len(Q.run(PAWNS, WORTH, env=env, max_steps=3).steps) == 3
    one = Q.run(PAWNS, WORTH, env=env, budget_cost=1e-9)
    assert len(one.steps) == 1                                                  # a step is started while the budget is not used up; the one begun is finished
    zero = Q.run(PAWNS, WORTH, env=env, budget_seconds=0.0)
    assert len(zero.steps) == 0
    free = Q.run(PAWNS, WORTH, env=env, max_steps=10 ** 6)
    capped = Q.run(PAWNS, WORTH, env=env, budget_cost=sum(s.cost for s in free.steps[:5]) * 0.999)
    budget = sum(s.cost for s in free.steps[:5]) * 0.999
    assert len(capped.steps) < len(free.steps) and sum(s.cost for s in capped.steps[:-1]) < budget                    # every step but the last began while the budget was not used up
    seen = []
    Q.run(PAWNS, WORTH, env=env, max_steps=4, on_step=lambda sheet, st, elapsed: seen.append((len(sheet.steps), st.question.rank, elapsed)))
    assert [x[0] for x in seen] == [1, 2, 3, 4] and all(x[2] > 0 for x in seen)
    path = str(tmp_path / "c.json")
    Q.save_chances(path, {("theirs", "unread", "-", "-"): {4: [39, 40], 8: [39, 40]}, ("mine", "unread", "-", "-"): {4: [0, 40], 8: [0, 40]}}, {})
    timed = Q.run(PAWNS, WORTH, env=env, budget_seconds=1.0, params=Q.Params(clock="chance", chances=path))
    assert timed.steps and timed.steps[0].question.owner == Q.OTHER             # the first step is the favoured one even when the clock cuts the rest


def test_measure_j_counts_what_a_hand_made_game_holds_and_the_tallies_merge(env, tmp_path):
    d = _game_dir(tmp_path, ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"])
    path = str(tmp_path / "c.json")
    Q.save_chances(path, Q.cells_from_records(RECORDS), {})
    games = Q.Games([d], env)
    acc: dict = {}
    Q.j_position(env, WORTH, Q.Params(), games.position(0, 0, 2), lambda place: games.outcomes(0, 0, 2, place), acc, path)
    assert {c.split("|")[0] for c in acc} == set(Q.J_ORDERS) and all(a["positions"] == 1 for a in acc.values())
    assert f"present|steps|{Q.J_STEPS[0]} steps" in acc and "chance|time|20 ms" in acc and "plain|none|unbudgeted" in acc
    free = acc["plain|none|unbudgeted"]
    assert free["mine_units"] + free["theirs_units"] > 0
    both = Q.merge_counts([acc, acc])
    assert both["plain|none|unbudgeted"]["positions"] == 2 and both["plain|none|unbudgeted"]["mine_units"] == 2 * free["mine_units"]
    assert any(ln.startswith("OBS (J) unbudgeted, order plain") for ln in Q.format_j(acc))


def test_measure_j_can_be_limited_to_one_order_and_skip_the_wall_clock_runs(env, tmp_path):
    d = _game_dir(tmp_path, ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"])
    path = str(tmp_path / "c.json")
    Q.save_chances(path, Q.cells_from_records(RECORDS), {})
    games = Q.Games([d], env)
    acc: dict = {}
    Q.j_position(env, WORTH, Q.Params(risk=0.0), games.position(0, 0, 2), lambda place: games.outcomes(0, 0, 2, place), acc, path, ("chance",), False)
    assert {c.split("|")[0] for c in acc} == {"chance"} and not any("|time|" in c for c in acc)


# ---------------------------------------------------------------------------------------------------------------------------------------------- plan 78.42: the outcome is the loss; the question of the move

def _events(env, start, moves):
    return env.codec.events(start, moves)


def _loss(env, start, moves, unit_place, horizon=4, ply=0):
    """unit_loss of the unit standing on `unit_place` before move number `ply`."""
    by_ply, _ = env.codec.tracks(start, moves)
    uid = by_ply[ply][unit_place][0]
    return Q.unit_loss(_events(env, start, moves), uid, ply, horizon, env.worth_fn(WORTH))


def test_the_loss_of_an_even_exchange_is_even(env):
    res = _loss(env, "4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", ["e4d5", "c6d5"], "e4")           # a pawn takes a pawn and is taken back
    assert res["status"] == Q.EVEN and res["balance"] == 0 and res["run"] == [0, 1] and res["arrival"] == 0


def test_a_unit_that_takes_a_heavier_unit_and_is_retaken_gained(env):
    res = _loss(env, "3qk3/8/8/8/8/8/8/3RK3 w - - 0 1", ["d1d8", "e8d8"], "d1")               # a rook takes a queen, the king takes the rook
    assert res["status"] == Q.GAINED and res["balance"] == 8 - 14


def test_a_plain_loss_is_the_worth_of_the_unit(env):
    res = _loss(env, "4k3/8/8/3p4/4P3/8/8/4K3 b - - 0 1", ["d5e4"], "e4")                      # their pawn takes my pawn and nothing is taken back
    assert res["status"] == Q.LOST and res["balance"] == 2 and res["run"] == [0] and res["arrival"] is None


def test_a_unit_that_moves_and_is_then_taken_for_nothing_is_lost_at_its_worth(env):
    res = _loss(env, "4k3/8/8/4b3/8/8/8/1N2K3 w - - 0 1", ["b1c3", "e5c3"], "b1")
    assert res["status"] == Q.LOST and res["balance"] == 4 and res["place"] == "c3" and res["run"] == [1] and res["arrival"] is None


def test_the_units_own_arriving_capture_counts_even_when_the_taking_back_comes_later(env):
    start = "4r1k1/8/8/4p3/8/3N4/8/4K3 w - - 0 1"
    res = _loss(env, start, ["d3e5", "e8e5"], "d3")                                           # takes a pawn, is taken at once
    assert res["status"] == Q.LOST and res["balance"] == 4 - 2 and res["run"] == [0, 1]
    late = _loss(env, start, ["d3e5", "g8g7", "e1d1", "e8e5"], "d3")                          # the same, the taking back two plies later
    assert late["status"] == Q.LOST and late["balance"] == 2 and late["run"] == [0, 3] and late["arrival"] == 0
    assert _loss(env, start, ["d3e5", "g8g7", "e1d1", "e8e5"], "d3", horizon=3) is None       # not taken inside the horizon


def test_a_run_that_began_before_the_start_ply_is_not_counted(env):
    start = "4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1"
    by_ply, _ = env.codec.tracks(start, ["e4d5", "c6d5"])
    res = Q.unit_loss(_events(env, start, ["e4d5", "c6d5"]), by_ply[1]["d5"][0], 1, 4, env.worth_fn(WORTH))        # the pawn that stands on d5 after the first move
    assert res["status"] == Q.LOST and res["balance"] == 2 and res["run"] == [1]


def test_the_game_outcome_by_loss_is_false_for_an_even_exchange_and_the_fit_can_use_it(env, tmp_path):
    d = _game_dir(tmp_path, ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"])
    g = Q.Games([d], env)
    wf = env.worth_fn(WORTH)
    assert g.outcomes(0, 0, 0, "e2") == {4: True, 8: True}                                      # taken
    assert g.losses(0, 0, 0, "e2", wf) == {4: False, 8: False}                                   # but an even exchange: the pawn took a pawn
    assert g.outcomes(0, 0, 2, "d5") == {4: True, 8: True} and g.losses(0, 0, 2, "d5", wf) == {4: False, 8: False}      # their pawn is taken and the queen takes back: even
    d2 = _game_dir_named(tmp_path, "g2", ["b1c3", "d7d5", "c3d5", "d8d5"])
    g2 = Q.Games([d2], env)
    assert g2.losses(0, 0, 2, "c3", wf) == {4: True, 8: True}                                   # my knight takes a pawn (2) and is taken by the queen for nothing: 4 - 2 is lost
    assert g.losses(0, 0, 0, "h5", wf) is None
    cells, header = Q.fit_chances([d], env, WORTH, Q.Params(), stride=3, say=None, outcome="loss")
    assert header["outcome"] == "loss" and "lost" in header["source"] and header["records"] > 0
    with pytest.raises(ValueError):
        Q.fit_chances([d], env, WORTH, Q.Params(), say=None, outcome="nonsense")


def test_measure_j_has_the_loss_columns_and_old_tallies_still_merge(env, tmp_path):
    d = _game_dir(tmp_path, ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"])
    path = str(tmp_path / "c.json")
    Q.save_chances(path, Q.cells_from_records(RECORDS), {})
    games = Q.Games([d], env)
    wf = env.worth_fn(WORTH)
    acc: dict = {}
    Q.j_position(env, WORTH, Q.Params(), games.position(0, 0, 2), lambda place: games.outcomes(0, 0, 2, place), acc, path, ("plain",), False, loss_of=lambda place: games.losses(0, 0, 2, place, wf))
    free = acc["plain|none|unbudgeted"]
    assert free["theirs_lost"] == 0 and free["mine_lost"] == 0                                 # the pawn taken and taken back is no loss (and a pawn is no unit above the least worth)
    d2 = _game_dir_named(tmp_path, "g2", ["b1c3", "d7d5", "c3d5", "d8d5"])
    g2 = Q.Games([d2], env)
    acc2: dict = {}
    Q.j_position(env, WORTH, Q.Params(), g2.position(0, 0, 2), lambda place: g2.outcomes(0, 0, 2, place), acc2, path, ("plain",), False, loss_of=lambda place: g2.losses(0, 0, 2, place, wf))
    free2 = acc2["plain|none|unbudgeted"]
    assert free2["mine_taken"] >= 1 and 1 <= free2["mine_lost"] <= free2["mine_taken"]
    free = {k: free[k] + free2[k] for k in free}
    acc = {"plain|none|unbudgeted": free}
    old = {k: {f: v for f, v in a.items() if "lost" not in f} for k, a in acc.items()}
    assert Q.merge_counts([old, acc])["plain|none|unbudgeted"]["theirs_lost"] == free["theirs_lost"]
    assert any(ln.startswith("OBS (J, by LOSS) unbudgeted, order plain") for ln in Q.format_j(acc))


# ---- the question of the move

HUNG = "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"                 # their queen strikes the d-file; my rook on d2 strikes it back
GUARDED_QUEEN = "4k3/8/4p3/3q4/8/8/3R4/4K3 w - - 0 1"     # their queen on d5 is guarded by a pawn
GUARDED_PAWN = "4k3/8/4p3/3p4/8/8/3R4/4K3 w - - 0 1"      # their pawn on d5 is guarded by a pawn, and worth less than my rook


def _game_dir_named(tmp_path, name, moves):
    d = tmp_path / name
    d.mkdir()
    (d / "games.jsonl").write_text(json.dumps({"variant": "chess", "game": 0, "start": None, "moves": moves}) + "\n")
    return str(d)


def _move(env, fen, mv, params=None):
    r = Q.Reading(env, fen, WORTH)
    return Q.move_question(env, r, mv, params or Q.Params())


def test_a_unit_that_moves_onto_a_struck_unguarded_place_loses_it_and_the_line_is_named(env):
    res = _move(env, HUNG, "d2d3")
    assert res["verdict"] == Q.LOSES_V and res["named"]["part"] == "b" and res["named"]["unit"].kind == "rook"
    assert (res["low"], res["high"]) == (8.0, 8.0) and res["line"] == ("d5d3",)
    assert res["took"] == 0.0 and res["threats"][0]["kind"] == "lands"


def test_a_taking_that_wins_more_than_it_loses_is_safe_and_one_that_loses_more_is_not(env):
    win = _move(env, GUARDED_QUEEN, "d2d5")                    # the rook (8) takes the queen (14) and is taken by the pawn
    assert win["took"] == 14.0 and win["verdict"] == Q.SAFE_V and win["threats"][0]["value"] == 8.0 and "the balance is +6" in win["answers"]["b"]
    lose = _move(env, GUARDED_PAWN, "d2d5")                    # the rook (8) takes a pawn (2) and is taken by the pawn
    assert lose["took"] == 2.0 and lose["verdict"] == Q.LOSES_V and (lose["low"], lose["high"]) == (6.0, 6.0) and "the balance is -6" in lose["answers"]["b"]


def test_a_move_that_uncovers_another_unit_loses_it_and_names_that_unit(env):
    res = _move(env, DUTY, "b3a5")                              # the knight guarded the rook that their rook strikes
    assert res["verdict"] == Q.LOSES_V and res["named"]["part"] == "a" and res["named"]["unit"].kind == "rook" and res["named"]["place"] == "d4"
    assert res["threats"][0]["kind"] == "left unguarded" and res["answers"]["b"].endswith("is not lost where it lands (real moves)")


def test_a_move_that_leaves_everything_guarded_is_safe_with_every_answer_given(env):
    res = _move(env, DUTY_GUARDED, "b3a5")                      # a second guard stays
    assert res["verdict"] == Q.SAFE_V and res["threats"] == [] and res["named"] is None and res["low"] == res["high"] == 0.0
    assert all(res["answers"][k] for k in "abc") and set(res["ms"]) == {"a", "read", "b", "c", "total"} and res["ms"]["total"] > 0
    ok = _move(env, DUTY, "d4c4")                              # the rook steps aside: nothing is lost
    assert ok["verdict"] == Q.SAFE_V


def test_c_reads_the_units_on_changed_places_that_a_did_not(env, monkeypatch):
    blind = {"read": True, "mark": "", "duties": []}
    monkeypatch.setattr(Q, "cost_of_leaving", lambda *a, **k: blind)
    monkeypatch.setattr(Q, "_leaving_candidates", lambda *a, **k: (set(), False))
    res = _move(env, DUTY, "b3a5")
    assert res["verdict"] == Q.LOSES_V and res["named"]["part"] == "c" and res["named"]["unit"].kind == "rook" and res["threats"][0]["kind"] == "newly struck"
    assert "is lost where it stands after the move" in res["answers"]["c"]


def test_a_unit_lost_before_the_move_is_not_newly_lost(env):
    res = _move(env, "7k/3r4/8/6p1/3R4/1N6/8/6K1 b - - 0 1".replace(" b ", " w "), "g1f2")     # a king step: the rook stays guarded as before
    assert res["verdict"] == Q.SAFE_V


def test_the_move_graph_has_a_node_per_answer_with_bounds(env):
    r = Q.Reading(env, DUTY, WORTH)
    res = Q.move_question(env, r, "b3a5", Q.Params())
    nodes = r.graph.nodes
    root, a, b, c = (nodes[res["nodes"][k]] for k in ("move", "a", "b", "c"))
    assert root.verdict == Q.LOSES_V and (root.stake_low, root.stake_high) == (8.0, 8.0)
    assert (a.stake_low, a.stake_high) == (8.0, 8.0) and (b.stake_low, b.stake_high) == (0.0, 0.0) and c.stake_high == 0.0
    assert [n.id for _e, n in r.graph.out(root)] == [a.id, b.id, c.id]


def test_explain_move_says_because_and_names_the_unit_the_bounds_and_the_line(env):
    text = "\n".join(Q.explain_move(_move(env, HUNG, "d2d3")))
    assert text.startswith("the move d2d3: my rook on d2 goes to d3: LOSES")
    assert text.count("because") >= 4 and "it LOSES my rook on d2" in text and "between 8 and 8" in text and "the line: d5d3" in text
    safe = "\n".join(Q.explain_move(_move(env, DUTY, "d4c4")))
    assert "SAFE -- because nothing of its own is lost" in safe
    take = "\n".join(Q.explain_move(_move(env, GUARDED_QUEEN, "d2d5")))
    assert "taking their queen on d5 (worth 14)" in take


def test_the_move_sheet_asks_struck_landing_places_first_and_the_heavier_unit_first(env):
    r = Q.Reading(env, HUNG, WORTH)
    order = Q.move_order(r, r.legal)
    struck = {Q.place_text(x) for x in r.struck_by("them")}
    flags = [m[2:4] in struck for m in order]
    assert any(flags) and not all(flags) and flags == sorted(flags, reverse=True)           # every move that lands on a struck place comes before every one that does not
    assert sorted(order) == sorted(r.legal)
    first_struck = [m for m in order if m[2:4] in struck]
    stakes = [r.weight(next(u for u in r.units if u.side == "us" and Q.place_text(u.at) == m[:2])) for m in first_struck]
    assert stakes == sorted(stakes, reverse=True)


def test_the_move_sheet_gives_a_verdict_to_every_move_and_a_budget_cuts_it(env):
    r = Q.Reading(env, HUNG, WORTH)
    full = Q.move_sheet(env, r, Q.Params())
    assert len(full.results) == len(r.legal) and full.unread == [] and set(full.verdicts()) == set(r.legal)
    assert "d2d3" in {x["move"] for x in full.loses()} and full.verdicts()["d2d5"] == Q.SAFE_V
    cut = Q.move_sheet(env, r, Q.Params(), max_moves=3)
    assert [x["move"] for x in cut.results] == full.order[:3] and cut.unread == full.order[3:]
    none = Q.move_sheet(env, r, Q.Params(), budget_seconds=0.0)
    assert none.results == [] and none.unread == full.order
    one = Q.move_sheet(env, r, Q.Params(), budget_cost=1e-9)                                  # the cost used up after the first move
    assert len(one.results) == 1 and one.results[0]["move"] == full.order[0]
    given = Q.move_sheet(env, r, Q.Params(), moves=["d2d3", "e1e2"])
    assert set(given.verdicts()) == {"d2d3", "e1e2"}


def test_the_move_question_is_off_by_default_and_on_it_is_kept_on_the_sheet_and_explained(env):
    plain = Q.run(HUNG, WORTH, env=env, max_steps=3)
    assert plain.moves is None and not any("question of the move" in ln for ln in Q.explain(plain))
    sheet = Q.run(HUNG, WORTH, env=env, max_steps=3, params=Q.Params(move_question=True))
    assert sheet.moves is not None and len(sheet.moves.results) == len(sheet.reading.legal)
    text = "\n".join(Q.explain(sheet))
    assert "the question of the move:" in text and "the move d2d3" in text and "LOSES" in text
    gone = Q.run(HUNG, WORTH, env=env, max_steps=3, budget_seconds=0.0, params=Q.Params(move_question=True))
    assert gone.moves.results == []


# ---------------------------------------------------------------------------------------------------------------------------------------------- plan 78.42 (next): a loss is charged to the last own move

def _charged(env, start, moves, unit_place, ply=0, horizon=6):
    by_ply, _ = env.codec.tracks(start, moves)
    ev = _events(env, start, moves)
    res = Q.unit_loss(ev, by_ply[ply][unit_place][0], ply, horizon, env.worth_fn(WORTH))
    return Q.charged_ply(ev, res), res


def test_a_loss_is_charged_to_the_last_own_move_before_the_first_taking(env):
    start = "4r1k1/8/8/4p3/8/3N4/8/4K3 w - - 0 1"
    assert _charged(env, start, ["d3e5", "e8e5"], "d3")[0] == 0                                  # taken at once: charged to the move that arrived
    got, res = _charged(env, start, ["d3e5", "g8g7", "e1d1", "e8e5"], "d3")                      # a second own move lies between: it is the one charged
    assert res["status"] == Q.LOST and got == 2
    got, res = _charged(env, "4k3/8/8/4b3/8/2N5/8/4K3 w - - 0 1", ["e1d1", "e8d8", "d1c1", "e5c3"], "c3")        # the unit never moved
    assert res["status"] == Q.LOST and got == 2
    got, res = _charged(env, "4k3/8/8/4b3/8/2N5/8/4K3 b - - 0 1", ["e5c3"], "c3")             # no own move of the unit's side before the taking
    assert res["status"] == Q.LOST and got is None


def test_measure_k_prints_the_charged_table(env, tmp_path):
    d = _game_dir(tmp_path, ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"])
    (Path(d) / "surprises.jsonl").write_text("")
    recs = Q.measure_k([d], env, WORTH, Q.Params(), stride=3, say=None)
    assert recs and all("lost_charged" in x for x in recs)
    assert any(ln.startswith("OBS (K, charged) table") for ln in Q.format_k(recs))


# ---------------------------------------------------------------------------------------------------------------------------------- 78.42 cheap: the light reading, a unit that changes kind

LIGHT_POSITIONS = [HANGING, PAWNS, SPARSE, "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "r3k2r/ppp2ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPP2PPP/R3K2R w KQkq - 0 1",
                   "4k3/8/8/3q4/8/8/3R4/3rK3 w - - 0 1"]


@pytest.mark.parametrize("fen", LIGHT_POSITIONS)
def test_the_light_reading_holds_what_the_full_reading_holds_for_the_units_that_stand(env, fen):
    full, light = Q.Reading(env, fen, WORTH), Q.Reading(env, fen, WORTH, light=True)
    assert light.units == full.units and light.mover == full.mover and light.legal == full.legal
    for u in full.units:
        assert light.strikes_of(u) == full.strikes_of(u)
    assert light.kind_key == full.kind_key
    assert [light.worth_of(u.kind) for u in light.units] == [full.worth_of(u.kind) for u in full.units]


def test_the_light_reading_reads_the_move_tables_only_when_asked(env):
    light = Q.Reading(env, HANGING, WORTH, light=True)
    assert light._go is None
    assert light.go[0] == Q.Reading(env, HANGING, WORTH).go[0] and light._go is not None


@pytest.mark.parametrize("fen", LIGHT_POSITIONS)
def test_the_light_question_of_the_move_says_what_the_full_one_says(env, fen):
    full, light = Q.Reading(env, fen, WORTH), Q.Reading(env, fen, WORTH, light=True)
    for mv in full.legal:
        a, b = Q.move_question(env, full, mv), Q.move_question(env, light, mv)
        assert (a["verdict"], a["low"], a["took"]) == (b["verdict"], b["low"], b["took"]), (fen, mv)
        assert (a["named"] or {}).get("place") == (b["named"] or {}).get("place")


def test_the_light_reading_says_what_it_leaves_unread(env):
    # the goal unit's sign is not asked by the light reading (it costs a whole scenario read): no part c answer about it, and the verdict never rests on it
    light = Q.Reading(env, "4k3/8/8/3q4/8/8/3R4/3rK3 w - - 0 1", WORTH, light=True)
    assert Q._goal_newly_under_sign(light, Q.Question(light.units[0], Q.SELF, 1.0, 0), light.position, Q.Params()) is False


PROMOTES_TAKING = "1rk5/P7/8/8/8/8/8/4K3 w - - 0 1"              # the pawn takes the rook and becomes a queen the king may take
PROMOTES_AND_LOST = "2k5/P7/8/8/8/8/r7/4K3 w - - 0 1"             # the pawn becomes a queen on a place the rook bears on


@pytest.mark.parametrize("light", [False, True])
def test_a_unit_that_arrives_as_another_kind_is_the_same_unit_with_its_new_worth(env, light):
    r = Q.Reading(env, PROMOTES_TAKING, WORTH, light=light)
    res = Q.move_question(env, r, "a7b8q")
    assert res["took"] == 8.0 and res["credit"] == 12.0                                  # the queen is worth 12 more than the pawn was
    assert res["threats"] and res["threats"][0]["value"] == 14.0
    assert res["verdict"] == Q.SAFE_V and "arrives as a queen" in res["answers"]["b"] and "the balance is +6" in res["answers"]["b"]
    r2 = Q.Reading(env, PROMOTES_AND_LOST, WORTH, light=light)
    res2 = Q.move_question(env, r2, "a7a8q")
    assert res2["verdict"] == Q.LOSES_V and res2["low"] == 2.0 and res2["credit"] == 12.0      # what it loses is the pawn it was: 14 less the 12 it had gained


def _events_with(env, start, moves):
    return env.codec.events(start, moves)


def test_the_events_say_what_kind_a_unit_has_after_its_move(env):
    ev = _events_with(env, PROMOTES_TAKING, ["a7b8q", "c8b8"])
    assert ev[0]["mover"][2] == "pawn" and ev[0]["arrives"] == "queen" and ev[1]["arrives"] == "king"
    assert ev[0]["mover"][0] == ev[1]["taken"][0]                                           # the queen taken is the pawn that moved


def test_a_unit_that_changes_kind_on_taking_and_is_retaken_is_not_lost_by_the_change(env):
    ev = _events_with(env, PROMOTES_TAKING, ["a7b8q", "c8b8"])
    res = Q.unit_loss(ev, ev[0]["mover"][0], 0, 4, env.worth_fn(WORTH))
    # a queen (14) is taken, the rook (8) was taken by it, and the change itself gained 12: the side is ahead by 6
    assert res["status"] == Q.GAINED and res["balance"] == 14 - 8 - 12 and res["arrival"] == 0 and res["run"] == [0, 1]


def test_a_unit_that_changes_kind_with_no_taking_and_is_taken_on_arrival_is_lost_at_what_it_was(env):
    ev = _events_with(env, PROMOTES_AND_LOST, ["a7a8q", "a2a8"])
    res = Q.unit_loss(ev, ev[0]["mover"][0], 0, 4, env.worth_fn(WORTH))
    assert res["status"] == Q.LOST and res["balance"] == 14 - 12 and res["arrival"] == 0


def test_events_without_the_arrival_kind_read_as_before(env):
    ev = [{"mover": ("u", "w", "pawn"), "origin": "a7", "dest": "a8", "taken": None, "taken_at": None}, {"mover": ("v", "b", "rook"), "origin": "a1", "dest": "a8", "taken": ("u", "w", "queen"), "taken_at": "a8"}]
    res = Q.unit_loss(ev, "u", 0, 4, env.worth_fn(WORTH))
    assert res["balance"] == 14 and res["arrival"] is None
