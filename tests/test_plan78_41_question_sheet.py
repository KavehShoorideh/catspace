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
    assert f.level == 2 and f.status == Q.LIVE and f.value == 14 and f.line[0] == "d2d5"


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

def test_load_positions_reads_the_units_lost_and_the_unit_moved(tmp_path):
    moves = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"]
    ours = {str(p): {"balance": 0} for p in range(6)}
    (tmp_path / "games.jsonl").write_text(json.dumps({"variant": "chess", "game": 0, "start": None, "moves": moves, "ours": ours}) + "\n")
    (tmp_path / "surprises.jsonl").write_text("")
    items = {i[0][1]: i for i in Q.load_positions(str(tmp_path), "chess", game="chess")}
    assert items[1][4] == {"pawn"}                      # black had 8 pawns at ply 1 and 7 at ply 5
    assert items[0][4] == {"pawn"}                      # white's pawn is taken at ply 3
    assert items[0][5] == "e2" and items[1][5] == "d7" and items[2][5] == "e4"
    assert items[3][4] is None                          # the game ends before ply 7


def _rec(bad, rows, lost=None, quiet_at=None, moved=None):
    return Q.Record((0, 0), bad, rows, None, [(0, 0.001)], lost=lost, quiet_at=quiet_at, moved=moved)


def test_measure_b_by_unit_and_measure_d_on_hand_made_records():
    row = lambda kind, closed, live=False, owner="us": Q.Row(owner, live, closed, kind=kind)
    records = [
        _rec(False, [row("pawn", True), row("rook", False)], lost={"pawn"}),
        _rec(False, [row("pawn", True), row("rook", False)], lost=set()),
        _rec(False, [row("pawn", True), row("rook", False, owner="them")], lost={"rook"}),
        _rec(True, [row("pawn", True)], lost=set(), quiet_at="a2", moved="a2"),
        _rec(False, [row("pawn", True)], lost=set(), quiet_at="a2", moved="b1"),
        _rec(True, [row("pawn", True)], lost=set(), quiet_at="a2", moved="b1"),
        _rec(False, [row("pawn", False, live=True)], lost=set(), quiet_at="a2", moved="a2"),    # a question is live: not a quiet position
    ]
    lines = Q.format_obs(records)
    closed = next(ln for ln in lines if "my units that closed at level 1" in ln)
    stayed = next(ln for ln in lines if "my units that stayed open" in ln)
    assert "closed at level 1: 6; " in closed and "in 1 (16.7%)" in closed                    # pawn closed in 6 rows, lost in the first only
    assert "stayed open: 3; " in stayed and "in 0 (0.0%)" in stayed                              # the rook's two rows of mine and the live pawn; none of those kinds lost
    d = next(ln for ln in lines if ln.startswith("OBS (d)"))
    assert "nothing live at level 0 or 1: 6 " in d and "the quiet question named and the move read: 3" in d
    assert "a move of the unit it names in 1 (33.3%)" in d
    assert "bad move follows 100.0% when it was (1), 50.0% when it was not (2)" in d
