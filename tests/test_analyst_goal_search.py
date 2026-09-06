"""goal_search.py: proof-number search toward a material goal, the planner as prior, the engine as the judge at
the leaves (the no-engine material rule exists but is slow and strict; the engine is the design)."""
import chess
import pytest

from catspace.incubator.analyst.goal_search import GoalSearch, pursue, result_text
from catspace.incubator.analyst.memo import Memo
from catspace.incubator.analyst.engine import Engine


@pytest.fixture(scope="module")
def eng():
    with Engine(depth=12) as e:
        yield e


def test_direct_win_is_proven(eng):
    # AUjG3sl4 after 16.c4, Black: the c4 pawn falls; the proof line is Bxc4 with the recapture answered
    b = chess.Board("1r1r2k1/p1p2ppp/b3pb2/2Pq4/2PP4/8/P2NNPPP/R2Q1RK1 b - - 0 16")
    r = GoalSearch(b, chess.BLACK, chess.C4, memo=Memo(), engine=eng, horizon=6, budget=400, engine_depth=8).run()
    assert r.proven and r.line[0] == "Bxc4" and r.tried[0] == "Bxc4" and r.engine_calls > 0


def test_the_goal_loop_gives_up_and_moves_on(eng):
    # the d4 pawn cannot be won while the queen hangs on d5; the loop gives d4 up and proves c4
    b = chess.Board("1r1r2k1/p1p2ppp/b3pb2/2Pq4/2PP4/8/P2NNPPP/R2Q1RK1 b - - 0 16")
    rs = pursue(b, chess.BLACK, memo=Memo(), engine=eng, budget=300, horizon=6, max_goals=3)
    assert rs[-1].proven and rs[-1].line[0] == "Bxc4"
    assert not rs[0].proven and rs[0].goal.endswith("d4")


def test_counterplay_is_weighed_and_an_escaping_target_is_given_up(eng):
    # AsXfL6OX before 18, White: cxb6 wins the b6 pawn; Black's desperado Qxf3 is judged by the engine
    b = chess.Board("2kr1bnr/p1pp4/1p3p2/2P2qp1/8/4BNp1/PP2PP1P/2RQ1RK1 w - - 0 18")
    r = GoalSearch(b, chess.WHITE, chess.B6, memo=Memo(), engine=eng, horizon=6, budget=300, engine_depth=8).run()
    assert r.proven and r.line[0] == "cxb6"
    # the f3 knight cannot be won: g4 is tried first (the planner's prior), the knight steps away, the search gives up
    r2 = GoalSearch(b, chess.BLACK, chess.F3, memo=Memo(), engine=eng, horizon=6, budget=200, engine_depth=8).run()
    assert not r2.proven and r2.tried[0] == "g4"
    assert "given up" in result_text(r2) or "refuted" in result_text(r2)
