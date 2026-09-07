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
    # AsXfL6OX before 18, White: cxb6 takes the b6 pawn, but Black's doomed g3 pawn cashes h2 (gxh2+), so the
    # count never settles a pawn up within the horizon; the greedy recapture Nxh2 is refuted by the judge (Qh7)
    b = chess.Board("2kr1bnr/p1pp4/1p3p2/2P2qp1/8/4BNp1/PP2PP1P/2RQ1RK1 w - - 0 18")
    gs = GoalSearch(b, chess.WHITE, chess.B6, memo=Memo(), engine=eng, horizon=6, budget=300, engine_depth=8)
    gs.verbose = True
    r = gs.run()
    assert not r.proven and r.tried[0] == "cxb6"
    assert any("AND cxb6 gxh2+ Nxh2:" in l and "refuted by" in l for l in gs.trace)
    # the f3 knight cannot be won: g4 is tried first (the planner's prior), the knight steps away, the search gives up
    r2 = GoalSearch(b, chess.BLACK, chess.F3, memo=Memo(), engine=eng, horizon=6, budget=200, engine_depth=8).run()
    assert not r2.proven and r2.tried[0] == "g4"
    assert "given up" in result_text(r2) or "refuted" in result_text(r2)


def test_latest_game_goals(eng):
    # tPVi7fZF (Kaveh as Black), three goals from the game: the piece the game won, the pawn it won, the knight it missed
    # after 12.a3: the b4 bishop; axb4 is answered by Rxa1 (the a-file opened), cxb6 by Qxb6 first
    b = chess.Board("r2q1rk1/4bppp/npp1pn2/2Pp4/1B1P4/P3PN2/5PPP/RN1QK2R b KQ - 0 12")
    r = GoalSearch(b, chess.BLACK, chess.B4, memo=Memo(), engine=eng, horizon=6, budget=300, engine_depth=8).run()
    assert r.proven and r.line[0] == "Nxb4" and r.work[:5] == ["Nxb4", "cxb6", "Qxb6", "axb4", "Rxa1"]
    # after 15.O-O: the c5 pawn; the deepest try is the desperado dxc5 met by Qxe5 (the knight is undefended once d4 left)
    b = chess.Board("r4rk1/2q1bppp/npp1pn2/2PpN3/3P4/P1N1P3/5PPP/R2Q1RK1 b - - 5 15")
    r = GoalSearch(b, chess.BLACK, chess.C5, memo=Memo(), engine=eng, horizon=6, budget=600, engine_depth=8).run()
    assert r.proven and r.line[0] in ("bxc5", "Bxc5", "Nxc5")
    # after 16.dxc5: the e5 knight hangs (the game played Nxc5); the goal loop finds it first
    b = chess.Board("r4rk1/2q1bppp/n1p1pn2/2PpN3/8/P1N1P3/5PPP/R2Q1RK1 b - - 0 16")
    rs = pursue(b, chess.BLACK, memo=Memo(), engine=eng, budget=300, horizon=6)
    assert rs[0].proven and rs[0].goal.endswith("e5") and rs[0].line[0] == "Qxe5"
