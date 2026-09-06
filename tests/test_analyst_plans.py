"""plans.py: suggestions by a chain of reasoning, with the opponent's counter-goals."""
import chess

from catspace.incubator.analyst.plans import derive, plans_for, answer, Goal, plan_text


def test_defend_first_finds_the_game_move():
    # AUjG3sl4 after 16.c4: the c4 pawn attacks the queen; the planner's first plan is to hold the queen by
    # removing the attacker -- 16...Bxc4, the move played
    b = chess.Board("1r1r2k1/p1p2ppp/b3pb2/2Pq4/2PP4/8/P2NNPPP/R2Q1RK1 b - - 0 16")
    pls = plans_for(b, chess.BLACK, top=5)
    assert pls[0].goal.kind == "defend" and pls[0].steps[0].san == "Bxc4" and pls[0].race == "urgent"
    txt = plan_text(b, pls[0])
    assert txt.startswith("First, hold the queen on d5") and "attacker on c4 is removed" in txt


def test_reinforcement_plan_and_the_race():
    # AsXfL6OX before 18.fxg3, Black: the f3 knight is held; g4 brings the pawn; White can step the knight away in one
    b = chess.Board("2kr1bnr/p1pp4/1p3p2/2P2qp1/8/4BNp1/PP2PP1P/2RQ1RK1 w - - 0 18")
    pls = derive(b, chess.BLACK, Goal("win", chess.F3))
    assert pls and pls[0].steps[0].san == "g4" and pls[0].hops == 1
    answer(b, pls[0])
    assert pls[0].race == "met" and pls[0].answers and pls[0].answers[0].goal.kind == "defend"
    assert any("reaches g4 in 1" in c for c in pls[0].chain)


def test_undermining_chain():
    # c5 is held by d4; d4 falls; the chain says the defender falls first, and the counter-goal finds the queen capture
    b = chess.Board("1r1r2k1/p1p2ppp/b3pb2/2Pq4/2PP4/8/P2NNPPP/R2Q1RK1 b - - 0 16")
    pls = [p for p in derive(b, chess.BLACK, Goal("win", chess.C5)) if p.hops == 2]
    assert pls and "falls first" in " ".join(pls[0].chain) and pls[0].steps[0].san == "Bxd4"
    answer(b, pls[0])
    assert pls[0].race == "met" and pls[0].answers[0].steps[0].san == "cxd5"


def test_nothing_to_plan_in_a_quiet_opening():
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
    assert plans_for(b, chess.WHITE) == []
