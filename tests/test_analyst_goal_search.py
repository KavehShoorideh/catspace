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
    assert r.proven and r.line[0] == "Nxb4" and r.work[0] == "Nxb4"      # (the longest try used to be cxb6 Qxb6 axb4 Rxa1; the judge now settles it earlier)
    # after 15.O-O: the c5 pawn; the deepest try is the desperado dxc5 met by Qxe5 (the knight is undefended once d4 left)
    b = chess.Board("r4rk1/2q1bppp/npp1pn2/2PpN3/3P4/P1N1P3/5PPP/R2Q1RK1 b - - 5 15")
    r = GoalSearch(b, chess.BLACK, chess.C5, memo=Memo(), engine=eng, horizon=6, budget=600, engine_depth=8).run()
    assert r.proven and r.line[0] in ("bxc5", "Bxc5", "Nxc5")
    # after 16.dxc5: the e5 knight hangs (the game played Nxc5); the goal loop finds it first
    b = chess.Board("r4rk1/2q1bppp/n1p1pn2/2PpN3/8/P1N1P3/5PPP/R2Q1RK1 b - - 0 16")
    rs = pursue(b, chess.BLACK, memo=Memo(), engine=eng, budget=300, horizon=6)
    assert rs[0].proven and rs[0].goal.endswith("e5") and rs[0].line[0] == "Qxe5"


def test_refutation_revises_the_sketch(eng):
    """The Lean-error-to-next-sketch loop (Kaveh 2026-09-07): a refuted plan move is diagnosed, the counter-goal
    becomes the root prior, the horizon grows by the preparation's two plies, and the search runs again."""
    from catspace.incubator.analyst.goal_search import search_goal
    # Nxe6?? Rxd1#: the knight shields the queen from the d8 rook. With horizon 2 the plain search is refuted; the
    # refinement says 'first secure the queen on d1', tries the queen moves first, and proves the pawn win
    b = chess.Board("3r2k1/6pp/4p3/8/3N4/4P3/5PPP/3Q2K1 w - - 0 1")
    r = search_goal(b, chess.WHITE, chess.E6, memo=Memo(), engine=eng, budget=300, horizon=2, engine_depth=8, refinements=1)
    assert r.proven and r.refinements and r.refinements[0].reply == "Nxe6 Rxd1#"
    assert r.refinements[0].diagnosis == "takes my queen on d1" and r.refinements[0].counter == "first secure the queen on d1"
    assert r.line[0].startswith("Q") and ("Nxe6" in r.line or r.line[-1].endswith("#"))   # the pawn, or a mate found on the way (a win meets every goal)
    # the rook ending after ...Kxh5: bxa5 Rxa4 is diagnosed, the only securing move (Rxa5+) is tried, the goal stays given up
    b = chess.Board("8/8/1p3p2/p6k/RPr5/P5K1/8/8 w - - 0 2")
    r = search_goal(b, chess.WHITE, chess.A5, memo=Memo(), engine=eng, budget=300, horizon=6, engine_depth=8, refinements=1)
    assert not r.proven and r.refinements[0].reply == "bxa5 Rxa4" and r.refinements[0].diagnosis == "takes my rook on a4"
    assert r.refinements[0].prior == ["Rxa5+"] and r.refinements[0].result is not None and not r.refinements[0].result.proven
    assert "revised: first secure the rook on a4 (Rxa5+): given up again" in result_text(r)


def test_proofs_are_stored_and_reused(eng):
    """The concept store keeps every proof twice: exactly (a transposition is served as it stands) and as a sketch on
    the proof's footprint (served as the prior and re-verified when the same pieces stand on it)."""
    from catspace.incubator.analyst.goal_search import search_goal
    memo = Memo()
    b = chess.Board("r2q1rk1/4bppp/npp1pn2/2Pp4/1B1P4/P3PN2/5PPP/RN1QK2R b KQ - 0 12")
    r0 = search_goal(b, chess.BLACK, chess.B4, memo=memo, engine=eng, budget=300)
    assert r0.proven and r0.served == ""
    r1 = search_goal(b, chess.BLACK, chess.B4, memo=memo, engine=eng, budget=300)
    assert r1.proven and r1.served == "transposition" and r1.line == r0.line
    c = b.copy(); c.push_san("h6"); c.push_san("h3")                       # quiet moves off the footprint
    r2 = search_goal(c, chess.BLACK, chess.B4, memo=memo, engine=eng, budget=300)
    assert r2.proven and r2.served == "sketch" and r2.line[0] == "Nxb4"
    assert memo.stats["kinds"]["sketch"]["hit"] == 1 and memo.stats["kinds"]["proof"]["hit"] == 1
    d = b.copy(); d.push_san("h6"); d.push_san("Nc3")                      # the b1 knight leaves: the a1 rook is defended, the sketch no longer applies
    r3 = search_goal(d, chess.BLACK, chess.B4, memo=memo, engine=eng, budget=60, refinements=0)
    assert r3.served == "" and not r3.proven


def test_the_terminal_goal_is_the_rules_outcome():
    """target=None: win the game by the rules' terminal outcome, proven without an engine; the proof-number pricing
    (few replies = cheap) finds mates with no 'checks first' rule."""
    from catspace.incubator.analyst.goal_search import GoalSearch
    b = chess.Board("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 10")     # Legal's mate pattern
    r = GoalSearch(b, chess.WHITE, None, memo=Memo(), horizon=6, budget=400).run()
    assert r.proven and r.line == ["Nf6+", "gxf6", "Bxf7#"] and r.engine_calls == 0 and r.goal == "win the game"
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    assert GoalSearch(b, chess.WHITE, None, memo=Memo(), horizon=2, budget=50).run().line == ["Ra8#"]
    b = chess.Board("r1bqkbnr/ppp2ppp/2np4/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4")          # no mate: given up, honestly
    assert not GoalSearch(b, chess.WHITE, None, memo=Memo(), horizon=4, budget=60).run().proven
