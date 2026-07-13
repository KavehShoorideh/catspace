"""
Real-board layer tests: play_board_game legality/determinism/PGN round-trip
(no torch needed), FBBoardPolicy legality + mate-taking (torch), and a live
UCIBoardPolicy smoke (skipped without a stockfish binary).
"""
import shutil

import chess
import numpy as np
import pytest

from catspace.realboard import (BoardGameRecord, RandomBoardPolicy, play_board_game,
                                   record_to_pgn)


def test_random_game_terminates_and_is_legal():
    rec = play_board_game(RandomBoardPolicy(), RandomBoardPolicy(),
                          max_plies=250, rng=np.random.default_rng(0))
    board = chess.Board()
    for uci in rec.moves:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        board.push(move)
    assert rec.result in ("1-0", "0-1", "1/2-1/2", "*")
    assert rec.final_fen == board.fen()


def test_game_deterministic_per_seed():
    a = play_board_game(RandomBoardPolicy(), RandomBoardPolicy(), opening_plies=4,
                        max_plies=120, rng=np.random.default_rng(7))
    b = play_board_game(RandomBoardPolicy(), RandomBoardPolicy(), opening_plies=4,
                        max_plies=120, rng=np.random.default_rng(7))
    c = play_board_game(RandomBoardPolicy(), RandomBoardPolicy(), opening_plies=4,
                        max_plies=120, rng=np.random.default_rng(8))
    assert a.moves == b.moves
    assert a.moves != c.moves


def test_record_to_pgn_roundtrip():
    rec = play_board_game(RandomBoardPolicy(), RandomBoardPolicy(),
                          max_plies=60, rng=np.random.default_rng(1))
    game = record_to_pgn(rec, "a", "b")
    assert game.headers["Result"] == rec.result
    assert len(list(game.mainline_moves())) == rec.n_plies


def test_fb_policy_legal_and_takes_mate():
    torch = pytest.importorskip("torch")
    from catspace.nn.fb import TorchFB
    from catspace.nn.policy_fb import FBBoardPolicy

    fb = TorchFB(d=16, channels=16, blocks=2, enc_out=64, dh=64, omega_dim=4, seed=0)
    z = np.zeros(16, dtype=np.float32)
    rng = np.random.default_rng(0)

    for depth in (1, 2):
        pol = FBBoardPolicy(fb, z, depth=depth)
        board = chess.Board()
        for _ in range(6):
            move = pol.move(board, rng)
            assert move in board.legal_moves
            board.push(move)

        # back-rank mate in 1 must be taken via the terminal short-circuit,
        # whatever the (random) field says
        mate_board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        chosen = pol.move(mate_board, rng)
        mate_board.push(chosen)
        assert mate_board.is_checkmate()


def test_fb_policy_avoids_being_mated_at_depth2():
    torch = pytest.importorskip("torch")
    from catspace.nn.fb import TorchFB
    from catspace.nn.policy_fb import FBBoardPolicy

    fb = TorchFB(d=16, channels=16, blocks=2, enc_out=64, dh=64, omega_dim=4, seed=0)
    pol = FBBoardPolicy(fb, np.zeros(16, dtype=np.float32), depth=2)
    # black threatens Qh4#; white's only non-losing tries block/defend -- with
    # MIN over replies, any move allowing mate scores MATED and is avoided.
    # (fools-mate setup: after 1.f3 e5 2.g4?? comes Qh4#; here white to move
    # must NOT play g4.)
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2")
    move = pol.move(board, np.random.default_rng(0))
    board.push(move)
    can_be_mated = any(
        (lambda b: (b.push(r) or b.is_checkmate()))(board.copy(stack=False))
        for r in board.legal_moves
    )
    assert not can_be_mated


def test_fb_policy_move_scored_matches_move_and_reports_feared():
    torch = pytest.importorskip("torch")
    from catspace.nn.fb import TorchFB
    from catspace.nn.policy_fb import FBBoardPolicy

    fb = TorchFB(d=16, channels=16, blocks=2, enc_out=64, dh=64, omega_dim=4, seed=0)
    z = np.zeros(16, dtype=np.float32)
    rng = np.random.default_rng(0)

    for depth in (1, 2):
        pol = FBBoardPolicy(fb, z, depth=depth)
        board = chess.Board()
        move, cands = pol.move_scored(board, rng)
        # move() must agree exactly with move_scored()'s first return value
        move2 = pol.move(board, np.random.default_rng(0))
        assert move == move2
        assert len(cands) == len(list(board.legal_moves))
        assert sum(c["chosen"] for c in cands) == 1
        chosen = next(c for c in cands if c["chosen"])
        assert chess.Move.from_uci(chosen["uci"]) == move
        # sorted descending by score (mate/reach/draw/mated ordering)
        scores = [c["score"] for c in cands]
        assert scores == sorted(scores, reverse=True)

    # depth-2 candidates that allow a mate-in-1 reply must carry a feared_* triple
    pol2 = FBBoardPolicy(fb, z, depth=2)
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2")
    _, cands = pol2.move_scored(board, rng)
    mated = [c for c in cands if c["kind"] == "mated"]
    assert mated and all("feared_san" in c for c in mated)


def test_fb_policy_move_scored_mate_shortcircuit():
    torch = pytest.importorskip("torch")
    from catspace.nn.fb import TorchFB
    from catspace.nn.policy_fb import FBBoardPolicy

    fb = TorchFB(d=16, channels=16, blocks=2, enc_out=64, dh=64, omega_dim=4, seed=0)
    pol = FBBoardPolicy(fb, np.zeros(16, dtype=np.float32), depth=2)
    mate_board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    move, cands = pol.move_scored(mate_board, np.random.default_rng(0))
    assert len(cands) == 1 and cands[0]["kind"] == "mate" and cands[0]["chosen"]
    mate_board.push(move)
    assert mate_board.is_checkmate()


def test_fb_search_policy_legal_and_takes_mate():
    torch = pytest.importorskip("torch")
    from catspace.nn.fb import TorchFB
    from catspace.nn.policy_fb import FBSearchPolicy

    fb = TorchFB(d=16, channels=16, blocks=2, enc_out=64, dh=64, omega_dim=4, seed=0)
    z = np.zeros(16, dtype=np.float32)
    rng = np.random.default_rng(0)

    for max_nodes, beam in ((40, 4), (200, 4), (3000, 6)):
        pol = FBSearchPolicy(fb, z, max_nodes=max_nodes, beam=beam)
        board = chess.Board()
        for _ in range(4):
            move = pol.move(board, rng)
            assert move in board.legal_moves
            board.push(move)

        mate_board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        chosen = pol.move(mate_board, rng)
        mate_board.push(chosen)
        assert mate_board.is_checkmate()


def test_fb_search_policy_avoids_being_mated_at_depth4():
    torch = pytest.importorskip("torch")
    from catspace.nn.fb import TorchFB
    from catspace.nn.policy_fb import FBSearchPolicy

    fb = TorchFB(d=16, channels=16, blocks=2, enc_out=64, dh=64, omega_dim=4, seed=0)
    pol = FBSearchPolicy(fb, np.zeros(16, dtype=np.float32), max_nodes=3000, beam=6)
    # same fool's-mate setup as the FBBoardPolicy depth-2 test: white must
    # not play g4?? (Qh4# follows). max_nodes=3000 comfortably reaches
    # depth>=2 from this position's ~20-ish branching, matching the old
    # depth=4 test's intent.
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2")
    move = pol.move(board, np.random.default_rng(0))
    board.push(move)
    can_be_mated = any(
        (lambda b: (b.push(r) or b.is_checkmate()))(board.copy(stack=False))
        for r in board.legal_moves
    )
    assert not can_be_mated


def test_fb_search_policy_finds_forced_mate_in_2():
    """z=0 makes every non-terminal leaf score exactly 0 -- with no reach
    signal at all, move selection is driven ENTIRELY by the MATE_SCORE/
    MATED_SCORE terminal propagation, isolating tree-search correctness
    from embedding quality. depth=2 (FBBoardPolicy-equivalent) can only
    ever see a mate-in-1; this position needs depth>=3 (my move, black's
    forced reply, my mating move) to find deliberately. Two rooks vs a lone
    king in the corner: controlling the 7th rank with one rook forces the
    black king to g8 (its only legal square), and the other rook mates on
    the 8th next move."""
    torch = pytest.importorskip("torch")
    from catspace.nn.fb import TorchFB
    from catspace.nn.policy_fb import FBSearchPolicy

    fb = TorchFB(d=16, channels=16, blocks=2, enc_out=64, dh=64, omega_dim=4, seed=0)
    # max_nodes=3000, beam=6 comfortably reaches depth>=3 from this sparse
    # (~30 legal moves) position: R + R*6 + R*36 ~= 30*43 = 1290 <= 3000.
    pol = FBSearchPolicy(fb, np.zeros(16, dtype=np.float32), max_nodes=3000, beam=6)
    rng = np.random.default_rng(0)
    board = chess.Board("7k/8/8/8/8/8/8/RR4K1 w - - 0 1")

    m1 = pol.move(board, rng)
    assert pol.last_depth_used >= 3, f"expected depth>=3 from the node budget, got {pol.last_depth_used}"
    board.push(m1)
    assert not board.is_game_over()
    replies = list(board.legal_moves)
    assert len(replies) == 1, f"expected the king cornered to one square, got {replies}"
    board.push(replies[0])

    m2 = pol.move(board, rng)
    board.push(m2)
    assert board.is_checkmate()


def test_fb_search_policy_plan_matches_move_and_has_subgoal():
    torch = pytest.importorskip("torch")
    from catspace.nn.fb import TorchFB
    from catspace.nn.policy_fb import FBSearchPolicy

    fb = TorchFB(d=16, channels=16, blocks=2, enc_out=64, dh=64, omega_dim=4, seed=0)
    pol = FBSearchPolicy(fb, np.zeros(16, dtype=np.float32), max_nodes=3000, beam=6)
    rng = np.random.default_rng(0)
    board = chess.Board("7k/8/8/8/8/8/8/RR4K1 w - - 0 1")

    move, subgoal = pol.plan(board, rng)
    move2 = pol.move(board, np.random.default_rng(0))
    assert move == move2, "plan()'s chosen move must agree with move()'s"
    assert move in board.legal_moves
    # the PV subgoal is a real position several plies deeper than the root,
    # not just the root move applied once
    after_root = board.copy(stack=False)
    after_root.push(move)
    assert subgoal.board_fen() != after_root.board_fen() or subgoal.is_checkmate()

    # forced mate-in-2 from this position: the PV should walk all the way
    # down to the actual mate, i.e. the subgoal itself is checkmate
    assert subgoal.is_checkmate()


def test_fb_plan_policy_legal_and_takes_mate():
    torch = pytest.importorskip("torch")
    from catspace.nn.fb import TorchFB
    from catspace.nn.policy_fb import FBPlanPolicy

    fb = TorchFB(d=16, channels=16, blocks=2, enc_out=64, dh=64, omega_dim=4, seed=0)
    z = np.zeros(16, dtype=np.float32)
    rng = np.random.default_rng(0)

    pol = FBPlanPolicy(fb, z, plan_nodes=200, plan_beam=4, shallow_nodes=40, shallow_beam=3)
    board = chess.Board()
    for _ in range(6):
        move = pol.move(board, rng)
        assert move in board.legal_moves
        board.push(move)

    mate_board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    pol2 = FBPlanPolicy(fb, z, plan_nodes=200, plan_beam=4, shallow_nodes=40, shallow_beam=3)
    chosen = pol2.move(mate_board, rng)
    mate_board.push(chosen)
    assert mate_board.is_checkmate()


def test_fb_plan_policy_holds_plan_across_plies():
    """z=0 means every shallow reach call sees the SAME subgoal_z against a
    board that (for a static/unreachable subgoal) won't move reach much --
    the point is just that the executor, not the deep planner, is doing the
    picking on non-replan plies, and plans_made stays well below the ply
    count instead of replanning every single move."""
    torch = pytest.importorskip("torch")
    from catspace.nn.fb import TorchFB
    from catspace.nn.policy_fb import FBPlanPolicy

    fb = TorchFB(d=16, channels=16, blocks=2, enc_out=64, dh=64, omega_dim=4, seed=0)
    z = np.zeros(16, dtype=np.float32)
    rng = np.random.default_rng(0)

    pol = FBPlanPolicy(fb, z, plan_nodes=200, plan_beam=4, shallow_nodes=40, shallow_beam=3,
                        max_plies_per_plan=6, drop_delta=2.0, achieved_cos=2.0)
    board = chess.Board()
    n_plies = 8
    for _ in range(n_plies):
        move = pol.move(board, rng)
        assert move in board.legal_moves
        board.push(move)
    # drop_delta/achieved_cos set outside [-1,1] so only the plies-cap can
    # force a replan: 8 plies / max_plies_per_plan=6 -> exactly 2 plans.
    assert pol.plans_made == 2, f"expected exactly 2 plans (initial + one stall-replan), got {pol.plans_made}"


def test_fb_search_policy_goal_bank_readout():
    """Bank (m,d) goals: identical-exemplar bank must reproduce the single-
    goal scores exactly (soft-min normalizer), the policy must stay legal,
    and terminal short-circuits (mate-in-1) must be unaffected by banks."""
    torch = pytest.importorskip("torch")
    from catspace.nn.fb import TorchFB
    from catspace.nn.policy_fb import FBSearchPolicy, soft_min_bank

    fb = TorchFB(d=16, channels=16, blocks=2, enc_out=64, dh=64, omega_dim=4, seed=0)
    rng = np.random.default_rng(0)
    z = np.random.default_rng(1).normal(size=16).astype(np.float32)

    f = torch.nn.functional.normalize(torch.randn(5, 16), dim=1)
    zt = torch.from_numpy(z)
    dup_bank = zt[None, :].repeat(7, 1)
    single = fb.score(f, zt)
    banked = soft_min_bank(fb, f, dup_bank, tau=0.1)
    assert torch.allclose(single, banked, atol=1e-5)

    bank = np.stack([z, -z, np.roll(z, 3)])
    pol = FBSearchPolicy(fb, bank, max_nodes=100, beam=3)
    board = chess.Board()
    for _ in range(4):
        move = pol.move(board, rng)
        assert move in board.legal_moves
        board.push(move)

    mate_board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    chosen = pol.move(mate_board, rng)
    mate_board.push(chosen)
    assert mate_board.is_checkmate()


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="no stockfish binary")
def test_uci_policy_smoke():
    from catspace.uci import UCIBoardPolicy
    rng = np.random.default_rng(0)
    with UCIBoardPolicy(movetime=0.01, elo=1320) as sf:
        board = chess.Board()
        for _ in range(4):
            move = sf.move(board, rng)
            assert move in board.legal_moves
            board.push(move)
