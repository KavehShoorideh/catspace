"""
Real-board layer tests: play_board_game legality/determinism/PGN round-trip
(no torch needed), FBBoardPolicy legality + mate-taking (torch), and a live
UCIBoardPolicy smoke (skipped without a stockfish binary).
"""
import shutil

import chess
import numpy as np
import pytest

from latentchess.realboard import (BoardGameRecord, RandomBoardPolicy, play_board_game,
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
    from latentchess.nn.fb import TorchFB
    from latentchess.nn.policy_fb import FBBoardPolicy

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
    from latentchess.nn.fb import TorchFB
    from latentchess.nn.policy_fb import FBBoardPolicy

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


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="no stockfish binary")
def test_uci_policy_smoke():
    from latentchess.uci import UCIBoardPolicy
    rng = np.random.default_rng(0)
    with UCIBoardPolicy(movetime=0.01, elo=1320) as sf:
        board = chess.Board()
        for _ in range(4):
            move = sf.move(board, rng)
            assert move in board.legal_moves
            board.push(move)
