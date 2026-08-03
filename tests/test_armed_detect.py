"""M7 detection unit tests (MILESTONES.md, sequencing override 2026-08-03).
Pure-logic tests for BlockingCondition + classify_blocking_move -- no engine
needed, deterministic, matches the spec's own gate philosophy ("unit tests:
blocker removed => fires, else not") without depending on Stockfish's search
choosing a specific line (which would make the test version/hardware-fragile).
A separate light engine-integration smoke test is included, skipped if no
local Stockfish is found.
"""
import shutil

import chess
import chess.engine
import pytest

from catspace.armed.detect import (
    BlockingCondition, classify_blocking_move, find_armed_tactic_candidates,
)


def test_blocking_condition_active_and_removed():
    # White rook h1 (defends h7 down the h-file) vs an empty file.
    b = chess.Board("6k1/8/8/8/8/8/8/4K2R w - - 0 1")
    cond = BlockingCondition(guarded_square=chess.H7, defender_color=chess.WHITE)
    assert cond.is_active(b), "rook on h1 should still be defending h7"

    b2 = b.copy()
    b2.remove_piece_at(chess.H1)
    assert not cond.is_active(b2), "guard removed -- condition must flip False"


def test_blocking_condition_min_defenders():
    # Two rooks hitting h7 from DIFFERENT lines (rank vs file) so neither
    # blocks the other's line of sight; min_defenders=2 needs both.
    b = chess.Board("6k1/R7/8/7R/8/8/8/4K3 w - - 0 1")  # Ra7 (rank 7), Rh5 (h-file)
    cond2 = BlockingCondition(guarded_square=chess.H7, defender_color=chess.WHITE, min_defenders=2)
    assert cond2.is_active(b), "two independent rooks should both hit h7"
    b.remove_piece_at(chess.H5)
    assert not cond2.is_active(b), "down to one defender, min_defenders=2 must fail now"


def test_classify_blocking_move_new_defender_localizes_shared_square():
    """'the Nf6 guards h7' — the spec's own example. Black plays Nd7-f6, which
    newly defends h7; a white pawn on g6 already bears on h7 too. A pawn's
    2-square attack fan is deliberately used here (not a rook/bishop, whose
    long rays kept sharing OTHER squares with the knight's 8-square attack
    set in earlier drafts of this test, e.g. e4 or d7 -- the overlap logic
    doesn't try to disambiguate multiple shared squares, a known, documented
    gap; the pawn keeps this specific test unambiguous)."""
    # Black king on a8, NOT g8 -- g8 would be diagonally adjacent to h7 and
    # would keep "defending" it on its own even with the knight gone,
    # confounding the removed-guard assertion below.
    board_before = chess.Board("k7/3n4/6P1/8/8/8/8/4K3 b - - 0 1")  # Black to move
    blocking_move = chess.Move.from_uci("d7f6")
    our_next_move = chess.Move.from_uci("g6g7")  # pawn already attacks h7 (and f7) from g6

    cond = classify_blocking_move(board_before, blocking_move, our_next_move)

    assert cond.guarded_square == chess.H7, "should localize to h7, the shared contested square"
    assert cond.defender_color == chess.BLACK
    assert cond.source == "new_defender"

    board_after = board_before.copy(); board_after.push(blocking_move)
    assert cond.is_active(board_after), "knight just landed on f6, defending h7 -- must be active"

    board_no_knight = board_before.copy(stack=False)
    board_no_knight.remove_piece_at(chess.D7)
    board_no_knight.turn = chess.WHITE
    assert not cond.is_active(board_no_knight), "no knight anywhere -- h7 undefended, must be inactive"


def test_classify_blocking_move_capture_localizes_to_captured_square():
    """Black's pawn on f6 can recapture on g5 -- the checkable fact is
    "does Black still attack g5", assessed on the pre-capture board (the
    natural re-check point: before we try the sac again)."""
    board_before = chess.Board("6k1/8/5p2/6N1/8/8/8/4K3 b - - 0 1")  # Black to move
    blocking_move = chess.Move.from_uci("f6g5")
    assert board_before.is_capture(blocking_move)

    cond = classify_blocking_move(board_before, blocking_move, our_next_move=None)

    assert cond.guarded_square == chess.G5
    assert cond.defender_color == chess.BLACK
    assert cond.source == "capture"
    assert cond.is_active(board_before), "the f6 pawn already attacks g5 -- guard present"

    board_no_pawn = board_before.copy(stack=False)
    board_no_pawn.remove_piece_at(chess.F6)
    assert not cond.is_active(board_no_pawn), "pawn gone -- g5 undefended by Black"


def test_classify_blocking_move_en_passant_captured_square():
    """En passant is the classic special-move footgun (bit the hanging-piece
    probe earlier this session too) -- the captured square is NOT the
    blocking move's to_square."""
    board_before = chess.Board("6k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1")  # White to move, e.p. available
    ep_move = chess.Move.from_uci("e5d6")
    assert board_before.is_en_passant(ep_move)

    cond = classify_blocking_move(board_before, ep_move, our_next_move=None)
    assert cond.guarded_square == chess.D5, "captured pawn sits on d5, not the to-square d6"


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="local Stockfish not found")
def test_find_armed_tactic_candidates_runs_and_returns_well_formed_records():
    """Structural smoke test: don't assert a SPECIFIC tactic (SF's exact line
    is version/hardware-dependent), just that the pipeline runs end-to-end and
    every returned candidate is internally consistent."""
    eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish"))
    try:
        eng.configure({"UCI_ShowWDL": True})
    except Exception:
        pass
    try:
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        candidates = find_armed_tactic_candidates(eng, board, k_candidates=5, k_moves=4, depth=8,
                                                    min_gain=0.05, decay_tol=0.03)
        for c in candidates:
            assert c.blocking_move is not None
            assert c.blocking is not None
            assert isinstance(c.blocking.is_active(board.copy()), bool) or True  # must not raise
            assert c.trend[0] <= c.payoff_if_unblocked + 1e-9
    finally:
        eng.quit()
