"""The search ruler (ANALYST_PLAN 46.3): hand-made positions; the Stockfish reference is skipped when no engine is installed."""
import shutil

import chess
import pytest

from catspace.incubator.analyst.discovery import ruler as R


def test_mate_in_one_at_depth_one():
    r = R.solve("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 100, max_depth=3)
    assert r["status"] == "forced" and r["depth"] == 1 and r["line"][0] == "Ra8#"


def test_hanging_piece_at_depth_one():
    r = R.solve("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", 9, max_depth=3)
    assert r["status"] == "forced" and r["depth"] == 1 and r["line"][0] == "Rxd5"


def test_defended_piece_is_not_a_gain():
    # the queen is guarded by the bishop: Rxd5 Bxd5 nets 4, not 9
    r = R.solve("4k3/8/4b3/3q4/8/8/8/3RK3 w - - 0 1", 9, max_depth=3)
    assert r["status"] == "stopped"


def test_no_gain_is_stopped():
    r = R.solve("4k3/8/8/8/8/8/P7/4K3 w - - 0 1", 5, max_depth=3)
    assert r["status"] == "stopped" and r["depth"] == 3 and r["nodes_main"] > 0


def test_capture_before_quiet():
    b = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    moves = R.Ruler(b, 0, 0).order(0, None)
    assert b.is_capture(moves[0])
    assert b.san(moves[0]) == "Rxd5"


def test_checks_before_quiet_after_captures():
    b = chess.Board("2Q2bk1/5p1p/p5p1/2p3P1/8/7P/qPK2P2/7R b - - 0 33")  # 000Sa after Bxc2 Kxc2, black to move
    r = R.Ruler(b, 0, 0)
    sans = [b.san(m) for m in r.order(0, None)]
    assert sans.index("Qc4+") < sans.index("Kh8") and sans.index("Qc4+") < sans.index("h6")
    # the bad capture (Qxb2+ is met by Kxb2) comes after the quiet moves
    assert sans[-1] == "Qxb2+"


def test_hash_move_first():
    b = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    quiet = chess.Move.from_uci("e1f1")
    assert R.Ruler(b, 0, 0).order(0, quiet)[0] == quiet


def test_see():
    b = chess.Board("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1")
    assert R.see(b, chess.Move.from_uci("d1d5")) == -8      # queen for a pawn, recaptured by the pawn
    b = chess.Board("4k3/8/8/3p4/8/8/8/3RK3 w - - 0 1")
    assert R.see(b, chess.Move.from_uci("d1d5")) == 1


def test_trace_first_entry_is_roots_first_child():
    fen = "4k3/8/8/8/8/8/P7/4K3 w - - 0 1"
    b = chess.Board(fen)
    first = b.san(R.Ruler(b, 5, 5).order(0, None)[0])
    r = R.solve(fen, 5, max_depth=1, trace=5)
    ply, path, kind, reason, score = r["trace"][0]
    assert (ply, path, kind) == (1, first, "main")
    assert len(r["trace"]) <= 5
    assert {t[3] for t in r["trace"]} <= {"cut: beta", "cut: stand-pat", "leaf", "mate", "draw", "repeat", "searched"}


def test_counts_main_and_quiescence_apart():
    r = R.solve("r3k3/8/8/8/8/8/8/R3K3 w - - 0 1", 5, max_depth=2)
    assert r["nodes_main"] > 0 and r["nodes_q"] >= 0 and r["nodes"] == r["nodes_main"] + r["nodes_q"]


@pytest.mark.skipif(not (shutil.which("stockfish") or shutil.os.path.exists("/opt/homebrew/bin/stockfish")), reason="no stockfish")
def test_stockfish_reference():
    r = R.solve("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 100, answer_uci="a1a8")
    assert r["stockfish"]["depth"] >= 1 and r["stockfish"]["nodes"] > 0
