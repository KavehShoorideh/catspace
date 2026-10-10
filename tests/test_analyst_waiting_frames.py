"""Waiting frames (rust/analyst_core/src/waiting.rs): missing conditions, breakers, the reverse lookup, the live pool.
Fixture: Kaveh's castling position (2026-10-09) -- Ne7+ mates once a rook stands on the d-file."""
import chess
import pytest

analyst_core = pytest.importorskip("analyst_core")

CASTLE = "2kr1n1r/ppp3p1/6p1/3Nn1Pp/5N1P/P3PP2/1PP5/R3K2R w KQ - 0 1"


def _frame(rows, owner, uci):
    return [f for f in rows if f[0] == owner and f[1] == uci]


def test_ne7_waits_on_a_rook_on_the_d_file():
    (f,) = _frame(analyst_core.waiting_frames(CASTLE), "w", "d5e7")
    assert f[2] == "cornering king" and f[6] == "dormant"
    waits = {(k, r, s) for k, r, s, _ in f[7]}
    assert ("add ours", "R", "d1") in waits
    # the control: no condition that wins by itself (a pawn on e7 promotes on d8)
    assert not any(r == "P" and s == "e7" for _, r, s, _ in f[7])


def test_castling_long_arms_the_mate():
    arms, hands, defuses = analyst_core.arming_moves(CASTLE)
    armed = {(m, seed) for m, seed, *_ in arms}
    assert ("e1c1", "d5e7") in armed and ("a1d1", "d5e7") in armed
    assert all(v >= 500 for m, seed, _, _, v, _ in arms if seed == "d5e7")


def test_after_castling_c6_defuses_and_nxf3_does_not():
    b = chess.Board(CASTLE); b.push_san("O-O-O")
    (f,) = _frame(analyst_core.waiting_frames(b.fen()), "w", "d5e7")
    assert f[6] == "active" and f[8], "an active frame carries its breakers"
    _, _, defuses = analyst_core.arming_moves(b.fen())
    d = {m for m, seed, *_ in defuses if seed == "d5e7"}
    assert "c7c6" in d
    assert "e5f3" not in d


def test_pool_reports_the_threat_turning_active():
    pool = analyst_core.FramePool()
    b = chess.Board(CASTLE)
    pool.observe(b.fen())
    b.push_san("O-O-O")
    ev = pool.observe(b.fen())
    assert any(e[0] == "w" and e[1] == "d5e7" and e[6] == "dormant" and e[7] == "active" for e in ev)
    b.push_san("Nxf3")
    pool.observe(b.fen())
    assert any(f[0] == "w" and f[1] == "d5e7" and f[6] == "active" for f in pool.frames())
