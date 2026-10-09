"""Maia in Rust (analyst_core.maia_policy / frame_outs): a distribution over the legal moves; the frame's outs on
Kaveh's castling trap (2026-10-09): after O-O-O, Nxf3 walks into Ne7+ Kb8 Rxd8#."""
import chess
import pytest

from catspace.io import paths

MODEL = str(paths.assets_dir() / "engines" / "maia_onnx" / "maia-1500.onnx")


def test_policy_is_a_distribution_over_the_legal_moves():
    import analyst_core
    b = chess.Board("2kr1n1r/ppp3p1/6p1/3Nn1Pp/5N1P/P3PP2/1PP5/R3K2R w KQ - 0 1")
    pol = dict(analyst_core.maia_policy(b.fen(), MODEL, 1.0))
    assert set(pol) == {m.uci() for m in b.legal_moves}
    assert abs(sum(pol.values()) - 1) < 1e-4


def test_the_castling_trap_has_a_likely_out():
    import analyst_core
    b = chess.Board("2kr1n1r/ppp3p1/6p1/3Nn1Pp/5N1P/P3PP2/1PP5/R3K2R w KQ - 0 1")
    rows = analyst_core.frame_outs(b.fen(), b.parse_san("O-O-O").uci(), MODEL, 1.0, 12)
    nxf3 = [r for r in rows if r[0] == "e5f3"]
    assert nxf3 and nxf3[0][1] > 0.3 and nxf3[0][2] >= 40          # likely, and it walks into mate
