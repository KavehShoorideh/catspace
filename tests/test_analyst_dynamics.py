"""L4 of the analyst: effects, null-move threats, zugzwang. Needs a Stockfish binary."""
import shutil
import chess
import pytest

from catspace.incubator.analyst import dynamics as L4
from catspace.incubator.analyst.engine import Engine

pytestmark = pytest.mark.skipif(not (shutil.which("stockfish") or shutil.which("/opt/homebrew/bin/stockfish")), reason="no stockfish")


@pytest.fixture(scope="module")
def eng(tmp_path_factory):
    e = Engine(depth=12, cache_path=str(tmp_path_factory.mktemp("c") / "cache.sqlite"))
    yield e
    e.close()


def test_move_effects_capture_and_pin():
    b = chess.Board("r1bqkbnr/ppp2ppp/2n5/1B1pp3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq d6 0 4")
    e = L4.move_effects(b, chess.Move.from_uci("e4d5"))
    assert e.captured == ("pawn", 1)
    b = chess.Board("r1bqkbnr/ppp2ppp/2n5/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq d6 0 4")
    e = L4.move_effects(b, chess.Move.from_uci("f1b5"))            # Bb5 pins Nc6 to the king
    assert any(r.name == "pin" and r.names() == ("b5", "c6", "e8") for r in e.created)


def test_mate_threat_is_real_and_named(eng):
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3")   # 3.Qh5, black to move
    th = L4.threats(b, eng)
    assert th and th[0].real and th[0].line.mate == 1 and "Qxf7#" in th[0].san_line(b)


def test_quiet_position_has_no_real_threat(eng):
    b = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")     # after 2.Nf3
    th = L4.threats(b, eng)
    assert th and th[0].gain_cp < 150                                # a tempo, nothing more


def test_cache_hits(eng):
    b = chess.Board()
    eng.analyse(b); c0 = eng.calls; eng.analyse(b)
    assert eng.calls == c0 and eng.hits >= 1
