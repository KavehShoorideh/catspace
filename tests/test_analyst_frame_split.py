"""Kaveh's bet, measured (catspace/incubator/analyst/frame_split.py): frames, the critical path, passes."""
import chess

from catspace.incubator.analyst.frame_split import chain, split, with_passes


def _line(fen, sans):
    b = chess.Board(fen); bb = b.copy(); ms = []
    for s in sans:
        m = bb.parse_san(s); ms.append(m); bb.push(m)
    return b, ms


def test_two_interleaved_pawn_runs_are_two_chains():
    b, ms = _line("4k3/7p/8/8/8/8/P7/4K3 w - - 0 1", ["a4", "h5", "a5", "h4", "a6", "h3", "a7", "h2"])
    n, path = chain(b, ms)
    assert (n, path) == (4, [0, 2, 4, 6])                             # one pawn's run: each push needs the last
    assert with_passes(path) == 7                                     # three passes for the other side in between
    assert len(split(b, ms)) == 2


def test_an_exchange_chains_through_its_square_and_a_flight_through_the_attack():
    b, ms = _line("4k3/8/2n5/3p4/4P3/5N2/8/4K3 w - - 0 1", ["exd5", "Nb4", "Nd4", "Nxd5"])
    assert chain(b, ms) == (2, [0, 3])                                # the capture and the recapture on d5
    assert chain(b, ms, attacks=True) == (3, [0, 1, 3])               # Nb4 flees the pawn that now attacks c6
    assert with_passes([0, 1, 3]) == 4
