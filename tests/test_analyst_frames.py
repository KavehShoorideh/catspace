"""Ideas as live frames: value from the frame's episode, logic from its signature, the listener naming who is
affected and why (Kaveh 2026-09-24, the Bxh2+ position)."""
import chess

from catspace.incubator.analyst.chess_domain import domain
from catspace.incubator.analyst.frames import Listener, signature, value_of

FEN = "2kr3r/ppp2ppp/2nb1n2/7q/3P2b1/2P2N2/PP1NBPPP/R1BQ1RK1 b - - 0 1"


def test_same_tactic_only_a_pawn_after_re1():
    """The bishop sacrifice frame: +4 with the rook on f1; after Rf1-e1 the same frame, re-derived, is worth +1, and
    the changed relation is named: the rook now defends e2."""
    dom = domain(oracle="frame")
    ids = [chess.D6, chess.H2, chess.F3, chess.E2, chess.D1, chess.F1, chess.G1, chess.H5]
    L = Listener(dom, chess.Board(FEN))
    L.board.turn = chess.WHITE                                                # white to move first in this test: Re1
    sac = L.register("bishop sacrifice", ids, None)
    assert value_of(chess.Board(FEN), ids, chess.Move.from_uci("d6h2")) == 4.0
    sac.move = chess.Move.from_uci("d6h2")
    rep = L.apply(chess.Move.from_uci("f1e1"), by="white")
    (name, touched, changed, old, new), = rep
    assert touched and new == 1.0
    assert any(r == "defends" and b is True and idea_sq(L, sl) == ("e1", "e2") for (r, sl), a, b in changed), changed
    assert "defends(e1,e2) False->True" in L.why("bishop sacrifice")


def idea_sq(L, sl):
    ids = L.ideas["bishop sacrifice"].ids
    return tuple(chess.square_name(ids[s]) for s in sl)


def test_listener_names_the_frames_a_piece_affects_and_recalculates_the_other():
    """Two frames share the knight: the check-response frame (Bxh2+, the knight recaptures) and the frame of the
    light bishop taking e2. When the first frame's line moves the knight, the second is touched and re-derived."""
    dom = domain(oracle="frame")
    b = chess.Board(FEN)
    L = Listener(dom, b)
    L.register("check response", [chess.D6, chess.H2, chess.F3, chess.G1, chess.H5], chess.Move.from_uci("d6h2"))
    L.register("take e2", [chess.G4, chess.E2, chess.D1, chess.F1, chess.F3, chess.H5], chess.Move.from_uci("g4e2"))
    assert set(L.affected(chess.F3)) == {"check response", "take e2"}
    assert L.affected(chess.A1) == []
    assert L.ideas["take e2"].value is None                                  # Bxe2 is not even legal yet: the knight on f3 blocks the diagonal (the armed tactic)
    L.apply(chess.Move.from_uci("d6h2"), by="check response")               # the sacrifice
    rep = L.apply(chess.Move.from_uci("f3h2"), by="check response")         # the knight recaptures: the other frame is touched
    take = next(r for r in rep if r[0] == "take e2")
    assert take[1] and L.ideas["take e2"].ids[4] == chess.H2                 # the knight followed to h2 inside the other frame
    assert L.ideas["take e2"].value is not None and L.ideas["take e2"].value > 0   # now the bishop takes e2 and the frame gains
    assert "touched by check response" in L.why("take e2") and "attacks(g4,e2) False->True" in L.why("take e2")
