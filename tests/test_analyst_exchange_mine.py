import chess

from catspace.incubator.analyst.exchange_mine import _hanging, _move_cost


def test_a_move_that_hangs_a_piece_is_flagged():
    b = chess.Board("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1")            # Rd2-a2?? leaves... no: Rd2 is attacked by the queen
    b = chess.Board("4k3/8/2n5/8/8/8/8/R3K3 w - - 0 1")
    m = chess.Move.from_uci("a1a5")                                   # Ra5?? the knight takes it
    assert b.is_legal(m) and _hanging(b, m) == 5
    assert _move_cost(b, m, checks=True) >= 1
    ok = chess.Move.from_uci("a1a2")
    assert _hanging(b, ok) == 0 and _move_cost(b, ok, checks=True) < 1
