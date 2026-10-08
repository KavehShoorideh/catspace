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


def test_hanging_nets_the_move_s_own_capture_and_respects_check():
    b = chess.Board("r2qk2r/2pnbppp/pp2pn2/4p3/2PP4/2NB1Q2/PP3PPP/R1B2RK1 w kq - 0 11")
    assert _hanging(b, b.parse_san("dxe5")) <= 0                   # dxe5 Nxe5: a pawn for a pawn
    b2 = chess.Board("8/p2k1pp1/4p2p/p2nP3/3P4/PK4P1/R4r1P/1B6 b - - 1 30")
    m = b2.parse_san("a4+")
    nb = b2.copy(); nb.push(m); assert nb.is_check()
    assert _hanging(b2, m) < 5                                      # no capture of five that White, in check, can make
