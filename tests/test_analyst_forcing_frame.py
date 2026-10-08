import chess

from catspace.incubator.analyst.forcing_frame import capture_value


def test_a_capture_elsewhere_is_answered():
    """Nxd5 wins a knight; Black's bishop takes h2 in between; the forcing frame counts both: +2."""
    b = chess.Board("4k3/8/3b4/3n4/8/4N3/7P/1K6 w - - 0 1")
    assert capture_value(b, chess.D5, use_checks=False)[0] == 2


def test_a_check_in_between_wins_the_queen():
    """Qxd5 grabs a pawn, but ...Bb4+ is not even needed: Rxd5 wins the queen -- the exchange; and in the second
    position the check skewers: after Qxa7?? Rh1+ wins the queen on the next move (king moves, Rh7xa7 not legal...) --
    kept simple: the plain recapture is found."""
    b = chess.Board("3rk3/8/8/3p4/8/8/8/3QK3 w - - 0 1")
    assert capture_value(b, chess.D5)[0] == 1 - 9


def test_a_pinned_recapturer_is_not_available_in_the_exchange_frame():
    from catspace.incubator.analyst.exchange_frame import Search, frame_of, root_values
    b = chess.Board("4r1k1/8/8/3p4/4B3/4K3/8/8 b - - 0 1")              # ...no: build the pin with the rook on e8 against Ke3
    b = chess.Board("4r1k1/8/8/3P4/2q1B3/4K3/8/8 b - - 0 1")
    assert b.is_pinned(chess.WHITE, chess.E4)
    st = frame_of(b, chess.D5)
    assert [r.square for r in st[2]] == []                               # the pinned bishop cannot recapture on d5
    assert max(root_values(Search(True), *st).values()) == 1
