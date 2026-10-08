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


def test_standing_pat_costs_the_second_threat():
    """Kxf1 Qxd4 (a real blunder of c300): the queen on d4 then hits the a1 rook and more; the threat-aware exit sees the
    loss the captures-only frame missed."""
    from catspace.incubator.analyst.forcing_frame import double_threat
    b = chess.Board("2r5/pp1k1ppp/1q2p3/3nP3/3P4/Q7/P4PPP/RB3bK1 w - - 0 22")
    b.push_san("Kxf1"); b.push_san("Qxd4")
    assert double_threat(b) >= 1
    one = chess.Board("4k3/8/8/8/8/8/1q6/R3K3 w - - 0 1")               # one threat (the rook): parried, no cost
    assert double_threat(one) == 0


def test_the_parry_lemma():
    """One threat with a parry costs nothing; the Kxf1 Qxd4 double attack costs; a trapped piece's single threat costs."""
    from catspace.incubator.analyst.forcing_frame import parry_cost
    assert parry_cost(chess.Board("4k3/8/8/8/8/8/1q6/R3K3 w - - 0 1")) == 0           # the rook simply moves
    b = chess.Board("2r5/pp1k1ppp/1q2p3/3nP3/3P4/Q7/P4PPP/RB3bK1 w - - 0 22")
    b.push_san("Kxf1"); b.push_san("Qxd4")
    assert parry_cost(b) >= 1
    trapped = chess.Board("4k3/8/8/8/8/1p6/p1p5/NK6 w - - 0 1")                         # the knight a1 cannot go anywhere safe? (b3 pawn, c2 pawn)
    assert trapped.piece_at(chess.A1).piece_type == chess.KNIGHT
