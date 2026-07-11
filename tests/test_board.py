from latentchess.board import (NSQ, chebyshev, king_moves, knight_moves, rook_slides,
                                rook_attacks, sq, rc, KING_MOVES, KNIGHT, KN_ATT)


def test_sq_rc_roundtrip():
    assert rc(sq(2, 3)) == (2, 3)
    assert sq(*rc(7)) == 7


def test_chebyshev():
    assert chebyshev(0, 0) == 0
    assert chebyshev(sq(0, 0), sq(4, 4)) == 4
    assert chebyshev(sq(1, 2), sq(2, 3)) == 1


def test_king_moves():
    assert sq(1, 1) in king_moves(sq(0, 0))
    assert sq(0, 0) not in king_moves(sq(0, 0))
    assert len(KING_MOVES) == NSQ


def test_knight_moves():
    center = sq(2, 2)
    assert set(knight_moves(center)) == {
        sq(0, 1), sq(0, 3), sq(1, 0), sq(1, 4),
        sq(3, 0), sq(3, 4), sq(4, 1), sq(4, 3)
    }
    assert len(KNIGHT) == NSQ
    assert len(KN_ATT) == NSQ
    assert KN_ATT[center] == set(knight_moves(center))


def test_rook_slides_blocked():
    actual = rook_slides(sq(2, 2), {sq(2, 4)})
    assert sq(2, 4) not in actual
    assert sq(2, 3) in actual


def test_rook_attacks():
    # no blocker on the line -> the rook attacks all the way down the row
    assert rook_attacks(sq(0, 0), sq(0, 4), {sq(4, 4)})
    # a blocker strictly between rook and target -> blocked, regardless of
    # where exactly it sits in between (the original test asserted the
    # opposite for a blocker at sq(0,2) -- a pre-existing bug: both sq(0,2)
    # and sq(0,3) sit strictly between sq(0,0) and sq(0,4), so both block)
    assert not rook_attacks(sq(0, 0), sq(0, 4), {sq(0, 2)})
    assert not rook_attacks(sq(0, 0), sq(0, 4), {sq(0, 3)})
