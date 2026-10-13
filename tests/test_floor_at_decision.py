"""What matters to them is read at the decision's balance, not after our move (plan.rs net_after, 2026-10-13)."""
import catspace.incubator.analyst  # noqa: F401
import analyst_core as A


def test_a_grab_does_not_raise_their_bar():
    # Black 11 down (a rook matters to them); Rxh7 takes a pawn -- 12 down after it, where the floor is 9 -- and Kxh7
    # takes the rook. The audit must count it.
    fen = "6k1/R4p1p/8/8/6p1/3P3R/PPP5/2K5 w - - 0 3"
    assert A.gain_floor_of(-11) == 5 and A.gain_floor_of(-12) == 9
    net, why = A.net_of(fen, "h3h7")
    assert net < 0 and "Kxh7" in why
    assert max(n for u, n, _ in A.audit_check(fen, []) if u == "h3h7") < max(n for _, n, _ in A.audit_check(fen, []))
