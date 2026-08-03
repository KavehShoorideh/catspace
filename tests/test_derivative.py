"""Phase 2 tests: directional derivative orientation bug (spec 3.1's explicit
warning) and basic ascent-cone sanity."""
import chess
import numpy as np

from catspace.controlfield.control import weighted_attacker_field, orient
from catspace.controlfield.derivative import move_derivatives, ascent_cone, ConeConfig


def test_derivative_orientation_not_naive_diff():
    """The spec's flagged bug: naively subtracting the raw (un-reoriented) post-move
    field from the pre-move field produces nonsense (sign/orientation mismatch across
    the color flip). This only manifests for BLACK's moves -- when the original mover
    is White, orient() is an identity no-op and the naive diff happens to coincide,
    so the test case must be a Black move to actually exercise the bug."""
    b = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")   # Black to move
    moves, D = move_derivatives(b)
    e7e5 = moves[[m.uci() for m in moves].index("e7e5")]
    idx = moves.index(e7e5)

    # naive (buggy) computation: raw post-move field minus raw pre-move field,
    # with NO re-orientation for the color flip.
    c_before_raw = weighted_attacker_field(b)
    b2 = b.copy(); b2.push(e7e5)
    c_after_raw = weighted_attacker_field(b2)
    naive_D = c_after_raw - c_before_raw

    assert not np.allclose(D[idx], naive_D), (
        "move_derivatives matches the naive un-reoriented diff -- the orientation "
        "bug the spec explicitly warns about is present")

    # correct computation, spelled out independently as a second check:
    c_before = orient(c_before_raw, mover_is_white=False)   # original mover was Black
    c_after = orient(c_after_raw, mover_is_white=False)
    assert np.allclose(D[idx], c_after - c_before, atol=1e-6)


def test_ascent_cone_starting_position_runs():
    b = chess.Board()
    out = ascent_cone(b)
    assert 0.0 <= out["cone_size"] <= 1.0
    assert len(out["moves"]) == len(list(b.legal_moves))
    assert out["D"].shape == (len(out["moves"]), 64)


def test_damage_ignores_squares_opponent_cannot_legally_reach():
    """Regression test for the gate-2 fix (reports/phase-2.md): a discovered-check
    move that "loses" defense of a critical square must NOT be penalized by
    damage() if the opponent's only legal replies (forced to resolve check)
    can't actually reach that square. White Rd1+Bd3+Re4, Black Kd8+Re8: Bb5+ is
    discovered check via the d-file; it drops the bishop's defense of e4
    (D_m(e4) very negative, M(e4)=1.0 since a rook sits there) but Black's only
    legal replies are king moves (d8c7/d8c8) -- Re4 is illegal (doesn't resolve
    check) even though the black rook could reach e4 in an unchecked position."""
    b = chess.Board("3kr3/8/8/8/4R3/3B4/8/3RK3 w - - 0 1")
    moves, D, E = move_derivatives(b, exploitable=True)
    idx = [m.uci() for m in moves].index("d3b5")
    assert D[idx, chess.E4] < -0.5, "expected a large negative D_m(e4) from losing the bishop's defense"
    assert not E[idx, chess.E4], "Black has no legal reply reaching e4 (forced to respond to check)"

    out = ascent_cone(b)
    bb5_idx = out["moves"].index(chess.Move.from_uci("d3b5"))
    naive_damage = (np.minimum(D[idx], 0.0) * 1.0)[chess.E4]   # what the old (unfixed) formula would count
    assert naive_damage < -0.5
    # the actual (fixed) damage must be strictly greater (less negative) than
    # what the naive, exploitability-blind formula would have produced, since
    # at minimum the e4 contribution is now excluded.
    assert out["damage"][bb5_idx] > naive_damage + 0.4


def test_is_squeezed_when_cone_empty():
    # tau=0, king_zone target on a position where no move plausibly gains on the
    # enemy king zone without any cost -- just check the flag is internally
    # consistent with in_cone, not a specific chess claim.
    b = chess.Board()
    out = ascent_cone(b, cone_cfg=ConeConfig(tau=0.0))
    assert out["is_squeezed"] == (not out["in_cone"].any())
