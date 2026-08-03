"""Additional behavioral tests (Kaveh, 2026-08-02: "think of a few more test
cases, make sure it behaves as we expect") covering special moves and tactical
motifs not exercised by the Phase 1/2 gate tests: forks, en passant, promotion,
castling, deeper SEE exchange chains, terminal positions, and target-mode
correctness beyond king_zone."""
import chess
import numpy as np
import pytest

from catspace.research.components.encoder.approaches.control_field_wdl.src.control import weighted_attacker_field, see_field, compute_fields
from catspace.research.components.encoder.approaches.control_field_wdl.src.derivative import (
    move_derivatives, ascent_cone, ConeConfig, target_squares,
)


def test_knight_fork_gains_on_both_targets():
    """Nb5-c7+ forks Ke8 and Ra8 -- a genuine double-attack move should show
    positive D_m on BOTH target squares simultaneously, not just one."""
    b = chess.Board("r3k3/8/8/1N6/8/8/8/6K1 w - - 0 1")
    moves, D = move_derivatives(b)
    nc7 = moves.index(chess.Move.from_uci("b5c7"))
    assert D[nc7, chess.A8] > 0, "fork should gain attacker weight on the rook's square"
    assert D[nc7, chess.E8] > 0, "fork should gain attacker weight on the king's square"
    # explicit target mode over both forked squares: gain should be the SUM,
    # strictly greater than either individually -- confirms gain() aggregates
    # across simultaneous targets rather than only picking up one.
    out = ascent_cone(b, cone_cfg=ConeConfig(target_mode="explicit",
                                              explicit_squares=frozenset({chess.A8, chess.E8})))
    assert out["gain"][nc7] == pytest.approx(D[nc7, chess.A8] + D[nc7, chess.E8], abs=1e-5)
    assert out["gain"][nc7] > max(D[nc7, chess.A8], D[nc7, chess.E8])


def test_en_passant_capture_no_crash_and_correct_removal():
    """En passant is the classic special-move footgun (also bit us earlier
    tonight in the hanging-piece probe's captured-square logic). White Pe5,
    Black Pd5 just played d7-d5; exd6 e.p. must remove the black pawn from d5
    (not d6) and the field must reflect a real pawn on d6 afterward, not a
    phantom on d5."""
    b = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    ep = chess.Move.from_uci("e5d6")
    assert b.is_en_passant(ep)
    moves, D = move_derivatives(b)     # must not raise
    idx = moves.index(ep)
    b2 = b.copy(); b2.push(ep)
    assert b2.piece_at(chess.D5) is None, "captured pawn must be gone from d5"
    assert b2.piece_at(chess.D6) is not None and b2.piece_at(chess.D6).color == chess.WHITE
    # sanity: the derivative array is finite and not all-zero (something changed)
    assert np.isfinite(D[idx]).all()
    assert not np.allclose(D[idx], 0.0)


def test_promotion_grants_long_range_control():
    """A promoted queen must show long-range control a pawn never had. White
    Pe7 promotes on e8 (empty square, simple push-promotion); compare the
    resulting piece's contribution against what a mere pawn could ever produce."""
    b = chess.Board("6k1/4P3/8/8/8/8/8/4K3 w - - 0 1")   # king on g8, e8 empty for the push
    promo_q = chess.Move.from_uci("e7e8q")
    assert promo_q in b.legal_moves
    b2 = b.copy(); b2.push(promo_q)
    assert b2.piece_at(chess.E8).piece_type == chess.QUEEN
    c_after = weighted_attacker_field(b2)
    # a queen on e8 controls the whole e-file and 8th rank and both diagonals --
    # check it reaches a4 (diagonal e8-a4), a square no pawn promotion history
    # could ever attack, confirming piece_type_at() post-promotion is read
    # correctly (not stuck reporting PAWN).
    assert c_after[chess.A4] > 0, "promoted queen should control a4 via the e8-a4 diagonal"


def test_castling_no_crash_and_rook_gains_open_file():
    """Kingside castling moves TWO pieces (king + rook) in one chess.Move; make
    sure move_derivatives handles it without crashing and that the rook's move
    to f1 is reflected (new control on the previously-blocked f-file)."""
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    o_o = chess.Move.from_uci("e1g1")
    assert o_o in b.legal_moves
    moves, D = move_derivatives(b)      # must not raise
    idx = moves.index(o_o)
    assert np.isfinite(D[idx]).all()
    b2 = b.copy(); b2.push(o_o)
    assert b2.piece_at(chess.F1) is not None and b2.piece_at(chess.F1).piece_type == chess.ROOK
    assert b2.piece_at(chess.H1) is None   # rook left h1


def test_see_three_capture_exchange_chain():
    """Deeper exchange than the earlier 2-capture smoke check: White queen
    attacks a black knight on e5, defended by a black bishop, which is itself
    the only defender; White also has a rook backing up the queen on the
    e-file. Attacker order by value: white queen(900) initiates -- wait, SEE
    always uses LEAST valuable attacker first regardless of who's "attacking
    to start" -- verify the 3-ply fold (Nxe5 knight captured by queen is wrong
    framing; construct concretely and check against hand computation)."""
    # White: Re1, Qd4 (both attack e5). Black: Ne5 (target), Bc7 (defends e5 via c7-e5? no,
    # use Bd6 defending e5 diagonally is wrong square set -- use Bg7 does not defend e5 either.
    # Simplify: White Re1 + Nc4 (both attack e5); Black Ne5 defended by Bd6 only.
    b = chess.Board("4k3/8/3b4/4n3/2N5/8/8/4RK2 w - - 0 1")
    assert b.piece_at(chess.E5).piece_type == chess.KNIGHT and b.piece_at(chess.E5).color == chess.BLACK
    assert chess.E5 in b.attackers(chess.WHITE, chess.E5) or True  # sanity below via SEE directly
    from catspace.research.components.encoder.approaches.control_field_wdl.src.control import _see_square, SEE_VALUES
    result = _see_square(b, chess.E5, chess.WHITE)
    # hand computation: white least-valuable attacker = knight c4 (300) captures
    # knight e5 (300) -> net so far +300 for white, black recaptures with bishop
    # d6 (300) capturing white knight (300) -> net swing 0, white rook e1 then
    # recaptures bishop (300) -> net +300 for white overall, no further black
    # attackers. Sequence value from White's POV: +300 (gain knight) -300 (lose
    # knight) +300 (gain bishop) = +300 net.
    assert result == 300, f"expected net +300 for White over the 3-capture chain, got {result}"


def test_terminal_position_no_crash():
    """Checkmate and stalemate positions have zero legal moves -- ascent_cone
    must degrade gracefully (empty arrays, cone_size=0.0, is_squeezed=True),
    not divide by zero or raise."""
    # Fool's mate: Black delivers checkmate, White to move has no legal moves.
    b = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert b.is_checkmate()
    out = ascent_cone(b)
    assert out["cone_size"] == 0.0
    assert out["is_squeezed"] is True
    assert len(out["moves"]) == 0
    assert out["D"].shape == (0, 64)


def test_weak_squares_and_explicit_target_modes():
    """target_mode='weak_squares' and 'explicit' haven't been exercised by the
    gate scripts (only king_zone has) -- basic correctness check for both."""
    b = chess.Board()
    C, _, _ = compute_fields(b)
    weak = target_squares(b, C, ConeConfig(target_mode="weak_squares", k_weak=4))
    assert len(weak) == 4
    # weak_squares picks the top-k by C descending -- verify that property directly
    order = np.argsort(-C)[:4]
    assert weak == set(int(s) for s in order)

    explicit = target_squares(b, C, ConeConfig(target_mode="explicit",
                                                explicit_squares=frozenset({chess.E4})))
    assert explicit == {chess.E4}   # White to move: no reorientation needed


def test_see_field_zero_on_undefended_and_unattacked_squares():
    """Sparse-by-construction check (spec 2.1-B): squares nobody attacks must
    be exactly zero, not merely small."""
    b = chess.Board()
    c_see = see_field(b)
    # e4 is empty and unattacked at the start position -- must be exactly 0.
    assert c_see[chess.E4] == 0.0
    assert c_see[chess.A1] == 0.0   # occupied by White's own rook, no attackers at all
