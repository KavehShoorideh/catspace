"""The opening subgoals (catspace/incubator/analyst/opening_subgoals.py): each principle on positions whose properties are
asserted first, the lexicographic order, the opening-only gate, the move-history check, and the conjecture tier in
frame_player (it only orders the ledger's ties)."""
import chess
import pytest

from catspace.incubator.analyst import frame_player as FP
from catspace.incubator.analyst import opening_subgoals as O


@pytest.fixture()
def rules():
    old, old_ledger = list(FP.RULES), FP.LEDGER
    FP.RULES[:] = []
    FP.LEDGER = "one_ply"
    yield FP.RULES
    FP.RULES[:] = old
    FP.LEDGER = old_ledger


def _top(claims):
    hi = max(v for v, _ in claims.values())
    return {u for u, (v, _) in claims.items() if v == hi}


def test_start_position_develops_a_knight_toward_the_centre():
    b = chess.Board()
    claims = O.opening_claims(b)
    assert _top(claims) == {"g1f3", "b1c3"}                     # develop + central + steady: 13
    assert claims["g1h3"][0] == 9 and claims["e2e4"][0] == 3     # develop + steady; centre pawn + steady


def test_castling_outranks_development():
    b = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 6 5")
    m = chess.Move.from_uci("e1g1")
    assert m in b.legal_moves and b.is_castling(m)             # the position's own properties first
    assert chess.Move.from_uci("b1c3") in b.legal_moves
    claims = O.opening_claims(b)
    assert _top(claims) == {"e1g1"} and claims["e1g1"][0] == 17


def test_early_queen_king_walk_and_flank_pawn_are_not_claimed():
    b = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")   # 1.e4 e5
    for u in ("d1h5", "e1e2", "h2h3"):
        assert chess.Move.from_uci(u) in b.legal_moves
    assert sum(1 for sq in O.MINOR_HOME[chess.WHITE] if b.piece_at(sq)) == 4   # every minor piece at home
    claims = O.opening_claims(b)
    assert "d1h5" not in claims and "e1e2" not in claims and "h2h3" not in claims
    assert claims["g1f3"][0] == 13 and "g1f3" in _top(claims)


def test_black_develops_a_knight_toward_the_centre():
    b = chess.Board()
    b.push_san("e4")
    assert b.turn == chess.BLACK
    claims = O.opening_claims(b)
    assert _top(claims) == {"g8f6", "b8c6"}
    assert claims["e7e5"][0] == 3                               # it occupies e5: a centre pawn, steady


def test_only_in_the_opening():
    b = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/P3BPPP/R2QKB1R w KQ - 0 30")
    assert b.fullmove_number > O.OPENING_MOVES
    assert O.opening_claims(b) == {}


def test_a_piece_that_already_moved_is_not_steady_with_history_only():
    b = chess.Board()
    b.push_san("Nf3"); b.push_san("Nf6")
    assert chess.Move.from_uci("f3g5") in b.legal_moves
    claims = O.opening_claims(b)
    assert "f3g5" not in claims                                 # the knight moved before: a second move is not steady
    assert claims["b1c3"][0] == 13
    nb = chess.Board(b.fen())                                   # the same position without its history
    assert not nb.move_stack
    assert O.opening_claims(nb)["f3g5"][0] == 1                 # history unknown: not counted against it


def test_rook_move_before_castling_and_flank_capture():
    b = chess.Board("rnbqkbnr/1ppppppp/8/p7/1P6/8/P1PPPPPP/RNBQKBNR w KQkq - 0 2")   # 1.b4 a5
    assert b.piece_at(chess.B4) and b.piece_at(chess.A5)
    m = chess.Move.from_uci("b4a5")
    assert m in b.legal_moves and b.is_capture(m)
    assert O.steady(b, m)                                       # a flank pawn may capture
    assert not O.steady(b, chess.Move.from_uci("a2a3"))         # a flank pawn push is not steady
    b2 = chess.Board("rnbqkbnr/pppppppp/8/8/7P/8/PPPPPPP1/RNBQKBNR w KQkq - 0 2")
    r = chess.Move.from_uci("h1h3")
    assert r in b2.legal_moves and b2.king(chess.WHITE) == chess.E1
    assert not O.steady(b2, r)                                  # a rook move with the king still at home


def test_the_rule_only_orders_the_ledgers_ties(rules):
    from catspace.incubator.analyst.fates import clear_caches
    fens = [chess.STARTING_FEN,
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 6 5",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"]
    for fen in fens:
        clear_caches()
        FP.RULES[:] = []
        base = FP.choose(chess.Board(fen))
        clear_caches()
        FP.RULES[:] = [FP.RULE_BOOK["opening"]()]
        got = FP.choose(chess.Board(fen))
        assert got["best"] and set(got["best"]) <= set(base["best"])   # never a move outside the ledger's top
        top = max(base["values"].values())
        assert all(abs(got["values"][u] - base["values"][u]) <= 1e-3 + 1e-9 for u in base["values"])   # values move by a tie-break at most
        assert max(got["values"].values()) == top
    clear_caches()
    FP.RULES[:] = [FP.RULE_BOOK["opening"]()]
    assert set(FP.choose(chess.Board())["best"]) == {"g1f3", "b1c3"}


# --- the review of 2026-10-06: regressions ---

def test_a_developed_position_from_an_epd_is_not_the_opening():
    b = chess.Board("r4rk1/ppp2ppp/2n5/2bqp3/8/P2PB3/1PP1NPPP/R2Q1RK1 w - - 0 1")   # WAC.016: castled, developed, "move 1"
    assert b.fullmove_number == 1 and not b.move_stack
    assert O.undeveloped_count(b, chess.WHITE) == 0 and b.king(chess.WHITE) != chess.E1
    assert not O.in_opening(b) and O.opening_claims(b) == {}


def _play(sans):
    b = chess.Board()
    for s in sans.split():
        b.push_san(s)
    return b


def test_a_rook_that_castled_has_moved():
    b = _play("e4 e5 Nf3 Nc6 Bc4 Bc5 O-O Nf6")
    assert b.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)
    assert chess.F1 in O.moved_squares(b) and not O.steady(b, chess.Move.from_uci("f1e1"))
    b = _play("d4 d5 Nc3 Nc6 Bf4 Bf5 Qd2 Qd7 O-O-O O-O-O e3")
    assert b.piece_at(chess.D8) == chess.Piece(chess.ROOK, chess.BLACK) and b.turn == chess.BLACK
    assert chess.D8 in O.moved_squares(b) and not O.steady(b, chess.Move.from_uci("d8e8"))


def test_a_knight_on_a_bishops_square_and_a_piece_back_home_are_not_undeveloped():
    b = _play("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7 Nbd2 Bb7 Bc2 Re8 Nf1 Bf8")
    assert b.piece_at(chess.F1) == chess.Piece(chess.KNIGHT, chess.WHITE) and b.turn == chess.WHITE
    assert not O.develop(b, chess.Move.from_uci("f1e3"))       # a knight on f1 is not a bishop at home
    assert O.undeveloped_count(b, chess.WHITE) == 1            # only the bishop on c1
    assert O.in_opening(b) and O.steady(b, chess.Move.from_uci("d1e2"))   # one minor at home: the queen may move
    b = _play("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4")
    assert b.piece_at(chess.B8) == chess.Piece(chess.KNIGHT, chess.BLACK) and b.turn == chess.BLACK
    assert not O.develop(b, chess.Move.from_uci("b8d7"))       # back home is not undeveloped


def _moved_reference(b):
    """An independent reference: replay the game and follow the board's diffs (no castling or en passant cases): a
    square whose occupant after a move is the mover's piece and differs from before received a moving piece."""
    r = chess.Board(b.root().fen())
    moved = set()
    for m in b.move_stack:
        side = r.turn
        before = r.piece_map()
        r.push(m)
        after = r.piece_map()
        for sq in set(before) | set(after):
            if sq not in after:
                moved.discard(sq)                                # vacated (or a captured pawn en passant)
            elif after[sq].color == side and before.get(sq) != after[sq]:
                moved.add(sq)                                    # a piece of the mover arrived
            elif after[sq].color == side and sq in before and before[sq] == after[sq] and sq in (m.to_square,):
                moved.add(sq)
    return moved


def test_moved_squares_agrees_with_a_board_diff_replay():
    import random
    rng = random.Random(7)
    checked = 0
    for _ in range(150):
        b = chess.Board()
        for _ in range(rng.randint(10, 90)):
            moves = list(b.legal_moves)
            if not moves:
                break
            castles = [m for m in moves if b.is_castling(m)]
            b.push(rng.choice(castles) if castles and rng.random() < 0.7 else rng.choice(moves))
            assert set(O.moved_squares(b)) == _moved_reference(b), b.move_stack
            checked += 1
    assert checked > 3000


def test_rules_are_applied_in_the_canonical_order():
    names = [r.name for r in FP.rules_in_order(["opening", "fork_hint", "fork", "refute"])]
    assert names == ["fork", "refute", "fork_hint", "opening"]
    with pytest.raises(KeyError):
        FP.rules_in_order(["no_such_rule"])
