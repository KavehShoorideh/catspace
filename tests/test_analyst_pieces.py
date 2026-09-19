import chess

from catspace.incubator.analyst.pieces import track, fates, move_between, boards_from_fens, KEPT, EXCHANGED, LOST, PROMOTED, describe


def _game(sans):
    b = chess.Board(); moves = []
    for s in sans:
        m = b.parse_san(s); moves.append(m); b.push(m)
    return moves


def test_track_castling_en_passant_and_exchange():
    # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.O-O Nxe4 5.Re1 d5 6.Bxd5 Qxd5 : the knight takes a pawn (lost pawn), the bishop
    # takes a pawn and is taken by the queen (exchanged), the rook castles
    moves = _game(["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "Nxe4", "Re1", "d5", "Bxd5", "Qxd5"])
    ids = track(chess.Board(), moves)
    assert ids[7][chess.G1] == (chess.E1, chess.KING) and ids[7][chess.F1] == (chess.H1, chess.ROOK)     # after O-O
    assert ids[8][chess.E4] == (chess.G8, chess.KNIGHT)                                              # the knight on e4 is the g8 knight
    assert (chess.E2, chess.PAWN) not in ids[8].values()                                             # the e-pawn is gone
    f = fates(chess.Board(), moves, 0, 12)
    assert f[(chess.E2, chess.PAWN)].kind == LOST and f[(chess.E2, chess.PAWN)].other_type == chess.KNIGHT
    assert f[(chess.F1, chess.BISHOP)].kind == EXCHANGED and f[(chess.F1, chess.BISHOP)].other_type == chess.QUEEN
    assert f[(chess.D7, chess.PAWN)].kind == LOST
    assert f[(chess.H1, chess.ROOK)].kind == KEPT and f[(chess.H1, chess.ROOK)].square == chess.E1
    # a window that ends before the exchange: the bishop is kept
    f2 = fates(chess.Board(), moves, 0, 10)
    assert f2[(chess.F1, chess.BISHOP)].kind == KEPT and f2[(chess.F1, chess.BISHOP)].square == chess.C4


def test_en_passant_and_promotion():
    b = chess.Board("4k3/1p6/8/8/4p3/8/3P4/4K3 w - - 0 1")
    moves = []
    for s in ["d4", "exd3", "Kd2", "b5", "Kxd3", "b4", "Kc2", "b3+", "Kb1", "b2", "Ka2", "b1=Q+"]:
        m = b.parse_san(s); moves.append(m); b.push(m)
    start = chess.Board("4k3/1p6/8/8/4p3/8/3P4/4K3 w - - 0 1")
    ids = track(start, moves)
    assert (chess.D2, chess.PAWN) not in ids[2].values()                    # taken en passant
    assert ids[12][chess.B1] == (chess.B7, chess.PAWN)                      # the promoted queen keeps its id
    f = fates(start, moves, 0, 12)
    assert f[(chess.D2, chess.PAWN)].kind == LOST
    assert f[(chess.E4, chess.PAWN)].kind == EXCHANGED                      # took the d-pawn, was taken by the king
    assert f[(chess.B7, chess.PAWN)].kind == PROMOTED and f[(chess.B7, chess.PAWN)].other_type == chess.QUEEN


def test_move_between_and_describe():
    b0 = chess.Board(); b1 = chess.Board(); b1.push_san("Nf3")
    assert move_between(b0, b1) == chess.Move.from_uci("g1f3")
    boards, moves = boards_from_fens([b0.fen(), b1.fen()])
    assert moves == [chess.Move.from_uci("g1f3")]
    assert describe(chess.Board(), chess.C1, chess.WHITE) == "our dark-squared bishop on c1"
    assert describe(chess.Board(), chess.G8, chess.WHITE) == "their kingside knight on g8"
