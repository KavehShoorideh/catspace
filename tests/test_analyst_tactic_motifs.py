"""The structural motif labeller (catspace/incubator/analyst/tactic_motifs.py) on constructed positions whose tactical
properties are asserted first."""
import chess

from catspace.incubator.analyst import tactic_motifs as T


def _seq(b, sans):
    out = []; bb = b.copy()
    for s in sans:
        m = bb.parse_san(s); out.append(m); bb.push(m)
    return out


def test_knight_fork():
    b = chess.Board("3k4/8/7r/6N1/8/8/8/6K1 w - - 1 2")                 # Nf7+ forks king and rook
    seq = _seq(b, ["Nf7+", "Ke8", "Nxh6"])
    assert b.is_check() is False and seq[0].to_square == chess.F7
    assert "knight fork" in T.label(b, seq[0], chess.H6, seq)


def test_discovered_attack_and_check():
    b = chess.Board("4k3/8/8/8/8/8/3B4/3RK3 w - - 0 1")                  # the bishop on d2 masks the rook d1 against d8? build a real one:
    b = chess.Board("3q4/8/8/8/8/8/3N4/3RK1k1 w - - 0 1")                 # rook d1, knight d2 in front, black queen d8: a knight move discovers Rxd8
    assert b.piece_at(chess.D2).piece_type == chess.KNIGHT and T._attacks_through(b, chess.D1, chess.D8)
    seq = _seq(b, ["Nf3+", "Kh1", "Rxd8"])                                # Nf3 gives check to the king on g1? the black king is on g1: Nf3+ attacks g1
    assert chess.Board(b.fen()).is_attacked_by(chess.WHITE, chess.G1) is False
    b1 = b.copy(); b1.push(seq[0]); assert b1.is_check()
    labels = T.label(b, seq[0], chess.D8, seq)
    assert "discovered attack" in labels


def test_absolute_pin_wins_the_pinned_piece():
    b = chess.Board("4k3/8/8/8/8/8/4n3/4RK2 w - - 0 1")                   # Re1 pins the knight e2 to the king e8? the knight is on e2 between e1 and e8
    assert T._attacks_through(b, chess.E1, chess.E8)
    seq = _seq(b, ["Kxe2"])                                              # simplest: the pinned knight is taken (it could not move)
    labels = T.label(b, chess.Move.from_uci("f1e2"), chess.E2, seq)
    assert "absolute pin" in labels or "trapped piece" in labels


def test_capture_of_the_defender():
    # White Ka1, Bb3, Rh3; Black Ke8, Rf7, Nf3: the knight f3 is attacked by the rook h3 and defended by the rook f7;
    # 1.Bxf7+ Kxf7 2.Rxf3 wins it -- the defender was captured (the knight gives no check to a1)
    b = chess.Board("4k3/5r2/8/8/8/1B3n1R/8/K7 w - - 0 1")
    assert not b.is_check()
    assert chess.H3 in b.attackers(chess.WHITE, chess.F3) and b.attackers(chess.BLACK, chess.F3) == chess.SquareSet([chess.F7])
    assert chess.F7 in b.attacks(chess.B3) and T.VAL[chess.ROOK] > T.VAL[chess.KNIGHT]
    seq = _seq(b, ["Bxf7+", "Kxf7", "Rxf3"])
    assert "capture of the defender" in T.label(b, seq[0], chess.F3, seq)


def test_trapped_bishop():
    b = chess.Board("4k3/8/8/8/8/8/1P6/bK6 w - - 0 1")                     # the bishop a1 has no square but b2 (a pawn) -- trapped; Kxa1? the king is adjacent
    assert b.piece_at(chess.A1) == chess.Piece(chess.BISHOP, chess.BLACK)
    seq = _seq(b, ["Kxa1"])
    labels = T.label(b, chess.Move.from_uci("b1a1"), chess.A1, seq)
    assert labels == [] or "trapped piece" in labels                      # a king capture of an undefended piece is a hang, not a tactic
    b2 = chess.Board("4k3/8/8/8/8/1P6/P7/bK6 w - - 0 1")                   # Kb1-c2? no: trap the bishop with b3-b4? The bishop on a1 attacks b2: a2 pawn... use:
    b2 = chess.Board("4k3/8/8/8/8/8/PP6/b1K5 w - - 0 1")                   # Kc1 to move; the bishop a1 has b2 (defended by the king? c1 defends b2) -> after Kb1 it is lost
    seq2 = _seq(b2, ["Kb1"])
    assert "trapped piece" in T.label(b2, seq2[0], chess.A1, seq2)


def test_deflection_needs_the_defender_to_leave_the_square_s_defence():
    """Review 2026-10-07: the destination was tested against the defenders' squares before the move, so a rook that moved
    along its own defending file was 'deflected'."""
    from catspace.incubator.analyst.tactic_motifs import label
    b = chess.Board("4k2r/8/8/7n/8/8/6PP/6K1 w - - 0 1")
    assert chess.H8 in b.attackers(chess.BLACK, chess.H5)
    stays = [chess.Move.from_uci(u) for u in ("g2g4", "h8h7", "g4h5", "h7h5")]
    assert "deflection / overload" not in label(b, stays[0], chess.H5, stays)
    leaves = [chess.Move.from_uci(u) for u in ("g2g4", "h8a8", "g4h5")]
    assert "deflection / overload" in label(b, leaves[0], chess.H5, leaves)


def test_trapped_piece_is_not_read_when_the_move_takes_it_or_gives_check():
    from catspace.incubator.analyst.tactic_motifs import label
    b = chess.Board("6k1/1r4BR/4p1p1/5p2/2pP4/3r2P1/5PK1/7R b - - 0 1")      # Rb7xg7 takes the bishop outright
    m = chess.Move.from_uci("b7g7")
    assert b.is_capture(m) and b.piece_at(chess.G7).piece_type == chess.BISHOP
    assert "trapped piece" not in label(b, m, chess.G7, [m])
    b2 = chess.Board("r2k4/8/8/3N4/8/8/8/3R2K1 w - - 0 1")                   # Nc7+ discovers the rook's check; the rook a8 has squares
    m2 = chess.Move.from_uci("d5c7")
    b3 = b2.copy(); b3.push(m2); assert b3.is_check() and chess.A8 in b3.attacks(chess.C7)
    labs = label(b2, m2, chess.A8, [m2])
    assert "discovered check" in labs and "trapped piece" not in labs


def test_promotion_label_is_a_s_own_pawn():
    from catspace.incubator.analyst.tactic_motifs import label
    b = chess.Board("3k2r1/8/4p3/8/2B5/8/8/7K w - - 0 1")                    # a bishop's trip through a pawn's square is no promotion
    seq = [chess.Move.from_uci(u) for u in ("c4e6", "d8c7", "e6f7", "c7b6", "f7g8")]
    assert "promotion / breakthrough" not in label(b, seq[0], chess.G8, seq)
    b2 = chess.Board("4k3/8/8/8/8/8/1p2K3/R6r w - - 0 1")                    # the VICTIM promotes: not A's motif
    seq2 = [chess.Move.from_uci(u) for u in ("a1h1", "b2b1q", "h1b1")]
    assert "promotion / breakthrough" not in label(b2, seq2[0], chess.H1, seq2)
    b3 = chess.Board("4k3/P7/8/8/8/8/8/K6r w - - 0 1")
    seq3 = [chess.Move.from_uci("a7a8q")]
    assert "promotion / breakthrough" in label(b3, seq3[0], chess.H1, seq3)
