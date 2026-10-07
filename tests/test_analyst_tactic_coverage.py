"""The coverage measurement's event definition (catspace/incubator/analyst/tactic_coverage.py) on constructed games whose
properties are asserted first: a knight fork wins a visibly safe rook (a tactic); a hung piece is not one; a piece that
moves away before being captured is followed."""
import chess

from catspace.incubator.analyst import tactic_coverage as C

V = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def _rows(fens_evals):
    """A game as census rows: [(fen, eval from the side to move)], material from the side to move's view; each FEN must
    follow the previous by one legal move (asserted)."""
    plies = {}
    for t, (fen, ev) in enumerate(fens_evals):
        b = chess.Board(fen); us = b.turn
        if t:
            prev = chess.Board(fens_evals[t - 1][0])
            assert any((prev.push(m), prev.board_fen() == b.board_fen(), prev.pop())[1] for m in prev.legal_moves), f"ply {t} does not follow"
        mat = sum(len(b.pieces(p, c)) * v * (1 if c == us else -1) for p, v in V.items() for c in (True, False))
        plies[t] = dict(fen=fen, eval=ev, material=mat, side="w" if us else "b", stm=[], opp=[], elo=1600, elo_opp=1600)
    return plies


def test_a_knight_fork_wins_a_visibly_safe_rook():
    # White Ng5, Black Kd8 and Rh6: 1.Nf7+ forks the king and the rook; the rook was not attacked before
    b = chess.Board("3k4/8/7r/6N1/8/8/8/6K1 w - - 1 2")
    assert chess.Move.from_uci("g5f7") in b.legal_moves and C.visibly_safe(b, chess.H6) == "safe"
    b1 = b.copy(); b1.push_san("Nf7+"); assert b1.is_check() and b1.attackers(chess.WHITE, chess.H6)
    seq = [("2k5/8/7r/6N1/8/8/8/6K1 b - - 0 1", 0), ("3k4/8/7r/6N1/8/8/8/6K1 w - - 1 2", 50),
           ("3k4/5N2/7r/8/8/8/8/6K1 b - - 2 2", -450), ("4k3/5N2/7r/8/8/8/8/6K1 w - - 3 3", 500), ("4k3/8/7N/8/8/8/8/6K1 b - - 0 3", -500)]
    ev = C.events_of(_rows(seq))
    tactics = [e for e in ev if e[3] != "hanging"]
    assert len(tactics) == 1
    t, sq, pt, kind, forced, net, cls = tactics[0]
    assert (t, sq, pt, kind, forced, net) == (1, chess.H6, chess.ROOK, "safe", True, 5)


def test_a_plain_hang_is_not_a_tactic():
    # Black plays ...Rf1+?? into the white king and rook; 2.Rxf1
    seq = [("4k3/8/8/8/8/8/5r2/4K2R b - - 0 1", 0), ("4k3/8/8/8/8/8/8/4Kr1R w - - 1 2", 500), ("4k3/8/8/8/8/8/8/4KR2 b - - 0 2", -500)]
    plies = _rows(seq)
    b = chess.Board(seq[1][0])
    assert b.is_check() and C.visibly_safe(b, chess.F1) == "hanging"
    ev = C.events_of(plies)
    assert ev and all(e[3] == "hanging" for e in ev)


def test_follow_tracks_a_piece_that_moves():
    b = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    moves = [chess.Move.from_uci(u) for u in ("a1a8", "e8d7", "a8a7", "d7c6", "a7b7", "c6b7")]
    assert C.follow(moves, b, chess.A1) == ("captured", 5)            # the rook, followed to b7, is taken there
    assert C.follow(moves[:5], b, chess.A1) == ("kept", None)
