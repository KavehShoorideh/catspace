import chess

from catspace.incubator.analyst.threat_lemmas import claims


def test_a_quiet_knight_fork_is_certified_by_the_kernel():
    """Nc7 forks two rooks that defend each other; no black check exists (the f1 rook seals the first rank); whatever
    Black plays, a rook falls for the knight: +2, certified by the kernel's localCheck."""
    b = chess.Board("r3r2k/8/8/1N6/8/8/5PPP/5RK1 w - - 0 1")
    m = chess.Move.from_uci("b5c7")
    assert b.is_legal(m) and not b.is_capture(m) and not b.gives_check(m)
    fork = [c for c in claims(b, m) if c.name == "fork"]
    assert fork and fork[0].k == 2 and fork[0].certified


def test_a_fork_that_the_defender_escapes_with_check_is_not_certified():
    """The same fork shape with the black queen on e8 and the first rank open: ...Qe1+ escapes -- matched, not certified."""
    b = chess.Board("r3q1k1/8/8/1N6/8/8/8/6K1 w - - 0 1")
    fork = [c for c in claims(b, chess.Move.from_uci("b5c7")) if c.name == "fork"]
    assert fork and not fork[0].certified


def test_a_royal_fork_is_certified():
    b = chess.Board("q3k3/8/8/1N6/8/8/8/6K1 w - - 0 1")
    cs = claims(b, chess.Move.from_uci("b5c7"))
    assert any(c.name == "fork with check" and c.certified and c.k == 9 for c in cs)


def test_a_single_parriable_threat_is_matched_but_not_certified():
    b = chess.Board("4k3/8/8/8/8/2n5/8/R3K3 w - - 0 1")
    cs = claims(b, chess.Move.from_uci("a1a3"))                  # Ra3 attacks the knight, which simply moves
    assert cs and not any(c.certified for c in cs)


def test_a_mate_threat_is_matched():
    """Rc2-a2 opens the a-file for Ra8#; Black has luft moves, so it is matched, not certified."""
    b = chess.Board("6k1/2p2ppp/8/8/8/8/2R2PPP/6K1 w - - 0 1")
    cs = claims(b, chess.Move.from_uci("c2a2"))
    assert any(c.name == "mate threat" for c in cs)
