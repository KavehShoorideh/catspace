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


def test_the_rust_mirror_agrees_with_the_kernel_on_real_positions():
    """Bridge: on real census positions, every quiet move that newly attacks something, at the claimed k, the Rust
    local_check equals the Lean kernel's localCheck."""
    import analyst_core
    from catspace.incubator.analyst.lean_ideas import detector
    from catspace.incubator.analyst.tactic_census_measure import load
    li = detector(); n = 0
    for r in load("c300")[:400:7]:
        b = chess.Board(r["fen"])
        for m in list(b.legal_moves)[:12]:
            for k in (1, 3):
                rust = analyst_core.local_check(b.fen(), m.uci(), k)
                assert rust == li.fork(b, m, k), (r["fen"], m.uci(), k)
                n += 1
    assert n > 500


def test_the_rust_rule_layer_matches_the_python_matcher_on_the_cases():
    import analyst_core
    cases = {("r3r2k/8/8/1N6/8/8/5PPP/5RK1 w - - 0 1", "b5c7"): ("fork", 2, True),
             ("r3q1k1/8/8/1N6/8/8/8/6K1 w - - 0 1", "b5c7"): ("fork", 2, False),
             ("q3k3/8/8/1N6/8/8/8/6K1 w - - 0 1", "b5c7"): ("fork with check", 9, True),
             ("6k1/2p2ppp/8/8/8/8/2R2PPP/6K1 w - - 0 1", "c2a2"): ("cornered king (threat)", 50, False)}
    for (fen, uci), want in cases.items():
        got = [(c[1], c[2], c[3]) for c in analyst_core.threat_claims(fen) if c[0] == uci]
        assert want in got, (fen, uci, got)


def test_the_cornering_frame_sees_a_trap_closed_by_a_quiet_move():
    """The black knight on a1 is attacked by the h1 rook; its escapes are b3 (covered by the a2 pawn) and c2. Ke2-d2
    covers c2 and attacks nothing new: the trap is closed by a quiet move -- a cornered piece, certified."""
    import analyst_core
    b = chess.Board("6k1/8/8/8/8/8/P3K3/n6R w - - 0 1")
    assert b.is_valid()                                               # (the first fixture left Black in check: shakmaty refused it)
    assert chess.H1 in b.attackers(chess.WHITE, chess.A1) and chess.A2 in b.attackers(chess.WHITE, chess.B3)
    rows = [c for c in analyst_core.threat_claims(b.fen()) if c[0] == "e2d2"]
    assert any(c[1] == "cornered piece" and c[2] == 3 and (c[3] or c[5]) for c in rows), rows


def test_a_mate_threat_by_check_is_named_a_mating_attack():
    """Re2-e8+ forces Rxe8, then Rxe8 mates: a certificate that holds by mating is a mating attack."""
    import analyst_core
    b = chess.Board("r5k1/5ppp/8/8/8/8/4RPPP/4R1K1 w - - 0 1")
    nb = b.copy(); nb.push_san("Re8+"); assert nb.is_check() and [nb.san(x) for x in nb.legal_moves] == ["Rxe8"]
    rows = [c for c in analyst_core.threat_claims(b.fen()) if c[0] == "e2e8"]
    assert any(c[1] == "cornered king" and c[5] for c in rows), rows       # mate is cornering the king (Kaveh)


def test_the_promotion_frame_uses_the_proved_passer():
    from catspace.incubator.analyst.threat_lemmas import passer_claim
    b = chess.Board("8/8/8/1P6/8/8/6k1/K7 w - - 0 1")                 # b5-b6: the black king is far; the pawn runs
    assert passer_claim(b, chess.Move.from_uci("b5b6"))
