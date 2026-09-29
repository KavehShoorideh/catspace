"""Frames read off exchanges (Kaveh 2026-09-27): the exchange frame's outcome, the pass at each ply, the ledger across
frames, the square conditions, and the exchange labeler."""
import chess

from catspace.incubator.analyst import chess_domain as cd
from catspace.incubator.analyst import fates as F

D4 = "r1bq1rk1/pp3p1p/2n1pbp1/2pp4/2B1P3/2PPB3/PP1Q1PPP/RN2K2R w KQ - 0 11"   # the error that started it: d4 is attacked three times and defended three times


def test_the_exchange_frame_counts_the_defenders():
    b = chess.Board(D4); m = chess.Move.from_uci("d3d4")
    full = lambda bb, s: F.direct_frame(bb, s)
    assert F.exchange_move_value(b, m, full)[0] == 0.0                                   # defended: the exchange does not start
    small = lambda bb, s: {s, chess.C5}                                                   # the defenders left out
    assert F.exchange_move_value(b, m, small)[0] == -1.0


def test_known_cases_and_the_ledger():
    cases = [("8/2k1r2p/1p1R2p1/p1pKP3/3p4/6B1/1P4PP/8 w - - 8 33", "d6d8", -5.0),        # the rook walks into the king, undefended
             ("4k3/8/8/4p3/3Q4/8/8/4K3 w - - 0 1", "d4a1", 9.0),                           # the queen leaves the pawn's attack: worth what standing still loses
             ("4k3/8/4p3/8/8/8/8/3QK3 w - - 0 1", "d1d5", -9.0)]                           # Qd5?? hangs to e6
    for fen, u, want in cases:
        b = chess.Board(fen); m = chess.Move.from_uci(u)
        assert F.exchange_move_value(b, m, F.direct_frame)[0] == want, u
        assert F.ledger_move_value(b, m, F.direct_frame)[0] == want, u
    b = chess.Board("r1bQr1k1/pp3pbp/2p2np1/4n3/4N3/2P2NP1/PP3PBP/R1B1R1K1 b - - 0 12")    # the white queen is en prise on d8
    assert F.ledger_move_value(b, chess.Move.from_uci("e8d8"), F.direct_frame)[0] >= 9.0   # the queen, and more: standing still allowed Nxf6+ Bxf6 Qxe8+ (forcing lines, 2026-09-28)
    from catspace.incubator.analyst.frame_player import choose
    assert "e8d8" in choose(b)["best"]


def test_the_pass_at_each_ply_has_a_stake():
    b = chess.Board("r2qk3/8/8/3b4/8/8/3R4/3QK3 w - - 0 1")                              # the rook takes a bishop the queen defends, our queen behind
    v = F.exchange_variations(b, chess.D5, F.direct_frame(b, chess.D5) | {chess.D1}, chess.WHITE)
    assert v[0]["mover"] == "us" and v[0]["stake"] == 3.0                                   # leaving the frame at once costs us the bishop
    assert v[1]["mover"] == "them" and v[1]["stake"] == 0.0                                 # taking back would cost them: they stop


def test_square_conditions():
    b = chess.Board(D4); b.push(chess.Move.from_uci("d3d4")); st = cd.State(b, [chess.D4])
    assert not cd.attacked_by_lower(st, chess.D4) and not cd.outnumbered(st, chess.D4)
    b2 = chess.Board("4k3/8/4p3/3Q4/8/8/8/4K3 w - - 0 1"); st2 = cd.State(b2, [chess.D5])
    assert cd.attacked_by_lower(st2, chess.D5) and cd.outnumbered(st2, chess.D5) and cd.last_attacker(st2, chess.E6, chess.D5)


def test_exchanges_are_chains_on_one_square():
    from catspace.incubator.analyst.exchanges import exchanges
    b = chess.Board(); moves = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"]
    meta = []
    for ply, u in enumerate(moves):
        meta.append(dict(game=7, ply=ply, fen=b.fen(), played=u)); b.push(chess.Move.from_uci(u))
    ex = exchanges(meta)
    assert len(ex) == 1 and ex[0].squares == [chess.D5] and len(ex[0].captures) == 2
    assert {"d5", "e4", "d8"} <= set(ex[0].pieces)                                        # the pawn taken, the pawn that took, the queen that took back


def test_generation_computes_relations_the_core_does_not_know():
    """The compiled core answers False for a relation it does not implement; the generator must compute those in
    Python, not cache an all-false column (the square conditions of 2026-09-27)."""
    from catspace.incubator.analyst import lemma_gen as lg
    dom = cd.domain(oracle="frame", depth=0)
    if dom.batch_literals is None:
        return                                                                               # no compiled core here: nothing to guard
    g = lg.Generator(dom, cap=60, log=lambda *a, **k: None, sequential=False)
    rows, _ = g.rows_of(lg.Frame(("wQ", "bP"), (("attacks", (1, 0), True),)), 0)
    assert rows
    cache = {}
    for name in ("outnumbered", "attacked_by_lower"):
        want = sum(1 << i for i, (st, ids, a, _) in enumerate(rows) if g.rel[name].fn(st, ids[0]))
        assert g.lit_mask(rows, name, (0,), cache) == want, name
    assert any(g.lit_mask(rows, "attacked_by_lower", (0,), cache) for _ in [0])                # the queen attacked by a pawn: some rows true


def test_the_frames_see_mates_and_forcing_lines():
    """Kaveh 2026-09-28, from the frame player's worst misses: a side that can neither act nor pass has lost, and the
    moves that take the pass away join the frame's line. Each old choice allowed a mate; the frames now refuse it."""
    from catspace.incubator.analyst.frame_player import choose
    cases = [("2r1r1k1/pp3pp1/q7/8/3Qbb2/4N2P/PPP1RP2/2K3R1 b - - 7 22", "Qxe2"),     # allowed Qxg7#
             ("5r1k/p1pp3p/1pP1p3/6p1/P3P2q/2N5/1PPB2KP/R2Q4 w - - 1 21", "cxd7"),     # allowed Rf2+ Kh1 Qxh2#
             ("8/5pk1/p6p/5B2/2np1p2/8/PP2rPPP/3R2K1 w - - 0 29", "Rxd4")]              # allowed Re1#
    for fen, old in cases:
        b = chess.Board(fen); c = choose(b); u = b.parse_san(old).uci()
        assert u not in c["best"], old
        for pick in c["best"]:                                                        # whatever the frames choose, it allows no mate in one
            nb = b.copy(); nb.push(chess.Move.from_uci(pick))
            assert not any((lambda x: (x.push(r), x.is_checkmate())[1])(nb.copy()) for r in nb.legal_moves), (old, pick)


def test_a_square_frame_holds_the_pieces_behind_its_pieces():
    """Kaveh 2026-09-28, the Rxg3 miss: the rook behind the recapturing rook belongs to the frame (it attacks the square
    once the frame's pieces are lifted), a rook behind a pawn that never captures there does not."""
    from catspace.incubator.analyst.frame_player import choose
    b = chess.Board("6r1/p1k2pr1/3p1p1b/1b1PpP2/2p1P3/PpPn1RpP/1P2N1B1/1K1N2R1 w - - 3 30")
    nb = b.copy(); nb.push(chess.Move.from_uci("f3g3"))
    assert chess.G8 in F.square_frame(nb, chess.G3) and chess.G8 not in F.direct_frame(nb, chess.G3)
    c = choose(b)
    assert "f3g3" not in c["best"] and c["values"]["f3g3"] < 0
    b2 = chess.Board("4k3/8/8/8/8/5n2/5P2/5RK1 w - - 0 1")                                  # Rf1 behind the f2 pawn: the pawn never takes on f3
    assert chess.F1 not in F.square_frame(b2, chess.F3)


def test_joint_frames_find_the_intermezzo_and_keep_recaptures():
    """Kaveh 2026-09-28: frames resolved jointly with the pieces they pull in. After Bxd5 exd5 the b1 rook's Rb6+ (a
    line the e6 pawn opens) makes the capture lose, so the frames choose Bxd5 (Stockfish's move); and a capture opens
    its square to every piece bearing on it (the d7 pawn the d8 queen defends is not 'won' by Qxd7)."""
    from catspace.incubator.analyst.frame_player import choose
    b = chess.Board("1n3r2/1Q2qp2/p3pkp1/3p3p/3P4/2P3P1/B1P2PKP/1R6 w - - 0 24")
    assert choose(b)["best"] == ["a2d5"]
    after = b.copy(); after.push_san("Bxd5")
    fr = F.square_frame(after, chess.D5)
    assert F.local_resolve(after, {chess.D5}, set(fr))[0] == 2.0                           # the d5 frame alone: exd5 Qxd5
    squares, kings, members = F.joint_frame(after, {chess.D5}, set())
    assert chess.WHITE in kings                                                            # lifting e6 gives the b1 rook a check: White's king frame joins
    assert F.local_resolve(after, squares, set(members), kings)[0] == 0.0                  # with what it pulls in: taking does not pay
    b2 = chess.Board("r2q1rk1/1bppnp1p/1pn1p1p1/p3P3/P3P3/5N2/BPP1NPPP/R2Q1RK1 b - - 3 13")
    stand = F.passed(b2)
    assert F.joint_threats(stand, F.square_frame)[0] == 0.0                                 # White has nothing: Qxd7 is met by Qxd7


def test_a_line_a_frame_opens_brings_in_what_it_now_attacks():
    """Kaveh 2026-09-28, the g6 miss: after g6, exd6 opens the d6 frame's own queen onto h8 -- the rook joins the frame
    (its square, with the pieces bearing on it), so Black cannot both take back on d6 and save the rook."""
    from catspace.incubator.analyst.frame_player import choose
    b = chess.Board("r1bqkbnr/ppp2ppp/3p4/4P3/3Q4/8/PPP2PPP/RNB1KB1R b KQkq - 0 6")
    after = b.copy(); after.push_san("g6")
    assert chess.H8 in F.interactors(after, F.square_frame(after, chess.D6))[0]            # h8's square joins, not a recursive closure
    c = choose(b)
    assert c["values"][b.parse_san("g6").uci()] < 0 and b.parse_san("g6").uci() not in c["best"]


def test_frames_are_squares_and_kings_live_in_king_frames():
    """Kaveh 2026-09-28: a frame is its squares -- a reply is on the same square, anything elsewhere is an intermezzo --
    and kings live in king frames. The holes the restructure opened, each by its example: a check's other targets
    (Nc7+ forks the a8 rook); the square a check lands on (f4? Ng4+ Rxg4 fxg4 wins the exchange); a king hunt of three
    checks (h3? Rf2+ Kg1 Qg3+ Kh1 Rh2#); a line a check opens onto a piece that is a frame piece but not on a frame
    square (Rg2? d4+ Kf2 Bxg2)."""
    from catspace.incubator.analyst.frame_player import choose
    b = chess.Board("r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1")
    assert choose(b)["best"] == [b.parse_san("Nc7+").uci()]
    for fen, san in [("r6r/pbp2kp1/1p2pnnp/5p2/3P3P/2PB1P2/PP1NNK1R/6R1 w - - 2 20", "f4"), ("5r1k/p1pp3p/1pP1p3/6p1/P3P2q/2N5/1PPB2KP/R2Q4 w - - 1 21", "h3"),
                     ("2r2r2/1b1k1p2/4p3/pp1p3P/8/P1P1K3/PB6/6RR w - - 2 24", "Rg2")]:
        b = chess.Board(fen); c = choose(b); u = b.parse_san(san).uci()
        assert u not in c["best"] and c["values"][u] < max(c["values"].values()), (fen, san)
    sq, members = F.king_frame(chess.Board("r3k3/8/8/3N4/8/8/8/4K3 b - - 0 1"), chess.BLACK)
    assert chess.A8 in sq and chess.C7 in sq                                               # the fork target and the check's square


def test_the_persistent_frame_store_serves_only_what_it_would_recompute():
    """Kaveh 2026-09-28 ("persistent frames from move to move ... like the NNUE"): a frame's resolution is stored under
    what it reads (fates.frame_key) and reused wherever the key recurs; in verify mode every hit is recomputed and must
    be equal, over every legal move of several positions."""
    from catspace.incubator.analyst.frame_player import choose
    old = F.VERIFY_FRAMES
    F.VERIFY_FRAMES = True
    try:
        before = dict(F.RESOLVED_STATS)
        for fen in ("1n3r2/1Q2qp2/p3pkp1/3p3p/3P4/2P3P1/B1P2PKP/1R6 w - - 0 24", "r2q1rk1/1bppnp1p/1pn1p1p1/p3P3/P3P3/5N2/BPP1NPPP/R2Q1RK1 b - - 3 13",
                    "7R/3b2r1/1p1kp3/p2p4/3PpN2/PP2P3/1KP2P2/8 b - - 0 29"):
            choose(chess.Board(fen))
        assert F.RESOLVED_STATS["hits"] > before["hits"]                                   # the store was used, and every hit matched
        assert F.RESOLVED_STATS.get("lemma_mismatch", 0) == before.get("lemma_mismatch", 0)  # a lemma (the same relations on other squares) never served a different value
        assert F.RESOLVED_STATS.get("lemma_hits", 0) > before.get("lemma_hits", 0)
    finally:
        F.VERIFY_FRAMES = old


def test_the_fast_resolver_equals_the_reference():
    """2026-09-28 'make frames faster': the in-place resolver (transposition table, check screen, captures onto the
    frame's squares) and the bitboard interactors must give exactly what the reference code gives."""
    positions = ["7R/3b2r1/1p1kp3/p2p4/3PpN2/PP2P3/1KP2P2/8 b - - 0 29", "r4r2/1pq3kp/p2p2p1/4p3/4bPn1/1Q2PN2/PP1KB1P1/R6R w - - 0 21",
                 "r5k1/p1p2p2/2p4p/3b1Nq1/3p1Q2/1PbPp1P1/P1P1P2P/1R4K1 w - - 1 25", "1n3r2/1Q2qp2/p3pkp1/3p3p/3P4/2P3P1/B1P2PKP/1R6 w - - 0 24"]
    old_gate = F.KING_JOIN_GATE
    F.KING_JOIN_GATE = None
    try:
        for fen in positions:
            b = chess.Board(fen)
            for m in list(b.legal_moves)[:8]:
                nb = b.copy(); nb.push(m)
                if nb.is_game_over():
                    continue
                for fr, _, v, ln in F.frame_book(nb, F.square_frame).values():
                    assert F.local_resolve(nb, fr[0], set(fr[2]), fr[1])[0] == F._local_resolve_reference(nb, fr[0], set(fr[2]), fr[1])[0], (nb.fen(), fr)
                for sq, pc in nb.piece_map().items():
                    if pc.color != nb.turn and pc.piece_type != chess.KING and nb.attackers(nb.turn, sq):
                        mem = F.square_frame(nb, sq)
                        assert F.interactors(nb, mem, {sq}) == F._interactors_reference(nb, mem, {sq})
    finally:
        F.KING_JOIN_GATE = old_gate


def test_a_frame_whose_line_gives_check_is_stored_by_the_whole_board():
    # verify mode 2026-09-29: the d5 frames of these two boards shared a local key; cxd5+ is answered by Kc5 in one and by
    # Kxc5 (a pawn outside the frame) in the other, so the stored value was served to the wrong board
    import catspace.incubator.analyst.fates as F
    a = chess.Board("8/8/2p5/2pP3p/2K1k3/7P/6P1/8 b - - 0 45")
    b = chess.Board("8/8/2p5/3P3p/2K1k3/7P/6P1/8 b - - 0 45")
    for first, second in ((a, b), (b, a)):
        F.clear_caches(keep_lemmas=False)
        v1 = F.resolve_frame(first, {chess.D5}, set(), F.square_frame(first, chess.D5))[0]
        v2 = F.resolve_frame(second, {chess.D5}, set(), F.square_frame(second, chess.D5))[0]
        assert {first.fen(): v1, second.fen(): v2} == {a.fen(): 0.0, b.fen(): 1.0}
    quiet = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    assert F.frame_key(quiet, {chess.E5}, set(), F.square_frame(quiet, chess.E5))[0] == "local"


def test_a_promotion_push_is_answered_on_its_square():
    # verify mode 2026-09-29: after Rgxe8 the frame let d7-d8=Q stand (+8) because d8 was not a frame square; the reply to
    # a promotion is on its square (Kaveh), so Rxd8 answers it and Black's Rxe8 nets +1 (Rgxe8 dxe8=Q Raxe8)
    import catspace.incubator.analyst.fates as F
    b = chess.Board("r3R1r1/3P3R/p7/2b1N1pp/P3P1pk/1P6/2P4P/7K b - - 1 38")
    F.clear_caches(keep_lemmas=False)
    assert F.resolve_frame(b, {chess.E8}, set(), F.square_frame(b, chess.E8))[0] == 1.0
    assert F.frame_key(b, {chess.E8}, set(), F.square_frame(b, chess.E8))[0] == "board"


def test_a_capturer_behind_a_member_is_in_the_key():
    # verify mode 2026-09-29: after Bxh1 the e-file opens and the rook behind the bishop takes on e3; the rook on e8 and
    # a rook on e7 gave the {h1, e3} frame the same local key and different resolutions
    import catspace.incubator.analyst.fates as F
    b = chess.Board("4rk2/pp3q2/2p2b2/6P1/4b3/4P3/PP3Q2/K3R2R b - - 2 38")
    c = chess.Board("5k2/pp2rq2/2p2b2/6P1/4b3/4P3/PP3Q2/K3R2R b - - 2 38")
    sq = {chess.H1, chess.E3}
    members = F.square_frame(b, chess.H1) | F.square_frame(b, chess.E3)
    assert chess.E8 in chess.SquareSet(F.potential_capturers(b, sq, members))
    assert F.frame_key(b, sq, set(), members) != F.frame_key(c, sq, set(), members)
