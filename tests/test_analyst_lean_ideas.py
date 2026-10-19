"""The Lean detector (lean/ideas): the back-rank lemma's verdicts on hand-made positions, every proved mate confirmed by
python-chess, and the kernel's legal moves equal to python-chess's (the bridge). Skipped when the binary is not built
(cd lean/ideas && lake build)."""
import chess
import pytest

from catspace.incubator.analyst.lean_ideas import LeanIdeas, default_binary, proved_back_rank_mates, rank_check_possible

pytestmark = pytest.mark.skipif(not default_binary().exists(), reason="lean/ideas not built (cd lean/ideas && lake build)")

# (fen, move, status, failing hypothesis, target squares)
CASES = [
    ("6k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1", "d1d8", "MATE", None, []),
    ("3r2k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", "d8d1", "MATE", None, []),                       # Black mates White
    ("6k1/5pp1/7p/8/8/8/8/3R2K1 w - - 0 1", "d1d8", "ONEAWAY", "boxed", ["h7"]),            # luft on h7
    ("6k1/5ppp/8/8/8/8/3N4/3R2K1 w - - 0 1", "d1d8", "ONEAWAY", "legal", ["d2"]),           # our knight in the path
    ("6k1/3n1ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8", "ONEAWAY", "noInterpose", ["d7"]),      # Nb8 or Nf8 blocks
    ("r5k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1", "d1d8", "ONEAWAY", "noDefender", ["a8"]),       # Rxd8
]


@pytest.fixture(scope="module")
def li():
    with LeanIdeas() as d:
        yield d


@pytest.mark.parametrize("fen,uci,status,failing,targets", CASES)
def test_backrank_verdicts(li, fen, uci, status, failing, targets):
    b = chess.Board(fen)
    insts = {i.move.uci(): i for i in li.backrank(b)}
    inst = insts[uci]
    assert inst.status == status
    assert inst.failing == failing
    assert sorted(chess.square_name(s) for s in inst.targets) == sorted(targets)
    if status == "MATE":
        nb = b.copy()
        nb.push(inst.move)
        assert nb.is_checkmate()
        assert rank_check_possible(b)
        assert inst.move in proved_back_rank_mates(b)
    else:
        assert inst.move not in proved_back_rank_mates(b)


@pytest.mark.parametrize("fen", [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",               # Kiwipete: castling, pins
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",                       # en passant
    "8/P5k1/8/8/8/8/6Kp/1r6 w - - 0 1",                                                   # promotions
])
def test_bridge_legal_moves(li, fen):
    b = chess.Board(fen)
    assert li.legal(b) == {m.uci() for m in b.legal_moves}


def test_checkmate_bridge(li):
    b = chess.Board("3R2k1/5ppp/8/8/8/8/8/6K1 b - - 1 1")
    assert li.checkmate(b) and b.is_checkmate()
    b = chess.Board("3R2k1/5pp1/7p/8/8/8/8/6K1 b - - 1 1")
    assert not li.checkmate(b) and not b.is_checkmate()


# (fen, pawn, status, failing, plies, targets)
PASSER_CASES = [
    ("8/8/8/8/8/4P3/8/K6k w - - 0 1", "e3", "PROVED", None, 9, []),                 # bare king far away
    ("8/8/8/8/k7/4P3/8/K7 w - - 0 1", "e3", "ONEAWAY", "kingOutside", None, ["a4"]),   # the king is in the square
    ("8/8/8/4N3/8/4P3/8/K6k w - - 0 1", "e3", "ONEAWAY", "pathClear", None, ["e5"]),  # our own knight blocks the path
    ("R7/8/8/8/8/4P3/8/K6k w - - 0 1", "e3", "ONEAWAY", "noSliders", None, []),       # our rook: this form has no sliders
    ("8/8/8/4b3/8/4P3/8/K6k w - - 0 1", "e3", "FAIL", None, None, []),                 # an enemy bishop blocks: two hypotheses fail
    ("8/8/8/8/8/4P3/8/K5Nk w - - 0 1", "e3", "PROVED", None, 9, []),                 # our knight is allowed (no check from g1)
]


@pytest.mark.parametrize("fen,sq,status,failing,plies,targets", PASSER_CASES)
def test_passer_verdicts(li, fen, sq, status, failing, plies, targets):
    b = chess.Board(fen)
    insts = {chess.square_name(i.square): i for i in li.passer(b)}
    inst = insts[sq]
    assert inst.status == status
    assert inst.failing == failing
    assert inst.plies == plies
    assert sorted(chess.square_name(s) for s in inst.targets) == sorted(targets)


def test_passer_proved_is_true(li):
    """A PROVED claim holds in an exhaustive race search."""
    from catspace.incubator.analyst.lean_passer_check import promotes_within
    b = chess.Board("8/8/8/8/8/4P3/8/K6k w - - 0 1")
    inst = {chess.square_name(i.square): i for i in li.passer(b)}["e3"]
    assert inst.status == "PROVED"
    assert promotes_within(b, chess.square_file(inst.square), inst.plies)
    assert not promotes_within(b, chess.square_file(inst.square), inst.plies - 2)


# the second form: enemy king and pawns (PROVED by either form).  (fen, pawn, status, failing, plies)
PASSER2_CASES = [
    ("8/8/8/8/8/4P3/8/K6k w - - 0 1", "e3", "PROVED", None, 9),                     # bare king: by the first form
    ("8/8/8/7P/5P1p/3k3K/8/8 w - - 0 1", "h5", "PROVED", None, 5),                   # an enemy pawn present, far
    ("K7/8/8/p2P4/P7/5p2/5P2/6k1 w - - 0 1", "d5", "ONEAWAY", "pawnsFar", None),     # an enemy pawn can interfere
    ("1K6/8/4P2k/2p5/3P4/8/8/8 w - - 0 1", "d4", "ONEAWAY", "kingOutside", None),    # the king is in the square
    ("8/8/3Pp1P1/2K1p2k/8/8/8/8 w - - 0 1", "g6", "PROVED", None, 3),               # its pawns cannot reach the king's corner in time (2026-10-06)
    ("7k/4N1pp/P7/8/8/8/8/1K6 w - - 0 1", "a6", "ONEAWAY", "kingRoom", None),        # the king's own pawns on its last free squares
    ("8/8/8/4b3/8/4P3/8/K6k w - - 0 1", "e3", "FAIL", None, None),                   # an enemy bishop on our king's colour, on the path's diagonal: two hypotheses fail
    # plan 52.16: the bishop as a stopper that does not stop (Ideas.Passer2.bishopsFarB, bdist); 0Nv29 after b6 Bb4
    ("8/8/1P6/2P1k3/1b2P3/3K4/8/8 w - - 1 56", "b6", "PROVED", None, 3),             # the B on b4 (the other colour from Kd3) needs 2 moves to land on b8, the pawn 2 pushes
    ("8/8/1P6/2P1k3/8/3K4/8/b7 w - - 1 56", "b6", "PROVED", None, 3),                # the B on a1 is two moves from b8
    ("8/1P6/8/3b4/8/3K4/8/7k w - - 0 1", "b7", "ONEAWAY", "bishopsFar", None),       # the B on d5 stands on our king's colour: it could check or pin
    ("8/8/1P6/8/8/3K4/2b5/7k w - - 0 1", "b6", "FAIL", None, None),                  # the B on c2 is on the path's diagonal to b3? no: it stands on our king's colour and is one move from b1-h7/a4-e8 lines
]


@pytest.mark.parametrize("fen,sq,status,failing,plies", PASSER2_CASES)
def test_passer2_verdicts(li, fen, sq, status, failing, plies):
    b = chess.Board(fen)
    inst = {chess.square_name(i.square): i for i in li.passer2(b)}[sq]
    assert inst.status == status
    assert inst.failing == failing
    assert inst.plies == plies
    if status == "PROVED":
        from catspace.incubator.analyst.lean_passer_check import promotes_within
        assert promotes_within(b, chess.square_file(inst.square), inst.plies)


def test_passer_setup_certificate():
    """setsUpPasser2 (Ideas/Setup.lean, the composition over one reply): after the move every legal reply leaves the
    same pawn a proved passer of the second form. The position's properties are asserted first, by the exhaustive
    race search and the direct detector (constructed positions were wrong four times on 2026-10-06)."""
    import pytest
    from catspace.incubator.analyst.lean_ideas import detector
    from catspace.incubator.analyst.lean_passer_check import promotes_within
    li = detector()
    b = chess.Board("1K6/3P2P1/8/k4P2/8/8/3PP3/8 w - - 0 1")
    m = chess.Move.from_uci("b8c8")
    assert m in b.legal_moves
    q = b.copy(stack=False); q.push(m)
    replies = list(q.legal_moves)
    assert replies
    for r in replies:                                   # after every reply the d-pawn queens on the next move
        q2 = q.copy(stack=False); q2.push(r)
        assert promotes_within(q2, chess.square_file(chess.D7), 1)
        assert {i.square: i.status for i in li.passer2(q2)}[chess.D7] == "PROVED"
    got = li.passer_setup(b, m)
    assert got[chess.D7] == 2                             # the reply and the push
    # a capturing reply refutes a set-up: the kernel refuses
    b2 = chess.Board("8/8/8/8/3Pk3/8/8/1K6 w - - 0 1")
    k = chess.Move.from_uci("b1c1")
    assert k in b2.legal_moves
    q = b2.copy(stack=False); q.push(k)
    assert chess.Move.from_uci("e4d4") in q.legal_moves  # Kxd4
    assert li.passer_setup(b2, k) == {}
    with pytest.raises(ValueError):
        li.passer_setup(b, chess.Move.from_uci("b8b6"))   # not a legal move


def test_exchange_law_hypotheses_and_certificate():
    """Ideas/ExchangeLaw.lean through the binary: an honest frame (a knight takes an undefended knight; no check, no
    arrival possible) has every hypothesis and the kernel's certificate SETTLED; a pinned defender is named."""
    from catspace.incubator.analyst.lean_ideas import detector
    li = detector()
    b = chess.Board("4k3/8/8/3n4/8/4N3/8/K7 w - - 0 1")
    assert chess.E3 in b.attackers(chess.WHITE, chess.D5) and not b.attackers(chess.BLACK, chess.D5)
    hyps, value, honest = li.exchange(b, chess.D5)
    assert honest and all(hyps.values()) and value == 3
    assert li.exchange_cert(b, chess.D5)
    b2 = chess.Board("4k3/8/4b3/3p4/8/2N5/8/K3R3 w - - 0 1")
    assert b2.is_pinned(chess.BLACK, chess.E6) and chess.E6 in b2.attackers(chess.BLACK, chess.D5)
    hyps2, value2, honest2 = li.exchange(b2, chess.D5)
    assert not honest2 and hyps2["noPin"] is False and value2 == -2       # the capture itself: a knight for a pawn
    hyps3, value3, honest3 = li.exchange(chess.Board("8/8/8/8/8/8/8/8 w - - 0 1"), chess.D5)   # an empty square: no target
    assert not honest3 and hyps3["target"] is False and value3 == 0


def test_a_query_that_does_not_answer_in_time_is_abandoned_and_the_binary_restarted(tmp_path):
    """A stand-in binary that answers after 5 s: a 0.3 s timeout raises TimeoutError and the next query (fresh process)
    still answers."""
    import os
    from catspace.incubator.analyst.lean_ideas import LeanIdeas
    fake = tmp_path / "slow.sh"
    fake.write_text("#!/bin/sh\nwhile read l; do case \"$l\" in slow*) sleep 5; echo late;; *) echo quick;; esac; done\n")
    os.chmod(fake, 0o755)
    li = LeanIdeas(binary=fake)
    assert li._ask("hello", until_end=False, timeout=2) == ["quick"]
    with pytest.raises(TimeoutError):
        li._ask("slow", until_end=False, timeout=0.3)
    assert li._ask("again", until_end=False, timeout=2) == ["quick"]
    li.close()


def test_the_kernel_composed_tier_matches_the_prototype_on_the_intermezzo_frame():
    from catspace.incubator.analyst.lean_ideas import detector
    li = detector()
    b = chess.Board("4k3/8/3b4/3n4/8/4N3/7P/1K6 w - - 0 1")
    hyps, value, honest = li.exchange(b, chess.D5)
    assert value == 3 and not honest and hyps["noOtherCapture"] is False and hyps["noOverload"] is True
    assert li.exchange_composed(b, chess.D5) == (2, True)
    assert li.exchange_cert(b, chess.D5, timeout=30, composed=True)


def test_the_kernel_sees_alekhines_gun_as_an_xray_and_a_still_blocker_as_none():
    from catspace.incubator.analyst.lean_ideas import detector
    li = detector()
    assert li.exchange(chess.Board("4k3/8/8/3n4/8/3R4/3R4/1K1Q4 w - - 0 1"), chess.D5)[0]["noXray"] is False
    assert li.exchange(chess.Board("4k3/8/8/3n4/3P4/4N3/8/1K1R4 w - - 0 1"), chess.D5)[0]["noXray"] is True
    assert li.exchange(chess.Board("4k3/8/8/3n4/2P5/1B6/Q7/1K6 w - - 0 1"), chess.D5)[0]["noXray"] is False


def test_the_kernel_capture_game_with_chains_counts_the_gun():
    from catspace.incubator.analyst.exchange_law import see_x
    from catspace.incubator.analyst.lean_ideas import detector
    li = detector()
    gun = chess.Board("4k3/8/4b3/3n4/3R4/3R4/3Q4/1K6 w - - 0 1")
    xv, xh, seex = li.exchange_xray(gun, chess.D5)
    assert seex == 1 == see_x(gun, chess.D5)[0] and xv == 1 and xh
    assert li.exchange(gun, chess.D5)[1] == -2                         # the plain count
    # the x-ray certificate's horizon on the gun is six plies full-width: beyond the certificate (timeouts), not asserted


def test_passer2_with_enemy_bishops_agrees_with_the_race_search():
    """Plan 52.16 bridge (not the proof -- the proof is Ideas.Passer2.unstoppablePasser2_sound): on random positions with one or two enemy bishops, every
    position the Lean detector PROVES is confirmed by the exhaustive race search. A single disagreement means the statement and the search differ."""
    import random
    from catspace.incubator.analyst.lean_ideas import detector
    from catspace.incubator.analyst.lean_passer_check import promotes_within
    rng = random.Random(7)
    li = detector()
    proved = tried = 0
    while tried < 3000:
        b = chess.Board(None)
        k, ek, pw, eb1, eb2, ep1, ep2 = rng.sample(range(64), 7)
        if chess.square_rank(pw) not in (4, 5, 6):
            pw = chess.square(chess.square_file(pw), rng.choice((4, 5, 6)))
        if pw in (k, ek):
            continue
        b.set_piece_at(k, chess.Piece(chess.KING, chess.WHITE)); b.set_piece_at(ek, chess.Piece(chess.KING, chess.BLACK))
        b.set_piece_at(pw, chess.Piece(chess.PAWN, chess.WHITE))
        used = {k, ek, pw}
        ok = True
        for s in (eb1, eb2)[:rng.randint(1, 2)]:
            ok = ok and s not in used
            used.add(s); b.set_piece_at(s, chess.Piece(chess.BISHOP, chess.BLACK))
        for s in (ep1, ep2)[:rng.randint(0, 2)]:
            ok = ok and s not in used and chess.square_rank(s) not in (0, 7)
            used.add(s); b.set_piece_at(s, chess.Piece(chess.PAWN, rng.choice((chess.WHITE, chess.BLACK))))
        b.turn = chess.WHITE
        if not ok or not b.is_valid() or b.is_check() or b.attackers_mask(chess.WHITE, ek):
            continue
        tried += 1
        for i in li.passer2(b):
            if i.status == "PROVED":
                proved += 1
                assert promotes_within(b, chess.square_file(i.square), i.plies), (b.fen(), chess.square_name(i.square))
    assert proved > 100                                                  # the bridge is not vacuous
