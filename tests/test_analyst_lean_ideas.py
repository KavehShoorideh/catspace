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
    ("8/8/3Pp1P1/2K1p2k/8/8/8/8 w - - 0 1", "g6", "ONEAWAY", "kingRoom", None),      # the enemy king could run out of moves
    ("8/8/8/4b3/8/4P3/8/K6k w - - 0 1", "e3", "FAIL", None, None),                   # an enemy bishop: two hypotheses fail
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
