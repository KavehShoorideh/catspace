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
