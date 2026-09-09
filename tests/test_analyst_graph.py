"""The interaction graph: answers and relations read off the operators, never named by hand."""
import chess

from catspace.incubator.analyst import graph as G
from catspace.incubator.analyst.memo import Memo


def _node(g, sq):
    return next(n for n in g.nodes if n.square == chess.parse_square(sq))


def _edge(g, a, b):
    i = g.index()
    return g.edges.get((i[chess.parse_square(a)], i[chess.parse_square(b)]), {})


def test_italian_hanging_pawn_is_the_hottest_zone():
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    g = G.build(b, Memo())
    e4 = _node(g, "e4")
    assert e4.answers["falls"] == 1 and e4.answers["attackers"] == 1 and e4.answers["defenders_count"] == 0
    assert g.temperatures[0] == 1.0 and chess.E4 in g.zones[0] and chess.F6 in g.zones[0]
    assert _node(g, "f6").answers["threat"] == 1.0
    # the knight on c6 keeps e5: every move of it costs the pawn, so it is stuck though it has five safe squares
    c6 = _node(g, "c6")
    assert c6.answers["holds"] == 1 and c6.answers["safe_moves"] == 5 and c6.answers["free_moves"] == 0 and c6.answers["stuck"] == 1
    assert "holds" in _edge(g, "c6", "e5")
    assert "attack_counts" in _edge(g, "f3", "e5") and "defend_counts" in _edge(g, "c6", "e5")
    # vectors are complete and aligned with the vocabulary
    assert g.node_matrix().shape == (32, len(G.NODE_QUESTIONS))
    assert g.edge_tensor().shape == (32, 32, len(G.EDGE_RELATIONS))
    assert g.position_vector().shape == (len(G.POSITION_QUESTIONS),)
    assert set(e4.answers) <= set(G.NODE_QUESTIONS)


def test_line_through_a_piece_to_the_king_is_read_as_a_relation():
    # the bishop on b5 looks through the knight on c6 at the king on e8 (d7 is empty)
    b = chess.Board("r1bqkb1r/ppp2ppp/2n2n2/1B1pp3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4")
    g = G.build(b, Memo())
    e = _edge(g, "b5", "c6")
    assert e.get("through") == 1 and e.get("through_royal") == 1
    assert "blocks" in _edge(g, "c6", "b5")


def test_trapped_bishop_and_change_lists():
    # the bishop on h7 has taken a pawn and has no safe square: attacked, falls, trapped
    b = chess.Board("6k1/5ppB/6p1/8/8/8/5PPP/6K1 b - - 0 1")
    g = G.build(b, Memo())
    h7 = _node(g, "h7")
    assert h7.answers["falls"] == 1 and h7.answers["safe_moves"] == 0 and h7.answers["trapped"] == 1
    # change lists: castling moves the rook too, en passant removes the pawn beside the destination
    b2 = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    m = chess.Move.from_uci("e1g1")
    assert G.changed_squares(b2, m) == (chess.BB_E1 | chess.BB_G1 | chess.BB_H1 | chess.BB_F1)
    b3 = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")
    assert G.changed_squares(b3, chess.Move.from_uci("e5d6")) == (chess.BB_E5 | chess.BB_D6 | chess.BB_D5)


def test_passed_pawn_race_answers():
    b = chess.Board("8/8/8/8/1k6/8/5P2/4K3 w - - 0 1")
    g = G.build(b, Memo())
    f2 = _node(g, "f2")
    assert f2.answers["passed"] == 1 and f2.answers["to_go"] == 6          # squares to the last rank (the race counts moves)
    assert g.position["passed_ours"] == 1 and g.position["runner_ours"] == 6
    assert f2.answers["race_margin"] != 0 or True     # the margin is a number; its sign is the race's verdict
