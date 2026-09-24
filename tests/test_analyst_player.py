"""Invariants for the MVP player's search (2026-09-23 bug hunt): the transposition table must not change any root
value; a mate found at depth d is found at every deeper depth; the resolved exchange reads known exchanges right;
the puzzle set's tactics are found within the budget."""
import chess
import pytest

from catspace.incubator.analyst.player import MATE, Search, evaluate, material, Player

POSITIONS = [
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r2q1rk1/ppp2ppp/2n1bn2/3p4/3P4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 9",
    "4k3/7K/8/8/7b/8/8/2r4R w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/1b1P4/2NBPN2/PPP2PPP/R2QK2R w KQ - 0 8",
    "2r3k1/pp3ppp/8/3q4/8/2N5/PPP2PPP/R2QK2R b KQ - 0 1",
]


def root_values(fen, depth, use_tt):
    b = chess.Board(fen); s = Search(node_cap=10 ** 9)
    if not use_tt:
        s.tt = _NoTT()
    return s.root(b, list(b.legal_moves), depth)


class _NoTT(dict):
    def get(self, k, d=None):
        return None

    def __setitem__(self, k, v):
        pass


@pytest.mark.parametrize("fen", POSITIONS)
def test_transposition_table_changes_no_root_value(fen):
    a = root_values(fen, 3, True); b = root_values(fen, 3, False)
    assert a.keys() == b.keys()
    for m in a:
        assert a[m] == pytest.approx(b[m], abs=1e-9), f"{fen} {m}: with TT {a[m]} without {b[m]}"


def test_material_is_antisymmetric():
    for fen in POSITIONS:
        b = chess.Board(fen)
        assert material(b, chess.WHITE) == -material(b, chess.BLACK)


def test_mate_in_one_found_at_every_depth():
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1")     # Qd8#
    for d in (1, 2, 3, 4):
        s = Search(node_cap=10 ** 9); vals = s.root(b, list(b.legal_moves), d)
        best = max(vals, key=vals.get)
        assert best == chess.Move.from_uci("d1d8"), f"depth {d}: {best}"
        assert vals[best] >= MATE - 1


def test_resolved_exchange_reads_a_defended_square_as_a_trade():
    b = chess.Board("4k3/pppp4/5n2/8/8/1BN5/PPPP4/4K3 w - - 0 1")  # Nd5 attacked by Nf6, defended by Bb3 (pawns: no insufficient-material draw)
    s = Search(node_cap=10 ** 9)
    b.push(chess.Move.from_uci("c3d5")); v = s.quiesce(b, chess.WHITE, -MATE * 2, MATE * 2, 6); b.pop()
    assert v == pytest.approx(material(b, chess.WHITE), abs=0.5)    # a trade: material unchanged after it settles (the leaf adds a mobility term under 0.5)
    b2 = chess.Board("4k3/pppp4/5n2/8/8/2N5/PPPP4/4K3 w - - 0 1")  # undefended: a loss of three
    b2.push(chess.Move.from_uci("c3d5")); v2 = s.quiesce(b2, chess.WHITE, -MATE * 2, MATE * 2, 6); b2.pop()
    assert v2 == pytest.approx(material(b2, chess.WHITE) - 3.0, abs=0.5)


def test_deeper_keeps_a_won_exchange():
    """The search must not prefer a move that loses material at depth 3 when depth 1 already sees the gain."""
    b = chess.Board("4k3/7K/8/8/7b/8/8/2r4R w - - 0 1")
    s1 = Search(node_cap=10 ** 9); v1 = s1.root(b, list(b.legal_moves), 1)
    s3 = Search(node_cap=10 ** 9); v3 = s3.root(b, list(b.legal_moves), 3)
    assert max(v1, key=v1.get) == chess.Move.from_uci("h1c1")
    assert max(v3, key=v3.get) == chess.Move.from_uci("h1c1")
    assert v3[chess.Move.from_uci("h1c1")] >= v3[chess.Move.from_uci("h1h4")]


def test_player_finds_the_hanging_piece_and_the_mate():
    pl = Player(None, None, depth=3, node_cap=4000, use_lemmas=False)
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1")
    m, info = pl.choose(b); assert m == chess.Move.from_uci("d1d8")
    b = chess.Board("4k3/7K/8/8/7b/8/8/2r4R w - - 0 1")
    m, info = pl.choose(b); assert m == chess.Move.from_uci("h1c1")


def test_graph_cache_does_not_reuse_across_different_bodies():
    """The same mover, square and move in two positions whose other pieces differ must not share a verdict
    (2026-09-23 bug: keyed by the mover alone)."""
    from pathlib import Path
    pl = Player(Path("data/derived/lemmas/library_v2_size3.json"), Path("data/derived/lemmas/proved_v2_size3.json"), depth=1, node_cap=200, use_lemmas=True)
    b1 = chess.Board("4k3/7K/8/8/7b/8/8/2r4R w - - 0 1")          # Rg1 loses the rook to Rxg1
    b2 = chess.Board("4k3/7K/8/8/7b/8/8/7R w - - 0 1")            # no black rook: Rg1 is safe
    m = chess.Move.from_uci("h1g1")
    v1 = pl.graph_verdicts(b1, [m])[m]
    v2 = pl.graph_verdicts(b2, [m])[m]
    assert v1[1] > 0, f"Rg1 with the black rook present should be called losing: {v1}"
    assert v2[1] == 0, f"Rg1 without the black rook should not be called losing: {v2} (cache reused across different bodies?)"


def test_analysis_values_are_exact_not_bounds():
    """With alpha-beta at the root, the second-best moves come back as bounds; the analysis must search a full window
    so a move that allows mate in one reads as lost (2026-09-23: Bxb5 read +0.24 in game 669)."""
    b = chess.Board("r4rk1/pb1n1p2/2p1p1p1/1p5q/3P3Q/3B4/PPP2P2/1K1R2NR w - - 0 19")
    s = Search(node_cap=10 ** 9)
    vals = s.root(b, list(b.legal_moves), 2, full_window=True)
    assert vals[chess.Move.from_uci("d3b5")] <= -MATE + 1
    assert max(vals, key=vals.get) == chess.Move.from_uci("h4h5")
