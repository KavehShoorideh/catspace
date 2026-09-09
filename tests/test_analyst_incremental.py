"""Incremental exchange ledgers must equal fresh ones: inside the goal search (parent's ledger carried over where the
move touched no footprint) and inside the sure-gain reply loop."""
import chess

from catspace.incubator.analyst.goal_search import GoalSearch
from catspace.incubator.analyst.defense import all_defenses
from catspace.incubator.analyst import rank
from catspace.incubator.analyst.memo import Memo


def _walk(n):
    yield n
    for c in n.children or []:
        yield from _walk(c)


def test_goal_search_node_ledgers_equal_fresh_ledgers():
    fens = ["r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 10",
            "2r3k1/pp3ppp/4p3/3nP3/3P4/2N5/PP3PPP/2R3K1 b - - 0 20"]
    checked = 0
    for fen in fens:
        b = chess.Board(fen)
        target = max((sq for sq, p in b.piece_map().items() if p.color != b.turn and p.piece_type != chess.KING), key=lambda s: b.piece_at(s).piece_type)
        gs = GoalSearch(b, b.turn, target, memo=Memo(), engine=None, horizon=4, budget=25)
        gs.run()
        for n in _walk(gs.root):
            if n.ledger is None:
                continue
            fresh = {sq: d.see for sq, d in all_defenses(n.board).items() if d.falls}
            inc = {sq: d.see for sq, (d, _) in n.ledger.items() if d is not None and d.falls}
            assert fresh == inc, (fen, n.ply, fresh, inc)
            checked += 1
    assert checked > 20


def test_sure_gain_incremental_equals_full():
    b = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7")
    for m in rank.attacking_moves(b)[:6]:
        full = rank.base_ledger
        rank.base_ledger = lambda *a: None
        ref = rank.sure_gain(b, m, Memo(), signed=True)
        rank.base_ledger = full
        assert rank.sure_gain(b, m, Memo(), signed=True) == ref
