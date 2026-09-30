"""The job ledger (catspace/incubator/analyst/jobs.py): pins, guards and discoveries found by removing pieces."""
import chess

from catspace.incubator.analyst import jobs as J
from catspace.incubator.analyst.fates import clear_caches


def test_the_relative_pin_is_a_shield_job():
    clear_caches()
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p1B1/4P3/2N5/PPPP1PPP/R2QKBNR b KQkq - 0 4")
    js = J.ledger(b)
    pin = next(j for j in js if j.holder == chess.F6 and j.kind == "shield" and j.target == chess.D8)
    assert pin.attacker == chess.G5 and pin.value >= 5
    assert J.abandons(b, b.parse_san("Nd5"), js) > 0                   # the pinned knight walks away from the queen
    assert J.abandons(b, b.parse_san("Be7"), js) <= 0                  # another piece's move abandons nothing


def test_the_back_rank_guard_is_a_job_against_mate():
    clear_caches()
    b = chess.Board("2r2k1r/2p1ppbp/2Q1bqpn/p7/8/2P1BPN1/PP4PP/3R1RK1 w - - 1 18")
    js = J.ledger(b)
    assert any(j.holder == chess.C8 and j.target == chess.F8 and j.value >= 400 for j in js)


def test_a_piece_pinned_to_its_king_is_decisive():
    clear_caches()
    b = chess.Board("4k3/8/4n3/8/8/8/4R3/4K3 b - - 0 1")
    js = J.ledger(b)
    assert any(j.holder == chess.E6 and j.kind == "shield" and j.target == chess.E8 and j.value == J.DECISIVE for j in js)


def test_knowledge_orders_the_search_without_changing_its_values():
    from catspace.incubator.analyst.zone_search import ZoneSearch, TacticalSpec
    clear_caches()
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p1B1/4P3/2N5/PPPP1PPP/R2QKBNR b KQkq - 0 4")
    ms = [b.parse_san(s) for s in ("Be7", "Nd5", "h6")]
    plain = ZoneSearch(spec=TacticalSpec(budget=20000)).rank(b, ms)
    clear_caches()
    known = ZoneSearch(spec=TacticalSpec(budget=20000, knowledge=tuple(J.ledger(b)))).rank(b, ms)
    assert {m: plain[m][0] for m in ms} == {m: known[m][0] for m in ms}
