"""threats.py: a threat with its required defences; a reply that is not one of them is punished at once."""
import chess

from catspace.incubator.analyst.memo import Memo
from catspace.incubator.analyst.threats import threats_after, judge_reply, mate_threat


def test_scholars_mate_defences_and_execution():
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3")
    ts = threats_after(b, b.parse_san("Qh5"), Memo(), None, budget=60)
    mate = next(t for t in ts if t.kind == "mate")
    assert mate.text == "mate (Qxf7#)"
    assert mate.defences["g6"] == "blocks the line to f7" and mate.defences["Qe7"] == "defends f7" and mate.defences["Ke7"] == "the king steps away"
    assert mate.punishes["Nf6"] == "Qxf7#" and "Nf6" not in mate.defences
    after = b.copy(); after.push_san("Qh5")
    out = judge_reply(ts, after, after.parse_san("Nf6"))
    assert any("Nf6 does not: Qxf7# executes" in l for l in out)
    out = judge_reply(ts, after, after.parse_san("g6"))
    assert any("parried by g6 (blocks the line to f7)" in l for l in out)
    # after the mate itself nothing is threatened
    after.push_san("Nf6")
    assert threats_after(after, after.parse_san("Qxf7#"), Memo(), None) == []


def test_no_threat_where_every_reply_holds():
    b = chess.Board("r1bqkbnr/ppp2ppp/2np4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4")
    ts = threats_after(b, b.parse_san("Bb5"), Memo(), None, budget=60)          # a pin is pressure, not yet a threat
    assert all(t.kind != "win" or t.target != chess.C6 for t in ts)
