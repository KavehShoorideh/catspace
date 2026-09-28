"""plan_layers: every registered tactical plan layer answers the same interface; the frame-graph layer's cases."""
import chess
import pytest

from catspace.incubator.analyst import plan_layers
from catspace.incubator.analyst import frame_goals as G

KAVEH_BEFORE_A3 = "7R/3b2r1/1p1kp3/p2p4/3PpN2/1P2P3/PKP2P2/8 w - - 0 29"
KAVEH_AFTER_A3 = "7R/3b2r1/1p1kp3/p2p4/3PpN2/PP2P3/1KP2P2/8 b - - 0 29"


@pytest.mark.parametrize("name", sorted(plan_layers.LAYERS))
def test_every_layer_scores_the_moves_it_is_given(name):
    layer = plan_layers.get(name)
    b = chess.Board("8/5pk1/p6p/5B2/2np1p2/8/PP2rPPP/3R2K1 w - - 0 29")
    moves = [b.parse_san(s) for s in ("Bd3", "f3", "h3")]
    sc = layer.score_moves(b, moves)
    assert set(sc) == {m.uci() for m in moves} and all(isinstance(v, float) for v in sc.values())
    assert isinstance(layer.explain(b), list)


def test_frame_goals_reads_kavehs_plans():
    """Kaveh 2026-09-28: White breaks the d5 frame with c4 and a rook that takes e6 off it (either alone breaks
    nothing); Black's rook heads for f2 (Rg1 Rf1) and White's answers are Rh2 and Nh3; the double attack Bd3."""
    b = chess.Board(KAVEH_BEFORE_A3)
    ps = G.plans_for(b, chess.WHITE, targets={chess.D5})
    sans = {p.san(b) for p in ps}
    assert "c4 Rh6" in sans and "Rh6 c4" in sans and "c4" not in sans and "Rh6" not in sans
    b = chess.Board(KAVEH_AFTER_A3)
    p = next(p for p in G.plans_for(b, chess.BLACK, targets={chess.F2}) if p.san(b) == "Rg1 Rf1")
    ok, _ = G.goes_through(b, p)
    assert not ok and {"Rh2", "Nh3"} <= {a.san for a in G.final_answers(b, p)}
    b = chess.Board("8/5pk1/p6p/5B2/2np1p2/8/PP2rPPP/3R2K1 w - - 0 29")
    sc = G.score_moves(b, list(b.legal_moves))
    assert max(sc, key=sc.get) == b.parse_san("Bd3").uci()
