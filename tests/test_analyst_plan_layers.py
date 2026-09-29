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
    # since any piece's check counts on a joined king (2026-09-28), c4 alone threatens d5: cxd5 exd5 opens the 6th rank
    # for Rh6+ (Stockfish: exd5 Rh6+ Ke7 Nxd5+, -4.77 for Black); before, only c4 with Rh6 (the pin) broke d5
    assert "c4" in sans and "Rh6" not in sans
    b = chess.Board(KAVEH_AFTER_A3)
    p = next(p for p in G.plans_for(b, chess.BLACK, targets={chess.F2}) if p.san(b) == "Rg1 Rf1")
    ok, _ = G.goes_through(b, p)
    assert not ok and {"Rh2", "Nh3"} <= {a.san for a in G.final_answers(b, p)}
    b = chess.Board("8/5pk1/p6p/5B2/2np1p2/8/PP2rPPP/3R2K1 w - - 0 29")
    sc = G.score_moves(b, list(b.legal_moves))
    assert max(sc, key=sc.get) == b.parse_san("Bd3").uci()


def test_ablation_finds_the_shield_and_its_removal():
    """Kaveh 2026-09-28, 6k1/6p1/8/2r1P2Q/6K1/.. after Kf5: the e5 pawn shields the king from the c5 rook; Black removes
    it with Rxe5+ and Rc5+ skewers king and queen (+5); the threat depends on both rooks and the queen."""
    from catspace.incubator.analyst import ablation as A
    b = chess.Board("6k1/6p1/8/2r1PK1Q/8/3P2P1/2r5/8 b - - 2 38")
    amap = A.ablation_map(b)
    assert amap[0].square == chess.E5 and amap[0].shield
    assert {h.square for h in A.support(b, chess.BLACK, amap)} == {chess.H5, chess.C5, chess.C2}
    q = A.attack(b, chess.BLACK, amap)[0]
    assert q.target == chess.E5 and q.ways[0][0] == "Rxe5+" and q.ways[0][1] >= 4
