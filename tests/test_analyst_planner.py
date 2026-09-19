"""The planning layer: the codebook reads, the plan is a distribution with value and sharpness, the needs follow
from the named differences, the budget follows the sharpness."""
import numpy as np
import chess

from catspace.incubator.analyst import regimes as RG
from catspace.incubator.analyst import planner as PL
from catspace.incubator.analyst.graph import build, POSITION_QUESTIONS
from catspace.incubator.analyst.memo import Memo


def _codebook():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(600, len(POSITION_QUESTIONS))).astype(np.float32)
    P = {q: i for i, q in enumerate(POSITION_QUESTIONS)}
    pos[:, P["queens_on"]] = rng.integers(0, 3, 600); pos[:, P["threat_ours"]] = rng.choice([0, 1, 5], 600); pos[:, P["mate_ours"]] = 0; pos[:, P["mate_theirs"]] = 0
    res = rng.choice([0.0, 0.5, 1.0], 600)
    return RG.fit(pos, np.ones(600, bool), res, k=6), pos


def test_codebook_assigns_describes_and_marks_sharp():
    cb, pos = _codebook()
    codes = cb.assign(pos)
    assert codes.shape == (600,) and codes.max() < 6
    for c in range(6):
        t = cb.describe(c)
        assert ("queen" in t) and ("threat" in t)
    assert cb.sharp().dtype == bool and 0 < cb.sharp().sum() < 6


def test_needs_follow_the_named_differences():
    b = chess.Board("r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 10")
    g = build(b, Memo())
    now = {k: 0.0 for k in RG.CODE_KEYS}; now.update(queens_on=2, material_diff=0, phase=50, passed_ours=0, runner_ours=0, passed_theirs=0, king_attackers=0, mate=0, threat_max=0, rooks_connected_ours=0, open_files=0)
    tgt = dict(now); tgt.update(queens_on=0, material_diff=1.5, king_attackers=3, passed_ours=1)
    needs = PL.needs_for(b, g, now, tgt)
    kinds = [k for k, _, _ in needs]
    assert "remove" in kinds and "win" in kinds and "mate" in kinds and "lever" in kinds
    rm = next(sq for k, sq, _ in needs if k == "remove")
    assert b.piece_at(rm).piece_type == chess.QUEEN and b.piece_at(rm).color == chess.BLACK


def test_budget_scales_with_sharpness():
    assert PL.budget_for(0.1)["level"] == "shallow" and PL.budget_for(0.3)["level"] == "normal" and PL.budget_for(0.7)["level"] == "deep"
    assert PL.budget_for(0.7)["combo_cap"] > PL.budget_for(0.1)["combo_cap"]


def test_piece_serving_moves_trade_and_remove():
    import chess
    from catspace.incubator.analyst import rank
    from catspace.incubator.analyst.planner import PieceIdea
    from catspace.incubator.analyst.memo import Memo

    class P:
        pass
    b = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5")
    plan = P()
    plan.pieces = [PieceIdea(chess.C4, "our light-squared bishop on c4", True, .4, .2, .1, .2, .2, op="trade", value=0.03),
                   PieceIdea(chess.C5, "their dark-squared bishop on c5", False, .2, .3, .4, .2, .2, op="remove", value=0.03),
                   PieceIdea(chess.F6, "their kingside knight on f6", False, .2, .3, .4, .2, .2, op="", value=0.03)]      # no operator: no moves
    got = {b.san(m): how for m, (idea, how) in rank.piece_serving_moves(b, plan, list(b.legal_moves), Memo()).items()}
    assert got.get("Bd5") == "offers the trade"            # Nxd5 is the even exchange the move offers
    assert got.get("Na4") == "goes for it"                 # the knight attacks the c5 bishop and does not fall
    assert "Ng5" not in got and "Bxf7+" not in got         # the idea-less knight on f6 and a sacrifice are not served
