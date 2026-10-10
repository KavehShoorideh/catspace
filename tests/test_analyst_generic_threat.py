"""The generic threat (rust/analyst_core/src/threat.rs; Kaveh 2026-10-10: a threat is losing a piece or the king, with
a value; a job is a threat a piece parries)."""
import chess
import pytest

analyst_core = pytest.importorskip("analyst_core")


def test_material_threat():
    mine, _, _ = analyst_core.threat_check("6k1/5ppp/8/3n4/8/8/5PPP/3R2K1 w - - 0 1")
    assert ("d5", 3, False) in mine


def test_mate_threat_by_either_side_to_move():
    assert ("g8", 500, True) in analyst_core.threat_check("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")[0]
    assert ("g8", 500, True) in analyst_core.threat_check("6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1")[1]


def test_mate_by_checks_through_a_spite_block():
    # Rc1 Qb2: Rc8+ Bf8 Rxf8# -- two checks, a block in between
    b = chess.Board("7k/4b2p/p3Npp1/1p3p2/1P6/P7/1Bq2PPP/1R4K1 w - - 6 32"); b.push_san("Rc1"); b.push_san("Qb2")
    assert ("h8", 500, True) in analyst_core.threat_check(b.fen())[0]


def test_job_guarding_the_back_rank():
    _, _, jobs = analyst_core.threat_check("3r2k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1")
    assert ("d8", [("g8", 500, True)]) in jobs


def test_promotion_is_a_threat():
    b = chess.Board("8/p3P1kp/6p1/2R5/PPp5/2P3Pb/2K5/4n3 w - - 0 37"); b.push_san("Kd1"); b.push_san("Ng2")
    assert ("e8", 8, False) in analyst_core.threat_check(b.fen())[0]


def test_single_threat_lemma_proves_a_job_and_refuses_a_lone_attack():
    # Rg7: the g3 queen shields its king -- every escape leaves a threat: proved, and it wins the queen
    tr, dec, *_, imm = analyst_core.plan_trace("2k3q1/ppp1bp1r/8/5P2/P1PpP3/3P2Q1/2PB3P/1R4K1 b - - 1 27")
    assert imm and imm[1] == "h7g7"
    # a lone attack on a free piece: refused without proof
    tr = analyst_core.plan_trace("4k3/8/8/3q4/8/8/8/2R1K3 w - - 0 1")[0]
    assert any("single-threat lemma" in l for l in tr if l.strip().startswith("Rc5"))
