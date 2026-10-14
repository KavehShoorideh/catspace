"""The reasoning ledger (docs/ANALYST_PLAN.md section 39; rust/analyst_core/src/reasoning.rs) on Kaveh's position: phase 1 the pin on the e-pawn
as a PROVEN chain the solver confirms; phase 2 the roll forward through 1...Qc7 2.Rbd1 Rc6 3.h5 exd4."""
import time

import analyst_core as A
from catspace.incubator.analyst.discovery import ledger as L

FEN = L.KAVEH            # Black to move; the ledger reads either side, so the board is not mirrored


def test_read_sets_name_the_squares_a_fact_looked_at():
    L._a()
    v, reads = A.vocab_reads("5k1r/1p1n1p2/p1qb1p1p/4p3/2rP2PP/2PQ1PB1/PP2NK2/1R5R w - - 0 1", "(count (and (attacks d3) (at c4)))")
    assert v == 1 and {"d3", "c4"} <= set(reads) and "h8" not in reads and len(reads) < 20
    v2, reads2 = A.vocab_reads("5k1r/1p1n1p2/p1qb1p1p/4p3/2rP2PP/2PQ1PB1/PP2NK2/1R5R w - - 0 1", "(without (at e2) (count (and (attacks d3) (at c4))))")
    assert "e2" in reads2


def test_phase_1_the_pin_on_the_e_pawn_is_a_proven_chain_the_solver_confirms():
    t0 = time.time()
    rep = L.analyse(FEN)
    secs = time.time() - t0
    fs = L.by_id(rep)
    pin = [f for f in rep["findings"] if f["type"] == "PIN" and f["claim"].startswith("e5 is pinned")]
    assert len(pin) == 1 and pin[0]["status"] == "PROVEN" and pin[0]["lo"] >= 2           # bishop (3) - pawn (1)
    lines = L.chain(rep, pin[0]["id"])
    text = "\n".join(lines)
    for must in ("if e5 leaves, Bg3 hits Bd6", "effectively unguarded", "Qc6 has another job", "no longer sees Rc4", "gives check"):
        assert must in text, must
    assert rep["confirmations"] and rep["confirmations"][0]["confirmed"] and rep["confirmations"][0]["finding"] == pin[0]["id"]
    assert all(m["status"] == "forced" for m in rep["confirmations"][0]["moves"]) and len(rep["confirmations"][0]["moves"]) == 2   # exd4 and e4 both lose
    assert rep["questions_asked"] <= rep["budget"] == 200
    assert set(f["status"] for f in rep["findings"]) <= {"PROVEN", "CONDITIONAL"}
    assert rep["back_pocket"] and rep["back_pocket"][0]["premise"].startswith("e5 leaves")   # the conjecture waiting on its premise
    print(f"OBS ledger test phase 1: {rep['questions_asked']} questions, {rep['facts_evaluated']} facts, {secs:.2f} s, counts {rep['counts']}")
    for line in lines:
        print("OBS   " + line)


def test_phase_2_the_pin_retracts_when_the_rook_reaches_c6_and_exd4_is_playable():
    recs = L.roll(FEN, "Qc7 Rbd1 Rc6 h5 exd4".split(), verbose=False)
    pin = lambda r: r["pin_on_square"][0] if r["pin_on_square"] else None
    assert recs[0]["pins"][0]["lo"] >= 2
    assert pin(recs[1])["lo"] >= 2 and pin(recs[2])["lo"] >= 2                       # the queen still holds both jobs
    assert pin(recs[3])["lo"] <= 0 and "may move" in pin(recs[3])["claim"]            # Rc6: the guard is jobless, the pin is gone
    assert recs[3]["retracted"] > 0 and recs[3]["premise_listeners_woken"] > 0
    assert recs[5]["move"] == "e5d4" and recs[5]["touched"] and recs[5]["retracted"] > 0
    L_ = L.new(FEN)
    L_.run()
    for uci in ("c6c7", "b1d1", "c4c6"):
        L_.push(uci)
    rep = L_.report_dict()
    old = [f for f in rep["stale"] if f["type"] == "PIN" and f["claim"].startswith("e5 is pinned")]
    assert [f for f in old if f["stale"] == 3 and f["lo"] >= 2]                       # "true at ply 2, refuted at 3" stays readable
    now = [f for f in rep["findings"] if f["type"] == "PIN" and "e5" in f["key"].split("|")[2]]
    assert now and now[0]["lo"] <= 0
    costs = {m["move"]: m["cost"] for m in __import__("json").loads(now[0]["note"].split(" | ")[0])}
    assert costs["e5d4"][1] <= 0                                                      # ...exd4 reads as playable (it wins the pawn it takes)
    for r in recs:
        print(f"OBS ledger test ply {r['ply']}: retracted {r['retracted']}, looked up {r['lookups']}, re-asked {r['re_asked']}")
