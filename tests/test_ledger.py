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


def test_phase_3_decisions_on_hand_positions():
    import chess
    assert A.Ledger("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", 200).decide()["move"] is None      # mated: no move, no panic
    b = chess.Board("rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 4")      # 3.Bb5+ : in check, several replies
    assert b.is_check()
    d = A.Ledger(b.fen(), 200).decide()
    assert d["legal"] == b.legal_moves.count() and d["candidates"] == d["legal"]         # in check every legal reply is a candidate
    # a mate in one is decided from the rules alone, PROVEN, with no question asked
    d = A.Ledger("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1", 200).decide()
    assert d["move"] == "a1a8" and d["status"] == "PROVEN" and d["questions"] == 0 and d["interval"][0] >= 1000
    # ...Qg5+ discovers the rook on the queen (the "discovery" reason fires) and the line through Rxe6 is valued: Black wins about the queen
    d = A.Ledger("1r2r3/1kp1q2p/4Q3/p2P4/1P1P4/P1p2P2/2P3PP/2KR3R b - - 0 30", 200).decide()
    assert d["move"] == "e7g5" and d["interval"][0] >= 4
    assert any("discovery" in a[1] for a in d["all"] if a[0] == "e7g5")
    assert d["nodes"] > 1000 and d["confirm"]["result"] == "CONFIRMED"


def test_trace_reads_in_plain_chess_order():
    t = L.new("r5k1/2p1q2p/2Qp2p1/1rpPpp2/2P1P3/P4P2/6PP/4R1K1 w - - 0 27", 1000).trace(4)
    text = "\n".join(t["trace"])
    assert "THREATS TO White: none above the margin." in text and "BRANCH (b)" in text and "an open threat of ours on Ra8" in text
    assert t["move"] == "c6a8" and t["status"] == "PROVEN (solver-confirmed)" and t["positions"] <= 6 and "DECISION: Qxa8" in text
    t = L.new(FEN, 1000).trace(4)                              # Kaveh's position: the latent threat comes first, the pin of the e-pawn in words
    assert "LATENT THREATS to Black" in t["trace"][1] and any("e5 is pinned to Bd6" in l for l in t["trace"][:8])
    assert t["positions"] <= 200 and any(l.startswith("DECISION") for l in t["trace"][-3:])


def test_d2_the_attacker_is_seen_when_the_mover_is_in_check():
    # Black to move IN CHECK (Qxa8+): the c4 pawn still attacks Rb5 -- the position with the other side to move is not one shakmaty will build,
    # so the capture is read on the board (a pseudo-legal capture that leaves the capturer's king safe)
    rep = A.Ledger("Q5k1/2p1q2p/3p2p1/1rpPpp2/2P1P3/P4P2/6PP/4R1K1 b - - 0 27", 200).run()
    assert any(f["type"] == "SAFE" and f["claim"].startswith("Rb5 loses 5") and "c4 takes first" in f["claim"] for f in rep["findings"])
    # the relaxed board survives the other side being in check (the checked side becomes the mover; board words read the same)
    v, reads = A.vocab_reads("Q5k1/2p1q2p/3p2p1/1rpPpp2/2P1P3/P4P2/6PP/4R1K1 b - - 0 27", "(without (at b4) (count (and (attacks a1) (at g1))))")   # (squares named from the mover's side: Black)
    assert v == 1


def test_c2_a_piece_is_saved_only_where_it_is_safe_and_a_trapped_piece_counts_in_full():
    # after b3 the rook on c4 has no safe square and nothing saves it
    t = L.new("5k1r/1p1n1p2/p1qb1p1p/4p3/2rP2PP/1PPQ1PB1/P3NK2/1R5R b - - 0 1", 1000).trace(2)
    text = "\n".join(t["trace"])
    assert "Rc4 attacked by [b3, Qd3]" in text and "TRAPPED" in text
    # a rook that can step to a safe square is "saved by" that step
    t2 = L.new("4k3/8/8/8/3r4/8/2P5/4K3 b - - 0 1", 1000).trace(1)
    assert "THREATS TO Black, highest first" not in "\n".join(t2["trace"]) or "TRAPPED" not in "\n".join(t2["trace"])


def test_c3_lines_are_read_both_ways_the_discovered_attack_is_listed():
    t = L.new("1r2r3/1kp1q2p/4Q3/p2P4/1P1P4/P1p2P2/2P3PP/2KR3R b - - 0 30", 1000).trace(2)
    text = "\n".join(t["trace"])
    assert "LATENT OPPORTUNITIES" in text and "if Qe7 leaves, Re8 hits Qe6" in text and "wins 4" in text    # our slider, our shield, their piece
    assert t["move"] == "e7g5" and t["status"].startswith("PROVEN")
