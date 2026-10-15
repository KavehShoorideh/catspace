"""THE EVENT LOOP'S KERNEL (Kaveh 2026-10-15, docs/ANALYST_PLAN.md section 40, phase 1): events, conditions, an agenda by stake, chains as data; the two
new words MOBILITY and ATTACKABLE (the plan's REACH; `reach` was taken by the king's walk). rust/analyst_core/src/agenda.rs and reasoning/chain.rs.
Numbers are printed as OBS lines; one position each, so they are directional."""
import subprocess
from pathlib import Path

import pytest

import catspace.incubator.analyst  # noqa: F401
import analyst_core as A
from catspace.incubator.analyst.discovery import base_vocabulary as V
from catspace.incubator.analyst.discovery import ledger as LG

MOVE30 = "2r2rk1/5pp1/1p4qp/p1b5/6QR/6P1/1B2RP2/5K2 w - - 6 30"
# the same position with White's queen on h3: it no longer guards e2, so Re2 holds Bb2 alone
MOVE30_QH3 = "2r2rk1/5pp1/1p4qp/p1b5/7R/6PQ/1B2RP2/5K2 w - - 6 30"
NXC6 = "1r1qk2r/p3bpp1/2p1p2p/2p1N1P1/4PPn1/2NPp3/PPP1Q1P1/1R3RK1 w k - 1 16"


def chain(fen: str, questions: int = 120) -> dict:
    L = LG.new(fen, 5000)
    LG.load_transitions(L)
    return L.chain(questions)


def recs(r: dict, kind: str) -> list:
    return [c for c in r["agenda"]["chains"] if c["kind"] == kind]


# ---- the words ------------------------------------------------------------------------------------------------------------------------------

def test_mobility_is_the_legal_squares_that_are_safe_by_the_exchange():
    fen = "k7/8/8/8/1p6/8/8/1N5K w - - 0 1"        # Nb1: a3 and c3 are taken by the b4 pawn, d2 is safe
    legal, safe = A.vocab_eval(fen, "(count (moves b1))"), A.vocab_eval(fen, "(count (mobility b1))")
    print(f"OBS mobility of Nb1: {safe} safe square(s) of {legal} legal")
    assert (legal, safe) == (3, 1)
    assert A.vocab_eval(fen, "(count (and (mobility b1) (at d2)))") == 1
    assert A.vocab_eval(fen, "(count (and (mobility b1) (at a3)))") == 0


def test_attackable_is_reach_within_k_moves_with_every_landing_safe():
    near = "r6k/8/8/8/3N4/8/8/6K1 w - - 0 1"       # Ra4 or Rd8 attacks the knight at once
    assert A.vocab_eval(near, "(attackable d4 1)") == 1
    far = "r6k/8/8/3P4/3R4/1P6/8/6K1 w - - 0 1"    # pawn b3 takes on a4, pawn d5 shuts the file, every other square of the rook's rank or file is its own
    got = [A.vocab_eval(far, f"(attackable d4 {k})") for k in (1, 2, 3)]
    print(f"OBS attackable Rd4 within 1 / 2 / 3 moves of Black: {got}")
    assert got == [0, 0, 1]


def test_new_words_have_glosses_and_lean_renderings():
    assert "safe on by the exchange" in V.gloss(V.parse("(mobility W)"))
    assert "within 2 of its moves" in V.gloss(V.parse("(attackable P 2)"))
    assert "mobilityOf" in V.lean(V.parse("(count (mobility P))")) and "attackableOf" in V.lean(V.parse("(attackable P 2)"))


@pytest.mark.skipif(not Path("lean/ideas/.lake").exists(), reason="the Lean project is not built")
def test_new_words_typecheck_in_lean(tmp_path):
    terms = ["(count (mobility P))", "(attackable P 2)", "(count (and (mobility O) (not unhit)))"]
    body = "\n".join(f"def t{i} (p : Position) (c : Color) (W L P O : Square) (hist : List Position) : Nat := {V.lean(V.parse(t))}" for i, t in enumerate(terms))
    src = tmp_path / "VocabCheckMobility.lean"
    src.write_text(f"import Ideas.Base.Vocabulary\nset_option linter.unusedVariables false\nopen Chess Ideas.Vocabulary\n{body}\n")
    r = subprocess.run(["lake", "env", "lean", str(src)], cwd="lean/ideas", capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr


# ---- the loop -------------------------------------------------------------------------------------------------------------------------------

def test_a_reason_event_opens_a_question_for_the_other_side():
    r = chain(MOVE30)
    reasons = [e for e in r["agenda"]["events"] if e["kind"] == "reason" and e["subject"].startswith("HELD|")]
    assert reasons
    sides = {e["actor"] for e in reasons}
    assert sides == {0, 1}, "both sides' findings are held by something"
    deflects = recs(r, "DEFLECT")
    for e in reasons:
        owner, holder, piece = e["subject"].split("|")[1], e["subject"].split("|")[3], e["subject"].split("|")[2]
        want = 1 - e["actor"]                               # the question is the OTHER side's
        assert any(d["actor"] == want and d["key"].startswith(f"DEFLECT|{want}|{holder},{piece}") for d in deflects), (e["subject"], owner)
    # the example: Qg4 is held by Rh4 -> Black asks whether Rh4 can be got off its post, and answers with its mobility and reach in ONE line
    rh4 = [d for d in deflects if d["actor"] == 1 and "Rh4" in d["text"]]
    print("OBS", rh4[0]["answer"][:200])
    assert rh4 and "MOBILITY Rh4" in rh4[0]["answer"] and "REACH" in rh4[0]["answer"]
    assert not any(c["parent"] == rh4[0]["q"] for c in r["agenda"]["chains"])      # no chain below it: one line


def test_a_question_whose_stake_does_not_beat_the_best_known_value_is_not_opened():
    r = chain(NXC6)
    best = r["agenda"]["best"]
    print(f"OBS best known value per side after the chain on the Nxc6 position: White {best[0]} Black {best[1]}; refused: {len(r['agenda']['refused'])}")
    assert best[0] >= 5                                      # Nxc6 settles first (stake 10)
    refused = [x for x in r["agenda"]["refused"] if "does not beat the best known value" in x]
    assert refused
    # a smaller candidate of the same side (Nxg4, stake 4) was opened as a question but its targets (stake 1) were not
    nxg4 = [c for c in recs(r, "POKE") if c["text"].startswith("Nxg4")]
    assert nxg4 and not any(c["parent"] == nxg4[0]["q"] for c in r["agenda"]["chains"])
    # a chain is its own budget: every question carries its stake
    assert all(c["stake"] >= 0 for c in r["agenda"]["chains"])


def test_a_chain_settles_and_prints():
    r = chain(NXC6)
    lines = r["lines"]
    assert lines[0].startswith("THE CHAIN")
    assert any("assess Nxc6" in l for l in lines)
    assert lines[-1].startswith("EVENTS:")
    # the records are data: question, answer, reason, stake, cost
    for c in r["agenda"]["chains"]:
        assert set(c) >= {"kind", "key", "actor", "text", "answer", "reason", "stake", "cost", "value"}
    # nothing is asked twice
    keys = [c["key"] for c in r["agenda"]["chains"]]
    assert len(keys) == len(set(keys))


def test_the_nxc6_chain_settles_at_the_pawn_and_a_rook():
    r = chain(NXC6)
    poke = next(c for c in recs(r, "POKE") if c["text"].startswith("Nxc6"))
    kids = [c["text"] for c in r["agenda"]["chains"] if c["parent"] == poke["q"] and c["kind"] == "SAFE"]
    assert any("Qd8" in k for k in kids) and any("Rb8" in k for k in kids)
    ass = next(c for c in recs(r, "ASSESS") if "Nxc6" in c["text"])
    print("OBS", ass["answer"])
    assert "no single move of Black saves both Qd8 and Rb8" in ass["answer"]
    lo, hi = ass["value"]
    cand = next(c for c in r["candidates"] if c["san"] == "Nxc6")
    print(f"OBS Nxc6 settles at {cand['lo']}..{cand['hi']} (the pawn and a rook; the exact solver confirms the gain)")
    assert lo >= 5 and "SOLVER: forces a gain of" in ass["answer"]


def test_the_move_30_two_move_threat_needs_the_deflection_to_hold():
    # Qg4 guards e2, so Re2 is held twice: the static reading finds "...Rfe8, then Qb1+ wins Bb2 (3)" but the exact solver does not confirm it
    r = chain(MOVE30)
    d = next(c for c in recs(r, "DEFLECT") if c["actor"] == 1 and "Re2" in c["text"] and "Bb2" in c["text"])
    print("OBS", d["answer"][:260])
    assert "Bb2 falls: 3" in d["answer"] and "does NOT confirm" in d["answer"] and d["value"] == [0, 3]
    # with the queen on h3 the deflection works and is SOLVER-CONFIRMED: a conditional two-move threat of 3 for Black
    r2 = chain(MOVE30_QH3)
    d2 = next(c for c in recs(r2, "DEFLECT") if c["actor"] == 1 and "Re2" in c["text"] and "Bb2" in c["text"])
    print("OBS", d2["answer"][:300])
    assert "SOLVER-CONFIRMED" in d2["answer"] and d2["value"] == [3, 3]
    assert r2["agenda"]["best"][1] >= 3                       # the armed tactic raised Black's best known value


def test_move_30_chain_has_the_trade_the_deflections_and_the_discovery():
    r = chain(MOVE30)
    txt = "\n".join(r["lines"])
    assert "is Qg4 safe" in txt and "trade" in txt
    assert any("transition table" in c["text"] for c in r["agenda"]["chains"])        # the stake of the trade is read from data/derived/transitions.json when it exists
    assert "the c-file opens Rc8 onto Qc4 when Bc5 moves" in txt
    qc4 = next(c for c in r["candidates"] if c["san"] == "Qc4")
    print(f"OBS Qc4: sure cost of what it opens {qc4['constraint']}, risk up to {qc4['risk']}")


def test_trace_keeps_its_answer_on_the_event_loop_and_the_script():
    fen = "r5k1/2p1q2p/2Qp2p1/1rpPpp2/2P1P3/P4P2/6PP/4R1K1 w - - 0 27"
    out = []
    for loop in (True, False):
        L = LG.new(fen, 1000)
        LG.load_defence_frame(L)
        if loop:
            LG.load_transitions(L)
        r = L.trace(4, "defined", "frame", loop)
        out.append(r)
    assert out[0]["move"] == out[1]["move"] == "c6a8"
    assert out[0]["chain"] and not out[1]["chain"]
    assert "PROVEN" in out[0]["status"]
