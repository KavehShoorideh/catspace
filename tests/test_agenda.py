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
    assert any(l.startswith("THE CHAIN") for l in out[0]["trace"]) and not any(l.startswith("THE CHAIN") for l in out[1]["trace"])
    assert "PROVEN" in out[0]["status"] and "decision" in out[0]


# ---- PHASE 2: the decision read off the chain (ANALYST_PLAN 40.4) ------------------------------------------------------------------------

def test_move_30_is_decided_by_the_chain_and_names_the_constraints():
    # Whether the queens stay on is the TRANSITION TABLE's reading, not a rule: the population cell (Lichess bands pooled, behind -4..-2) says a
    # queen trade changes the mover's result by about +0.01 (neutral), so the chain may trade; the player's own cell says -0.09 (keep them). Kaveh's
    # knob (2026-10-15 night). What the test holds fixed: Qc4 is never chosen (the c-file discovery), and the reason names the Re2 / e8 constraint.
    d = LG.decide(MOVE30)
    print("OBS", d["lines"][-1][:220])
    assert d["san"] != "Qc4"
    assert "Re2 must not be deflectable" in d["lines"][-1]                 # the reason names the Re2 / e8 constraint it meets
    assert any("Re2 must not be deflectable" in c["text"] for c in d["constraints"]) and any("Qg4" in c["text"] for c in d["constraints"])
    names = {c["san"]: c for c in d["candidates"]}
    assert "Qxg6" in names and "Qc4" in names                               # both were checked, neither is preferred
    assert (names["Qxg6"]["lo"], names["Qxg6"]["hi"]) <= (d["interval"][0], d["interval"][1]) or names["Qxg6"]["lo"] <= d["interval"][0]


def chess_capture_of_queen(fen: str, uci: str) -> bool:
    import chess
    b = chess.Board(fen)
    p = b.piece_at(chess.parse_square(uci[2:4]))
    return p is not None and p.piece_type == chess.QUEEN


def test_nxc6_is_played_and_proven():
    d = LG.decide(NXC6)
    print("OBS", d["lines"][-1][:200])
    assert d["move"] == "e5c6" and d["status"].startswith("PROVEN") and d["interval"][0] >= 5
    assert any("Qd8 can be won" in o["text"] for o in d["opportunities"]) or d["opportunities"]


def test_the_three_trace_positions_keep_their_answers_on_the_chain_decision():
    for fen, want in (("5k1r/1p1n1p2/p1qb1p1p/4p3/2rP2PP/2PQ1PB1/PP2NK2/1R5R b - - 0 1", "c4a4"),
                      ("r5k1/2p1q2p/2Qp2p1/1rpPpp2/2P1P3/P4P2/6PP/4R1K1 w - - 0 27", "c6a8"),
                      ("1r2r3/1kp1q2p/4Q3/p2P4/1P1P4/P1p2P2/2P3PP/2KR3R b - - 0 30", "e7g5")):
        L = LG.new(fen, 1000)
        LG.load_defence_frame(L)
        LG.load_transitions(L)
        r = L.trace(4, "defined", "frame", True, LG.opening_bonus(fen))
        print(f"OBS {fen.split()[0][:20]}...: {r['san']} {r['interval']} {r['status']}")
        assert r["move"] == want
        assert any(l.startswith("CONSTRAINTS") for l in r["trace"]) and any(l.startswith("DECISION:") for l in r["trace"])


def test_nothing_threatened_and_nothing_to_win_plays_the_opening_default():
    import chess
    d = LG.decide(chess.STARTING_FEN)
    assert d["default"] and d["move"] in ("b1c3", "g1f3", "b1c3", "e2e4", "d2d4") or d["default"]
    bonus = LG.opening_bonus(chess.STARTING_FEN)
    top = max(bonus.values())
    assert bonus[d["move"]] == top                          # the tie-break principles' best moves first
    assert "the opening default" in d["lines"][-1]


def test_the_script_loop_is_still_there():
    L = LG.new("r5k1/2p1q2p/2Qp2p1/1rpPpp2/2P1P3/P4P2/6PP/4R1K1 w - - 0 27", 1000)
    LG.load_defence_frame(L)
    r = L.trace(4, "defined", "frame", False)
    assert r["move"] == "c6a8" and any(l.startswith("DECISION:") for l in r["trace"])


# ---- PHASE 3: lemmas by chunking (ANALYST_PLAN 40.3) -------------------------------------------------------------------------------------

def _learn(fen: str):
    L = LG.new(fen, 5000)
    LG.load_transitions(L)
    L.set_lemmas("", 3, 5, True)
    L.chain(120)
    return L.lemma_report()


def _store(tmp_path, reports, threshold=5):
    from catspace.incubator.analyst.discovery import chain_lemmas as CL
    p = str(tmp_path / "lemmas.json")
    CL.merge(reports, p, threshold)
    return p


def test_a_shape_is_canonical_under_mirror_and_translation():
    import chess
    fen = MOVE30
    base = LG.new(fen, 100).shape_for("SAFE", "Qg4", 1)
    assert base and base.startswith("SAFE|")
    flipped = chess.Board(fen).transform(chess.flip_horizontal)              # a left-right mirror
    assert LG.new(flipped.fen(), 100).shape_for("SAFE", "Qb4", 1) == base
    swapped = chess.Board(fen).mirror()                                      # the colours swapped: the actor and the pieces change sides together
    assert LG.new(swapped.fen(), 100).shape_for("SAFE", "Qg5", 0) == base
    # a translation of a position with nothing on the edge it moves away from: the same pieces, the same relations
    b = chess.Board(None); b.set_piece_at(chess.D4, chess.Piece(chess.ROOK, chess.WHITE)); b.set_piece_at(chess.D6, chess.Piece(chess.QUEEN, chess.BLACK))
    b.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE)); b.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK)); b.turn = chess.WHITE
    c = chess.Board(None); c.set_piece_at(chess.E3, chess.Piece(chess.ROOK, chess.WHITE)); c.set_piece_at(chess.E5, chess.Piece(chess.QUEEN, chess.BLACK))
    c.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE)); c.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK)); c.turn = chess.WHITE
    assert LG.new(b.fen(), 100).shape_for("SAFE", "Rd4", 1) == LG.new(c.fen(), 100).shape_for("SAFE", "Re3", 1)


def test_a_promoted_lemma_answers_in_one_step_and_the_chain_is_skipped(tmp_path):
    rep = _learn(NXC6)
    store = _store(tmp_path, [rep] * 5, 5)
    import json
    promoted = [e for e in json.load(open(store))["lemmas"].values() if e["promoted"]]
    assert promoted and all(e["outcome"] and e["id"] for e in promoted)
    L0 = LG.new(NXC6, 5000); LG.load_transitions(L0)
    base = L0.chain(120)
    L = LG.new(NXC6, 5000); LG.load_transitions(L)
    L.set_lemmas(open(store).read(), 9, 1_000_000, False)
    r = L.chain(120)
    rep2 = L.lemma_report()
    print(f"OBS the Nxc6 chain: {base['questions']} questions without lemmas, {r['questions']} with {rep2['hits']} answered by a lemma ({len(promoted)} promoted)")
    assert rep2["hits"] > 0 and r["questions"] < base["questions"]
    assert any("by lemma" in c["answer"] for c in r["agenda"]["chains"])
    # a question answered by a lemma has no chain below it
    for c in r["agenda"]["chains"]:
        if "by lemma" in c["answer"]:
            assert not any(k["parent"] == c["q"] for k in r["agenda"]["chains"])


def test_a_contradiction_demotes_the_lemma_and_is_kept(tmp_path):
    rep = _learn(NXC6)
    # corrupt one SAFE premise: promote it with an outcome the chain will not give
    pk = next(p for p in rep["observations"] if p.startswith("SAFE|"))
    wrong = {"observations": {pk: {"alone => 7..7": {"count": 6, "value": [7, 7], "evidence": ["x"]}}}}
    store = _store(tmp_path, [wrong], 5)
    L = LG.new(NXC6, 5000); LG.load_transitions(L)
    L.set_lemmas(open(store).read(), 9, 1, False)                            # every match is a held-out check: run and compared
    L.chain(120)
    rep2 = L.lemma_report()
    assert rep2["contradictions"] and rep2["contradictions"][0]["premise"] == pk and rep2["contradictions"][0]["expected"] == "alone => 7..7"
    from catspace.incubator.analyst.discovery import chain_lemmas as CL
    CL.merge([rep2], store, 5)
    e = CL.load(store)["lemmas"][pk]
    assert e["demoted"] and not e["promoted"] and e["contradictions"]
    L2 = LG.new(NXC6, 5000); L2.set_lemmas(open(store).read(), 9, 1, False)
    assert L2.lemma_report()["promoted_loaded"] == 0


# ---- PHASE 4: the proposer against puzzle surprises (ANALYST_PLAN 40.5) ----------------------------------------------------------------

def _puz(answer="a1a2", generated=("a1a2",), rows=None, correct=False, value=3, themes=("short",), hits=0, plies=2, move="b1b2"):
    rows = [{"uci": u, "lo": lo, "hi": hi, "depth": d} for u, lo, hi, d in (rows or [])]
    return dict(id="X", fen="", answer=answer, themes=list(themes), value=value,
                ledger=dict(generated=list(generated), rows=rows, correct=correct, move=move, lemma={"hits": hits}, plies=plies, tags={}))


def test_a_surprise_is_classified_into_one_of_the_four_classes():
    from catspace.incubator.analyst.discovery import proposer as P
    # the event was missing: no condition made the puzzle's move a candidate
    r = P.classify(_puz(answer="a1a2", generated=("c1c2",), rows=[("c1c2", 0, 0, 1)]))
    assert r["surprise"] and r["a"] and r["klass"] == "event missing"
    # the question was never opened: generated, but the cap / the budget cut it
    r = P.classify(_puz(answer="a1a2", generated=("a1a2", "c1c2"), rows=[("c1c2", 0, 0, 1)]))
    assert r["surprise"] and r["klass"] == "question never opened"
    # the answer was wrong: evaluated, its interval stops below the line's gain
    r = P.classify(_puz(answer="a1a2", generated=("a1a2",), rows=[("a1a2", 0, 1, 1)], value=3, hits=2))
    assert r["surprise"] and r["b"] and r["klass"] == "answer wrong" and r["lemma_involved"] and not r["no_lemma"]
    # no surprise: evaluated, its value reaches the line's, a short chain, solved
    r = P.classify(_puz(answer="a1a2", rows=[("a1a2", 3, 3, 1)], correct=True, value=3))
    assert not r["surprise"] and r["klass"] == "no surprise" and r["no_lemma"]
    # (c) alone: our chain for it is more than one link longer than the line's
    r = P.classify(_puz(answer="a1a2", rows=[("a1a2", 3, 3, 6)], correct=True, value=3, plies=2))
    assert r["surprise"] and r["c"] and not r["a"] and not r["b"]


def test_a_proposed_condition_is_kept_only_if_it_survives_the_fresh_check():
    from catspace.incubator.analyst.discovery import proposer as P
    base = {"p1": False, "p2": True, "p3": False}
    assert P.compare(base, {"p1": True, "p2": True, "p3": False})["kept"]                       # fixes a fresh surprise, no new miss
    assert not P.compare(base, {"p1": True, "p2": False, "p3": False})["kept"]                  # fixes one but breaks another
    assert not P.compare(base, {"p1": False, "p2": True, "p3": False})["kept"]                  # fixes nothing
    c = P.compare(base, {"p1": True, "p2": False, "p3": False})
    assert c["fixed"] == ["p1"] and c["new_misses"] == ["p2"] and (c["base"], c["new"]) == (1, 1)


def test_a_rule_adds_the_moves_after_which_a_word_rose_or_fell():
    from catspace.incubator.analyst.discovery import proposer as P
    fen = "4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1"
    caps = P.rule_moves(fen, [{"term": "material_them", "sign": -1}])
    assert "d1d5" in caps and len(caps) <= P.MAX_EXTRA                                         # their material fell: the capture
    assert P.rule_moves(fen, [{"term": "material_them", "sign": 1}]) == []


def test_the_priority_proposal_is_learned_by_the_strict_pairwise_ranking():
    from catspace.incubator.analyst.discovery import proposer as P
    res = [_puz(answer="a1a2", generated=("a1a2", "c1c2", "d1d2"), rows=[("c1c2", 0, 0, 1), ("d1d2", 0, 0, 1)])]
    res[0]["ledger"]["tags"] = {"a1a2": ["check"], "c1c2": ["touch"], "d1d2": ["touch", "capture"]}
    res[0]["id"] = "X"
    recs = [dict(id="X", answer="a1a2", klass="question never opened", surprise=True)]
    p = P.propose_stakes(res, recs, log=lambda *_: None)
    assert p and p["ranking"][0][0] == "check" and p["tag_bonus"]["check"] > 0
    assert P.propose_stakes(res, [dict(id="X", answer="a1a2", klass="no surprise", surprise=False)], log=lambda *_: None) is None
