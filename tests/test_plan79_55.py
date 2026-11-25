"""PLAN 79.55 A RULE LEMMA OVER THE BITS OF A STATE, CLOSED BY LEAN'S `bv_decide` (SYSTEM_SPEC 11 "One lemma machinery, used twice", SYSTEM_SPEC 13 the decision of 2026-11-25; piece: the long-term
memory's statements; ledger rows 49, 56, 33). The gate of the WRITER (discovery/rule_lemma_bits.py), not of any standing:
  (1) the expressions the writer makes equal the evaluator's rows at the fixtures and at states of seeded games, for the line lemma (every binding the solver is given);
  (2) one binding of the line lemma, written by the writer, is closed by `bv_decide` (Lean's own run; the wall is printed);
  (3) a planted false lemma is refused by the solver, its counterexample is a board, and the evaluator confirms the board is a counterexample;
  (4) a statement that needs a hypothesis is false without it and the hypothesis the document names is written into the text;
  (5) the writer is simple where it must be (constants, sharing, a cycle refused) and names no game; the test is in the gate.
No proof is written by hand in this file: the only proof line is the `bv_decide` the writer emits.
"""
from __future__ import annotations

import copy
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DISC = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
HYPS = ROOT / "rules" / "lemmas" / "chess" / "bits"


def mods():
    from catspace.incubator.analyst.discovery import rule_lemma as RL
    from catspace.incubator.analyst.discovery import rule_lemma_bits as B
    from catspace.incubator.analyst.discovery import rule_lemma_certify as RC
    return RL, B, RC


needs_lean = pytest.mark.skipif(shutil.which("lake") is None or not (ROOT / "lean" / "ideas" / ".lake").exists(), reason="no Lean project with its packages here")

LINE = ("rook", "white", "d", "4", "d", "7")


def planted(RL):
    """A false lemma made in memory: a unit of the line-mover kind on a file attacks every other square of the file (blockers ignored)."""
    base = RL.load("chess", "lone_rook_attacks_its_file")
    doc = copy.deepcopy(base.doc())
    doc["name"] = "planted_rook_on_file"
    doc["says"] = "PLANTED FALSE: a rook on a file attacks every other square of the file."
    doc["left"] = {"relation": "rook_on_file", "places": 6}
    doc["delta"] = ["(signature rook_on_file kind side file rank file rank)",
                    "(<= (rook_on_file rook ?c ?f ?r1 ?f ?r2) (true (cell ?f ?r1 ?c rook)) (rank ?r2) (distinct ?r1 ?r2))"]
    doc["standing"] = {"is": "CONJECTURE"}
    return RL.parse(doc)


def test_the_expressions_equal_the_evaluators_rows():
    RL, B, RC = mods()
    lemma = RL.load("chess", "lone_rook_attacks_its_file")
    bits = B.Bits(lemma)
    bs = [b for b in bits.bindings() if bits.statement(b).settled is None]
    assert len(bs) == 896                                                   # 1 kind x 2 sides x 8 files x 8 x 7 places: the others have a constant false left side
    res = B.agree(bits, RC.corpus_states(lemma, games=1), bs, limit_states=120)
    print(f"OBS 79.55 test agree: states {res['states']}, comparisons {res['compared']}, disagreements {res['disagree']}, left true {res['left_true']}")
    assert res["states"] >= 100 and res["disagree"] == 0 and res["left_true"] > 0      # (the left side held at some states: the test is not vacuous)


def test_the_expressions_are_not_equal_to_a_wrong_evaluation():
    """Negative control: a wrong expression (the line's blockers ignored) disagrees with the evaluator at a state that has a blocker."""
    RL, B, RC = mods()
    lemma = RL.load("chess", "lone_rook_attacks_its_file")
    bits = B.Bits(lemma)
    b = (LINE, ())
    st = bits.statement(b)
    d = bits.g.dag
    state = RC.state_of_text("(cell d 4 white rook) (cell d 5 black bishop)")
    prog = bits.prog
    db = prog.run(state, wanted=("lone_on_file", "attack_by"))
    true_vars = {B.var_name(rel, args) for rel, args in state}
    assert d.eval(st.left, true_vars, {}) == (LINE in db.facts("lone_on_file")) is False        # a unit on the file: not alone
    assert d.eval(st.right, true_vars, {}) == (LINE in db.facts("attack_by")) is False         # the bishop blocks the line to d7
    wrong = d.eval(d.neg(st.right), true_vars, {})                                              # the planted wrong expression
    assert wrong != (LINE in db.facts("attack_by"))


@needs_lean
def test_one_binding_is_closed_by_bv_decide(tmp_path):
    RL, B, RC = mods()
    lemma = RL.load("chess", "lone_rook_attacks_its_file")
    bits = B.Bits(lemma)
    p = bits.write(tmp_path / "one.lean", [(LINE, ())])
    text = p.read_text()
    assert text.count("  bv_decide\n") == 1 and "sorry" not in text and "native_decide" not in text and "axiom" not in text
    r = B.run_lean(p, 300)
    print(f"OBS 79.55 test one binding: ok {r['ok']}, wall {r['secs']:.1f} s")
    assert r["ok"], r["out"][:500]


@needs_lean
def test_a_planted_false_lemma_is_refused_with_a_counterexample_that_is_a_board(tmp_path):
    RL, B, RC = mods()
    lemma = planted(RL)
    bits = B.Bits(lemma, [{"fluent": "cell", "key": [0, 1]}])
    b = (("rook", "white", "d", "1", "d", "3"), ())
    p = bits.write(tmp_path / "planted.lean", [b])
    r = B.run_lean(p, 300)
    assert not r["ok"] and not r["timeout"] and "counterexample" in r["out"]
    text = B.state_text(bits, B.counterexample(r["out"]))
    print(f"OBS 79.55 test planted false lemma: counterexample state {text}")
    state = RC.state_of_text(text)
    assert len([a for a in state if a[0] == "true.cell" and a[1][:2] == ("d", "2")]) <= 1    # at most one unit on the square the hypothesis names
    db = bits.prog.run(state, wanted=("rook_on_file", "attack_by"))
    row = ("rook", "white", "d", "1", "d", "3")
    assert row in db.facts("rook_on_file") and row not in db.facts("attack_by")                # the evaluator confirms: left holds, right does not


@needs_lean
def test_a_case_of_a_split_is_closed_and_carries_its_assumptions(tmp_path):
    """The cases lemma at one binding is too large whole; with the king's square and the moved unit taken as known (a case of a split) the same statement closes. The case is a theorem that carries the
    facts it assumes as hypotheses, and its expressions equal the evaluator's at a state that meets them."""
    RL, B, RC = mods()
    lemma = RL.load("chess", "exposed_by_cases")
    prog = RL.program(lemma)
    hyps = B.hypotheses_of(HYPS / "exposed_by_cases.json")
    spec = [["cell", "*", "*", "white", "king", False], ["cell", "e", "1", "white", "king", True],
            ["cell", "e", "2", "*", "*", False], ["cell", "e", "2", "white", "pawn", True]]
    assume = B.assume_of(prog, spec)
    assert assume["f_cell_e_1_white_king"] is True and assume["f_cell_a_1_white_king"] is False and assume["f_cell_e_2_white_pawn"] is True
    bits = B.Bits(lemma, hyps, assume, prog)
    b = (("e", "2", "e", "3"), ("white",))
    p = bits.write(tmp_path / "case.lean", [b])
    text = p.read_text()
    assert "CASE:" in text and "hasm0" in text and text.count("  bv_decide\n") == 1
    state = RC.state_of_text("(control white) (cell e 1 white king) (cell e 2 white pawn) (cell e 8 black king)")
    db = prog.run(state, wanted=("exposed", "exposed_by_cases"))
    true_vars = {B.var_name(rel, args) for rel, args in state}
    st = bits.statement(b)
    assert bits.g.dag.eval(st.left, true_vars, {}) == (("e", "2", "e", "3") in db.facts("exposed"))
    r = B.run_lean(p, 900)
    print(f"OBS 79.55 test case closed: ok {r['ok']}, wall {r['secs']:.1f} s, text {len(text)} bytes")
    assert r["ok"], r["out"][:500]


def test_the_closed_lemmas_have_their_record_and_no_document_changed():
    """The two lemmas all of whose bindings the solver was given closed have a record beside the document, read back with the file's hash; the standing of each document is as it was (PROVED by the
    library, CERTIFIED, CERTIFIED), and the cases lemma has no record."""
    RL, B, RC = mods()
    for name in ("lone_rook_attacks_its_file", "candidate_is_mover_unit"):
        lemma = RL.load("chess", name)
        word, detail = B.standing(lemma)
        print(f"OBS 79.55 test standing {name}: {word} ({detail})")
        assert word == "SOLVER-PROVED", detail
    assert B.standing(RL.load("chess", "exposed_by_cases"))[0] == "NONE"
    for name, is_ in (("candidate_is_mover_unit", "PROVED"), ("exposed_by_cases", "CERTIFIED"), ("lone_rook_attacks_its_file", "CERTIFIED")):
        assert RL.load("chess", name).standing["is"] == is_


def test_the_hypotheses_a_statement_needs_are_written_and_listed():
    RL, B, RC = mods()
    lemma = RL.load("chess", "exposed_by_cases")
    hyps = B.hypotheses_of(HYPS / "exposed_by_cases.json")
    assert {h["fluent"] for h in hyps} == {"cell", "control"} and all("why" in h for h in hyps)
    bits = B.Bits(lemma, hyps)
    b = (("e", "1", "e", "2"), ("white",))
    st = bits.statement(b)
    assert st.settled is None
    text = bits.theorem(st, bits.label_of(b))
    assert "hy0x1" in text and "hy1x1" in text                                                  # both groups occur in this binding
    assert "(!(" in text and "bv_decide" in text
    header = bits.header(1)
    assert "HYPOTHESES" in header and "at most one" not in header.lower() or '"fluent"' in header
    assert '"key"' in header and "TRUST" in header


def test_constants_sharing_and_a_cycle():
    RL, B, RC = mods()
    d = B.Dag()
    a, b = d.var("xa"), d.var("xb")
    assert d.conj([a, B.TRUE]) == a and d.conj([a, B.FALSE]) == B.FALSE and d.disj([a, d.neg(a)]) == B.TRUE
    assert d.conj([a, b]) == d.conj([b, a]) and d.conj([a, d.conj([b, a])]) == d.conj([a, b])
    assert d.neg(d.neg(a)) == a and d.conj([a, d.neg(a)]) == B.FALSE
    base = RL.load("chess", "lone_rook_attacks_its_file")
    doc = copy.deepcopy(base.doc())
    doc["name"] = "cyclic_toy"
    doc["left"] = {"relation": "ring_a", "places": 1}
    doc["right"] = {"relation": "ring_b", "places": 1}
    doc["delta"] = ["(signature ring_a file)", "(signature ring_b file)",
                    "(<= (ring_a ?f) (ring_b ?f) (true (cell ?f 1 white rook)))", "(<= (ring_b ?f) (ring_a ?f) (true (cell ?f 2 white rook)))"]
    doc["standing"] = {"is": "CONJECTURE"}
    with pytest.raises(B.Untranslatable, match="cycle"):                                        # a positive cycle is stratified for the reader; among ground atoms the writer refuses it
        B.Bits(RL.parse(doc)).statement((("a",), ()))


def test_the_module_names_no_game_the_hypotheses_are_a_document_and_the_gate_has_the_test():
    src = (DISC / "rule_lemma_bits.py").read_text()
    assert not re.search(r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|check|mate|stalemate|uci|fen|san|chess\w*|[a-h][1-8])\b", src, re.I)
    assert (HYPS / "exposed_by_cases.json").is_file()
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert "test_plan79_55" in gate
