"""PLAN 79.44 THE LEMMA MACHINERY, FIRST STEP (SYSTEM_SPEC 11 "One lemma machinery, used twice"; pieces: rules, and the long-term memory's statements; ledger rows 49, 44).
A lemma about the rules (or about play) is a document beside the description (rules/lemmas/<game>/), of two forms (SAME, IMPLIES), with a delta of clauses in the description's language,
a condition in the learned lemmas' literal form, an origin and a standing; certified by the reader's evaluator on fixtures and seeded games; its Lean obligation written by the existing
generator; loadable as a rep `Statement`. THIS file is the gate of it:
  (1) the form: the documents of the first game load; the form refuses a delta that retracts, an unknown relation, a wrong place count, a clause that is unsafe or not stratified, a
      relation the description defines, a word of the language, a condition with an unbound negation; the delta composes onto the description;
  (2) the certificate: each lemma of the first game has a fresh certificate (the condition held, rows were compared, no disagreement); a fresh small run agrees; the twelve certificates of the
      old bindings are still twelve and fresh and the lemma certificates are counted apart;
  (3) a deliberately wrong lemma is not certified and its disagreements are printed; a lemma whose condition never holds is vacuous and not certified; a certificate goes stale when the delta changes;
  (4) the Lean obligations: written as the generator writes them (the file on disk is the regenerated text), OPEN where no proof is written, closed for the lemma about play, which the library
      proves; the index lists them so; the standing PROVED is refused when its evidence is missing;
  (5) the statements: each lemma is the JSON of a `Statement` (Proved or Conjecture); with RULE_LEMMA_RUST set, the crate's own reader reads them (rust/engine/rep/tests/rule_lemma_statements.rs);
  (6) the machinery names no game and no relation of any description; nothing under rules/ that is a description, no reader and no source of the engine's crates changed; the test is in the gate.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BEFORE = "4f7e4828"                      # the commit before this plan's code
GAME = "chess"
MODULES = [ROOT / "catspace" / "incubator" / "analyst" / "discovery" / n for n in ("rule_lemma.py", "rule_lemma_certify.py", "rule_lemma_lean.py")]
NAMES = ("exposed_by_cases", "candidate_is_mover_unit", "lone_rook_attacks_its_file")


def RL():
    from catspace.incubator.analyst.discovery import rule_lemma as m
    return m


def RC():
    from catspace.incubator.analyst.discovery import rule_lemma_certify as m
    return m


def RLEAN():
    from catspace.incubator.analyst.discovery import rule_lemma_lean as m
    return m


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def have_before() -> bool:
    return git("cat-file", "-e", f"{BEFORE}^{{commit}}").returncode == 0


def lemma(name: str):
    return RL().load(GAME, name)


def doc_of(name: str) -> dict:
    return copy.deepcopy(lemma(name).doc())


def problems(doc: dict) -> list:
    L = lemma("exposed_by_cases")
    return RL().problems(doc, RL().description_text(L))


def made_states(note_part: str, n: int = 1) -> list:
    """The fixture lines whose note holds `note_part`."""
    lines = (RL().lemma_dir(GAME) / "states.txt").read_text().splitlines()
    out = [lines[i + 1] for i, ln in enumerate(lines) if ln.startswith(";") and note_part in ln and i + 1 < len(lines)]
    assert len(out) >= 1, note_part
    return out[:n]


# ------------------------------------------------------------------------------------------------------------------------------------------ (1) the form

def test_the_documents_of_the_first_game_load_from_a_folder_beside_the_descriptions():
    lemmas = RL().load_all(GAME)
    assert sorted(l.name for l in lemmas) == sorted(NAMES)
    folder = RL().lemma_dir(GAME)
    assert folder == ROOT / "rules" / "lemmas" / GAME and (ROOT / "rules" / "chess.gdl").is_file()
    assert "learned" not in folder.parts and "derived" not in folder.parts and "data" not in folder.parts
    forms = {l.name: l.form for l in lemmas}
    assert forms == {"exposed_by_cases": "SAME", "candidate_is_mover_unit": "IMPLIES", "lone_rook_attacks_its_file": "IMPLIES"}
    assert all(l.origin == "HAND" for l in lemmas)
    for l in lemmas:
        assert list(l.doc()) == list(RL().KEYS) and l.path.parent == folder
        print(f"OBS 79.44 A: lemma {l.name} ({l.form}, origin {l.origin}, claims {l.standing['is']}): {l.left['relation']}/{l.left['places']} against {l.right['relation']}/{l.right['places']}, {len(l.delta)} clauses of delta, {len(l.condition)} literals of condition")


def test_the_delta_composes_onto_the_description_and_the_program_reads():
    for l in RL().load_all(GAME):
        base, text = RL().description_text(l), RL().composed_text(l)
        from catspace.incubator.analyst.discovery import gdl as G
        assert len(G.parse(text)) == len(G.parse(base)) + len(l.delta)
        prog = RL().program(l)
        assert prog.typecheck() == []
        for end in (l.left, l.right):
            assert len(prog.signature(end["relation"])) == end["places"]


def test_the_form_refuses_a_delta_that_retracts():
    d = doc_of("exposed_by_cases")
    d["delta"] = d["delta"] + ["(retract (opponent white black))"]
    bad = problems(d)
    print(f"OBS 79.44 F refusal: a retraction -- {bad}")
    assert bad and "additive only" in bad[0]


def test_the_form_refuses_a_relation_the_description_does_not_have_and_a_wrong_place_count():
    d = doc_of("exposed_by_cases")
    d["right"] = {"relation": "no_such_relation", "places": 4}
    assert any("not a relation of the description" in b for b in problems(d))
    d = doc_of("exposed_by_cases")
    d["right"]["places"] = 3
    assert any("has 4 places, the document says 3" in b for b in problems(d))
    d = doc_of("exposed_by_cases")
    d["delta"] = [c for c in d["delta"] if not c.startswith("(signature exposed_by_cases")]
    bad = problems(d)
    print(f"OBS 79.44 F refusal: a relation with no signature -- {bad[:2]}")
    assert bad
    d = doc_of("exposed_by_cases")
    d["condition"] = [{"relation": "no_such_relation", "args": ["?c"], "neg": False}]
    assert any("is not in the description" in b for b in problems(d))


def test_the_form_refuses_an_unsafe_clause_and_a_clause_that_is_not_stratified():
    d = doc_of("exposed_by_cases")
    d["delta"] = d["delta"] + ["(signature unsafe_one side)", "(<= (unsafe_one ?c) (not (opponent ?c ?o)))"]
    bad = problems(d)
    print(f"OBS 79.44 F refusal: an unsafe clause -- {bad[0][:160]}")
    assert bad and "unsafe" in bad[0]
    d = doc_of("exposed_by_cases")
    d["delta"] = d["delta"] + ["(signature odd_one side)", "(signature even_one side)", "(<= (odd_one ?c) (role ?c) (not (even_one ?c)))", "(<= (even_one ?c) (role ?c) (not (odd_one ?c)))"]
    bad = problems(d)
    print(f"OBS 79.44 F refusal: a delta not stratified -- {bad[0][:160]}")
    assert bad and "not stratified" in bad[0]


def test_the_form_refuses_to_redefine_a_relation_of_the_description_and_a_word_of_the_language():
    d = doc_of("exposed_by_cases")
    d["delta"] = d["delta"] + ["(<= (exposed ?f1 ?r1 ?f2 ?r2) (candidate ?c ?f1 ?r1 ?f2 ?r2))"]
    assert any("the delta defines relations the description defines" in b and "exposed" in b for b in problems(d))
    d = doc_of("exposed_by_cases")
    d["delta"] = d["delta"] + ["(<= (goal ?c 50) (role ?c))"]
    assert any("words of the language" in b for b in problems(d))
    d = doc_of("exposed_by_cases")
    d["delta"] = d["delta"][:4] + ["(<= (somewhere_else ?f1 ?r1 ?f2 ?r2) (candidate ?c ?f1 ?r1 ?f2 ?r2))"]
    assert any("defines neither end" in b for b in problems(d)) or any("not a relation" in b or "type check" in b or "sort test" in b for b in problems(d))


def test_the_form_refuses_a_condition_with_an_unbound_negation_and_a_document_with_other_keys():
    d = doc_of("exposed_by_cases")
    d["condition"] = [{"relation": "check", "args": ["?c"], "neg": True}]
    assert any("a condition needs a positive literal" in b for b in problems(d))
    d["condition"] = [{"relation": "control", "args": ["?c"], "neg": False}, {"relation": "check", "args": ["?z"], "neg": True}]
    assert any("which no positive literal binds" in b for b in problems(d))
    d = doc_of("exposed_by_cases")
    d["form"] = "EQUAL"
    assert any("the form is one of" in b for b in problems(d))
    d = doc_of("exposed_by_cases")
    d["extra"] = 1
    assert "the keys are" in problems(d)[0]
    d = doc_of("exposed_by_cases")
    d["condition"] = [{"relation": "control", "args": ["nobody"], "neg": False}]
    assert any("no member of the sort" in b for b in problems(d))


def test_the_condition_reads_the_state_through_the_description():
    L = lemma("exposed_by_cases")
    prog = RL().program(L)
    C = RC()
    pos = C.state_of_text(made_states("a pinned rook that may slide")[0])
    db = prog.run(pos, wanted=C.wanted_of(L, prog))
    assert RL().condition_rows(L, db, prog) == [{"?c": "white"}]
    check = C.state_of_text(made_states("a rule corner")[0])
    L2 = lemma("exposed_by_cases")
    st = C.state_of_text("(cell e 1 white king) (cell e 8 black rook) (cell a 8 black king) (control white) (halfmove 0)")
    db2 = prog.run(st, wanted=C.wanted_of(L2, prog))
    assert db2.facts("check") and RL().condition_rows(L2, db2, prog) == []


# ------------------------------------------------------------------------------------------------------------------------------------------ (2) the certificate

def test_the_twelve_old_certificates_are_still_twelve_and_fresh_and_the_lemma_ones_are_apart():
    from catspace.incubator.analyst.discovery import certify as CF
    st = CF.status()
    assert len(CF.BINDINGS) == 12 and all(v == "fresh" for v in st.values()), st
    assert not any("lemma" in n and "chess" in n for n in CF.BINDINGS)
    print(f"OBS 79.44 B: the old bindings {len(CF.BINDINGS)}, all fresh; the lemma certificates are counted apart")


@pytest.mark.parametrize("name", NAMES)
def test_each_lemma_has_a_fresh_certificate_that_held_and_compared_rows_and_found_nothing(name):
    L = lemma(name)
    c = RC().load(L)
    assert c is not None, "run: python -m catspace.incubator.analyst.discovery.rule_lemma_certify run chess --jobs 8 --adopt"
    print(RC().obs(c))
    assert RC().status(L) == "fresh", RC().status(L)
    assert c["states_held"] > 0 and c["rows_left"] > 0 and c["rows_compared"] > 0 and c["disagreements"] == 0 and not c["vacuous"]
    assert c["origin"] == "HAND" and c["form"] == L.form
    assert L.standing.get("key", c["key"]) == c["key"] or L.standing["is"] == "PROVED"


def test_a_fresh_small_run_of_each_lemma_agrees_with_the_oracle():
    fixtures = made_states("en passant opens a diagonal") + made_states("en passant takes both units from the king's row", 2) + made_states("a unit alone between the king and a rook") + \
        made_states("a pinned rook") + made_states("a pawn pinned on a diagonal") + made_states("the king steps next to a king") + made_states("a lone rook on its file", 2)
    for name in NAMES:
        L = lemma(name)
        c = RC().certify(L, jobs=1, games=0, fixtures=fixtures)
        print(RC().obs(c))
        assert c["disagreements"] == 0 and c["states_held"] > 0 and c["rows_left"] > 0 and not c["vacuous"], (name, c["examples"])


def test_the_small_run_over_two_workers_gives_the_one_run_s_counts():
    L = lemma("exposed_by_cases")
    fx = made_states("a rule corner", 3) + made_states("a unit alone between the king and a rook")
    one = RC().certify(L, jobs=1, games=0, fixtures=fx)
    two = RC().certify(L, jobs=2, games=0, fixtures=fx)
    keys = ("positions", "states_held", "rows_left", "rows_right", "rows_compared", "disagreements")
    assert {k: one[k] for k in keys} == {k: two[k] for k in keys}


# ------------------------------------------------------------------------------------------------------------------------------------------ (3) a wrong lemma, a vacuous one, a stale one

CASES_FIXTURES = ["en passant opens a diagonal", "en passant takes both units from the king's row", "a unit alone between the king and a rook", "a pawn pinned on a diagonal",
                  "the king steps next to a king", "castling with the rook attacked"]


def corpus() -> list:
    return [s for part in CASES_FIXTURES for s in made_states(part, 2)]


@pytest.mark.parametrize("clause,why", [("(ep_victim ?c ?f2 ?r2 ?vf ?vr) (walled", "the case where the captured unit leaves the line"),
                                         ("(true (cell ?f1 ?r1 ?c king)) (opponent ?c ?o) (attacked", "the case of the king itself"),
                                         ("(walled ?c ?d ?f1 ?r1 ?sf ?sr) (not (on_wall", "the case of the unit held on a line")])
def test_a_lemma_with_a_case_left_out_is_not_certified_and_its_disagreements_are_printed(clause, why):
    d = doc_of("exposed_by_cases")
    d["name"] = "wrong_lemma"
    d["delta"] = [c for c in d["delta"] if clause not in c]
    assert len(d["delta"]) == len(doc_of("exposed_by_cases")["delta"]) - 1
    L = RL().parse(d)
    c = RC().certify(L, jobs=1, games=0, fixtures=corpus())
    print(RC().obs(c))
    for e in c["examples"][:2]:
        print(f"   OBS 79.44 F the wrong lemma ({why}) disagrees at {e['at']}: {e['state'][:90]}... rows {e['rows'][:3]}")
    assert c["disagreements"] > 0 and c["examples"] and c["examples"][0]["rows"]
    assert c["rows_left"] > c["rows_right"]                    # (the case left out loses rows of exposed: the lemma says too little)


def test_an_invented_case_is_not_certified_either():
    d = doc_of("exposed_by_cases")
    d["name"] = "wrong_lemma"
    d["delta"] = d["delta"] + ["(<= (exposed_by_cases ?f1 ?r1 ?f2 ?r2) (candidate ?c ?f1 ?r1 ?f2 ?r2) (true (cell ?f1 ?r1 ?c pawn)))"]
    c = RC().certify(RL().parse(d), jobs=1, games=0, fixtures=corpus())
    print(RC().obs(c))
    assert c["disagreements"] > 0 and c["rows_right"] > c["rows_left"]


def test_a_lemma_with_a_wrong_inclusion_is_not_certified():
    d = doc_of("candidate_is_mover_unit")
    d["name"] = "wrong_inclusion"
    d["standing"] = {"is": "CONJECTURE"}
    d["lean"] = {"module": d["lean"]["module"], "namespace": d["lean"]["namespace"], "position": d["lean"]["position"]}
    d["delta"] = ["(signature mover_unit_move side file rank file rank)", "(<= (mover_unit_move ?c ?f1 ?r1 ?f2 ?r2) (true (control ?c)) (true (cell ?f1 ?r1 ?c pawn)) (square ?f2 ?r2))"]
    c = RC().certify(RL().parse(d), jobs=1, games=0, fixtures=corpus())
    print(RC().obs(c))
    assert c["disagreements"] > 0


def test_a_lemma_whose_condition_never_holds_is_vacuous_and_not_certified():
    d = doc_of("exposed_by_cases")
    d["name"] = "vacuous_lemma"
    d["condition"] = [{"relation": "control", "args": ["white"], "neg": False}, {"relation": "control", "args": ["black"], "neg": False}]
    L = RL().parse(d)
    c = RC().certify(L, jobs=1, games=0, fixtures=corpus())
    print(RC().obs(c))
    assert c["states_held"] == 0 and c["vacuous"] and c["disagreements"] == 0


def test_a_certificate_goes_stale_when_the_delta_the_condition_or_the_fixtures_change(tmp_path, monkeypatch):
    rl, rc = RL(), RC()
    fx = made_states("a unit alone between the king and a rook")
    more = made_states("a pinned rook")
    d = doc_of("exposed_by_cases")
    monkeypatch.setattr(rl, "LEMMAS", tmp_path)
    folder = tmp_path / GAME
    folder.mkdir()
    (folder / "states.txt").write_text("\n".join(fx) + "\n")
    L = rl.parse(d)
    rl.write(L)
    L = rl.load(GAME, L.name)
    cert = rc.certify(L, jobs=1, games=0)
    rc.write(L, cert)
    assert rc.status(L) == "fresh"
    assert rc.adopt(L, cert) is not None and rl.load(GAME, L.name).standing == {"is": "CERTIFIED", "key": cert["key"]}
    assert rc.standing(rl.load(GAME, L.name))[0] == "CERTIFIED"
    # (1) the delta changes: a clause more
    d2 = copy.deepcopy(rl.load(GAME, L.name).doc())
    d2["delta"] = d2["delta"] + ["(signature spare file rank)", "(<= (spare ?f ?r) (square ?f ?r))"]
    rl.write(rl.parse(d2))
    L2 = rl.load(GAME, L.name)
    assert rc.status(L2) == "stale" and rc.standing(L2)[0] == "CONJECTURE"
    print(f"OBS 79.44 F: after a clause more in the delta the certificate is {rc.status(L2)} and the lemma's standing {rc.standing(L2)}")
    # (2) the condition changes
    d3 = copy.deepcopy(L.doc())
    d3["condition"] = d3["condition"][:1]
    rl.write(rl.parse(d3))
    assert rc.status(rl.load(GAME, L.name)) == "stale"
    # (3) the fixtures change
    rl.write(L)
    assert rc.status(rl.load(GAME, L.name)) == "fresh"
    (folder / "states.txt").write_text("\n".join(fx + more) + "\n")
    assert rc.status(rl.load(GAME, L.name)) == "stale"
    # a certificate that found a disagreement is failing, and a CERTIFIED claim on it is not held
    bad = dict(cert, disagreements=1)
    rc.write(L, bad)
    assert rc.status(rl.load(GAME, L.name)) == "failing"


# ------------------------------------------------------------------------------------------------------------------------------------------ (4) the Lean obligations

def test_the_obligation_on_disk_is_the_regenerated_text_and_follows_the_generator_s_conventions():
    for name in NAMES:
        L = lemma(name)
        text = RLEAN().obligation(L)
        assert RLEAN().path_of(L).read_text() == text, f"{name}: regenerate with rule_lemma_lean write {GAME}"
        assert "AUTO-GENERATED" in text and f"namespace Ideas.Mined.rule_lemma_{GAME}_{name}" in text and "inductive " in text
        assert f"theorem {name}_{L.form.lower()}" in text
        assert RLEAN().path_of(L).parent == ROOT / "lean" / "ideas" / "Ideas" / "Mined"
        proved = bool(L.lean.get("proof"))
        body = re.sub(r"/-.*?-/", "", text, flags=re.S)
        assert ("sorry" not in body) == proved
        print(f"OBS 79.44 C: obligation of {name}: {len(text.splitlines())} lines, {'closed by the library' if proved else 'OPEN'}")
    same = RLEAN().obligation(lemma("exposed_by_cases"))
    assert "↔" in same and "(hcond : ∃ (c : Chess.Color), p.turn = c ∧ ¬ (check p c))" in same
    imp = RLEAN().obligation(lemma("candidate_is_mover_unit"))
    assert "candidate p a1 a2 a3 a4 a5 → moverUnitMove p a1 a2 a3 a4 a5" in imp and "Ideas.LegalEquiv.candidate_unit" in imp


def test_the_index_lists_the_open_obligations_as_open_and_the_proved_one_as_proved():
    text = (ROOT / "docs" / "LEAN_INDEX.md").read_text()
    assert "**OPEN theorem `Ideas.Mined.rule_lemma_chess_exposed_by_cases.exposed_by_cases_same`**" in text
    assert "**OPEN theorem `Ideas.Mined.rule_lemma_chess_lone_rook_attacks_its_file.lone_rook_attacks_its_file_implies`**" in text
    assert "- **theorem `Ideas.Mined.rule_lemma_chess_candidate_is_mover_unit.candidate_is_mover_unit_implies`**" in text
    lean = (ROOT / "lean" / "ideas" / "Ideas.lean").read_text()
    assert "import Ideas.Mined.rule_lemma_chess_candidate_is_mover_unit" in lean and "rule_lemma_chess_exposed_by_cases" not in lean       # (what is proved is built; an OPEN statement is not)


def test_the_standing_of_each_lemma_is_what_its_evidence_holds():
    got = {n: RC().standing(lemma(n)) for n in NAMES}
    print(f"OBS 79.44 E: standings {got}")
    assert got["exposed_by_cases"][0] == "CERTIFIED" and got["lone_rook_attacks_its_file"][0] == "CERTIFIED"
    assert got["candidate_is_mover_unit"] == ("PROVED", "Ideas.Mined.rule_lemma_chess_candidate_is_mover_unit.candidate_is_mover_unit_implies")


def test_a_proved_claim_without_its_evidence_is_held_as_a_conjecture():
    rl = RL()
    L = lemma("exposed_by_cases")          # (an OPEN obligation: no proof written)
    d = copy.deepcopy(L.doc())
    d["standing"] = {"is": "PROVED", "theorem": "Ideas.Mined.rule_lemma_chess_exposed_by_cases.exposed_by_cases_same"}
    claim = rl.parse(d)
    w, why = rl.standing_of(claim, True)
    print(f"OBS 79.44 F refusal: PROVED claimed on an OPEN obligation -- {w}: {why}")
    assert w == "CONJECTURE" and "sorry" in why and "holds no proof" in why
    d["standing"] = {"is": "PROVED", "theorem": "Ideas.Spec.Rook.no_such_theorem"}
    w, why = rl.standing_of(rl.parse(d), True)
    assert w == "CONJECTURE" and "no_such_theorem" in why
    P = lemma("candidate_is_mover_unit")
    assert rl.standing_of(P, False)[0] == "PROVED"                     # (the proof does not depend on a certificate being fresh)
    assert rl.standing_of(P, True, failing=True)[0] == "CONJECTURE"      # (but a certificate that contradicts it is heard)
    d = copy.deepcopy(P.doc())
    d["lean"] = dict(d["lean"], uses=["Ideas.LegalEquiv.candidate_unit", "Ideas.Nowhere.not_a_theorem"])
    w, why = rl.standing_of(rl.parse(d), True)
    assert w == "CONJECTURE" and "not_a_theorem" in why


def lake() -> "str | None":
    return shutil.which("lake") if (ROOT / "lean" / "ideas" / ".lake").exists() else None


@pytest.mark.skipif(lake() is None, reason="lake or the library's build is not on this machine")
def test_lean_accepts_each_obligation_the_open_ones_with_a_warning_only_and_the_proved_one_with_none():
    for name in NAMES:
        L = lemma(name)
        r = subprocess.run([lake(), "env", "lean", str(RLEAN().path_of(L).relative_to(ROOT / "lean" / "ideas"))], cwd=ROOT / "lean" / "ideas", capture_output=True, text=True, timeout=900)
        out = (r.stdout + r.stderr).strip()
        print(f"OBS 79.44 C: lean on {name}: exit {r.returncode}; {out[:140] or 'no message'}")
        assert r.returncode == 0 and "error" not in out
        assert ("declaration uses `sorry`" in out) == (not L.lean.get("proof"))


# ------------------------------------------------------------------------------------------------------------------------------------------ (5) the statements

STATEMENT_KEYS = {"id", "pattern", "says", "standing", "holder", "also"}


def statement_of(name: str) -> dict:
    L = lemma(name)
    c = RC().load(L)
    return RL().statement_of(L, RC().status(L) == "fresh", c["key"] if c else None, bool(c and c["disagreements"]))


def test_each_lemma_is_the_json_of_a_statement_and_its_standing_follows_the_lemma_s():
    s = {n: statement_of(n) for n in NAMES}
    for n, st in s.items():
        assert set(st) == STATEMENT_KEYS and st["id"] == n and st["holder"] == "ours" and st["says"] == {"kind": "holds"}
        assert all(set(l) == {"relation", "args", "neg"} for l in st["pattern"])
        assert all(isinstance(k, str) for k in st["also"]) and st["also"][0].startswith("lemma:")
    assert s["candidate_is_mover_unit"]["standing"] == {"kind": "proved", "theorem": "Ideas.Mined.rule_lemma_chess_candidate_is_mover_unit.candidate_is_mover_unit_implies"}
    for n in ("exposed_by_cases", "lone_rook_attacks_its_file"):
        assert s[n]["standing"]["kind"] == "conjecture" and s[n]["standing"]["sample"].startswith("certificate ") and "origin HAND" in s[n]["standing"]["sample"]
    assert s["exposed_by_cases"]["pattern"] == [{"relation": "control", "args": ["?c"], "neg": False}, {"relation": "check", "args": ["?c"], "neg": True}]
    # a lemma with no certificate is a conjecture saying so
    d = copy.deepcopy(lemma("exposed_by_cases").doc())
    d["standing"] = {"is": "CONJECTURE"}
    none = RL().statement_of(RL().parse(d), False)
    assert none["standing"] == {"kind": "conjecture", "sample": "no certificate: no certificate named; origin HAND"}
    print(f"OBS 79.44 D: statements {{{', '.join(n + ': ' + st['standing']['kind'] for n, st in s.items())}}}")


@pytest.mark.skipif(not os.environ.get("RULE_LEMMA_RUST"), reason="set RULE_LEMMA_RUST=1 to run cargo (offline) on the statements; a long first build")
def test_the_crates_own_reader_reads_the_statements(tmp_path):
    for n in NAMES:
        (tmp_path / f"{n}.json").write_text(json.dumps(statement_of(n)))
    env = dict(os.environ, CS_LEMMA_STATEMENTS=str(tmp_path))
    r = subprocess.run(["cargo", "test", "--offline", "-p", "cs_rep", "--test", "rule_lemma_statements", "--", "--nocapture"], cwd=ROOT / "rust" / "engine", env=env, capture_output=True, text=True, timeout=3000)
    print(r.stdout[-600:])
    assert r.returncode == 0 and "OBS 79.44 D: 3 lemma statements read by cs_rep: 1 Proved, 2 Conjecture" in r.stdout


# ------------------------------------------------------------------------------------------------------------------------------------------ (6) the scan, the untouched files, the gate

def sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


GAME_WORDS = r"\b(king\w*|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|mate|stalemate|uci|fen|san|chess\w*|shakmaty|atomic|antichess|horde|960)\b"
SCHEMA_WORDS = {"kind"}           # rep's own JSON key for a standing and a saying (rust/engine/rep/src/statement.rs): the machinery writes a statement in that schema


def words_of(text: str) -> set:
    return set(re.findall(r"[a-z_][a-z0-9_]*", text.lower()))


def scan(text: str, names: set) -> tuple:
    return sorted(set(re.findall(GAME_WORDS, text, flags=re.I))), sorted(words_of(text) & names)


def test_the_machinery_names_no_game_and_no_relation_of_any_description():
    names = sibling("test_plan79_40").all_names()
    assert len(names) > 250
    for p in MODULES:
        game_words, hit = scan(p.read_text(), names)
        print(f"OBS 79.44 G scan {p.name}: game words {game_words}, names of the descriptions {hit}")
        assert not game_words, (p.name, game_words)
        assert set(hit) <= SCHEMA_WORDS, (p.name, hit)


def test_the_scan_sees_a_planted_word():
    names = sibling("test_plan79_40").all_names()
    g, hit = scan("# a rook slides along its file\nx = 'slide'", names)
    assert g == ["rook"] and "slide" in hit and "file" in hit


def test_no_relative_data_path_no_path_insert_and_no_new_top_level_package():
    for p in MODULES + [ROOT / "scripts" / "rule_lemma_states_print.py"]:
        text = p.read_text()
        assert "sys.path.insert" not in text and not re.search(r"""["']data/(derived|experiments)""", text), p.name


@pytest.mark.skipif(not have_before(), reason="the commit before the plan is not on this machine")
def test_no_description_no_reader_and_no_source_of_the_engine_s_crates_differs_from_the_commit_before_the_plan():
    assert git("diff", "--quiet", BEFORE, "--", "rules/chess.gdl", "rules/variants", "rules/concepts", "rules/learned", "rules/surprises", "catspace/incubator/analyst/discovery/gdl.py",
               "catspace/incubator/analyst/discovery/gdl_chess.py", "catspace/incubator/analyst/discovery/certify.py", "scripts/gdl_to_lean.py").returncode == 0
    # (the crates this plan could have touched: the representation, the base, the description reader and the memory; another plan's translated files under rust/engine/game/tests are its own)
    changed = git("diff", "--name-only", BEFORE, "--", "rust/engine/rep", "rust/engine/base", "rust/engine/gdl", "rust/engine/memory", "rust/engine/Cargo.lock").stdout.split()
    print(f"OBS 79.44 G: files of those crates that differ from {BEFORE}: {changed}")
    assert changed == ["rust/engine/rep/tests/rule_lemma_statements.rs"], changed
    assert not [c for c in git("diff", "--name-only", BEFORE, "--", "rules").stdout.split() if not c.startswith("rules/lemmas/")]


def test_the_test_is_in_the_gate_after_the_one_of_79_42():
    text = (ROOT / "scripts" / "gate.py").read_text()
    assert "test_plan79_44" in text and text.index("test_plan79_42") < text.index("test_plan79_44")
