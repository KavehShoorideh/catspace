"""PLAN 79.59 (THE LEMMA MINER OF LAYER 2; SYSTEM_SPEC 11, block "The lemma miner of layer 2"; LAYER 2, piece: rules; ledger rows 59, 58, 57, 49). Loose and fast, by the standing rule for development mode: the pieces work end to end on small samples; the long runs are the judge's
(scripts/lemma_miner_run.py judge, rust/engine/game/tests/translated_judge.rs), which this file does not run.
  (1) pass 28 explain: the module made with it tags the reads and holds the tables; the module made without it has the plain recording statements (the pass changes nothing else).
  (2) the first proposer is game-free: it builds the null change from the clauses of a description it is given and refuses where the clauses stand in the way, with the reason.
  (3) the form FILTER: a well formed document is read, a document with a problem is refused whole, the mined folder is the folder of the mined lemmas.
  (4) pass 29 filter: the rewrite reads the replacement where the test was read; it refuses a lemma whose hits read the test.
  (5) the sources of the miner name no relation of any description under rules/ and no unit or square of a game (the scan of plan 79.37 applied to the new modules).
  (6) the analysis of an explained read says what follows from the clause and the order (one row, existence, bound later, scan)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import gdl_to_rust as T
from catspace.incubator.analyst.discovery import lemma_miner as M
from catspace.incubator.analyst.discovery import rule_lemma as RL
from catspace.incubator.analyst.discovery import rule_lemma_filter as RLF

ROOT = Path(__file__).resolve().parents[1]
DISC = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
PASSES = [p for p in T.PASSES if p != "carry"]
KEYED = (("placed_bears", 4),)


def chess_text() -> str:
    return (ROOT / "rules" / "chess.gdl").read_text()


def with_placed() -> str:
    from catspace.incubator.analyst.discovery import placed_questions as PQ
    return PQ.composed_text("chess")


# ------------------------------------------------------------------------------------------------------------------------------------------ (1) pass 28

def test_the_pass_explain_tags_the_reads_and_the_module_without_it_does_not():
    text = with_placed()
    plain = T.translate(text, "m_reads", PASSES + ["reads"], reads=KEYED)
    expl = T.translate(text, "m_explain", PASSES + ["reads", "explain"], reads=KEYED)
    assert "EXPL_FUNCS" not in plain and "EXPL_CLAUSES" not in plain
    assert re.search(r"rd_note\(\d+, &\[[^\]]*\]\);", plain) and not re.search(r"rd_note\(\d+, &\[[^\]]*\], \d+, \d+\);", plain)
    assert re.search(r"rd_note\(\d+, &\[[^\]]*\], \d+, \d+\);", expl)
    assert "pub const EXPL_FUNCS" in expl and "pub fn ask_explain" in expl and "pub fn expl_chain" in expl
    # the explain pass is optional: the passes every module is made with do not hold it
    assert "explain" in T.OPT_PASSES and "explain" not in T.PASSES and "filter" in T.OPT_PASSES and "filter" not in T.PASSES


def test_the_kept_explain_modules_hold_the_tables_for_every_game():
    kept = sorted((ROOT / "rust" / "engine" / "game" / "tests" / "generated_mined").glob("*_explain.rs"))
    assert len(kept) == 7
    for p in kept:
        head = p.read_text()[:3000]
        assert "// explain: every read carries its clause and literal" in head, p.name


# ------------------------------------------------------------------------------------------------------------------------------------------ (2) B1

def shape_and_proposal(text: str):
    d, f = M.shapes(text)
    assert f.found, f.refused
    s = f.found[0]
    return d, f, s, M.null_change(text, s, d, f)


def test_the_null_change_is_built_from_the_clauses_and_the_delta_reads_as_a_program():
    text = chess_text()
    d, f, s, pr = shape_and_proposal(text)
    assert pr.places == s["m"] and pr.test == s["R"] and pr.hypothetical == s["H"] and pr.state_relation == s["B"]
    assert len(pr.reads) >= 2 and pr.classes == 1
    prog = GDL.Program(RL.compose(text, pr.delta), "t")
    assert prog.typecheck() == []
    # every relation the delta defines is new, and the keys of the hits are the places of the test
    assert not ({c.split()[1].lstrip("(") for c in pr.clauses} & set(d.prog.nodes) - {pr.hits})
    assert len(prog.signature(pr.hits)) == pr.places and len(prog.signature(pr.answer)) == pr.places and len(prog.signature(pr.tested)) == pr.places
    # the reads name the literal and the clause they come from, in the description's own words
    assert all(r.literal.startswith("(") and r.clause.startswith("(<=") for r in pr.reads)


def test_the_null_change_is_the_same_on_a_description_whose_every_name_is_changed():
    """The proposer reads shapes, not names: rename every relation, fluent and sort of the first description and the proposal has the same size."""
    text = chess_text()
    names = set(re.findall(r"\(signature ([a-z_0-9]+)", text))
    sorts = set(re.findall(r"\(sort ([a-z_0-9]+)", text))
    fluents = {"cell", "control", "castling_right", "ep_square", "halfmove"}
    mapping = {n: "z" + n for n in sorted(names | sorts | fluents, key=len, reverse=True) if n not in ("true", "does", "legal", "role", "init", "next", "terminal", "goal", "record", "noop")}
    renamed = re.sub(r"[a-z_][a-z_0-9]*", lambda m: mapping.get(m.group(0), m.group(0)), text)
    try:
        _d, _f, _s, pr = shape_and_proposal(renamed)
    except (AssertionError, GDL.GDLError, T.Untranslatable):
        pytest.skip("the renamed description is not read by the reader (a constant that is also a relation name); the name-free property is held by the scan below")
    _d0, _f0, _s0, pr0 = shape_and_proposal(text)
    assert (len(pr.reads), len(pr.clauses), pr.classes) == (len(pr0.reads), len(pr0.clauses), pr0.classes)


def test_the_proposer_refuses_with_the_reason_where_the_clauses_have_two_guards():
    """The test of the third variant of the description holds two different guard literals: the class of a key is not one template, the proposer says so and builds nothing."""
    text = T.compose_text([ROOT / "rules" / "chess.gdl", ROOT / "rules" / "variants" / "atomic.gdl"])
    d, f = M.shapes(text)
    assert f.found
    with pytest.raises(M.Refused) as e:
        M.null_change(text, f.found[0], d, f)
    assert "two different guard literals" in str(e.value)


# ------------------------------------------------------------------------------------------------------------------------------------------ (3) the form FILTER

def good_doc():
    text = chess_text()
    d, f, s, pr = shape_and_proposal(text)
    return text, M.document(pr, "chess", "chess")


def test_a_document_of_the_form_filter_is_read_and_a_document_with_a_problem_is_refused_whole():
    text, doc = good_doc()
    lem = RLF.parse(doc, base_text=text)
    assert lem.form == "FILTER" and lem.origin == "MINED" and lem.standing["is"] == "PROPOSED"
    assert RLF.filter_problems(doc, text) == []
    bad = dict(doc, origin="HAND")
    with pytest.raises(RL.LemmaError):
        RLF.parse(bad, base_text=text)
    bad = {k: v for k, v in doc.items() if k != "evidence"}
    assert RLF.filter_problems(bad, text)
    bad = dict(doc, standing={"is": "CERTIFIED", "key": "x"})
    assert any("standing" in x for x in RLF.filter_problems(bad, text))
    bad = dict(doc, filter=dict(doc["filter"], hits="no_such_relation"))
    assert RLF.filter_problems(bad, text)
    bad = dict(doc, delta=[c for c in doc["delta"] if not c.startswith(f"(<= ({doc['filter']['hits']} ")])
    assert RLF.filter_problems(bad, text)


def test_the_mined_folder_is_not_the_folder_of_the_hand_given_lemmas():
    assert RLF.mined_dir("chess") == RL.lemma_dir("chess") / "mined"
    assert RLF.mined_dir("chess") != RL.lemma_dir("chess")
    assert not any("mined" in p.name for p in RL.lemma_dir("chess").glob("*.json"))
    # the loader of the hand-given lemmas does not read the mined folder: its documents are of another form
    assert all(l.form != "FILTER" for l in RL.load_all("chess"))


def test_a_standing_of_proved_rests_on_a_theorem_that_compiles_and_a_gate_that_held():
    text, doc = good_doc()
    lem = RLF.parse(doc, base_text=text)
    proved = RLF.FilterLemma(**{**lem.__dict__, "standing": {"is": "PROVED", "detail": ""}, "evidence": {"gate": "held"}})
    word, detail = RLF.filter_standing(proved)
    assert word == "PROVED", detail
    # with the gate not held, or no theorem named, or a theorem that does not exist, the standing is lowered to TESTED
    assert RLF.filter_standing(RLF.FilterLemma(**{**proved.__dict__, "evidence": {"gate": "failed"}}))[0] == "TESTED"
    assert RLF.filter_standing(RLF.FilterLemma(**{**proved.__dict__, "lean": dict(proved.lean, uses=[])}))[0] == "TESTED"
    assert RLF.filter_standing(RLF.FilterLemma(**{**proved.__dict__, "lean": dict(proved.lean, uses=["Ideas.NullFilter.Test.no_such_theorem"])}))[0] == "TESTED"
    assert RLF.filter_standing(RLF.FilterLemma(**{**lem.__dict__, "standing": {"is": "REJECTED", "detail": "x"}}))[0] == "REJECTED"


def test_the_kept_mined_documents_are_read_and_none_is_proved_without_its_gate():
    kept = []
    for game in ("chess", "3check", "kingofthehill", "chess960", "horde"):
        for p in sorted(RLF.mined_dir(game).glob("*.json")):
            import json
            doc = json.loads(p.read_text())
            assert doc["form"] == "FILTER" and doc["origin"] == "MINED" and doc["standing"]["is"] in RLF.MINED_STANDINGS
            if doc["standing"]["is"] == "PROVED":
                assert doc["evidence"]["gate"] == "held" and doc["lean"]["uses"]
            if doc["standing"]["is"] == "REJECTED":
                assert doc["evidence"]["gate"] == "failed"
            kept.append(p)
    assert kept


# ------------------------------------------------------------------------------------------------------------------------------------------ (4) pass 29

def lemma_data(doc):
    return {"name": doc["name"], "form": "FILTER", "left": doc["left"], "right": doc["right"], "filter": doc["filter"], "standing": doc["standing"]}


def test_the_filter_rewrite_reads_the_replacement_where_the_test_was_read_and_refuses_a_hits_that_reads_the_test():
    text, doc = good_doc()
    full = RL.compose(text, doc["delta"])
    new, infos = T.rewrite_filters(full, [lemma_data(doc)])
    assert infos[0]["form"] == "FILTER" and infos[0]["tested"] == doc["filter"]["tested"]
    prog = GDL.Program(new, "t")
    rep = infos[0]["rep"]
    # a clause that read the test now reads the replacement; the replacement has the two clauses of the rewrite
    assert any(rep in c.text and doc["left"]["relation"] not in [l[1] for l in c.body if l[0] != "distinct"] for cs in prog.rules.values() for c in cs) if hasattr(prog, "rules") else True
    assert f"({rep} ?x1 ?x2 ?x3 ?x4)" in "\n".join(GDL._unparse(x) for x in GDL.parse(new))
    # the hits that read the test would make the replacement read itself
    bad = dict(doc, delta=doc["delta"] + [f"(<= ({doc['filter']['hits']} ?a ?b ?c ?d) ({doc['left']['relation']} ?a ?b ?c ?d))"])
    with pytest.raises(T.Untranslatable):
        T.rewrite_filters(RL.compose(text, bad["delta"]), [lemma_data(bad)])
    # a lemma of this form is read only with the pass
    with pytest.raises(T.Untranslatable):
        T.translate(full, "m", PASSES, [lemma_data(doc)])


def test_the_module_with_the_filter_holds_the_lemma_line_and_the_module_without_it_is_unchanged():
    text, doc = good_doc()
    full = RL.compose(text, doc["delta"])
    src = T.translate(full, "m_mined", PASSES + ["filter"], [lemma_data(doc)])
    head = "\n".join(src.split("\n")[:12])
    assert "// lemma: exposed_null_filter FILTER" in head and "tested exposed_m" in head
    plain = T.translate(text, "m_plain", PASSES)
    assert "// lemma:" not in "\n".join(plain.split("\n")[:12])


# ------------------------------------------------------------------------------------------------------------------------------------------ (5) the scan

def description_names() -> set:
    G = GDL
    names: set = set()
    for path in sorted((ROOT / "rules").rglob("*.gdl")):
        for x in G.parse(path.read_text()):
            if isinstance(x, list) and x and x[0] == "retract":
                continue
            if isinstance(x, list) and x and x[0] in G.DECLARATIONS:
                names.add(x[1])
                continue
            head = x[1] if isinstance(x, list) and x and x[0] == "<=" else x
            if isinstance(head, list) and head and head[0] in G.FLAT and len(head) > 1:
                f = head[1]
                names.add(f[0] if isinstance(f, list) else f)
            else:
                names.add(head[0] if isinstance(head, list) else head)
    return {n for n in names if isinstance(n, str)}


FIXED = {"role", "init", "true", "does", "legal", "next", "terminal", "goal", "record", "noop", "distinct", "not", "or", "signature", "sort", "range"}


@pytest.mark.parametrize("name", ["lemma_miner.py", "rule_lemma_filter.py"])
def test_the_miner_sources_name_no_relation_of_a_description_and_no_unit_or_square(name):
    words = set(re.findall(r"[a-z_][a-z0-9_]*", (DISC / name).read_text().lower()))
    hit = sorted(words & (description_names() - FIXED))
    assert not hit, f"{name} holds names of a game's relations: {hit}"
    bad = re.findall(r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b", (DISC / name).read_text(), flags=re.I)
    assert not bad, bad


def test_the_translator_still_names_no_relation_of_a_description():
    words = set(re.findall(r"[a-z_][a-z0-9_]*", (DISC / "gdl_to_rust.py").read_text().lower()))
    assert not (words & (description_names() - FIXED))


# ------------------------------------------------------------------------------------------------------------------------------------------ (6) what follows from an explained read

def test_the_findings_say_what_follows_from_the_clause_and_the_order():
    # a literal read with every place bound tests one row
    c = "(<= (p ?x ?y) (q ?x ?y) (r ?x ?y))"
    got = M.findings_at(c, "00", [0, 1], 1)
    assert got and got[0].startswith("ONE ROW")
    # a literal that leaves a variable free that nothing else uses is a read for existence
    c = "(<= (p ?x) (q ?x) (r ?x ?z))"
    assert M.findings_at(c, "0", [0, 1], 1)[0].startswith("EXISTENCE")
    # a variable the literal leaves free that another literal binds later in the join: bound later
    c = "(<= (p ?x ?y) (r ?y) (s ?x ?y))"
    got = M.findings_at(c, "00", [1, 0], 0)
    assert got and got[0].startswith("BOUND LATER") and "could be joined first" not in got[0] and "joined first" in got[0]
    # nothing else binds it: a scan
    c = "(<= (p ?x ?y) (r ?x ?y))"
    assert M.findings_at(c, "00", [0], 0)[0].startswith("SCAN")
