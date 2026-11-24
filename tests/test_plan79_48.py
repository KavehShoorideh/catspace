"""PLAN 79.48 (SYSTEM_SPEC 11, "The description is the source of the rules; fast code is only a certified copy"; piece: rules; ledger rows 44, 42): the gate of the placed questions of layer 2 said over the description and held to it
  (1) PART A, the list: the methods of the trait `Placed` are read from rust/engine/game/src/placed.rs, each with the call sites above layer 2 by crate; the fast answers of kernel.rs and outside.rs are read from those sources;
  (2) PART B, the definitions: the documents of rules/placed/ compose on the description of each of the seven games, the reader accepts the composed text (stratified, safe, well sorted), every relation a document names is defined,
      no document redefines a relation of the description, and none of the twelve certificates went stale (no description changed);
  (3) PART C, the module: the kept module of each game (generated_placed/<game>.rs) is the fresh translation of the composed text, and the translator was given the text as it is;
  (4) PART D, the record of each game (copy_records/placed/<game>.json) is fresh, read the certificate's own corpus, compared every question, held the module to the evaluator, and holds exactly the causes this file names for the game;
  (5) PART F, the gate's list: every fast answer of layer 2 has a certificate, a fresh record or a line of the short list of exceptions with a cause in words; a line of the list that names nothing in the sources fails; a planted answer
      that nothing holds is found (a negative control); the records of the kernel against the translated module and the certificates are fresh;
  (6) the name scan: the Python modules of this plan name no object of a game; the gate lists this file after the one of 79.47.
The cargo test of cs_game (translated_placed.rs) holds the kernel's placed answers to the module of the definitions and the module to the evaluator."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
SEVEN = ["chess", "atomic", "3check", "kingofthehill", "antichess", "chess960", "horde"]
GEN = ENGINE / "game" / "tests" / "generated_placed"
RUST_TEST = ENGINE / "game" / "tests" / "translated_placed.rs"
RUST_KNOWN = ENGINE / "game" / "tests" / "translated_placed_known.rs"
needs_cargo = pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo (offline) to run the small co-play")


def PQ():
    from catspace.incubator.analyst.discovery import placed_questions as m
    return m


def PT():
    from catspace.incubator.analyst.discovery import placed_translated as m
    return m


def KT():
    from catspace.incubator.analyst.discovery import kernel_translated as m
    return m


# THE CAUSES NAMED FOR A GAME (the record holds exactly these, each a finding with its whole state in the record's examples); the explanation of each is in the BUILT block of docs/ANALYST_PLAN.md and in KNOWN of placed_translated.py
TWO_PLACES = {"reach_from: only module, pawn, a two-place step along a file", "reach_from (given): only module, pawn, a two-place step along a file"}
KNOWN = {g: set(TWO_PLACES) for g in SEVEN}
KNOWN["atomic"].add("strikes_place: only kernel, the owner's own king beside the place")


# ------------------------------------------------------------------------------------------------------------------------------------------ (1) part A: the list

TRAIT = ["place_names", "kind_names", "cells", "bearing_from", "reach_from", "move_ends", "kind_at", "find_step", "why_not", "move_info", "strikes_place", "without", "relocate", "steps"]


def test_the_list_of_placed_questions_is_read_from_the_trait_and_each_has_its_callers_counted():
    qs = PQ().questions()
    assert [q["name"] for q in qs] == TRAIT, [q["name"] for q in qs]
    assert all(q["signature"].startswith("fn ") and q["says"] for q in qs)
    by = {q["name"]: q for q in qs}
    assert [q["name"] for q in qs if q["provided"]] == ["steps"]
    above = {n: sum(v for k, v in q["callers"].items() if k != "game") for n, q in by.items()}
    assert above["move_ends"] > above["why_not"] > 0 and above["cells"] > 0 and above["strikes_place"] > 0
    for line in PQ().obs_lines():
        if line.startswith("OBS 79.48 A") and " fast " not in line and "totals" not in line:
            print(line[:300])


def test_the_fast_answers_are_read_from_the_two_sources_that_hold_them():
    items = PQ().fast_answers()
    names = {(i["file"], i["trait"] or i["owner"], i["name"]) for i in items}
    for want in [("kernel.rs", "Game", "moves"), ("kernel.rs", "Game", "apply"), ("kernel.rs", "Game", "outcome"), ("outside.rs", "Placed", "reach_from"), ("outside.rs", "Placed", "strikes_place"),
                 ("outside.rs", "KernelGame", "state_facts"), ("outside.rs", "StateText", "state_text"), ("outside.rs", "", "spec_term")]:
        assert want in names, want
    assert all("Placed" in {i["trait"] for i in items if i["name"] == q} or q in ("steps",) for q in TRAIT if q != "steps"), "every method of the trait is implemented in outside.rs"
    print(f"OBS 79.48 A: {len(items)} fast answers in kernel.rs and outside.rs, {len(PQ().questions())} methods of the trait")


# ------------------------------------------------------------------------------------------------------------------------------------------ (2) part B: the definitions

def test_the_documents_compose_on_every_description_and_the_reader_accepts_the_text():
    from catspace.incubator.analyst.discovery import gdl as GDL
    pq = PQ()
    for g in SEVEN:
        text = pq.composed_text(g)
        prog = GDL.Program(text, f"placed/{g}")                      # the reader's own tests: stratified, safe, well sorted
        assert prog is not None
        base = {d["relation"] for doc in pq.question_documents() for d in doc["questions"]}
        for rel in base:
            assert f"(<= ({rel} " in text, (g, rel)                   # each relation a document names is defined by a clause
    print(f"OBS 79.48 B: the documents of rules/placed/ compose on the descriptions of {len(SEVEN)} games and the reader accepts each text")


def test_no_document_redefines_a_relation_of_the_description_and_the_game_document_only_retracts_what_the_base_one_added():
    pq = PQ()
    head = re.compile(r"\(<= \(([a-z_0-9]+) ")
    for g in SEVEN:
        paths = pq.description_paths(g)
        described = set()
        for p in paths:
            described |= {m.group(1) for m in head.finditer(p.read_text())}
        docs = pq.documents(g)
        for d in docs:
            for cl in json.loads(d.read_text())["delta"]:
                if cl.startswith("(retract"):
                    assert d.name != "base.json"
                    inner = head.search(cl).group(1)
                    assert inner.startswith("placed_"), (g, cl)
                    continue
                m = head.match(cl)
                if m:
                    assert m.group(1).startswith("placed_") and m.group(1) not in described, (g, cl)
    print("OBS 79.48 B: every clause of the documents defines a relation named placed_*; none is a relation of a description; a game's document retracts only a clause of the base document")


def test_the_definitions_are_aliases_of_relations_the_description_has_and_each_question_names_the_methods_it_answers():
    pq = PQ()
    docs = pq.question_documents()
    asks = {m.strip() for doc in docs for q in doc["questions"] for m in q["asks"].split(",")}
    assert asks <= set(TRAIT), asks - set(TRAIT)
    held = {q["name"] for q in pq.questions()}
    for m in pq.HELD_BY_PLACED:
        assert m in held, m
    # the methods the definitions answer are the ones the placed record holds; the trait's others are exceptions with a cause
    assert pq.HELD_BY_PLACED == {"place_names", "kind_names", "cells", "bearing_from", "reach_from", "move_ends", "kind_at", "find_step", "move_info", "strikes_place"}
    assert {"why_not", "without", "relocate"} <= {k.split("::")[1] for k in json.loads((ROOT / "rules" / "placed" / "held.json").read_text())["exceptions"]}


def test_the_twelve_certificates_and_the_descriptions_are_unchanged():
    from catspace.incubator.analyst.discovery import certify as CF
    st = CF.status()
    stale = [k for k, v in st.items() if v != "fresh" and not str(k).startswith("uncertified")]
    assert not stale, stale


# ------------------------------------------------------------------------------------------------------------------------------------------ (3) part C: the modules

def test_there_is_a_kept_module_for_each_game_and_it_is_the_fresh_translation_of_the_composed_text():
    from catspace.incubator.analyst.discovery import gdl_to_rust as T
    from catspace.incubator.analyst.discovery import placed_translated as pt
    assert pt.games() == sorted(SEVEN), pt.games()
    kt = KT()
    for g in SEVEN:
        text = PQ().composed_text(g)
        want = T.translate(text, "placed_" + kt.rust_name(g), list(T.PASSES))
        assert (GEN / f"{g}.rs").read_text() == want, f"{g}: python scripts/gdl_to_rust_make.py --placed"
    print(f"OBS 79.48 C: the {len(SEVEN)} kept modules are the fresh translations of the composed descriptions, every pass, the translator unchanged")


def test_the_translator_source_names_no_relation_of_the_placed_documents():
    text = (ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "gdl_to_rust.py").read_text()
    for doc in PQ().question_documents():
        for q in doc["questions"]:
            assert q["relation"] not in text, q["relation"]
    assert "placed_" not in text


# ------------------------------------------------------------------------------------------------------------------------------------------ (4) part D: the records

def causes_of(r: dict) -> set:
    return set(r["small"].get("causes", {})) | set(r["large"].get("causes", {}))


def test_there_is_a_record_for_each_of_the_seven_games():
    pt = PT()
    for g in SEVEN:
        assert pt.path_of(g).exists(), f"no record for {g}: python -m catspace.incubator.analyst.discovery.placed_translated run {g}"


@pytest.mark.parametrize("game", SEVEN)
def test_the_record_of_a_game_is_fresh_counted_over_every_question_and_holds_exactly_the_named_causes(game):
    pt, kt = PT(), KT()
    r = pt.load(game)
    st = pt.status(game)
    assert st == "fresh", f"{game}: {st}: python -m catspace.incubator.analyst.discovery.placed_translated run {game}"
    s, lg = r["small"], r["large"]
    assert s["states"] == s["certificate_positions"] == kt.certificate_positions(game) and s["equals_certificate"]
    assert lg["states"] > s["states"] and lg["probes"] > 0 and lg["cores"] >= 1 and lg["ended"] > 0
    assert causes_of(r) == set(KNOWN[game]), (game, causes_of(r) ^ set(KNOWN[game]))
    assert set(pt.KNOWN[game]) == set(KNOWN[game])
    assert not any(c.startswith("lower link") for c in causes_of(r))
    for q in pt.QUESTIONS:
        assert s["compared"][q][0] > 0 and lg["compared"][q][0] > 0, (game, q)
    for rel in pt.LOWER:
        assert s["lower"][rel][0] > 0, (game, rel)
    print(f"OBS 79.48 D record {game}: fresh; small {s['states']} states (the certificate's {s['certificate_positions']}), {s['probes']} probes, the lower link at {s['lower_states']} states; large {lg['games']} games {lg['states']} states {lg['probes']} probes, "
          f"{sum(v[0] for v in lg['compared'].values())} comparisons of {sum(v[1] for v in lg['compared'].values())} rows; causes {sorted(causes_of(r))}")


def test_a_record_with_a_cause_that_is_not_named_is_failing_and_one_that_is_old_is_stale_and_one_that_missed_a_question_is_short():
    import copy
    pt = PT()
    good = copy.deepcopy(pt.load("chess"))
    assert pt.status("chess", good) == "fresh"
    bad = copy.deepcopy(good)
    bad["small"]["causes"]["a cause nobody named"] = 1
    assert pt.status("chess", bad) == "failing"
    old = copy.deepcopy(good)
    old["key"] = "0" * 16
    assert pt.status("chess", old) == "stale"
    short = copy.deepcopy(good)
    short["large"]["compared"].pop("strikes_place")
    assert pt.status("chess", short) == "short"
    nolow = copy.deepcopy(good)
    nolow["small"]["lower"].pop("placed_takes")
    assert pt.status("chess", nolow) == "short"


@needs_cargo
def test_a_fresh_small_run_of_each_game_counts_the_certificates_positions_and_holds_the_named_causes_only():
    """The seven small runs at once (each is one core and the longest is a few minutes: the evaluator is asked at every state of the small corpus)."""
    from concurrent.futures import ThreadPoolExecutor
    pt, kt = PT(), KT()
    with ThreadPoolExecutor(max_workers=len(SEVEN)) as ex:
        got = dict(zip(SEVEN, ex.map(pt.small_counts, SEVEN)))
    for game, c in got.items():
        assert c["states"] == kt.certificate_positions(game)
        assert set(c["causes"]) == set(KNOWN[game]), (game, c["causes"])
        assert c["probes"] > c["states"] and c["lower_states"] >= c["states"] and not c["left_out"].get("a state the module cannot read")
        print(f"OBS 79.48 D small {game}: {c['games']} games, {c['states']} states (the certificate's {kt.certificate_positions(game)}), {c['probes']} probes, lower link at {c['lower_states']} states, disagreements {sum(c['causes'].values())}")


def test_the_large_runs_are_ignored_tests_with_a_named_cause_and_the_small_ones_and_the_control_are_not():
    text = RUST_TEST.read_text()
    large = re.findall(r'#\[test\]\n#\[ignore = "([^"]+)"\]\nfn (\w+_large_\w+)', text)
    assert len(large) == 7 and all("large run" in why and "run by hand" in why for why, _n in large), large
    small = re.findall(r"#\[test\]\nfn (\w+_on_the_certificates_games)", text)
    assert len(small) == 7
    assert re.search(r"#\[test\]\nfn a_kernel_of_other_rules_is_caught_by_the_placed_comparison", text)


# ------------------------------------------------------------------------------------------------------------------------------------------ (5) part F: the gate's list

def test_every_fast_answer_of_layer_2_has_a_holder_and_the_holders_are_fresh():
    pq, kt = PQ(), KT()
    rows = pq.gate_rows()
    none = [f"{r['file']}: {r['trait'] or r['owner']}::{r['name']}" for r in rows if r["held"]["kind"] == "none"]
    assert not none, f"fast answers with no certificate, no fresh record and no exception: {none}"
    kinds: dict = {}
    for r in rows:
        kinds[r["held"]["kind"]] = kinds.get(r["held"]["kind"], 0) + 1
    # what the holders must be: the record of the kernel against the translated module (seven games), the placed record (seven games)
    if kinds.get("record"):
        assert all(kt.status(g) == "fresh" for g in SEVEN), {g: kt.status(g) for g in SEVEN}
    if kinds.get("placed"):
        assert all(PT().status(g) == "fresh" for g in SEVEN), {g: PT().status(g) for g in SEVEN}
    for r in rows:
        if r["held"]["kind"] == "exception":
            assert len(r["held"]["cause"]) > 25, r
    print(f"OBS 79.48 F: {len(rows)} fast answers of layer 2; held by {kinds}; every exception has its cause in words")


def test_every_line_of_the_list_of_exceptions_and_of_the_record_names_an_answer_of_the_sources_and_the_list_is_short():
    pq = PQ()
    held = json.loads((ROOT / "rules" / "placed" / "held.json").read_text())
    keys = {f"{r['trait'] or r['owner']}::{r['name']}" for r in pq.fast_answers()}
    for part in ("record", "exceptions"):
        gone = sorted(k for k in held[part] if k not in keys)
        assert not gone, f"{part}: lines that name no answer of kernel.rs or outside.rs: {gone}"
    assert not (set(held["record"]) & set(held["exceptions"]))
    assert len(held["exceptions"]) <= 24, len(held["exceptions"])
    print(f"OBS 79.48 F: {len(held['record'])} answers held by the record of the kernel, {len(held['exceptions'])} exceptions (the limit is 24), none stale")


def test_a_planted_answer_that_nothing_holds_is_found_and_a_planted_line_in_the_list_is_found():
    import tempfile
    pq = PQ()
    src = pq.OUTSIDE_RS.read_text()
    planted = src.replace("impl Placed for KernelGame {", "impl Placed for KernelGame {\n    fn a_new_answer(&self) -> u8 { 0 }", 1)
    assert planted != src
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "outside.rs"
        p.write_text(planted)
        items = pq._impl_items(p)
    new = [i for i in items if i["name"] == "a_new_answer"]
    assert len(new) == 1 and pq.held_by(new[0])["kind"] == "none", new
    assert pq.held_by({"trait": "Placed", "owner": "Placed", "name": "reach_from"})["kind"] == "placed"
    assert pq.held_by({"trait": "Placed", "owner": "Placed", "name": "why_not"})["kind"] == "exception"


def test_the_records_of_the_kernel_against_the_module_and_the_twelve_certificates_are_fresh():
    kt = KT()
    for g in SEVEN:
        assert kt.status(g) == "fresh", (g, kt.status(g))
    from catspace.incubator.analyst.discovery import certify as CF
    st = CF.status()
    assert sum(1 for k, v in st.items() if v == "fresh") >= 12, st


# ------------------------------------------------------------------------------------------------------------------------------------------ (6) the names and the gate

GAME_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"


@pytest.mark.parametrize("name", ["placed_questions.py", "placed_translated.py"])
def test_the_python_modules_of_the_plan_name_no_object_of_a_game(name):
    text = (ROOT / "catspace" / "incubator" / "analyst" / "discovery" / name).read_text()
    hits = [m.group(0) for m in re.finditer(GAME_WORDS, text, flags=re.I)]
    assert not hits, (name, hits[:8])


def test_the_known_states_of_the_causes_are_ignored_tests_with_the_cause_in_their_text():
    text = RUST_KNOWN.read_text()
    named = re.findall(r'#\[ignore = "([^"]+)"\]\nfn (\w+_known_cause_\w+)', text)
    assert named and all("known cause" in why and "79.48" in why for why, _n in named), named


@needs_cargo
def test_the_known_states_are_run_and_show_their_causes():
    """The ignored tests of translated_placed_known.rs are run by the gate: each asserts that the kernel and the module still differ at its state as the cause says (the cause is gone when one fails)."""
    pt = PT()
    rc, text = pt.cargo(["--test", "translated_placed_known", "--", "--ignored", "--nocapture"], {})
    assert rc == 0, text[-2000:]
    lines = [ln for ln in text.splitlines() if ln.startswith("OBS 79.48 E known cause")]
    assert len(lines) == 2, lines
    for ln in lines:
        print(ln[:300])


def test_the_gate_lists_this_test_after_the_one_of_plan_79_47():
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert "test_plan79_47 test_plan79_48" in gate
