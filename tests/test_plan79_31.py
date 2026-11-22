"""PLAN 79.31 LAYER 0 MADE RIGHT: AN ORDER FOR EACH CHOOSER, THE MIXED CHOICE, THE BELIEF, EXACT CHANCES, AND ITS MEANING HELD AGAINST THE SEARCH (SYSTEM_SPEC 11 "Layer 0 is made right", "Belief about the opponent, and choosing by chance", 13 item 49; ledger rows 36, 37).
Rust holds the unit tests (rust/engine/base) and the test that holds the reading of layer 0 against layer 4 (rust/engine/tactics/tests/layer0_meaning.rs). THIS file is what belongs to Python: the scans.
  (1) cs_base depends on NO crate at all, of ours or outside (its Cargo file, its lock entry, its sources);
  (2) cs_base's sources and the layer 4 test of the meaning name no game and none of the words of a world;
  (3) the readings of part A exist as signatures: a chooser is an identity, a party is a set of choosers, a choice at once is between two parties, opposed and allied are asked of the orders, the readings take a set;
  (4) the mixed choice, the belief and the choice against a belief exist; the exact fractions are in the number type and a chance is a number;
  (5) the JSON of a literal is written and read in cs_rep (a trait), and no outside crate is in cs_base's sources;
  (6) layer 0's laws of this plan are theorems in Lean, none is a sorry, and the one law not proved (the minimax theorem) is not stated as proved anywhere;
  (7) the representation uses layer 0's set where it is the same thing (part G);
  (8) the test is in the gate after the one of 79.30.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "rust" / "engine" / "base"
REP = ROOT / "rust" / "engine" / "rep"
TACTICS = ROOT / "rust" / "engine" / "tactics"
LEAN = ROOT / "lean" / "ideas" / "Ideas" / "Layer0"


def _plan30():
    spec = importlib.util.spec_from_file_location("plan79_30_scan", ROOT / "tests" / "test_plan79_30.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P30 = _plan30()
EXTRA_WORDS = re.compile(r"\b(opponents?|adversar\w+|enem\w+|foes?)\b", flags=re.I)


def src(path: str) -> str:
    return (BASE / "src" / path).read_text()


def test_cs_base_depends_on_nothing_at_all():
    toml = (BASE / "Cargo.toml").read_text()
    deps = toml.split("[dependencies]", 1)[1]
    names = re.findall(r"^([A-Za-z0-9_\-]+)\s*=", deps, flags=re.M)
    assert names == [], names
    lock = (ROOT / "rust" / "engine" / "Cargo.lock").read_text()
    block = lock.split('name = "cs_base"', 1)[1].split("[[package]]", 1)[0]
    assert "dependencies" not in block, block
    for p in sorted((BASE / "src").rglob("*.rs")) + sorted((BASE / "tests").glob("*.rs")):
        t = p.read_text()
        assert not re.search(r"\b(serde|serde_json|cs_(?!base\b)[a-z]+)::", t), p.name
        assert not re.search(r"extern crate", t), p.name
    print("OBS 79.31 scan: cs_base's Cargo file lists no dependency, its lock entry has none, no source names an outside crate or a crate of ours")


# the words of a game that no layer names (the list of tests/test_plan78.py): the layer 4 test of the meaning must say "goal", "unit" and "board" as every layer 4 test does, and names no game
GAME_WORDS = re.compile(r"\b(king\w*|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b", flags=re.I)


def test_cs_base_names_no_game_and_no_world_word_and_the_layer_4_test_of_the_meaning_no_game():
    files = P30.base_sources()
    bad = {}
    for p in files:
        text = p.read_text()
        hits = [m.group(0) for m in P30.WORLD.finditer(text)] + [m.group(0) for m in EXTRA_WORDS.finditer(text)]
        if hits:
            bad[p.relative_to(ROOT).as_posix()] = sorted(set(hits))
    assert not bad, bad
    meaning = TACTICS / "tests" / "layer0_meaning.rs"
    hits = GAME_WORDS.findall(meaning.read_text())
    assert not hits, hits
    assert "can_force_set" in meaning.read_text() and "ask(&q)" in meaning.read_text()
    print(f"OBS 79.31 scan: {len(files)} sources of cs_base hold none of the world words and none of the words of a side against a side; the layer 4 test of the meaning names no game")


def test_a_chooser_is_an_identity_a_choice_at_once_is_between_two_parties_and_opposed_and_allied_are_asked_of_the_orders():
    c = src("choice/mod.rs")
    assert re.search(r"pub struct Chooser\(u32\)", c) and "pub enum Chooser" not in c
    assert "pub type Party = Set<Chooser>" in c
    assert re.search(r"Table \{ row_party: Party, col_party: Party, rows: usize, cols: usize, cells", c)
    for sig in ("pub fn can_force_set<R>(t: &Tree<R>, who: &Party", "pub fn guarantee<R: 'static>(t: &Tree<R>, side: &Party", "pub fn guarantee_allied<R: 'static>", "pub fn value_between<R: 'static>(t: &Tree<R>, wants: &Wants<R>, p: &Party, q: &Party)",
                "pub fn opposed(&self, a: Chooser, b: Chooser", "pub fn allied(&self, a: Chooser, b: Chooser", "pub fn zero_sum(&self, a: Chooser, b: Chooser", "pub struct Wants<R>"):
        assert sig in c, sig
    # the two-chooser calls stay, as thin cases
    assert "pub fn can_force<R>(t: &Tree<R>, who: Chooser" in c and "pub fn value<R: 'static>(t: &Tree<R>, a_order: &Order<R>)" in c
    # the value between sides refuses what it does not read: not opposed, not covered, not zero-sum with chance
    for e in ("NotOpposed", "NotAllied", "NotCovered", "NotZeroSum", "SplitParty", "NoOrder"):
        assert re.search(rf"\b{e}\b", c), e
    print("OBS 79.31 scan: a chooser is an identity, a table is between two parties, the readings take a set, opposed and allied are asked of the orders")


def test_the_mixed_choice_the_belief_and_exact_fractions_exist():
    c = src("choice/mod.rs")
    assert "pub fn mixed_value<R: 'static>" in c and "pub fn mixed_table<R: 'static>" in c and "pub fn choose_against<O, S: Clone>" in c
    lp = src("choice/lp.rs")
    assert "pub fn solve_matrix_game(m: &[Vec<Num>])" in lp and "NotExact" in lp and "Overflow" in lp and "f64" not in re.sub(r"//.*", "", lp)
    n = src("number.rs")
    assert "pub fn frac(numer: i64, denom: i64)" in n and "Frac(i64, i64)" in n and "pub fn exact(v: f64)" in n
    ch = src("chance.rs")
    assert "pub struct Spread<T> { outcomes: Vec<(T, Num)> }" in ch and "pub struct Belief<S>" in ch and "pub fn after_seeing" in ch and "Impossible" in ch
    assert "pub fn product<A: Clone, B: Clone>(a: &Spread<A>, b: &Spread<B>) -> Result<Spread<(A, B)>, ChanceError>" in ch
    print("OBS 79.31 scan: the exact program has no real number in its code; chances are numbers; the belief, the mixed choice and the choice against a belief exist")


def test_the_json_of_a_literal_is_written_in_cs_rep_and_not_in_cs_base():
    assert not (BASE / "src" / "logic" / "json.rs").exists()
    lemma = (REP / "src" / "lemma.rs").read_text()
    assert "pub trait LitJson" in lemma and "impl LitJson for Lit" in lemma
    for f in ("statement.rs", "claim.rs"):
        assert "LitJson" in (REP / "src" / f).read_text(), f
    assert "LitJson" in (TACTICS / "tests" / "layer4_goals.rs").read_text()
    print("OBS 79.31 scan: LitJson is cs_rep's trait; statement.rs, claim.rs and the layer 4 test bring it into scope; cs_base has no json file")


def lean_code(module: str) -> str:
    return re.sub(r"/-.*?-/", "", (LEAN / f"{module}.lean").read_text(), flags=re.S)


def test_the_laws_of_this_plan_are_theorems_and_nothing_is_a_sorry():
    needed = {"Number": ["frac_add_val", "frac_mul_val", "frac_le_iff", "frac_one_whole", "frac_scale", "example_third_sixth", "example_tenths", "example_ninths"],
              "Chance": ["belief_sums_to_one", "belief_two_sightings", "example_expectation", "example_belief_first", "example_belief_second"],
              "Choice": ["coalition_exactly_one", "guarantees_meet_of_certificate", "certificate_value_unique", "no_mix_beats_best", "best_is_an_option", "example_pennies_low", "example_pennies_up",
                         "example_saddle_low", "example_saddle_up", "example_coin_low"]}
    for module, names in needed.items():
        code = lean_code(module)
        for n in names:
            assert re.search(rf"theorem {n}\b", code), (module, n)
    for module in ("Logic", "Sets", "Number", "Structure", "Chance", "Choice"):
        code = lean_code(module)
        assert not re.search(r"\bsorry\b|native_decide|\bdecide\b|^axiom\b|\baxiom\s", code, flags=re.M), module
    # the minimax theorem is OPEN: it is said so in the Choice file, and no theorem of that name is stated
    text = (LEAN / "Choice.lean").read_text()
    assert "minimax theorem" in text and "OPEN" in text and not re.search(r"theorem\s+minimax", text)
    # every Rust test that names a theorem still resolves (the scan of 79.30 covers all sources, these included)
    P30.test_every_law_a_rust_test_says_it_mirrors_is_a_theorem_and_the_lean_files_are_clean()
    print("OBS 79.31 scan: the new laws are theorems with no sorry, no enumeration, no axiom; the minimax theorem is said to be OPEN and is not stated")


def test_the_representation_uses_layer_0s_set_for_the_facts_of_a_position():
    h = (REP / "src" / "history.rs").read_text()
    assert "use cs_base::sets::Set;" in h and "-> Result<Set<Fact>, String>" in h and "pub fn apply_change(set: &mut Set<Fact>" in h
    s = src("sets.rs")
    assert "pub fn remove(&mut self, t: &T)" in s
    print("OBS 79.31 scan: history.rs derives the facts of a position as cs_base's Set")


def test_the_test_is_registered_in_the_gate_after_the_one_of_79_30():
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert gate.index("test_plan79_30") < gate.index("test_plan79_31")
    print("OBS 79.31 scan: the test is in the gate after 79.30")
