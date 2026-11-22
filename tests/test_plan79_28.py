"""PLAN 79.28 A GOAL IS REACHED ONLY IF IT OUTLASTS THE REPLY (SYSTEM_SPEC 11 "Reached means the goal outlasts the reply", "Layer 4 takes a goal as an input"; ledger rows 36, 14; piece: tactical reasoner).
Rust holds the tests (cargo test --release --offline -p cs_rep -p cs_tactics): rust/engine/tactics/tests/layer4_goals.rs asks the entry `meansends::ask` against a slow plain reference written apart in the new meaning (held, reached, touched), and
rust/engine/rep/tests/layer1.rs reads layer 1's new `differ` literal against its plain matcher. THIS file is what belongs to Python: the scans of the remade fixture and of the printing scripts, the fixed rule that chooses the lines a reader sees, and that the layer's files
name no object of a game.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
TACTICS = ENGINE / "tactics"
FIXTURES = TACTICS / "tests" / "fixtures"
DISCOVERY = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
LAYER_FILES = [TACTICS / "src" / "meansends.rs", TACTICS / "src" / "reach.rs", ENGINE / "rep" / "src" / "matcher.rs", TACTICS / "tests" / "layer4_goals.rs"]
GAME_WORDS = r"\b(king\w*|queen\w*|rook\w*|bishop\w*|knight\w*|pawn\w*|castl\w*|promot\w*|en passant|check\w*|mate|stalemate|uci|fen|san|chess\w*|shakmaty)\b"
SQUARE = r"\b[a-h][1-8]\b"


def goals() -> list:
    out = []
    for l in (FIXTURES / "goals.fixture").read_text().splitlines():
        if l.startswith("pos\t"):
            f = l.split("\t")
            out += [(f[1], g["kind"], g["pattern"]) for g in json.loads(f[3])]
    return out


def test_the_remade_goals_fixture_holds_five_kinds_of_eighty_and_two_units_are_said_with_differ():
    gs = goals()
    kinds: dict = {}
    for _, k, _ in gs:
        kinds[k] = kinds.get(k, 0) + 1
    assert len(kinds) == 5 and set(kinds.values()) == {80}, kinds
    two = [p for _, k, p in gs if k == "two_bear"]
    assert two and all(sum(l["relation"] == "differ" for l in p) == 1 for p in two), "every two-units goal says the two units differ"
    for p in two:
        d = [l for l in p if l["relation"] == "differ"][0]
        assert len(d["args"]) == 2 and all(a.startswith("?") for a in d["args"]) and not d["neg"]
        bound = {a for l in p if l["relation"] == "bear" for a in l["args"] if a.startswith("?")}
        assert set(d["args"]) <= bound, "a placeholder of differ is bound by a row literal"
    assert {l["relation"] for _, _, p in gs for l in p} <= {"unit", "bear", "may_move", "on_line", "on_edge", "tempo", "differ"}
    old = (FIXTURES / "goals_79_27.fixture").read_text()
    assert old.count("\npos\t") == 80 and '"differ"' not in old, "the questions of plan 79.27 are kept as they were"
    print(f"OBS 79.28 scan: goals.fixture holds {len(gs)} goals of {len(kinds)} kinds ({sorted(kinds)}), {len(two)} of them with a differ literal; goals_79_27.fixture is the earlier fixture kept")


def test_the_pairs_fixture_is_the_references_and_names_positions_of_the_goals_fixture():
    lines = (FIXTURES / "pairs.fixture").read_text().splitlines()
    assert lines[0].startswith("note\t") and "reference" in lines[0] and "79.28" in lines[0]
    ids = {l.split("\t")[1] for l in (FIXTURES / "goals.fixture").read_text().splitlines() if l.startswith("pos\t")}
    pairs = [l.split("\t") for l in lines if l.startswith("pair\t")]
    assert len(pairs) >= 10 and all(p[1] in ids and p[2].isdigit() and p[3].isdigit() and int(p[2]) < int(p[3]) for p in pairs)
    print(f"OBS 79.28 scan: pairs.fixture holds {len(pairs)} (position, goal, goal) triples the reference found one line to reach")


def test_the_layer_files_name_no_object_of_a_game_and_the_test_has_the_new_answers():
    for p in LAYER_FILES:
        txt = p.read_text()
        bad = [m.group(0) for m in re.finditer(GAME_WORDS, txt, flags=re.I)] + [m.group(0) for m in re.finditer(SQUARE, txt)]
        assert not bad, (p.name, bad)
    t = (TACTICS / "tests" / "layer4_goals.rs").read_text()
    assert "seconds: None" in t and "seconds: Some" not in t
    for word in ("BUDGETS", "Reference", "LAYER4_MAKE_PAIRS", "reach_or_touch", "Touched", "KIND_FLOOR", "plies_old"):
        assert word in t, word
    src = (TACTICS / "src" / "meansends.rs").read_text()
    assert "pub struct Touched" in src and "Touched(Touched)" in src and "OutOfReach" in src and "NotSettled" in src
    lem = (ENGINE / "rep" / "src" / "lemma.rs").read_text()
    assert 'pub const DIFFER: &str = "differ"' in lem
    print("OBS 79.28 scan: the entry, the body, layer 1's language and matcher and both tests name no object of a game; the answer has its fourth case, the language its one literal that is not a row")


def test_the_two_makers_are_printing_scripts_that_fit_nothing():
    for name in ("layer4_goals_print.py", "layer4_lines_print.py"):
        p = DISCOVERY / name
        assert p.name.endswith("_print.py")
        t = p.read_text()
        assert not re.search(r"sklearn|\.fit\(|torch|urlopen|requests|anthropic|stockfish", t, flags=re.I)
    assert "random.Random(" in (DISCOVERY / "layer4_goals_print.py").read_text(), "seeded"
    assert "paths.experiment" in (DISCOVERY / "layer4_lines_print.py").read_text()
    print("OBS 79.28 scan: the goals maker is seeded; the lines maker writes through catspace.io.paths; neither fits anything")


def test_the_lines_for_a_reader_are_chosen_by_the_fixed_rule():
    from catspace.incubator.analyst.discovery import layer4_lines_print as lp

    kinds = ["bears", "may_take", "on_edge", "on_line", "two_bear"]
    rows = [{"status": s, "kind": k, "position": f"p{i}", "goal": 0, "plies": 1 + (i % 3)} for s in ("reached", "touched") for k in kinds for i in range(4)]
    rows += [{"status": "reached", "kind": "bears", "position": "root", "goal": 0, "plies": 0}]
    r = lp.pick(rows, "reached", 12)
    assert len(r) == 12 and all(x["plies"] >= 1 for x in r), "a goal held at the root has no line to read and is never chosen"
    assert [x["kind"] for x in r[:5]] == kinds, "one of each kind in turn, the kinds in alphabetical order"
    assert [x["plies"] for x in r if x["kind"] == "bears"] == sorted((x["plies"] for x in r if x["kind"] == "bears"), reverse=True), "the longest lines first within a kind"
    assert r == lp.pick(rows, "reached", 12), "the rule is a function of the answers"
    assert len(lp.pick(rows, "touched", 6)) == 6
    assert len(lp.pick([x for x in rows if x["kind"] == "bears"], "touched", 6)) == 4, "fewer when a kind has no more"
    print("OBS 79.28 scan: twelve reached and six touched are picked round the kinds, longest lines first, none held at the root")


def test_no_dependency_moved_and_the_layers_depend_downward_only():
    lock = (ENGINE / "Cargo.lock").read_text()
    assert "cs_tactics" in lock
    def deps(p: Path) -> set:
        return {l.split("=")[0].strip() for l in p.read_text().splitlines() if l.startswith("cs_") and "path" in l}

    assert not deps(TACTICS / "Cargo.toml") & {"cs_decision", "cs_learn", "cs_planning", "cs_memory"}, "layer 4 depends on nothing above it"
    assert not deps(ENGINE / "rep" / "Cargo.toml") & {"cs_game", "cs_tactics", "cs_decision"}
    print("OBS 79.28 scan: cs_tactics names no crate above it; cs_rep names no crate that holds the rules")
