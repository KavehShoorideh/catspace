"""PLAN 79.27 LAYER 4 TAKES AN ORDERED LIST OF GOALS (SYSTEM_SPEC 11 the layers, "Layer 4 takes a goal as an input" and its amendment; ledger rows 36, 14; piece: tactical reasoner).
Rust holds the tests (cargo test --release --offline -p cs_tactics --test layer4_goals --test layer4 -p cs_rep): layer4_goals.rs asks the entry `meansends::ask` with lists of statements over layer 1's graph facts on two games against a slow plain reference written apart
in the test, and the two old goals through the entry against the ledger search. THIS file is what belongs to Python: the scans of the new data and its makers, and that the layer's new files name no object of a game.
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
LAYER_FILES = [TACTICS / "src" / "meansends.rs", TACTICS / "src" / "reach.rs", ENGINE / "game" / "src" / "graph_facts.rs", TACTICS / "tests" / "layer4_goals.rs"]
GAME_WORDS = r"\b(king\w*|queen\w*|rook\w*|bishop\w*|knight\w*|pawn\w*|castl\w*|promot\w*|en passant|check\w*|mate|stalemate|uci|fen|san|chess\w*|shakmaty)\b"
SQUARE = r"\b[a-h][1-8]\b"


def test_the_goals_fixture_holds_two_games_with_lists_of_patterns_and_no_verdict():
    lines = (FIXTURES / "goals.fixture").read_text().splitlines()
    assert lines[0].startswith("note\t")
    games = [l.split("\t")[1] for l in lines if l.startswith("game\t")]
    assert len(games) == 2 and len(set(games)) == 2
    pos = [l.split("\t") for l in lines if l.startswith("pos\t")]
    assert len(pos) >= 60
    kinds = set()
    goals = 0
    for p in pos:
        assert len(p) == 4
        for g in json.loads(p[3]):
            kinds.add(g["kind"])
            goals += 1
            assert g["pattern"] and all(set(l) == {"relation", "args", "neg"} for l in g["pattern"])
            assert {l["relation"] for l in g["pattern"]} <= {"unit", "bear", "may_move", "on_line", "on_edge", "tempo", "differ"}   # plan 79.28: layer 1's differ literal
    assert len(kinds) >= 4
    assert not any(re.search(r"\b(reached|out of reach|verdict)\b", l.split("\t", 1)[1]) for l in lines if l.startswith("pos\t")), "no verdict is written"
    print(f"OBS 79.27 scan: goals.fixture holds {len(pos)} positions of games {games}, {goals} goals of {len(kinds)} kinds ({sorted(kinds)}), patterns over layer 1's graph relations, no verdict")


def test_the_pairs_fixture_is_the_references_and_names_positions_of_the_goals_fixture():
    lines = (FIXTURES / "pairs.fixture").read_text().splitlines()
    assert lines[0].startswith("note\t") and "reference" in lines[0]
    ids = {l.split("\t")[1] for l in (FIXTURES / "goals.fixture").read_text().splitlines() if l.startswith("pos\t")}
    pairs = [l.split("\t") for l in lines if l.startswith("pair\t")]
    assert len(pairs) >= 10 and all(p[1] in ids and p[2].isdigit() and p[3].isdigit() and int(p[2]) < int(p[3]) for p in pairs)
    print(f"OBS 79.27 scan: pairs.fixture holds {len(pairs)} (position, goal, goal) triples the reference found one line to reach")


def test_the_layer_files_name_no_object_of_a_game_and_the_test_is_bounded_by_nodes():
    for p in LAYER_FILES:
        txt = p.read_text()
        bad = [m.group(0) for m in re.finditer(GAME_WORDS, txt, flags=re.I)] + [m.group(0) for m in re.finditer(SQUARE, txt)]
        assert not bad, (p.name, bad)
    t = (TACTICS / "tests" / "layer4_goals.rs").read_text()
    assert "seconds: None" in t and "seconds: Some" not in t
    assert "BUDGETS" in t and "Reference" in t and "LAYER4_MAKE_PAIRS" in t
    src = (TACTICS / "src" / "meansends.rs").read_text()
    assert "pub fn ask<" in src and "pub enum Goal" in src and "OutOfReach" in src and "NotSettled" in src
    print("OBS 79.27 scan: the entry, the body, layer 1's facts reader and the test name no object of a game; the test reads budgets of nodes and a reference of its own")


def test_the_goals_maker_is_a_printing_script_and_fits_nothing():
    p = DISCOVERY / "layer4_goals_print.py"
    assert p.name.endswith("_print.py")
    t = p.read_text()
    assert not re.search(r"sklearn|\.fit\(|torch|urlopen|requests|anthropic|stockfish", t, flags=re.I)
    assert "random.Random(" in t, "seeded"
    print("OBS 79.27 scan: the goals maker is seeded, fits nothing, reads no outside engine")


def test_no_dependency_moved_and_the_layers_depend_downward_only():
    lock = (ENGINE / "Cargo.lock").read_text()
    assert "cs_tactics" in lock
    def deps(p: Path) -> set:
        return {l.split("=")[0].strip() for l in p.read_text().splitlines() if l.startswith("cs_") and "path" in l}

    assert not deps(TACTICS / "Cargo.toml") & {"cs_decision", "cs_learn", "cs_planning", "cs_memory"}, "layer 4 depends on nothing above it"
    assert not deps(ENGINE / "rep" / "Cargo.toml") & {"cs_game", "cs_tactics", "cs_decision"}
    print("OBS 79.27 scan: cs_tactics names no crate above it; cs_rep names no crate that holds the rules")
