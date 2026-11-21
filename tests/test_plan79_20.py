"""PLAN 79.20 THE TESTS OF LAYER 1 CALL NOTHING OF THE RULES (SYSTEM_SPEC 11 "a layer's test uses only that layer and what is below it"; ledger row 36; piece: representation).
Rust holds the tests (cargo test --release --offline -p cs_memory): rust/engine/memory/tests/layer1.rs reads positions as sets of facts from rust/engine/memory/tests/fixtures/*.fixture.
THIS file is the scan: no source of a layer-1 test (every .rs file under rust/engine/memory/tests) may name a crate that holds the rules, the rules' kernel or description, a moving part of the rules' trait
(moves, apply, outcome, initial, find_move, move_name, base_facts) or the game's text form. The one name taken from the crate that holds the rules is the type of a fact, `cs_game::Fact` (it is defined there, beside the trait; plan 79.20 part D).
The fixtures are data: they hold the facts of the games and are not scanned for names, only checked to be there and small.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "rust" / "engine" / "memory" / "tests"

CRATES = r"\b(cs_gdl|cs_decision|cs_tactics|cs_planning|cs_learn|cs_operator)\b"
RULES_PARTS = r"\b(moves|apply|outcome|initial|find_move|move_name|legal_moves|base_facts|terminal|mover)\b"
TEXT_FORM = r"\b(fen|from_fen|san|uci)\b"
RULES_TEXT = r"(\.gdl\b|\brules/|KernelGame|GdlGame|\bGame\b|\bkernel\b|gdl_game)"


def sources() -> list[Path]:
    return sorted(TESTS.rglob("*.rs"))


def code_of(text: str) -> str:
    """the source without its comments (a `//` to the end of the line) and without the text inside strings: a name is found where it is used, not where a sentence says there is none"""
    out = []
    for line in text.splitlines():
        line = re.sub(r'"(\\.|[^"\\])*"', '""', line)
        out.append(line.split("//")[0])
    return "\n".join(out)


def violations(text: str) -> list[str]:
    code = code_of(text)
    found = [m.group(0) for r in (CRATES, RULES_PARTS, TEXT_FORM, RULES_TEXT) for m in re.finditer(r, code)]
    for m in re.finditer(r"cs_game::(\w+|\{[^}]*\})", code):                              # of the crate that holds the rules only the type of a fact is named
        if m.group(1) != "Fact":
            found.append(m.group(0))
    return found


def test_the_sources_of_layer_1_tests_are_there():
    names = {p.name for p in sources()}
    assert {"layer1.rs", "part_b.rs", "mod.rs"} <= names


def test_no_test_of_layer_1_names_a_rules_crate_a_moving_part_of_the_rules_or_the_games_text_form():
    bad = {p.relative_to(ROOT).as_posix(): violations(p.read_text()) for p in sources()}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, bad


def test_the_scan_sees_what_it_is_for():
    """the negative control, in the scan itself: a source that names what is forbidden is found; a comment that says so is not"""
    assert violations("use cs_game::kernel::KernelGame;\nlet s = k.apply(&s, &m);\n")
    assert violations("use cs_gdl::compose;")
    assert violations("let t = k.fen(&s);")
    assert violations("use cs_game::{Fact, Game};")
    assert violations("let m = g.outcome(&s);")
    assert not violations("use cs_game::Fact; // no move, no outcome, no fen here\nlet s = \"apply\";")


def test_the_fixtures_are_there_two_games_and_small():
    files = sorted((TESTS / "fixtures").glob("*.fixture"))
    assert len(files) >= 2
    assert sum(p.stat().st_size for p in files) < 1_000_000
    for p in files:
        head = p.read_text().splitlines()[0]
        assert head.startswith("key\t")
        print(f"OBS 79.20 fixture {p.name}: {p.stat().st_size} bytes")
