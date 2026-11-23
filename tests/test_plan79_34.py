"""PLAN 79.34 THE CODE FOLLOWS THE CUT OF LAYERS 1 AND 2: WHAT IS SEEN IN ONE STATE, AND WHAT THE RULES SAY OF IT (SYSTEM_SPEC 11 "Layers 0, 1 and 2 by what they hold", "The cut of layers 1 and 2 is carried into the code"; ledger rows 37, 36).
Rust holds the tests (rust/engine/rep/tests/seen.rs for layer 1, rust/engine/game/tests/rules_by_kind.rs and relations.rs for layer 2). THIS file is what belongs to Python: the scans.
  (1) the sources of the representation crate (layer 1) name none of the rules' relations (bear, may_move, shut, through) nor the module that made their facts, and the crate does not depend on the rules crate, not even as a dev-dependency;
  (2) the rules crate (layer 2) holds them: the names, the making of the lines through and the profile of the kinds, and the fact form;
  (3) the interface layer 1 keeps for a layer above to write relations is static (a value, no boxed function, no trait object);
  (4) the two test files exist and print OBS lines; the layer 1 test reads no starting arrangement and names no game in its code;
  (5) the answers recorded before the change and after agree, line by line (when the files are on this machine);
  (6) the files that must not be touched are not;
  (7) the test is in the gate after the one of 79.33.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
REP = ENGINE / "rep"
GAME = ENGINE / "game"

# the words of the rules' relations and of the module that made their fact form; whole words, in the code and in the string literals of layer 1 (comments aside)
RULE_WORDS = re.compile(r"\b(bears?|may_move|shuts?|throughs?|derive_throughs|graph_facts|FactReader|struck_by|MayMove|Bear|Shut|Through)\b")


def code_and_strings(text: str) -> str:
    """A Rust source without its comments (strings kept: a word in a string is a word of the layer)."""
    out = []
    for line in text.splitlines():
        in_str, cut = False, len(line)
        i = 0
        while i < len(line):
            c = line[i]
            if c == "\\" and in_str:
                i += 2
                continue
            if c == '"':
                in_str = not in_str
            elif not in_str and line.startswith("//", i):
                cut = i
                break
            i += 1
        out.append(line[:cut])
    return "\n".join(out)


def test_layer_1_sources_name_none_of_the_rules_relations_and_the_crate_does_not_depend_on_the_rules():
    sources = sorted((REP / "src").rglob("*.rs")) + sorted((REP / "tests").rglob("*.rs"))
    assert {"graph.rs", "live.rs", "history.rs", "fact.rs"} <= {p.name for p in sources}
    assert not (REP / "src" / "graph_facts.rs").exists(), "the fact form of the rules' relations is layer 2's"
    bad = {}
    for p in sources:
        found = [m.group(0) for m in RULE_WORDS.finditer(code_and_strings(p.read_text()))]
        if found:
            bad[p.relative_to(ROOT).as_posix()] = sorted(set(found))
    assert not bad, bad
    toml = (REP / "Cargo.toml").read_text()
    assert "cs_game" not in toml and "cs_tactics" not in toml and "shakmaty" not in toml, "no rules crate, not even as a dev-dependency"
    print(f"OBS 79.34 scan: {len(sources)} sources of layer 1 (code and strings) hold none of the rules' relations; Cargo.toml names no crate of the rules")


def test_the_scan_sees_a_planted_word():
    assert RULE_WORDS.search(code_and_strings('let w = "bear";'))
    assert RULE_WORDS.search(code_and_strings("p.bears.iter()"))
    assert RULE_WORDS.search(code_and_strings("use crate::graph_facts::FactReader;"))
    assert not RULE_WORDS.search(code_and_strings("// the unit bears on a place, in a comment\nlet next = 1;"))


def test_the_rules_crate_holds_the_relations_their_making_and_their_fact_form():
    rel = (GAME / "src" / "relations.rs").read_text()
    for need in ('"bear"', '"may_move"', '"shut"', '"through"', "pub trait PartRelations", "pub trait UnitRelations", "pub const BEAR", "pub const THROUGH"):
        assert need in rel, need
    graph = (GAME / "src" / "graph.rs").read_text()
    for need in ("pub fn derive_throughs", "pub fn learn", "pub fn continues", "pub fn part_of", "pub cont:"):
        assert need in graph, need
    facts = (GAME / "src" / "graph_facts.rs").read_text()
    assert "pub struct FactReader" in facts and 'TEMPO: &str = "tempo"' in facts and "MAY_MOVE_REL.word" in facts
    print("OBS 79.34 scan: the rules crate holds the names, the lines through, the profile of the kinds and the fact form of bear, may_move, shut, through and the turn")


def test_the_door_layer_1_keeps_is_static():
    g = (REP / "src" / "graph.rs").read_text() + (REP / "src" / "live.rs").read_text()
    code = code_and_strings(g)
    assert not re.search(r"Box\s*<\s*dyn|&\s*(mut\s+)?dyn\b|\bdyn\s+Fn", code), "no boxed function and no trait object in layer 1's graph"
    assert "pub fn write(&self, part: &mut PositionPart, rel: Rel," in g, "the door takes the relation as a value"
    assert "pub struct Rel" in g and "pub const REL_SLOTS" in g
    print("OBS 79.34 scan: layer 1's door `Board::write` takes the relation as a value (Rel); no boxed function or trait object")


def test_the_two_test_files_exist_print_obs_lines_and_layer_1_reads_no_starting_arrangement():
    seen = (REP / "tests" / "seen.rs").read_text()
    assert seen.count("OBS 79.34 D") >= 6 and (REP / "tests" / "fixtures" / "seen_chess.expect").exists()
    # the start is layer 2's: the layer 1 test never asks for it
    assert not re.search(r"\binitial\b|\bstart(ing)?\b.*position_from", code_and_strings(seen))
    rules = (GAME / "tests" / "rules_by_kind.rs").read_text()
    assert rules.count("OBS 79.34 E") >= 9
    for need in ("the_start_puts_each_unit", "each_kind_alone_on_the_board", "a_unit_of_its_own_side_in_the_way", "a_unit_of_the_other_side_in_the_way", "the_smallest_kind_may_advance_two_places", "taking_in_passing",
                 "with_the_corner_unit", "reaching_the_last_row", "the_turn_passes", "each_end_of_the_game", "the_kernels_own_moves"):
        assert need in rules, need
    assert "OBS 79.34 E relations" in (GAME / "tests" / "relations.rs").read_text()
    print("OBS 79.34 scan: both test files exist and print OBS lines; the layer 1 test asks layer 1 for no starting arrangement")


def test_the_recorded_dumps_agree_before_and_after_when_they_are_on_this_machine():
    from catspace.io import paths
    f32 = Path(paths.experiment("whole_79_32/x")).parent
    f33 = Path(paths.experiment("whole_79_33/x")).parent
    ran = 0
    if (f32 / "before34.goals.dump").exists() and (f32 / "after34.goals.dump").exists():
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "whole_79_32_compare.py"), "before34", "after34"], capture_output=True, text=True, cwd=ROOT)
        print(r.stdout[-1500:])
        assert r.returncode == 0, r.stdout[-2000:]
        ran += 1
    kinds = [k for k in ("values", "toward", "tail", "frame") if (f33 / f"before34.{k}.dump").exists() and (f33 / f"after34.{k}.dump").exists()]
    if kinds:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "whole_79_33_compare.py"), "before34", "after34", *kinds], capture_output=True, text=True, cwd=ROOT)
        print(r.stdout[-1500:])
        assert r.returncode == 0, r.stdout[-2000:]
        ran += 1
    if not ran:
        pytest.skip("the dumps of plans 79.32 and 79.33 are not on this machine: made by scripts/whole_79_32_run.sh and whole_79_33_run.sh dump <kind> before34 / after34")


def test_the_files_that_must_not_move_do_not_name_the_new_door():
    # the kernel, the description evaluator, the old player's solver and the description reader hold no word of the new interface (they were not touched)
    for rel in ("game/src/kernel.rs", "game/src/gdl_game.rs", "gdl/src/lib.rs", "tactics/src/ledger.rs"):
        text = (ENGINE / rel).read_text()
        assert "Rel::new" not in text and "PartRelations" not in text and "UnitRelations" not in text, rel
    print("OBS 79.34 scan: kernel.rs, gdl_game.rs, gdl/src/lib.rs and ledger.rs hold none of the new interface")


def test_the_test_is_registered_in_the_gate_after_the_one_of_79_33():
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert gate.index("test_plan79_33") < gate.index("test_plan79_34")
    print("OBS 79.34 scan: the test is in the gate after 79.33")
