"""PLAN 79.36 EVERY UNIT OF LAYERS 0, 1 AND 2 HAS A UNIT TEST, AND EVERYTHING CHECKED BY HAND IS A TEST (SYSTEM_SPEC 11, the last block; ledger rows 43, 41, 36; pieces: base, representation, rules).
Rust holds the tests (rust/engine/base/tests/units_*.rs, rust/engine/rep/tests/units_*.rs, rust/engine/game/tests/units_*.rs, kinds_by_game.rs). THIS file is what belongs to Python: the gate that keeps it so.
  (1) scripts/layer_units_print.py finds every public item of the three crates and the tests that call it; the gate FAILS when a public item is called by no test and is not on the list of exceptions with a cause (scripts/layer_units_exceptions.txt),
      and when a line of the list names an item that is tested or does not exist;
  (2) the program sees what it is for: a planted public item with no test, one named only in a comment or a string, one called by a test that asserts nothing, one called only by an ignored test are all found untested; one called by a test that asserts is not;
  (3) the table of kinds by game is complete: seven games, six kinds, three families, each scenario's sets written out by hand;
  (4) every test marked to be skipped names its cause; the files that must not be touched are not (kernel.rs, gdl_game.rs, gdl/src/lib.rs, ledger.rs, the sources of layer 0, every Cargo.toml and Cargo.lock);
  (5) the answers recorded before the plan and after agree, line by line (when the files are on this machine);
  (6) the test is in the gate after the one of 79.35.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
BEFORE = "d5decd8d"                      # the commit before this plan's code (the plan block's own commit: it changes documents only)
GAMES = ("chess", "kingofthehill", "3check", "chess960", "horde", "atomic", "antichess")
KINDS = ("bishop", "king", "knight", "pawn", "queen", "rook")


def program():
    spec = importlib.util.spec_from_file_location("layer_units_print", ROOT / "scripts" / "layer_units_print.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["layer_units_print"] = mod
    spec.loader.exec_module(mod)
    return mod


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def have_before() -> bool:
    return git("cat-file", "-e", f"{BEFORE}^{{commit}}").returncode == 0


def test_every_public_item_of_the_three_crates_is_called_by_a_test_or_is_an_exception_with_a_cause():
    p = program()
    res = p.inventory()
    names = {"cs_base", "cs_rep", "cs_game"}
    assert names <= set(res), set(res)
    bad = {}
    for crate in sorted(names):
        r = res[crate]
        assert len(r["items"]) > 100, f"{crate}: the program found only {len(r['items'])} items"
        untested = [f"{it.file}:{it.line} {it.name}" for it in r["untested"]]
        if untested:
            bad[crate] = untested[:20]
        print(f"OBS 79.36 gate {crate}: items {len(r['items'])} tested {len(r['tested']) + len(r['weak_only'])} untested {len(r['untested'])} excepted {len(r['excepted'])}")
    assert not bad, bad
    assert not res["_stale_exceptions"], f"the list of exceptions names items that are tested or are not there: {sorted(res['_stale_exceptions'])}"


def test_every_exception_names_a_cause():
    text = (ROOT / "scripts" / "layer_units_exceptions.txt").read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item, _, cause = line.partition("|")
        assert item.strip() and len(cause.strip()) > 10, f"an exception without its cause: {line}"


CONTROL_SRC = """
pub struct Counter { n: u32 }
impl Counter {
    pub fn new() -> Counter { Counter { n: 0 } }
    pub fn bump(&mut self) { self.n += 1; }
    pub fn read(&self) -> u32 { self.n }
    pub fn size(&self) -> u32 { 1 }
    pub(crate) fn hidden(&self) -> u32 { 2 }
}
pub struct Other { n: u32 }
impl Other { pub fn size(&self) -> u32 { 2 } }
pub fn free_one() -> u32 { 1 }
pub fn free_two() -> u32 { 2 }
pub fn free_three() -> u32 { 3 }
pub fn free_four() -> u32 { 4 }
pub const LIMIT: u32 = 9;
"""

CONTROL_TESTS = """
use ctl::*;
// free_one() in a comment is no call
#[test]
fn names_a_function_only_in_a_string_and_a_comment() {
    let s = "free_one() and Counter::new()";
    assert_eq!(s.len(), 29);
}
#[test]
fn calls_with_no_assertion() { let _ = free_two(); }
#[test]
#[ignore = "skipped on purpose"]
fn calls_but_is_skipped() { assert_eq!(free_three(), 3); }
#[test]
fn calls_and_asserts() {
    assert_eq!(free_four(), 4);
    let mut c = Counter::new();
    Counter::bump(&mut c);
    assert_eq!(Counter::read(&c), 1);
    assert_eq!(LIMIT, 9);
}
"""


def make_control(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "lib.rs").write_text(CONTROL_SRC)
    (tmp_path / "tests" / "t.rs").write_text(CONTROL_TESTS)
    return tmp_path


def test_the_program_sees_what_it_is_for_a_planted_item_with_no_test_is_found_and_one_with_an_asserting_test_is_not(tmp_path):
    p = program()
    items, ntests, nasserting = p.analyse("ctl", make_control(tmp_path))
    by = {it.name: it for it in items}
    assert "Counter::hidden" not in by, "a `pub(crate)` method is no item"
    assert by["free_one"].tests == set() and by["free_one"].weak == set(), "named only in a string and a comment: not called"
    assert by["free_two"].tests == set(), "called by a test that asserts nothing: not tested"
    assert by["free_three"].tests == set(), "called only by a test that is skipped: not tested"
    assert by["free_four"].tests, "called by a test that asserts: tested"
    assert by["Counter::new"].tests and by["Counter::bump"].tests and by["Counter::read"].tests and by["LIMIT"].tests
    assert by["Counter::size"].tests == set() and by["Other::size"].tests == set(), "two types share the name `size` and no test calls either"
    assert (ntests, nasserting) == (3, 2)
    print("OBS 79.36 gate controls: a planted item with no test, one named in a string, one called by a test that asserts nothing and one called only by a skipped test are each found untested")


def test_two_types_that_share_a_method_name_are_told_apart_by_the_path_of_the_call(tmp_path):
    p = program()
    root = make_control(tmp_path)
    (root / "tests" / "t.rs").write_text(CONTROL_TESTS + """
#[test]
fn calls_the_one_by_its_type() { let c = Counter::new(); assert_eq!(Counter::size(&c), 1); }
""")
    items, _, _ = p.analyse("ctl", root)
    by = {it.name: it for it in items}
    assert by["Counter::size"].tests, "the path names the type"
    assert by["Other::size"].tests == set(), "the other type of that name is not credited by a call that names the first"


def test_the_table_of_kinds_by_game_is_complete():
    src = (ENGINE / "game" / "tests" / "kinds_by_game.rs").read_text()
    m = re.search(r'pub const GAMES: \[&str; (\d+)\] = \[([^\]]*)\];', src)
    assert m and int(m.group(1)) == 7
    assert tuple(re.findall(r'"([^"]+)"', m.group(2))) == GAMES
    k = re.search(r'pub const KINDS: \[&str; (\d+)\] = \[([^\]]*)\];', src)
    assert k and tuple(re.findall(r'"([^"]+)"', k.group(2))) == KINDS
    f = re.search(r'pub const FAMILIES: \[&str; (\d+)\] = \[([^\]]*)\];', src)
    assert f and tuple(re.findall(r'"([^"]+)"', f.group(2))) == ("alone", "own in the way", "other in the way")
    # every kind has a scenario in every family (the Rust test of the same name counts them; here the table is read as text)
    rows = re.findall(r's\("[^"]+", "(\w+)", "([a-z ]+)",', src)
    for kind in KINDS:
        for family in ("alone", "own in the way", "other in the way"):
            assert (kind, family) in set(rows), f"no scenario of {kind} in the family {family}"
    for need in ("the_table_has_each_kind_in_each_family_for_each_game", "king_of_the_hill_and_three_check_move_every_kind_as_chess_does_the_same_table", "chess960_moves_every_kind_as_chess_does_the_same_table",
                 "horde_moves_every_kind_as_chess_does", "atomic_moves_every_kind_as_chess_does", "antichess_moves_every_kind_as_chess_does", "the_smallest_kind_on_the_first_row",
                 "in_atomic_the_leader_beside_a_unit", "in_antichess_a_taking_that_is_possible", "in_antichess_a_smallest_unit_reaching_the_last_row"):
        assert need in src, need
    for game in GAMES:
        assert f'"{game}"' in src, game
    assert src.count("OBS 79.36 D") >= 7
    print(f"OBS 79.36 scan: kinds_by_game.rs holds {len(rows)} scenarios over {len(KINDS)} kinds and 3 families, run on {len(GAMES)} games")


def test_every_test_marked_to_be_skipped_names_its_cause_and_the_new_test_files_exist():
    marks = {}
    for crate in ("base", "rep", "game"):
        for p in sorted((ENGINE / crate / "tests").rglob("*.rs")) + sorted((ENGINE / crate / "src").rglob("*.rs")):
            text = p.read_text()
            for m in re.finditer(r"#\[ignore([^\]]*)\]", text):
                cause = m.group(1).strip()
                assert cause.startswith("=") and len(cause) > 20, f"{p.relative_to(ROOT)}: a test is skipped with no cause: {m.group(0)}"
                marks[p.relative_to(ROOT).as_posix()] = marks.get(p.relative_to(ROOT).as_posix(), 0) + 1
    for need in ("base/tests/units_choice.rs", "base/tests/units_walk.rs", "base/tests/units_fold.rs", "base/tests/units_logic.rs", "rep/tests/units_values.rs", "rep/tests/units_store.rs", "rep/tests/units_board.rs",
                 "game/tests/units_game.rs", "game/tests/units_builds.rs", "game/tests/units_hand_checked.rs", "game/tests/kinds_by_game.rs"):
        assert (ENGINE / need).exists(), need
    print(f"OBS 79.36 scan: tests marked to be skipped, each with its cause: {marks}")


def test_the_files_that_must_not_move_do_not_move_and_no_dependency_changes():
    if not have_before():
        pytest.skip(f"the commit {BEFORE} is not in this clone")
    frozen = ["rust/engine/gdl/src/lib.rs", "rust/engine/game/src/gdl_game.rs", "rust/engine/game/src/kernel.rs", "rust/engine/tactics/src/ledger.rs", "rust/engine/base/src",
              "rust/engine/Cargo.lock", "rust/engine/Cargo.toml"] + [f"rust/engine/{c}/Cargo.toml" for c in ("base", "rep", "memory", "game", "gdl", "value", "tactics", "decision", "planning", "learn", "py", "operator")]
    d = git("diff", "--stat", BEFORE, "--", *frozen)
    assert d.returncode == 0 and d.stdout.strip() == "", d.stdout
    print("OBS 79.36 scan: kernel.rs, gdl_game.rs, gdl/src/lib.rs, ledger.rs, the sources of layer 0, Cargo.lock and every Cargo.toml are the files of the commit before")


def test_the_recorded_dumps_agree_before_and_after_when_they_are_on_this_machine():
    from catspace.io import paths
    f32 = Path(paths.experiment("whole_79_32/x")).parent
    f33 = Path(paths.experiment("whole_79_33/x")).parent
    ran = 0
    if (f32 / "before36.goals.dump").exists() and (f32 / "after36.goals.dump").exists():
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "whole_79_32_compare.py"), "before36", "after36"], capture_output=True, text=True, cwd=ROOT)
        print(r.stdout[-1500:])
        assert r.returncode == 0, r.stdout[-2000:]
        ran += 1
    kinds = [k for k in ("values", "toward", "tail", "frame") if (f33 / f"before36.{k}.dump").exists() and (f33 / f"after36.{k}.dump").exists()]
    if kinds:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "whole_79_33_compare.py"), "before36", "after36", *kinds], capture_output=True, text=True, cwd=ROOT)
        print(r.stdout[-1500:])
        assert r.returncode == 0, r.stdout[-2000:]
        ran += 1
    if not ran:
        pytest.skip("the dumps of plans 79.32 and 79.33 are not on this machine: made by scripts/whole_79_32_run.sh and whole_79_33_run.sh dump <kind> before36 / after36")


def test_the_test_is_registered_in_the_gate_after_the_one_of_79_35():
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert gate.index("test_plan79_35") < gate.index("test_plan79_36")
    print("OBS 79.36 scan: the test is in the gate after 79.35")
