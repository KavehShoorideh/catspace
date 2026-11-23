"""PLAN 79.33 ONE GENERAL ANSWER FOR LAYER 0'S WALK, AND THE OTHER WALKS AS CASES OF IT (SYSTEM_SPEC 11 "What plan 79.32 left of one algorithm", 13 item 51; ledger rows 37, 36).
Rust holds the tests (rust/engine/base/tests/value_dump.rs, fold_steps.rs, rust/engine/tactics/tests/walk_values.rs). THIS file is what belongs to Python: the scans.
  (1) the moved walks hold no recursion of their own over options: the search toward claims (toward.rs), the tail of a line (lines.rs `settle_tail`) and the other side's gain (forcing.rs `their_gain`) are cases of `cs_base::choice::fold`;
  (2) cs_base still depends on nothing and names no game and no word of a world (the scans of 79.30, 79.31 and 79.32 run again over the new file), and its interface is static;
  (3) the general walk's interface is the one written first (Node, Kind, Bounds, Fold) and the steps walk's node is the same `Node`;
  (4) the tests of speed and of the recorded answers exist and print OBS lines;
  (5) the answers recorded before the change and after agree, line by line (when the files are on this machine);
  (6) the walks that were NOT moved are named, with their copies still in place (the explicit tree's value, the search inside a frame, the steps walk), and the old player's solver is untouched;
  (7) the test is in the gate after the one of 79.32.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "rust" / "engine" / "base"
TACTICS = ROOT / "rust" / "engine" / "tactics"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_scan", ROOT / "tests" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P30 = _load("test_plan79_30")
P31 = _load("test_plan79_31")


def code(path: Path) -> str:
    """A Rust source without its comments and strings (what the compiler reads)."""
    t = path.read_text()
    t = re.sub(r"//[^\n]*", "", t)
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', t)


def test_the_moved_walks_hold_no_recursion_of_their_own_over_options():
    toward = code(TACTICS / "src" / "toward.rs")
    assert "cs_base::choice::fold" in (TACTICS / "src" / "toward.rs").read_text()
    assert not re.search(r"\bfn value\b", toward) and not re.search(r"\bself\.value\(", toward), "the recursion is layer 0's"
    assert not re.search(r"\.path\.contains|path\.contains\(", toward) or "fn before" in toward, "the line's repeat is read in the kind's `before`"
    lines = code(TACTICS / "src" / "lines.rs")
    assert "cs_base::choice::fold" in (TACTICS / "src" / "lines.rs").read_text()
    body = lines[lines.index("pub fn settle_tail"):lines.index("fn read_on")]
    assert not re.search(r"\bfn go\b", body) and not re.search(r"\bgo\(g, probe", body), "the tail's recursion is layer 0's"
    forcing = code(TACTICS / "src" / "sheet" / "forcing.rs")
    assert "cs_base::choice::fold" in (TACTICS / "src" / "sheet" / "forcing.rs").read_text()
    assert "fn their_gain_read" not in forcing and not re.search(r"self\.their_gain\(&c\)", forcing), "the other side's gain is read by layer 0's walk"
    stepping = (TACTICS / "src" / "stepping.rs").read_text()
    assert "impl<G: Game> Node for Stepped" in stepping
    print("OBS 79.33 scan: toward.rs, lines.rs `settle_tail` and forcing.rs `their_gain` hold no recursion over options of their own; they walk layer 0's Fold over the game's positions (stepping.rs)")


def test_cs_base_still_depends_on_nothing_and_names_no_game_with_the_new_file_and_the_interface_is_static():
    P31.test_cs_base_depends_on_nothing_at_all()
    P31.test_cs_base_names_no_game_and_no_world_word_and_the_layer_4_test_of_the_meaning_no_game()
    P30.test_cs_base_sources_name_no_game_and_no_word_of_a_world()
    names = {p.name for p in (BASE / "src" / "choice").glob("*.rs")}
    assert {"walk.rs", "tree_space.rs", "mod.rs", "lp.rs", "fold.rs"} <= names, names
    c = code(BASE / "src" / "choice" / "fold.rs")
    assert not re.search(r"Box\s*<\s*dyn|&\s*(mut\s+)?dyn\b|\bdyn\s+Fn", c), "no boxed function and no trait object"
    print("OBS 79.33 scan: cs_base holds no dependency, no game and no word of a world with fold.rs; the general walk is static")


def test_the_interface_of_the_general_walk_is_the_one_written_first():
    f = (BASE / "src" / "choice" / "fold.rs").read_text()
    for sig in ("pub trait Node", "pub trait Kind<S: Node>", "pub struct Bounds<V>", "pub enum Pre<V, T>", "pub struct Fold<S: Node, K: Kind<S>>",
                "pub fn go(&mut self, left: u32, ply: u32, win: Bounds<K::Val>) -> Result<K::Val, K::Err>",
                "fn before(&mut self, space: &S, left: u32, ply: u32, win: &mut Bounds<Self::Val>)", "fn finish(&mut self, space: &S, token: Self::Token, got: Option<Self::Val>, entry: &Bounds<Self::Val>)",
                "const WINDOWED: bool;", "const CHANCE: bool;"):
        assert sig in f, sig
    w = (BASE / "src" / "choice" / "walk.rs").read_text()
    assert "pub trait Space: Node" in w, "the steps walk's node extends the shared stepping"
    print("OBS 79.33 scan: the interface of cs_base::choice::fold is as committed first; walk::Space extends fold::Node")


def test_the_tests_of_speed_and_of_the_recorded_answers_exist_and_print_obs_lines():
    v = (BASE / "tests" / "value_dump.rs").read_text()
    assert "OBS 79.33 cs_base value" in v and "LAYER0_VALUES" in v
    s = (BASE / "tests" / "fold_steps.rs").read_text()
    assert "OBS 79.33 steps as a case" in s
    t = (TACTICS / "tests" / "walk_values.rs").read_text()
    for o in ("OBS 79.33 walk toward", "OBS 79.33 walk tail", "OBS 79.33 walk frame"):
        assert o in t, o
    for env in ("WALK_TOWARD", "WALK_TAIL", "WALK_FRAME"):
        assert env in t, env
    assert "OBS 79.33 forcing" in (ROOT / "scripts" / "whole_79_33_forcing_print.py").read_text()
    print("OBS 79.33 scan: the speed tests print OBS lines and the answers are written behind env variables for the comparison")


def test_the_answers_recorded_before_the_change_and_the_answers_now_agree_line_by_line():
    from catspace.io import paths
    folder = Path(paths.experiment("whole_79_33/x")).parent
    kinds = [k for k in ("values", "toward", "tail", "frame", "forcing") if (folder / f"before.{k}.dump").exists() and (folder / f"after.{k}.dump").exists()]
    if not kinds:
        pytest.skip(f"the recorded answers are not on this machine ({folder}): they are made by scripts/whole_79_33_run.sh dump <kind> before / after")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "whole_79_33_compare.py"), "before", "after", *kinds], capture_output=True, text=True, cwd=ROOT)
    print(r.stdout)
    assert r.returncode == 0, r.stdout[-2000:]


def test_the_walks_not_moved_are_named_and_the_old_players_solver_is_untouched():
    # what stayed a walk of its own (plan 79.33 BUILT block says why): the explicit tree's value fold, the search inside a frame, the steps walk (a case of its own interface)
    assert re.search(r"pub fn value<", code(BASE / "src" / "choice" / "tree_space.rs")) or "fn value" in code(BASE / "src" / "choice" / "tree_space.rs")
    frame = code(TACTICS / "src" / "sheet" / "frame.rs")
    assert "fn search" in frame
    assert (BASE / "src" / "choice" / "walk.rs").exists()
    assert "cs_base::choice::fold" not in (TACTICS / "src" / "ledger.rs").read_text(), "the old player's solver is a copy and stays one"
    assert "choice::fold" not in (ROOT / "rust" / "engine" / "game" / "src" / "kernel.rs").read_text()
    print("OBS 79.33 scan: the value fold of the explicit tree, the search inside a frame and the steps walk are still walks of their own; ledger.rs does not use the general walk")


def test_the_test_is_registered_in_the_gate_after_the_one_of_79_32():
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert gate.index("test_plan79_32") < gate.index("test_plan79_33")
    print("OBS 79.33 scan: the test is in the gate after 79.32")
