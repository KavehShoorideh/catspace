"""PLAN 79.32 THE PURE SEARCH IS LAYER 0'S: ONE ALGORITHM OVER A NODE THAT GIVES ITS OPTIONS, AND LAYER 4 A CASE OF IT (SYSTEM_SPEC 11 "The pure search is mathematics and lives in layer 0; what steers it does not", 13 item 50; ledger rows 36, 37).
Rust holds the tests (rust/engine/base/tests/walk_speed.rs, the unit tests of rust/engine/base/src/choice/, rust/engine/tactics/tests/walk_speed.rs and layer4_goals.rs). THIS file is what belongs to Python: the scans.
  (1) layer 4's reach.rs holds no recursion of its own over options: its walk is layer 0's (`cs_base::choice::walk::Walk`), and cs_tactics names cs_base as a dependency;
  (2) cs_base still depends on nothing and names no game and no word of a world (plan 79.30's and 79.31's scans run again over the new files);
  (3) the interface of the walk is static: no boxed function and no reference to a trait object in the walk, the node, or the explicit tree's case; the numbers of the hot path are bare whole numbers;
  (4) the explicit tree is a case of the interface and its readings go through it (`can_force_set` is a `Walk`; the values are one fold over a `TreeSpace`);
  (5) the tests of speed exist and print OBS lines; the laws the walk relies on are theorems with no sorry;
  (6) the answers recorded on main's code before the change and the answers now agree, question by question (when both files are on this machine);
  (7) the test is in the gate after the one of 79.31.
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
LEAN = ROOT / "lean" / "ideas" / "Ideas" / "Layer0"


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


def test_reach_holds_no_recursion_over_options_of_its_own_and_cs_tactics_names_cs_base():
    reach = code(TACTICS / "src" / "reach.rs")
    assert "cs_base::choice::walk" in (TACTICS / "src" / "reach.rs").read_text()
    # no function of reach.rs is the and-or walk: its only calls into the walk are the asks of `Walk::go` (from the deepening and from the hand-walked lines)
    assert not re.search(r"\bfn go\b", reach), "the recursion is layer 0's"
    assert not re.search(r"\bself\.go\(", reach), "no call of its own walk"
    assert "HashMap<(u64, u16)" not in reach and "struct Entry" not in reach, "the answers are kept by layer 0's walk"
    assert not re.search(r"\bpath\b\s*:\s*Path|Path::new", reach), "the line's keys are layer 0's"
    toml = (TACTICS / "Cargo.toml").read_text()
    deps = toml.split("[dependencies]", 1)[1].split("[dev-dependencies]", 1)[0]
    assert re.search(r"^cs_base\s*=", deps, flags=re.M), "cs_tactics depends on cs_base (the walk is layer 0's)"
    print("OBS 79.32 scan: reach.rs has no walk of its own, keeps no table and no line of its own; cs_tactics depends on cs_base")


def test_cs_base_still_depends_on_nothing_and_names_no_game_with_the_new_files():
    P31.test_cs_base_depends_on_nothing_at_all()
    P31.test_cs_base_names_no_game_and_no_world_word_and_the_layer_4_test_of_the_meaning_no_game()
    P30.test_cs_base_sources_name_no_game_and_no_word_of_a_world()
    names = {p.name for p in (BASE / "src" / "choice").glob("*.rs")}
    assert {"walk.rs", "tree_space.rs", "mod.rs", "lp.rs"} <= names, names
    print("OBS 79.32 scan: cs_base's new files hold no dependency, no game and no word of a world")


def test_the_interface_is_static_and_the_numbers_of_the_hot_path_are_bare_whole_numbers():
    for f in ("walk.rs", "tree_space.rs"):
        c = code(BASE / "src" / "choice" / f)
        assert not re.search(r"Box\s*<\s*dyn|&\s*(mut\s+)?dyn\b|\bdyn\s+Fn", c), f
    w = (BASE / "src" / "choice" / "walk.rs").read_text()
    for sig in ("pub trait Space", "pub trait Steer<S: Space>", "pub trait Whole", "pub struct Walk<S: Space, T: Steer<S>, N: Whole>", "fn key(&self) -> u64", "pub fn go(&mut self, depth: N, ask: Ask, later: bool, by_own: bool, out: &mut Out<N>)"):
        assert sig in w, sig
    for sig in ("fn enter(&mut self, o: &Self::Opt)", "fn leave(&mut self)", "fn options(&mut self, out: &mut Vec<Self::Opt>)"):          # plan 79.33: the stepping moved to `fold::Node`, which `Space` extends
        assert sig in (BASE / "src" / "choice" / "fold.rs").read_text(), sig
    assert "whole!(u8, u16, u32, u64, usize);" in w
    # layer 4's walk runs on machine numbers (u8), not on the exact number type
    assert "Walk<Position<'a, G>, Policy<'a, G>, u8>" in (TACTICS / "src" / "reach.rs").read_text()
    assert "Num" not in code(BASE / "src" / "choice" / "walk.rs")
    print("OBS 79.32 scan: no boxed function and no trait object in the walk or the tree's case; Whole is satisfied by the bare whole numbers; layer 4 walks on u8; the walk names no exact number")


def test_the_explicit_tree_is_a_case_of_the_interface_and_its_readings_go_through_it():
    m = (BASE / "src" / "choice" / "mod.rs").read_text()
    assert "walk::Walk::<_, _, u16>::new(tree_space::TreeSpace::forcing(" in m
    assert "tree_space::TreeSpace::valuing(t, side).value(" in m
    assert "pub fn can_force_set<R>(t: &Tree<R>, who: &Party, holds: &dyn Fn(&R) -> bool) -> bool" in m, "the public readings keep their signatures"
    assert "fn secure<R>" in m and "fn pure_choice" not in m
    t = (BASE / "src" / "choice" / "tree_space.rs").read_text()
    assert "impl<'t, R, F: Fn(&R) -> bool> Space for TreeSpace<'t, R, F>" in t
    print("OBS 79.32 scan: can_force_set is a Walk over TreeSpace; guarantee, value and force_chance are one fold over the same stepping")


def lean_code(module: str) -> str:
    return re.sub(r"/-.*?-/", "", (LEAN / f"{module}.lean").read_text(), flags=re.S)


def test_the_tests_of_speed_exist_and_the_laws_the_walk_relies_on_are_theorems():
    speed = (BASE / "tests" / "walk_speed.rs").read_text()
    assert "OBS 79.32 cs_base walk" in speed and "nodes a second" in speed
    goals = (TACTICS / "tests" / "walk_speed.rs").read_text()
    assert "OBS 79.32 speed" in goals and "nodes a second" in goals
    assert "the_whole_reply_of_every_question" in (TACTICS / "tests" / "layer4_goals.rs").read_text()
    code_ = lean_code("Choice")
    for n in ("canIn_succ", "canIn_mono", "not_canIn_of_le", "canS_iff_canIn"):
        assert re.search(rf"theorem {n}\b", code_), n
    assert not re.search(r"\bsorry\b|native_decide|\bdecide\b|^axiom\b|\baxiom\s", code_, flags=re.M)
    print("OBS 79.32 scan: the speed tests print OBS lines; met within n steps is monotone in n, and a set can force P exactly when it can within some number of steps (theorems, no sorry)")


def test_the_answers_recorded_before_the_change_and_the_answers_now_agree_question_by_question():
    folder = Path(sys.modules["catspace.io.paths"].experiment("whole_79_32/x")).parent if "catspace.io.paths" in sys.modules else None
    if folder is None:
        from catspace.io import paths
        folder = Path(paths.experiment("whole_79_32/x")).parent
    need = [f"{lab}.{k}.dump" for lab in ("before", "after") for k in ("goals", "meaning", "answers")]
    if not all((folder / n).exists() for n in need):
        pytest.skip(f"the recorded answers are not on this machine ({folder}): they are made by scripts/whole_79_32_run.sh dump before / after")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "whole_79_32_compare.py"), "before", "after"], capture_output=True, text=True, cwd=ROOT)
    print(r.stdout)
    assert r.returncode == 0, r.stdout[-2000:]


def test_the_test_is_registered_in_the_gate_after_the_one_of_79_31():
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert gate.index("test_plan79_31") < gate.index("test_plan79_32")
    print("OBS 79.32 scan: the test is in the gate after 79.31")
