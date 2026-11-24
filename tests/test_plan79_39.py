"""PLAN 79.39 THE SIDE TO MOVE IS ASKED WITHOUT READING THE END OF THE GAME (SYSTEM_SPEC 11, the block "Layer 2 answers whose turn it is without asking whether the game has ended"; piece: rules; ledger row 46).
The Rust tests hold the method itself (rust/engine/game/tests/turn_79_39.rs: seven games, seeded random play, a made game of four roles). THIS file is the gate around it:
  (1) the Game trait has the method `turn` with a default body, and `Ruled` forwards it;
  (2) kernel.rs differs from the commit before the plan (bb08451c) only by added lines of the one new method (no removed line); gdl_game.rs and gdl/src/lib.rs are unchanged;
  (3) the shared search (tactics/src/ledger.rs) asks the side by the new question and no longer asks `mover`;
  (4) the answers recorded before and after (the dumps of plans 79.32 and 79.33) agree, line by line (when the files are on this machine);
  (5) no Cargo.toml or Cargo.lock differs from the commit before;
  (6) the test is in the gate after the one of 79.37.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
BEFORE = "bb08451c"                      # the commit before this plan's code
LABELS = ("t7939_before", "t7939_after")


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def have_before() -> bool:
    return git("cat-file", "-e", f"{BEFORE}^{{commit}}").returncode == 0


def test_the_trait_has_the_turn_with_a_default_and_the_wrapper_forwards_it():
    lib = (ENGINE / "game" / "src" / "lib.rs").read_text()
    m = re.search(r"fn turn\(&self, s: &Self::State\) -> Role \{ self\.mover\(s\)\.unwrap_or\(0\) \}", lib)
    assert m, "the trait's `turn` has a default body built from `mover`"
    assert lib.count("fn turn(") == 1
    assert "fn mover(&self, s: &Self::State) -> Option<Role>;" in lib, "mover stays, unchanged in meaning"
    rules = (ENGINE / "game" / "src" / "rules.rs").read_text()
    assert "fn turn(&self, s: &G::State) -> Role { self.game.turn(s) }" in rules
    print("OBS 79.39 trait: turn has a default body and Ruled forwards it")


@pytest.mark.skipif(not have_before(), reason="the commit before the plan is not on this machine")
def test_the_kernel_differs_only_by_added_lines_of_the_one_new_method_and_the_evaluator_is_untouched():
    d = git("diff", "-U0", BEFORE, "--", "rust/engine/game/src/kernel.rs").stdout
    removed = [ln for ln in d.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    added = [ln[1:] for ln in d.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    assert removed == [], removed
    assert added, "the kernel answers the new question"
    assert sum("fn turn(" in ln for ln in added) == 1
    assert all(ln.strip().startswith("///") or "fn turn(" in ln for ln in added), added
    assert "outcome" not in [ln for ln in added if "fn turn(" in ln][0], "the kernel's answer asks no end"
    for rel in ("rust/engine/game/src/gdl_game.rs", "rust/engine/gdl/src/lib.rs"):
        assert git("diff", "--quiet", BEFORE, "--", rel).returncode == 0, rel
    print(f"OBS 79.39 kernel: {len(added)} lines added, 0 removed; gdl_game.rs and gdl/src/lib.rs unchanged")


def test_the_shared_search_asks_the_side_by_the_new_question():
    src = (ENGINE / "tactics" / "src" / "ledger.rs").read_text()
    assert ".mover(" not in src
    assert src.count(".turn(") >= 13
    print(f"OBS 79.39 ledger: {src.count('.turn(')} asks of turn, none of mover")


@pytest.mark.skipif(not have_before(), reason="the commit before the plan is not on this machine")
def test_no_manifest_changed():
    for rel in ("rust/engine/Cargo.lock", "rust/engine/game/Cargo.toml", "rust/engine/tactics/Cargo.toml", "rust/engine/Cargo.toml"):
        if (ROOT / rel).exists():
            assert git("diff", "--quiet", BEFORE, "--", rel).returncode == 0, rel


def test_the_answers_before_and_after_agree_line_by_line():
    from catspace.io import paths
    found = 0
    for dirname, kinds in (("whole_79_32", ("goals", "meaning", "answers")), ("whole_79_33", ("values", "toward", "tail", "frame", "forcing"))):
        base = Path(paths.experiment(f"{dirname}/x")).parent
        for kind in kinds:
            a, b = (base / f"{LABELS[0]}.{kind}.dump"), (base / f"{LABELS[1]}.{kind}.dump")
            if not (a.exists() and b.exists()):
                print(f"OBS 79.39 compare {dirname} {kind}: skipped, a file is not on this machine")
                continue
            la, lb = a.read_text().splitlines(), b.read_text().splitlines()
            differ = sum(1 for x, y in zip(la, lb) if x != y) + abs(len(la) - len(lb))
            print(f"OBS 79.39 compare {dirname} {kind}: {len(la)} lines, identical {len(la) - differ}, differ {differ}")
            assert differ == 0
            found += 1
    if not found:
        pytest.skip("the recorded answers are not on this machine")


def test_the_gate_lists_this_test_after_the_one_of_79_37():
    text = (ROOT / "scripts" / "gate.py").read_text()
    assert "test_plan79_37 test_plan79_39" in text
