"""PLAN 79.51 FUTURE PREDICTION, FIRST STEP: A GOAL SOME UNMET FACTS AWAY ON A LINE, THE FORCING MOVES THAT MEET THEM, THE JOIN OF TWO GOALS AND THE STOP SAID WITH ITS CAUSE (SYSTEM_SPEC 7 "Future prediction: one channel, two sub-layers, and the decision to
stop"; piece: tactical reasoning and the opportunity policy; ledger row 50).
Rust holds the tests (cargo test --release --offline -p cs_decision --test future_fixture, and the measure future_count): the fixture printed as Kaveh reads it, the proof that the decision is the same with the channel on and off. THIS file is what belongs
to Python: the scans.
  (1) the module and its tests of the channel name no object of a game, in code, strings or comments (the fixture position is the fixture test's own and the printing scripts');
  (2) the interface is written in the module's doc before the body: what it takes, returns and who calls it; every statement carries its sub-layer;
  (3) NOTHING OF THE CHANNEL IS READ BY A CHOICE: no other source of the decision crate reads `future`, the wake only writes it into the log after the decision is complete, the default is None;
  (4) the files this plan must not touch are not touched (the shared search, the first game's description, Cargo.lock; no new dependency; no decision-path crate depends on cs_learn);
  (5) the printing scripts fit nothing and call no outside engine;
  (6) the test is in the gate after the last test of plan 79.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
DECISION = ENGINE / "decision"
DISCOVERY = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
BEFORE = "12409195"                      # the commit this plan was built from
GAME_WORDS = r"\b(king\w*|queen\w*|rook\w*|bishop\w*|knight\w*|pawn\w*|castl\w*|promot\w*|en passant|check\w*|mate|stalemate|uci|fen|san|chess\w*|shakmaty)\b"
SQUARE = r"\b[a-h][1-8]\b"


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def have_before() -> bool:
    return git("cat-file", "-e", f"{BEFORE}^{{commit}}").returncode == 0


def test_the_module_names_no_object_of_a_game():
    p = DECISION / "src" / "future.rs"
    txt = p.read_text()
    bad = [m.group(0) for m in re.finditer(GAME_WORDS, txt, flags=re.I)] + [m.group(0) for m in re.finditer(SQUARE, txt)]
    assert not bad, bad
    print(f"OBS 79.51 scan: {p.name} names no object of a game in code, strings or comments ({len(txt.splitlines())} lines)")


def test_the_interface_is_written_first_and_every_statement_says_its_sublayer():
    txt = (DECISION / "src" / "future.rs").read_text()
    head = txt.split("use crate::")[0]
    for word in ("INTERFACE", "WHO CALLS IT", "TAKES", "RETURNS", "NEVER ADDS TO THE VALUE", "NEAR sub-layer", "FAR sub-layer", "ALIVE", "THREAT NOW", "KEPT", "LEFT"):
        assert word in head, word
    for item in ("pub fn read<", "pub enum Sub", "pub struct LineGoal", "pub struct TargetStmt", "pub struct JoinStmt", "pub struct NearRead", "pub enum Stop"):
        assert item in txt, item
    assert re.search(r"pub struct LineGoal \{ pub sub: Sub", txt) and re.search(r"pub struct TargetStmt \{ pub sub: Sub", txt), "a statement carries the sub-layer that made it"
    assert "pub read: Vec<String>" in txt, "a statement carries what it read"
    print("OBS 79.51 scan: the interface heads the module; goals and targets carry their sub-layer and what they read")


def test_nothing_of_the_channel_is_read_by_a_choice():
    readers = []
    for p in sorted((DECISION / "src").glob("*.rs")):
        t = p.read_text()
        if p.name in ("future.rs", "wake.rs", "log.rs", "lib.rs"):
            continue
        if re.search(r"\bfuture\b|\bFutureRead\b|\bFutureParams\b", re.sub(r"//.*", "", t)):
            readers.append(p.name)
    assert not readers, f"a source reads the channel: {readers}"
    wake = (DECISION / "src" / "wake.rs").read_text()
    uses = [l for l in wake.splitlines() if "future" in l and not l.strip().startswith("//")]
    assert len(uses) == 3, uses                      # the field, its default, the line that writes the log
    assert any("log.future = Some(" in l for l in uses) and any("future: None" in l for l in uses)
    assert wake.index("log.future = Some(") > wake.index("log.measure = measure_log;") > wake.index("log.chosen_goal ="), "the channel is read after the choice and the log of it are made"
    for name in ("rule.rs", "pick.rs", "chooser.rs", "goals.rs", "threats.rs", "after.rs", "gain.rs", "carry.rs"):
        assert "crate::future" not in (DECISION / "src" / name).read_text(), name
    print("OBS 79.51 scan: no source but the module, the wake (which writes the log after the choice) and the log reads the channel; the default is None")


def test_the_files_the_plan_must_not_touch_are_not_touched_and_no_dependency_is_added():
    for rel in ("rust/engine/Cargo.lock", "rust/engine/tactics/src/ledger.rs", "rust/engine/gdl/src/lib.rs", "rust/engine/decision/Cargo.toml"):
        if have_before():
            assert git("diff", "--quiet", BEFORE, "--", rel).returncode == 0, rel
    assert "cs_learn" not in (DECISION / "Cargo.toml").read_text(), "no decision-path crate depends on the learning crate"
    print("OBS 79.51 scan: Cargo.lock, the shared search, the translator of the description and the decision crate's Cargo.toml are as they were" if have_before() else "OBS 79.51 scan: the commit before this plan is not on this machine; the diff scan is skipped")


def test_the_printing_scripts_fit_nothing_and_call_no_outside_engine():
    for name in ("future_79_51_lists_print.py", "future_79_51_today_print.py"):
        p = DISCOVERY / name
        assert p.name.endswith("_print.py")
        t = p.read_text()
        assert not re.search(r"sklearn|\.fit\(|torch|urlopen|requests|anthropic|stockfish", t, flags=re.I), name


def test_the_test_is_in_the_gate_after_the_last_test_of_plan_79():
    gate = (ROOT / "scripts" / "gate.py").read_text()
    names = re.findall(r"test_plan79_\w+", gate)
    assert "test_plan79_51" in names
    assert names.index("test_plan79_51") > max(names.index(n) for n in names if re.fullmatch(r"test_plan79_(\d\d)", n) and int(n[-2:]) <= 48)
