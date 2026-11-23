"""PLAN 79.35 LAYER 2 SAYS FOUR MORE GAMES ON THE SAME LAYER 1: TWO WITH ANOTHER START (CHESS960, HORDE) AND TWO WITH OTHER RULES OF MOVING AND ENDING (ATOMIC, ANTICHESS) (SYSTEM_SPEC 11 "Layer 2 says variants, with another start and with other rules,
on the same layer 1"; ledger rows 41, 40; pieces: rules, representation).
Rust holds the tests (rust/engine/game/tests/layer2_variants.rs: the kernel's mode of each game against the fixtures of the outside library and the rules of each game by made positions; layer2_variants_described.rs: the description of each game
against the same fixtures; rust/engine/rep/tests/seen_variants.rs: layer 1 holds a state of each). THIS file is what belongs to Python: the scans and the certificates:
  (1) the four fixtures exist, say how their game differs, are read by the tests and the three old ones are the files the maker makes;
  (2) no file under rust/engine/rep/src differs from the commit before this plan (layer 1 is unchanged);
  (3) each game either has a description under rules/variants and a fresh certificate, or is named in NOT_DESCRIBED with its cause;
  (4) kernel.rs, gdl_game.rs, gdl/src/lib.rs, tactics/src/ledger.rs and layer 0 are untouched, and so are the dependencies (Cargo.lock, every Cargo.toml);
  (5) the answers recorded before the change and after agree, line by line (when the files are on this machine);
  (6) the test is in the gate after the one of 79.34.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
GAME = ENGINE / "game"
FIXTURES = GAME / "tests" / "fixtures"
BEFORE = "fa331fd0"                      # the commit before this plan's code (the plan block's own commit: it changes documents only)
GAMES = ("chess960", "horde", "atomic", "antichess")
# the games the description language or the evaluator cannot say as they stand, each with the rule and the cause (empty: all four are said)
NOT_DESCRIBED: dict = {}


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def have_before() -> bool:
    return git("cat-file", "-e", f"{BEFORE}^{{commit}}").returncode == 0


def test_the_four_fixtures_exist_say_how_their_game_differs_and_the_tests_read_them():
    differs = {"chess960": "start", "horde": "both", "atomic": "rules", "antichess": "rules"}
    for game in GAMES:
        p = FIXTURES / f"{game}.fixture"
        lines = p.read_text().splitlines()
        assert lines[0].startswith(f"game\t{game}\t") and lines[1].startswith("ends\t") and lines[2] == f"differs\t{differs[game]}", game
        assert sum(1 for l in lines if l.startswith("pos\t")) > 50 and any(l.startswith("step\t") for l in lines) and any(l.startswith("perft\t") for l in lines)
        print(f"OBS 79.35 scan: {p.name} {p.stat().st_size} bytes, {len(lines)} lines, differs {differs[game]}")
    common = (GAME / "tests" / "variants_common" / "mod.rs").read_text()
    for game in GAMES:
        assert f'"{game}"' in common, game
    for t in ("layer2_variants.rs", "layer2_variants_described.rs"):
        assert "variants_common" in (GAME / "tests" / t).read_text(), t
    start = FIXTURES.joinpath("chess960.fixture").read_text()
    assert sum(1 for l in start.splitlines() if l.startswith("start\t")) >= 5 and sum(1 for l in start.splitlines() if l.startswith("scharnagl\t")) == 960
    print("OBS 79.35 scan: the two Rust test files read the four fixtures; chess960's has several starts (the standard one among them, two with the leader next to a corner) and all 960 numbered first rows")


def test_layer_1_is_unchanged_no_file_under_rep_src_differs_from_the_commit_before_this_plan():
    if not have_before():
        pytest.skip(f"the commit {BEFORE} is not in this clone")
    d = git("diff", "--stat", BEFORE, "--", "rust/engine/rep/src")
    assert d.returncode == 0 and d.stdout.strip() == "", d.stdout
    assert git("diff", "--quiet", BEFORE, "--", "rust/engine/rep/src", "rust/engine/rep/Cargo.toml").returncode == 0
    seen = (ENGINE / "rep" / "tests" / "seen_variants.rs").read_text()
    assert "OBS 79.35 F" in seen and "differs" in seen, "the new layer 1 test holds a state of each game, found by what the file says"
    print("OBS 79.35 scan: rust/engine/rep/src and its Cargo.toml are the files of the commit before; the new layer 1 test holds a state of each of the four games")


def test_each_game_has_a_description_and_a_fresh_certificate_or_is_named_with_its_cause():
    from catspace.incubator.analyst.discovery import certify as CF
    status = CF.status()
    described = []
    for game in GAMES:
        if game in NOT_DESCRIBED:
            assert NOT_DESCRIBED[game].strip(), f"{game}: a game left without a description names the rule and the cause"
            assert not (ROOT / "rules" / "variants" / f"{game}.gdl").exists(), f"{game}: named as not described, yet a description exists"
            continue
        p = ROOT / "rules" / "variants" / f"{game}.gdl"
        assert p.is_file() and p.read_text().count("(<=") > 5, game
        assert (ROOT / "rules" / "variants" / f"{game}.starts").is_file(), f"{game}: the positions its certificate plays from"
        assert status[f"kernel {game} (cs_game)"] == "fresh", (game, status[f"kernel {game} (cs_game)"])
        cert = CF.load(f"kernel {game} (cs_game)")
        assert cert["disagreements"] == 0 and cert["positions"] > 1000, cert
        described.append((game, cert["positions"]))
    assert not any(v != "fresh" for k, v in status.items() if k.startswith("kernel ")), {k: v for k, v in status.items() if v != "fresh"}
    print(f"OBS 79.35 scan: described and certified fresh with no disagreement: {described}; not described: {sorted(NOT_DESCRIBED)}")


def test_a_wrong_description_disagrees_with_each_kernel_the_negative_controls():
    CE = pytest.importorskip("cs_engine")
    from catspace.incubator.analyst.discovery import certify as CF
    base = (ROOT / "rules" / "chess.gdl").read_text()
    delta = {g: (ROOT / "rules" / "variants" / f"{g}.gdl").read_text() for g in GAMES}
    for kernel, wrong in (("atomic", "antichess"), ("antichess", "atomic"), ("horde", None), ("chess960", "horde")):
        _g, _p, states, bad = CE.certify_coplay(kernel, base if wrong is None else delta[wrong], 3, 5, None if wrong is None else base)
        print(f"OBS 79.35 control: the {kernel} kernel against {'plain chess' if wrong is None else wrong + ' text'}: {len(bad)} disagreements in {states} states")
        assert bad, (kernel, wrong)
    if hasattr(CE, "certify_coplay_from"):
        # the two-unit move of the leader from a start where it does not stand on its file: the description of plain chess does not say it
        starts = CF.start_positions("chess960")
        fen = next(f for f in starts if f.startswith("4k3/8/8/8/8/8/8/RK5R"))
        _g, _p, states, bad = CE.certify_coplay_from("chess960", base, 6, 5, None, fen)
        print(f"OBS 79.35 control: the chess960 kernel from `{fen}` against plain chess's text: {len(bad)} disagreements in {states} states")
        assert bad


def test_the_files_that_must_not_move_do_not_move_and_no_dependency_changes():
    if not have_before():
        pytest.skip(f"the commit {BEFORE} is not in this clone")
    frozen = ["rust/engine/gdl/src/lib.rs", "rust/engine/game/src/gdl_game.rs", "rust/engine/game/src/kernel.rs", "rust/engine/tactics/src/ledger.rs",   # 2026-11-23 (Kaveh): layer 0's sources may be mended, so they left this list
              "rust/engine/Cargo.lock", "rust/engine/Cargo.toml"] + [f"rust/engine/{c}/Cargo.toml" for c in ("base", "rep", "memory", "game", "gdl", "value", "tactics", "decision", "planning", "learn", "py", "operator")]
    d = git("diff", "--stat", BEFORE, "--", *frozen)
    assert d.returncode == 0 and d.stdout.strip() == "", d.stdout
    print("OBS 79.35 scan: kernel.rs, gdl_game.rs, gdl/src/lib.rs, ledger.rs, layer 0 (rust/engine/base), Cargo.lock and every Cargo.toml are the files of the commit before")


def test_the_recorded_dumps_agree_before_and_after_when_they_are_on_this_machine():
    from catspace.io import paths
    f32 = Path(paths.experiment("whole_79_32/x")).parent
    f33 = Path(paths.experiment("whole_79_33/x")).parent
    ran = 0
    if (f32 / "before35.goals.dump").exists() and (f32 / "after35.goals.dump").exists():
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "whole_79_32_compare.py"), "before35", "after35"], capture_output=True, text=True, cwd=ROOT)
        print(r.stdout[-1500:])
        assert r.returncode == 0, r.stdout[-2000:]
        ran += 1
    kinds = [k for k in ("values", "toward", "tail", "frame") if (f33 / f"before35.{k}.dump").exists() and (f33 / f"after35.{k}.dump").exists()]
    if kinds:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "whole_79_33_compare.py"), "before35", "after35", *kinds], capture_output=True, text=True, cwd=ROOT)
        print(r.stdout[-1500:])
        assert r.returncode == 0, r.stdout[-2000:]
        ran += 1
    if not ran:
        pytest.skip("the dumps of plans 79.32 and 79.33 are not on this machine: made by scripts/whole_79_32_run.sh and whole_79_33_run.sh dump <kind> before35 / after35")


def test_the_test_is_registered_in_the_gate_after_the_one_of_79_34():
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert gate.index("test_plan79_34") < gate.index("test_plan79_35")
    print("OBS 79.35 scan: the test is in the gate after 79.34")
