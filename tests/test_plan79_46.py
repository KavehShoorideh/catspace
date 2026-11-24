"""PLAN 79.46 THE KERNEL HELD TO THE TRANSLATED MODULE (SYSTEM_SPEC 11 "The description is the source of the rules; fast code is only a certified copy"; piece: rules; ledger rows 44, 45).
The chain: description, evaluator, translated module (held to the evaluator and to the plain translation, tests of 79.37 and 79.38), kernel (held to the module by rust/engine/game/tests/translated_kernel.rs).
THIS file is the gate of it:
  (1) the record of each game with a kept translated module (discovery/lemmas/copy_records/<game>.json, apart from the twelve certificates) is FRESH: its key is the current files', it found no disagreement, its small
      run played the existing kernel certificate's corpus and counted the certificate's positions, its large run counted states;
  (2) a fresh small run of the Rust test agrees (the counts it writes equal the certificate's, no disagreement), and the negative control (a kernel of other rules) is caught;
  (3) the status is a function of the hashes and the counts: a stale, a failing, a short and a missing record are each named, and none is fresh;
  (4) the twelve certificates are still twelve and fresh and the records are counted apart; the module names no game; the large runs are ignored tests with a named cause;
  (5) nothing on the engine's path changed (no engine crate source, no Cargo file, no description, no reader).
The large run is by hand: `python -m catspace.incubator.analyst.discovery.kernel_translated run <game>` (about five minutes a game on all cores)."""
from __future__ import annotations

import copy
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BEFORE = "93a2c09d"                      # the commit before this plan's code
MODULE = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "kernel_translated.py"
RUST_TEST = ROOT / "rust" / "engine" / "game" / "tests" / "translated_kernel.rs"


def KT():
    from catspace.incubator.analyst.discovery import kernel_translated as m
    return m


def CFY():
    from catspace.incubator.analyst.discovery import certify as m
    return m


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def have_before() -> bool:
    return git("cat-file", "-e", f"{BEFORE}^{{commit}}").returncode == 0


needs_cargo = pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo (offline) to run the small co-play")


def test_there_is_a_kept_module_and_a_record_for_each_game_that_has_one():
    kt = KT()
    gs = kt.games()
    assert len(gs) >= 2, gs
    for g in gs:
        assert kt.path_of(g).exists(), f"no record for {g}: python -m catspace.incubator.analyst.discovery.kernel_translated run {g}"
    print(f"OBS 79.46 E: games with a kept translated module {gs}, each with a record")


@pytest.mark.parametrize("game", ["chess", "atomic"])
def test_the_record_is_fresh_and_counted(game):
    kt = KT()
    r = kt.load(game)
    st = kt.status(game)
    assert st == "fresh", f"{game}: {st} (the files the answers depend on changed, or the run found a disagreement or counted less than the certificate): python -m catspace.incubator.analyst.discovery.kernel_translated run {game}"
    assert r["disagreements"] == 0
    s, lg = r["small"], r["large"]
    assert s["states"] == s["certificate_positions"] == kt.certificate_positions(game) and s["equals_certificate"]
    assert lg["states"] > s["states"] and lg["nexts"] >= lg["states"] and lg["ended"] > 0 and lg["results"] == lg["ended"] and lg["cores"] >= 1
    assert r["chain"]["held_to_the_evaluator"]["coplay_states"] > 0
    print(f"OBS 79.46 E record {game}: fresh; small {s['states']} states (the certificate's {s['certificate_positions']}), large {lg['games']} games {lg['states']} states {lg['moves']} allowed moves {lg['nexts']} next states "
          f"{lg['ended']} ended, disagreements {r['disagreements']}, left out {lg['left_out']}, {lg['seconds']} s on {lg['cores']} cores, {r['date']}")


@needs_cargo
@pytest.mark.parametrize("game", ["chess", "atomic"])
def test_a_fresh_small_run_agrees_and_counts_the_certificates_positions(game):
    kt = KT()
    c = kt.small_counts(game)
    assert c["test_returned"] == 0 and not c["causes"], c["causes"]
    assert c["states"] == kt.certificate_positions(game)
    assert c["moves"] > c["states"] and c["nexts"] == c["moves"] and c["ended"] > 0 and c["results"] == c["ended"]
    print(f"OBS 79.46 E small {game}: {c['games']} games, {c['states']} states (the certificate's {kt.certificate_positions(game)}), {c['moves']} allowed moves, {c['nexts']} next states compared, {c['ended']} ended")


@needs_cargo
def test_the_negative_control_a_kernel_of_other_rules_is_caught():
    kt = KT()
    rc, text = kt.cargo(["--test", "translated_kernel", "a_kernel_of_other_rules", "--", "--nocapture"], {})
    assert rc == 0, text[-2000:]
    line = next(ln for ln in text.splitlines() if ln.startswith("OBS 79.46 control"))
    print(line[:300])
    assert "disagreements 0 " not in line


def test_the_status_names_a_stale_a_failing_a_short_and_a_missing_record(monkeypatch):
    kt = KT()
    good = copy.deepcopy(kt.load("chess"))
    assert kt.status("chess", good) == "fresh"
    stale = copy.deepcopy(good)
    stale["key"] = "0" * 16
    assert kt.status("chess", stale) == "stale"
    failing = copy.deepcopy(good)
    failing["disagreements"] = 1
    assert kt.status("chess", failing) == "failing"
    short = copy.deepcopy(good)
    short["small"]["states"] -= 1
    assert kt.status("chess", short) == "short"
    zero = copy.deepcopy(good)
    zero["large"]["nexts"] = 0
    assert kt.status("chess", zero) == "short"
    monkeypatch.setattr(kt, "load", lambda g: None)
    assert kt.status("chess") == "missing"
    # the key is a function of the hashes: a changed hash is a changed key
    monkeypatch.undo()
    h = dict(kt.hashes_of("chess"))
    assert set(h) == {"description", "translator", "module", "kernel", "bridge", "test", "lock"}
    before = kt.key_of("chess")
    monkeypatch.setattr(kt, "hashes_of", lambda g: dict(h, module="f" * 16))
    assert kt.key_of("chess") != before
    print("OBS 79.46 E status: stale, failing, short, missing are each named; a changed hash changes the key")


def test_the_twelve_certificates_are_still_twelve_and_fresh_and_the_records_are_apart():
    cf, kt = CFY(), KT()
    st = cf.status()
    assert len(cf.BINDINGS) == 12 and all(v == "fresh" for v in st.values()), st
    assert kt.DIR.parent == cf.DIR.parent and kt.DIR != cf.DIR
    assert not list(cf.DIR.glob("*copy*")) and not list(cf.DIR.glob("**/copy_records"))
    print(f"OBS 79.46 E: the old bindings {len(cf.BINDINGS)}, all fresh; the records live in {kt.DIR.relative_to(ROOT)}, counted apart")


def test_the_module_names_no_game():
    cf = CFY()
    text = MODULE.read_text()
    for g in list(cf.VARIANT_TEXTS) + ["python-chess", "shakmaty"]:
        assert not re.search(rf"\b{re.escape(g)}\b", text), g


def test_the_large_runs_are_ignored_tests_with_a_named_cause_and_the_small_ones_are_not():
    text = RUST_TEST.read_text()
    large = re.findall(r'#\[test\]\n#\[ignore = "([^"]+)"\]\nfn (\w+_large_\w+)', text)
    assert len(large) >= 2 and all("large run" in why and "run by hand" in why for why, _n in large), large
    small = re.findall(r"#\[test\]\nfn (\w+_agree_on_the_certificates_games)", text)
    assert len(small) >= 2
    assert "#[ignore" not in text.split("fn chess_the_kernel_and_the_translated_module_agree_on_the_certificates_games")[0].rsplit("#[test]", 1)[-1]


def test_no_source_of_the_engine_no_cargo_file_no_description_and_no_reader_differs_from_the_commit_before_the_plan():
    if not have_before():
        pytest.skip(f"{BEFORE} not in this clone")
    changed = git("diff", "--name-only", BEFORE, "--", "rust/engine").stdout.split()
    # 2026-11-24 (plan 79.47 part E): the one source of an engine crate that plan 79.47 changed is the bridge `state_facts` of rust/engine/game/src/outside.rs (the checks of three-check; tests/test_plan79_47.py holds that it is one place and nothing else under a src)
    assert not [c for c in changed if (re.match(r"rust/engine/[^/]+/src/", c) and c != "rust/engine/game/src/outside.rs") or c.endswith("Cargo.toml") or c.endswith("Cargo.lock")], changed
    # 2026-11-24 (plan 79.48): rules/placed/ holds the additive texts of the placed questions and their lists (documents, no description): excluded as rules/lemmas is.
    assert git("diff", "--quiet", BEFORE, "--", "rules", ":(exclude)rules/lemmas", ":(exclude)rules/placed", "catspace/incubator/analyst/discovery/gdl.py").returncode == 0
    print(f"OBS 79.46 E: files under rust/engine that differ from {BEFORE}: {changed}")
