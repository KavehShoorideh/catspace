"""PLAN 79.50 (SYSTEM_SPEC 11, "The description is the source of the rules; fast code is only a certified copy"; piece: rules; ledger row 44): the kernel's `reach_from` gives the first move of two places as the description's `move_by` does.
  (1) rules/placed/known.json names no cause for any game, and the seven placed records are fresh with 0 disagreements and no cause;
  (2) the one source of an engine crate that differs from the commit before the plan is rust/engine/game/src/outside.rs, and in it only the function `reach_from`; no Cargo file, no description, no reader, no certificate changed."""
from __future__ import annotations

import json
import re
import subprocess as sp
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEVEN = ["chess", "atomic", "3check", "kingofthehill", "antichess", "chess960", "horde"]
BEFORE = "a0170507"


def PT():
    from catspace.incubator.analyst.discovery import placed_translated as m
    return m


def git(*a):
    return sp.run(["git", *a], cwd=ROOT, capture_output=True, text=True)


def test_known_json_holds_no_cause_for_any_game():
    games = json.loads((ROOT / "rules" / "placed" / "known.json").read_text())["games"]
    assert sorted(games) == sorted(SEVEN)
    assert all(v == {} for v in games.values()), {g: list(v) for g, v in games.items() if v}


@pytest.mark.parametrize("game", SEVEN)
def test_the_placed_record_is_fresh_and_reads_no_disagreement(game):
    pt = PT()
    r = pt.load(game)
    assert pt.status(game) == "fresh", f"{game}: python -m catspace.incubator.analyst.discovery.placed_translated run {game}"
    assert r["disagreements"] == 0 and not r["small"]["causes"] and not r["large"]["causes"], (game, r["disagreements"])
    for run in ("small", "large"):
        assert r[run]["compared"]["reach_from"][0] > 0 and r[run]["compared"]["reach_from (given)"][0] > 0
    print(f"OBS 79.50 B {game}: reach_from compared small {r['small']['compared']['reach_from'][0]} large {r['large']['compared']['reach_from'][0]}, given small {r['small']['compared']['reach_from (given)'][0]} large {r['large']['compared']['reach_from (given)'][0]}, disagreements {r['disagreements']}")


def test_only_the_function_reach_from_of_the_bridge_differs_from_the_commit_before_the_plan():
    if git("cat-file", "-e", f"{BEFORE}^{{commit}}").returncode:
        pytest.skip(f"{BEFORE} not in this clone")
    changed = git("diff", "--name-only", BEFORE, "--", "rust/engine").stdout.split()
    srcs = [c for c in changed if re.match(r"rust/engine/[^/]+/src/", c)]
    assert srcs in ([], ["rust/engine/game/src/outside.rs"]), srcs
    assert not [c for c in changed if c.endswith("Cargo.toml") or c.endswith("Cargo.lock")], changed
    if srcs:
        text = (ROOT / "rust/engine/game/src/outside.rs").read_text().splitlines()
        start = next(i for i, ln in enumerate(text) if "fn reach_from" in ln) + 1
        end = next(i for i in range(start, len(text)) if text[i].startswith("    fn ")) + 1       # the next method of the impl
        diff = git("diff", "-U0", BEFORE, "--", "rust/engine/game/src/outside.rs").stdout
        hunks = [(int(a), int(b or 1)) for a, b in re.findall(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.M)]
        assert len(hunks) == 1 and all(start <= a and a + b - 1 < end for a, b in hunks), (hunks, start, end)
    for protected in ("rust/engine/gdl/src/lib.rs", "rust/engine/game/src/gdl_game.rs", "rust/engine/game/src/kernel.rs", "rust/engine/tactics/src/ledger.rs"):
        assert git("diff", "--quiet", BEFORE, "--", protected).returncode == 0, protected
    assert git("diff", "--quiet", BEFORE, "--", "rules", ":(exclude)rules/lemmas", ":(exclude)rules/placed", "catspace/incubator/analyst/discovery/gdl.py", "catspace/incubator/analyst/discovery/certify.py").returncode == 0
    print(f"OBS 79.50 B: files under a crate's src that differ from {BEFORE}: {srcs}")


def test_the_gate_lists_this_test_after_the_one_of_plan_79_48():
    text = (ROOT / "scripts" / "gate.py").read_text()
    assert text.index("test_plan79_48") < text.index("test_plan79_50")
