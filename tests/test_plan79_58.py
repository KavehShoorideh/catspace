"""PLAN 79.58 (SYSTEM_SPEC 11, blocks "A divergence found and named" and "The support filter"; ledger rows 58, 57, 44; piece: rules; LAYER 2, on layer 1's kept graph): the Python part of the gate. The cargo tests hold the Rust side
(rust/engine/game/tests/translated_reads.rs: the measures of part A, and part B's gates when they are built). This file holds:
  (1) the kept reads modules (generated_placed/<game>_reads.rs) are the ones the translator and the make script make now, and name the keyed questions of the plan: the bearing and the reach of a unit (four places) and of one row (six), and the allowed moves of a unit (three);
  (2) the make script's unfolding of the literal with a compound term (`allowed_unfolded`): one clause for each pair of an ends clause and a legal clause of the same functor, no compound term in the lines it adds, and nothing of the plain placed modules changed;
  (3) layer 0 (rust/engine/base) is as it was at the plan's start.
The text of this file names the games on purpose (a test file may)."""
from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLACED = ROOT / "rust" / "engine" / "game" / "tests" / "generated_placed"
GAMES = ("chess", "atomic", "3check", "kingofthehill", "antichess", "chess960", "horde")
PARENT = "6cd6b4f3"                                    # main at the plan's start


def make_script():
    spec = importlib.util.spec_from_file_location("gdl_to_rust_make", ROOT / "scripts" / "gdl_to_rust_make.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


@pytest.mark.parametrize("game", GAMES)
def test_the_kept_reads_module_names_the_keyed_questions_of_the_plan(game):
    from catspace.incubator.analyst.discovery import gdl_to_rust as T
    src = (PLACED / f"{game}_reads.rs").read_text()
    head = src.split("\n", 20)[:20]
    assert f"// translator-sha256: {T.translator_hash()}" in head
    for needle in ("// reads: placed_bears with its first 4 places as the key", "// reads: placed_reaches with its first 4 places as the key", "// reads: placed_bears with its first 6 places as the key",
                   "// reads: placed_reaches with its first 6 places as the key", "// reads: placed_allowed with its first 3 places as the key"):
        assert needle in head, needle
    assert not [x for x in head if "reads refused" in x]
    plain = (PLACED / f"{game}.rs").read_text()
    assert "placed_allowed" not in plain and "ask_reads" not in plain


@pytest.mark.parametrize("game", GAMES)
def test_the_unfolding_of_the_allowed_moves_adds_clauses_without_a_compound_term(game):
    from catspace.incubator.analyst.discovery import placed_questions as PQ
    M = make_script()
    text = PQ.composed_text(game)
    ends = [l for l in text.splitlines() if l.startswith("(<= (placed_ends ")]
    added = M.allowed_unfolded(text)
    assert added[0].startswith("(signature placed_allowed ")
    clauses = added[1:]
    # one clause for each pair of an ends clause and a legal clause of its functor
    assert len(ends) == 2 and len(clauses) == 2, (len(ends), len(clauses))
    for c in clauses:
        assert c.startswith("(<= (placed_allowed ?c ?f1 ?r1 ?f2 ?r2) ")
        body = c[len("(<= (placed_allowed ?c ?f1 ?r1 ?f2 ?r2) "):]
        assert not re.search(r"\((move|promote|legal)[ )]", body), c


def test_layer_0_is_as_it_was_at_the_plans_start():
    if not git("cat-file", "-t", PARENT).strip():
        pytest.skip("the plan's parent commit is not in this checkout")
    changed = git("diff", "--name-only", PARENT, "--", "rust/engine/base").split()
    assert changed == [], f"layer 0 changed: {changed}"
