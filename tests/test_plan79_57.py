"""PLAN 79.57 (SYSTEM_SPEC 11, blocks "A divergence found and named" and "The support filter"; ledger rows 58, 57, 44, 40; piece: rules; LAYER 2, on layer 1's kept graph): the Python part of the gate. The cargo tests
rust/engine/game/tests/translated_graph.rs (part A: the graph kept from the description equals the fresh graph and the bridge's) and translated_reads.rs (part B: the graph kept under the support rule equals the fresh graph;
a support with a place struck out is caught) hold the Rust side. This file holds:
  (1) pass (27) `reads` on a toy description: the module made with it has `ask_reads`, the recording of every read of a fluent, the numbers of the fluents and the relations given; the module made without it has none of these;
      a relation that is not in the description, or is not derived, is refused with its reason in the head;
  (2) the kept module of each game (generated_placed/<game>_reads.rs) is the one the translator makes now (hashes), names the two relations it was given, and the plain placed modules have no trace of the pass;
  (3) layer 0 and layer 1 are as they were before the plan (git: no change in rust/engine/base and rust/engine/rep), and the keeper's interface text names the second rule.
The text of this file names the games on purpose (a test file may)."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLACED = ROOT / "rust" / "engine" / "game" / "tests" / "generated_placed"
GAMES = ("chess", "atomic", "3check", "kingofthehill", "antichess", "chess960", "horde")
PARENT = "63045a2d"                                    # main at the plan's start


def sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def translator():
    from catspace.incubator.analyst.discovery import gdl_to_rust as T
    return T


def toy() -> str:
    return sibling("test_plan79_56").small("", sibling("test_plan79_56").WORLD_OK)


def test_the_reads_pass_records_every_read_of_a_fluent_and_a_module_without_it_has_none():
    T = translator()
    base = [p for p in T.PASSES if p != "carry"]
    with_reads = T.translate(toy(), "g", base + ["reads"], reads=(("touched", 1),))
    without = T.translate(toy(), "g", base)
    heads = [x for x in with_reads.split("\n", 12)[:12] if x.startswith("// reads")]
    assert heads == ["// reads: touched with its first 1 places as the key"], heads
    for needle in ("pub fn ask_reads", "rd_note(", "pub const READS_FLUENTS", "pub const READS: &[(&str, usize)] = &[(\"touched\", 1)]"):
        assert needle in with_reads, needle
    for needle in ("ask_reads", "rd_note", "READS_FLUENTS", "RD_REC"):
        assert needle not in without, needle
    n = with_reads.count("rd_note(")
    # the toy's test reads two fluent literals through the relations made from them; every form a literal takes is recorded
    assert n >= 2, n
    print(f"OBS 79.57 B toy: the module with the pass records {n} reads in its top-down functions; the module without it has no trace of the pass")


def test_a_relation_that_cannot_be_asked_by_key_is_refused_with_its_reason():
    T = translator()
    base = [p for p in T.PASSES if p != "carry"]
    src = T.translate(toy(), "g", base + ["reads"], reads=(("nothing_of_the_kind", 1), ("square", 5), ("touched", 1)))
    heads = [x for x in src.split("\n", 14)[:14] if x.startswith("// reads")]
    assert any("nothing_of_the_kind: the description has no such relation" in h for h in heads), heads
    assert any("square: the key of 5 places does not fit" in h for h in heads), heads
    assert any(h.startswith("// reads: touched") for h in heads), heads


@pytest.mark.parametrize("game", GAMES)
def test_the_kept_reads_module_is_the_one_the_translator_makes_now_and_the_plain_placed_module_has_no_trace_of_it(game):
    T = translator()
    src = (PLACED / f"{game}_reads.rs").read_text()
    head = src.split("\n", 14)[:14]
    assert f"// translator-sha256: {T.translator_hash()}" in head, "the kept module was made by another translator: run scripts/gdl_to_rust_make.py --placed"
    assert "// reads: placed_bears with its first 4 places as the key" in head and "// reads: placed_reaches with its first 4 places as the key" in head
    assert "pub fn ask_reads" in src and "reads" in [x for x in head if x.startswith("// passes:")][0]
    plain = (PLACED / f"{game}.rs").read_text()
    for needle in ("ask_reads", "rd_note", "READS_FLUENTS"):
        assert needle not in plain, needle
    assert f"// translator-sha256: {T.translator_hash()}" in plain.split("\n", 5)[:5]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


def test_layers_0_and_1_are_unchanged_and_the_keepers_text_names_the_second_rule():
    if not git("cat-file", "-t", PARENT).strip():
        pytest.skip("the plan's parent commit is not in this checkout")
    changed = git("diff", "--name-only", PARENT, "--", "rust/engine/base", "rust/engine/rep").split()
    assert changed == [], f"layer 0 or layer 1 changed: {changed}"
    text = (ROOT / "rust" / "engine" / "game" / "src" / "keeper.rs").read_text()
    for needle in ("PLAN 79.57 PART B, THE SECOND RULE", "pub fn make_supported", "pub fn new_supported", "pub trait Supported", "pub struct Support"):
        assert needle in text, needle
    print("OBS 79.57 B layers: rust/engine/base and rust/engine/rep are as they were at the plan's start; the keeper names the second rule beside the hand rule")
