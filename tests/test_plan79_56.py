"""PLAN 79.56 (SYSTEM_SPEC 11, blocks "Layer 2 unlocked for lemmas" and "The support filter"; ledger rows 55, 56, 57; piece: rules; LAYER 2): the gate of pass (26) `support`, the Python part. The cargo test
rust/engine/game/tests/translated_support.rs holds the modules with the pass to the modules without it at every state of the walks; this file holds:
  (1) the shape is found from the clauses in each of the seven games (a test R, the hypothetical relation H, the state relation B), and a description with every name replaced gives the same shape;
  (2) the pass refuses, naming the reason, where a clause of the test reads the key outside H (a relation that does not read the state, a test `distinct` over a key variable), where a view of H has clauses of both forms, and
      where H is not a view; a description with no such relation is left unchanged and the head says so;
  (3) the modules made without the pass are those of the kept files (no `support` line, no support item), and a module with the pass names what it found in its head;
  (4) the translator's source names no relation of any description (the scan of tests/test_plan79_37.py covers the whole file) and the Lean lemma has no `sorry`;
  (5) the kept module of each game where the shape is found is the one the translator makes now (hashes), and no kept module exists where it is not.
The text of this file names the games on purpose (a test file may)."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "rust" / "engine" / "game" / "tests" / "generated"
GAMES = ("chess", "atomic", "3check", "kingofthehill", "antichess", "chess960", "horde")


def sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def translator():
    from catspace.incubator.analyst.discovery import gdl_to_rust as T
    return T


def description_of(game: str) -> str:
    return sibling("test_plan79_37").description_of(game)


def found(text: str):
    T = translator()
    d = T.Description(text)
    f = T.SupportFinder(d)
    f.run()
    return d, f


def key_of(s: dict) -> tuple:
    """The shape of a found test with no name in it."""
    return (s["m"], s["p"], len(s["dparts"]), sorted(p[0] for p in s["dparts"]), len(s["guards"]), len(s["hd"]), len(s["views"]))


@pytest.mark.parametrize("game", GAMES)
def test_the_shape_is_found_from_the_clauses_and_by_no_name(game):
    T = translator()
    m = sibling("test_plan79_40")
    text = description_of(game)
    d, f = found(text)
    assert f.found, (game, f.refused)
    d2 = T.Description(m.rename(text, d))
    f2 = T.SupportFinder(d2)
    f2.run()
    assert sorted(map(key_of, f.found)) == sorted(map(key_of, f2.found)), (game, [key_of(s) for s in f.found], [key_of(s) for s in f2.found])
    s = f.found[0]
    print(f"OBS 79.56 B shape {game}: test {s['R']}, hypothetical relation {s['H']} from {s['B']} (key {s['m']} places, row {s['p']}), differing places read from {len(s['dparts'])} clauses, {len(s['guards'])} guards, relations made from it {','.join(s['hd'])}; the same shape with every name replaced")


def small(extra: str, world_clauses: str) -> str:
    """A description of a toy with the shape (and no chess): a key (a, b) says 'the unit at a goes to b'; the world of the key is the cell relation outside what the key touches, and the unit placed at b."""
    return f"""(sort file f1 f2 f3)
(sort side x o)
(sort score 0 1)
(sort move (go file file) noop)
(signature cell file side) (signature control side) (signature pick file) (signature rm file file file) (signature put file file file side) (signature world file file file side)
(signature stay file file file side) (signature bad file file) (signature square file) (signature touched file file) (signature known file) (signature role side) (signature does side move) (signature legal side move) (signature terminal) (signature goal side score)
(role x) (role o) (known f1) (known f2)
(init (cell f1 x)) (init (cell f2 o)) (init (control x))
(<= (square ?a) (true (cell ?a ?s)))
(<= (pick ?a) (true (cell ?a x)))
(<= (touched ?a ?b) (pick ?a) (square ?b) (distinct ?a ?b))
(<= (rm ?a ?b ?a) (touched ?a ?b))
(<= (rm ?a ?b ?b) (touched ?a ?b))
(<= (put ?a ?b ?b ?s) (touched ?a ?b) (true (cell ?a ?s)))
{world_clauses}
(<= (world ?a ?b ?c ?s) (put ?a ?b ?c ?s))
(<= (world ?a ?b ?c ?s) (stay ?a ?b ?c ?s))
(<= (bad ?a ?b) (touched ?a ?b) (world ?a ?b ?c o) (distinct ?c f3))
{extra}
(<= (legal ?r (go ?a ?b)) (true (control ?r)) (touched ?a ?b) (not (bad ?a ?b)))
(<= (legal ?o noop) (true (control ?c)) (distinct ?c ?o) (role ?o))
(<= (next (control o)) (true (control x)))
(<= (next (control x)) (true (control o)))
(<= (next (cell ?c ?s)) (true (cell ?c ?s)))
(<= terminal (true (control o)))
(<= (goal x 1) terminal) (<= (goal o 0) terminal)
"""


WORLD_OK = "(<= (stay ?a ?b ?c ?s) (touched ?a ?b) (true (cell ?c ?s)) (not (rm ?a ?b ?c)))"


def test_a_toy_with_the_shape_is_found_and_a_toy_that_reads_the_key_outside_the_world_is_refused():
    T = translator()
    good = small("", WORLD_OK)
    d, f = found(good)
    assert [s["R"] for s in f.found] == ["bad"], (f.found, f.refused)
    s = f.found[0]
    assert (s["H"], s["B"], s["m"], s["p"]) == ("world", "true.cell", 2, 2)
    # a literal over a relation that does not read the state with a key variable, inside the test
    bad_static = small("(<= (bad ?a ?b) (touched ?a ?b) (known ?a) (world ?a ?b ?c x))", WORLD_OK)
    d2, f2 = found(bad_static)
    assert not f2.found and "does not read the state holds a key variable" in " ".join(f2.refused.values()), f2.refused
    # a test distinct over a key variable
    bad_dist = small("(<= (bad ?a ?b) (touched ?a ?b) (distinct ?a ?b) (world ?a ?b ?c x))", WORLD_OK)
    d3, f3 = found(bad_dist)
    assert not f3.found and "distinct" in " ".join(f3.refused.values()), f3.refused
    # the hypothetical relation has a clause that is not a view
    not_view = small("", WORLD_OK + "\n(<= (world ?a ?b ?c ?s) (true (cell ?c ?s)) (touched ?a ?b))")
    d4, f4 = found(not_view)
    assert not f4.found and "not a view" in " ".join(f4.refused.values()), f4.refused
    print("OBS 79.56 B toy: the shape found in the toy (test bad, world from the cell relation, key 2 places), refused with its reason where a clause reads the key through a static relation, through a distinct, and where the world has a clause that is not a view")


def test_a_description_with_no_hypothetical_relation_is_left_unchanged_and_the_head_says_so():
    T = translator()
    text = "(role x) (init (a 1)) (signature a n) (sort n (1 2)) (<= (legal x go) (true (a ?n))) (<= (next (a ?n)) (true (a ?n))) (<= terminal (true (a 2))) (<= (goal x 1) terminal)"
    base = [p for p in T.PASSES if p != "carry"]
    src = T.translate(text, "g", base + ["support"])
    heads = [x for x in src.split("\n", 12)[:12] if x.startswith("// support")]
    assert heads and "run_support" not in src and "SpCount" not in src, heads
    without = T.translate(text, "g", base)
    assert "support" not in without
    stripped = "\n".join(x for x in src.split("\n") if not x.startswith("// support")).replace("// passes: " + ",".join(base) + ",support", "// passes: " + ",".join(base)).replace('"topdown", "support"]', '"topdown"]')
    assert stripped == without                    # nothing but the head differs
    print(f"OBS 79.56 B no shape: a description with no hypothetical relation gives the module without the pass, but for the head lines ({heads[0][:100]})")


@pytest.mark.parametrize("game", GAMES)
def test_the_modules_without_the_pass_have_no_trace_of_it_and_the_kept_module_names_what_it_found(game):
    for variant in ("", "_fast", "_carry"):
        src = (GEN / f"{game}{variant}.rs").read_text()
        assert "support" not in src.split("\n", 6)[3] and "SpCount" not in src and "run_support" not in src, (game, variant)
    path = GEN / f"{game}_support.rs"
    if game == "antichess":
        assert not path.exists()            # the shape is there, no question reads the test: no module is kept
        return
    head = path.read_text().split("\n", 12)
    lines = [x for x in head if x.startswith("// support")]
    assert any(x.startswith("// support: test ") for x in lines), (game, lines)
    assert "passes: " in head[3] and head[3].endswith(",support")


def test_the_kept_support_modules_are_those_the_translator_makes_now():
    T = translator()
    import importlib
    spec = importlib.util.spec_from_file_location("gdl_to_rust_make", ROOT / "scripts" / "gdl_to_rust_make.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    for game, sources in mk.GAMES.items():
        path = GEN / f"{game}_support.rs"
        if not path.exists():
            continue
        text = T.compose_text(sources) if len(sources) > 1 else sources[0].read_text()
        head = path.read_text().split("\n", 3)
        assert head[1] == f"// description-sha256: {T.sha(text.encode())}", game
        assert head[2] == f"// translator-sha256: {T.translator_hash()}", game


def test_the_lean_lemma_has_no_sorry_and_the_translator_hash_is_in_the_kept_files():
    lean = (ROOT / "lean" / "ideas" / "Ideas" / "Support.lean").read_text()
    code = "\n".join(x for x in lean.splitlines() if not x.strip().startswith("--"))
    assert "sorry" not in code and "axiom " not in code and "theorem Test.run_congr" in lean
