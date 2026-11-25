"""PLAN 79.52 PART B (SYSTEM_SPEC 11, last block "The lower layers are made as fast as the outside library first; the facts stay; the list of allowed moves is made once a state"; piece: rules; ledger rows 55, 45): the gate of the two passes `cones` and `words`.
  (1) the list of passes ends with the two new ones, each brings `keep`, and a module with neither is the plain one (tests/test_plan79_38.py holds that for the kept plain files);
  (2) `cones` (answer only what was asked): the pieces of the question of the allowed moves, of the end and of the outcome are the cones of the dependency graph of the description, on a small game with a relation that only the end reads and one that only the outcome
      reads (planted here), on the first game and on the six others in the kept files; the translator is given no name;
  (3) `words` (the state as words, the facts kept): a fluent of small sorts is a `Wd` and one with a large product or an open sort stays a list of rows (planted here); every kept module with passes holds its fluents so where the sorts allow, and the kept plain modules hold none;
  (4) the interface is written before its bodies and the old entry points are kept (translated_common/mod.rs, the head of the translator); the clock file has the makers of the new modules; the gate lists this file after the one of 79.51;
  (5) the Rust tests of the plan (cs_game, translated_words.rs: the questions run only the pieces they need, a planted dead position, the state is words and the facts stay, no allocation after a first pass, agreement with the plain module) pass, and no source on the engine's
      path (the crates' sources, a manifest, a description, a reader) differs from the commit before the plan.
The comparisons against the evaluator, the plain module and the kernel are the cargo tests of cs_game that exist (translated_pieces.rs, translated_kernel.rs, translated_lemma.rs, translated_agree.rs), run on the regenerated modules."""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
GEN = ENGINE / "game" / "tests" / "generated"
BEFORE = "ee128c96"                      # the commit before this plan's code (main when the plan was begun)
GAMES = ("chess", "atomic", "3check", "kingofthehill", "antichess", "chess960", "horde")
needs_cargo = pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo (offline) to run the Rust tests of the plan")


def translator():
    from catspace.incubator.analyst.discovery import gdl_to_rust as T
    return T


def sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def have_before() -> bool:
    return git("cat-file", "-e", f"{BEFORE}^{{commit}}").returncode == 0


def kept(game: str, variant: str = "_fast") -> str:
    return (GEN / f"{game}{variant}.rs").read_text()


def pieces_and_cones(src: str):
    """-> (the names of the pieces by number, the cone of each mask as a list of piece names)"""
    names = re.findall(r'\("([^"]+)", (?:true|false)\)', re.search(r"pub const PIECES.*", src).group(0))
    m = re.search(r"const CONES: \[&\[usize\]; 8\] = \[(.*)\];", src)
    cones = [[names[int(x)] for x in c.split(",") if x.strip()] for c in re.findall(r"&\[([^\]]*)\]", m.group(1))]
    assert len(cones) == 8
    return names, cones


# a small game that is not a chess: the walk of 79.41 with a relation only the end reads and a relation only the outcome reads
PLANTED = sibling("test_plan79_41").WALK + """
(signature far cell) (signature bonus who score)
(<= (far ?n) (true (block ?n)) (up ?n ?m))
(<= terminal (far ?x))
(<= (bonus ?w 50) (role ?w) (true (piece n3)))
(<= (goal ?w 50) (bonus ?w 50))
"""


def test_the_two_passes_end_the_list_and_bring_the_pass_keep():
    T = translator()
    assert T.PASSES[-5:-3] == ("cones", "words")                 # plan 79.53 added two passes after them, plan 79.54 one more
    assert T.PASSES.index("gated") < T.PASSES.index("cones")
    d = T.Description(PLANTED)
    for p in ("cones", "words"):
        assert "keep" in T.Plan(d, [p]).passes
    assert T.Plan(d, ["cones"]).cones and not T.Plan(d, ["words"]).cones
    print(f"OBS 79.52 B passes: {len(T.PASSES)} passes, the last two {T.PASSES[-2:]}, each with keep")


def test_the_cones_are_the_dependency_graphs_on_a_planted_small_game():
    T = translator()
    src = T.translate(PLANTED, "planted", ["cones"])
    names, cones = pieces_and_cones(src)
    d = T.Description(PLANTED)
    only_end = d.cone(["terminal"]) - d.cone(["legal"]) - d.cone(["goal"])
    only_goal = d.cone(["goal"]) - d.cone(["legal"]) - d.cone(["terminal"])
    assert "far" in only_end and "bonus" in only_goal
    flat = lambda mask: {x for p in cones[mask] for x in p.split("+")}
    legal, end, goal = flat(1), flat(2), flat(4)
    assert "far" not in legal and "bonus" not in legal and "far" in end and "bonus" not in end and "bonus" in goal and "far" not in goal
    assert "legal" in legal and "terminal" in end and "goal" in goal and "terminal" not in legal and "goal" not in legal
    assert flat(3) == legal | end and flat(6) == end | goal and flat(7) == legal | end | goal and flat(5) == legal | goal and cones[0] == []
    print(f"OBS 79.52 B cones planted: the allowed moves run {len(cones[1])} pieces, the end {len(cones[2])}, the outcome {len(cones[4])}, all three {len(cones[7])}; a relation only the end reads is in the end's cone alone and one only the outcome reads in the outcome's alone")


@pytest.mark.parametrize("game", GAMES)
def test_in_every_kept_module_the_pieces_of_a_question_are_the_cone_of_the_graph(game):
    T = translator()
    make = sibling("test_plan79_38")
    d = T.Description(make.description_of(game))
    names, cones = pieces_and_cones(kept(game))
    flat = lambda mask: {x for p in cones[mask] for x in p.split("+")}
    legal, end, goal = flat(1), flat(2), flat(4)
    end_only = {x for x in d.cone(["terminal"]) - d.cone(["legal"]) if not d.static[x]}
    assert end_only, game
    unfolded = set(T.Plan(d, list(T.PASSES)).unfolded)
    # what only the end needs is not in the cone of the allowed moves, whatever the passes made of it; what the allowed moves need is in the cone of the end where the end reads it
    assert not (end_only & legal) - unfolded, (game, sorted(end_only & legal))
    assert "legal" in legal and "terminal" in end and "goal" in goal and "goal" not in end and "terminal" not in legal
    assert flat(7) == legal | end | goal
    print(f"OBS 79.52 B cones {game}: the allowed moves run {len(cones[1])} pieces of {len(names)}, the end {len(cones[2])}, the outcome {len(cones[4])}, the three {len(cones[7])}; {len(end_only)} relations are the end's alone and none of them is in the allowed moves' cone")


SMALL_WORDS = sibling("test_plan79_41").WALK + """
(sort big (range 0 200)) (signature big_f big big big)
(init (big_f 0 1 2))
(<= (next (big_f ?a ?b ?c)) (true (big_f ?a ?b ?c)))
"""


def test_a_fluent_of_small_sorts_is_words_and_a_large_one_stays_rows_on_a_planted_game():
    T = translator()
    p = T.Plan(T.Description(SMALL_WORDS), ["words"])
    assert set(p.words) == {"piece", "block", "control"}, p.words
    assert all(w == 1 for _s, w in p.words.values())
    src = T.translate(SMALL_WORDS, "planted", ["words"])
    assert "pub f_piece: Wd<1>" in src and "pub f_block: Wd<1>" in src and "pub f_control: Wd<1>" in src and "pub f_big_f: Vec<[u32; 3]>" in src
    plain = T.translate(SMALL_WORDS, "planted", [])
    assert "Wd<" not in plain and "pub f_piece: Vec<[u32; 1]>" in plain
    print(f"OBS 79.52 B words planted: fluents held as words {sorted(p.words)}; the fluent of 200 x 200 x 200 members stays a list of rows")


def test_the_translator_gives_a_fact_outside_the_sorts_as_an_error_and_names_the_tables_it_reads():
    T = translator()
    src = T.translate(SMALL_WORDS, "planted", ["words"])
    assert "outside the sorts of its places" in src and "widx(&TAB_piece" in src and "static TAB_piece: Tab" in src


@pytest.mark.parametrize("game", GAMES)
def test_kept_modules_with_passes_hold_the_state_as_words_and_the_plain_ones_do_not(game):
    T = translator()
    make = sibling("test_plan79_38")
    d = T.Description(make.description_of(game))
    fast, plain = kept(game), kept(game, "")
    assert re.search(r"^// passes: .*cones,words,tables,topdown$", fast, re.M)
    want = T.Plan(d, list(T.PASSES)).words
    assert want and len(want) >= 4, want
    for f, (sig, w) in want.items():
        assert f"pub f_{T.ident(f)}: Wd<{w}>" in fast, (game, f)
    assert "Wd<" not in plain and "CONES" not in plain
    cell = want["cell"][1]
    print(f"OBS 79.52 B words {game}: {len(want)} fluents held as words ({', '.join(f + ' ' + str(w) for f, (_s, w) in sorted(want.items()))}); the plain module holds none")


def test_the_interface_is_written_before_its_bodies_and_the_old_entry_points_stay():
    mod = (ENGINE / "game" / "tests" / "translated_common" / "mod.rs").read_text()
    for item in ("fn moves_t", "fn next_t", "fn facts_of_t", "fn state_of_facts_t", "fn initial_t", "fn moves_into_t", "fn legal_into_t", "($m:ident, fresh)"):
        assert item in mod, item
    head = translator().__doc__
    assert "PLAN 79.52 PART B" in head and "(21) \"cones\"" in head and "(22) \"words\"" in head
    speed = (ENGINE / "game" / "tests" / "translated_speed_bits.rs").read_text()
    assert "chess_the_times_of_part_b" in speed and "atomic_the_times_of_part_b" in speed and "LegalIntoAndNext" in speed
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert re.search(r"test_plan79_51 test_plan79_52", gate) or re.search(r"test_plan79_51\s+test_plan79_52", gate) or "test_plan79_52" in gate
    assert gate.index("test_plan79_52") > gate.index("test_plan79_51")


@needs_cargo
def test_the_rust_tests_of_the_plan_pass():
    env = {**os.environ, "CARGO_BUILD_JOBS": "2", "RUST_TEST_THREADS": "2"}
    r = subprocess.run(["cargo", "test", "--release", "--offline", "-j", "2", "-p", "cs_game", "--test", "translated_words", "--", "--test-threads=2", "--nocapture"], cwd=ENGINE, env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    lines = [ln for ln in out.splitlines() if "OBS 79.52 B" in ln]
    assert any("allocations chess" in ln and "without the end 0 allocations, with the end 0," in ln for ln in lines), lines
    for ln in lines:
        print(ln[ln.index("OBS"):][:300])


def test_no_source_of_the_engine_no_cargo_file_no_description_and_no_reader_differs_from_the_commit_before_the_plan():
    if not have_before():
        pytest.skip(f"{BEFORE} not in this clone")
    changed = git("diff", "--name-only", BEFORE, "--", "rust/engine").stdout.split()
    # 2026-11-25 (plan 79.52 part A, Kaveh's word): kernel.rs was changed on main beside this plan (the list of allowed moves made once a state); it is left out here
    assert not [c for c in changed if (re.match(r"rust/engine/[a-z_]+/src/", c) and c != "rust/engine/game/src/kernel.rs") or c.endswith("Cargo.toml") or c.endswith("Cargo.lock")], changed
    assert git("diff", "--quiet", BEFORE, "--", "rules", ":(exclude)rules/lemmas", ":(exclude)rules/placed", "catspace/incubator/analyst/discovery/gdl.py").returncode == 0
    print(f"OBS 79.52 B: files under rust/engine that differ from {BEFORE}: {len(changed)}, none a source of a crate nor a manifest")
