"""PLAN 79.41 (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ... made fast piece by piece under tests"; ledger row 45; piece: rules): the gate of the passes base, shift, scan and cut.
  (1) the passes exist, the file with every pass of each game names them and is the kept file (the hashes, a fresh translation), and the files of the single passes are made here, not kept;
  (2) the passes decide from shapes: the translator's source names no relation, sort or constant of any description under rules/; the same decisions come from a description with every name replaced;
  (3) what each new pass recognises and refuses, on small descriptions written here, each refusal naming its clause: the pass base unfolds a view whose clause is a product and refuses a view that is read negated; the pass shift compiles
      a walk along a step relation whose rows are displacements and refuses, naming the relation, one that is not; the pass cut and the pass scan are shown in the text of the module;
  (4) a small game that is not a chess, translated with each new pass and compiled with rustc here, answers every relation as the reader's evaluator does on every state of a list;
  (5) the gate lists this file after the one of 79.40.
The piece-by-piece comparisons against the plain module on the chess games are the cargo tests of cs_game (translated_pieces.rs; the single passes with `--cfg singles`)."""
from __future__ import annotations

import importlib.util
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "rust" / "engine" / "game" / "tests" / "generated"
TRANSLATOR = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "gdl_to_rust.py"
GAMES = ("chess", "atomic")
NEW = ("scan", "cut", "shift", "base")


def sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def translator():
    from catspace.incubator.analyst.discovery import gdl_to_rust as T
    return T


def t40():
    return sibling("test_plan79_40")


# ------------------------------------------------------------------------------------------------------------------------------------------ (1)

def test_the_new_passes_are_named_and_the_every_pass_file_of_each_game_names_them():
    T = translator()
    assert set(NEW) <= set(T.PASSES)
    for game in GAMES:
        head = t40().head_line((GEN / f"{game}_fast.rs").read_text(), "passes")
        assert head.split(",") == [p for p in T.PASSES if p != "carry"], game
    print(f"OBS 79.41 F passes {','.join(T.PASSES)}: the kept file of each game names every one")


def test_a_pass_that_is_not_named_is_an_error_and_base_brings_scan_with_it():
    T = translator()
    with pytest.raises(T.Untranslatable):
        T.Plan(T.Description(WALK), ["nothing"])
    p = T.Plan(T.Description(WALK), ["bits", "base"])
    assert "scan" in p.passes and "base" in p.passes


# ------------------------------------------------------------------------------------------------------------------------------------------ (2)

def test_the_translator_names_no_relation_sort_or_constant_of_any_description():
    words = set(re.findall(r"[a-z_][a-z0-9_]*", TRANSLATOR.read_text().lower()))
    names = t40().all_names()
    hit = sorted(words & names)
    print(f"OBS 79.41 G scan: {len(names)} relation, sort and constant names of the descriptions under rules/, {len(words)} words in the translator's source, {len(hit)} in both")
    assert len(names) > 250 and not hit, hit


@pytest.mark.parametrize("game", GAMES)
def test_the_new_passes_give_the_same_decisions_on_a_description_with_every_name_replaced(game):
    T = translator()
    m = t40()
    text = m.description_of(game)
    d = T.Description(text)
    d2 = T.Description(m.rename(text, d))
    a, b = T.Plan(d, T.PASSES), T.Plan(d2, T.PASSES)

    def shape(p):
        return {"unfolded": len(p.unfolded), "refused": sorted(len(v) > 0 for v in p.base_refused.values()), "tables": sorted((v[2], len(v[1])) for v in p.shift_tables.values()), "shift_refused": len(p.shift_refused)}
    assert shape(a) == shape(b), (shape(a), shape(b))
    assert len(a.unfolded) >= 2 and a.shift_tables and not a.shift_refused
    print(f"OBS 79.41 G renamed {game}: {len(a.unfolded)} views unfolded, {len(a.base_refused)} refused, {len(a.shift_tables)} step table, {len(a.bit_rec)} walks; the same with every name replaced")


# ------------------------------------------------------------------------------------------------------------------------------------------ (3)

WALK = """(sort cell n0 n1 n2 n3 n4 n5 n6 n7)
(sort who p q)
(sort score 0 100)
(sort move (go cell) noop)
(signature cell cell) (signature up cell cell) (signature down cell cell) (signature ray_up cell cell) (signature ray_down cell cell) (signature block cell) (signature piece cell) (signature control who) (signature role who) (signature opponent who who)
(signature does who move) (signature legal who move) (signature terminal) (signature goal who score) (signature occupied cell) (signature free cell)
(signature ahead cell cell) (signature see cell)

(role p) (role q) (opponent p q) (opponent q p)
(cell n0) (cell n1) (cell n2) (cell n3) (cell n4) (cell n5) (cell n6) (cell n7)
(up n0 n1) (up n1 n2) (up n2 n3) (up n3 n4) (up n4 n5) (up n5 n6) (up n6 n7)
(down n1 n0) (down n2 n1) (down n3 n2) (down n4 n3) (down n5 n4) (down n6 n5) (down n7 n6)
(init (piece n0)) (init (block n4)) (init (control p))
(<= (occupied ?n) (true (block ?n)))
(<= (free ?n) (cell ?n) (not (occupied ?n)))
(<= (ray_up ?a ?b) (up ?a ?b))
(<= (ray_up ?a ?c) (ray_up ?a ?b) (free ?b) (up ?b ?c))
(<= (ray_down ?a ?b) (down ?a ?b))
(<= (ray_down ?a ?c) (ray_down ?a ?b) (free ?b) (down ?b ?c))
(<= (ahead ?a ?n) (true (piece ?a)) (cell ?n) (not (occupied ?n)))
(<= (ahead ?a ?n) (true (block ?a)) (true (piece ?n)))
(<= (see ?n) (ahead ?a ?n) (free ?n))
(<= (legal ?w (go ?b)) (true (control ?w)) (true (piece ?a)) (ray_up ?a ?b) (free ?b) (see ?b))
(<= (legal ?w (go ?b)) (true (control ?w)) (true (piece ?a)) (ray_down ?a ?b) (free ?b))
(<= (legal ?o noop) (true (control ?w)) (opponent ?w ?o))
(<= (next (piece ?c)) (does ?w (go ?c)))
(<= (next (block ?n)) (true (block ?n)))
(<= (next (control ?o)) (true (control ?w)) (opponent ?w ?o))
(<= terminal (true (piece n7)))
(<= (goal ?w 100) (role ?w) (true (control ?w)))
(<= (goal ?w 0) (role ?w) (not (true (control ?w))))
"""
BLIND = "(signature blind cell)\n(<= (blind ?n) (cell ?n) (not (ahead n0 ?n)))\n(<= (legal ?o noop) (true (control ?w)) (opponent ?w ?o) (blind n1))\n"


def plan(passes, text=WALK):
    T = translator()
    return T.Plan(T.Description(text), passes)


def test_the_pass_base_unfolds_a_view_whose_clause_is_a_product_into_the_clauses_that_read_it():
    p = plan(["bits", "base"])
    assert p.unfolded == ["ahead"], (p.unfolded, p.base_refused)
    assert not p.base_refused
    reader = next(r for r in p.rules[next(k for k, (c, _r) in enumerate(p.d.strata) if "see" in c)])
    assert len([r for r in p.rules[next(k for k, (c, _r) in enumerate(p.d.strata) if "see" in c)]]) == 2 and "(with ahead unfolded)" in reader.clause.text and all(st[1] != "ahead" for st in reader.steps if st[0] != "distinct")
    print(f"OBS 79.41 H base: unfolded {p.unfolded}, refused {p.base_refused}")


def test_a_view_that_is_read_negated_is_refused_with_the_reading_clause_named():
    p = plan(["base"], WALK + BLIND)
    assert "ahead" in p.base_refused and "blind" in p.base_refused["ahead"] and "negated" in p.base_refused["ahead"], p.base_refused
    assert not p.unfolded
    print(f"OBS 79.41 H base refusal: {p.base_refused['ahead'][:160]}")


def test_a_view_whose_head_is_not_a_list_of_distinct_variables_is_refused_with_the_clause_named():
    text = WALK.replace("(<= (ahead ?a ?n) (true (block ?a)) (true (piece ?n)))", "(<= (ahead ?a ?a) (true (block ?a)) (true (piece ?a)))")
    p = plan(["base"], text)
    assert "ahead" in p.base_refused and "distinct variables" in p.base_refused["ahead"], p.base_refused


def test_the_unfolded_view_is_not_made_for_the_questions_but_is_still_made_in_full_where_the_tests_ask():
    T = translator()
    src = T.translate(WALK, "walk", ["bits", "base"])
    assert 'pub const UNFOLDED: &[&str] = &["ahead"];' in src
    cone = re.search(r"const READ_CONE: &\[usize\] = &\[([0-9, ]*)\];", src).group(1)
    alld = re.search(r"const ALL_DYNAMIC: &\[usize\] = &\[([0-9, ]*)\];", src).group(1)
    assert len(cone.split(",")) < len(alld.split(",")), (cone, alld)
    assert "(with ahead unfolded)" in src


def test_the_pass_shift_compiles_a_walk_along_a_table_of_displacements():
    T = translator()
    p = plan(["bits", "shift"])
    assert len(p.shift_tables) == 2 and not p.shift_refused, p.shift_refused
    tabs = sorted(sorted(dd.items()) for _tid, by, _n, _sel in p.shift_tables.values() for dd in by)
    assert tabs == [[(-1, 0b11111110)], [(1, 0b01111111)]], tabs
    src = T.translate(WALK, "walk", ["bits", "shift"])
    assert "if dl >= 0 { y << dl } else { y >> (-dl) }" in src and "static SH0:" in src and "static SH1:" in src
    print(f"OBS 79.41 H shift: two tables, displacements and masks {tabs}")


def test_the_pass_shift_refuses_more_displacements_than_the_limit_naming_the_relation():
    T = translator()
    old = T.SHIFT_MAX
    try:
        T.SHIFT_MAX = 0
        p = plan(["bits", "shift"])
    finally:
        T.SHIFT_MAX = old
    why = " ".join(p.shift_refused.values())
    assert len(p.shift_refused) == 2 and "displacements" in why and "(<= (ray_up" in why, p.shift_refused
    assert not p.shift_tables


def test_the_pass_shift_refuses_a_hop_relation_that_reads_the_state_naming_the_relation():
    text = WALK.replace("(<= (ray_up ?a ?c) (ray_up ?a ?b) (free ?b) (up ?b ?c))", "(<= (ray_up ?a ?c) (ray_up ?a ?b) (free ?b) (hop ?b ?c))\n(signature hop cell cell)\n(<= (hop ?b ?c) (up ?b ?c) (not (occupied ?c)))")
    p = plan(["bits", "shift"], text)
    assert "ray_up" in p.shift_refused and "reads the state" in p.shift_refused["ray_up"] and "hop" in p.shift_refused["ray_up"], p.shift_refused
    assert "ray_down" not in p.shift_refused


def test_the_pass_shift_refuses_a_walk_with_a_distinct_test_naming_the_clause():
    text = WALK.replace("(<= (ray_down ?a ?c) (ray_down ?a ?b) (free ?b) (down ?b ?c))", "(<= (ray_down ?a ?c) (ray_down ?a ?b) (free ?b) (down ?b ?c) (distinct ?a ?c))")
    p = plan(["bits", "shift"], text)
    assert "distinct" in p.shift_refused["ray_down"] and "(<= (ray_down" in p.shift_refused["ray_down"], p.shift_refused


def test_the_pass_cut_leaves_the_literals_of_a_complete_head_in_a_block_and_the_pass_scan_loops_over_the_words_held():
    T = translator()
    cut = T.translate(WALK, "walk", ["bits", "cut"])
    plain = T.translate(WALK, "walk", ["bits"])
    assert "break 'h" in cut and "break 'h" not in plain
    text = (ROOT / "rules" / "chess.gdl").read_text()
    scan = T.translate(text, "m", ["bits", "scan"])
    assert "let sc" in scan and "nz.len() < nc" in scan and "nz.len() < nc" not in T.translate(text, "m", ["bits"])


# ------------------------------------------------------------------------------------------------------------------------------------------ (4)

def states(n: int) -> list:
    rng = random.Random(11)
    cells = [f"n{i}" for i in range(8)]
    out = []
    for _ in range(n):
        p = rng.choice(cells)
        blocks = rng.sample(cells, rng.randint(0, 5))
        facts = [("piece", (p,)), ("control", (rng.choice("pq"),))] + [("block", (b,)) for b in blocks]
        out.append(facts)
    return out


def reference(sts) -> list:
    G = sibling("test_plan79_40").reader()
    prog = G.Program(WALK)
    d = translator().Description(WALK)
    dynamic = [r for r in d.nodes if not d.static[r]]
    out = []
    for facts in sts:
        db = prog.run([(f"true.{f}", a) for f, a in facts], does=())
        out.append({r: sorted(" ".join(G.term_text(t) if not isinstance(t, str) else t for t in row) for row in db.facts(r)) for r in dynamic})
    return out


@pytest.mark.skipif(shutil.which("rustc") is None, reason="no rustc on this machine")
@pytest.mark.parametrize("passes", [["bits", "scan"], ["bits", "cut"], ["bits", "shift"], ["bits", "base"], ["bits", "scan", "cut", "shift", "base"], ["dense", "demand", "copy", "delta", "bits", "order", "keep", "scan", "cut", "shift", "base"]],
                         ids=["scan", "cut", "shift", "base", "scan-cut-shift-base", "every pass"])
def test_a_small_game_that_is_not_chess_translated_with_the_new_passes_answers_every_relation_as_the_evaluator_does(passes, tmp_path):
    m = t40()
    T = translator()
    (tmp_path / "game.rs").write_text(T.translate(WALK, "game", passes))
    (tmp_path / "main.rs").write_text(m.DRIVER)
    exe = tmp_path / "walk"
    r = subprocess.run(["rustc", "--edition", "2021", "-O", "-A", "warnings", "-o", str(exe), str(tmp_path / "main.rs")], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    sts = states(60)
    text = "\n".join(";".join(f"{f} {' '.join(a)}" for f, a in facts) for facts in sts) + "\n"
    r = subprocess.run([str(exe)], input=text, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    blocks = r.stdout.split("END\n")[:-1]
    assert len(blocks) == len(sts)
    ref = reference(sts)
    compared = 0
    for i, (blk, want) in enumerate(zip(blocks, ref)):
        rows = {}
        for line in blk.splitlines():
            parts = line.split("\t")
            if parts[0] == "REL":
                rows.setdefault(parts[1], [])
            elif parts[0] == "ROW":
                rows.setdefault(parts[1], []).append(parts[2])
        assert rows, "the module made no relation"
        for name in rows:
            assert sorted(set(rows[name])) == want[name], (i, passes, name, sorted(rows[name]), want[name], sts[i])
            compared += 1
    assert compared > 60 * 5
    print(f"OBS 79.41 I walk {passes}: {len(sts)} states, {compared} relations compared with the evaluator's, 0 disagreements")


# ------------------------------------------------------------------------------------------------------------------------------------------ (5)

def test_the_gate_lists_this_test_after_the_one_of_plan_79_40():
    text = (ROOT / "scripts" / "gate.py").read_text()
    assert "test_plan79_41" in text and text.index("test_plan79_40") < text.index("test_plan79_41")
