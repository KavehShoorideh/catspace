"""PLAN 79.53 (SYSTEM_SPEC 11, last block, (vi), (vii) and (viii); piece: rules; ledger rows 55, 45): the gate of the pass `tables` (tables for walks) and of the pass `topdown` (the top-down way beside the bottom-up one).
  (1) the list of passes ends "tables", "topdown"; tables brings bits and shift, topdown brings words and cones;
  (2) PART A. What the pass tables finds to be a walk in each of the seven games (printed); a planted near-walk is refused with its reason (a line that branches, a chain that returns to a member), and a walk it takes gives, in a small game that is not a chess compiled with rustc here, every relation the evaluator gives;
  (3) PART B. The top-down way: it is made for each of the seven games (`Plan.td_refused` is blank); a planted game with a cycle in its step relation (a walk that must end), a negation over a walk and a negation of a fact, compiled with rustc here, answers the end and the legal turn top-down as the evaluator and the bottom-up way do;
      a description the way cannot take is refused naming why and keeps its bottom-up answers; the way recorded for each question is the one the clocks gave (WAYS), the translator chooses nothing;
  (4) the interface is written before its bodies and the old entry points stay (translated_common/mod.rs, the head of the translator); the gate lists this file after the one of 79.52; no source on the engine's path differs from the commit before plan 79.52;
  (5) the Rust test of part B (cs_game, translated_topdown.rs: the end and the legal turn top-down equal bottom-up at every fixture position and every state of seeded games, for the seven games; each relation of the cone asked with no place bound) passes.
The comparisons of the tabled modules to the plain ones are the cargo tests that exist (translated_pieces.rs, translated_kernel.rs, translated_lemma.rs), run on the regenerated modules; the clocks are translated_speed_bits.rs (`chess_the_times_of_plan_53`, `atomic_the_times_of_plan_53`)."""
from __future__ import annotations

import importlib.util
import os
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
GEN = ENGINE / "game" / "tests" / "generated"
BEFORE = "ee128c96"
GAMES = ("chess", "atomic", "3check", "kingofthehill", "antichess", "chess960", "horde")
needs_cargo = pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo (offline) to run the Rust tests")
needs_rustc = pytest.mark.skipif(shutil.which("rustc") is None, reason="no rustc on this machine")


def sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def translator():
    from catspace.incubator.analyst.discovery import gdl_to_rust as T
    return T


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def kept(game: str, variant: str = "_fast") -> str:
    return (GEN / f"{game}{variant}.rs").read_text()


WALK = sibling("test_plan79_41").WALK


def test_the_passes_end_the_list_and_bring_what_they_read():
    T = translator()
    assert T.PASSES[-2:] == ("tables", "topdown") and T.PASSES.index("words") < T.PASSES.index("tables")
    d = T.Description(WALK)
    p = T.Plan(d, ["tables"])
    assert "bits" in p.passes and "shift" in p.passes and p.tables
    q = T.Plan(d, ["topdown"])
    assert "words" in q.passes and "cones" in q.passes and "keep" in q.passes and q.topdown
    print(f"OBS 79.53 passes: the last two {T.PASSES[-2:]}; tables brings {sorted(set(p.passes) - {'tables'})}, topdown brings {sorted(set(q.passes) - {'topdown'})}")


# ------------------------------------------------------------------------------------------------------------------------------------------ part A

@pytest.mark.parametrize("game", GAMES)
def test_what_the_pass_tables_finds_to_be_a_walk_in_each_game(game):
    T = translator()
    make = sibling("test_plan79_38")
    T.translate(make.description_of(game), "x", list(T.PASSES))
    p = T.translate.last_plan
    assert p.tables_found
    assert not p.tables_refused, p.tables_refused
    found = {rel: v for rel, v in sorted(p.tables_found.items())}
    for rel, (lines, words) in found.items():
        assert lines >= 1 and words <= T.TABLE_WORDS
    print(f"OBS 79.53 A tables {game}: walks with a table {', '.join(f'{r} ({n} lines, {w} words)' for r, (n, w) in found.items())}; refused: none")


def test_a_planted_near_walk_is_refused_with_its_reason_and_keeps_the_shifts():
    T = translator()
    # a line that branches: one member has two next ones
    branch = WALK.replace("(up n0 n1)", "(up n0 n1) (up n0 n2)")
    T.translate(branch, "g", ["bits", "shift", "tables"])
    p = T.translate.last_plan
    assert "ray_up" in p.tables_refused and "displacements" in p.tables_refused["ray_up"] and "ray_up" not in p.tables_found, (p.tables_refused, p.tables_found)
    assert "ray_down" in p.tables_found
    # a chain that returns to a member: the step relation is a ring
    ring = WALK.replace("(up n6 n7)", "(up n6 n7) (up n7 n0)").replace("(down n7 n6)", "(down n7 n6) (down n0 n7)")
    T.translate(ring, "g", ["bits", "shift", "tables"])
    q = T.translate.last_plan
    assert "branches" in q.tables_refused.get("ray_up", "") and "branches" in q.tables_refused.get("ray_down", ""), q.tables_refused
    print(f"OBS 79.53 A refused: a line that branches: {p.tables_refused['ray_up'][:110]}; a ring (the last member links to the first): {q.tables_refused['ray_up'][:110]}")


def test_the_chain_table_is_the_chains_of_the_displacements():
    T = translator()
    chains, inc = T.chain_table([{1: 0b0111}, {}, {-2: 0b11100}], 5)
    assert chains[0] == [0b01111, 0b01110, 0b01100, 0b01000, 0b10000] and inc[0]
    assert chains[1] == [1, 2, 4, 8, 16]
    assert chains[2][4] == 0b10101 and chains[2][3] == 0b01010 and not inc[2]
    with pytest.raises(T.ShiftRefuse):
        T.chain_table([{1: 0b1, 2: 0b1}], 5)
    with pytest.raises(T.ShiftRefuse):
        T.chain_table([{1: 0b1111}], 4)                         # a member with a next one whose next is outside the tile
    with pytest.raises(T.ShiftRefuse) as e:
        T.chain_table([{0: 0b100}], 4)                          # a member that links to itself: the chain returns
    assert "returns" in str(e.value)


def run_small(text: str, passes, driver: str, lines: list, tmp_path, ways=None) -> list:
    T = translator()
    (tmp_path / "game.rs").write_text(T.translate(text, "game", passes, ways=ways))
    (tmp_path / "main.rs").write_text(driver)
    exe = tmp_path / "small"
    r = subprocess.run(["rustc", "--edition", "2021", "-O", "-A", "warnings", "-o", str(exe), str(tmp_path / "main.rs")], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    r = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.splitlines()


@needs_rustc
def test_a_walk_with_a_table_gives_every_relation_the_evaluator_gives_in_a_small_game_that_is_not_chess(tmp_path):
    m41 = sibling("test_plan79_41")
    m40 = sibling("test_plan79_40")
    T = translator()
    passes = ["bits", "scan", "shift", "tables"]
    (tmp_path / "game.rs").write_text(T.translate(WALK, "game", passes))
    assert T.translate.last_plan.tables_found and "static CH" in (tmp_path / "game.rs").read_text()
    (tmp_path / "main.rs").write_text(m40.DRIVER)
    exe = tmp_path / "walk"
    r = subprocess.run(["rustc", "--edition", "2021", "-O", "-A", "warnings", "-o", str(exe), str(tmp_path / "main.rs")], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    sts = m41.states(120)
    text = "\n".join(";".join(f"{f} {' '.join(a)}" for f, a in facts) for facts in sts) + "\n"
    r = subprocess.run([str(exe)], input=text, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    blocks = r.stdout.split("END\n")[:-1]
    assert len(blocks) == len(sts)
    ref = m41.reference(sts)
    compared = 0
    for blk, want in zip(blocks, ref):
        rows = {}
        for line in blk.splitlines():
            parts = line.split("\t")
            if parts[0] == "REL":
                rows.setdefault(parts[1], [])
            elif parts[0] == "ROW":
                rows.setdefault(parts[1], []).append(parts[2])
        for rel in ("ray_up", "ray_down"):
            if rel in rows:
                assert sorted(rows[rel]) == want[rel], (rel, rows[rel], want[rel])
                compared += len(want[rel])
    assert compared > 100
    print(f"OBS 79.53 A small game: the walks ray_up and ray_down with a table, {len(sts)} states, {compared} rows equal to the evaluator's")


# ------------------------------------------------------------------------------------------------------------------------------------------ part B

# a small game that is not a chess: a ring of six places (a CYCLE in the step relation: the walk must end), a piece that walks round it, blocks; the end holds when the piece is shut in (a negation of a fact is in the walk's condition)
# or when it can go all the way round and no block is on the way (a negation over a relation that is a walk)
RING = """(sort spot n0 n1 n2 n3 n4 n5)
(sort who p q)
(sort score 0 100)
(sort move (go spot) noop)
(signature spot spot) (signature nxt spot spot) (signature reach spot spot) (signature block spot) (signature piece spot) (signature control who) (signature occupied spot) (signature free spot)
(signature trapped spot) (signature live spot) (signature opponent who who) (signature role who) (signature does who move) (signature legal who move) (signature terminal) (signature goal who score)

(role p) (role q) (opponent p q) (opponent q p)
(spot n0) (spot n1) (spot n2) (spot n3) (spot n4) (spot n5)
(nxt n0 n1) (nxt n1 n2) (nxt n2 n3) (nxt n3 n4) (nxt n4 n5) (nxt n5 n0)
(init (piece n0)) (init (block n3)) (init (control p))
(<= (occupied ?n) (true (block ?n)))
(<= (free ?n) (spot ?n) (not (occupied ?n)))
(<= (reach ?a ?b) (nxt ?a ?b))
(<= (reach ?a ?c) (reach ?a ?b) (free ?b) (nxt ?b ?c))
(<= (trapped ?a) (true (piece ?a)) (nxt ?a ?b) (occupied ?b))
(<= (live ?a) (reach ?a ?b) (occupied ?b))
(<= (legal ?w (go ?b)) (true (control ?w)) (true (piece ?a)) (reach ?a ?b) (free ?b))
(<= (legal ?o noop) (true (control ?w)) (opponent ?w ?o))
(<= (next (piece ?c)) (does ?w (go ?c)))
(<= (next (block ?n)) (true (block ?n)))
(<= (next (control ?o)) (true (control ?w)) (opponent ?w ?o))
(<= terminal (trapped ?a))
(<= terminal (true (piece ?a)) (reach ?a ?a) (not (live ?a)))
(<= (goal ?w 100) (role ?w) (true (control ?w)))
(<= (goal ?w 0) (role ?w) (not (true (control ?w))))
"""

DRIVER = r'''
mod game { include!("game.rs"); }
use std::io::Read;
fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let mut g = game::game::Game::new();
    for line in text.lines() {
        let facts: Vec<(String, Vec<String>)> = line.split(';').filter(|x| !x.trim().is_empty()).map(|f| { let mut it = f.split_whitespace(); (it.next().unwrap().to_string(), it.map(|x| x.to_string()).collect()) }).collect();
        let s = g.state_of_facts(&facts).unwrap();
        let mut b = vec![];
        g.legal_moves_into(&s, &mut b);
        println!("{} {} {} {} {}", g.terminal_td(&s), g.terminal_bu(&s), g.has_turn_td(&s), !b.is_empty(), g.td_over);
    }
}
'''


def ring_states(n: int) -> list:
    rng = random.Random(53)
    cells = [f"n{i}" for i in range(6)]
    out = []
    for _ in range(n):
        facts = [("piece", (rng.choice(cells),)), ("control", (rng.choice("pq"),))] + [("block", (b,)) for b in rng.sample(cells, rng.randint(0, 3))]
        out.append(facts)
    return out


def ring_reference(sts) -> list:
    G = sibling("test_plan79_40").reader()
    prog = G.Program(RING)
    out = []
    for facts in sts:
        db = prog.run([(f"true.{f}", a) for f, a in facts], does=())
        term = bool(db.facts("terminal"))
        txt = lambda x: G.term_text(x) if not isinstance(x, str) else x
        legal = [row for row in db.facts("legal") if txt(row[1]) != "noop"]
        out.append((term, bool(legal)))
    return out


@needs_rustc
def test_a_planted_game_with_a_cycle_and_two_negations_answers_the_end_and_the_legal_turn_top_down_as_the_evaluator_does(tmp_path):
    T = translator()
    sts = ring_states(300)
    lines = [";".join(f"{f} {' '.join(a)}" for f, a in facts) for facts in sts]
    got = run_small(RING, list(T.PASSES), DRIVER, lines, tmp_path)
    p = T.translate.last_plan
    assert p.td_refused == "", p.td_refused
    assert len(got) == len(sts)
    want = ring_reference(sts)
    ended = free = 0
    for line, facts, (term, turn) in zip(got, sts, want):
        td_end, bu_end, td_turn, bu_turn, over = line.split()
        assert (td_end == "true", bu_end == "true") == (term, term), (facts, line, term)
        assert (td_turn == "true", bu_turn == "true") == (turn, turn), (facts, line, turn)
        assert over == "0", (facts, line)
        ended += term
        free += turn
    assert 20 < ended < 280 and 20 < free < 280, (ended, free)
    # the tables pass refuses the ring (a chain that returns) and the walk is still answered, top-down by the worklist that does not repeat a row
    assert "branches" in " ".join(p.tables_refused.values()), p.tables_refused
    print(f"OBS 79.53 B planted ring: {len(sts)} states, {ended} where the end holds, {free} with a legal turn; the end, the legal turn, top-down and bottom-up equal to the evaluator's; the walk of the ring ends (a worklist that does not repeat a row), {p.tables_refused and 'the table refused it: the last link goes back to the first'}")


def test_a_description_the_top_down_way_cannot_take_is_refused_naming_why_and_the_module_has_no_top_down_way():
    T = translator()
    # a relation that reads itself twice in a clause (not a linear walk)
    twice = RING + "(signature both spot spot)\n(<= (both ?a ?b) (nxt ?a ?b) (free ?b))\n(<= (both ?a ?c) (both ?a ?b) (both ?b ?c))\n(<= terminal (true (piece ?a)) (both ?a ?a) (not (true (control p))))\n"
    src = T.translate(twice, "g", list(T.PASSES))
    p = T.translate.last_plan
    assert "twice" in p.td_refused and "both" in p.td_refused, p.td_refused
    assert "fn terminal_td" not in src and "fn terminal(" in src                     # the bottom-up way answers, as before
    # a compound term in a literal of the cone
    compound = RING + "(signature mv who move)\n(<= (mv ?w (go ?b)) (true (control ?w)) (true (piece ?b)))\n(<= terminal (mv ?w (go ?b)) (not (true (control p))) (occupied ?b))\n"
    src2 = T.translate(compound, "g", list(T.PASSES))
    assert "compound" in T.translate.last_plan.td_refused, T.translate.last_plan.td_refused
    print(f"OBS 79.53 B refused: a relation that reads itself twice: {p.td_refused[:100]}; a compound term in the cone: {T.translate.last_plan.td_refused[:100]}")


@pytest.mark.parametrize("game", GAMES)
def test_the_top_down_way_is_made_for_every_game_and_the_kept_module_records_the_way_of_each_question(game):
    T = translator()
    make = sibling("test_plan79_38")
    T.translate(make.description_of(game), "x", list(T.PASSES))
    p = T.translate.last_plan
    assert p.td_refused == "" and p.topdown, p.td_refused
    src = kept(game)
    assert "pub fn terminal_td" in src and "pub fn has_turn_td" in src and "pub fn ask_td" in src and 'pub const WAYS: &[(&str, &str)] = &[("end", "bottom-up"), ("legal-turn", "bottom-up")]' in src
    assert "terminal_td" not in kept(game, "") and "WAYS" not in kept(game, "")
    print(f"OBS 79.53 B {game}: the top-down way is made; the kept module records the end and the legal turn as bottom-up (the clocks of 79.53 gave that)")


def test_the_way_of_a_question_is_what_is_recorded_and_the_translator_chooses_nothing():
    T = translator()
    make = sibling("test_plan79_38")
    text = make.description_of("chess")
    a = T.translate(text, "x", list(T.PASSES), ways={"end": "top-down"})
    assert "const WAY_END_TOP_DOWN: bool = true;" in a and "const WAY_TURN_TOP_DOWN: bool = false;" in a and '("end", "top-down")' in a
    b = T.translate(text, "x", list(T.PASSES))
    assert "const WAY_END_TOP_DOWN: bool = false;" in b
    with pytest.raises(T.Untranslatable):
        T.translate(text, "x", list(T.PASSES), ways={"end": "sideways"})
    import json
    rec = json.loads((ROOT / "scripts" / "gdl_to_rust_ways.json").read_text())
    assert rec["chess"] == {"end": "bottom-up", "legal-turn": "bottom-up"} and rec["atomic"] == rec["chess"]
    src = (ROOT / "scripts" / "gdl_to_rust_make.py").read_text()
    assert "gdl_to_rust_ways.json" in src and "chooses nothing" in src


def test_the_interface_is_written_before_its_bodies_and_the_old_entry_points_stay():
    mod = (ENGINE / "game" / "tests" / "translated_common" / "mod.rs").read_text()
    for item in ("fn moves_t", "fn next_t", "fn facts_of_t", "fn state_of_facts_t", "fn moves_into_t", "fn legal_into_t", "fn terminal_td_t", "fn has_move_td_t", "fn ask_td_t", "($m:ident, topdown)"):
        assert item in mod, item
    head = translator().__doc__
    assert "PLAN 79.53 PART A" in head and "(23) \"tables\"" in head and "(24) \"topdown\"" in head and "HOW IT TERMINATES" in head and "HOW A NEGATED CONDITION IS ANSWERED" in head
    speed = (ENGINE / "game" / "tests" / "translated_speed_bits.rs").read_text()
    assert "chess_the_times_of_plan_53" in speed and "atomic_the_times_of_plan_53" in speed and "Work::EndTd" in speed
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert gate.index("test_plan79_53") > gate.index("test_plan79_52")


@needs_cargo
def test_the_rust_test_of_part_b_passes():
    env = {**os.environ, "CARGO_BUILD_JOBS": "2", "RUST_TEST_THREADS": "2"}
    r = subprocess.run(["cargo", "test", "--release", "--offline", "-j", "2", "-p", "cs_game", "--test", "translated_topdown", "--", "--test-threads=2", "--nocapture"], cwd=ENGINE, env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    held = [ln for ln in out.splitlines() if "OBS 79.53 B1 held" in ln]
    over = [ln for ln in out.splitlines() if "OBS 79.53 B1 overflows" in ln]
    assert len(held) == 8 and all(" 0 questions" in ln for ln in over), (len(held), over)
    for ln in held:
        print(ln[ln.index("OBS"):][:260])


def test_no_source_of_the_engine_no_cargo_file_no_description_and_no_reader_differs_from_the_commit_before_plan_79_52():
    if git("cat-file", "-e", f"{BEFORE}^{{commit}}").returncode != 0:
        pytest.skip(f"{BEFORE} not in this clone")
    changed = git("diff", "--name-only", BEFORE, "--", "rust/engine").stdout.split()
    assert not [c for c in changed if re.match(r"rust/engine/[a-z_]+/src/", c) or c.endswith("Cargo.toml") or c.endswith("Cargo.lock")], changed
    assert git("diff", "--quiet", BEFORE, "--", "rules", ":(exclude)rules/lemmas", ":(exclude)rules/placed", "catspace/incubator/analyst/discovery/gdl.py").returncode == 0
    print(f"OBS 79.53: files under rust/engine that differ from {BEFORE}: {len(changed)}, none a source of a crate nor a manifest")
