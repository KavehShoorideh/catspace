"""PLAN 79.54 (SYSTEM_SPEC 11, last block, (ix) "answers carried from the last state"; piece: rules; ledger rows 55, 45): THE COUNT AND THE PASS `carry`.
  (1) the pass is in the list, comes after the others, brings `cones` (and so `keep`); a module made without it has none of its items and a module made with it has them; the interface is written in the head of the translator and in translated_common/mod.rs
      before the bodies, and the old entry points are kept;
  (2) what the pass finds: for each piece, the fluents its cone reads, from the dependency graph of the clauses as written, on a small planted game that is not chess (the masks are the fluents the clauses name, reached through other relations);
  (3) a small planted game, compiled with rustc here, answered by the module with the pass over states asked ALONG a chain that changes one fact at a time (a change inside and outside what a piece reads), across siblings and at jumps, as the evaluator and the module without the pass answer;
  (4) the kept modules: for each of the seven games `<game>_carry.rs` is made with every pass, `<game>_fast.rs` with every pass but `carry` (the other kept modules are made with no `carry`), both with the hash of the translator now;
  (5) the Rust tests of the plan (translated_carry.rs: the module with the pass equals the module without it at every state along games, across siblings and at jumps, every relation of the kept evaluation held to the rows made again, seven games; translated_carry_count.rs
      runs and prints the count of part A) pass; the clock file exists and its short run passes; the gate list names this file after plan 79.53's;
  (6) no source of a crate of the engine, no manifest, no description and no reader differs from the commit before the plan.
This file names a game (the planted small one); a test file may."""
from __future__ import annotations

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
BEFORE = "30f5ec44"
GAMES = ("chess", "atomic", "3check", "kingofthehill", "antichess", "chess960", "horde")
needs_cargo = pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo (offline) to run the Rust tests of the plan")
needs_rustc = pytest.mark.skipif(shutil.which("rustc") is None, reason="needs rustc to compile the planted small game")


def translator():
    from catspace.incubator.analyst.discovery import gdl_to_rust as T
    return T


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


# a small game that is not chess: a piece moves along a ring of six places held back by blocks; two relations read different fluents
SMALL = """(sort spot n0 n1 n2 n3 n4 n5)
(sort who p q)
(sort score 0 100)
(sort move (go spot) noop)
(signature spot spot) (signature nxt spot spot) (signature reach spot spot) (signature piece spot) (signature block spot) (signature control who) (signature occupied spot) (signature free spot)
(signature trapped spot) (signature opponent who who) (signature role who) (signature does who move) (signature legal who move) (signature terminal) (signature goal who score)
(signature roomy spot) (signature mine spot)

(role p) (role q) (opponent p q) (opponent q p)
(spot n0) (spot n1) (spot n2) (spot n3) (spot n4) (spot n5)
(nxt n0 n1) (nxt n1 n2) (nxt n2 n3) (nxt n3 n4) (nxt n4 n5)
(init (piece n0)) (init (block n3)) (init (control p))
(<= (occupied ?n) (true (block ?n)))
(<= (free ?n) (spot ?n) (not (occupied ?n)))
(<= (roomy ?n) (free ?n) (nxt ?n ?m) (free ?m))
(<= (reach ?a ?b) (nxt ?a ?b))
(<= (reach ?a ?c) (reach ?a ?b) (free ?b) (nxt ?b ?c))
(<= (mine ?a) (true (piece ?a)) (true (control p)))
(<= (trapped ?a) (true (piece ?a)) (nxt ?a ?b) (occupied ?b))
(<= (legal ?w (go ?b)) (true (control ?w)) (true (piece ?a)) (reach ?a ?b) (free ?b))
(<= (legal ?o noop) (true (control ?w)) (opponent ?w ?o))
(<= (next (piece ?c)) (does ?w (go ?c)))
(<= (next (block ?n)) (true (block ?n)))
(<= (next (control ?o)) (true (control ?w)) (opponent ?w ?o))
(<= terminal (trapped ?a))
(<= terminal (mine ?a) (not (roomy ?a)))
(<= (goal ?w 100) (role ?w) (true (control ?w)))
(<= (goal ?w 0) (role ?w) (not (true (control ?w))))
"""


def small_states(n: int, seed: int, chain: bool) -> list:
    """States of the small game: a chain that changes one fact at a time (a move of the piece, a block put or taken, the side) when `chain`, else unrelated random states."""
    rng = random.Random(seed)
    cells = [f"n{i}" for i in range(6)]
    piece, control, blocks = "n0", "p", {"n3"}
    out = []
    for _ in range(n):
        if chain:
            k = rng.randrange(4)
            if k == 0:
                piece = rng.choice(cells)
            elif k == 1:
                blocks ^= {rng.choice(cells)}
            elif k == 2:
                control = "q" if control == "p" else "p"
            else:
                piece = rng.choice(cells)
                control = "q" if control == "p" else "p"
        else:
            piece, control, blocks = rng.choice(cells), rng.choice("pq"), set(rng.sample(cells, rng.randint(0, 3)))
        out.append([("piece", (piece,)), ("control", (control,))] + [("block", (b,)) for b in sorted(blocks)])
    return out


def pieces_and_masks(src: str):
    names = re.findall(r'\("([^"]+)", (?:true|false)\)', re.search(r"pub const PIECES.*", src).group(0))
    fluents = re.findall(r'"([^"]+)"', re.search(r"pub const FLUENTS: \[&str; \d+\] = \[(.*)\];", src).group(1))
    masks = [int(x) for x in re.search(r"const PIECE_FL: \[u32; \d+\] = \[(.*)\];", src).group(1).split(",")]
    assert len(names) == len(masks)
    return {n: {fluents[i] for i in range(len(fluents)) if m >> i & 1} for n, m in zip(names, masks)}, fluents


def test_the_pass_ends_the_list_brings_cones_and_a_module_without_it_has_none_of_its_items():
    T = translator()
    assert T.PASSES[-1] == "carry" and T.PASSES[-2] == "topdown"
    p = T.Plan(T.Description(SMALL), ["carry"])
    assert "cones" in p.passes and "keep" in p.passes and p.carry
    with_pass = T.translate(SMALL, "g", list(T.PASSES))
    without = T.translate(SMALL, "g", [x for x in T.PASSES if x != "carry"])
    for item in ("PIECE_FL", "fn clear_piece", "fn fluent_diff", "fn run_carried", "cvalid"):
        assert item in with_pass, item
        assert item not in without, item
    assert "fn carried_relations" in with_pass and "fn carried_relations" in without     # the tests' item exists in both, to hold one to the other
    assert "// passes: " in with_pass and with_pass.split("// passes: ")[1].split("\n")[0].endswith("topdown,carry")
    print("OBS 79.54 B the pass carry: in the list after topdown, brings cones and keep; the module without it has no PIECE_FL, clear_piece, fluent_diff, run_carried or cvalid")


def test_what_the_pass_finds_is_the_fluents_each_cone_reads_in_a_planted_game():
    T = translator()
    src = T.translate(SMALL, "g", list(T.PASSES))
    masks, fluents = pieces_and_masks(src)
    assert sorted(fluents) == ["block", "control", "piece"]
    found = {n: m for n, m in masks.items() if m}
    # a piece that reads the blocks only, through the relations it reads; one that reads the piece and the blocks; the legal rows read all three
    assert masks.get("legal") == {"block", "control", "piece"}, masks
    assert masks.get("terminal") == {"block", "control", "piece"} or masks.get("terminal") == {"block", "piece", "control"}, masks
    assert masks.get("trapped") in (None, {"block", "piece"}), masks
    assert masks.get("free") in (None, {"block"}), masks
    # the walk `reach` is made for the keys its readers ask (pass asked): the guards of its readers are in its written clauses, so its cone reads what they read (the piece and the side), not only the blocks
    assert masks.get("reach") in (None, {"block"}, {"block", "control", "piece"}), masks
    assert any(m == {"block"} for m in found.values()), found
    # the true.* pieces read their own fluent
    for f in fluents:
        assert masks.get(f"true.{f}") == {f}, (f, masks)
    print(f"OBS 79.54 B planted game, fluents read by the pieces: " + "; ".join(f"{n}: {sorted(m)}" for n, m in sorted(found.items())))


DRIVER = r'''
mod game { include!("game.rs"); }
mod plain { include!("plain.rs"); }
use std::io::Read;
fn facts_of(line: &str) -> Vec<(String, Vec<String>)> { line.split(';').filter(|x| !x.trim().is_empty()).map(|f| { let mut it = f.split_whitespace(); (it.next().unwrap().to_string(), it.map(|x| x.to_string()).collect()) }).collect() }
fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let mut c = game::game::Game::new();
    let mut p = plain::plain::Game::new();
    for line in text.lines() {
        let facts = facts_of(line);
        let sc = c.state_of_facts(&facts).unwrap();
        let sp = p.state_of_facts(&facts).unwrap();
        let mut b = vec![];
        c.legal_moves_into(&sc, &mut b);
        let mut bp = vec![];
        p.legal_moves_into(&sp, &mut bp);
        let names = |g: &dyn Fn(u32) -> String, v: &[u32]| { let mut x: Vec<String> = v.iter().map(|m| g(*m)).collect(); x.sort(); x.join(",") };
        let same = names(&|m| c.render(m), &b) == names(&|m| p.render(m), &bp) && c.terminal(&sc) == p.terminal(&sp) && c.mover(&sc) == p.mover(&sp);
        println!("{} {} {} {} {}", c.terminal(&sc), !b.is_empty(), same, c.carried_valid(), names(&|m| c.render(m), &b).len());
    }
}
'''


def run_small(tmp_path, lines: list) -> list:
    T = translator()
    (tmp_path / "game.rs").write_text(T.translate(SMALL, "game", list(T.PASSES)))
    (tmp_path / "plain.rs").write_text(T.translate(SMALL, "plain", [x for x in T.PASSES if x != "carry"]))
    (tmp_path / "main.rs").write_text(DRIVER)
    exe = tmp_path / "small"
    r = subprocess.run(["rustc", "--edition", "2021", "-O", "-A", "warnings", "-o", str(exe), str(tmp_path / "main.rs")], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    r = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.splitlines()


def reference(sts) -> list:
    from catspace.incubator.analyst.discovery import gdl as G
    prog = G.Program(SMALL)
    out = []
    for facts in sts:
        db = prog.run([(f"true.{f}", a) for f, a in facts], does=())
        term = bool(db.facts("terminal"))
        txt = lambda x: G.term_text(x) if not isinstance(x, str) else x
        legal = [row for row in db.facts("legal") if txt(row[1]) != "noop"]
        out.append((term, bool(legal)))
    return out


@needs_rustc
def test_a_planted_game_is_answered_along_a_chain_of_one_fact_changes_and_at_jumps_as_the_evaluator_and_the_module_without_the_pass_answer(tmp_path):
    chain, jumps = small_states(400, 54, True), small_states(300, 55, False)
    sts = chain + jumps + chain[::-1] + chain[::7]          # along, jumps, the same chain backwards (every state a change from the last), and a stride of the chain (changes of several facts)
    lines = [";".join(f"{f} {' '.join(a)}" for f, a in facts) for facts in sts]
    got = run_small(tmp_path, lines)
    want = reference(sts)
    assert len(got) == len(sts)
    ended = valid = 0
    for line, facts, (term, turn) in zip(got, sts, want):
        t, has, same, nvalid, _ = line.split()
        assert same == "true", (facts, line)
        assert (t == "true", has == "true") == (term, turn), (facts, line, term, turn)
        ended += term
        valid += int(nvalid)
    assert 20 < ended < len(sts) - 20
    print(f"OBS 79.54 B planted game: {len(sts)} states (a chain of one-fact changes, jumps, the chain backwards, a stride), {ended} where the end holds; the module with the pass equals the evaluator's end and legal turn and the module without the pass in every one; mean valid pieces {valid / len(sts):.1f}")


@pytest.mark.parametrize("game", GAMES)
def test_the_kept_modules_with_and_without_the_pass_carry_the_hash_of_the_translator_now(game):
    T = translator()
    now = T.translator_hash()
    carry, fast = (GEN / f"{game}_carry.rs").read_text(), (GEN / f"{game}_fast.rs").read_text()
    for src in (carry, fast, (GEN / f"{game}.rs").read_text()):
        assert f"// translator-sha256: {now}" in src
    assert re.search(r"// passes: .*topdown,carry\n", carry)
    assert "carry" not in re.search(r"// passes: (.*)\n", fast).group(1)
    assert "PIECE_FL" in carry and "PIECE_FL" not in fast and "cvalid" in carry
    masks, fluents = pieces_and_masks(carry)
    assert any(masks.values()) and all(masks[n] == {n[len("true."):]} for n in masks if n.startswith("true.") and masks[n])
    print(f"OBS 79.54 B {game}: kept _carry module with {sum(1 for m in masks.values() if m)} pieces that read a fluent of {len(fluents)}; the _fast module has no pass carry")


def test_the_lemma_module_and_the_placed_modules_are_made_without_the_pass():
    for f in [GEN / "chess_lemma.rs", *sorted((ENGINE / "game" / "tests" / "generated_placed").glob("*.rs"))]:
        m = re.search(r"// passes: (.*)\n", f.read_text())
        assert m and "carry" not in m.group(1), f.name


def test_the_interface_is_written_before_its_bodies_and_the_old_entry_points_stay():
    mod = (ENGINE / "game" / "tests" / "translated_common" / "mod.rs").read_text()
    for item in ("fn moves_t", "fn next_t", "fn facts_of_t", "fn moves_into_t", "fn terminal_td_t", "fn carried_t", "fn carried_valid_t", "($m:ident, carry)"):
        assert item in mod, item
    head = translator().__doc__
    assert "PLAN 79.54 PART B" in head and '(25) "carry"' in head and "B1, BY PIECE" in head and "B2, BY GROUP (NOT BUILT" in head and "SOUNDNESS" in head
    assert "chess_the_times_of_the_carry" in (ENGINE / "game" / "tests" / "translated_carry_speed.rs").read_text()
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert gate.index("test_plan79_54") > gate.index("test_plan79_53")


@needs_cargo
def test_the_rust_tests_of_the_plan_pass():
    env = {**os.environ, "CARGO_BUILD_JOBS": "2", "RUST_TEST_THREADS": "2"}
    r = subprocess.run(["cargo", "test", "--release", "--offline", "-j", "2", "-p", "cs_game", "--test", "translated_carry", "--", "--test-threads=2", "--nocapture"], cwd=ENGINE, env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    held = [ln for ln in out.splitlines() if "OBS 79.54 B" in ln and "carry equals no-carry" in ln]
    assert len(held) == 7 and all(", 0 differ" in ln for ln in held), held
    for ln in held:
        print(ln[ln.index("OBS"):][:260])
    r = subprocess.run(["cargo", "test", "--release", "--offline", "-j", "2", "-p", "cs_game", "--test", "translated_carry_count", "--", "--test-threads=2", "--nocapture"], cwd=ENGINE, env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert len([ln for ln in out.splitlines() if "OBS 79.54 A" in ln and "successor ceiling" in ln]) == 14      # successor and second successor, seven games
    r = subprocess.run(["cargo", "test", "--release", "--offline", "-j", "2", "-p", "cs_game", "--test", "translated_carry_speed", "--", "--test-threads=1"], cwd=ENGINE, env=env, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]


def test_no_source_of_the_engine_no_cargo_file_no_description_and_no_reader_differs_from_the_commit_before_plan_79_54():
    if git("cat-file", "-e", f"{BEFORE}^{{commit}}").returncode != 0:
        pytest.skip(f"{BEFORE} not in this clone")
    changed = git("diff", "--name-only", BEFORE, "--", "rust/engine").stdout.split()
    assert not [c for c in changed if re.match(r"rust/engine/[a-z_]+/src/", c) or c.endswith("Cargo.toml") or c.endswith("Cargo.lock")], changed
    assert git("diff", "--quiet", BEFORE, "--", "rules", ":(exclude)rules/lemmas", ":(exclude)rules/placed", "catspace/incubator/analyst/discovery/gdl.py").returncode == 0
    print(f"OBS 79.54: files under rust/engine that differ from {BEFORE}: {len(changed)}, none a source of a crate nor a manifest")
