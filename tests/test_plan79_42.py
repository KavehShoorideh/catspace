"""PLAN 79.42 (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ... made fast piece by piece under tests"; ledger row 45; piece: rules): the gate of the passes start, orient and ends (the pass lines is not built: the module text says which piece is missing).
  (1) the passes exist, the file with every pass of each game names them and is the kept file, and the files of the single passes are made here, not kept;
  (2) the passes decide from shapes: the translator's source names no relation, sort or constant of any description under rules/; the same decisions come from a description with every name replaced;
  (3) what each pass recognises and refuses, on a small description written here that is not a chess, each refusal naming its clause: the pass start writes a clause in two orders and a chooser, and refuses an order that asks an orientation no clause asks;
      the pass ends takes a walk whose readers entail a stop and refuses, naming the reader, a walk read by a clause that does not, a walk read negated, a walk whose stop literals are all positive; the pass orient is shown in the text of the module;
  (4) the doubling form of the pass ends and the lemma of the module text's part D, checked on random tables (the lemma with its conditions holds, and fails when a condition is left out: the negative control of the argument);
  (5) a small game that is not a chess, translated with each pass and compiled with rustc here, answers every relation and the moves as the reader's evaluator does on every state of a list;
  (6) the gate lists this file after the one of 79.41.
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
NEW = ("start", "orient", "ends")


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


def t41():
    return sibling("test_plan79_41")


# the small game of 79.41, with the stops of its two walks written as negations and its readers of them changed: the walk upward is read by a clause that holds only at a place where the walk stops (a unit is there), the walk downward by one that does not
WALK = t41().WALK
WALK2 = (WALK
         .replace("(<= (ray_up ?a ?c) (ray_up ?a ?b) (free ?b) (up ?b ?c))", "(<= (ray_up ?a ?c) (ray_up ?a ?b) (not (occupied ?b)) (up ?b ?c))")
         .replace("(<= (ray_down ?a ?c) (ray_down ?a ?b) (free ?b) (down ?b ?c))", "(<= (ray_down ?a ?c) (ray_down ?a ?b) (not (occupied ?b)) (down ?b ?c))")
         .replace("(<= (legal ?w (go ?b)) (true (control ?w)) (true (piece ?a)) (ray_up ?a ?b) (free ?b) (see ?b))", "(<= (legal ?w (go ?b)) (true (control ?w)) (true (piece ?a)) (ray_up ?a ?b) (true (block ?b)))"))
assert WALK2 != WALK and "(not (occupied ?b)) (up ?b ?c)" in WALK2 and "(ray_up ?a ?b) (true (block ?b))" in WALK2


def plan(passes, text=WALK2):
    T = translator()
    return T.Plan(T.Description(text), passes)


# ------------------------------------------------------------------------------------------------------------------------------------------ (1)

def test_the_new_passes_are_named_and_the_every_pass_file_of_each_game_names_them():
    T = translator()
    assert set(NEW) <= set(T.PASSES)
    for game in GAMES:
        head = t40().head_line((GEN / f"{game}_fast.rs").read_text(), "passes")
        assert head.split(",") == [p for p in T.PASSES if p != "carry"], game
    print(f"OBS 79.42 F passes {','.join(T.PASSES)}: the kept file of each game names every one")


def test_ends_brings_bits_and_shift_with_it():
    p = plan(["ends"])
    assert "bits" in p.passes and "shift" in p.passes and "ends" in p.passes


# ------------------------------------------------------------------------------------------------------------------------------------------ (2)

def test_the_translator_names_no_relation_sort_or_constant_of_any_description():
    words = set(re.findall(r"[a-z_][a-z0-9_]*", TRANSLATOR.read_text().lower()))
    names = t40().all_names()
    hit = sorted(words & names)
    print(f"OBS 79.42 G scan: {len(names)} relation, sort and constant names of the descriptions under rules/, {len(words)} words in the translator's source, {len(hit)} in both")
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
        return {"orders": sorted(v for v in p.start_written.values()), "refused_orders": len(p.start_refused), "ends": len(p.ends), "ends_refused": sorted(len(v) > 0 for v in p.ends_refused.values()), "walks": len(p.bit_rec)}
    assert shape(a) == shape(b), (shape(a), shape(b))
    assert any(v[0] > 1 for v in a.start_written.values()) and a.ends
    print(f"OBS 79.42 G renamed {game}: {len(a.start_written)} clauses with a choice of order ({sum(1 for v in a.start_written.values() if v[0] > 1)} with two or more orders written, {len(a.start_refused)} orders refused), {len(a.ends)} walks made as their ends, {len(a.ends_refused)} refused; the same with every name replaced")


# ------------------------------------------------------------------------------------------------------------------------------------------ (3)

def test_the_pass_start_writes_a_clause_in_two_orders_and_a_chooser_that_runs_the_one_of_the_least_estimate():
    T = translator()
    p = plan(["bits", "start"])
    many = {k: v for k, v in p.start_written.items() if v[0] > 1}
    assert many, p.start_written
    src = T.translate(WALK2, "walk", ["bits", "start"])
    name = sorted(many)[0][0]
    assert f"fn {name}v0(" in src and f"fn {name}v1(" in src and re.search(rf"fn {name}\(s: &St, d: &mut Dy, tm: &mut Terms, dw: &mut Dw\) -> u64 {{\n    let mut best", src)
    assert "saturating_mul" in src and "match best" in src
    plain = T.translate(WALK2, "walk", ["bits"])
    assert "match best" not in plain
    print(f"OBS 79.42 H start: {len(many)} clauses written in two or more orders on the small game, e.g. {sorted(many)[0][0]} ({many[sorted(many)[0]]} written, found)")


def test_the_pass_start_refuses_an_order_that_asks_an_orientation_no_clause_asks_naming_the_clause():
    T = translator()
    text = (ROOT / "rules" / "chess.gdl").read_text()
    p = T.Plan(T.Description(text), T.PASSES)
    why = " ".join(p.start_refused.values())
    assert p.start_refused and "orientation" in why and "(<=" in why, p.start_refused
    n = len(p.start_refused)
    first = sorted(p.start_refused.items(), key=lambda kv: (kv[0][0], kv[0][1]))[0]
    print(f"OBS 79.42 H start refusals on the first game: {n} orders refused, e.g. {first[1][:150]}")
    assert n > 5


def test_the_pass_ends_takes_a_walk_whose_readers_entail_a_stop_and_refuses_the_one_with_a_reader_that_does_not_naming_it():
    p = plan(["bits", "ends"])
    assert "ray_up" in p.ends and p.ends["ray_up"][0] == "neg" and p.ends["ray_up"][1] == "occupied", p.ends
    assert "ray_down" in p.ends_refused and "does not entail a stop at occupied" in p.ends_refused["ray_down"] and "(ray_down ?a ?b) (free ?b)" in p.ends_refused["ray_down"], p.ends_refused
    print(f"OBS 79.42 H ends: taken {sorted(p.ends)}, refused {p.ends_refused['ray_down'][:130]}")


def test_the_pass_ends_refuses_a_walk_read_negated_naming_the_reader_and_a_walk_whose_stop_literals_are_positive():
    text = WALK2 + "(signature lonely cell cell)\n(<= (lonely ?a ?b) (cell ?a) (cell ?b) (not (ray_up ?a ?b)))\n"
    p = plan(["bits", "ends"], text)
    assert "ray_up" in p.ends_refused and "read negated" in p.ends_refused["ray_up"] and "(<= (lonely" in p.ends_refused["ray_up"], p.ends_refused
    p2 = plan(["bits", "ends"], WALK)
    assert not p2.ends and all("no negated stop literal" in v or "does not entail" in v or "reads" in v for v in p2.ends_refused.values()) and p2.ends_refused, p2.ends_refused
    print(f"OBS 79.42 H ends refusals: {p.ends_refused['ray_up'][:100]} | {sorted(p2.ends_refused.values())[0][:80]}")


def test_the_pass_ends_refuses_a_walk_with_a_reader_that_reads_it_with_a_term_that_is_no_variable_or_twice():
    twice = WALK2 + "(signature both cell)\n(<= (both ?b) (ray_up ?a ?b) (ray_up ?b ?c) (true (block ?b)))\n"
    p = plan(["bits", "ends"], twice)
    assert "ray_up" in p.ends_refused and "twice" in p.ends_refused["ray_up"] and "(<= (both" in p.ends_refused["ray_up"], p.ends_refused
    const = WALK2 + "(signature top cell)\n(<= (top ?a) (ray_up ?a n7) (true (block n7)))\n"
    p = plan(["bits", "ends"], const)
    assert "ray_up" in p.ends_refused and "no variable" in p.ends_refused["ray_up"], p.ends_refused


def test_the_ends_walk_is_written_with_the_doubling_form_the_start_words_apart_and_the_whole_word_when_asked():
    T = translator()
    src = T.translate(WALK2, "walk", ["bits", "ends"])
    assert "pub const ENDS: &[&str] = &[\"ray_up\"];" in src
    assert "sb_r_ray_up" in src and "shw(live" in src and "for _ in 0..3 {" in src and "if dw.full { acc" in src
    assert "self.dw.full = true;" in src and "self.dw.full = false;" in src
    plain = T.translate(WALK2, "walk", ["bits", "shift"])
    assert "shw(" not in plain and "ENDS: &[&str] = &[];" in plain
    chess = T.translate((ROOT / "rules" / "chess.gdl").read_text(), "m", ["bits", "shift", "ends"])
    assert "for _ in 0..6 {" in chess and "pub const ENDS: &[&str] = &[\"king_ray\"];" in chess


def test_the_pass_orient_sets_the_other_orientations_from_the_new_bits_of_a_word_with_the_index_made_once():
    T = translator()
    text = (ROOT / "rules" / "chess.gdl").read_text()
    on = T.translate(text, "m", ["bits", "shift", "orient"])
    off = T.translate(text, "m", ["bits", "shift"])
    assert "let (bf" in on and "p * " in on and "let (bf" not in off
    print(f"OBS 79.42 H orient: {on.count('let (bf')} orientations written by stride on the first game")


# ------------------------------------------------------------------------------------------------------------------------------------------ (4)

def closure(start: int, live: int, table: list, size: int) -> int:
    """The walk of the module text: from the frontier, the members that pass (live) go to their next members by the displacements of the table, until no new member."""
    full = (1 << size) - 1
    acc = fr = start
    while True:
        x = fr & live
        if not x:
            break
        nx = 0
        for dl, m in table:
            y = x & m
            nx |= (y << dl) if dl >= 0 else (y >> -dl)
        nx &= ~acc & full
        if not nx:
            break
        acc |= nx
        fr = nx
    return acc


def shw(x: int, n: int, size: int) -> int:
    full = (1 << size) - 1
    if n >= 0:
        return (x << n) & full if n < 64 else 0
    return x >> -n if n > -64 else 0


def doubling(start: int, live: int, dl: int, m: int, size: int) -> int:
    """The doubling form of pass (15), as the generated function writes it."""
    rounds = max(1, (size - 1).bit_length())
    q = shw(live & m, dl, size)
    acc, s = start, dl
    for _ in range(rounds):
        acc |= q & shw(acc, s, size)
        q &= shw(q, s, size)
        s *= 2
    return acc


@pytest.mark.parametrize("size", [8, 20, 64])
def test_the_doubling_form_gives_the_word_of_the_closure_on_random_tables(size):
    rng = random.Random(5 + size)
    n = 0
    for _ in range(3000):
        dl = rng.choice([1, -1, 2, -2, 3, size // 8 or 1, -(size // 8 or 1)])
        m = rng.getrandbits(size) | rng.getrandbits(size)
        live = rng.getrandbits(size)
        start = rng.getrandbits(size) & rng.getrandbits(size) & rng.getrandbits(size)
        want = closure(start, live, [(dl, m & ((1 << size) - 1))], size)
        got = doubling(start, live, dl, m & ((1 << size) - 1), size)
        assert got == want, (size, dl, bin(m), bin(live), bin(start))
        n += 1
    print(f"OBS 79.42 I doubling size {size}: {n} random tables and words, the same word as the closure in every one")


def test_the_doubling_form_with_one_round_too_few_is_caught_on_the_same_tables():
    size = 64
    rng = random.Random(9)
    bad = 0
    for _ in range(3000):
        m = (1 << size) - 1
        live = rng.getrandbits(size) | rng.getrandbits(size) | rng.getrandbits(size)
        start = 1
        want = closure(start, live, [(1, m)], size)
        q = shw(live & m, 1, size)
        acc, s = start, 1
        for _r in range(2):                       # NEGATIVE CONTROL: two rounds where six are needed
            acc |= q & shw(acc, s, size)
            q &= shw(q, s, size)
            s *= 2
        bad += acc != want
    print(f"OBS 79.42 I doubling control: two rounds instead of six differ from the closure on {bad} of 3000 random words")
    assert bad > 0


def test_the_lemma_of_part_d_holds_when_its_conditions_do_and_fails_when_one_is_left_out():
    """If S_K = S_0 and R_0 and C = 0 then the closure of S_K under live_K is R_0 (module text, (16)); each condition is needed."""
    size = 16
    rng = random.Random(3)
    checked = holds = 0
    no_start = no_change = 0
    for _ in range(20000):
        dl = rng.choice([1, -1, 3, -3])
        table = [(dl, rng.getrandbits(size))]
        s0 = rng.getrandbits(size) & rng.getrandbits(size) & rng.getrandbits(size)
        live0 = rng.getrandbits(size)
        c = rng.getrandbits(size) & rng.getrandbits(size) & rng.getrandbits(size)
        r0 = closure(s0, live0, table, size)
        livek = live0 ^ c
        if r0 & c == 0:
            checked += 1
            holds += closure(s0, livek, table, size) == r0
            # a start word that is not the base's: the conclusion may fail (control 1)
            s1 = s0 | (1 << rng.randrange(size))
            no_start += closure(s1, livek, table, size) != r0
        else:
            # R_0 meets C: the conclusion may fail (control 2)
            no_change += closure(s0, livek, table, size) != r0
    print(f"OBS 79.42 I lemma: {checked} cases with S_K = S_0 and R_0 and C = 0, the closure is R_0 in {holds}; with another start word it differs in {no_start}, and where R_0 meets C it differs in {no_change}")
    assert checked > 1000 and holds == checked and no_start > 0 and no_change > 0


# ------------------------------------------------------------------------------------------------------------------------------------------ (5)

def reference(sts) -> list:
    G = t40().reader()
    prog = G.Program(WALK2)
    d = translator().Description(WALK2)
    dynamic = [r for r in d.nodes if not d.static[r]]
    out = []
    for facts in sts:
        db = prog.run([(f"true.{f}", a) for f, a in facts], does=())
        rel = {r: sorted(" ".join(G.term_text(t) if not isinstance(t, str) else t for t in row) for row in db.facts(r)) for r in dynamic}
        legal = [tuple(G.term_text(t) if not isinstance(t, str) else t for t in row) for row in db.facts("legal")]
        terminal = bool(db.facts("terminal"))
        moves: list = []
        if not terminal:
            for role in ("p", "q"):
                mine = sorted(m for r, m in legal if r == role and m != "noop")
                if mine:
                    moves = mine
                    break
        out.append((rel, moves))
    return out


@pytest.mark.skipif(shutil.which("rustc") is None, reason="no rustc on this machine")
@pytest.mark.parametrize("passes", [["bits", "start"], ["bits", "ends"], ["bits", "orient"], ["bits", "scan", "cut", "shift", "base", "start", "ends", "orient"], list(translator().PASSES)],
                         ids=["start", "ends", "orient", "bits and the new ones", "every pass"])
def test_a_small_game_that_is_not_chess_translated_with_the_new_passes_answers_every_relation_and_the_moves_as_the_evaluator_does(passes, tmp_path):
    m = t40()
    T = translator()
    (tmp_path / "game.rs").write_text(T.translate(WALK2, "game", passes))
    (tmp_path / "main.rs").write_text(m.DRIVER)
    exe = tmp_path / "walk"
    r = subprocess.run(["rustc", "--edition", "2021", "-O", "-A", "warnings", "-o", str(exe), str(tmp_path / "main.rs")], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    sts = t41().states(80)
    text = "\n".join(";".join(f"{f} {' '.join(a)}" for f, a in facts) for facts in sts) + "\n"
    r = subprocess.run([str(exe)], input=text, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    blocks = r.stdout.split("END\n")[:-1]
    assert len(blocks) == len(sts)
    ref = reference(sts)
    compared = moves_seen = 0
    for i, (blk, (want, want_moves)) in enumerate(zip(blocks, ref)):
        rows = {}
        got_moves = None
        for line in blk.splitlines():
            parts = line.split("\t")
            if parts[0] == "MOVES":
                got_moves = sorted(re.findall(r"\([^()]*\)|\S+", parts[1])) if len(parts) > 1 and parts[1] else []
            elif parts[0] == "REL":
                rows.setdefault(parts[1], [])
            elif parts[0] == "ROW":
                rows.setdefault(parts[1], []).append(parts[2])
        joined = got_moves
        assert joined == sorted(want_moves), (i, passes, joined, want_moves, sts[i])
        moves_seen += len(joined)
        assert rows, "the module made no relation"
        for name in rows:
            assert sorted(set(rows[name])) == want[name], (i, passes, name, sorted(rows[name]), want[name], sts[i])
            compared += 1
    assert compared > 60 * 5 and moves_seen > 20
    print(f"OBS 79.42 J walk {passes}: {len(sts)} states, {compared} relations and {moves_seen} moves compared with the evaluator's, 0 disagreements")


# ------------------------------------------------------------------------------------------------------------------------------------------ (6)

def test_the_gate_lists_this_test_after_the_one_of_plan_79_41():
    text = (ROOT / "scripts" / "gate.py").read_text()
    assert "test_plan79_42" in text and text.index("test_plan79_41") < text.index("test_plan79_42")
