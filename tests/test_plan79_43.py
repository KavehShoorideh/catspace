"""PLAN 79.43 (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ... made fast piece by piece under tests"; ledger row 45; piece: rules): the gate of the passes asked, project and worth.
  (1) the passes exist, the file with every pass of each game names them and is the kept file, and the files of the single passes are made here, not kept;
  (2) the passes decide from shapes: the translator's source names no relation, sort or constant of any description under rules/; the same decisions come from a description with every name replaced;
  (3) what each pass recognises and refuses, on small descriptions written here that are not a chess, each refusal naming its clause or its reader: the pass asked restricts a walk to the keys its readers ask, and refuses a walk read negated, a reader that asks it unbound or with one variable at two
      places of the key, a recursive clause that changes the key, and a walk whose base clause already holds the prefix of its reader; the pass project reads a literal from the projection of its relation where the places dropped are used nowhere else and the projection is made before the clause in every
      order of the strata, and leaves a literal alone where it is not; the pass worth makes the orientation of a relation that does not read the state and refuses that of one that does, naming it;
  (4) the argument of the pass asked on random tables (a walk that keeps its key, begun only at the keys asked, has the rows of the whole walk at those keys and no other; with the condition that the key is kept left out it does not: the negative control of the argument);
  (5) a small game that is not a chess, translated with each pass and compiled with rustc here, answers every relation and the moves as the reader's evaluator does on every state of a list;
  (6) the gate lists this file after the one of 79.42.
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
NEW = ("asked", "project", "worth")


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


def t42():
    return sibling("test_plan79_42")


# the small game of 79.42 (its walk upward read by a clause that holds only where the walk stops), with a fluent of two places, its projection, and a clause that reads the fluent where the projection will do
WALK3 = t42().WALK2 + """(sort tag red blue)
(signature mark cell tag) (signature marked cell) (signature see2 cell) (signature rd cell cell) (signature reach cell cell) (signature rw cell cell) (signature use cell) (signature anyb cell)
(init (mark n1 red))
(<= (next (mark ?n ?t)) (true (mark ?n ?t)))
(<= (marked ?n) (true (mark ?n ?t)))
(<= (see2 ?n) (marked ?n) (cell ?n))
(<= (see2 ?n) (true (mark ?n ?t)) (free ?n))
"""
assert "(see2 ?n) (true (mark ?n ?t)) (free ?n)" in WALK3


def plan(passes, text=WALK3):
    T = translator()
    return T.Plan(T.Description(text), passes)


# ------------------------------------------------------------------------------------------------------------------------------------------ (1)

def test_the_new_passes_are_named_and_the_every_pass_file_of_each_game_names_them():
    T = translator()
    assert set(NEW) <= set(T.PASSES)
    for game in GAMES:
        head = t40().head_line((GEN / f"{game}_fast.rs").read_text(), "passes")
        assert head.split(",") == list(T.PASSES), game
        assert "pub const ASKED: &[&str] = &[\"slide\"];" in (GEN / f"{game}_fast.rs").read_text()
    print(f"OBS 79.43 F passes {','.join(T.PASSES)}: the kept file of each game names every one")


def test_asked_and_project_and_worth_bring_what_they_read_with_them():
    for name in NEW:
        p = plan([name])
        assert "bits" in p.passes, name
    assert "start" in plan(["worth"]).passes


# ------------------------------------------------------------------------------------------------------------------------------------------ (2)

def test_the_translator_names_no_relation_sort_or_constant_of_any_description():
    words = set(re.findall(r"[a-z_][a-z0-9_]*", TRANSLATOR.read_text().lower()))
    names = t40().all_names()
    hit = sorted(words & names)
    print(f"OBS 79.43 G scan: {len(names)} relation, sort and constant names of the descriptions under rules/, {len(words)} words in the translator's source, {len(hit)} in both")
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
        return {"asked": len(p.asked), "asked_refused": len(p.asked_refused), "guarded": sum(len(v["guarded"]) for v in p.asked.values()), "projected": sorted(len(v) for _t, v in p.projected), "project_refused": len(p.project_refused),
                "worth": sorted(len(v) for v in p.worth_taken.values()), "refused_orders": len(p.start_refused), "orders": sorted(p.start_written.values())}
    assert shape(a) == shape(b), (shape(a), shape(b))
    assert a.asked and a.projected and a.worth_taken
    print(f"OBS 79.43 G renamed {game}: {len(a.asked)} walks made for the keys asked (refused {len(a.asked_refused)}), {len(a.projected)} clauses with a projection read ({len(a.project_refused)} left as written), {len(a.worth_taken)} orders admitted with an orientation of a relation that does not read the state; the same with every name replaced")


# ------------------------------------------------------------------------------------------------------------------------------------------ (3)

def test_the_pass_asked_makes_a_walk_for_the_keys_its_readers_ask_and_keeps_the_base_clause_for_when_everything_is_asked_for():
    T = translator()
    p = plan(["bits", "shift", "asked"])
    assert set(p.asked) == {"ray_up", "ray_down"}, (p.asked.keys(), p.asked_refused)
    k = p.asked["ray_up"]["k"]
    texts = [x.clause.text.replace("\n", " ") for x in p.rules[k]]
    assert any("(asked by:" in t and "(true (piece" in t.replace("true.piece", "(true (piece") or "true.piece" in str(x.clause.body) for t, x in zip(texts, p.rules[k]) if "(asked by:" in t)
    src = T.translate(WALK3, "walk", ["bits", "shift", "asked"])
    assert "pub const ASKED: &[&str] = &[\"ray_down\", \"ray_up\"];" in src and "pub full: bool" in src
    assert "if dw.full { made += " in src and "if !dw.full { made += " in src and "self.dw.full = true;" in src
    plain = T.translate(WALK3, "walk", ["bits", "shift"])
    assert "if !dw.full" not in plain and "ASKED: &[&str] = &[];" in plain
    print(f"OBS 79.43 H asked: taken {sorted(p.asked)}, the clauses of the walk {len(texts)} (the base clause as it was, with the prefix of the reader, the recursive clause)")


def test_the_pass_asked_refuses_a_walk_read_negated_naming_the_reader():
    text = WALK3 + "(<= (rd ?a ?b) (cell ?a) (cell ?b) (not (ray_up ?a ?b)))\n"
    p = plan(["bits", "shift", "asked"], text)
    assert "ray_up" in p.asked_refused and "negated" in p.asked_refused["ray_up"] and "(<= (rd" in p.asked_refused["ray_up"], p.asked_refused
    assert "ray_up" not in p.asked and "ray_down" in p.asked


def test_the_pass_asked_refuses_a_reader_that_asks_the_walk_unbound_or_with_one_variable_at_two_places_of_its_key():
    unbound = WALK3 + "(<= (anyb ?b) (ray_down ?a ?b))\n"
    p = plan(["bits", "shift", "asked"], unbound)
    assert "ray_down" in p.asked_refused and "asks it unbound" in p.asked_refused["ray_down"] and "(<= (anyb" in p.asked_refused["ray_down"], p.asked_refused
    twice = WALK3 + "(signature rk cell cell cell) (signature rd2 cell)\n(<= (rk ?a ?b ?c) (up ?a ?b) (up ?b ?c))\n(<= (rk ?a ?b ?d) (rk ?a ?b ?c) (free ?c) (up ?c ?d))\n(<= (rd2 ?a) (true (piece ?a)) (rk ?a ?a ?c))\n"
    p = plan(["bits", "shift", "asked"], twice)
    assert "rk" in p.asked_refused and "(<= (rd2" in p.asked_refused["rk"] and "one variable twice" in p.asked_refused["rk"] and "rk" not in p.asked, (p.asked_refused, p.bits_refused.get("rk"))


def test_the_pass_asked_refuses_a_walk_whose_recursive_clause_changes_the_key():
    text = WALK3 + "(<= (reach ?a ?b) (up ?a ?b) (free ?b))\n(<= (reach ?x ?c) (reach ?a ?b) (up ?x ?a) (up ?b ?c))\n(<= (rd ?a ?c) (true (piece ?a)) (reach ?a ?c))\n"
    p = plan(["bits", "shift", "asked"], text)
    assert "reach" in p.asked_refused and "does not keep the key" in p.asked_refused["reach"] and "(<= (reach ?x ?c)" in p.asked_refused["reach"], p.asked_refused
    assert "reach" not in p.asked


def test_the_pass_asked_leaves_a_walk_whose_base_clause_holds_the_prefix_of_every_reader_as_it_is_naming_the_reader():
    text = WALK3 + "(<= (rw ?a ?b) (true (piece ?a)) (up ?a ?b))\n(<= (rw ?a ?c) (rw ?a ?b) (free ?b) (up ?b ?c))\n(<= (use ?b) (true (piece ?a)) (rw ?a ?b))\n"
    p = plan(["bits", "shift", "asked"], text)
    assert "rw" in p.asked_refused and "already holds the prefix" in p.asked_refused["rw"] and "(<= (use" in p.asked_refused["rw"], p.asked_refused
    assert "rw" not in p.asked


def test_the_pass_project_reads_a_literal_from_the_projection_where_the_places_dropped_are_used_nowhere_else():
    T = translator()
    p = plan(["bits", "project"])
    assert any(rep == [("true.mark", "marked")] and "(see2 ?n) (true (mark ?n ?t)) (free ?n)" in text for text, rep in p.projected), p.projected
    assert not any("(marked ?n)" in text and "true (mark" in text for text, _r in p.projected if "(<= (marked" in text)         # the clause that makes the projection is not rewritten
    src = T.translate(WALK3, "walk", ["bits", "project"])
    off = T.translate(WALK3, "walk", ["bits"])
    assert src.count("d.r_marked.o") > off.count("d.r_marked.o") and "(with a projection read)" in src and "(with a projection read)" not in off
    print(f"OBS 79.43 H project: {len(p.projected)} clauses read a projection on the small game: {[(rep) for _t, rep in p.projected]}")


def test_the_pass_project_leaves_a_literal_as_written_where_the_projection_is_not_made_before_the_clause_in_every_order_naming_it():
    text = WALK3 + "(<= (use ?n) (true (mark ?n ?t)) (cell ?n))\n"
    p = plan(["bits", "project"], text)
    assert any("(<= (use ?n)" in t and "not made before this clause in every order" in why for t, why in p.project_refused.items()), p.project_refused
    assert not any("(<= (use ?n)" in t for t, _r in p.projected)


def test_the_pass_project_does_not_drop_a_variable_that_another_literal_uses():
    text = WALK3 + "(<= (see2 ?n) (true (mark ?n ?t)) (free ?n) (cell ?n) (true (mark ?n ?t)))\n"
    p = plan(["bits", "project"], text)
    assert not any("(true (mark ?n ?t)) (free ?n) (cell ?n) (true (mark ?n ?t))" in t for t, _r in p.projected), p.projected


def test_the_pass_worth_makes_the_orientation_of_a_relation_that_does_not_read_the_state_and_refuses_that_of_one_that_does_naming_it():
    text = (ROOT / "rules" / "chess.gdl").read_text()
    T = translator()
    p = T.Plan(T.Description(text), T.PASSES)
    assert p.worth_taken and all(all(T.Description(text).static[r] for r, _f in v) for v in p.worth_taken.values()), p.worth_taken
    why = " ".join(p.start_refused.values())
    assert "a relation that reads the state: each row set would set it too" in why and "(<=" in why, p.start_refused
    off = T.Plan(T.Description(text), [x for x in T.PASSES if x != "worth"])
    assert len(off.start_refused) > len(p.start_refused) and not off.worth_taken
    print(f"OBS 79.43 H worth on the first game: {len(p.worth_taken)} orders admitted with an orientation of a relation that does not read the state, {len(p.start_refused)} refused (without the pass {len(off.start_refused)})")


# ------------------------------------------------------------------------------------------------------------------------------------------ (4)

def closure_by_key(table: dict, keys: list, start: dict, live: dict) -> dict:
    """The walk of the module text for each key: from `start[key]`, to the members that pass (live), along the table of the key."""
    out = {}
    for k in keys:
        acc = set(start.get(k, ()))
        todo = list(acc)
        while todo:
            x = todo.pop()
            if x not in live[k]:
                continue
            for y in table.get((k, x), ()):
                if y not in acc:
                    acc.add(y)
                    todo.append(y)
        out[k] = acc
    return out


def test_a_walk_that_keeps_its_key_begun_only_at_the_keys_asked_has_the_rows_of_the_whole_walk_at_those_keys():
    """The argument of the pass asked (module text, (17)): the rows of a key are those derived from the base rows of that key; so the base rows of the keys asked give the rows of those keys and no others."""
    rng = random.Random(43)
    n, nk = 12, 6
    checked = same = 0
    for _ in range(3000):
        keys = list(range(nk))
        table = {(k, x): rng.sample(range(n), rng.randint(0, 2)) for k in keys for x in range(n)}
        live = {k: set(rng.sample(range(n), rng.randint(3, n))) for k in keys}
        start = {k: set(rng.sample(range(n), rng.randint(1, 3))) for k in keys}
        asked = set(rng.sample(keys, rng.randint(1, nk - 1)))
        whole = closure_by_key(table, keys, start, live)
        part = closure_by_key(table, keys, {k: v for k, v in start.items() if k in asked}, live)
        checked += 1
        same += all(part[k] == whole[k] for k in asked) and all(not part[k] for k in keys if k not in asked)
    print(f"OBS 79.43 I asked lemma: {checked} random tables, keys asked at random, the walk begun only at the keys asked has the rows of the whole walk at those keys and none at the others in {same}")
    assert same == checked


def test_the_argument_fails_when_the_recursive_clause_does_not_keep_the_key_the_negative_control():
    rng = random.Random(44)
    n, nk = 8, 4
    bad = 0
    for _ in range(2000):
        # a walk over pairs (key, member) whose step may move to another key: the key is not kept
        step = {(k, x): [(rng.randrange(nk), rng.randrange(n)) for _ in range(rng.randint(0, 2))] for k in range(nk) for x in range(n)}
        start = {k: {rng.randrange(n)} for k in range(nk)}

        def run(keys):
            acc = {(k, x) for k in keys for x in start[k]}
            todo = list(acc)
            while todo:
                p_ = todo.pop()
                for q in step[p_]:
                    if q not in acc:
                        acc.add(q)
                        todo.append(q)
            return acc
        asked = set(rng.sample(range(nk), 2))
        whole = run(range(nk))
        part = run(asked)
        bad += any(part_k != {m for m in whole if m[0] == k} for k in asked for part_k in [{m for m in part if m[0] == k}])
    print(f"OBS 79.43 I asked control: with a recursive clause that moves the key, the walk begun at the keys asked differs from the whole walk at those keys in {bad} of 2000 random tables")
    assert bad > 0


# ------------------------------------------------------------------------------------------------------------------------------------------ (5)

def states(n: int) -> list:
    rng = random.Random(43)
    cells = [f"n{i}" for i in range(8)]
    out = []
    for _ in range(n):
        p = rng.choice(cells)
        blocks = rng.sample(cells, rng.randint(0, 5))
        marks = [(c, rng.choice(["red", "blue"])) for c in rng.sample(cells, rng.randint(0, 4))]
        facts = [("piece", (p,)), ("control", (rng.choice("pq"),))] + [("block", (b,)) for b in blocks] + [("mark", m) for m in marks]
        out.append(facts)
    return out


def reference(sts) -> list:
    G = t40().reader()
    prog = G.Program(WALK3)
    d = translator().Description(WALK3)
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
@pytest.mark.parametrize("passes", [["bits", "asked"], ["bits", "shift", "asked"], ["bits", "project"], ["bits", "start", "worth"], ["bits", "scan", "cut", "shift", "base", "start", "ends", "orient", "asked", "project", "worth"], list(translator().PASSES)],
                         ids=["asked", "asked with shift", "project", "worth", "bits and the new ones", "every pass"])
def test_a_small_game_that_is_not_chess_translated_with_the_new_passes_answers_every_relation_and_the_moves_as_the_evaluator_does(passes, tmp_path):
    m = t40()
    T = translator()
    (tmp_path / "game.rs").write_text(T.translate(WALK3, "game", passes))
    (tmp_path / "main.rs").write_text(m.DRIVER)
    exe = tmp_path / "walk"
    r = subprocess.run(["rustc", "--edition", "2021", "-O", "-A", "warnings", "-o", str(exe), str(tmp_path / "main.rs")], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    sts = states(80)
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
        assert got_moves == sorted(want_moves), (i, passes, got_moves, want_moves, sts[i])
        moves_seen += len(got_moves)
        assert rows, "the module made no relation"
        for name in rows:
            assert sorted(set(rows[name])) == want[name], (i, passes, name, sorted(rows[name]), want[name], sts[i])
            compared += 1
    assert compared > 60 * 5 and moves_seen > 20
    print(f"OBS 79.43 J walk {passes}: {len(sts)} states, {compared} relations and {moves_seen} moves compared with the evaluator's, 0 disagreements")


# ------------------------------------------------------------------------------------------------------------------------------------------ (6)

def test_the_gate_lists_this_test_after_the_one_of_plan_79_42():
    text = (ROOT / "scripts" / "gate.py").read_text()
    assert "test_plan79_43" in text and text.index("test_plan79_42") < text.index("test_plan79_43")
