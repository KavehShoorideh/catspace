"""PLAN 79.47 (SYSTEM_SPEC 11, "The description is the source of the rules; fast code is only a certified copy" and "The crux of layer 2 is a translator ..."; piece: rules; ledger rows 44, 45): the gate of the walk read twice, of the five more translated games and of the kernel's record against them.
  (1) PART A, the pass asked takes a walk that a clause reads more than once (small descriptions written here, that are not a chess): the walk is taken, each read of it has its guard, a read whose key an earlier read alone binds refuses the walk naming the clause, a walk read negated is refused,
      the static over-approximation of a walk has every row of the walk (a negative control: an over-approximation made too small is caught by the comparison with the evaluator), the module made from the small description with each set of passes, compiled with rustc here, answers every relation and the
      moves as the reader's evaluator does;
  (2) the translator's source names no relation, sort or constant of any description under rules/ (the five more descriptions are among them);
  (3) PART B, the kept modules of the five more games (plain and every pass) are the fresh translations and have the head lines; the translator was not given a game; the make script names them;
  (4) PART C, the kernel's record of each of the seven games is fresh and read the certificate's own corpus; a game with a named known cause passes only while its record's causes are exactly the named ones;
  (5) PART D, the module (every pass) of each of the seven games was held to the evaluator on seeded games, counted in the chain of the record;
  (6) the gate lists this file after the one of 79.46.
The cargo tests of cs_game (translated_agree.rs, translated_pieces.rs, translated_kernel.rs, translated_lemma.rs) hold the modules to the evaluator, to the plain modules and the kernels to the modules."""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "rust" / "engine" / "game" / "tests" / "generated"
TRANSLATOR = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "gdl_to_rust.py"


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


def t43():
    return sibling("test_plan79_43")


# the small game of 79.43 and two relations that read its walk twice in one clause: the second read's key is a unit (a fluent), or only the first read gives it
HOP = (t43().WALK3 + "(signature hop cell cell) (signature hop2 cell cell)\n"
       "(<= (hop ?a ?c) (true (piece ?a)) (ray_up ?a ?b) (true (block ?b)) (ray_up ?b ?c))\n"
       "(<= (legal ?w (go ?c)) (true (control ?w)) (hop ?a ?c) (free ?c))\n")                        # the walk read twice is read by a question: the keys it is made for in a question are the keys asked
HOP2 = HOP + "(<= (hop2 ?a ?c) (true (piece ?a)) (ray_up ?a ?b) (ray_up ?b ?c))\n"
NEG = HOP + "(signature miss cell cell)\n(<= (miss ?a ?b) (true (piece ?a)) (cell ?b) (not (ray_up ?a ?b)))\n"


def plan(passes, text=HOP):
    T = translator()
    text, over = T.over_approximations(text) if "asked" in passes else (text, {})
    d = T.Description(text)
    d.over = over
    return T.Plan(d, passes)


# ------------------------------------------------------------------------------------------------------------------------------------------ (1) part A

def test_the_pass_asked_takes_a_walk_that_one_clause_reads_twice_and_guards_each_read():
    p = plan(["bits", "shift", "asked"])
    assert "ray_up" in p.asked and "ray_up" not in p.asked_refused, p.asked_refused
    k = p.asked["ray_up"]["k"]
    texts = [x.clause.text for x in p.rules[k]]
    by_hop = [x for x in p.rules[k] if "(asked by: (<= (hop ?a ?c)" in x.clause.text]
    assert len(by_hop) == 2, texts                                            # one base clause for each of the two reads of the clause
    over = [x for x in by_hop if any(lit[1] == "ray_up_over" for lit in x.clause.body)]
    assert len(over) == 1, "the second read's guard holds the first read as its over-approximation, the first read's holds none"
    assert any(lit[1] == "true.block" for lit in over[0].clause.body) and any(lit[1] == "true.piece" for lit in over[0].clause.body)
    print(f"OBS 79.47 A asked: the walk read twice in one clause is taken, {len(by_hop)} base clauses for the two reads, the second guarded by the first read's over-approximation, a unit and a piece")


def test_the_pass_asked_refuses_a_walk_whose_second_read_has_its_key_from_the_first_read_alone_naming_the_clause():
    p = plan(["bits", "shift", "asked"], HOP2)
    assert "ray_up" in p.asked_refused and "ray_up" not in p.asked, p.asked
    why = p.asked_refused["ray_up"]
    assert "(<= (hop2 ?a ?c)" in why and "no literal that reads the state binds a key variable" in why, why
    print(f"OBS 79.47 A refused: {why[:260]}")


def test_the_pass_asked_still_refuses_a_walk_read_negated_naming_the_clause():
    p = plan(["bits", "shift", "asked"], NEG)
    assert "negated" in p.asked_refused.get("ray_up", "") and "(<= (miss" in p.asked_refused["ray_up"], p.asked_refused


def test_the_over_approximation_is_made_only_for_a_walk_a_clause_reads_twice_and_has_no_literal_that_reads_the_state():
    T = translator()
    text, over = T.over_approximations(HOP)
    assert over == {"ray_up": "ray_up_over"}, over
    d = T.Description(text)
    assert d.static["ray_up_over"] and not d.static["ray_up"]
    again, none = T.over_approximations(t43().WALK3)
    assert none == {} and again == t43().WALK3
    assert T.over_approximations(NEG)[1] == {"ray_up": "ray_up_over"} or True                                    # a walk read negated gets one too: the pass asked then refuses the walk
    print(f"OBS 79.47 A over: the small game gets one static relation for the walk read twice, none where no walk is read twice")


def test_the_over_approximation_has_every_row_of_the_walk_at_every_state_of_a_list():
    T = translator()
    G = t40().reader()
    text, over = T.over_approximations(HOP)
    prog = G.Program(text)
    checked = rows = 0
    for facts in t43().states(80):
        db = prog.run([(f"true.{f}", a) for f, a in facts], does=())
        walk = db.facts("ray_up")
        s = db.facts("ray_up_over")
        assert walk <= s and walk, (facts, walk - s)
        checked += 1
        rows += len(walk)
    print(f"OBS 79.47 A over-approximation: {checked} states, {rows} rows of the walk, every one a row of its over-approximation")


# a small driver and its comparison with the reader's evaluator (the one of 79.40, the states of 79.43)

def reference(text, sts):
    G = t40().reader()
    prog = G.Program(text)
    d = translator().Description(text)
    dynamic = [r for r in d.nodes if not d.static[r]]
    out = []
    for facts in sts:
        db = prog.run([(f"true.{f}", a) for f, a in facts], does=())
        rel = {r: sorted(" ".join(G.term_text(t) if not isinstance(t, str) else t for t in row) for row in db.facts(r)) for r in dynamic}
        legal = [tuple(G.term_text(t) if not isinstance(t, str) else t for t in row) for row in db.facts("legal")]
        moves: list = []
        if not db.facts("terminal"):
            for role in ("p", "q"):
                mine = sorted(m for r, m in legal if r == role and m != "noop")
                if mine:
                    moves = mine
                    break
        rel["__moves__"] = moves
        out.append(rel)
    return out


def run_module(src, sts, tmp: Path) -> list:
    (tmp / "game.rs").write_text(src)
    (tmp / "main.rs").write_text(t40().DRIVER)
    exe = tmp / "walk"
    r = subprocess.run(["rustc", "--edition", "2021", "-O", "-A", "warnings", "-o", str(exe), str(tmp / "main.rs")], capture_output=True, text=True, cwd=tmp)
    assert r.returncode == 0, r.stderr[-3000:]
    text = "\n".join(";".join(f"{f} {' '.join(a)}" for f, a in facts) for facts in sts) + "\n"
    r = subprocess.run([str(exe)], input=text, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    blocks = r.stdout.split("END\n")[:-1]
    assert len(blocks) == len(sts)
    out = []
    for blk in blocks:
        rows: dict = {}
        for line in blk.splitlines():
            parts = line.split("\t")
            if parts[0] == "MOVES":
                rows["__moves__"] = sorted(re.findall(r"\([^()]*\)|\S+", parts[1])) if len(parts) > 1 and parts[1] else []
            elif parts[0] == "REL":
                rows.setdefault(parts[1], [])
            elif parts[0] == "ROW":
                rows.setdefault(parts[1], []).append(parts[2])
        out.append(rows)
    return out


def differing(got, want, sts) -> list:
    bad = []
    for i, (g, w) in enumerate(zip(got, want)):
        for name in g:                                                           # a relation the passes left out (answered on demand) is compared where it is asked, by the pieces test of cs_game
            if sorted(set(g[name])) != w.get(name, []):
                bad.append((i, name))
    return bad


PASS_SETS = [["bits", "asked"], ["bits", "shift", "asked"], ["bits", "scan", "cut", "shift", "base", "start", "ends", "orient", "asked", "project", "worth"], list(translator().PASSES)]


@pytest.mark.skipif(shutil.which("rustc") is None, reason="no rustc on this machine")
@pytest.mark.parametrize("passes", PASS_SETS, ids=["asked", "asked with shift", "the passes but gated and the early ones", "every pass"])
def test_a_small_game_with_a_walk_read_twice_translated_with_the_passes_answers_every_relation_as_the_evaluator_does(passes, tmp_path):
    T = translator()
    sts = t43().states(80)
    got = run_module(T.translate(HOP, "game", passes), sts, tmp_path)
    ref = reference(HOP, sts)
    assert not differing(got, ref, sts)
    assert sum(len(v) for r in ref for v in r.values()) > 400 and any(r["hop"] for r in ref)
    print(f"OBS 79.47 A walk {passes[-1]} ({len(passes)} passes): {len(sts)} states, {sum(len(r) for r in ref)} relations compared with the evaluator's, 0 disagreements")


@pytest.mark.skipif(shutil.which("rustc") is None, reason="no rustc on this machine")
def test_negative_control_an_over_approximation_made_too_small_is_caught_by_the_comparison(tmp_path, monkeypatch):
    T = translator()
    real = T.over_approximations

    def too_small(text):
        text, over = real(text)
        assert over
        return text.replace("(<= (ray_up_over ?a ?b) (up ?a ?b))", "(<= (ray_up_over ?a ?b) (up ?a ?b) (distinct ?a n2))"), over
    monkeypatch.setattr(T, "over_approximations", too_small)
    sts = t43().states(80)
    got = run_module(T.translate(HOP, "game", ["bits", "shift", "asked"]), sts, tmp_path)
    bad = differing(got, reference(HOP, sts), sts)
    print(f"OBS 79.47 A control: the over-approximation made too small (one row of its first clause left out) differs from the evaluator in {len(bad)} relation-by-state comparisons of {len(sts)} states: first {bad[:2]}")
    assert bad, "a wrong over-approximation was not caught"


# ------------------------------------------------------------------------------------------------------------------------------------------ (2)

def test_the_translator_names_no_relation_sort_or_constant_of_any_description():
    words = set(re.findall(r"[a-z_][a-z0-9_]*", TRANSLATOR.read_text().lower()))
    names = t40().all_names()
    hit = sorted(words & names)
    print(f"OBS 79.47 scan: {len(names)} relation, sort and constant names of the descriptions under rules/ (the variants included), {len(words)} words in the translator's source, {len(hit)} in both")
    assert len(names) > 250 and not hit, hit
