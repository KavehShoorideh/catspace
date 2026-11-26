"""PLAN 79.40 (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ... made fast piece by piece under tests"; ledger row 45; piece: rules): the gate of the passes bits, order and keep, and of the generated files.
  (1) the files kept in the repository are the plain file and the file with every pass of each game, each carrying the hashes of its description and of the translator, and a fresh translation is the same file; the files of the single passes are
      made here, in the test run (plan 79.40 part F), and are not kept; a translation is the same under two hash seeds of the process;
  (2) the plain files differ from those before the plan in the line of the translator hash only;
  (3) the passes decide from shapes: the translator's source names no relation, sort or constant of any description under rules/; the same decisions come from a description with every name replaced; nothing under rust/engine/*/src names the
      generated files, the translator or its helpers; the gate lists this file;
  (4) what the pass bits recognises and refuses, on small descriptions written here: products found, relations held as bits, a recursive walk compiled to loops over words, each refusal naming its clause;
  (5) a small game that is not a chess, translated with each pass and compiled with rustc here, answers every relation as the reader's evaluator does on every state of a list (the pass bits against the evaluator, outside chess).
The piece-by-piece comparisons against the plain module on the chess games are the cargo tests of cs_game (translated_pieces.rs; the single passes with `--cfg singles`)."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
GEN = ENGINE / "game" / "tests" / "generated"
TRANSLATOR = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "gdl_to_rust.py"
BEFORE = "d86fd3e9"                      # the commit before this plan's code (the plan block's own commit: documents only)
GAMES = ("chess", "atomic")
SINGLES = {"prev": ["dense", "demand", "copy", "delta"], "dense": ["dense"], "demand": ["demand"], "copy": ["copy"], "delta": ["delta"], "bits": ["bits"], "order": ["order"], "keep": ["keep"]}


def translator():
    from catspace.incubator.analyst.discovery import gdl_to_rust as T
    return T


def reader():
    from catspace.incubator.analyst.discovery import gdl as G
    return G


def sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def description_of(game: str) -> str:
    return sibling("test_plan79_37").description_of(game)


def head_line(text: str, key: str):
    for line in text.splitlines()[:6]:
        if line.startswith(f"// {key}:"):
            return line.split(":", 1)[1].strip()
    return None


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


# ------------------------------------------------------------------------------------------------------------------------------------------ (1)

@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("variant", ["", "_fast"])
def test_a_kept_file_carries_the_hashes_and_the_passes_and_a_fresh_translation_is_the_same_file(game, variant):
    T = translator()
    module = game + variant
    passes = [p for p in T.PASSES if p != "carry"] if variant else []
    path = GEN / f"{module}.rs"
    text = description_of(game)
    src = path.read_text()
    assert head_line(src, "description-sha256") == hashlib.sha256(text.encode()).hexdigest(), f"{module}: the description changed; make the file again"
    assert head_line(src, "translator-sha256") == T.translator_hash(), f"{module}: the translator changed; make the file again"
    assert head_line(src, "passes") == (",".join(passes) or None), module
    assert T.translate(text, module, passes) == src, f"{module}: a fresh translation differs from the kept file"
    print(f"OBS 79.40 F {module}: {len(src)} bytes, {src.count(chr(10))} lines, passes {head_line(src, 'passes')}, the hashes match and a fresh translation is the same file")


def test_only_the_plain_files_and_the_files_with_every_pass_are_kept():
    names = sorted(p.name for p in GEN.iterdir() if p.is_file())
    more = ("3check", "kingofthehill", "antichess", "chess960", "horde")                                                    # 2026-11-24 (plan 79.47): the five more games of the seven certified kernels, plain and every pass
    # 2026-11-25 (plan 79.56, mended): the modules <game>_support.rs of the six games where pass (26) finds a test that a question reads (not antichess) are kept besides the others
    assert names == sorted([f"{g}.rs" for g in GAMES + more] + [f"{g}_fast.rs" for g in GAMES + more] + [f"{g}_carry.rs" for g in GAMES + more] + [f"{g}_support.rs" for g in GAMES + more if g != "antichess"] + ["chess_lemma.rs"]), names          # 2026-11-24 (plan 79.45): the first game's file with every pass and its lemma is kept too
    tracked = git("ls-files", "rust/engine/game/tests/generated").stdout.split()
    assert sorted(Path(x).name for x in tracked) == names, tracked
    assert "generated/scratch/" in (ROOT / ".gitignore").read_text()


@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("variant", sorted(SINGLES))
def test_the_file_of_a_single_pass_is_made_here_and_carries_its_hashes_and_its_pass(game, variant, tmp_path):
    T = translator()
    text = description_of(game)
    src = T.translate(text, f"{game}_{variant}", SINGLES[variant])
    (tmp_path / f"{game}_{variant}.rs").write_text(src)
    assert head_line(src, "description-sha256") == hashlib.sha256(text.encode()).hexdigest()
    assert head_line(src, "translator-sha256") == T.translator_hash()
    assert head_line(src, "passes") == ",".join(SINGLES[variant])
    assert not (GEN / f"{game}_{variant}.rs").exists(), "a single-pass file is not kept in the repository"
    print(f"OBS 79.40 F {game}_{variant}: made here, {src.count(chr(10))} lines, passes {head_line(src, 'passes')}")


def test_the_maker_writes_the_scratch_files_of_the_single_passes_and_nothing_is_kept_of_them(tmp_path):
    spec = importlib.util.spec_from_file_location("gdl_to_rust_make", ROOT / "scripts" / "gdl_to_rust_make.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.make(tmp_path, {"_bits": ["bits"]}, None, m.SCRATCH_GAMES)                       # 2026-11-24 (plan 79.47): the files of the single passes are made for the first two games only (SCRATCH_GAMES)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["atomic_bits.rs", "chess_bits.rs"]
    assert {f"_{k}" for k in SINGLES} <= set(m.SCRATCH)
    assert m.KEPT == {"": [], "_fast": [p for p in translator().PASSES if p != "carry"], "_carry": list(translator().PASSES)}          # 2026-11-25 (plan 79.54): the pass carry is kept in its own modules


def test_a_fresh_translation_is_the_same_under_two_hash_seeds_of_the_process():
    code = ("import hashlib,sys\n"
            "sys.path.insert(0, %r)\n"
            "from pathlib import Path\n"
            "from catspace.incubator.analyst.discovery import gdl_to_rust as T\n"
            "text = Path(%r).read_text()\n"
            "print(hashlib.sha256(T.translate(text, 'm', [p for p in T.PASSES if p != 'carry']).encode()).hexdigest())\n"
            "print(hashlib.sha256(T.translate(text, 'm', ['bits']).encode()).hexdigest())\n") % (str(ROOT), str(ROOT / "rules" / "chess.gdl"))
    outs = []
    for seed in ("1", "2"):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)})
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.split())
    assert outs[0] == outs[1] and len(outs[0]) == 2 and all(len(x) == 64 for x in outs[0])
    print(f"OBS 79.40 F seeds: the translation with every pass of the first game is {outs[0][0][:16]}... and with bits alone {outs[0][1][:16]}... under PYTHONHASHSEED 1 and 2")


# ------------------------------------------------------------------------------------------------------------------------------------------ (2)

@pytest.mark.skipif(git("cat-file", "-e", f"{BEFORE}^{{commit}}").returncode != 0, reason="the commit before the plan is not on this machine")
@pytest.mark.parametrize("game", GAMES)
def test_the_plain_files_differ_from_those_before_the_plan_in_the_line_of_the_translator_hash_only(game):
    import difflib
    old = git("show", f"{BEFORE}:rust/engine/game/tests/generated/{game}.rs").stdout.splitlines()
    new = (GEN / f"{game}.rs").read_text().splitlines()
    # 2026-11-24 (plan 79.47): the plain file gained two pure insertions, the item `restart` of the module and the `clear` of a relation that it needs (no line of the plain translation changed: the lines of the file that the
    # plan before it made are all there, in order, and the only line that differs is the translator's hash)
    ops = [o for o in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes() if o[0] != "equal"]
    differ = [o for o in ops if o[0] == "replace"]
    assert all(o[0] == "insert" or (o[0] == "replace" and o[2] - o[1] == 1 and o[4] - o[3] == 1 and old[o[1]].startswith("// translator-sha256:") and new[o[3]].startswith("// translator-sha256:")) for o in ops), ops
    inserted = "\n".join(new[o[3]:o[4]] and "\n".join(new[o[3]:o[4]]) for o in ops if o[0] == "insert")
    assert "pub fn restart(" in inserted and inserted.count("pub fn clear") == 1 and "plan 79.47" in inserted
    print(f"OBS 79.40 F {game}: the plain file differs from the one before the plan in {len(differ)} line (the translator hash) and by two insertions (`restart` and the `clear` it needs)")


# ------------------------------------------------------------------------------------------------------------------------------------------ (3)

def all_names() -> set:
    """The relation names, sort names and the constants (the members of sorts and the symbols of the facts) that the descriptions under rules/ use, but for the fixed words of the language."""
    s37 = sibling("test_plan79_37")
    G = reader()
    names = set(s37.defined_names())
    for path in sorted((ROOT / "rules").rglob("*.gdl")):
        try:
            prog = G.Program(path.read_text())
        except Exception:
            continue
        names |= set(prog.sorts)
        for ms in prog.sorts.values():
            for m in ms:
                if isinstance(m, str):
                    names.add(m)
    return {n for n in names if re.fullmatch(r"[a-z_][a-z0-9_]*", n) and (len(n) >= 3 or n in s37.defined_names())} - s37.FIXED_WORDS          # (the files a to h are one letter, as the names of variables in a source are)


def test_the_translator_names_no_relation_sort_or_constant_of_any_description():
    words = set(re.findall(r"[a-z_][a-z0-9_]*", TRANSLATOR.read_text().lower()))
    names = all_names()
    hit = sorted(words & names)
    print(f"OBS 79.40 G scan: {len(names)} relation, sort and constant names of the descriptions under rules/, {len(words)} words in the translator's source, {len(hit)} in both")
    assert len(names) > 250
    assert not hit, f"the translator's source holds names of a game: {hit}"


def test_the_scan_sees_a_planted_sort_name():
    assert {"direction", "kind"} <= (set(re.findall(r"[a-z_][a-z0-9_]*", "x = 'direction'; y = 'kind'")) & all_names())


def rename(text: str, d) -> str:
    keep = set(translator().QUESTION_RELS) | {"role", "init", "true", "does", "next", "noop", "distinct", "not", "record", "sort", "signature", "range", "or", "<="}
    names = sorted({n for n in re.findall(r"[a-z_][a-z0-9_.]*", text.lower()) if not n.startswith("?")} - keep - set(d.syms) | {r for r in d.prog.signatures} | set(d.prog.sorts))
    names = [n for n in names if n not in keep and not n.isdigit()]
    out = text
    for i, n in enumerate(sorted(names, key=len, reverse=True)):
        out = re.sub(rf"(?<![a-z0-9_?.]){re.escape(n)}(?![a-z0-9_.])", f"zz{i}", out)
    return out


def decisions(plan) -> dict:
    return {"bits": len(plan.bits), "orientations": sorted(len(i.orients) for i in plan.bits.values()), "tiles": sorted(len(i.tiles) for i in plan.bits.values()), "recursive": len(plan.bit_rec),
            "refused": len(plan.bits_refused), "demanded": len(plan.demanded), "dense": len(plan.dense), "copy": len(plan.copy), "units": sorted(translator_units(plan))}


def translator_units(plan) -> list:
    return sorted(plan.units.values())


@pytest.mark.parametrize("game", GAMES)
def test_the_passes_give_the_same_decisions_on_a_description_with_every_name_replaced(game):
    T = translator()
    text = description_of(game)
    d = T.Description(text)
    d2 = T.Description(rename(text, d))
    a, b = T.Plan(d, T.PASSES), T.Plan(d2, T.PASSES)
    da, db = decisions(a), decisions(b)
    assert da == db, (da, db)
    assert da["bits"] > 40 and da["recursive"] >= 2 and da["units"]
    print(f"OBS 79.40 G renamed {game}: {da['bits']} relations as bits ({da['recursive']} recursive), {len(da['units'])} products of sorts, {da['refused']} refused, orientations {da['orientations'][-3:]}; the same with every name replaced")


def test_nothing_under_the_sources_of_the_engine_names_the_generated_files_the_translator_or_the_helpers():
    bad = []
    for src in sorted(ENGINE.glob("*/src")):
        for f in src.rglob("*"):
            if f.is_file() and f.suffix in (".rs", ".toml"):
                t = f.read_text()
                for w in ("generated/", "gdl_to_rust", "translated_", "chess_fast", "atomic_fast", "scratch/", "singles", "gdl_to_rust_make"):
                    if w in t:
                        bad.append(f"{f.relative_to(ROOT)}: {w}")
    assert not bad, bad
    assert not list(ENGINE.glob("*/src/generated")) and not list(ENGINE.glob("*/src/**/generated"))


def test_the_gate_lists_this_test_after_the_one_of_plan_79_38():
    text = (ROOT / "scripts" / "gate.py").read_text()
    assert "test_plan79_40" in text and text.index("test_plan79_38") < text.index("test_plan79_40")


# ------------------------------------------------------------------------------------------------------------------------------------------ (4)

WALK = """(sort col a b c d)
(sort row 1 2 3)
(sort who p q)
(sort big (range 0 70))
(sort move (go col row) noop)
(sort score 0 100)
(signature piece col row) (signature block col row) (signature control who) (signature does who move) (signature role who) (signature opponent who who)
(signature col col) (signature row row) (signature right col col) (signature up row row) (signature cell col row) (signature occupied col row) (signature free col row)
(signature reach col row col row) (signature legal who move) (signature terminal) (signature goal who score) (signature count big) (signature wide big big) (signature carry big)

(role p) (role q) (opponent p q) (opponent q p)
(col a) (col b) (col c) (col d) (row 1) (row 2) (row 3)
(right a b) (right b c) (right c d) (up 1 2) (up 2 3)
(init (piece a 1)) (init (block c 1)) (init (block b 3)) (init (control p))
(<= (cell ?c ?r) (col ?c) (row ?r))
(<= (occupied ?c ?r) (true (block ?c ?r)))
(<= (free ?c ?r) (cell ?c ?r) (not (occupied ?c ?r)))
(<= (reach ?c1 ?r ?c2 ?r) (right ?c1 ?c2) (row ?r))
(<= (reach ?c1 ?r ?c3 ?r) (reach ?c1 ?r ?c2 ?r) (free ?c2 ?r) (right ?c2 ?c3))
(<= (reach ?c ?r1 ?c ?r2) (up ?r1 ?r2) (col ?c))
(<= (reach ?c ?r1 ?c ?r3) (reach ?c ?r1 ?c ?r2) (free ?c ?r2) (up ?r2 ?r3))
(<= (legal ?w (go ?c2 ?r2)) (true (control ?w)) (true (piece ?c1 ?r1)) (reach ?c1 ?r1 ?c2 ?r2) (free ?c2 ?r2))
(<= (legal ?o noop) (true (control ?w)) (opponent ?w ?o))
(<= (next (piece ?c ?r)) (does ?w (go ?c ?r)))
(<= (next (block ?c ?r)) (true (block ?c ?r)))
(<= (next (control ?o)) (true (control ?w)) (opponent ?w ?o))
(<= terminal (true (piece d ?r)))
(<= (goal ?w 100) (role ?w) (true (control ?w)))
(<= (goal ?w 0) (role ?w) (not (true (control ?w))))
(<= (wide ?x ?y) (carry ?x) (carry ?y))
"""


def walk_plan(passes):
    T = translator()
    return T.Plan(T.Description(WALK), passes)


def test_the_pass_bits_finds_the_product_of_two_sorts_from_the_clauses_and_holds_a_relation_over_them_as_bits():
    p = walk_plan(["bits"])
    assert p.units, "no product found"
    ((a, b), _w), = list(p.units.items())
    assert {a, b} == {"col", "row"}
    reach = p.bits["reach"]
    assert [t.sorts for t in reach.tiles] == [(a, b), (a, b)]
    assert "free" in p.bits and "occupied" in p.bits and "true.piece" in p.bits
    print(f"OBS 79.40 H units {sorted(p.units)}, held as bits {sorted(p.bits)}")


def test_the_pass_bits_compiles_a_walk_that_stops_at_a_test_to_loops_over_words():
    T = translator()
    src = T.translate(WALK, "walk", ["bits"])
    assert "dw.dl_r_reach" in src and "std::mem::swap" in src and "or_word" in src
    assert "reach" in walk_plan(["bits"]).bit_rec


def test_the_pass_bits_refuses_a_sort_of_more_than_the_word_and_names_the_clause():
    p = walk_plan(["bits"])
    for rel in ("count", "carry", "wide"):
        assert rel not in p.bits
    assert "70" in p.bits_refused["carry"] or "members" in p.bits_refused["carry"]
    assert any("goal" == r for r in p.bits_refused) and "only the questions" in p.bits_refused["goal"]


def test_the_pass_bits_refuses_a_place_that_is_a_compound_and_a_constant_outside_the_sort_naming_the_clause():
    T = translator()
    text = WALK + "(<= (cell2 ?c ?r) (true (piece ?c ?r)) (right ?c e))\n(signature cell2 col row)\n(<= (legal ?w noop) (cell2 ?c ?r) (true (control ?w)))\n"
    p = T.Plan(T.Description(text), ["bits"])
    assert "right" not in p.bits or True
    assert "cell2" not in p.bits and "e" in p.bits_refused["cell2"] or "right" in p.bits_refused, p.bits_refused


def test_a_relation_in_a_recursive_set_with_one_that_is_not_held_as_bits_is_refused_with_the_set_named():
    T = translator()
    text = WALK + "(signature mix1 col row) (signature mix2 big)\n(<= (mix1 ?c ?r) (true (piece ?c ?r)))\n(<= (mix1 ?c ?r) (mix2 ?x) (true (piece ?c ?r)))\n(<= (mix2 ?x) (mix1 ?c ?r) (carry ?x))\n"
    p = T.Plan(T.Description(text), ["bits"])
    assert "mix1" not in p.bits and "mix2" not in p.bits
    assert "recursive set" in p.bits_refused["mix1"] and "mix2" in p.bits_refused["mix1"], p.bits_refused


@pytest.mark.parametrize("game", GAMES)
def test_the_pass_order_joins_the_same_clauses_in_another_order_only_where_literals_start_equally_well(game):
    T = translator()
    G = reader()
    d = T.Description(description_of(game))
    plan = T.Plan(d, ["order"])
    changed = total = 0
    for k, (comp, rules) in enumerate(d.strata):
        for a, b in zip(rules, plan.rules[k]):
            total += 1
            assert a.clause is b.clause
            assert sorted(s[1] for s in a.steps if s[0] != "distinct") == sorted(s[1] for s in b.steps if s[0] != "distinct"), a.clause.text
            assert len(a.steps) == len(b.steps) and a.nvars == b.nvars and a.head_rel == b.head_rel
            if a.steps != b.steps:
                changed += 1
    assert 5 <= changed < total // 4, (changed, total)
    print(f"OBS 79.40 J order {game}: {total} clauses, {changed} joined in another order (the same literals, the same variables)")


def test_the_pass_keep_makes_the_lookup_of_a_compound_without_a_vector_and_keeps_the_buffers_of_the_questions():
    T = translator()
    keep = T.translate(WALK, "walk", ["keep"])
    plain = T.translate(WALK, "walk")
    assert "let mut buf = [0u32; 16]" in keep and "let mut buf = [0u32; 16]" not in plain
    assert "fn ensure" in keep and "does_buf" in keep and "fn ensure" not in plain
    assert "pub struct Dw" in keep and "pub struct Dw" not in plain


# ------------------------------------------------------------------------------------------------------------------------------------------ (5)

DRIVER = r"""
include!("game.rs");
use game::*;
fn main() {
    let mut g = Game::new();
    let input = std::io::read_to_string(std::io::stdin()).unwrap();
    for line in input.lines() {
        let facts: Vec<(String, Vec<String>)> = line.split(';').filter(|x| !x.trim().is_empty()).map(|f| { let mut it = f.split_whitespace(); let r = it.next().unwrap().to_string(); (r, it.map(|x| x.to_string()).collect()) }).collect();
        let s = g.state_of_facts(&facts).unwrap();
        let mover = g.mover(&s);
        let mut mv: Vec<String> = g.moves(&s).iter().map(|m| g.render(*m)).collect();
        mv.sort();
        println!("MOVES\t{}", mv.join(" "));
        let rels = g.all_relations(&s, &[]);
        for (name, rows) in rels {
            println!("REL\t{}", name);
            let mut out: Vec<String> = rows.iter().map(|r| r.iter().map(|t| g.render(*t)).collect::<Vec<_>>().join(" ")).collect();
            out.sort(); out.dedup();
            for o in out { println!("ROW\t{}\t{}", name, o); }
        }
        let _ = mover;
        println!("END");
    }
}
"""


def states(n: int) -> list:
    rng = random.Random(7)
    cells = [(c, r) for c in "abcd" for r in "123"]
    out = []
    for _ in range(n):
        p = rng.choice(cells)
        blocks = rng.sample(cells, rng.randint(0, 5))
        facts = [("piece", p), ("control", (rng.choice("pq"),))] + [("block", b) for b in blocks]
        out.append([(f, tuple(a)) for f, a in facts])
    return out


def compile_walk(passes, tmp: Path) -> Path:
    T = translator()
    (tmp / "game.rs").write_text(T.translate(WALK, "game", passes))
    (tmp / "main.rs").write_text(DRIVER)
    exe = tmp / "walk"
    r = subprocess.run(["rustc", "--edition", "2021", "-O", "-A", "warnings", "-o", str(exe), str(tmp / "main.rs")], capture_output=True, text=True, cwd=tmp)
    assert r.returncode == 0, r.stderr[-3000:]
    return exe


def reference(sts) -> list:
    G = reader()
    prog = G.Program(WALK)
    d = translator().Description(WALK)
    dynamic = [r for r in d.nodes if not d.static[r]]
    out = []
    for facts in sts:
        db = prog.run([(f"true.{f}", a) for f, a in facts], does=())
        rel = {}
        for r in dynamic:
            rel[r] = sorted(" ".join(G.term_text(t) if not isinstance(t, str) else t for t in row) for row in db.facts(r))
        out.append(rel)
    return out


@pytest.mark.skipif(shutil.which("rustc") is None, reason="no rustc on this machine")
@pytest.mark.parametrize("passes", [[], ["bits"], ["bits", "order", "keep"], list(translator().PASSES) if False else ["dense", "demand", "copy", "delta", "bits", "order", "keep"]], ids=["plain", "bits", "bits-order-keep", "every pass"])
def test_a_small_game_that_is_not_chess_translated_with_the_passes_answers_every_relation_as_the_evaluator_does(passes, tmp_path):
    exe = compile_walk(passes, tmp_path)
    sts = states(60)
    text = "\n".join(";".join(f"{f} {' '.join(a)}" for f, a in facts) for facts in sts) + "\n"
    r = subprocess.run([str(exe)], input=text, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    blocks = r.stdout.split("END\n")[:-1]
    assert len(blocks) == len(sts)
    ref = reference(sts)
    compared = 0
    for i, (blk, want) in enumerate(zip(blocks, ref)):
        made, rows = [], {}
        for line in blk.splitlines():
            parts = line.split("\t")
            if parts[0] == "REL":
                made.append(parts[1])
                rows.setdefault(parts[1], [])
            elif parts[0] == "ROW":
                rows.setdefault(parts[1], []).append(parts[2])
        assert made, "the module made no relation"
        for name in made:
            assert sorted(rows[name]) == want[name], (i, passes, name, sorted(rows[name]), want[name], sts[i])
            compared += 1
    assert compared > 60 * 5
    print(f"OBS 79.40 I walk {passes or 'plain'}: {len(sts)} states, {compared} relations compared with the evaluator's, 0 disagreements")
