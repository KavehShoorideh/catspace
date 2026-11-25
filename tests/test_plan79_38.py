"""PLAN 79.38 THE FIRST GENERAL PASSES OF THE TRANSLATOR (SYSTEM_SPEC 11, "The crux of layer 2 is a translator from the description to Rust, made fast piece by piece under tests"; ledger row 45; piece: rules).
Rust holds the tests of the modules with passes (rust/engine/game/tests/translated_pieces.rs and translated_common/pieces.rs: each piece held to the plain translation; translated_agree.rs: the answers held to the evaluator;
translated_speed_passes.rs: the times). THIS file is what belongs to Python: the gate.
  (1) the translation with no pass is the committed plain file, and the plain files differ from those of the commit before this plan in the line of the translator hash only;
  (2) the translator's source, with its passes, names no relation of any description under rules/ (the scan of 79.37, again), and no pass special-cases a name;
  (3) every file with passes carries the hash of its description, the hash of the translator and the list of its passes, and a fresh translation is the same file; the same under two hash seeds of the process;
  (4) the join order of the translator, with no relation avoided, is the reader's on every clause of the descriptions;
  (5) the conditions of the passes hold or the pass refuses: pass (3) copies only a clause of the shape stated, pass (2) keeps a relation in full that is read whole, pass (1) leaves a relation that fails its condition a hash relation;
  (6) nothing under rust/engine/*/src names any of it, and no source of the engine and no manifest differs from the commit before the plan; (7) the test is in the gate after the one of 79.37.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
GEN = ENGINE / "game" / "tests" / "generated"
TRANSLATOR = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "gdl_to_rust.py"
BEFORE = "12f17e26"                      # the commit before this plan's code (the plan block's own commit: documents only)
KEPT = ("fast",)                         # plan 79.40 part F: of the files with passes only `fast` (every pass) is kept in the repository; the others are made here, in the test run
VARIANTS = {"dense": ["dense"], "demand": ["demand"], "copy": ["copy"], "delta": ["delta"], "bits": ["bits"], "order": ["order"], "keep": ["keep"], "fast": ["dense", "demand", "copy", "delta", "bits", "order", "keep", "scan", "cut", "shift", "base", "start", "ends", "orient", "asked", "project", "worth", "gated", "cones", "words"]}           # plan 79.43: `fast` is every pass, the new ones too; 2026-11-24 (plan 79.45): and the pass gated; 2026-11-24 (plan 79.52): and the passes cones and words
GAMES = ("chess", "atomic")


def translator():
    from catspace.incubator.analyst.discovery import gdl_to_rust as T
    return T


def reader():
    from catspace.incubator.analyst.discovery import gdl as G
    return G


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def have_before() -> bool:
    return git("cat-file", "-e", f"{BEFORE}^{{commit}}").returncode == 0


def sibling():
    spec = importlib.util.spec_from_file_location("test_plan79_37", ROOT / "tests" / "test_plan79_37.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def description_of(game: str) -> str:
    return sibling().description_of(game)


def head_line(path: Path, key: str):
    for line in path.read_text().splitlines()[:6]:
        if line.startswith(f"// {key}:"):
            return line.split(":", 1)[1].strip()
    return None


# ------------------------------------------------------------------------------------------------------------------------------------------ (1)

@pytest.mark.parametrize("game", GAMES)
def test_the_translation_with_no_pass_is_the_committed_plain_file(game):
    T = translator()
    text = description_of(game)
    path = GEN / f"{game}.rs"
    assert T.translate(text, game) == path.read_text() == T.translate(text, game, ())
    assert head_line(path, "passes") is None


@pytest.mark.skipif(not have_before(), reason="the commit before the plan is not on this machine")
@pytest.mark.parametrize("game", GAMES)
def test_the_plain_files_differ_from_those_before_the_plan_in_the_line_of_the_translator_hash_only(game):
    import difflib
    old = git("show", f"{BEFORE}:rust/engine/game/tests/generated/{game}.rs").stdout.splitlines()
    new = (GEN / f"{game}.rs").read_text().splitlines()
    # 2026-11-24 (plan 79.47): the plain file gained two pure insertions, the item `restart` of the module and the `clear` of a relation that it needs; no line of the plain translation changed but the translator's hash
    ops = [o for o in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes() if o[0] != "equal"]
    differ = [o for o in ops if o[0] == "replace"]
    assert all(o[0] == "insert" or (o[0] == "replace" and o[2] - o[1] == 1 and o[4] - o[3] == 1 and old[o[1]].startswith("// translator-sha256:") and new[o[3]].startswith("// translator-sha256:")) for o in ops), ops
    inserted = "\n".join("\n".join(new[o[3]:o[4]]) for o in ops if o[0] == "insert")
    assert "pub fn restart(" in inserted and inserted.count("pub fn clear") == 1 and "plan 79.47" in inserted
    print(f"OBS 79.38 E {game}: the plain file differs from the one before the plan in {len(differ)} line (the translator hash) and by two insertions (`restart` and the `clear` it needs)")


# ------------------------------------------------------------------------------------------------------------------------------------------ (2)

def test_the_translator_with_its_passes_names_no_relation_of_any_description_but_the_fixed_words_of_the_language():
    s = sibling()
    names = s.defined_names()
    words = set(re.findall(r"[a-z_][a-z0-9_]*", TRANSLATOR.read_text().lower()))
    hit = sorted(words & names)
    print(f"OBS 79.38 E scan: {len(names)} relation names defined by the descriptions under rules/, {len(words)} words in the translator's source, {len(hit)} in both")
    assert not hit, f"the translator's source holds names of a game's relations: {hit}"


def test_no_pass_branches_on_a_name_the_same_passes_give_the_same_decisions_on_a_renamed_description():
    """The passes decide from shapes: the description with every relation and sort renamed gets the same number of dense relations, of demanded relations and of copied next relations."""
    T = translator()
    G = reader()
    text = description_of("chess")
    d = T.Description(text)
    keep = set(T.QUESTION_RELS) | {"role", "init", "true", "does", "next", "noop", "distinct", "not", "record", "sort", "signature", "range", "or", "<="}
    names = sorted({n for n in re.findall(r"[a-z_][a-z0-9_.]*", text.lower()) if not n.startswith("?")} - keep - set(d.syms) | {r for r in d.prog.signatures} | set(d.prog.sorts))
    names = [n for n in names if n not in keep and not n.isdigit()]
    renamed = text
    for i, n in enumerate(sorted(names, key=len, reverse=True)):
        renamed = re.sub(rf"(?<![a-z0-9_?.]){re.escape(n)}(?![a-z0-9_.])", f"zz{i}", renamed)
    d2 = T.Description(renamed)
    four = ["dense", "demand", "copy", "delta"]                 # (the four passes of plan 79.38: the pass bits of 79.40 takes relations the pass dense and the pass demand would; tests/test_plan79_40.py renames for it)
    a, b = T.Plan(d, four), T.Plan(d2, four)
    assert len(a.demanded) == len(b.demanded) > 20
    assert len(a.dense) == len(b.dense) > 20
    assert len(a.copy) == len(b.copy) == 2
    print(f"OBS 79.38 E renamed: {len(a.demanded)} demanded, {len(a.dense)} dense, {len(a.copy)} copied; with every name replaced: {len(b.demanded)}, {len(b.dense)}, {len(b.copy)}")


# ------------------------------------------------------------------------------------------------------------------------------------------ (3)

@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_a_file_with_passes_carries_the_hashes_and_the_passes_and_a_fresh_translation_is_the_same_file(game, variant):
    T = translator()
    module = f"{game}_{variant}"
    path = GEN / f"{module}.rs"
    text = description_of(game)
    fresh = T.translate(text, module, VARIANTS[variant])
    lines = fresh.splitlines()[:6]
    got = lambda key: next((x.split(":", 1)[1].strip() for x in lines if x.startswith(f"// {key}:")), None)
    assert got("description-sha256") == hashlib.sha256(text.encode()).hexdigest() and got("translator-sha256") == T.translator_hash() and got("passes") == ",".join(VARIANTS[variant]), module
    if variant in KEPT:                          # the kept file carries the hashes and the passes and is the fresh translation; a file of a single pass is made in the test run and is not in the repository
        assert head_line(path, "description-sha256") == hashlib.sha256(text.encode()).hexdigest(), f"{module}: the description changed; make the file again"
        assert head_line(path, "translator-sha256") == T.translator_hash(), f"{module}: the translator changed; make the file again"
        assert head_line(path, "passes") == ",".join(VARIANTS[variant]), module
        assert fresh == path.read_text(), f"{module}: a fresh translation differs from the committed file"
    else:
        assert not path.exists(), f"{module}: the file of a single pass is not kept in the repository"
    print(f"OBS 79.38 E {module}: {len(fresh)} bytes, {fresh.count(chr(10))} lines, passes {got('passes')}, the hashes at its head match and a fresh translation is identical{' to the kept file' if variant in KEPT else ' (made here, not kept)'}")


def test_the_files_of_the_generated_folder_are_the_plain_ones_and_those_with_every_pass():
    more = ("3check", "kingofthehill", "antichess", "chess960", "horde")                                          # 2026-11-24 (plan 79.47): the five more games of the seven certified kernels, plain and every pass
    want = {f"{g}.rs" for g in GAMES + more} | {f"{g}_{v}.rs" for g in GAMES + more for v in KEPT} | {"chess_lemma.rs"}              # 2026-11-24 (plan 79.45): the first game's file with every pass and its lemma is kept too
    assert {p.name for p in GEN.iterdir() if p.is_file()} == want


def test_a_fresh_translation_is_the_same_under_two_hash_seeds_of_the_process():
    code = ("import hashlib,sys\n"
            "sys.path.insert(0, %r)\n"
            "from pathlib import Path\n"
            "from catspace.incubator.analyst.discovery import gdl_to_rust as T\n"
            "text = Path(%r).read_text()\n"
            "print(hashlib.sha256(T.translate(text, 'm', list(T.PASSES)).encode()).hexdigest())\n") % (str(ROOT), str(ROOT / "rules" / "chess.gdl"))
    outs = []
    for seed in ("1", "2"):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)})
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1] and len(outs[0]) == 64
    print(f"OBS 79.38 E seeds: the fast translation of the first game is {outs[0][:16]}... under PYTHONHASHSEED 1 and 2")


# ------------------------------------------------------------------------------------------------------------------------------------------ (4)

@pytest.mark.parametrize("game", GAMES)
def test_the_join_order_of_the_translator_with_no_relation_avoided_is_the_readers_on_every_clause(game):
    T = translator()
    G = reader()
    prog = G.Program(description_of(game))
    n = 0
    for comp, rules in prog.strata:
        for rule in rules:
            mine = T.JoinRule(rule.clause, comp)
            for f in ("nvars", "head_rel", "head", "steps", "rec_steps"):
                assert getattr(mine, f) == getattr(rule, f), (f, rule.clause.text)
            n += 1
    print(f"OBS 79.38 E join order {game}: {n} clauses, every one joined as the reader joins it")
    assert n > 300


# ------------------------------------------------------------------------------------------------------------------------------------------ (5)

MINI = """(role r)
(role s)
(init (f a))
(init (g a))
(<= (legal r go) (true (f ?x)))
(<= (legal s noop) (true (f ?x)))
(<= (next (f ?x)) (true (f ?x)) (not (gone ?x)))
(<= (next (f b)) (does r go))
(<= (gone ?x) (true (g ?x)) (does r go))
(<= (next (g ?x)) (true (g ?x)) (distinct ?x c))
(<= (terminal) (true (f b)))
"""


def plan_of(text: str, passes):
    T = translator()
    return T.Plan(T.Description(text), passes)


def test_pass_copy_takes_a_clause_of_the_stated_shape_and_only_that():
    p = plan_of(MINI, ["copy"])
    assert sorted(p.copy) == ["next.f", "next.g"]
    assert all(len(v) == 1 for v in p.copy.values())
    # an extra positive literal, a repeated variable in the head, a compound in a negated literal, another fluent's rows: none is a copy of the rows of one fluent
    for bad in ("(<= (next (f ?x)) (true (f ?x)) (true (g ?x)))\n",
                "(<= (next (h ?x ?x)) (true (h ?x ?x)))\n(init (h a a))\n",
                "(<= (next (f ?x)) (true (g ?x)))\n",
                "(<= (next (f ?x)) (true (f ?x)) (not (gone (w ?x))))\n(<= (gone (w ?x)) (true (g ?x)))\n"):
        base = MINI.replace("(<= (next (f ?x)) (true (f ?x)) (not (gone ?x)))\n", "").replace("(<= (next (g ?x)) (true (g ?x)) (distinct ?x c))\n", "")
        if "(init (h" in bad:
            base = base.replace("(init (g a))\n", "(init (g a))\n(<= (next (g ?x)) (true (g ?x)))\n")
        q = plan_of(base + bad, ["copy"])
        assert "next.f" not in q.copy, bad
        assert "next.h" not in q.copy, bad


def test_pass_demand_keeps_a_relation_in_full_that_is_read_whole_and_answers_one_that_is_asked_by_key():
    text = """(sort s a b)
(signature f s)
(role r)
(init (f a))
(init (f b))
(<= (legal r go) (true (f ?x)))
(<= (big ?x ?y) (true (f ?x)) (true (f ?y)))
(<= (some ?x) (true (f ?x)) (big ?x ?y))
(<= (next (f ?x)) (true (f ?x)) (some ?x))
(<= (terminal) (true (f ?x)) (not (big ?x ?x)))
"""
    p = plan_of(text, ["demand"])
    assert "big" in p.demanded                      # asked with ?x bound, or only whether a row exists
    q = plan_of(text.replace("(<= (legal r go) (true (f ?x)))", "(<= (legal r ?x) (big ?x ?y))"), ["demand"])
    assert "big" not in q.demanded and "big" in q.refused       # read whole by `legal` (a relation a question reads is made in full): it remains in full, with its reason
    assert "read whole" in str(q.refused["big"])


def test_pass_dense_leaves_a_relation_that_fails_its_condition_a_hash_relation():
    T = translator()
    text = """(sort n (range 0 99))
(sort t a b)
(sort k (pair n t))
(signature f n n n) (signature g n) (signature h k) (signature legal t t)
(role r)
(init (f 0 0 0))
(<= (g ?x) (true (f ?x ?y ?z)))
(<= (legal r a) (true (f ?x ?y ?z)))
(<= (next (f ?x ?y ?z)) (true (f ?x ?y ?z)) (g ?x))
(signature role t) (signature init t) (signature next t)
"""
    # a pattern sort is not dense; a relation whose product of sorts is over the limit is not dense
    big = text.replace("(range 0 99)", "(range 0 399)")
    d = T.Description(text)
    p = T.Plan(d, ["dense"])
    assert "g" in p.dense and "true.f" in p.dense           # 100^3 = 10^6 tuples: within 2^22
    d2 = T.Description(big)
    q = T.Plan(d2, ["dense"])
    assert "true.f" not in q.dense and "g" in q.dense        # 400^3 > 2^22


# ------------------------------------------------------------------------------------------------------------------------------------------ (6), (7)

def test_nothing_under_the_sources_of_the_engine_names_the_translator_its_files_or_the_helpers():
    bad = []
    for src in sorted(ENGINE.glob("*/src")):
        for f in src.rglob("*"):
            if f.is_file() and f.suffix in (".rs", ".toml"):
                t = f.read_text()
                for w in ("generated/", "gdl_to_rust", "translated_", "chess_fast", "atomic_fast", "chess_dense", "atomic_dense"):
                    if w in t:
                        bad.append(f"{f.relative_to(ROOT)}: {w}")
    assert not bad, bad


@pytest.mark.skipif(not have_before(), reason="the commit before the plan is not on this machine")
def test_no_source_of_the_engine_and_no_manifest_differs_from_the_commit_before_the_plan():
    changed = git("diff", "--name-only", BEFORE, "--", "rust").stdout.split()
    # 2026-11-23 (plan 79.39, Kaveh's word): sources under rust/engine/*/src now change for the side-to-move question (tests/test_plan79_39.py holds them); this test checks only what plan 79.38 owned
    srcs = [c for c in changed if re.match(r"rust/engine/[^/]+/src/", c) and re.search(r"generated|translated|gdl_to_rust", c)]
    srcs += [c for c in changed if c.endswith("Cargo.toml") or c.endswith("Cargo.lock")]
    assert not srcs, srcs
    for protected in ("rust/engine/gdl/src/lib.rs", "rust/engine/game/src/gdl_game.rs"):          # kernel.rs and ledger.rs left this list at plan 79.39 (the side to move, held by tests/test_plan79_39.py)
        assert git("diff", "--quiet", BEFORE, "--", protected).returncode == 0, protected
    # 2026-11-23 (plan 79.44): rules/lemmas/ holds the lemma documents of plan 79.44, beside the descriptions and not descriptions; this test checks only what plan 79.38 owned (a description and the reader)
    # 2026-11-24 (plan 79.48): rules/placed/ holds the additive texts of the placed questions and their lists (documents, no description): excluded as rules/lemmas is.
    assert git("diff", "--quiet", BEFORE, "--", "rules", ":(exclude)rules/lemmas", ":(exclude)rules/placed", "catspace/incubator/analyst/discovery/gdl.py").returncode == 0, "a description or the reader changed"
    print(f"OBS 79.38 E: files under rust/ changed since {BEFORE}: {len(changed)}, none a source of a crate, a manifest or a protected file; no description under rules/ and not the reader changed")


def test_the_gate_lists_this_test_after_the_one_of_plan_79_37():
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert "test_plan79_37 test_plan79_38" in gate
