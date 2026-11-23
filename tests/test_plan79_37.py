"""PLAN 79.37 A MEASUREMENT BEFORE THE TRANSLATOR IS PLANNED (SYSTEM_SPEC 11, the blocks "The description is the source of the rules" and "The crux of layer 2 is a translator"; ledger rows 45, 44; piece: rules).
Rust holds the tests of the translated code (rust/engine/game/tests/translated_agree.rs, translated_speed.rs, translated_common/). THIS file is what belongs to Python: the gate.
  (1) the translator (catspace/incubator/analyst/discovery/gdl_to_rust.py) knows no game: its source holds no relation name that a description under rules/ defines, but the fixed words of the language;
  (2) the generated files carry the hash of the description and of the translator they were made from, and a fresh translation gives the same file: a description or translator that changed without the file made again fails here;
  (3) the same translator wrote the chess file and the atomic file (the variant needed no change);
  (4) nothing under rust/engine/*/src names a generated file, the translator or the test helpers; the generated files live under rust/engine/game/tests/generated only;
  (5) the files that must not be touched are not: kernel.rs, gdl_game.rs, gdl/src/lib.rs, ledger.rs, every Cargo.toml and Cargo.lock, and no source under rust/engine/*/src differs from the commit before the plan;
  (6) the test is in the gate after the one of 79.36.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
GEN = ENGINE / "game" / "tests" / "generated"
TRANSLATOR = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "gdl_to_rust.py"
BEFORE = "2153532f"                      # the commit before this plan's code (the plan block's own commit: it changes documents only)
FIXED_WORDS = {"role", "init", "true", "does", "legal", "next", "terminal", "goal", "distinct", "not", "record", "noop"}


def gdl():
    from catspace.incubator.analyst.discovery import gdl as G
    return G


def translator():
    from catspace.incubator.analyst.discovery import gdl_to_rust as T
    return T


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def have_before() -> bool:
    return git("cat-file", "-e", f"{BEFORE}^{{commit}}").returncode == 0


def defined_names() -> set:
    """Every relation name and fluent name a description under rules/ defines (a clause's head), less the words of the language."""
    G = gdl()
    names: set = set()
    for path in sorted((ROOT / "rules").rglob("*.gdl")):
        for x in G.parse(path.read_text()):
            if isinstance(x, list) and x and x[0] == "retract":
                continue
            if isinstance(x, list) and x and x[0] in G.DECLARATIONS:
                names.add(x[1])
                continue
            if isinstance(x, list) and x and x[0] == "<=":
                head = x[1]
            else:
                head = x
            if isinstance(head, list) and head and head[0] in G.FLAT and len(head) > 1:
                f = head[1]
                names.add(f[0] if isinstance(f, list) else f)
            else:
                names.add(head[0] if isinstance(head, list) else head)
    return {n for n in names if isinstance(n, str)} - FIXED_WORDS


def test_the_translator_names_no_relation_of_any_description_but_the_fixed_words_of_the_language():
    names = defined_names()
    assert len(names) > 150, len(names)
    assert {"cell", "attack", "slide", "legal"} - FIXED_WORDS <= names
    words = set(re.findall(r"[a-z_][a-z0-9_]*", TRANSLATOR.read_text().lower()))
    hit = sorted(words & names)
    print(f"OBS 79.37 G scan: {len(names)} relation names defined by the descriptions under rules/, {len(words)} words in the translator's source, {len(hit)} in both")
    assert not hit, f"the translator's source holds names of a game's relations: {hit}"


def test_the_scan_sees_a_planted_name():
    names = defined_names()
    planted = "def f():\n    return 'attack_by'\n"
    assert {"attack_by"} <= set(re.findall(r"[a-z_][a-z0-9_]*", planted.lower())) & names


def description_of(module: str) -> str:
    base = (ROOT / "rules" / "chess.gdl").read_text()
    if module == "chess":
        return base
    import cs_engine as CE
    return CE.compose(base, (ROOT / "rules" / "variants" / f"{module}.gdl").read_text())


def head_hash(path: Path, key: str) -> str:
    for line in path.read_text().splitlines()[:6]:
        if line.startswith(f"// {key}:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"{path}: no {key} line at its head")


@pytest.mark.parametrize("module", ["chess", "atomic"])
def test_a_generated_file_carries_the_hashes_of_its_description_and_of_the_translator_and_a_fresh_translation_is_the_same_file(module):
    T = translator()
    path = GEN / f"{module}.rs"
    text = description_of(module)
    assert head_hash(path, "description-sha256") == hashlib.sha256(text.encode()).hexdigest(), f"{module}: the description changed; make the file again"
    assert head_hash(path, "translator-sha256") == T.translator_hash(), f"{module}: the translator changed; make the file again"
    fresh = T.translate(text, module)
    assert fresh == path.read_text(), f"{module}: a fresh translation differs from the committed file"
    print(f"OBS 79.37 G {module}: {len(fresh)} bytes, {fresh.count(chr(10))} lines, the hashes at its head match and a fresh translation is identical")


def test_the_variant_was_written_by_the_same_translator_as_chess():
    assert head_hash(GEN / "chess.rs", "translator-sha256") == head_hash(GEN / "atomic.rs", "translator-sha256") == translator().translator_hash()


def test_the_translator_refuses_what_it_cannot_translate_by_naming_it():
    T = translator()
    with pytest.raises(T.Untranslatable, match="role"):
        T.translate("(a 1)\n", "bad")                                  # a description with no role
    with pytest.raises(T.Untranslatable, match="arguments"):
        T.translate("(role r)\n(a 1)\n(<= (b ?x) (a ?x 2))\n", "bad")  # a relation used with one and with two arguments


def test_nothing_under_the_sources_of_the_engine_names_the_generated_files_the_translator_or_the_helpers():
    bad = []
    for src in sorted(ENGINE.glob("*/src")):
        for f in src.rglob("*"):
            if f.is_file() and f.suffix in (".rs", ".toml"):
                t = f.read_text()
                for w in ("generated/", "gdl_to_rust", "translated_common", "translated_agree", "translated_speed"):
                    if w in t:
                        bad.append(f"{f.relative_to(ROOT)}: {w}")
    assert not bad, bad
    assert not list(ENGINE.glob("*/src/generated")) and not list(ENGINE.glob("*/src/**/generated"))
    assert sorted(p.name for p in GEN.iterdir()) == ["atomic.rs", "chess.rs"]


@pytest.mark.skipif(not have_before(), reason="the commit before the plan is not on this machine")
def test_no_source_of_the_engine_and_no_manifest_differs_from_the_commit_before_the_plan():
    changed = git("diff", "--name-only", BEFORE, "--", "rust").stdout.split()
    srcs = [c for c in changed if re.match(r"rust/engine/[^/]+/src/", c) or c.endswith("Cargo.toml") or c.endswith("Cargo.lock")]
    assert not srcs, srcs
    for protected in ("rust/engine/gdl/src/lib.rs", "rust/engine/game/src/gdl_game.rs", "rust/engine/game/src/kernel.rs", "rust/engine/tactics/src/ledger.rs"):
        assert git("diff", "--quiet", BEFORE, "--", protected).returncode == 0, protected
    print(f"OBS 79.37 G: files under rust/ changed since {BEFORE}: {len(changed)}, none a source of a crate, a manifest or a protected file")


def test_the_gate_lists_this_test_after_the_one_of_plan_79_36():
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert "test_plan79_36 test_plan79_37" in gate
