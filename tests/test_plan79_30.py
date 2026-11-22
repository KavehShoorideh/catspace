"""PLAN 79.30 LAYER 0, MATHEMATICS AND LOGIC, AS ONE CRATE AT THE BOTTOM (SYSTEM_SPEC 11 "Layers 0, 1 and 2 by what they hold", "Layer 0 is wide enough for two unlike games", "Layer 0 holds six things"; ledger row 37; piece: the base under the representation).
Rust holds the unit tests of the six modules and the tests across them (rust/engine/base/src/*, rust/engine/base/tests/layer0.rs). THIS file is what belongs to Python: the scans.
  (1) cs_base's sources (the code, the comments, the strings, the tests) name no game and none of the words of a world;
  (2) cs_base's Cargo file depends on no crate of ours (one outside crate, serde_json, for the JSON form of a literal);
  (3) cs_rep takes the fact, the argument, the literal, differ, the check of a body, the index of facts and the matcher from cs_base and defines none of them itself; its graph reads its chains from cs_base;
  (4) the two fixtures of facts hold what each of the six modules reads, and may name their game;
  (5) every Lean law a Rust test says it mirrors is a theorem of the Lean file the test names, and the Lean files hold no sorry, no enumeration and no axiom;
  (6) the test is in the gate after the one of 79.29.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "rust" / "engine" / "base"
REP = ROOT / "rust" / "engine" / "rep"
LEAN = ROOT / "lean" / "ideas" / "Ideas" / "Layer0"
FIXTURES = BASE / "tests" / "fixtures"

# The words of a game and of a world that layer 0 does not say (whole words, plural too), and a square written as a letter a-h and a digit 1-8.
WORLD_WORDS = (r"boards?|pitch(es)?|balls?|pieces?|units?|tempo|squares?|ranks?|files?|fen|san|uci|checks?|checked|mates?|promotions?|promote[sd]?|chess|kings?|queens?|rooks?|bishops?|knights?|pawns?"
               r"|players?|places?|castl\w*|stalemate|shakmaty|soccer|football|goalkeeper|goals?")
WORLD = re.compile(rf"\b({WORLD_WORDS})\b|\b[a-h][1-8]\b|squar|chess|board", flags=re.I)
MODULES = ["chance", "choice", "logic", "number", "sets", "structure"]


def base_sources() -> list[Path]:
    out = sorted((BASE / "src").rglob("*.rs")) + sorted((BASE / "tests").glob("*.rs")) + [BASE / "Cargo.toml"]
    return out


def test_cs_base_sources_name_no_game_and_no_word_of_a_world():
    files = base_sources()
    names = {p.name for p in files}
    assert {"lib.rs", "mod.rs", "number.rs", "sets.rs", "chance.rs", "lp.rs", "maker.rs", "layer0.rs", "Cargo.toml", "matcher.rs", "formula.rs", "geometry.rs", "graph.rs", "sequence.rs"} <= names
    bad = {}
    for p in files:
        hits = [m.group(0) for m in WORLD.finditer(p.read_text())]
        if hits:
            bad[p.relative_to(ROOT).as_posix()] = sorted(set(hits))
    assert not bad, bad
    print(f"OBS 79.30 scan: {len(files)} sources of cs_base (code, comments, strings, tests) hold none of the world words; the fixtures are the only place a game is named")


def test_cs_base_depends_on_no_crate_of_ours_and_on_no_outside_crate():
    # plan 79.31 part E: the one outside crate (serde_json, for the JSON of a literal) went to cs_rep; cs_base depends on nothing
    toml = (BASE / "Cargo.toml").read_text()
    deps = toml.split("[dependencies]", 1)[1]
    names = re.findall(r"^([A-Za-z0-9_\-]+)\s*=", deps, flags=re.M)
    assert names == [], names
    assert "path" not in deps and "cs_" not in deps, "no crate of ours"
    members = (ROOT / "rust" / "engine" / "Cargo.toml").read_text()
    assert '"base"' in members
    lock = (ROOT / "rust" / "engine" / "Cargo.lock").read_text()
    block = lock.split('name = "cs_base"', 1)[1].split("[[package]]", 1)[0]
    assert "dependencies" not in block, block
    # the layers above read cs_base only through cs_rep; the dependency points down
    assert "cs_base" in (REP / "Cargo.toml").read_text()
    print("OBS 79.30 scan (as changed by 79.31): cs_base's Cargo file lists nothing; the lock gives cs_base no dependency; cs_rep depends on cs_base")


def test_cs_rep_hands_on_the_moved_items_and_defines_none_of_them():
    src = REP / "src"
    defs = [r"pub struct Fact\b", r"pub enum Arg\b", r"pub struct Lit\b", r"pub const DIFFER\b", r"pub fn validate_body\b", r"pub struct FactIndex\b", r"^pub fn bindings\b", r"^pub fn bindings_fused\b",
            r"pub type Binding\b", r"fn unify\b", r"fn differ_says\b"]
    for p in sorted(src.glob("*.rs")):
        t = p.read_text()
        for d in defs:
            assert not re.search(d, t, flags=re.M), (p.name, d)
    assert "pub use cs_base::logic::Fact;" in (src / "fact.rs").read_text()
    assert "pub use cs_base::logic::{validate_body, Arg, Lit, DIFFER};" in (src / "lemma.rs").read_text()
    assert "pub use cs_base::logic::matcher::{bindings, bindings_fused, Binding, FactIndex};" in (src / "matcher.rs").read_text()
    # the same items live in cs_base
    logic = (BASE / "src" / "logic")
    assert re.search(r"pub struct Fact\b", (logic / "mod.rs").read_text()) and re.search(r"pub fn bindings_fused\b", (logic / "matcher.rs").read_text())
    # the graph's pure core is lifted: cs_rep reads its chains from cs_base and no longer holds the loops
    graph = (src / "graph.rs").read_text()
    assert "cs_base::structure::chains" in graph and "found.distance_to_end" in graph and "let mut seen = vec![false; np];" not in graph
    assert re.search(r"pub fn chains\b", (BASE / "src" / "structure" / "graph.rs").read_text())
    print("OBS 79.30 scan: cs_rep re-exports the fact, the argument, the literal, differ, validate_body, the index and the matcher from cs_base and defines none; its graph takes the chains from cs_base")


def test_the_two_fixtures_hold_what_each_module_reads_and_name_their_game():
    games = {}
    for name in ("turns.fixture", "plane.fixture"):
        text = (FIXTURES / name).read_text()
        rels = {l.split("\t")[1] for l in text.splitlines() if l.startswith("fact\t")}
        games[name] = [l.split("\t", 1)[1] for l in text.splitlines() if l.startswith("game\t")][0]
        # logic reads every fact; sets: is_a; number and sequence: seq; structure: link, coord, region; chance: outcome; choice: result, turn or table, chance_node
        assert {"link", "is_a", "coord", "seq", "region", "outcome", "result", "chance_node"} <= rels, (name, rels)
        assert rels & {"turn", "table"}, name
        shapes = {l.split("\t")[3] for l in text.splitlines() if l.startswith("fact\tregion")}
        assert shapes & {"box", "disc", "poly", "half"}, shapes
    assert "chess" in games["turns.fixture"] and "football" in games["plane.fixture"]
    plane = (FIXTURES / "plane.fixture").read_text()
    assert "fact\ttable\t" in plane and "fact\tregion\t" in plane and "poly" in plane and "disc" in plane and "half" in plane, "at once, a region of every kind but one box, an uncertain act"
    turns = (FIXTURES / "turns.fixture").read_text()
    assert "fact\tturn\t" in turns and "fact\tlink\t" in turns
    t = (BASE / "tests" / "layer0.rs").read_text()
    assert "every_one_of_the_six_modules_is_used_by_both_fixtures" in t and all(f'("{m}"' in t for m in MODULES)
    print(f"OBS 79.30 scan: two fixtures ({games['turns.fixture']}; {games['plane.fixture']}) hold the relations of all six modules; the Rust test that runs the six on both is in layer0.rs")


def lean_theorems(module: str) -> set[str]:
    text = (LEAN / f"{module}.lean").read_text()
    return set(re.findall(r"^\s*(?:private\s+)?(?:theorem|def|noncomputable def)\s+([A-Za-z_][A-Za-z0-9_']*)", text, flags=re.M))


def test_every_law_a_rust_test_says_it_mirrors_is_a_theorem_and_the_lean_files_are_clean():
    named = 0
    for p in base_sources():
        if p.suffix != ".rs":
            continue
        for m in re.finditer(r"Ideas\.Layer0\.(\w+)\.(\w+)((?:, \w+)*)", p.read_text()):
            module = m.group(1)
            assert (LEAN / f"{module}.lean").exists(), (p.name, module)
            have = lean_theorems(module)
            for name in [m.group(2)] + [x.strip() for x in m.group(3).split(",") if x.strip()]:
                assert name in have, (p.name, module, name)
                named += 1
    assert named >= 30, named
    for module in ("Logic", "Sets", "Number", "Structure", "Chance", "Choice"):
        text = (LEAN / f"{module}.lean").read_text()
        code = re.sub(r"/-.*?-/", "", text, flags=re.S)                           # comments aside
        assert not re.search(r"\bsorry\b|native_decide|\bdecide\b|^axiom\b|\baxiom\s", code, flags=re.M), module
        assert lean_theorems(module), module
    ideas = (ROOT / "lean" / "ideas" / "Ideas.lean").read_text()
    assert all(f"import Ideas.Layer0.{m}" in ideas for m in ("Logic", "Sets", "Number", "Structure", "Chance", "Choice"))
    # the open laws are said to be open and nothing leans on them: none in this plan
    print(f"OBS 79.30 scan: {named} theorem names in the Rust tests all resolve in lean/ideas/Ideas/Layer0; no sorry, no enumeration, no axiom in the six files")


def test_the_test_is_registered_in_the_gate_after_the_one_of_79_29():
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert gate.index("test_plan79_29") < gate.index("test_plan79_30")
    print("OBS 79.30 scan: the test is in the gate after 79.29")
