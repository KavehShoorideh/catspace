"""PLAN 79.24 LAYER 3, HOW GOOD A POSITION IS: THE VALUE AND THE MEASURE OF PROGRESS BEHIND ONE INTERFACE (SYSTEM_SPEC 11 the layers; ledger rows 35 and 36; piece: value and measure of progress).
Rust holds the test of being done (cargo test --release --offline -p cs_value --test layer3): rust/engine/value/tests/layer3.rs reads rust/engine/value/tests/fixtures/{value,ending}.fixture, made by
catspace/incubator/analyst/discovery/layer3_fixtures_print.py (human games of the test split in four rating bands, generated games that fill the thin endgames, the endgame table's own best play in the two small endings).
THIS file is what belongs to Python: (1) the scans: the layer's sources and its test name no object of any game anywhere (code, comments, strings), no square-like token, and no crate above it; the test calls no search and no outside program;
no decision-path crate depends on the learning crate; (2) the fixtures are small and in the format the test reads, name no player, and print their endgame share; (3) the maker is the one place that names the game and reads the table, and only through the oracle module of the measure;
(4) the negative controls of the scans.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
VALUE = ENGINE / "value"
FIXTURES = VALUE / "tests" / "fixtures"
MAKER = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "layer3_fixtures_print.py"
ORACLE = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "measure_oracle_79_17.py"

LAYER_FILES = sorted((VALUE / "src").glob("*.rs")) + [VALUE / "tests" / "layer3.rs"]
GAME_WORDS = r"\b(king\w*|queen\w*|rook\w*|bishop\w*|knight\w*|pawn\w*|castl\w*|promot\w*|en passant|check\w*|mate|stalemate|uci|fen|san|chess\w*|shakmaty)\b"
SQUARE = r"\b[a-h][1-8]\b"
ABOVE = r"\b(cs_tactics|cs_decision|cs_planning|cs_learn|cs_operator|cs_memory|cs_engine)\b"
SEARCH = r"\b(toward\w*|Search|solve\w*|lowering|question|Budget|ledger|alpha|minimax|Command|spawn)\b"


def words_in(text: str) -> list[str]:
    """every word of a game in a source, comments and strings included (the layer names none anywhere)"""
    return [m.group(0) for m in re.finditer(GAME_WORDS, text, flags=re.I)] + [m.group(0) for m in re.finditer(SQUARE, text)]


def code_of(text: str) -> str:
    """the source without its comments and the text inside strings: a crate is named where it is used"""
    out = []
    for line in text.splitlines():
        line = re.sub(r'"(\\.|[^"\\])*"', '""', line)
        out.append(line.split("//")[0])
    return "\n".join(out)


def test_the_files_of_layer_3_name_no_object_of_any_game_in_code_comments_or_strings():
    names = {p.name for p in LAYER_FILES}
    assert {"lib.rs", "progress.rs", "worth.rs", "layer3.rs"} <= names, names
    for p in LAYER_FILES:
        assert not words_in(p.read_text()), (p.name, words_in(p.read_text()))
    print(f"OBS 79.24 scan: {len(LAYER_FILES)} files of layer 3 (the interface, the measure, the value, the test), no word of a game in code, comments or strings, no square-like token")


def test_layer_3_names_no_crate_above_it_calls_no_search_and_no_outside_program():
    t = (VALUE / "tests" / "layer3.rs").read_text()
    for p in LAYER_FILES:
        assert not re.findall(ABOVE, code_of(p.read_text())), (p.name, re.findall(ABOVE, code_of(p.read_text())))
        assert not re.findall(SEARCH, code_of(p.read_text())), (p.name, re.findall(SEARCH, code_of(p.read_text())))
    cargo = (VALUE / "Cargo.toml").read_text()
    assert not re.search(ABOVE, cargo.split("[dependencies]")[1]), "the layer's manifest names no crate above it in its dependencies"
    deps = re.findall(r"^(cs_\w+)\s*=", cargo, flags=re.M)
    assert sorted(deps) == ["cs_game", "cs_rep"], deps                                 # over layer 1 and layer 2, nothing else
    assert "data/" not in t and "sys.path" not in t and "artifacts" not in t
    # no decision-path crate depends on the learning crate (only the binding, the composition root, does)
    for c in ("decision", "tactics", "planning", "memory", "operator", "value", "rep", "game"):
        assert "cs_learn" not in (ENGINE / c / "Cargo.toml").read_text(), c
    # the crates above layer 3 that call it are the means-ends crate, the planning crate and the decision crate: each lists it
    for c in ("tactics", "planning", "decision"):
        assert "cs_value" in (ENGINE / c / "Cargo.toml").read_text(), c
    print("OBS 79.24 scan: the layer's sources and test name no crate above it, no search and no outside program; its manifest holds cs_game and cs_rep only; no decision-path crate depends on cs_learn")


def test_the_scans_see_what_they_are_for():
    assert words_in('let f = Fact::new("cell", &["a", "1", "w", "rook"]);')
    assert words_in("// the king stands on e1")
    assert words_in('let b1 = "b1";')
    assert not words_in("let b = 3; let units = 4; // a unit moves")
    assert re.findall(ABOVE, code_of("use cs_decision::wake;\nlet a = 1;"))
    assert not re.findall(ABOVE, code_of("// cs_decision is above\nlet a = \"cs_tactics\";"))
    assert re.findall(SEARCH, code_of("let r = toward_memo(g, s);"))
    assert re.findall(SEARCH, code_of("let b = Budget::new(1.0, 10);"))
    assert not re.findall(SEARCH, code_of("// a search is above\nlet x = 1;"))
    for p in LAYER_FILES:
        assert not re.search(r"(from_fen|\bfen\b|_fen\b|\buci\b|_uci\b|\bsan\b)", p.read_text()), p.name


def test_the_fixtures_are_small_in_the_format_the_test_reads_and_name_no_player():
    files = sorted(FIXTURES.glob("*.fixture"))
    assert [p.name for p in files] == ["ending.fixture", "value.fixture"], [p.name for p in files]
    total = sum(p.stat().st_size for p in files)
    assert total < 1_000_000
    tags = {}
    for p in files:
        lines = p.read_text().splitlines()
        assert lines[0].startswith("game\t")
        tags[p.name] = {l.split("\t")[0] for l in lines}
        print(f"OBS 79.24 fixture {p.name}: {p.stat().st_size} bytes, {len(lines)} lines")
        for l in lines:
            assert not re.search(r"\b(elo|rating|player|user|white_name|black_name|site)\b", l, flags=re.I), l[:80]
    assert tags["value.fixture"] == {"game", "model", "sample", "pos"}, tags
    assert tags["ending.fixture"] == {"game", "measure", "line"}, tags
    v = (FIXTURES / "value.fixture").read_text().splitlines()
    samples = [l.split("\t") for l in v if l.startswith("sample\t")]
    kinds = {s[2] for s in samples}
    assert kinds == {"human", "generated"}, kinds
    humans = [s for s in samples if s[2] == "human"]
    assert len(humans) == 4                                                              # four rating bands
    for s in samples:
        print(f"OBS 79.24 fixture sample {s[1]}: {s[3]} games, {s[4]} positions, endgame share {s[6]}")
    pos = [l.split("\t") for l in v if l.startswith("pos\t")]
    assert all(len(x) == 7 and x[5] in ("0", "1", "2") for x in pos)
    # a game number made by the maker, never an id of a player or of a game of the record: small whole numbers
    assert max(int(x[2]) for x in pos) < 10_000
    # the endgame share is printed for every human sample, and a thin human sample is filled by the generated games (a sample of their own)
    thin = [s for s in humans if float(s[6]) < 0.10]
    gen = [s for s in samples if s[2] == "generated"]
    assert gen and int(gen[0][4]) > 0
    print(f"OBS 79.24 fixture: human samples under 0.10 of endgame positions: {[s[1] for s in thin]}; the generated fill holds {gen[0][4]} positions, endgame share {gen[0][6]}")
    e = (FIXTURES / "ending.fixture").read_text().splitlines()
    assert {l.split("\t")[1] for l in e if l.startswith("measure\t")} == {"one", "two"}
    assert {l.split("\t")[1] for l in e if l.startswith("line\t")} == {"one", "two", "twofar"}


def test_the_maker_is_the_one_place_that_names_the_games_and_reads_the_table_only_through_the_oracle_of_the_measure():
    t = MAKER.read_text()
    assert MAKER.name.endswith("_print.py")
    assert "measure_oracle_79_17" in t and "chess.syzygy" not in t and "probe_dtz" not in t and "probe_wdl" not in t, "the table is read only in the oracle module"
    assert not re.search(r"\bsubprocess\b|stockfish|\bwhite_name\b|\bblack_name\b|\bsite\b", t, flags=re.I), "no outside engine, no player's name"
    assert "paths." in t and "Path(\"data" not in t and "'data/" not in t, "Python data paths through catspace.io.paths"
    # no number of an outside engine reaches the value: the maker fits the plain count on results only and reads the stored head fitted on results only
    assert "result_human_K" in t and "LogisticRegression" in t
    users = [p.name for p in (ROOT / "catspace").rglob("*.py") if "measure_oracle_79_17" in p.read_text() and p.name != "measure_oracle_79_17.py"]
    assert set(users) <= {"whole_79_17.py", "measure_miner_79_17.py", "whole_79_17_print.py", "measure_miner_79_17_print.py", "layer3_fixtures_print.py"}, users
    assert "def distance" in ORACLE.read_text()


def test_the_wake_and_the_search_call_layer_3_and_keep_no_second_copy():
    tac = (ENGINE / "tactics" / "src" / "measure.rs").read_text()
    assert "pub use cs_value::progress::*;" in tac
    for name in ("pub struct Geometry", "pub enum Atom", "pub enum Term", "pub struct Reader", "pub fn library", "pub fn value", "pub struct Group"):
        assert name not in tac, f"{name} lives in layer 3 only"
    for name in ("pub fn lowering", "pub fn question"):
        assert name in tac, f"{name} needs a search and stays in the means-ends crate"
    planning = (ENGINE / "planning" / "src" / "lib.rs").read_text()
    assert "cs_value::worth::material" in planning
    wake = (ENGINE / "decision" / "src" / "wake.rs").read_text()
    assert "cs_value::progress::Progress::given" in wake and "Assess" in wake
    print("OBS 79.24 scan: the wake reads the held term through Assess, the search's material count is cs_value::worth::material, cs_tactics::measure re-exports the layer and keeps lowering and question (they need a search)")
