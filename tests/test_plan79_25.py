"""PLAN 79.25 LAYER 4, MEANS-ENDS: THE SEARCH THAT ADVANCES THE VALUE, ITS INTERFACE AND A TEST ON POSITIONS THAT ARE NOT QUIET (SYSTEM_SPEC 11 the layers; ledger row 36; piece: tactical reasoner).
Rust holds the test of being done (cargo test --release --offline -p cs_tactics --test layer4): rust/engine/tactics/tests/layer4.rs reads rust/engine/tactics/tests/fixtures/{puzzle,quiet}.fixture, made by
catspace/incubator/analyst/discovery/layer4_fixtures_print.py (puzzles of the set the earlier plans graded on, an outside engine as a measure for the stored verdicts, the rule of "not quiet" stated in the fixture).
THIS file is what belongs to Python: (1) the scans: the layer's entry and its test name no object of any game anywhere (code, comments, strings), no square-like token, no crate above layer 4, no outside program, no clock that decides;
the entry reaches no module of cs_tactics above the search (goals, frames, techniques, the sheet, the measure's search); (2) the fixtures are small, in the format the test reads, and name no player; (3) the maker is the one place that names the game and the
outside engine, and it fits nothing; (4) the negative controls of the scans.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
TACTICS = ENGINE / "tactics"
FIXTURES = TACTICS / "tests" / "fixtures"
MAKER = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "layer4_fixtures_print.py"

# layer 4's own files (the cs_tactics sources before layer 4 are a mixed crate: the part C list of the plan's block says which of them belong above)
LAYER_FILES = [TACTICS / "src" / "meansends.rs", TACTICS / "tests" / "layer4.rs"]
GAME_WORDS = r"\b(king\w*|queen\w*|rook\w*|bishop\w*|knight\w*|pawn\w*|castl\w*|promot\w*|en passant|check\w*|mate|stalemate|uci|fen|san|chess\w*|shakmaty)\b"
SQUARE = r"\b[a-h][1-8]\b"
ABOVE = r"\b(cs_decision|cs_planning|cs_learn|cs_memory|cs_engine)\b"
# modules of cs_tactics that are not the search (they read goals, frames, techniques, the sheet, the measure's search, a toward search): the entry names none of them
NOT_THE_SEARCH = r"\b(sheet|scenario|measure|toward|reason|census|operators|lines|choice_audit|equal_audit|gap_audit)\b"


def words_in(text: str) -> list[str]:
    return [m.group(0) for m in re.finditer(GAME_WORDS, text, flags=re.I)] + [m.group(0) for m in re.finditer(SQUARE, text)]


def code_of(text: str) -> str:
    out = []
    for line in text.splitlines():
        line = re.sub(r'"(\\.|[^"\\])*"', '""', line)
        out.append(line.split("//")[0])
    return "\n".join(out)


def test_the_files_of_layer_4_name_no_object_of_any_game_in_code_comments_or_strings():
    for p in LAYER_FILES:
        assert p.exists(), p
        assert not words_in(p.read_text()), (p.name, words_in(p.read_text()))
    print(f"OBS 79.25 scan: {len(LAYER_FILES)} files of layer 4 (the entry, the test), no word of a game in code, comments or strings, no square-like token")


def test_layer_4_names_no_crate_above_it_calls_no_outside_program_and_lets_no_clock_decide():
    for p in LAYER_FILES:
        c = code_of(p.read_text())
        assert not re.findall(ABOVE, c), (p.name, re.findall(ABOVE, c))
        assert not re.search(r"std::process|Command::new|\bspawn\(.*\b(engine|stockfish)\b", c), p.name
    t = (TACTICS / "tests" / "layer4.rs").read_text()
    assert "data/" not in t and "sys.path" not in t and "artifacts" not in t and "stockfish" not in t.lower()
    assert "seconds: None" in t and "seconds: Some" not in t, "the test asks by nodes only"
    # the manifest: the crate's dependencies are layers 1 to 3 and the operator interface; nothing of the wake, the planning or the learning
    cargo = (TACTICS / "Cargo.toml").read_text().split("[dependencies]")[1]
    deps = sorted(re.findall(r"^(cs_\w+)\s*=", cargo, flags=re.M))
    assert deps == ["cs_game", "cs_operator", "cs_rep", "cs_value"], deps
    # the entry is read by the test through the one door
    assert "cs_tactics::meansends::{search, Ask, Ending, Found}" in t
    print("OBS 79.25 scan: the layer's entry and test name no crate above it, no outside program, no clock that decides; the manifest of cs_tactics holds cs_game, cs_operator, cs_rep, cs_value only")


def test_the_entry_reaches_only_the_search_and_no_module_of_a_layer_above():
    src = (TACTICS / "src" / "meansends.rs").read_text()
    code = code_of(src)
    uses = re.findall(r"^\s*use ([\w:{}, ]+);", code, flags=re.M)
    crate_uses = [u for u in uses if u.startswith("crate::")]
    assert crate_uses == ["crate::ledger::{Judge, Search, PROVED}", "crate::reach"], crate_uses  # plan 79.27: the entry with goals also reaches the statements' body
    assert not re.search(NOT_THE_SEARCH, "\n".join(uses)), uses
    assert "pub fn search<" in src and "pub fn ask<" in src and src.count("pub fn ") == 5, "the reading of the old goals, its ending, and (plan 79.27) the entry with goals: ask, ask_with, Answer::reached"
    print("OBS 79.25 scan: the entry uses the ledger search and nothing else of cs_tactics")


def test_the_scans_see_what_they_are_for():
    assert words_in('let f = Fact::new("cell", &["a", "1", "w", "rook"]);')
    assert words_in("// the king stands on e1")
    assert words_in('let b1 = "b1";')
    assert words_in("// to check the line")
    assert not words_in("let b = 3; let units = 4; // a unit moves")
    assert re.findall(ABOVE, code_of("use cs_decision::wake;\nlet a = 1;"))
    assert not re.findall(ABOVE, code_of("// cs_decision is above\nlet a = \"cs_planning\";"))
    assert re.search(NOT_THE_SEARCH, "crate::toward::toward_memo")
    assert not re.search(NOT_THE_SEARCH, "crate::ledger::Search")
    for p in LAYER_FILES:
        assert not re.search(r"(from_fen|\bfen\b|_fen\b|\buci\b|_uci\b|\bsan\b)", p.read_text()), p.name


def test_the_fixtures_are_small_in_the_format_the_test_reads_and_name_no_player():
    files = sorted(FIXTURES.glob("*.fixture"))
    assert [p.name for p in files] == ["before.fixture", "goals.fixture", "pairs.fixture", "puzzle.fixture", "quiet.fixture", "ratchet.fixture", "worth.fixture"], [p.name for p in files]  # plan 79.26 added before.fixture and worth.fixture, plan 79.27 goals.fixture and pairs.fixture
    total = sum(p.stat().st_size for p in files)
    assert total < 500_000
    tags = {}
    for p in files:
        lines = p.read_text().splitlines()
        assert lines[0].startswith("game\t") or p.name in ("ratchet.fixture", "before.fixture", "worth.fixture", "goals.fixture", "pairs.fixture")
        tags[p.name] = {l.split("\t")[0] for l in lines}
        print(f"OBS 79.25 fixture {p.name}: {p.stat().st_size} bytes, {len(lines)} lines")
        for l in lines:
            assert not re.search(r"\b(elo|rating|player|user|white_name|black_name|site)\b", l, flags=re.I), l[:80]
    assert tags["puzzle.fixture"] == {"game", "rule", "puzzle"}, tags
    assert tags["quiet.fixture"] == {"game", "rule", "quiet"}, tags
    assert tags["ratchet.fixture"] == {"note", "floor", "wake_floor"}, tags
    pz = [l.split("\t") for l in (FIXTURES / "puzzle.fixture").read_text().splitlines() if l.startswith("puzzle\t")]
    assert len(pz) >= 300 and all(len(x) == 8 for x in pz)
    assert all(int(x[3]) == (len(x[7].split()) + 1) // 2 for x in pz), "our moves is read off the solution"
    assert all(int(x[5]) >= 150 for x in pz), "the stated margin of the rule"
    q = [l.split("\t") for l in (FIXTURES / "quiet.fixture").read_text().splitlines() if l.startswith("quiet\t")]
    assert len(q) >= 40 and all(len(x) == 5 and abs(int(x[2])) <= 100 and int(x[3]) < 60 for x in q)
    print(f"OBS 79.25 fixture: {len(pz)} positions that are not quiet, {len(q)} quiet positions, {total} bytes in all")


def test_the_maker_is_the_one_place_that_names_the_game_and_the_outside_engine_and_fits_nothing():
    t = MAKER.read_text()
    assert MAKER.name.endswith("_print.py")
    assert "from catspace.incubator.analyst.engine import" in t, "the engine the repo already drives (the analyst adapter), not a new one"
    assert "ledger_puzzles" in t, "the puzzle set and the reader the earlier plans graded on"
    assert not re.search(r"sklearn|LogisticRegression|torch|\.fit\(", t), "the outside engine's numbers are stored, never fitted on"
    assert not re.search(r"\bwhite_name\b|\bblack_name\b|\bsite\b|urlopen|requests|anthropic", t, flags=re.I), "no player's name, no download, no outside model"
    assert "paths." in t and "Path(\"data" not in t and "'data/" not in t, "Python data paths through catspace.io.paths"
    assert "R1" in t and "R2" in t and "MARGIN" in t, "the rule of not quiet is stated in the maker"
    # the search never reads the fixture's engine numbers: the layer's entry takes no reference
    src = (TACTICS / "src" / "meansends.rs").read_text()
    assert "reference" not in code_of(src) and "stockfish" not in src.lower()
