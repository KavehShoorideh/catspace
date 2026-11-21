"""PLAN 79.23 LAYER 2, THE RULES: ITS WORK, ITS INTERFACE AND A TEST THAT SAYS IT IS DONE (SYSTEM_SPEC 11 the layers; ledger row 36; piece: rules).
Rust holds the test (cargo test --release --offline -p cs_game --test layer2): rust/engine/game/tests/layer2.rs reads rust/engine/game/tests/fixtures/*.fixture, made once by catspace/incubator/analyst/discovery/layer2_fixtures_print.py from the python
library `chess` (a reference that shares no code with the engine), for the standard game and both variants of the repo.
THIS file is what belongs to Python: (1) the scans: the layer's sources and its test name no object of any game and its test names no crate above it; (2) the fixtures are small, are the files the maker makes now, and are in the format the test reads;
(3) the negative controls of the scans.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAME = ROOT / "rust" / "engine" / "game"
FIXTURES = GAME / "tests" / "fixtures"
MAKER = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "layer2_fixtures_print.py"

# the files of layer 2 that are the rules' interface and its test, and so name no game: what plan 79.23 wrote and what it moved down from cs_tactics
LAYER_FILES = [GAME / "src" / n for n in ("rules.rs", "graph.rs", "keeper.rs")] + [GAME / "tests" / "layer2.rs"]
GAME_WORDS = r"\b(king\w*|queen\w*|rook\w*|bishop\w*|knight\w*|pawn\w*|castl\w*|promot\w*|en passant|check\w*|mate|stalemate|uci|fen|san|chess\w*|shakmaty)\b"
SQUARE = r"\b[a-h][1-8]\b"
ABOVE = r"\b(cs_tactics|cs_decision|cs_planning|cs_learn|cs_operator|cs_memory|cs_engine)\b"


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


def test_the_files_of_layer_2_name_no_object_of_any_game_in_code_comments_or_strings():
    for p in LAYER_FILES:
        assert p.exists(), p
        assert not words_in(p.read_text()), (p.name, words_in(p.read_text()))
    print(f"OBS 79.23 scan: {len(LAYER_FILES)} files of layer 2 (the interface, the graph and its keeper moved down, the test), no word of a game in code, comments or strings")


def test_the_test_of_layer_2_names_no_crate_above_it_and_calls_no_outside_program():
    t = (GAME / "tests" / "layer2.rs").read_text()
    assert not re.findall(ABOVE, code_of(t)), re.findall(ABOVE, code_of(t))
    assert not re.search(r"\b(Command|spawn|process::)\b", code_of(t))
    assert not re.search(r"\b(cs_tactics|cs_decision|cs_planning|cs_learn|cs_operator|cs_memory|cs_engine)\b", (GAME / "Cargo.toml").read_text())
    for n in ("rules.rs", "graph.rs", "keeper.rs"):                                # the layer's own sources read nothing above either
        assert not re.findall(ABOVE, code_of((GAME / "src" / n).read_text())), n
    # the test reads only the fixtures it is given and the rules' files, never a path of the repo's data
    assert "data/" not in t and "sys.path" not in t


def test_the_scan_sees_what_it_is_for():
    assert words_in("let x = Fact::new(\"cell\", &[\"a\", \"1\", \"white\", \"rook\"]);")
    assert words_in("// the king stands on e1")
    assert words_in("let t = \"e2e4\"; let b1 = \"b1\";")
    assert not words_in("let b = 3; let units = 4; // a unit moves")
    assert re.findall(ABOVE, code_of("use cs_decision::wake;\nlet a = 1;"))
    assert not re.findall(ABOVE, code_of("// cs_decision is above\nlet a = \"cs_tactics\";"))
    for p in LAYER_FILES:
        assert not re.search(r"(from_fen|\bfen\b|_fen\b|\buci\b|_uci\b|\bsan\b)", p.read_text()), p.name


def test_the_fixtures_are_the_three_games_small_and_the_files_the_maker_makes_now(tmp_path):
    files = sorted(FIXTURES.glob("*.fixture"))
    assert len(files) == 3
    total = sum(p.stat().st_size for p in files)
    assert total < 1_000_000
    tags = set()
    for p in files:
        lines = p.read_text().splitlines()
        assert lines[0].startswith("game\t") and lines[1].startswith("ends\t")
        tags |= {l.split("\t")[0] for l in lines}
        print(f"OBS 79.23 fixture {p.name}: {p.stat().st_size} bytes, {len(lines)} lines")
    assert tags == {"game", "ends", "perft", "pos", "step", "line", "record"}, tags
    # the maker, run again, writes the same bytes: a fixture is never edited by hand
    r = subprocess.run([sys.executable, "-m", "catspace.incubator.analyst.discovery.layer2_fixtures_print", str(tmp_path)], cwd=ROOT, capture_output=True, text=True, env={"PYTHONHASHSEED": "0", "PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    for p in files:
        assert (tmp_path / p.name).read_text() == p.read_text(), p.name
    print(f"OBS 79.23 fixtures total {total} bytes; the maker makes the same files again")


def test_the_maker_is_the_one_place_that_names_the_games_and_uses_the_reference_library():
    t = MAKER.read_text()
    assert "import chess" in t and "chess.variant" in t
    assert not re.search(r"\bcs_engine\b|\bshakmaty\b|\bsubprocess\b", t), "the reference shares no code with the engine"
    assert MAKER.name.endswith("_print.py")
