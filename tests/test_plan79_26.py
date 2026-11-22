"""PLAN 79.26 LAYER 4'S TWO FAULTS MENDED, MEASURED BY ITS OWN TEST AND JUDGED AS THE WHOLE (SYSTEM_SPEC 11 the layers, 8 the wake; ledger row 36; piece: tactical reasoner).
Rust holds the tests (cargo test --release --offline -p cs_tactics --test layer4 --test layer3_one_copy): layer4.rs reads the layer's fixtures with TWO judges (the plain count and the judge the wake plays with, from worth.fixture), and
with the deepening stopping on an end proved whatever was given up (before.fixture holds the nodes the build before that needed). THIS file is what belongs to Python: the scans of the new data and its makers, and that the layer's files still name no word of a game.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TACTICS = ROOT / "rust" / "engine" / "tactics"
FIXTURES = TACTICS / "tests" / "fixtures"
DISCOVERY = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
LAYER_FILES = [TACTICS / "src" / "meansends.rs", TACTICS / "tests" / "layer4.rs"]
GAME_WORDS = r"\b(king\w*|queen\w*|rook\w*|bishop\w*|knight\w*|pawn\w*|castl\w*|promot\w*|en passant|check\w*|mate|stalemate|uci|fen|san|chess\w*|shakmaty)\b"
SQUARE = r"\b[a-h][1-8]\b"


def test_the_worth_fixture_holds_the_records_worths_and_names_no_player():
    t = (FIXTURES / "worth.fixture").read_text().splitlines()
    assert t[0].startswith("note\t")
    rows = [l.split("\t") for l in t[1:]]
    assert len(rows) >= 3 and all(r[0] == "worth" and len(r) == 3 and r[2].lstrip("-").isdigit() for r in rows), rows
    assert not any(re.search(r"\b(elo|rating|player|user|site)\b", l, flags=re.I) for l in t)
    print(f"OBS 79.26 scan: worth.fixture holds {len(rows)} worths of kinds, read from the human-result record, none typed in a layer")


def test_the_before_fixture_is_the_nodes_of_proved_ends_in_the_format_the_test_reads():
    t = (FIXTURES / "before.fixture").read_text().splitlines()
    assert t[0].startswith("note\t")
    rows = [l.split("\t") for l in t[1:]]
    assert len(rows) >= 300 and all(r[0] == "before" and len(r) == 6 and r[1] in ("plain", "wake") and r[3] in "012" and r[4].isdigit() and r[5] in ("Won", "Lost") for r in rows)
    assert {r[1] for r in rows} == {"plain", "wake"}
    print(f"OBS 79.26 scan: before.fixture holds {len(rows)} proved ends with the nodes the build before part B needed, for both judges")


def test_the_layer_files_still_name_no_word_of_a_game_and_the_test_uses_both_judges_and_the_before_nodes():
    for p in LAYER_FILES:
        txt = p.read_text()
        bad = [m.group(0) for m in re.finditer(GAME_WORDS, txt, flags=re.I)] + [m.group(0) for m in re.finditer(SQUARE, txt)]
        assert not bad, (p.name, bad)
    t = (TACTICS / "tests" / "layer4.rs").read_text()
    assert "Which::Wake" in t and "Which::Plain" in t and "worth.fixture" in t
    assert "seconds: None" in t and "seconds: Some" not in t, "still nodes only"
    print("OBS 79.26 scan: layer 4's entry and test name no word of a game; the test reads both judges and the nodes of before")


def test_the_worth_maker_names_the_game_only_here_and_fits_nothing():
    t = (DISCOVERY / "layer4_worth_print.py").read_text()
    assert (DISCOVERY / "layer4_worth_print.py").name.endswith("_print.py")
    assert "record_play" in t and "worth_of" in t
    assert not re.search(r"sklearn|\.fit\(|torch|urlopen|requests|anthropic|stockfish", t, flags=re.I)
