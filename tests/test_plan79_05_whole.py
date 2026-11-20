"""PLAN 79.05 THE WHOLE WAKE (SYSTEM_SPEC 8 "the whole is judged, not the piece", 11 Rust in the hot paths). The wake is a Rust module (rust/engine/decision/src/wake.rs); its pieces are ported one at a time and unit-tested in Rust (cargo test --release --offline -p cs_decision
--test wake_graph, wake_old_player, threat_verdict; -p cs_tactics --test named_first). THIS file holds the checks that belong to Python: the wall (no oracle, no cs_learn in the decision path), the old player unchanged with the new parameter empty (through the binding), and ONE binding
test of the ported graph reads. No puzzle, no stored game, no match. This file names no square of the game."""
import re
import tomllib
from pathlib import Path

import pytest

import cs_engine as CE

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "rust" / "engine" / "decision"
WAKE_FILES = [DECISION / "src" / "wake.rs", DECISION / "src" / "threat_verdict.rs"]
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")
LAYER = ("wake.rs", "threat_verdict.rs")


def test_the_wake_sources_name_no_game_object():
    for p in WAKE_FILES:
        text = p.read_text().lower()
        assert not re.search(r"\b[a-h][1-8]\b", text), p.name
        for w in BANNED:
            assert not re.search(rf"\b{w}\b", text), (p.name, w)


def test_the_wall_no_oracle_and_no_learn_crate_in_the_decision_path():
    cargo = tomllib.loads((DECISION / "Cargo.toml").read_text())
    assert "cs_learn" not in cargo.get("dependencies", {}), "no decision-path crate depends on cs_learn"
    for p in WAKE_FILES:
        text = p.read_text().lower()
        assert not re.search(r"ladder_oracle|stockfish|cs_learn|analyse|reqwest|anthropic", text), p.name


def test_the_new_parameter_is_empty_by_default_in_the_rust_sources():
    text = (DECISION / "src" / "lib.rs").read_text()
    assert "pub named: Vec<(u64, Vec<String>)>" in text and "named: vec![] }" in text
    assert "pub mod wake;" in text and "pub mod threat_verdict;" in text


def _text(fen):
    return CE.state("chess", fen, [])[0]


FEN = "1k6/2p5/8/8/5B2/8/8/K1R5 w - - 0 1"


def test_the_old_player_is_unchanged_with_the_named_moves_empty_or_unmatched():
    worth = {"p": 475, "n": 1216, "b": 1301, "r": 1945, "q": 3583}
    kw = dict(worth=worth, ours_first=False, valuation="odds record", threads=1, ledger_share=0.8)
    # a node-bound comparison is not available through the binding, so the decision at a depth the clock reaches in full is compared: the move, the value, the depth, the equal set
    a = CE.solve_equals_batch([("chess", FEN, [])], 1.0, **kw)[0]
    b = CE.solve_equals_batch([("chess", FEN, [])], 1.0, named_moves=[], **kw)[0]
    c = CE.solve_equals_batch([("chess", FEN, [])], 1.0, named_moves=[(1234567, ["nothing"])], **kw)[0]
    for r in (b, c):
        assert (r[0], r[5], r[8], r[10]) == (a[0], a[5], a[8], a[10]), "move, equal set, value and legal count are the old player's"


def test_the_binding_reads_the_lines_onto_the_goal_unit_and_the_moves_that_bring_a_bearer():
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
    model = GL.model("chess", "chess")
    goal = G2M.goal_kind_of("chess")
    text = _text(FEN)
    out = model.wake_graph_read(text, goal, [])
    assert out["new_bearer_moves"] == []
    rows = {owner: (place, lines) for owner, place, lines in out["lines"]}
    assert len(rows) == 2
    shut = [ln for _p, lines in rows.values() if lines for ln in lines]
    assert len(shut) == 1 and (len(shut[0]["bearers"]), len(shut[0]["holders"])) == (2, 1), "one line, shut by one unit; two bear on it, one holds it"
    some_place = shut[0]["blocker"][2]
    out2 = model.wake_graph_read(text, goal, [some_place])
    assert isinstance(out2["new_bearer_moves"], list)
