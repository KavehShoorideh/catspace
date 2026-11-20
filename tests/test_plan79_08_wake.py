"""PLAN 79.08 THE WAKE'S LOOP, THE WHOLE IN RUST (SYSTEM_SPEC 8 all seven steps and the rule for choosing a move, 7 calculation and estimate, 4, 11; piece: the wake; rust/engine/decision/src/{wake,rule,log,carry,estimate,frame_rows,pick}.rs, rust/engine/tactics/src/sheet/threat_frames.rs).
Every piece is unit-tested in Rust (cargo test --release --offline -p cs_decision --test wake_repairs --test wake_pieces --test wake_rule --test wake_whole; -p cs_tactics --test named_ply) and the whole is integration-tested there. THIS file holds what belongs to Python: the wall (no game object named in the
sources, no oracle, no cs_learn in the decision path), the old player unchanged (the binding with no piece on is the search as it was), and the binding of the whole wake. No puzzle, no stored game, no match. This file names no square of the game."""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import cs_engine as CE
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "rust" / "engine" / "decision"
TACTICS = ROOT / "rust" / "engine" / "tactics"
FILES = [DECISION / "src" / n for n in ("wake.rs", "rule.rs", "log.rs", "carry.rs", "estimate.rs", "frame_rows.rs", "pick.rs", "goals.rs", "threats.rs")] + [TACTICS / "src" / "sheet" / "threat_frames.rs", TACTICS / "src" / "lines.rs", TACTICS / "src" / "ledger.rs"]
BINDING = ROOT / "rust" / "engine" / "py" / "src" / "base_graph_py.rs"
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")
WORTH = {"p": 475, "n": 1216, "b": 1301, "r": 1945, "q": 3583}
LINE = "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1"
FORCED = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"


def _text(fen):
    return CE.state("chess", fen, [])[0]


def test_the_wall_the_new_sources_name_no_game_object_and_reach_for_no_oracle():
    for p in FILES[:7] + [FILES[7], FILES[8], FILES[9]]:
        text = p.read_text()
        low = text.lower()
        assert not re.search(r"\b[a-h][1-8]\b", low), (p.name, re.search(r"\b[a-h][1-8]\b", low).group(0))
        if p.name in ("wake.rs", "rule.rs", "log.rs", "carry.rs", "estimate.rs", "frame_rows.rs", "pick.rs", "threat_frames.rs"):
            for w in BANNED:
                assert not re.search(rf"\b{w}\b", low), (p.name, w)
        assert not re.search(r"ladder_oracle|stockfish|cs_learn|analyse|reqwest|anthropic", low), p.name
        assert "rules/learned" not in low and "data/derived" not in low, p.name
    cargo = tomllib.loads((DECISION / "Cargo.toml").read_text())
    assert "cs_learn" not in cargo.get("dependencies", {}), "no decision-path crate depends on cs_learn"


def test_the_estimates_come_from_parameters_and_the_heads_over_whole_position_counts_are_not_called():
    text = (DECISION / "src" / "wake.rs").read_text()
    assert "heads_read" not in text and "heads_pick" not in text
    assert "not ported, not called" in text
    est = (DECISION / "src" / "estimate.rs").read_text()
    assert "PARAMETER" in est and "NEVER a value added to the judge" in est


def test_nothing_new_is_on_by_default():
    text = (DECISION / "src" / "wake.rs").read_text()
    assert "pieces: Pieces::old_player()" in text
    assert "pub named_ply: FxHashMap<i32, Vec<String>>" in (TACTICS / "src" / "ledger.rs").read_text()
    assert "share_table: false" in (DECISION / "src" / "threats.rs").read_text()
    lib = (DECISION / "src" / "lib.rs").read_text()
    assert "named: vec![] }" in lib and "pub mod rule;" in lib


def test_the_binding_with_no_piece_on_is_the_old_player_move_and_value():
    kw = dict(worth=WORTH, ours_first=False, valuation="odds record", threads=1, ledger_share=1.0)
    old = CE.solve_equals_batch([("chess", LINE, [])], 5.0, ledger_depth=4, **kw)[0]
    model = GL.model("chess", "chess")
    goal = G2M.goal_kind_of("chess")
    out = json.loads(model.wake_decide(_text(LINE), [], 5.0, 4, WORTH, {}, goal, [], 0, 0, None, None, None, None))
    assert out["move"] == old[0], "the move is the old player's"
    assert out["value"] == old[8] and out["depth"] == old[7], "and the value and depth of its line"
    assert out["unported"] == [] and out["log"]["moves"] == [] and len(out["steps"]) == 1, "no piece, no step but the search"


def test_the_binding_of_the_whole_wake_returns_a_move_a_chain_a_log_and_the_time_of_each_step():
    model = GL.model("chess", "chess")
    goal = G2M.goal_kind_of("chess")
    pieces = ["threats", "goals", "two_threat_goals", "reasons_pick"]
    out = json.loads(model.wake_decide(_text(LINE), [], 5.0, 3, WORTH, {}, goal, pieces, 4, 0, None, None, ([], [], (100, 25), 20), None))
    assert out["move"] and out["chain"][0]["by"] == "search"
    log = out["log"]
    assert log["chosen"] == out["move"] and log["goals"] and len(log["moves"]) > 5
    assert any(m["chance"] is not None for m in log["moves"] if m["serves"]) or not any(m["serves"] for m in log["moves"])
    assert any("no reasons given" in n for n in log["notes"]), "a piece asked for with nothing to read says so"
    assert {s["step"] for s in out["steps"]} >= {3, 4, 5, 6}
    try:
        model.wake_decide(_text(LINE), [], 1.0, 3, WORTH, {}, goal, ["nonsense"], 0, 0, None, None, None, None)
        raise AssertionError("an unknown piece must be refused")
    except ValueError:
        pass
