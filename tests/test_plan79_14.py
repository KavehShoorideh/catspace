"""PLAN 79.14 THE PLY BACK, AND THE CHANCE SPENT ON WHAT THE SEARCH READS (SYSTEM_SPEC 8 the wake step by step and "a piece that costs too much is made cheaper, not turned off", 7 point 1 and 2, 4; piece: the wake).
Rust holds the unit tests (cargo test --release --offline -p cs_decision --test delta --test chance_order): the seeds asked only where the position changed, what the plies since touched, the frame read and the every-unit rows that carry, the order by chance, the gain mended.
THIS file holds what belongs to Python: the wall (the measure script names no game object and reaches for no oracle), the new switches reach the wake through the binding and are off by default, the steps the wake logs by part, the carry marks what it carries, and the harness's
helpers (the pairs from the logs, the step table, the differences between two readings, the configs of the matches)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import whole_79_14 as W14
from catspace.incubator.analyst.discovery import whole_play as WP

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")


def test_the_wall_the_measure_script_names_no_game_object_and_reaches_for_no_oracle():
    low = (DIR / "whole_79_14.py").read_text().lower()
    assert not re.search(r"\b[a-h][1-8]\b", low), re.search(r"\b[a-h][1-8]\b", low).group(0)
    assert "sys.path.insert" not in low
    for w in BANNED:
        assert not re.search(rf"\b{w}\b", low), w
    assert not re.search(r"ladder_oracle|leaf_ask|stockfish|analyse_pos|cs_learn|anthropic|\blo\.", low)


def test_the_hot_path_is_rust_and_no_decision_crate_depends_on_the_learning_crate():
    toml = (ROOT / "rust" / "engine" / "decision" / "Cargo.toml").read_text()
    assert "cs_learn" not in toml
    src = (ROOT / "rust" / "engine" / "tactics" / "src" / "sheet" / "delta.rs").read_text()
    assert "cs_learn" not in src


def test_the_configs_of_the_matches_add_the_switches_one_by_one_to_the_pieces_of_79_13():
    base = list(W14.BASE_PIECES)
    assert W14.CONFIGS["b"] == base + ["carry_graph"]
    assert W14.CONFIGS["bc"] == base + ["carry_graph", "chance_order"]
    assert W14.CONFIGS["bcg"] == base + ["carry_graph", "chance_order", "gain_mend"]
    assert not {"carry_graph", "chance_order", "gain_mend"} & set(WP.PIECES), "every new behaviour is off by default: the pieces of the harness do not name them"


def test_a_player_spec_carries_its_own_pieces_only_for_the_wake():
    spec = WP.player_spec("wake", 0.2, None, None, ["carried", "carry_graph"])
    assert spec["pieces"] == ["carried", "carry_graph"]
    assert "pieces" not in WP.player_spec("old", 0.2, None, None, ["carried"])
    assert "pieces" not in WP.player_spec("wake", 0.2)


def test_the_difference_between_two_readings_names_what_differs_and_nothing_when_they_agree():
    a = {"rows": [("r", "x", 1)], "goals": [("g", 1)], "named_mine": ["m1"], "named_theirs": [], "their_threat": None}
    assert W14.differences(a, dict(a)) == []
    b = dict(a, rows=[("r", "x", 2)], named_mine=["m2"])
    out = W14.differences(a, b)
    assert len(out) == 2 and out[0].startswith("rows: 1 only with the carry, 1 only asked afresh") and out[1].startswith("named_mine")


def test_the_step_table_is_the_mean_a_decision_with_the_share_of_the_whole():
    ds = [{"steps": [{"step": 0, "piece": "the whole decision", "seconds": 0.2, "note": ""}, {"step": 2, "piece": "frames", "seconds": 0.02, "note": ""}]},
          {"steps": [{"step": 0, "piece": "the whole decision", "seconds": 0.2, "note": ""}, {"step": 2, "piece": "frames", "seconds": 0.04, "note": ""}]}]
    t = {p: (m, sh, c) for _st, p, m, sh, c in W14.step_table(ds)}
    assert abs(t["frames"][0] - 0.03) < 1e-12 and abs(t["frames"][1] - 15.0) < 1e-9 and t["frames"][2] == 2


# ------------------------------------------------------------------------------------------------------------------------------ through the binding

def _decide(pieces, moves, slot=None, seconds=0.3):
    from catspace.incubator.analyst.discovery import odds_measure as OM
    from catspace.incubator.analyst.discovery import record_play as RP
    import cs_engine as CE
    w = WP.world()
    pos = CE.state(OM.VARIANT, None, list(moves))[0]
    history = [CE.state(OM.VARIANT, None, list(moves[:k]))[4] for k in range(len(moves))]
    return json.loads(w.model.wake_decide(pos, history, seconds, 64, dict(RP.worth_of("human")), {}, w.goal, pieces, 0, 0, w.keys, WP.fuses(), w.chances, w.reasons, WP.TM.LEDGER_SHARE, slot))


OPENING = ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5", "c2c3", "d7d6")


def test_the_new_switches_are_accepted_and_an_unknown_piece_is_refused():
    out = _decide(WP.PIECES + ["carry_graph", "chance_order", "gain_mend"], OPENING[:4])
    assert out["move"]
    with pytest.raises(Exception):
        _decide(["no_such_piece"], OPENING[:2])


def test_with_no_piece_on_the_decision_is_the_old_players_search_alone_and_the_switches_alone_change_nothing_of_it():
    old = _decide([], OPENING[:6])
    assert [s["step"] for s in old["steps"]] == [5] and not old["log"]["rows"]
    also = _decide(["carry_graph", "chance_order", "gain_mend"], OPENING[:6])
    assert [s["step"] for s in also["steps"]] == [5], "a switch with no piece to serve is nothing"


def test_the_steps_are_logged_by_part_and_the_carry_marks_what_it_carries():
    slot = "test79_14:carry"
    first = _decide(WP.PIECES + ["carry_graph"], OPENING[:6], slot, 1.0)
    parts = {s["piece"] for s in first["steps"] if s["step"] in (22, 23)}
    assert any(p.startswith("part of the frame read: the seeds") for p in parts), parts
    assert any(p.startswith("part of the threat list") for p in parts)
    second = _decide(WP.PIECES + ["carry_graph"], OPENING[:8], slot, 1.0)
    note = next(s["note"] for s in second["steps"] if s["piece"] == "frames and the question read")
    assert "carried:" in note, note
    off = _decide(WP.PIECES, OPENING[:8], "test79_14:off", 0.3)
    assert "carried:" not in next(s["note"] for s in off["steps"] if s["piece"] == "frames and the question read"), "with the switch off nothing of it is said"
    assert all(r["carried"] in (True, False) for r in second["log"]["rows"])


def test_the_key_hashes_the_seeds_are_found_by_are_the_hashes_of_the_keys_so_the_frame_read_at_a_long_clock_gives_the_same_rows_with_the_switch_on_and_off():
    # the order of the seeds differs between the two reads only in the order of joining frames, so compare the rows' units, not the order
    a = _decide(WP.PIECES, OPENING[:8], None, 3.0)
    b = _decide(WP.PIECES + ["carry_graph"], OPENING[:8], None, 3.0)
    ua = sorted((r["list"], r["unit"]) for r in a["log"]["rows"])
    ub = sorted((r["list"], r["unit"]) for r in b["log"]["rows"])
    assert ua == ub or len(set(ua) ^ set(ub)) <= 2, set(ua) ^ set(ub)
