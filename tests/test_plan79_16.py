"""PLAN 79.16 THE SEARCH DOES NOT STOP ON A FAILED FIRST MOVE; THE WHOLE JUDGED AT A CLOCK WHERE EVERY PIECE RUNS (SYSTEM_SPEC 8 the whole is judged and the rule for choosing a move, 7 the budget, 11; piece: the wake).
Rust holds the unit tests (cargo test --release --offline -p cs_tactics --test fail_extend; -p cs_decision --test wake_time --test chance_order --test profile_steps): the reading on of a first move, the clock given to the search, the one order of fixes by stake times chance, the rows equal after the lean search.
THIS file holds what belongs to Python: the wall (the measure script names no game object and reaches for no oracle), the configs of the matches, the watchdog scaled to the clock, the means of a decision from a built log, and the binding taking the piece.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import whole_79_16 as W16
from catspace.incubator.analyst.discovery import whole_play as WP

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")
POSITION = "r2k2n1/3n3r/4R2p/7P/3p1p2/BR3P2/P2K4/8 w - - 0 39"                      # a position text is data here, in a test


def test_the_wall_the_measure_script_names_no_game_object_and_reaches_for_no_oracle():
    low = (DIR / "whole_79_16.py").read_text().lower()
    assert not re.search(r"\b[a-h][1-8]\b", low)
    assert "sys.path.insert" not in low
    for w in BANNED:
        assert not re.search(rf"\b{w}\b", low), w
    assert not re.search(r"ladder_oracle|leaf_ask|stockfish|analyse_pos|cs_learn|anthropic|\blo\.", low)


def test_no_decision_crate_depends_on_the_learning_crate():
    assert "cs_learn" not in (ROOT / "rust" / "engine" / "decision" / "Cargo.toml").read_text()
    assert "cs_learn" not in (ROOT / "rust" / "engine" / "tactics" / "Cargo.toml").read_text()


def test_the_configs_are_the_pieces_of_the_plan():
    c = W16.CONFIGS
    assert "read_on" in c["abc"] and "chance_order" not in c["abc"], "A to C: the reading on, the chance order off"
    assert "read_on" in c["abcd"] and "chance_order" in c["abcd"], "A to D: the reading on and the one order of fixes"
    assert "read_on" not in c["main"] and "chance_order" in c["main"], "the control of part A: no reading on"
    for k in c.values():
        assert "carry_graph" not in k and "diag_first" not in k, "the carry by the graph is not equal to asking afresh (79.14); the measure piece is never in play"


def test_the_watchdog_is_scaled_to_the_clock_and_reaches_the_workers_through_the_environment():
    assert W16.watchdog_for(0.2) == 300.0
    assert W16.watchdog_for(2.0) == 1500.0
    assert isinstance(WP.WATCHDOG, float)


def _decision(steps: list, ff: dict | None) -> dict:
    return {"kind": "decision", "gid": "g", "ply": 20, "position": "x", "played": "a", "search_first": "a", "wall": 2.0,
            "wake": {"value": 0, "depth": 8, "equal": ["a"], "steps": steps, "log": {"moves": [], "failed_first": ff}}}


def test_the_means_of_a_decision_count_the_frames_built_and_not_the_questions_and_the_reading_on(tmp_path):
    steps = [{"step": 2, "piece": "frames and the question read", "seconds": 0.02, "note": "3 frames, 5 rows, 40 questions, 1 frames not reached by the clock"},
             {"step": 22, "piece": "part of the frame read: the frames built", "seconds": 0.01, "note": "2 frames built"},
             {"step": 3, "piece": "threat list of both sides", "seconds": 0.1, "note": "4 rows against us, 6 against them, 0 dropped, 9 questions; not asked: nothing"}]
    ff = {"from_depth": 7, "depth": 8, "first_value": 0, "goal": 10, "outcome": "other_held", "seconds": 0.05, "nodes": 100, "allowed": 0.12}
    tmp_path.mkdir(exist_ok=True)
    with gzip.open(tmp_path / "g1.jsonl.gz", "wt") as f:
        f.write(json.dumps(_decision(steps, ff)) + "\n")
        f.write(json.dumps(_decision(steps, None)) + "\n")
    d = W16.asked16(tmp_path)
    assert d["decisions"] == 2 and d["built"] == 2.0 and d["frames"] == 3.0 and d["frame_questions"] == 40.0 and d["list_rows_us"] == 4.0
    assert d["ff_n"] == 1 and d["ff_outcomes"] == {"other_held": 1} and abs(d["ff_seconds"] - 0.05) < 1e-9
    assert "frames BUILT 2.00" in W16.asked_line("t", d)


@pytest.fixture(scope="module")
def world():
    from catspace.incubator.analyst.discovery import whole_79_14 as W14
    try:
        return W14.world()
    except FileNotFoundError as e:                                                # the untracked inputs of the harness (a worktree without them)
        pytest.skip(str(e))


def test_the_binding_takes_the_piece_and_logs_the_reading_on_within_its_share(world):
    from catspace.incubator.analyst.discovery import whole_79_14 as W14
    on = W14.Runner(world, W16.CONFIGS["abc"])
    off = W14.Runner(world, W16.CONFIGS["main"])
    asked = 0
    for secs in (0.1, 0.2, 0.2, 0.4):
        o = on.fresh(POSITION, secs)
        assert off.fresh(POSITION, secs)["log"].get("failed_first") is None
        ff = o["log"].get("failed_first")
        if ff:
            asked += 1
            assert ff["outcome"] in ("first_held", "other_held", "all_failed", "cut")
            assert ff["seconds"] <= ff["allowed"] + 0.05
    print(f"OBS 79.16 binding: the first move was asked again in {asked} of 4 asks")
    with pytest.raises(Exception):
        W14.Runner(world, ["no_such_piece"]).fresh(POSITION, 0.05)
