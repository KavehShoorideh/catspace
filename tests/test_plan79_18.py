"""PLAN 79.18 THE TWO FAULTS OF THE HELD GOAL MENDED (SYSTEM_SPEC 7 the measure of progress, 8 the rule for choosing a move and the wall; ledger rows 13 and 35; piece: the wake).
Rust holds the unit tests (cargo test --release --offline -p cs_tactics --test measure; -p cs_decision --test wake_measure): the fall counted against the lowest held, the recovery, a position met before, the goal's states (held, asleep, woken, dropped for the class only).
THIS file holds what belongs to Python: the wall for the new harness scripts (no game object named), the log of the held goal as the harness reads it, the means of a game's log, and the proof that with no goal given the wake decides exactly as it did on main (a snapshot made on main before this plan's Rust).
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import whole_79_18 as W18

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
SNAPSHOT = ROOT / "tests" / "baselines" / "plan79_18_no_goal_main.json"
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")

# positions of several kinds: the first position, middle games, a quiet ending, a tactical one (the texts are the game's own)
POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
    "7k/8/8/8/8/8/R7/K7 w - - 0 1",
    "8/8/R4K2/8/8/3k4/8/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "8/5pk1/6p1/7p/7P/6P1/5PK1/8 w - - 0 1",
    "r3k2r/pp3ppp/2n1b3/q2pP3/3P4/2P2N2/P1Q2PPP/R3KB1R w KQkq - 0 12",
    "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1",
    "2r3k1/5ppp/p7/1p6/8/P1R2P2/1P3KPP/8 w - - 0 1",
]


def test_the_harness_of_this_plan_names_no_game_object():
    for name in ("whole_79_18.py", "whole_79_18_a_print.py"):
        low = (DIR / name).read_text().lower()
        if name.endswith("_print.py"):
            continue                                              # a print script may show positions and moves as the game writes them
        assert not re.search(r"\b[a-h][1-8]\b", low), name
        assert "sys.path.insert" not in low
        for w in BANNED:
            assert not re.search(rf"\b{w}\b", low), (name, w)
        assert not re.search(r"syzygy|stockfish|cs_learn|anthropic", low), name


def test_the_crates_name_no_game_object_in_the_new_code():
    for p in (ROOT / "rust" / "engine" / "tactics" / "src" / "measure.rs", ROOT / "rust" / "engine" / "decision" / "src" / "carry.rs"):
        low = p.read_text().lower()
        assert not re.search(r"\b[a-h][1-8]\b", low), p.name
        for w in ("fen", "san", "uci", "check", "mate", "promotion"):
            assert not re.search(rf"\b{w}\b", low), (p.name, w)
        assert not re.search(r"syzygy|tablebase", low), p.name


@pytest.fixture(scope="module")
def model():
    from catspace.incubator.analyst.discovery import whole_79_14 as W14
    try:
        return W14.world().model
    except FileNotFoundError as e:                                                # the untracked inputs of the harness (a worktree without them)
        pytest.skip(str(e))


def test_with_no_goal_given_the_wake_decides_exactly_as_it_did_on_main(model):
    """The snapshot (tests/baselines/plan79_18_no_goal_main.json) was made with the binding built from main before this plan's change (79.17's merge): the same ten positions, no goal, a fixed depth the clock does not reach. A decision with no goal given must equal it: the move, the value, the depth, the chain, the moves held equal and named, the reason."""
    snap = json.loads(SNAPSHOT.read_text())
    now = W18.no_goal_rows(model, POSITIONS, "king")
    assert len(now) == len(snap["rows"]) == len(POSITIONS)
    for text, a, b in zip(POSITIONS, snap["rows"], now):
        assert a == b, text
        assert b["measure"] is None


def test_a_goal_given_adds_only_the_measure_log_and_the_old_fields_stay(model):
    kinds = model.kinds()
    lead = kinds.index("king")
    text = POSITIONS[3]
    args = (text, [], 2.0, 6, W18.NO_GOAL_ARGS["worth"], {}, "king", W18.NO_GOAL_ARGS["pieces"], 0, 0, None, None, None, None)
    plain = json.loads(model.wake_decide(*args))
    held = json.loads(model.wake_decide(*args, measure=(f"region(1,{lead})", 2), carry_slot="t18"))
    m = held["log"]["measure"]
    assert plain["log"]["measure"] is None
    for key in ("recovery", "asleep", "woke", "without_line", "recovered_total", "slept_total", "woken_total"):
        assert key in m, key
    assert m["recovery"] is False and m["asleep"] is None and m["woke"] is False


# ------------------------------------------------------------------------------------------------------------------------------ the means of a game's log

def _row(reached=False, cut=False, toward=False, dropped=None, rec=0, slept=0, woken=0, without=0):
    return {"kind": "decision", "wake": {"log": {"measure": {"reached": reached, "cut": cut, "played_toward": toward, "dropped": dropped, "recovered_total": rec, "slept_total": slept, "woken_total": woken, "without_line": without}}}}


def test_the_falls_of_a_game_count_falls_the_longest_run_recoveries_sleeps_wakes_and_cuts(tmp_path):
    rows = [_row(reached=True), _row(), _row(cut=True), _row(), _row(reached=True, rec=1), _row(slept=1, rec=1), _row(woken=1, slept=1, rec=2, without=3)]
    with gzip.open(tmp_path / "g_0.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    f = W18.falls18(tmp_path)
    assert f["games"] == 1 and f["falls"] == 2 and f["longest_run"] == 3 and f["cut_decisions"] == 1
    assert f["recoveries"] == 2 and f["asleep"] == 1 and f["woken"] == 1 and f["longest_without_a_line"] == 3 and f["dropped_for_good"] == {}


def test_a_goal_dropped_is_counted_once_by_its_cause(tmp_path):
    rows = [_row(), _row(dropped="the class changed: the units"), _row(dropped="the class changed: the units")]
    with gzip.open(tmp_path / "g_0.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    assert W18.falls18(tmp_path)["dropped_for_good"] == {"the class changed": 1}
