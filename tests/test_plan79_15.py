"""PLAN 79.15 A FIRST MOVE THE PROBE LOWERED IS NOT PLAYED; THE STEPS' SHARE OF THE CLOCK JUDGED BY PLAY (SYSTEM_SPEC 8 the wall and the rule for choosing a move, 7 the budget, 11; piece: the wake).
Rust holds the unit tests (cargo test --release --offline -p cs_decision --test wall --test standing_line): the candidates of the choice (decision 18 of 79.14 is the first case), the wake with no lowered first move being main's, a line on the goal unit and the reason that says the move answers it.
THIS file holds what belongs to Python: the wall (the measure scripts name no game object and reach for no oracle), the measure of a lowered first move on a built log, the shares of the matches, the reason of mode Gain, and the binding (`first`, `pre_share_scale`, `diag_first`)."""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import lowered_first as LF
from catspace.incubator.analyst.discovery import whole_79_15 as W15
from catspace.incubator.analyst.discovery import whole_play as WP

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")
POSITION_18 = "1n1qkbnr/2pbpppp/8/3p4/8/2PPPP2/1B4PP/QN2KBNR b Kk - 2 11"          # decision 18 of 79.14 (game bc_v_old_A:A9100:1, ply 21); a position text is data here, in a test


def test_the_wall_the_measure_scripts_name_no_game_object_and_reach_for_no_oracle():
    for name in ("whole_79_15.py", "lowered_first.py"):
        low = (DIR / name).read_text().lower()
        assert not re.search(r"\b[a-h][1-8]\b", low), (name, re.search(r"\b[a-h][1-8]\b", low).group(0))
        assert "sys.path.insert" not in low
        for w in BANNED:
            assert not re.search(rf"\b{w}\b", low), (name, w)
        assert not re.search(r"ladder_oracle|leaf_ask|stockfish|analyse_pos|cs_learn|anthropic|\blo\.", low), name


def test_no_decision_crate_depends_on_the_learning_crate_and_the_wall_is_a_module_of_its_own():
    toml = (ROOT / "rust" / "engine" / "decision" / "Cargo.toml").read_text()
    assert "cs_learn" not in toml
    src = (ROOT / "rust" / "engine" / "decision" / "src" / "wall.rs").read_text()
    assert "cs_learn" not in src and "Instant" not in src, "the wall reads numbers only: no clock, no search"


def _write_game(folder: Path, name: str, rows: list) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / f"{name}.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _decision(played: str, first: str, value: int, probes: dict, equal_from: str = "as_before") -> dict:
    moves = [{"move": m, "value": v, "equal": v >= value, "named": m == first} for m, v in probes.items()]
    return {"kind": "decision", "gid": "g", "ply": 20, "position": "x", "played": played, "search_first": first, "wall": 0.2,
            "wake": {"value": value, "depth": 6, "equal": [played], "steps": [], "log": {"moves": moves, "equal_from": equal_from}}}


def test_the_measure_counts_a_lowered_first_move_played_and_one_that_is_not(tmp_path):
    base = {"a": -1301, "b": 0, "c": 0, "d": -390}
    only_first_low = {"a": -1301, "b": -400, "c": -500}                           # no other move holds: not counted
    _write_game(tmp_path, "g1", [_decision("a", "a", 0, base), _decision("b", "a", 0, base, "probe_only"), _decision("a", "a", 0, only_first_low), _decision("a", "a", 0, {"a": 0, "b": 0})])
    c = LF.count(tmp_path)
    assert (c["decisions"], c["lowered"], c["big"], c["played"], c["played_big"], c["games"], c["games_played_big"]) == (4, 2, 2, 1, 1, 1, 1), c
    assert c["by_from"] == {"as_before": 1, "probe_only": 1}
    assert LF.lowered_of(_decision("a", "a", 0, only_first_low))["others_hold"] is False
    assert "PLAYED the lowered first move: 1" in LF.line("t", c)


def test_the_shares_are_three_and_main_is_one_of_them_and_only_the_wake_has_a_scale():
    assert W15.SHARES == {"s1": 1.0, "s2": 0.5, "s4": 0.25}
    assert W15.PIECES[-1] == "chance_order" and "carry_graph" not in W15.PIECES, "every piece on, the carry by the graph off (79.14: not equal to asking afresh)"
    spec = WP.player_spec("wake", 0.2, pieces=["carried"], pre_scale=0.5)
    assert spec["pre_scale"] == 0.5
    assert "pre_scale" not in WP.player_spec("old", 0.2, pre_scale=0.5)
    assert "pre_scale" not in WP.player_spec("wake", 0.2)


def test_the_reason_of_mode_gain_is_logged_as_its_own_class():
    assert WP.reason_of("the gain: among the 3 moves the search holds equal, the largest gain of our chains' stake times chance less theirs (0.1); the rule would pick e2e4") == "gain"
    assert "gain" in WP.REASONS
    assert WP.reason_of("the equal reading put the search's first move lower, 1301 below the best at that reading; this move is held equal to the best") == "first_lowered"


@pytest.fixture(scope="module")
def world():
    from catspace.incubator.analyst.discovery import whole_79_14 as W14
    try:
        return W14.world()
    except FileNotFoundError as e:                                                # the untracked inputs of the harness (a worktree without them)
        pytest.skip(str(e))


def test_the_binding_gives_the_first_move_and_never_plays_a_lowered_first_with_another_move_holding(world):
    from catspace.incubator.analyst.discovery import whole_79_14 as W14
    r = W14.Runner(world, W15.PIECES)
    lowered = 0
    for secs in (0.05, 0.1, 0.2, 0.2, 0.2, 0.4):
        out = r.fresh(POSITION_18, secs)
        assert "first" in out and out["first"] is not None
        lg = out["log"]
        assert out["equal"], "the candidates are never empty where a move was read"
        assert "first_lowered" in lg and "equal_from" in lg and "depth_reads" in lg
        if lg["first_lowered"] is not None:
            lowered += 1
            if lg["equal_from"] in ("verified", "probe_only"):
                assert out["move"] != out["first"], (secs, lg["equal_from"])
                assert out["first"] not in out["equal"]
    print(f"OBS 79.15 binding: lowered first moves seen in 6 asks of the position of decision 18: {lowered}")


def test_the_share_scale_and_the_measure_piece_reach_the_wake_through_the_binding(world):
    from catspace.incubator.analyst.discovery import whole_79_14 as W14
    r = W14.Runner(world, W15.PIECES + ["diag_first"])
    full = r.fresh(POSITION_18, 0.2)
    small = r.fresh(POSITION_18, 0.2, pre_share_scale=0.25)
    assert small["move"] and full["move"]
    with pytest.raises(Exception):
        W14.Runner(world, ["no_such_piece"]).fresh(POSITION_18, 0.05)
