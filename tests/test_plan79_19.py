"""PLAN 79.19 ONE FORM OF STATEMENT, TWO MEMORIES, AND A WAY TO CHANGE THEM (SYSTEM_SPEC 11 the layers and the two memories; ledger rows 36, 34, 14, 13; piece: the memory, read by the wake).
Rust holds the unit tests (cargo test --release --offline -p cs_memory -p cs_decision): the statement and its four tests (part A, on two games), the two memories and the changes (part B), the wake reading through them (part C).
THIS file holds what belongs to Python: the wall for the new files (no game object named in a layer crate or the harness), and THE GUARD of part C: the wake's logged decisions through the binding are equal to main's on a fixed set of positions
(tests/baselines/plan79_19_main.json, made with the binding built from main before this plan's Rust): no goal given; every optional input given; a held measure goal across several plies; the open tactics carried across several plies. Two runs of the
scenarios are equal (no clock enters them).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import whole_79_19 as G
from catspace.incubator.analyst.discovery import whole_79_19_print as P

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests" / "baselines" / "plan79_19_main.json"
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")
CRATE_FILES = [ROOT / "rust" / "engine" / "memory" / "src" / n for n in ("statement.rs", "store.rs")] + [ROOT / "rust" / "engine" / "decision" / "src" / n for n in ("memory.rs", "carry.rs")]


def test_the_new_code_names_no_game_object():
    for p in CRATE_FILES + [ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "whole_79_19.py"]:
        if not p.exists():
            continue
        low = p.read_text().lower()
        assert not re.search(r"\b[a-h][1-8]\b", low), p.name
        for w in ("fen", "san", "uci", "check", "mate", "promotion"):
            assert not re.search(rf"\b{w}\b", low), (p.name, w)
        if p.suffix == ".py":
            assert "sys.path.insert" not in low
            for w in BANNED:
                assert not re.search(rf"\b{w}\b", low), (p.name, w)
            assert not re.search(r"syzygy|stockfish|cs_learn|anthropic", low), p.name


def test_the_strip_takes_out_the_clocks_and_nothing_else():
    d = {"seconds": 1.5, "step_seconds": [{"seconds": 2.0, "name": "x"}], "wall": 3, "note": "27 read, 0.0041 s of it", "nodes": 5, "depth_reads": [{"nodes": 7, "seconds": 0.1}]}
    assert G.strip(d) == {"note": "27 read, <s> of it", "nodes": 5, "depth_reads": [{"nodes": 7}]}


@pytest.fixture(scope="module")
def world():
    from catspace.incubator.analyst.discovery import whole_79_14 as W14
    try:
        return W14.world()
    except FileNotFoundError as e:                                                # the untracked inputs of the harness (a worktree without them)
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def rows(world):
    return G.all_rows(world, P.POSITIONS, "t")


@pytest.mark.parametrize("name", G.NAMES)
def test_the_wake_decides_exactly_as_it_did_on_main(rows, name):
    """Every scenario row (move, value, depth, nodes, chain, moves held equal and named, the whole log of reasons, the clocks taken out) equals the baseline made on main's binding."""
    snap = json.loads(BASELINE.read_text())
    assert len(rows[name]) == len(snap[name]) > 0
    for i, (a, b) in enumerate(zip(snap[name], json.loads(json.dumps(rows[name])))):
        assert a == b, (name, i)


def test_the_scenarios_hold_what_they_say(rows):
    """The scenarios are not empty ones: a goal held across plies is in the measure log, and with the carry by the graph a row of the threat list is marked carried."""
    held = [r["words"]["log"]["measure"] for r in rows["held_one"]]
    assert sum(1 for m in held if m and m.get("held_before")) >= 8
    assert any(x["carried"] for r in rows["carried"] if r["variant"] == "graph_on" for x in r["words"]["log"]["rows"])
    assert all(len(rows[n]) > 0 for n in G.NAMES)


@pytest.fixture(scope="module")
def through_the_file(world, tmp_path_factory):
    """The long-term memory of this game written as partition files (under a temporary folder here; the plan's own run writes under artifacts/experiments/whole_79_19) and the scenarios run reading them."""
    root = tmp_path_factory.mktemp("long_term")
    G.LONG_TERM = {kind: Path(G.write_memory(world, root, kind)[0]) for kind in ("nogoal", "tables", "full")}
    try:
        yield G.all_rows(world, P.POSITIONS, "f")
    finally:
        G.LONG_TERM = None


@pytest.mark.parametrize("name", G.NAMES)
def test_the_wake_reading_its_tables_and_its_goal_from_the_long_term_file_decides_as_main(through_the_file, name):
    """The tables the wake read as arguments and the mined measure are now a partition of the long-term memory, written to a file named by its content and read by the wake as the game's reader: the decisions are equal to the baseline made on main."""
    snap = json.loads(BASELINE.read_text())
    assert len(through_the_file[name]) == len(snap[name]) > 0
    for i, (a, b) in enumerate(zip(snap[name], json.loads(json.dumps(through_the_file[name])))):
        assert a == b, (name, i)


def test_a_partition_of_another_game_is_refused_by_the_wake(world, tmp_path):
    """The wake of one game given the partition file of another (another name, other rules) refuses it: what was learned in one game is not read in another."""
    other = tmp_path / "other"
    path, _id, _parent = world.model.memory_write(str(other), dict(world.worth), {}, None, None, None, None, None, None)
    # (the file is this game's: its game key is rewritten to another game's and the file is renamed by its new content)
    body = json.loads(Path(path).read_text())
    body["game"] = {"name": "another game", "rules": "0123456789abcdef"}
    import hashlib
    raw = json.dumps(body, separators=(",", ":")).encode()
    fake = tmp_path / (hashlib.sha256(raw).hexdigest()[:16] + ".json")
    fake.write_bytes(raw)
    args = (P.POSITIONS[3], [], 1.0, 3, dict(world.worth), {}, world.goal, ["goals"], 0, 0, None, None, None, None)
    with pytest.raises(ValueError, match="another game"):
        world.model.wake_decide(*args, long_term=str(fake))
