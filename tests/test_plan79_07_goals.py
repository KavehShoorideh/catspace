"""PLAN 79.07 THE PROPOSER IN RUST (SYSTEM_SPEC 7 the proposer, 4 the budget of questions, 8 step 3, 11; piece: the wake; rust/engine/decision/src/goals.rs). The Rust proposer is unit-tested in Rust (cargo test --release --offline -p cs_decision --test goals).
THIS file holds what belongs to Python: the wall (the new source names no game object and reaches for no oracle), and the MIRROR CHECK: the Rust goals through the binding (`BaseGraph.wake_goals`) against their Python original (subgoal_goals.goals,
future_goals.goals_of_list) on a few small constructed positions, the same candidate rows given to both. The Python runs without its text editor (a fix that needs one is not compared: the Rust fixes are legal moves). No puzzle, no stored game, no match, no Python wake."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import future_goals as FG
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import subgoal_units as SU
from catspace.incubator.analyst.discovery import threat_concrete as TC

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "rust" / "engine" / "decision"
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")
WORTH = {"q": 4343, "b": 1878, "n": 1661, "r": 1605, "p": 1045}
HALF = {"q": 120.0, "b": 606.0, "n": 628.0, "r": 534.0, "p": 508.0}
REC = TC.Record(WORTH, HALF, "synthetic")
PLACES = sorted(f + r for f in "abcdefgh" for r in "12345678")
NAMES = {"places": PLACES, "roles": ["white", "black"], "kinds": ["bishop", "king", "knight", "pawn", "queen", "rook"]}
ROLES = ["white", "black"]


def test_the_proposer_source_names_no_game_object_and_reaches_for_no_oracle():
    text = (DECISION / "src" / "goals.rs").read_text().lower()
    assert not re.search(r"\b[a-h][1-8]\b", text)
    for w in BANNED:
        assert not re.search(rf"\b{w}\b", text), w
    assert not re.search(r"ladder_oracle|stockfish|cs_learn|analyse|reqwest|anthropic", text)
    assert "rules/learned" not in text and "data/derived" not in text


def test_the_decision_crate_does_not_depend_on_the_learning_crate():
    assert "cs_learn" not in (DECISION / "Cargo.toml").read_text()


def _py_row(owner, kind, place, klass, by, resp, mover, stake=None):
    return {"side": "mover" if owner == mover else "other", "owner": owner, "kind": kind, "place": place, "place_ix": PLACES.index(place), "class": klass, "by": [(k, p) for _o, k, p in by],
            "response_ix": [(PLACES.index(a), PLACES.index(b), h, u) for a, b, h, u in resp], "stake": stake, "frames": [], "read_set": ("t", place), "ended": "quiet"}


@pytest.fixture(scope="module")
def world():
    model = GL.model("chess", "chess")
    facts = SG.Facts("chess")
    facts.goal_name = "king"
    return model, facts, TC.Lines(REC, threads=1)


def _fact_py(f):
    c = f["cause"]
    return (f["name"], f["status"], None if c is None else (c["type"], tuple(c["stoppers"]), c["victim"], c["basis"]))


def _fact_rust(f):
    c = f["cause"]
    return (f["name"], f["status"], None if c is None else (c["type"], tuple(c["stoppers"]), c["victim"], c["basis"]))


def _py_view(goals):
    out = []
    for g in goals:
        out.append((g["target"], g["source"], tuple(g["unmet"]), g["key"], g["stake"], g["n_unmet"], tuple(g["stoppers"]), tuple(_fact_py(f) for f in g["facts"]),
                    tuple((e["unit"], e["removable"], tuple(e["resolves"]), e["left"], e["left_stoppers"]) for e in g["fix_set"] if e["kind"] == "remove"), g["two_moves"], g["lead"]))
    return out


def _rust_view(goals):
    out = []
    for g in goals:
        assert not any(f["name"] == "outnumbers" for f in g["facts"]), "the cases chosen have no line onto the goal unit that one unit shuts: the new fact has no Python original"
        out.append((g["target"], g["source"], tuple(g["unmet"]), g["key"], g["stake"], g["n_unmet"], tuple(g["stoppers"]), tuple(_fact_rust(f) for f in g["facts"]),
                    tuple((e["unit"], e["removable"], tuple(e["resolves"]), e["left"], e["left_stoppers"]) for e in g["fix_set"] if e["kind"] == "remove"), g["two_moves"], g["lead"]))
    return out


CASES = [
    # (name, position, mover, rows [(owner, kind, place, class, by, response_ix)])
    ("a free taking works: no goal for it, the goal unit's goal only", "k7/8/8/3p4/8/8/8/K2R4 w - - 0 1", 0, [(1, "pawn", "d5", "threat", [(0, "rook", "d1")], [])]),
    ("a guarded unit: the taking is refuted, the guard is the stopper", "k7/8/2p5/3p4/8/8/8/K2R4 w - - 0 1", 0, [(1, "pawn", "d5", "threat", [(0, "rook", "d1")], [])]),
    ("a unit nothing bears on: every unit read, the quiet first move", "7k/8/8/8/8/5b2/8/R3K3 w - - 0 1", 0, []),
    ("like for like: an exchange is a goal", "k7/8/2p5/3n4/8/4N3/8/K7 w - - 0 1", 0, [(1, "knight", "d5", "threat", [(0, "knight", "e3")], [])]),
    ("a bearer that may not take: it is pinned, the holder is named", "k3r3/8/8/8/8/3p4/4B3/4K3 w - - 0 1", 0, [(1, "pawn", "d3", "threat", [(0, "bishop", "e2")], [])]),
    ("the end of the game forced against the mover: the goal unit's goal of the side to move", "6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1", 1, []),
]


@pytest.mark.parametrize("name,fen,mover,rows", CASES, ids=[c[0] for c in CASES])
def test_the_rust_goals_are_the_python_goals_on_a_small_constructed_position(world, name, fen, mover, rows):
    model, facts, lines = world
    py_rows = {"mover": [], "other": []}
    for owner, kind, place, klass, by, resp in rows:
        r = _py_row(owner, kind, place, klass, [(o, k, p) for o, k, p in by], resp, mover)
        py_rows[r["side"]].append(r)
    item = {"start": fen, "moves": [], "raw": {"mover": mover, "board": []}, "names": NAMES, "list": py_rows}
    c = TC.concrete_lists([item], lines)[0]
    SU.every_unit([item], [[mover]], [c], facts, lines, [None])
    want = SG.goals(item, mover, c, facts, editor=None)
    rust_rows = [(ROLES[o], k, p, klass, [(ROLES[bo], bk, bp) for bo, bk, bp in by], [(a, b, h, u) for a, b, h, u in resp], None) for o, k, p, klass, by, resp in rows]
    got = model.wake_goals(fen, WORTH, HALF, "king", [ROLES[mover]], rust_rows, 1000)
    assert got["unasked"] == [], "a budget this large asks every unit"
    pv = _py_view(want)
    rv = _rust_view(got["goals"])
    assert [x[0] for x in rv] == [x[0] for x in pv], name
    for a, b in zip(rv, pv):
        assert a == b, (name, a, b)
    assert len(rv) >= 1


def test_the_rust_goals_of_two_threats_are_the_python_ones(world):
    model, facts, lines = world
    fen, mover = "4k3/7p/8/p7/8/8/8/RB2K3 w - - 0 1", 0
    rows = [(1, "pawn", "a5", "threat", [(0, "rook", "a1")], []), (1, "pawn", "h7", "threat", [(0, "bishop", "b1")], [])]
    py_rows = {"mover": [], "other": [_py_row(o, k, p, kl, by, resp, mover) for o, k, p, kl, by, resp in rows]}
    item = {"start": fen, "moves": [], "raw": {"mover": mover, "board": []}, "names": NAMES, "list": py_rows}
    c = TC.concrete_lists([item], lines)[0]
    want = FG.goals_of_list(c, "mover", REC)
    rust_rows = [(ROLES[o], k, p, kl, [(ROLES[bo], bk, bp) for bo, bk, bp in by], resp, None) for o, k, p, kl, by, resp in rows]
    got = model.wake_goals(fen, WORTH, HALF, "king", ["white"], rust_rows, 1000)
    two = dict(got["two"])["white"]
    py_pairs = [(g["kind"], g["stake"], sorted((PLACES[t["place"]], t["net"], t["replies"]) for t in g["threats"])) for g in want["goals"]]
    rs_pairs = [(k, st, sorted((u[2], net, rep) for u, net, rep in th)) for k, st, th in two]
    assert rs_pairs == py_pairs
    assert len(py_pairs) >= 1
