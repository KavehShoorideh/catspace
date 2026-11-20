"""PLAN 79.06 THE THREAT LIST IN RUST (SYSTEM_SPEC 8 step 3, 4, 11; piece: the wake; rust/engine/decision/src/threats.rs). The Rust list is unit-tested in Rust (cargo test --release --offline -p cs_decision --test threats). THIS file holds what belongs to
Python: the wall (the new source names no game object and reaches for no oracle), and the MIRROR CHECK: the Rust list through the binding (`BaseGraph.wake_threats`) against its Python original (threat_concrete.concrete_lists and subgoal_units.every_unit) on a few
small constructed positions, the same candidate rows given to both. No puzzle, no stored game, no match, no Python wake."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

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


def test_the_threat_list_source_names_no_game_object_and_reaches_for_no_oracle():
    text = (DECISION / "src" / "threats.rs").read_text().lower()
    assert not re.search(r"\b[a-h][1-8]\b", text)
    for w in BANNED:
        assert not re.search(rf"\b{w}\b", text), w
    assert not re.search(r"ladder_oracle|stockfish|cs_learn|analyse|reqwest|anthropic", text)
    assert "rules/learned" not in text and "data/derived" not in text


# ------------------------------------------------------------------------------------------------------------------------------ the mirror

def _rust_row(r_owner, kind, place, klass, by, resp, stake=None):
    return (r_owner, kind, place, klass, by, resp, stake)


def _py_row(owner, kind, place, klass, by, resp, mover, stake=None):
    return {"side": "mover" if owner == mover else "other", "owner": owner, "kind": kind, "place": place, "place_ix": PLACES.index(place), "class": klass, "by": [(k, p) for _o, k, p in by],
            "response_ix": [(PLACES.index(a), PLACES.index(b), h, u) for a, b, h, u in resp], "stake": stake, "frames": [], "read_set": ("t", place), "ended": "quiet"}


@pytest.fixture(scope="module")
def world():
    model = GL.model("chess", "chess")
    facts = SG.Facts("chess")
    facts.goal_name = "king"
    return model, facts, TC.Lines(REC, threads=1)


def _norm_rows(rows):
    return {r["place"] if "place" in r else r["unit"][2]: (r["verdict"], r["net"], tuple(r["line"]), bool(r["two_moves"])) for r in rows if not r.get("goal")}


def _py_view(c):
    out = {}
    for side in ("mover", "other", "exchange"):
        out[side] = {r["place"]: (r["verdict"], r["net"], tuple(r["line"]), bool(r["two_moves"]), tuple((x["names"][0], x["net"]) for x in r["response"]), r.get("every", False)) for r in c[side] if not r.get("goal")}
    out["dropped"] = {r["place"]: (r.get("why"), r.get("every", False)) for r in c["dropped_rows"]}
    out["goal"] = sorted((r["owner"], tuple(r["line"]), tuple(sorted(a["names"][0] for a in r["response"]))) for r in c["goal"])
    return out


def _rust_view(o):
    out = {}
    for side in ("mover", "other", "exchange"):
        out[side] = {r["unit"][2]: (r["verdict"], r["net"], tuple(r["line"]), bool(r["two_moves"]), tuple((x[0][0], x[1]) for x in r["responses"]), r["every"]) for r in o[side] if not r["goal"]}
    out["dropped"] = {r["unit"][2]: (r["why"], r["every"]) for r in o["dropped"]}
    out["goal"] = sorted((ROLES.index(r["owner"]), tuple(r["line"]), tuple(sorted(a[0][0] for a in r["responses"]))) for r in o["goal"])
    return out


CASES = [
    # (name, position, mover, rows [(owner, kind, place, class, by, response_ix)])
    ("a free taking and nothing else", "k7/8/8/3p4/8/8/8/K2R4 w - - 0 1", 0, [(1, "pawn", "d5", "threat", [(0, "rook", "d1")], [])]),
    ("a threat against the mover with a named answer", "4k3/8/8/8/8/8/1b6/R3K3 w - - 0 1", 0, [(0, "rook", "a1", "threat", [(1, "bishop", "b2")], [("a1", "b1", 1, True), ("a1", "a2", 1, True)])]),
    ("the end of the game forced against the side to move", "6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1", 1, []),
    ("a guarded unit: refuted, and every unit read", "4k3/8/3p4/2p5/8/8/8/2Q1K3 w - - 0 1", 0, []),
    ("like for like", "k7/8/2p5/3n4/8/4N3/8/K7 w - - 0 1", 0, [(1, "knight", "d5", "threat", [(0, "knight", "e3")], [])]),
]


@pytest.mark.parametrize("name,fen,mover,rows", CASES, ids=[c[0] for c in CASES])
def test_the_rust_list_is_the_python_list_on_a_small_constructed_position(world, name, fen, mover, rows):
    model, facts, lines = world
    py_rows = {"mover": [], "other": []}
    for owner, kind, place, klass, by, resp in rows:
        r = _py_row(owner, kind, place, klass, [(o, k, p) for o, k, p in by], resp, mover)
        py_rows[r["side"]].append(r)
    item = {"start": fen, "moves": [], "raw": {"mover": mover, "board": []}, "names": NAMES, "list": py_rows}
    c = TC.concrete_lists([item], lines)[0]
    SU.every_unit([item], [[mover]], [c], facts, lines, [None])
    text = fen
    rust_rows = [(ROLES[o], k, p, klass, [(ROLES[bo], bk, bp) for bo, bk, bp in by], [(a, b, h, u) for a, b, h, u in resp], None) for o, k, p, klass, by, resp in rows]
    got = model.wake_threats(text, WORTH, HALF, "king", rust_rows, [ROLES[mover]])
    assert _rust_view(got) == _py_view(c), name
