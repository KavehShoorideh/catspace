"""Plan 78.47: THE FORCING READ, SPENT WHERE IT MATTERS (rust/engine/tactics/src/sheet/forcing.rs). A: the read accounts for the positions it spends per kind of work. B: a reply that touches nothing the read looked at
is answered from the stored result and changes no verdict and no value. C: the other side's gain is read once per position. D: the order of deepening. E: the ledger search has the last word on a CONFIRMED verdict
(NOT BORNE OUT, with both lines). Small fixed positions written here; the measures over real games and puzzles are forcing_shortcut_measure.py, forcing_cost_measure.py and forcing_soundness_measure.py."""
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import forcing_cost_measure as FC
from catspace.incubator.analyst.discovery import forcing_shortcut_measure as FSH
from catspace.incubator.analyst.discovery import forcing_soundness_measure as FS
from catspace.incubator.analyst.discovery import puzzle_sheet_measure as PM
from catspace.incubator.analyst.discovery import question_sheet as Q
from catspace.incubator.analyst.discovery import sheet_rust as SR

WORTH = {"k": 0, "q": 14, "r": 8, "b": 5, "n": 4, "p": 2}
ROOT = Path(__file__).resolve().parents[1]
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"

KNIGHT_FORK = "r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1"
PAWN_FORK = "4k3/8/8/3n1n2/8/4P3/8/6K1 w - - 0 1"
GUARD = "4k3/8/2n5/1B2r3/8/8/4R3/4K3 w - - 0 1"
# the answer of 78.46's soundness measure that the search contradicted (held games, game 1 ply 50): the read confirmed Rd8 on value 1 and the search from there says 0
CONTRADICTED = "2b1kb1r/8/p4n1p/p4p1R/PqPR1P2/NP6/8/2BQ1K2 w k - 3 26"
# far pawns the exchange on the knight's place does not involve
FAR_PAWNS = "4k3/pp6/8/3n1n2/8/4P3/8/6K1 w - - 0 1"


@pytest.fixture(scope="module")
def model():
    return SR.model(Q.Env("chess"), WORTH)


# ------------------------------------------------------------------------------------------------------------------------------ A: where the positions go

def test_the_cost_account_adds_up_to_the_positions_read(model):
    read = model.forcing(KNIGHT_FORK)
    cost = read["cost"]
    works = {k: v for k, v in cost.items() if isinstance(v, dict)}
    assert sum(w["positions"] for w in works.values()) == read["nodes"]                   # every position the fuse counts is charged to one kind of work
    assert works["root"]["positions"] == 1
    assert cost["replies_took"] + cost["replies_quiet"] > 0


def test_the_cost_measure_merges_and_prints_per_kind_of_work(model):
    FC._ST["model"] = model
    parts = [FC.run_one((KNIGHT_FORK, None)), FC.run_one((PAWN_FORK, None))]
    tot = FC.merge(parts)
    lines = FC.obs_lines("t", tot, None)
    assert tot["read"] == 2 and any("answered from memory" in line for line in lines) and any(line.startswith("OBS forcing cost t: replies read") for line in lines)


# ------------------------------------------------------------------------------------------------------------------------------ B: replies by what they touch

def test_the_shortcut_changes_no_verdict_and_no_value_on_fixed_positions(model):
    for fen in (KNIGHT_FORK, PAWN_FORK, GUARD, FAR_PAWNS):
        full = model.forcing(fen, 400_000, None, False)
        short = model.forcing(fen, 400_000, None, True)
        assert not full["cut"] and not short["cut"]
        assert FSH.compare(full, short) == []                                            # the comparison of the measure: verdict, value, every reply
        assert short["nodes"] <= full["nodes"]


def test_a_reply_that_touches_nothing_is_answered_from_the_stored_result(model):
    short = model.forcing(FAR_PAWNS, 400_000, None, True)
    full = model.forcing(FAR_PAWNS, 400_000, None, False)
    assert short["cost"]["untouched"] > 0 and full["cost"]["untouched"] == 0              # the old path (shortcut False) never asks the question
    assert short["cost"]["unchanged_quiet"] > 0 and short["nodes"] < full["nodes"]       # fewer positions read for the same answer


def test_a_reply_that_takes_or_moves_a_unit_the_claim_rests_on_is_read_in_full(model):
    short = model.forcing(KNIGHT_FORK, 400_000, None, True)
    # every reply of the fork either moves the rook (the target), the goal unit, or is read as touching: none is taken from a stored result without being untouched
    assert short["cost"]["unchanged_took"] == 0
    t = {x["move"]: x for x in short["threats"]}["d5c7"]
    assert t["verdict"] == "CONFIRMED" and all(r["value"] > 0 for r in t["replies"])


def test_the_comparison_finds_a_difference_when_there_is_one():
    a = {"threats": [{"move": "a1a2", "verdict": "CONFIRMED", "value": 4.0, "replies": [{"reply": "x", "value": 4.0, "open": False, "ends": None}]}]}
    b = {"threats": [{"move": "a1a2", "verdict": "CONFIRMED", "value": 3.0, "replies": [{"reply": "x", "value": 3.0, "open": False, "ends": None}]}]}
    d = FSH.compare(a, b)
    assert len(d) == 1 and d[0][1] == "value"
    assert FSH.compare(a, a) == []


# ------------------------------------------------------------------------------------------------------------------------------ C and D: memory and order

def test_the_read_is_deterministic_and_the_memory_of_positions_answers_repeats(model):
    a = model.forcing(GUARD, 400_000)
    b = model.forcing(GUARD, 400_000)
    assert FSH.compare(a, b) == [] and a["nodes"] == b["nodes"]
    works = {k: v for k, v in a["cost"].items() if isinstance(v, dict)}
    assert sum(w["memory"] for w in works.values()) > 0                                    # the same position reached again is not read again


# ------------------------------------------------------------------------------------------------------------------------------ E: the search as the last word

def test_a_confirmed_verdict_carries_the_search_that_bore_it_out(model):
    for fen, mv in ((KNIGHT_FORK, "d5c7"), (PAWN_FORK, "e3e4"), (GUARD, "b5c6")):
        t = {x["move"]: x for x in model.forcing(fen)["threats"]}[mv]
        assert t["verdict"] == "CONFIRMED" and t["search"]["status"] == "holds" and not t["search"]["cut"]


def test_a_claim_the_search_contradicts_is_not_confirmed_and_names_both_lines(model):
    read = model.forcing(CONTRADICTED, 20_000)
    t = {x["move"]: x for x in read["threats"]}.get("d4d8")
    assert t is None or t["verdict"] != "CONFIRMED"                                        # the claim the 78.46 soundness measure found contradicted is not given as confirmed
    for t in read["threats"]:
        if t["verdict"] == "NOT BORNE OUT":
            assert t["search"]["status"] == "contradicted" and t["search"]["line"] and t["replies"]
    assert read["not_borne_out"] == sum(1 for t in read["threats"] if t["verdict"] == "NOT BORNE OUT")
    assert read["found"] == read["confirmed"] + read["not_borne_out"] + read["unsettled"] + read["no_claim"] + read["unread"]


def test_the_soundness_measure_checks_two_plies_deeper_than_the_verdict():
    assert FS.EXTRA_PLIES == 2
    t = {"plies": 3, "level": 1}
    assert FS.needed_depth(t) + FS.EXTRA_PLIES == 8


def test_the_puzzle_measure_reads_not_borne_out_as_class_f():
    read = {"threats": [{"move": "c1c2", "verdict": "NOT BORNE OUT", "value": 2.0, "level": 0, "claims": [{"target": "n@c6", "kind": "takes", "value": 4.0}], "replies": [{"reply": "x", "value": 2.0, "open": False, "line": ["x"], "took": []}]}],
            "found": 1, "confirmed": 0, "unsettled": 0, "no_claim": 0, "unread": 0, "nodes": 10, "cut": False, "level": 0}
    view = PM.forcing_view(read, "c1c2")
    assert not view["answer_confirmed"] and view["pick"] is None and view["answer"]["verdict"] == "NOT BORNE OUT"
    assert PM.forcing_class(view, "R", False) == "F"


# ------------------------------------------------------------------------------------------------------------------------------ the layer stays game-free

def test_the_new_source_and_scripts_name_no_game_object():
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', (ROOT / "rust" / "engine" / "tactics" / "src" / "sheet" / "forcing.rs").read_text())
    hit = re.search(CHESS_WORDS, text, flags=re.I)
    assert hit is None, hit.group(0) if hit else None
    for name in ("forcing_cost_measure.py", "forcing_shortcut_measure.py", "forcing_soundness_measure.py"):
        src = (ROOT / "catspace" / "incubator" / "analyst" / "discovery" / name).read_text()
        assert not re.search(r"\b(king|queen|rook|bishop|knight|pawn)\b", src, flags=re.I), name
