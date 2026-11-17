"""Plan 78.72: the threat list made concrete (catspace/incubator/analyst/discovery/threat_concrete.py, the checks in threat_concrete_measure.py, the print in threat_print.py; the Rust line read in
rust/engine/tactics/src/lines.rs and its binding py/src/lines_py.rs). SYSTEM_SPEC 8 and 4. The verdict and the response logic on synthetic lines; the Rust line read on real positions (an exchange is an
exchange, a free taking is confirmed, the end of the game is read, a pass is refused where the game gives none); a scan of the files that must name no game object."""
from __future__ import annotations

import re
from pathlib import Path

import cs_engine as CE
import pytest

from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import threat_concrete_measure as TCM
from catspace.incubator.analyst.discovery import threat_list as TL

ROOT = Path(__file__).resolve().parents[1]
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
ENGINE = ROOT / "rust" / "engine"
WORTH = {"q": 4343, "b": 1878, "n": 1661, "r": 1605, "p": 1045}
HALF = {"q": 120.0, "b": 606.0, "n": 628.0, "r": 534.0, "p": 508.0}
REC = TC.Record(WORTH, HALF, "synthetic")
PLACES = sorted(f + r for f in "abcdefgh" for r in "12345678")


def place(name: str) -> int:
    return PLACES.index(name)


def read(moves, queries, **kw):
    return CE.read_lines("chess", [(None, moves)], queries, WORTH, kw.get("depth", 6), kw.get("fuse", 40000), kw.get("end_plies", 7), kw.get("end_fuse", 1500000), 1)


def _line(events, value, end=0, cut=False, depth=6):
    return {"events": events, "value": value, "end": end, "cut": cut, "depth": depth, "names": ["m1", "m2", "m3"], "settled": 3, "nodes": 10, "first": (1, 2), "lead": None, "prefix": []}


# ------------------------------------------------------------------------------------------------------------------------------ the verdict, on synthetic lines

def test_a_unit_taken_for_nothing_is_confirmed_a_trade_of_equals_is_an_exchange_a_loss_is_refuted_a_cut_line_is_unread():
    assert TC.verdict_of(_line([(False, "p")], 1045), REC)["verdict"] == "CONFIRMED"
    assert TC.verdict_of(_line([(False, "n"), (True, "n")], 0), REC)["verdict"] == "EXCHANGE"
    assert TC.verdict_of(_line([(False, "p"), (True, "q")], 1045 - 4343), REC)["verdict"] == "REFUTED"
    v = TC.verdict_of(_line([(False, "p")], 1045, cut=True), REC)
    assert v["verdict"] == "UNREAD" and v["ended"] == TL.ENDED[3]


def test_ahead_is_the_records_not_a_tables_and_the_gap_it_cannot_tell_apart_is_an_exchange():
    """A bishop for a pawn is ahead by more than the record's resolution; a pawn for a pawn and a knight is not told apart."""
    assert TC.verdict_of(_line([(False, "b"), (True, "p")], 1878 - 1045), REC)["verdict"] == "CONFIRMED"
    narrow = TC.Record(WORTH, {k: 600.0 for k in WORTH}, "synthetic wide")
    assert TC.verdict_of(_line([(False, "b"), (True, "p")], 1878 - 1045), narrow)["verdict"] == "EXCHANGE"
    assert REC.resolution(["p", "p"]) == pytest.approx(508.0 * 2 ** 0.5)


def test_the_end_of_the_game_decides_a_line_concretely():
    assert TC.verdict_of(_line([], 1 << 40, end=1), REC)["ended"] == TL.ENDED[0]
    assert TC.verdict_of(_line([], -(1 << 40), end=2), REC)["verdict"] == "REFUTED"
    assert TC.verdict_of(_line([(False, "p")], 1045), REC)["ended"] == TL.ENDED[1]          # the settled end


def _after(whole_value, takes, cut=False, events=()):
    return {"whole": dict(_line(list(events), whole_value), cut=cut), "takes": takes}


def test_a_response_is_kept_only_if_the_taker_is_no_longer_ahead_and_nothing_larger_is_lost():
    passed = _line([(False, "p")], 1045)
    saves = TC.response_of(_after(0, []), passed, REC, "the unit moves", "t")
    assert saves["state"] == "KEPT"
    still = TC.response_of(_after(1045, [_line([(False, "p")], 1045)]), passed, REC, "a guard is added", "t")
    assert still["state"] == "STILL WON"
    loses_more = TC.response_of(_after(4343, [], events=[(False, "q")]), passed, REC, "a counter threat", "t")
    assert loses_more["state"] == "WORSE THAN PASSING"
    cut = TC.response_of(_after(0, [], cut=True), passed, REC, "the unit moves", "t")
    assert cut["state"] == "UNREAD"
    exch = TC.response_of(_after(0, [_line([(False, "n"), (True, "n")], 0)]), passed, REC, "a taking", "t")
    assert exch["state"] == "KEPT"


# ------------------------------------------------------------------------------------------------------------------------------ the Rust line read on real positions

def test_a_taking_that_is_retaken_is_an_exchange_and_a_free_taking_is_confirmed():
    moves = "d2d4 d7d5 e2e4 b8c6 e4e5 f7f6 g1f3 f6e5 f3e5 c8e6".split()                     # the knight on e5 may take the knight on c6, and is retaken
    r = read(moves, [(0, 0, 0, 1, place("c6"), 0, 0)])[0]
    assert r["ok"] and len(r["reads"]) == 1
    v = TC.verdict_of(r["reads"][0], REC)
    assert v["verdict"] == "EXCHANGE" and sorted(v["taker_lost"]) == ["n"] and sorted(v["taker_won"]) == ["n"]
    free = "e2e4 e7e5 g1f3 b8c6 f1c4 d8f6 b1c3 f6f2".split()                                  # (white to move: the queen on f2 hangs to the king)
    r = read(free[:-1] + ["g8e7", "c3d5"], [(0, 0, 0, 1, place("f7"), 0, 0)])[0]
    assert r["ok"]


def test_the_other_side_after_one_pass_and_a_game_with_no_pass_for_a_side_that_must_answer():
    moves = "e2e4 e7e5 d1h5 b8c6 f1c4 g8f6".split()                                           # white to move; black's knight took nothing: a pass of white is allowed
    assert read(moves, [(0, 0, 1, 1, place("f7"), 0, 0)])[0]["ok"]
    in_check = "e2e4 f7f6 d1h5".split()                                                       # black is in check: the game gives it no pass
    assert read(in_check, [(0, 0, 1, 1, place("e4"), 0, 0)])[0]["ok"] is False


def test_the_forced_end_of_the_game_is_found_by_the_search_and_confirmed_by_the_exact_read():
    mate_in_one = "e2e4 e7e5 d1h5 b8c6 f1c4 g8f6".split()
    r = read(mate_in_one, [(0, 4, 0, 1, 0, 0, 0)])[0]
    assert r["ok"] and r["forced"] == "proved" and r["depth"] == 1 and r["line"][0] == "h5f7"
    quiet = "e2e4 e7e5".split()
    assert read(quiet, [(0, 4, 0, 1, 0, 0, 0)])[0]["forced"] == "none"
    # the answers to a forced end: the moves of the defender after which the other side finds none (the first of the list is a legal move of the defender, and there are some that do not answer it)
    ans = read("e2e4 e7e5 d1h5 b8c6 f1c4".split(), [(0, 5, 0, 1, 0, 0, 0)])[0]["answers"]
    assert any(a[2] == 0 for a in ans) and any(a[2] == 1 for a in ans)


def test_the_line_of_a_row_ends_at_its_last_capture_and_a_quiet_move_that_the_other_side_answers_is_no_threat():
    moves = "d2d4 d7d5 e2e4 b8c6 e4e5 f7f6 g1f3 f6e5 f3e5 c8e6".split()
    r = read(moves, [(0, 0, 0, 1, place("c6"), 0, 0)])[0]["reads"][0]
    assert 1 <= r["settled"] <= len(r["names"])
    two = read(moves, [(0, 3, 0, 1, place("c6"), 0, 0)])[0]
    assert two["ok"]
    for x in two["reads"]:
        assert x["lead"] is not None and x["names"][0] and x["prefix"] == []


def test_a_response_is_read_two_ways_the_whole_board_line_and_the_taking_of_the_unit_after_it():
    moves = "d2d4 d7d5 g1f3 g8f6 c2c4 e7e6 b1c3 d5c4 e2e3 c7c5 f1c4 c5d4".split()              # white to move: the pawn on d4 may be taken
    best = read(moves, [(0, 8, 0, 1, 0, 0, 0)])[0]["reads"][0]
    assert best["value"] > 0
    afters = read(moves, [(0, 7, 1, 0, 0, 0, place("d4"))])[0]
    assert afters["ok"] and len(afters["afters"]) > 10
    assert all("whole" in a and "takes" in a for a in afters["afters"])


# ------------------------------------------------------------------------------------------------------------------------------ the rules of the repo

def test_the_new_files_name_no_game_object_and_no_relative_data_path():
    for mod in (TC, TCM):
        src = Path(mod.__file__).read_text()
        hit = re.search(CHESS_WORDS, src, flags=re.I)
        assert hit is None, (mod.__name__, hit.group(0))
        assert not re.search(r"[\"']data/", src)
        assert "sys.path" not in src
        assert "experiments." not in src
    for p in (ENGINE / "tactics" / "src" / "lines.rs", ENGINE / "py" / "src" / "lines_py.rs"):
        hit = re.search(CHESS_WORDS, p.read_text(), flags=re.I)
        assert hit is None, (p.name, hit.group(0))
    s = Path(TL.__file__).read_text()
    hit = re.search(CHESS_WORDS, s, flags=re.I)
    assert hit is None, ("threat_list", hit.group(0))


def test_the_interface_is_written_before_the_body():
    for mod in (TC, TCM):
        for word in ("INTERFACE", "takes", "returns", "Called by"):
            assert word in mod.__doc__, (mod.__name__, word)
    for word in ("THE LINE OF A CLAIM", "THE RESPONSES", "THE GOAL", "HOW A LINE ENDED"):
        assert word in TC.__doc__
    assert "INTERFACE" in (ENGINE / "tactics" / "src" / "lines.rs").read_text()


def test_the_tactics_crate_does_not_depend_on_the_learning_crate():
    assert "cs_learn" not in (ENGINE / "tactics" / "Cargo.toml").read_text()
