"""PLAN 78.68 (first run), SYSTEM_SPEC 9: the ladder's rungs on built cases, the wall (no oracle field in the list), and the scan of the ladder's files for names of the game."""
import json
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import ladder as LD

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "catspace/incubator/analyst/discovery/ladder.py", ROOT / "catspace/incubator/analyst/discovery/ladder_measure.py"]
SENTINEL = "ZZ9SENTINEL"


class Fake:
    """Readers of our own reasoning with scripted answers: `at` maps a clock to the equal set the search holds there."""

    def __init__(self, at, heads=None, forced=None):
        self.at, self.hd, self.fc = at, heads or {}, forced or {"kind": "none", "first": None}
        self.asked = []

    def search(self, position, seconds):
        self.asked.append(seconds)
        eq = self.at[seconds]
        return {"move": eq[0], "equals": list(eq), "nodes": 10, "depth": 3, "proved": False}

    def forced(self, position):
        return dict(self.fc)

    def heads(self, position, equals):
        return dict(self.hd)

    def line(self, position, scale):
        return {"names": ["a", "b"], "events": [(False, "k1")], "end": 0, "depth": 2, "cut": False, "nodes": 5}

    def features(self, position, deep):
        return {"frame": ["us_lost"], "phase": ["mid"], "line": ["wins_k1"], "our_move": ["quiet"]}


LIM = {"wake_seconds": 0.2, "ladder": (1.0, 4.0, 16.0), "head": "arrival"}
POS = {"moves": ["m1", "m2"]}


def test_rung_1_when_every_equal_move_is_accepted_and_the_ladder_stops():
    t = Fake({0.2: ["x"]})
    r = LD.ladder(POS, [SENTINEL, "x"], t, LIM)
    assert r["rung"] == 1 and t.asked == [0.2] and r["features"] is None


def test_rung_1_by_the_forcing_read():
    t = Fake({0.2: ["x", "y"]}, forced={"kind": "proved", "first": "y"})
    assert LD.ladder(POS, ["y"], t, LIM)["rung"] == 1
    t = Fake({0.2: ["x", "y"], 1.0: ["x", "y"]}, forced={"kind": "suspected", "first": "y"})   # a suspected line is not a proof
    assert LD.ladder(POS, ["y"], t, dict(LIM, ladder=(1.0,)))["rung"] == 0


def test_rung_2_the_head_named_decides_and_the_others_are_measured():
    t = Fake({0.2: ["x", "y", "z"]}, heads={"arrival": "y", "trees": "x", "linear": "y"})
    r = LD.ladder(POS, ["y"], t, LIM)
    assert r["rung"] == 2 and t.asked == [0.2]
    assert r["tie"]["eligible"] and r["tie"]["hits"] == {"arrival": True, "trees": False, "linear": True}
    assert abs(r["tie"]["random_hit"] - 1 / 3) < 1e-9 and r["tie"]["control_hit"] is False


def test_rung_3_the_search_at_a_longer_clock_finds_it_and_the_clock_is_named():
    t = Fake({0.2: ["x", "y"], 1.0: ["x", "y"], 4.0: ["y"], 16.0: ["y"]}, heads={"arrival": "x", "trees": "x", "linear": "x"})
    r = LD.ladder(POS, ["y"], t, LIM)
    assert r["rung"] == 3 and r["found_at"] == 4.0 and t.asked == [0.2, 1.0, 4.0]
    assert r["features"]["frame"] == ["us_lost"] and r["deep"]["names"] == ["a", "b"]
    assert r["cost"][3]["queries"] >= 3


def test_rung_3_when_the_accepted_move_was_not_among_the_equal_moves():
    t = Fake({0.2: ["x"], 1.0: ["x"], 4.0: ["x"], 16.0: ["y"]})
    r = LD.ladder(POS, ["y"], t, LIM)
    assert r["rung"] == 3 and r["found_at"] == 16.0 and not r["tie"]["eligible"]


def test_rung_0_when_the_last_clock_does_not_find_it():
    t = Fake({0.2: ["x"], 1.0: ["x"], 4.0: ["x"], 16.0: ["x"]})
    r = LD.ladder(POS, ["y"], t, LIM)
    assert r["rung"] == 0 and r["found_at"] is None and r["deep"] is not None


def test_the_wall_no_oracle_move_or_field_reaches_the_result_the_row_or_the_file(tmp_path):
    t = Fake({0.2: ["x"], 1.0: ["x"], 4.0: ["x"], 16.0: ["x"]})
    r = LD.ladder(POS, [SENTINEL], t, LIM)
    row = LD.surprise_row("p1", "own", POS["moves"], r)
    assert set(row) <= set(LD.ROW_FIELDS) and LD.wall_violations(row) == []
    f = tmp_path / "list.jsonl"
    LD.write_surprise_list(f, [row])
    text = f.read_text()
    assert SENTINEL not in text and SENTINEL not in json.dumps(r)
    for bad in ("score", "eval", "oracle", "centipawn", "multipv", "accepted", "best_move"):
        assert not re.search(bad, text, re.I), bad
    for poisoned in (dict(row, score=12), dict(row, accepted=["x"]), dict(row, deep=dict(row["deep"], eval=3)), dict(row, features={"frame": ["oracle says"]})):
        assert LD.wall_violations(poisoned)
        with pytest.raises(ValueError):
            LD.write_surprise_list(tmp_path / "bad.jsonl", [poisoned])


def test_the_real_files_of_the_run_hold_no_oracle_field():
    root = ROOT / "artifacts/experiments/ladder_78_68/full"
    lst = root / "surprise_list.jsonl"
    if not lst.exists():
        pytest.skip("no run on disk")
    rows = [json.loads(x) for x in lst.read_text().splitlines() if x.strip()]
    assert rows and all(LD.wall_violations(r) == [] for r in rows)
    ans = {json.loads(x)["id"]: set(json.loads(x)["accepted"]) for x in (root / "answers.jsonl").read_text().splitlines() if x.strip()}
    assert not any("accepted" in r for r in rows)
    res = (root / "results.jsonl").read_text()
    assert not re.search(r"oracle|centipawn|multipv|\"score\"|accepted", res, re.I)
    assert ans and set(ans) >= {r["id"] for r in rows}


def test_groups_largest_first_and_a_feature_everyone_has_is_not_a_key():
    rows = []
    for i in range(6):
        rows.append({"id": f"a{i}", "rung": 3, "features": {"frame": ["us_safe", "them_lost"], "phase": ["many"], "our_move": ["taking"]}})
    for i in range(3):
        rows.append({"id": f"b{i}", "rung": 0, "features": {"frame": ["us_safe"], "phase": ["few"], "our_move": ["quiet"]}})
    g = LD.groups(rows)
    assert g[0]["size"] >= g[-1]["size"] and g[0]["size"] == 6
    assert not any("frame:us_safe" in grp["key"] for grp in g)                    # in every row: separates nothing
    assert sum(x["size"] for x in g) == 9


def test_the_ladders_files_name_no_object_of_the_game():
    words = re.compile(r"\b(fen|san|uci|check|mate|king|queen|rook|bishop|knight|pawn|chess)s?\b", re.I)
    square = re.compile(r"\b[a-h][1-8]\b")
    for p in FILES:
        text = p.read_text()
        assert words.search(text) is None, (p.name, words.search(text).group(0) if words.search(text) else None)
        assert square.search(text) is None, (p.name, square.search(text).group(0) if square.search(text) else None)
