"""PLAN 78.68 stage G (parts 1 to 3) and stage H as the score, SYSTEM_SPEC 9 and 6: the pairs on a built case, the wall (pairs and fit run with no answers file and no oracle; the answers are read only by score), the
split without overlap, stage H's counts on built cases, the leaf pick, and the scan of the new files for names of the game."""
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import deep_heads as DH

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "catspace/incubator/analyst/discovery/deep_heads.py", ROOT / "catspace/incubator/analyst/discovery/deep_heads_measure.py"]
NCOL = 6


class FakeTools:
    """Readers of our own reasoning with scripted answers. Moves are words of no game: the leaf of a position is its moves and the scripted line."""

    def __init__(self, cut_ids=(), end_ids=()):
        self.cut, self.end = set(cut_ids), set(end_ids)
        self.skipped = __import__("collections").Counter()
        self.played = []

    def leaf_of(self, moves, clock):
        if moves[-1] in self.end:
            return {"leaf": list(moves) + ["w1"], "end": 1, "cut": False, "names": ["w1"], "nodes": 1}
        return {"leaf": list(moves) + ["w1", "w2"], "end": 0, "cut": False, "names": ["w1", "w2", "w3"], "nodes": 1}

    def inputs(self, moves):
        return np.arange(NCOL, dtype=np.float32) + len(moves)

    def quiet(self, moves):
        return True

    def mover_of(self, moves):
        return "first" if len(moves) % 2 == 0 else "second"

    def ended(self, moves):
        return moves[-1] == "over"

    def read_best(self, moves, clock):
        if moves[2] in self.cut:
            return {"names": ["d1"], "settled": 1, "end": 0, "cut": True, "nodes": 9}
        return {"names": ["d1", "d2", "over"], "settled": 3, "end": 1, "cut": False, "nodes": 9}

    def target_of(self, moves, leaf_ply):
        n = len(moves) - leaf_ply
        return {"us": [n, False, False], "them": [1, True, False], "kept": {"us": [], "them": ["k"]}}

    def playouts(self, leaf, gid, k, seconds, plies):
        return [{"moves": list(leaf) + ["p1", "p2"], "ended": False} for _ in range(k)]


def positions():
    return [{"id": "own:g1:10", "source": "own", "band": "mid", "units": 12, "moves": ["a", "b", "c", "d"]},
            {"id": "human:g2:12", "source": "human", "band": "few", "units": 6, "moves": ["a", "b", "x", "d", "e"]}]


def test_pairs_on_a_built_case_the_deep_reading_makes_the_target_where_it_settles_and_playouts_where_it_is_cut():
    t = FakeTools(cut_ids={"x"})
    rows = DH.pairs(positions(), DH.WAKE_CLOCK, DH.DEEP_CLOCK, t, control_every=10 ** 9)
    first = [r for r in rows if r["pid"] == "own:g1:10"]
    assert [r["made_by"] for r in first] == ["deep"] and first[0]["anchor"] is True        # the deep line ends in the game's end
    assert first[0]["rec"] == 3 and first[0]["target"]["us"][0] == 3 and first[0]["kept"]["them"] == ["k"]
    assert len(first[0]["x"]) == NCOL and first[0]["leaf_ply"] == 6
    second = [r for r in rows if r["pid"] == "human:g2:12"]
    assert [r["made_by"] for r in second] == ["playout"] * DH.PLAYOUTS and not any(r["anchor"] for r in second)
    assert len({r["id"] for r in rows}) == len(rows)


def test_a_wake_line_that_ends_the_game_gives_no_row_and_is_counted():
    t = FakeTools(end_ids={"d"})
    rows = DH.pairs(positions()[:1], DH.WAKE_CLOCK, DH.DEEP_CLOCK, t)
    assert rows == [] and sum(t.skipped.values()) == 1


def test_the_wall_pairs_and_fit_run_with_no_answers_file_and_the_modules_name_no_oracle(monkeypatch):
    real = Path.read_text

    def no_answers(self, *a, **k):
        assert "answers" not in str(self), f"pairs or fit read {self}"
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", no_answers)
    rows = DH.pairs(positions(), DH.WAKE_CLOCK, DH.DEEP_CLOCK, FakeTools(cut_ids={"x"}))
    assert rows
    h = DH.fit(synthetic_rows(300), {"columns": list("abcdef"), "mean_rating": 1500.0}, source="deep", say=lambda m: None)
    assert h["source"] == "deep"
    for fn in (DH.pairs, DH.fit, DH.arrays_of, DH.prepare, DH.leaf_pick, DH.root_pick):
        src = inspect.getsource(fn)
        assert not re.search(r"answers|accepted|oracle", src, re.I), fn.__name__
    assert "ladder_oracle" not in sys.modules
    assert "answers" in inspect.signature(DH.score).parameters                               # the one place the answers come in
    for row in rows:
        assert not any(re.search(r"oracle|accepted|answer|score|eval|centipawn", k, re.I) for k in row)


def synthetic_rows(n=500, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        x = rng.normal(size=NCOL)
        p = 1 / (1 + np.exp(-(x[0] - 1.0)))
        ev = rng.random() < p
        d = int(rng.integers(0, 8)) if ev else 12
        ev2 = rng.random() < 0.3
        d2 = int(rng.integers(0, 8)) if ev2 else 12
        rows.append({"id": f"p{i}#0", "pid": f"p{i}", "x": [float(v) for v in x], "made_by": "deep", "anchor": False, "rec": 12,
                     "target": {"us": [d, bool(ev), False], "them": [d2, bool(ev2), False]}})
    return rows


def test_fit_gives_the_same_head_forms_as_78_67_and_a_learned_signal():
    rows = synthetic_rows()
    h = DH.fit(rows, {"columns": list("abcdef"), "mean_rating": 1500.0}, source="deep", say=lambda m: None)
    assert set(h) >= {"linear", "trees", "active", "columns", "horizon", "source"} and h["horizon"] == DH.RECORD_PLIES
    for rel in ("us", "them"):
        assert set(h["linear"][rel]) >= {"w", "b", "mean", "scale", "lam"} and "clf" in h["trees"][rel]
    a = DH.arrays_of(rows)
    lo = DH.loss_rows(h, a, np.arange(len(rows)))
    assert lo["linear"].sum() / lo["steps"].sum() < 0.6                                       # it learned the planted signal (the base rate alone is higher)
    hi = DH.race_number(h, "linear", np.array([[3.0] + [0.0] * (NCOL - 1)], np.float32), 40)
    lw = DH.race_number(h, "linear", np.array([[-3.0] + [0.0] * (NCOL - 1)], np.float32), 40)
    assert hi > lw                                                                            # a larger first input: the side to move wins a unit first more often


def test_the_split_has_no_overlap_and_is_by_game():
    ids = [f"own:g{i}:{k}" for i in range(40) for k in (5, 9)] + [f"human:h{i}:{k}" for i in range(40) for k in (7,)]
    parts = {i: DH.split_part(i) for i in ids}
    assert {"scored", "fit"} == set(parts.values())
    by_game = {}
    for i, p in parts.items():
        by_game.setdefault(DH.game_key(i), set()).add(p)
    assert all(len(v) == 1 for v in by_game.values())                                         # a game is wholly in one part
    lad = ROOT / "artifacts/experiments/ladder_78_68/full/positions.jsonl"
    mine = ROOT / "artifacts/experiments/deep_heads_78_68/full/positions.jsonl"
    if lad.exists() and mine.exists():
        used = {DH.game_key(json.loads(x)["id"]) for x in lad.read_text().splitlines() if x.strip()}
        pair_games = {DH.game_key(json.loads(x)["id"]) for x in mine.read_text().splitlines() if x.strip()}
        assert not used & pair_games                                                          # no position used for pairs shares a game with any of the 400


def prep(eq, move, forced=None):
    return {"eq": eq, "move": move, "forced": forced or {"kind": "none", "first": None}}


def test_the_rungs_of_the_cheap_ladder_are_rerun_by_the_candidate_and_counted():
    acc = {"a", "b"}
    assert DH.rung_under(prep(["a"], "a"), acc, lambda p: "z") == 1
    assert DH.rung_under(prep(["a", "c"], "c", {"kind": "proved", "first": "b"}), acc, lambda p: "z") == 1
    assert DH.rung_under(prep(["a", "c"], "c"), acc, lambda p: "a") == 2
    assert DH.rung_under(prep(["a", "c"], "c"), acc, lambda p: "c") == 3
    assert DH.rung_under(prep(["c", "d"], "c"), acc, lambda p: "c") == 3                      # the accepted move is not held equal: no head can reach it


def test_score_counts_resolved_on_the_open_and_created_on_the_resolved_and_only_on_the_scored_part():
    ids = [f"own:g{i}:3" for i in range(60)]
    scored = [i for i in ids if DH.split_part(i) == "scored"]
    other = [i for i in ids if DH.split_part(i) == "fit"]
    assert scored and other
    preps = {i: prep(["a", "c"], "c") for i in ids}
    ans = {i: ["a"] for i in ids}
    good = {"name": "good", "pick": lambda pid, p: "a", "cost": lambda p: 0.5, "random": False}
    bad = {"name": "bad", "pick": lambda pid, p: "c", "cost": lambda p: 0.25, "random": False}
    half = scored[: len(scored) // 2]
    s = DH.score(good, preps, half, scored[len(scored) // 2:] + other, ans, "scored")
    assert s["n_open"] == len(half) and s["resolved"] == len(half) and s["created"] == 0 and s["cost"] == 0.5
    s = DH.score(bad, preps, half, scored[len(scored) // 2:] + other, ans, "scored")
    assert s["resolved"] == 0 and s["created"] == s["n_resolved_sample"] > 0                   # equal numbers are drawn from the resolved
    assert all(DH.split_part(i) == "scored" for i in half)
    rnd = DH.score({"name": "r", "pick": None, "cost": lambda p: 0.0, "random": True}, preps, half, scored[len(scored) // 2:], ans, "scored")
    assert abs(rnd["resolved"] - 0.5 * len(half)) < 1e-9
    h = DH.hit_rate(good, preps, ids, ans, "scored")
    assert h["hits"] == h["n"] == len(scored) and h["lo"] <= h["rate"] <= h["hi"]


def test_leaf_pick_scores_the_head_at_the_leaf_for_the_mover_and_a_line_that_ends_the_game_scores_by_the_end(monkeypatch):
    nums = {1.0: 0.30, 2.0: 0.10}                                                             # the head's number at a leaf, by the first input

    def fake(heads, kind, x, remaining):
        return nums[float(x[0, 0])]
    monkeypatch.setattr(DH, "race_number", fake)
    pr = {"move": "m1", "leaf": {"m1": {"x": np.array([1.0], np.float32), "end": 0, "plies": 2, "line": []},
                                 "m2": {"x": np.array([2.0], np.float32), "end": 0, "plies": 2, "line": []}}}
    pos = {"moves": ["w"] * 10}
    assert DH.leaf_pick(pos, ["m1", "m2"], {}, pr) == "m1"                                    # an even count of plies: the number is the mover's own
    pr["leaf"]["m1"]["plies"], pr["leaf"]["m2"]["plies"] = 3, 3
    assert DH.leaf_pick(pos, ["m1", "m2"], {}, pr) == "m2"                                    # an odd count: the number is the other side's, so the lower one is better
    pr["leaf"]["m2"] = {"x": None, "end": 1, "plies": 3, "line": []}                          # the line ends the game for the side to move after it: the worst
    assert DH.leaf_pick(pos, ["m1", "m2"], {}, pr) == "m1"
    pr["leaf"]["m2"] = {"x": None, "end": 2, "plies": 3, "line": []}                          # ... or in the mover's favour: the best
    assert DH.leaf_pick(pos, ["m1", "m2"], {}, pr) == "m2"


def test_the_new_files_name_no_object_of_the_game():
    words = re.compile(r"\b(fen|san|uci|check|mate|king|queen|rook|bishop|knight|pawn|chess)s?\b", re.I)
    square = re.compile(r"\b[a-h][1-8]\b")
    for p in FILES:
        text = p.read_text()
        assert words.search(text) is None, (p.name, words.search(text).group(0) if words.search(text) else None)
        assert square.search(text) is None, (p.name, square.search(text).group(0) if square.search(text) else None)
