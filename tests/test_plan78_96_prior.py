"""PLAN 78.96 WHERE THE CLASS IS LOST, THE SEARCH OR THE CHOICE; A PRIOR FITTED ON WHICH EQUAL MOVE KEEPS IT (SYSTEM_SPEC 7 the prior, 9 the oracle in sleep and the wall, 11 the layers are game-free): the sample of the labelled positions, the class a move keeps, the room of a
choice, the labelled record, the fitted prior's pick (silent on a tie), the shares with their bands, and the wall (the fitted prior's module never imports the oracle wrapper). The layer files name no game object."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import ladder_oracle as LO
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import prior_measure as PM
from catspace.incubator.analyst.discovery import prior_pick as PP

DIR = Path(PE.__file__).parent
LAYER = ("prior_equal.py", "prior_pick.py", "prior_measure.py", "prior_play.py")
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")


def test_the_layer_files_name_no_game_object():
    for f in LAYER:
        text = (DIR / f).read_text().lower()
        assert not re.search(r"\b[a-h][1-8]\b", text), f
        assert "sys.path.insert" not in text, f
        for w in BANNED:
            assert not re.search(rf"\b{w}\b", text), (f, w)


def test_this_test_file_names_no_game_object_either():
    assert not re.search(r"\b[a-h][1-8]\b", Path(__file__).read_text().lower())


def test_the_fitted_prior_never_imports_the_oracle_wrapper():
    """THE WALL (SYSTEM_SPEC 9): the prior's module and everything it loads at run time leave the oracle wrapper out; it was fitted on stored labels."""
    for f in ("prior_pick.py", "prior_play.py"):
        text = (DIR / f).read_text()
        assert not re.search(r"^\s*(from|import) .*(ladder_oracle|prior_equal|future_labels)", text, flags=re.M), f
        assert "analyse" not in text, f
    for mod in ("prior_pick", "prior_play"):
        code = f"import sys; import catspace.incubator.analyst.discovery.{mod} as P; print('LOADED' if any(m.endswith('ladder_oracle') for m in sys.modules) else 'CLEAN')"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        assert out.stdout.strip().endswith("CLEAN"), (mod, out.stdout, out.stderr[-500:])


# ---- the sample, the class a move keeps, the room

def recs(n_games=6, per=40):
    out = []
    for g in range(n_games):
        out.append({"ord": g, "part": "find" if g % 2 == 0 else "confirm", "positions": [{"ply": 10 + 2 * k, "units": 4 + (k * 3) % 28, "cls": (k % 3) - 1} for k in range(per)]})
    return out


def test_the_sample_is_stratified_capped_and_in_a_fixed_order():
    r = recs()
    a, b = PE.sample(r, 5, 2), PE.sample(r, 5, 2)
    assert a == b
    cells = {}
    for i in a:
        cells.setdefault((i["part"], i["cls"], i["band"]), []).append(i)
    assert all(len(v) <= 5 for v in cells.values())
    for part in ("find", "confirm"):
        for c in (1, 0, -1):
            assert any(k[0] == part and k[1] == c for k in cells)
    per_game = {}
    for i in a:
        per_game.setdefault((i["part"], i["cls"], i["band"], i["ord"]), 0)
        per_game[(i["part"], i["cls"], i["band"], i["ord"])] += 1
    assert max(per_game.values()) <= 2
    assert {i["part"] for i in a} == {"find", "confirm"}


def test_a_move_keeps_the_class_when_it_is_not_lower():
    assert PE.keeps(1, 1) and PE.keeps(0, 1) and PE.keeps(-1, 0) and PE.keeps(-1, -1)
    assert not PE.keeps(1, 0) and not PE.keeps(1, -1) and not PE.keeps(0, -1)


def test_the_room_of_a_choice():
    base = {"cls": 1, "cm": [1, 1, 1]}
    assert PE.differ(base) == {"keep": False, "class": False}
    assert PE.differ({"cls": 1, "cm": [1, 0, 1]}) == {"keep": True, "class": True}
    assert PE.differ({"cls": 1, "cm": [0, -1]}) == {"keep": False, "class": True}                 # none keeps it: the choice has no room, the classes differ
    assert PE.differ({"cls": 0, "cm": [0, 1]}) == {"keep": False, "class": True}                  # both keep it (a class not lower): classes differ, keeping does not


def test_the_labelled_record_is_a_pure_function_of_its_parts():
    item = {"ord": 3, "part": "find", "ply": 20, "units": 12, "band": "mid", "cls": 1}
    s = {"first": "a", "equals": ["b", "a", "c"], "depth": 7}
    r = PE.label_record(item, s, [1, 0, 1], "d", -1, [5, None, 3], {"search": 0.1, "oracle": 1.0})
    assert r["tied"] == ["a", "b", "c"] and r["cm"] == [1, 0, 1] and r["calls"] == 4 and r["id"] == "3_20"
    r2 = PE.label_record(item, s, [1, 0, 1], "b", 0, [5, None, 3], {"search": 0.1, "oracle": 1.0})
    assert r2["calls"] == 3


def test_the_class_after_a_move_that_ends_the_game_is_a_win_for_the_mover():
    a = LO.analyse_after(["f2f3", "e7e5", "g2g4"], "d8h4")
    assert FL.class_of(a) == 1 and a["pv"] == []


# ---- the fitted prior's pick

class Stub:
    def predict_proba(self, z):
        s = z[:, 0]
        p = 1.0 / (1.0 + np.exp(-s))
        return np.stack([1 - p, p], axis=1)


HEAD = {"kind": "linear", "m": Stub(), "mu": np.zeros(2), "sd": np.ones(2)}


def test_the_pick_is_the_greatest_chance_and_silent_on_a_tie():
    tied = ["a", "b", "c"]
    assert PP.Prior.pick_of(tied, np.array([0.1, 0.9, 0.3])) == "b"
    assert PP.Prior.pick_of(tied, np.array([0.5, 0.5, 0.5])) is None
    assert PP.Prior.pick_of(tied, np.array([0.5, 0.5000000001, 0.2])) == "a"                    # a tie within the rounding is no preference: the first of the top
    assert PP.Prior.pick_of(tied, np.array([0.2, 0.7, 0.7])) == "b"


def test_a_position_is_read_against_its_own_mean_and_a_headless_prior_is_silent():
    x = np.array([[1.0, 5.0], [3.0, 5.0]], np.float32)
    assert np.allclose(PP.centred(x), [[-1.0, 0.0], [1.0, 0.0]])
    p = PP.Prior(HEAD, None)
    sc = p.scores_of([{"x": x}])[0]
    assert sc[1] > sc[0]
    silent = PP.Prior({"kind": "tree", "m": None}, None)
    assert PP.Prior.pick_of(["a", "b"], silent.scores_of([{"x": x}])[0]) is None


def test_a_share_has_a_band_over_games():
    vals = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0] * 20)
    game = np.repeat(np.arange(12), 10)
    v, lo, hi = PM.share(vals, game)
    assert lo <= v <= hi and abs(v - vals.mean()) < 1e-9
    assert PM.share(np.array([]), np.array([]))[0] != PM.share(np.array([]), np.array([]))[0]
