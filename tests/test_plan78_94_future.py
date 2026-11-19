"""PLAN 78.94 THE FUTURE JUDGE (SYSTEM_SPEC 7 the future imagination layer, 9 the oracle's labels in sleep, 11 the layers are game-free): the goal as the threat list reads it, the label table, the loss and the heads against planted truths, the wall (no oracle number in a fit),
the split (FIND only), the chooser. The layer files name no game object."""
import copy
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import future_choose as FC
from catspace.incubator.analyst.discovery import future_goals as FG
from catspace.incubator.analyst.discovery import future_imagine as FI
from catspace.incubator.analyst.discovery import future_judge as FJ
from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import future_measure as FM
from catspace.incubator.analyst.discovery import threat_concrete as TC

DIR = Path(FG.__file__).parent
LAYER = ("future_goals.py", "future_labels.py", "future_judge.py", "future_imagine.py", "future_choose.py", "future_measure.py")
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")


def test_the_layer_files_name_no_game_object():
    for f in LAYER:
        text = (DIR / f).read_text().lower()
        assert not re.search(r"\b[a-h][1-8]\b", text), f
        assert "sys.path.insert" not in text, f
        for w in BANNED:
            assert not re.search(rf"\b{w}\b", text), (f, w)


# ---- the goal

REC = TC.Record({"p": 1, "q": 3}, {}, "test")


def row(place, kind, replies, net, verdict="CONFIRMED", two=False):
    return {"goal": False, "verdict": verdict, "two_moves": two, "place_ix": place, "kind": kind, "net": net, "first": (1, place), "response": [{"first": r} for r in replies]}


def test_two_threats_that_no_one_reply_meets_are_a_goal():
    c = {"mover": [], "other": [row(1, "p", [(0, 1)], 5), row(2, "q", [(0, 2)], 7)]}
    g = FG.goals_of_list(c, "mover", REC)
    assert len(g["goals"]) == 1
    top = g["goals"][0]
    assert top["kind"] == "fork" and top["stake"] == 5 and top["shape"] == ["q", "p"] and top["n_pairs"] == 1
    assert FG.goals_of_list(c, "other", REC)["goals"] == []                 # the threats are against the other side's units: nothing for the other side


def test_a_common_reply_the_same_unit_or_a_cut_line_is_no_goal():
    common = {"mover": [], "other": [row(1, "p", [(0, 1), (3, 3)], 5), row(2, "q", [(3, 3)], 7)]}
    assert FG.goals_of_list(common, "mover", REC)["goals"] == []
    same = {"mover": [], "other": [row(1, "p", [], 5), row(1, "q", [], 7)]}
    assert FG.goals_of_list(same, "mover", REC)["goals"] == []
    cut = {"mover": [], "other": [row(1, "p", [], 5), row(2, "q", [], 7, verdict="UNREAD"), row(3, "q", [], 7, two=True)]}
    g = FG.goals_of_list(cut, "mover", REC)
    assert g["goals"] == [] and g["unread"] == 1 and g["deferred"] == 1 and g["threats"] == 1


def test_a_unit_with_no_reply_beside_another_threat_is_an_open_goal():
    c = {"mover": [row(4, "p", [], 2), row(5, "p", [(9, 9)], 2)], "other": []}
    g = FG.goals_of_list(c, "other", REC)
    assert g["goals"][0]["kind"] == "open"


# ---- the labels

def pos(ply, cls=0, mover_goal=False, other_goal=False, mover=0, units=20):
    return {"ply": ply, "units": units, "n_us": units // 2, "mover": mover, "cp": 17, "forced_in": None, "cls": cls, "pv": [],
            "mover_g": {"goal": mover_goal}, "other_g": {"goal": other_goal}}


def toy(part="find", n_moves=60):
    return {"ord": 1, "part": part, "y": 0, "n_moves": n_moves, "positions": [pos(10), pos(12), pos(14, mover_goal=True), pos(16, cls=1), pos(18, cls=-1)]}


def test_the_targets_look_forward_within_k_plies_and_never_past_the_end():
    rows = FJ.rows_from_labels([toy()], ks=(4, 8))
    mover = rows["mover_side"] == 1
    at10 = (rows["ply"] == 10) & mover
    assert rows["G4"][at10][0] == 1                                          # a goal of the mover at ply 14, within 4 plies of 10
    assert rows["W4"][at10][0] == 0 and rows["W8"][at10][0] == 1             # the mover is winning at 16: within 8 plies, not within 4
    other10 = (rows["ply"] == 10) & ~mover
    assert rows["W8"][other10][0] == 1                                       # the class at 18 is losing for the mover: a winning point for the other side
    assert rows["now_G"][(rows["ply"] == 14) & mover][0] == 1
    short = FJ.rows_from_labels([toy(n_moves=15)], ks=(8,))
    assert short["G8"][(short["ply"] == 10) & (short["mover_side"] == 1)][0] == -1       # the window leaves the game: no label


def test_no_oracle_number_enters_the_rows_or_any_fit():
    base = FJ.rows_from_labels([toy()], ks=(4, 8))
    rec = toy()
    for q in rec["positions"]:
        q["cp"], q["forced_in"] = 999, 5                                     # the raw number and the forced-end field changed: the class stays
    other = FJ.rows_from_labels([rec], ks=(4, 8))
    for k in base.c:
        assert np.array_equal(base[k], other[k]), k
    for fn in (FJ.rows_from_labels, FJ.side_flags, FJ.pair_rows, FJ.FutureJudge.fit, FC.Chooser.scores):
        src = inspect.getsource(fn)
        assert '"cp"' not in src and "'cp'" not in src and "forced_in" not in src, fn.__name__
    for f in ("future_judge.py", "future_choose.py", "future_goals.py", "future_imagine.py"):
        text = (DIR / f).read_text()
        assert '["cp"]' not in text and "forced_in\"]" not in text, f


def test_the_class_is_made_from_the_setting_and_the_score_stays_beside_it():
    assert FL.class_of({"cp": FL.WIN_CP, "forced_in": None}) == 1
    assert FL.class_of({"cp": -FL.WIN_CP - 1, "forced_in": None}) == -1
    assert FL.class_of({"cp": FL.WIN_CP - 1, "forced_in": None}) == 0
    assert FL.class_of({"cp": None, "forced_in": 3}) == 1 and FL.class_of({"cp": None, "forced_in": -2}) == -1
    assert FL.FUTURE_DEPTH == 16 and FL.STRIDE == 2


def test_the_games_are_split_by_game_and_the_sample_plies_keep_the_stride():
    g = {"ord": 12345, "moves": ["x"] * 61, "y": 0, "part": "find"}
    pl = FL.plies_of(g)
    assert pl[0] >= FL.MIN_PLY and all(b - a == FL.STRIDE for a, b in zip(pl, pl[1:])) and pl[-1] < 61
    games = [{"ord": i, "moves": [], "y": 0, "part": "find" if i % 2 else "confirm"} for i in range(40)]
    ch = FL.choose_games(games, 5)
    assert len(ch) == 10 and {x["part"] for x in ch[:5]} == {"find"} and {x["part"] for x in ch[5:]} == {"confirm"}
    assert ch == FL.choose_games(copy.deepcopy(games), 5)


# ---- the loss and the heads, against planted truths

def test_the_loss_the_ranking_and_the_calibration_are_what_they_say():
    assert FJ.logloss(np.array([0.5, 0.5]), np.array([0, 1])) == pytest.approx(np.log(2))
    assert FJ.logloss(np.array([0.9, 0.1]), np.array([1, 0])) == pytest.approx(-np.log(0.9))
    assert FJ.brier(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    assert FJ.auc(np.array([0.1, 0.4, 0.8]), np.array([0, 0, 1])) == 1.0
    assert FJ.auc(np.array([0.9, 0.4, 0.1]), np.array([0, 0, 1])) == 0.0
    assert FJ.auc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
    assert np.isnan(FJ.auc(np.array([0.1, 0.2]), np.array([1, 1])))
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    for pred, obs, _n in FJ.calibration(p, y, 5):
        assert abs(pred - obs) < 0.03


def test_the_linear_head_recovers_planted_weights_and_the_tree_a_planted_pair():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 4))
    y = (rng.uniform(size=3000) < 1 / (1 + np.exp(-(2.0 * X[:, 0] - 1.0 * X[:, 1])))).astype(int)
    g = np.arange(3000) // 10
    lin = FJ.fit_linear(X, y, g)
    w = lin["m"].coef_[0] / lin["sd"]
    assert w[0] > 1.0 and w[1] < -0.4 and abs(w[2]) < 0.3 and abs(w[3]) < 0.3
    assert FJ.auc(FJ.predict(lin, X), y) > 0.8
    Z = rng.normal(size=(4000, 3))
    yx = ((Z[:, 0] > 0) ^ (Z[:, 1] > 0)).astype(int)                        # a pair of inputs the linear head cannot read
    gz = np.arange(4000) // 10
    tree = FJ.fit_tree(Z, yx, gz)
    lin2 = FJ.fit_linear(Z, yx, gz)
    assert FJ.auc(FJ.predict(tree, Z), yx) > 0.9 and FJ.auc(FJ.predict(lin2, Z), yx) < 0.6


def test_the_effective_rank_sees_a_collapse():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(500, 8))
    assert FJ.effective_rank(A) > 7.0
    B = np.repeat(rng.normal(size=(500, 1)), 8, axis=1) + 1e-3 * rng.normal(size=(500, 8))
    assert FJ.effective_rank(B) < 2.0


def test_the_judge_is_fitted_on_find_games_only():
    rows = FJ.rows_from_labels([toy(part="confirm")], ks=(4,))
    X = np.zeros((len(rows), 3))
    with pytest.raises(ValueError):
        FJ.FutureJudge().fit(rows, X)
    with pytest.raises(ValueError):
        FJ.FutureJudge().fit_pair(rows, X, {"row": np.arange(len(rows)), "target": np.zeros(len(rows), int), "label": np.zeros(len(rows), int)}, [("x",)], 4)


def test_the_judge_reads_no_worth_statement():
    class Layer:
        def position_table(self, wires, threads):
            return ["a", "b"], np.zeros(sum((len(m) + 1) * 6 for _s, m in wires), np.int16).tobytes()

    def statements(games, threads):
        names = [("moves", "places its units may move to"), ("material", "the worth of its units, summed"), ("loose_worth", "the worth of those loose units")]
        return names, [np.ones((len(g["moves"]) + 1, 6), np.float32) for g in games]
    inp = FJ.Inputs(Layer(), statements)
    out = inp.inputs_of([(None, ["x", "y"])], [[1, 2]])
    assert inp.stat_names == ["moves"] and out["su"].shape == (2, 1) and out["so"].shape == (2, 1)
    X = FJ.side_matrix(out, True)
    assert X.shape == (2, 2 + 2 + 1 + 1 + 1)


# ---- the choice among equal moves

def test_the_chooser_prefers_the_move_that_raises_ours_less_theirs_and_is_silent_on_a_tie():
    ch = FC.Chooser(None, ("either", 8), None)
    assert ch.pick_of(["m1", "m2", "m3"], {"m1": (0.2, 0.1), "m2": (0.4, 0.1), "m3": (0.3, 0.1)}) == "m2"
    assert ch.pick_of(["m1", "m2"], {"m1": (0.2, 0.1), "m2": (0.2, 0.1)}) is None
    assert ch.pick_of(["m1", "m2"], {"m1": (0.3, 0.3), "m2": (0.2, 0.0)}) == "m2"


# ---- the imagine run needs the threat reader's data on disk: skipped where it is not there

def test_imagine_asks_one_question_a_node_and_a_refused_pass_gives_no_goals():
    try:
        im = FI.Imaginer()
    except Exception as e:                                                   # the data the reader needs is untracked
        pytest.skip(f"the threat reader's inputs are not on this machine: {type(e).__name__}")
    out = im.imagine([], "mover", 3)
    assert out["questions"] >= 1 and out["goals"] == [] and not out["refused"]
    out2 = im.imagine([], "other", 3)
    assert out2["questions"] >= 1
