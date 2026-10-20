"""Plan 53: the question log and the prior that orders the questions (it never decides)."""
import chess

from catspace.incubator.analyst.discovery import question_prior as QP
from catspace.incubator.analyst.discovery import regress as R


def _q(goal, stake, **pat):
    return R.new_question(goal, stake, dict(pat), text=f"{goal} {stake}")


def test_features_are_generic_words_with_no_square_names():
    b = chess.Board("2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 6 62")
    f = QP.features(_q("win", 5, target=chess.C1), b, "w", 0)
    assert f["goal=win"] == 1 and f["stake"] == 5 and f["target=rook"] == 1 and f["target_ours"] == 0
    assert f["target_attacked_by_us"] == 0 and f["target_guarded"] == 0 and f["in_check"] == 0
    assert all(not any(ch.isdigit() for ch in k.split("=")[-1]) for k in f if "=" in k)        # no square names: the prior must read shapes, not places


def test_score_is_zero_without_a_model_and_linear_with_one():
    f = {"goal=mate": 1, "stake": 100}
    assert QP.score(f, None) == 0.0
    assert QP.score(f, {"weights": {"goal=mate": 2.0, "stake": 0.01}, "bias": -1.0}) == 2.0


def test_labels_mark_the_deciding_question_and_the_ones_asked_before_it():
    rows = [dict(order=i, decided=(i == 2), decision="b6h6", sf="RIGHT", match=True, answer="b6h6", target_sq=None, move=None, features={}) for i in range(5)]
    lab = QP._label(rows)
    assert [l for _, l in lab] == [0, 0, 1]                      # the two asked before it are 0, it is 1, the ones after are not used


def test_a_wrong_decision_labels_the_question_naming_the_answer_target():
    ans = "d1d5"
    rows = [dict(order=0, decided=True, decision="e2e4", sf="WRONG", match=False, answer=ans, target_sq=chess.A1, move=None, features={}),
            dict(order=1, decided=False, decision="e2e4", sf="WRONG", match=False, answer=ans, target_sq=chess.D5, move=None, features={})]
    assert [l for _, l in QP._label(rows)] == [0, 1]
    rows[1]["target_sq"] = chess.H8
    assert QP._label(rows) == []                                 # no question named the answer: nothing to learn from this agenda
