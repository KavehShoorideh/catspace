"""Plan 78.31: the RESULT standard (a mistake is a change of result) beside the score standard: the rank source's result grading (UciTeacher grade_mode), the all-named quantifier of the
gates (cards.eval all_named, abstraction.Gates.all_named), the sleep's Settings.standard, the audit and the regret under it. Fast: fakes, no outside engine."""
import json

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import abstraction as AB
from catspace.incubator.analyst.discovery import detectors as D
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import initialize as IN
from catspace.incubator.analyst.discovery import judges as JG
from catspace.incubator.analyst.discovery import sleep as SL

FEN = "8/8/8/8/2k2P2/8/4K3/8 w - - 9 6"
VOCAB = "relational"


def teacher(mode, **kw):
    t = object.__new__(SL.UciTeacher)               # (no engine: only the grading is read)
    t.grade_mode = mode
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def card_set(values: dict):
    spec, vocab = GS.spec_texts("chess", VOCAB), json.dumps(GS.vocabulary("chess", VOCAB))
    cs = CE.build_cards([("chess", FEN, [], "p", "g")], spec, vocab, 1)
    return cs.revalued([values])


def lemma_naming(*moves_effects):
    """A lemma that always binds (control(us)) and names the moves that REMOVE a cell fact of the given squares."""
    return {"name": "t", "body": [AB.lit("control", ["us"])], "response": {"removed": [AB.lit(e[0], list(e[1:])) for e in moves_effects], "added": []}, "stats": {}, "meta": {}}


def test_result_grading_maps_scores_to_win_draw_loss_and_leaves_the_score_standard_alone():
    r, s = teacher("result"), teacher("score")
    for sc, cls in [(99990, 1), (504, 1), (300, 1), (299, 0), (0, 0), (-2, 0), (-299, 0), (-326, -1), (-99990, -1)]:
        assert r.grade(sc) == cls, sc
    assert r.regret_by_grade and not s.regret_by_grade
    assert s.grade(504) == 5 and s.grade(99990) == 99990                      # (the score grading as it was: a faster mate is strictly better)
    assert r.grade(99990) == r.grade(99000)                                   # a faster mate no longer beats a slower one
    assert teacher("result", decisive=500).grade(400) == 0


def test_the_cache_keeps_raw_scores_so_regrading_does_not_ask_again(tmp_path):
    class Raw:
        source = "raw"
        asked = 0
        def rank(self, variant, moves, cap=None, start=None):
            Raw.asked += 1
            return [("a", 504), ("b", 0)]
    c = JG.Cached(Raw(), tmp_path / "c.json")
    assert c.rank("chess", [], start=None) == [("a", 504), ("b", 0)] and c.rank("chess", [], start=None) and Raw.asked == 1
    assert json.loads((tmp_path / "c.json").read_text() if (tmp_path / "c.json").exists() else (c.save() or (tmp_path / "c.json").read_text()))                  # (raw [move, score] pairs on disk)


def test_all_named_holds_only_when_every_named_move_is_best_and_any_named_is_the_old_reading():
    legal = sorted(CE.state("chess", FEN, [])[2])
    good = {"e2f3"}
    cs = card_set({m: (1 if m in good else 0) for m in legal})
    both = lemma_naming(("cell", "e", "2", "us", "king"))                      # names every king move: one keeps, the others lose
    any_ev = cs.eval(json.dumps(both), None, 64, None)
    all_ev = cs.eval(json.dumps(both), None, 64, None, True)
    assert any_ev["holds"] == 1 and any_ev["refuted"] == 0                      # (a best move is among the named: the old claim)
    assert all_ev["holds"] == 0 and all_ev["refuted"] == 1                      # (a named move is below the best class: refuted)
    only = lemma_naming(("cell", "f", "4", "us", "pawn"))                       # names the pawn move only, which loses
    assert cs.eval(json.dumps(only), None, 64, None, True)["refuted"] == 1


def test_a_card_where_every_move_is_best_is_trivial_not_support_under_all_named_and_chance_is_all_k_best():
    legal = sorted(CE.state("chess", FEN, [])[2])
    cs = card_set({m: 1 for m in legal})
    lem = lemma_naming(("cell", "e", "2", "us", "king"))
    ev = cs.eval(json.dumps(lem), None, 64, None, True)
    assert ev["trivial"] == 1 and ev["fires"] == 0 and ev["holds"] == 0
    old = cs.eval(json.dumps(lem), None, 64, None)
    assert old["fires"] == 1 and old["holds"] == 1                              # (as before: counted as support)
    cs2 = card_set({m: (1 if m in ("e2f3", "e2d3", "e2e3") else 0) for m in legal})
    ev2 = cs2.eval(json.dumps(lemma_naming(("cell", "e", "2", "us", "king"))), None, 64, None, True)
    assert ev2["expected"] < 1.0 and ev2["refuted"] == 1


def test_gates_carry_the_quantifier_and_the_default_is_unchanged():
    assert AB.Gates().all_named is False
    legal = sorted(CE.state("chess", FEN, [])[2])
    cs = card_set({m: (1 if m == "e2f3" else 0) for m in legal})
    lem = lemma_naming(("cell", "e", "2", "us", "king"))
    assert AB.evaluate(cs, lem, None, AB.Gates())["holds"] == 1
    assert AB.evaluate(cs, lem, None, AB.Gates(all_named=True))["refuted"] == 1


def test_settings_standard_sets_the_gates_and_refuses_a_source_that_grades_the_other_way():
    s = SL.apply_standard(SL.Settings(standard="result"), teacher("result"))
    assert s.gates.all_named and s.held.all_named
    assert not SL.apply_standard(SL.Settings(), teacher("score")).gates.all_named
    with pytest.raises(ValueError):
        SL.apply_standard(SL.Settings(standard="result"), teacher("score"))
    with pytest.raises(ValueError):
        SL.apply_standard(SL.Settings(), teacher("result"))
    assert SL.standard_text(s, teacher("result")).startswith("RESULT") and SL.standard_text(SL.Settings(), teacher("score")).startswith("SCORE")


def test_regret_under_the_result_standard_is_a_lower_class_not_a_score_gap():
    r = teacher("result", gap=100)
    ranking = [("a", 504), ("b", 350), ("c", 0), ("d", -400)]              # a and b both win, c draws, d loses
    reg = D.Regret(r)
    assert reg.rule(ranking, "b", 3) is None                                    # a slower win is not a mistake
    assert reg.rule(ranking, "c", 3)["rule"] == "result"
    assert reg.rule(ranking, "d", 3)["rule"] == "result"
    assert D.Regret(teacher("score", gap=100)).rule(ranking, "b", 3) is not None   # (the score standard: 154 cp below the best, beyond its gap: a regret)


def test_run_size_refuses_a_mixed_standard():
    with pytest.raises(ValueError):
        IN.run_size("chess", 3, source=teacher("score"), standard="result")
