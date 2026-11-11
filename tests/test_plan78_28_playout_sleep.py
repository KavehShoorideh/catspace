"""Plan 78.28: playouts to a real result, diagnosed at sleep (playout_sleep.py, cs_engine.playout_game, the problem file, the cycle option of initialize.py). Fast: a fake judge, tmp dirs, no
outside engine (the one Stockfish test skips without it)."""
import json
import re
import shutil
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import initialize as IN
from catspace.incubator.analyst.discovery import playout_sleep as PS
from catspace.incubator.analyst.discovery import wdl_sleep as WS

ROOT = Path(__file__).resolve().parents[1]
STOCKFISH = "/opt/homebrew/bin/stockfish"
SANDBOX = ROOT / "data" / "derived" / "init_store_w27"          # read only here: copied to tmp before use
CAPTURE = {"worth": {}, }


class FakeJudge(WS.WdlJudge):
    """Verdicts by a function of the moves played: `rule(moves) -> (w, d, l)` for the side to move."""

    def __init__(self, rule):
        self.rule, self.asked = rule, []

    def wdl(self, variant, moves, start=None):
        self.asked.append(tuple(moves))
        return self.rule(list(moves))


# ---------------------------------------------------------------------------------------------------------------------------------------- playout_game

def test_a_careful_playout_ends_in_a_result_or_is_capped_and_is_reproducible():
    a = CE.playout_game("chess", None, [], seed=3, max_plies=400)
    assert a["outcome"] is not None and not a["capped"] and a["moves"] and a["named"] == 0 and a["plies"] == len(a["moves"])
    assert CE.playout_game("chess", None, [], seed=3, max_plies=400) == a
    cap = CE.playout_game("chess", None, [], seed=3, max_plies=6)
    assert cap["capped"] and cap["outcome"] is None and len(cap["moves"]) == 6


def test_a_playout_from_a_given_position_continues_it():
    r = CE.playout_game("chess", None, ["e2e4", "e7e5"], seed=1, max_plies=4)
    fen, mover, legal, outcome, _k = CE.state("chess", None, ["e2e4", "e7e5"] + r["moves"])
    assert len(r["moves"]) == 4 and outcome is None


def test_rules_need_a_store_and_an_unknown_rollout_is_refused():
    with pytest.raises(ValueError):
        CE.playout_game("chess", None, [], rollout="rules")
    with pytest.raises(ValueError):
        CE.playout_game("chess", None, [], rollout="nonsense")


@pytest.mark.skipif(not (SANDBOX / "snapshots" / "chess" / "6d926f1d52d5bda0.json").exists(), reason="the sandbox store of plan 78.27 is not there")
def test_rules_playout_reads_the_snapshot_and_counts_the_plies_a_rule_spoke():
    store = str(SANDBOX / "snapshots" / "chess" / "6d926f1d52d5bda0.json")
    specs = GS.spec_texts("chess", "relational")
    games = PS.playout_games([{"start": None, "moves": []}] * 3, variant="chess", store=store, spec=specs, n_per_position=2, seed=5, rollout="rules", max_plies=60)
    assert len(games) == 6 and all(g["rollout"] == "rules" for g in games)
    assert all(g["capped"] or g["outcome"] is not None for g in games)
    again = PS.playout_games([{"start": None, "moves": []}] * 3, variant="chess", store=store, spec=specs, n_per_position=2, seed=5, rollout="rules", max_plies=60)
    assert [g["moves"] for g in again] == [g["moves"] for g in games]


# ---------------------------------------------------------------------------------------------------------------------------------------- verdict classes, turning, match share

def test_verdict_class_is_the_largest_share_ties_to_the_lower_class():
    assert PS.verdict_class((0.7, 0.2, 0.1)) == "win" and PS.verdict_class((0.1, 0.2, 0.7)) == "loss" and PS.verdict_class((0.1, 0.8, 0.1)) == "draw"
    assert PS.verdict_class((0.5, 0.5, 0.0)) == "draw" and PS.verdict_class((0.0, 0.5, 0.5)) == "loss"
    assert PS.flip("win") == "loss" and PS.flip("draw") == "draw"
    assert PS.outcome_class([100, 0], 0) == "win" and PS.outcome_class([100, 0], 1) == "loss" and PS.outcome_class([50, 50], 0) == "draw"


def finished_game(prefix, n=40, seed=1):
    """A real finished playout from the position after `prefix`."""
    g = PS.playout_games([{"start": None, "moves": prefix}], variant="chess", store=None, spec={}, n_per_position=1, seed=seed, rollout="careful", max_plies=400)[0]
    assert g["outcome"] is not None
    return g


def test_turning_is_the_first_ply_where_the_movers_class_drops_and_none_when_the_result_matches():
    g = finished_game(["e2e4", "e7e5"])
    n0 = len(g["prefix"])
    # the judge: white (to move at ply 0 of the playout, an even number of moves) is winning until the playout's 3rd move, then it is a draw for everyone
    def rule(moves):
        k = len(moves) - n0
        if k < 3:
            return (0.9, 0.05, 0.05) if len(moves) % 2 == 0 else (0.05, 0.05, 0.9)
        return (0.05, 0.9, 0.05)
    judge = FakeJudge(rule)
    start_role = 0
    g = {**g, "outcome": [50, 50]}                               # the playout drew
    t = PS.turning(g, judge, "chess")
    assert t is not None and t["ply"] == 2 and t["played"] == g["moves"][2] and t["moves"] == g["prefix"] + g["moves"][:2]
    assert (t["before"], t["after"], t["start_class"], t["result_class"]) == ("win", "draw", "win", "draw")
    # a result equal to the start's verdict, or better, has no turning move; neither has a playout that did not finish
    assert PS.turning({**g, "outcome": [100, 0]}, judge, "chess") is None
    assert PS.turning({**g, "outcome": None}, judge, "chess") is None
    assert PS.turning(g, FakeJudge(lambda m: (0.05, 0.9, 0.05)), "chess") is None


def test_turning_caches_the_judge_calls():
    g = {**finished_game([]), }
    g["outcome"] = [0, 100]
    judge = FakeJudge(lambda m: (0.9, 0.05, 0.05) if len(m) % 2 == 0 else (0.05, 0.05, 0.9))
    v = PS.Verdicts(judge, "chess")
    PS.turning(g, judge, "chess", verdicts=v)
    PS.turning(g, judge, "chess", verdicts=v)
    assert len(judge.asked) == len(set(judge.asked)) == v.calls


def test_match_share_counts_results_that_equal_the_start_verdict():
    games = [{"start": None, "prefix": ["e2e4"], "moves": [], "outcome": o, "capped": o is None, "named": 0} for o in ([50, 50], [100, 0], [0, 100], None)]
    # the side to move after e2e4 is black; the judge says draw there
    m = PS.match_share(games, FakeJudge(lambda moves: (0.1, 0.8, 0.1)), "chess")
    assert (m["games"], m["finished"], m["capped"], m["matched"], m["worse"], m["better"], m["share"]) == (4, 3, 1, 1, 1, 1, round(1 / 3, 4))
    m = PS.match_share(games, FakeJudge(lambda moves: (0.8, 0.1, 0.1)), "chess")           # black judged winning: a draw is worse, black's loss is worse, its win matches
    assert (m["matched"], m["worse"], m["better"]) == (1, 2, 0)


def test_extra_positions_are_one_per_distinct_turning_position():
    t = {"start": None, "moves": ["e2e4"], "played": "e7e5"}
    xs = PS.extra_positions_of([t, dict(t), {"start": None, "moves": ["d2d4"], "played": "d7d5"}])
    assert len(xs) == 2 and all(set(x) == {"id", "start", "moves", "played", "group"} for x in xs) and len({x["id"] for x in xs}) == 2


# ---------------------------------------------------------------------------------------------------------------------------------------- the lemma sleep's extra seeds

def test_sleep_with_extra_positions_builds_cards_for_them_and_without_is_unchanged(tmp_path, monkeypatch):
    log = tmp_path / "empty.jsonl"
    log.write_text("")
    root = tmp_path / "store"
    IN.prepare_sandbox("chess", root)
    monkeypatch.setattr(GS, "LEARNED_ROOT", root / "learned")
    monkeypatch.setattr(GS, "SNAPSHOT_ROOT", root / "snapshots")
    from catspace.incubator.analyst.discovery import sleep as SL
    s = SL.Settings(depth=2, threads=1, vocabulary="relational") if (root / "learned" / "chess" / "vocabulary.relational.json").exists() else SL.Settings(depth=2, threads=1)
    base = SL.sleep("chess", str(log), settings=s, say=lambda *_: None)
    assert "extra_positions" not in base["funnel"] and base["funnel"]["cards"] == 0
    extra = [{"id": "playout:0:2", "start": None, "moves": ["e2e4", "e7e5"], "played": "g1f3", "group": "playout:0"}]
    r = SL.sleep("chess", str(log), settings=s, say=lambda *_: None, extra_positions=extra)
    assert r["funnel"]["extra_positions"] == 1 and r["funnel"]["cards"] == 1


# ---------------------------------------------------------------------------------------------------------------------------------------- problems

class FirstMove:
    """Pretend decision: used through monkeypatching CE.solve_batch."""


def fake_solve(moves_for):
    def solve(positions, seconds, **kw):
        assert kw.get("ties") is True and "store" in kw and kw["snapshot"] == "check"
        return [(moves_for(p), False, 1, 0.0, None) for p in positions]
    return solve


def test_check_runs_the_wake_decision_and_asks_the_judge_whether_the_move_keeps_the_verdict(monkeypatch):
    # white to move after e2e4 e7e5: the judge says win for the mover before, and the move g1f3 leaves a position where the side to move (black) is "draw" -> the mover dropped
    def rule(moves):
        if moves == ["e2e4", "e7e5"]:
            return (0.9, 0.05, 0.05)
        if moves == ["e2e4", "e7e5", "g1f3"]:
            return (0.05, 0.9, 0.05)
        if moves == ["e2e4", "e7e5", "d2d4"]:
            return (0.05, 0.05, 0.9)
        return (0.3, 0.4, 0.3)
    monkeypatch.setattr(CE, "solve_batch", fake_solve(lambda p: "g1f3" if p[2] == ["e2e4", "e7e5"] else "d2d4"))
    probs = [{"start": None, "moves": ["e2e4", "e7e5"]}]
    out = PS.check(probs, variant="chess", store=None, spec={}, judge=FakeJudge(rule), seconds=0.01)
    assert out[0]["move"] == "g1f3" and out[0]["keeps"] is False and (out[0]["before"], out[0]["after"]) == ("win", "draw")
    monkeypatch.setattr(CE, "solve_batch", fake_solve(lambda p: "d2d4"))
    out = PS.check(probs, variant="chess", store=None, spec={}, judge=FakeJudge(rule), seconds=0.01)
    assert out[0]["keeps"] is True and out[0]["after"] == "win"


def test_problem_file_round_trip_and_merge(tmp_path):
    path = tmp_path / "learned" / "chess" / "playout_problems.json"
    assert PS.load_problems(path) == []
    rows, new = PS.merge_problems([], [{"start": None, "moves": ["e2e4"], "played": "e7e5"}, {"start": None, "moves": ["e2e4"]}], cycle=1)
    assert len(rows) == 1 and len(new) == 1 and rows[0]["first_seen"] == 1
    PS.save_problems(path, rows)
    assert PS.load_problems(path) == rows
    rows2, new2 = PS.merge_problems(PS.load_problems(path), [{"start": None, "moves": ["e2e4"]}, {"start": None, "moves": ["d2d4"]}], cycle=2)
    assert len(rows2) == 2 and [r["first_seen"] for r in rows2] == [1, 2] and len(new2) == 1
    (tmp_path / "x.json").write_text("{}")
    with pytest.raises(AssertionError):
        PS.load_problems(tmp_path / "x.json")


def res(keeps):
    return {"move": "m", "keeps": keeps, "before": "win", "after": "win" if keeps else "draw", "steps": []}


def test_problem_counts_solved_still_open_and_came_back():
    rows = [{"start": None, "moves": [str(i)], "played": None, "first_seen": 1, "history": []} for i in range(5)]
    # cycle 1: rows 0-3 open before; the sleep solves 0 and 1; row 4 is new in cycle 2
    PS.record(rows[:4], [res(False)] * 4, 1, "before")
    PS.record(rows[:4], [res(True), res(True), res(False), res(False)], 1, "after")
    c1 = PS.problem_counts(rows[:4], 1, 4)
    assert c1 == {"total": 4, "new": 4, "solved_before": 0, "solved_after": 2, "still_open": 2, "came_back": 0}
    # cycle 2: row 0 stays solved, row 1 is open again (came back), row 2 is solved before the sleep, row 3 stays open, row 4 is new and open then solved
    PS.record(rows, [res(True), res(False), res(True), res(False), res(False)], 2, "before")
    PS.record(rows, [res(True), res(True), res(True), res(False), res(True)], 2, "after")
    c2 = PS.problem_counts(rows, 2, 1)
    assert c2 == {"total": 5, "new": 1, "solved_before": 2, "solved_after": 2, "still_open": 1, "came_back": 1}
    lines = []
    PS.say_counts(c2, lines.append)
    assert lines == ["OBS playout_sleep problems: total 5, new 1, solved before 2, solved after 2, still open 1, came back 1"]


def test_the_wake_and_the_snapshot_writer_never_read_the_problem_file():
    for p in (ROOT / "catspace/incubator/analyst/discovery/wake.py", ROOT / "catspace/incubator/analyst/discovery/game_store.py", ROOT / "catspace/incubator/analyst/discovery/wdl_table.py"):
        text = p.read_text()
        assert "playout_problems" not in text and "playout_sleep" not in text and "problems.json" not in text, p.name
    for p in sorted((ROOT / "rust/engine").glob("*/src/**/*.rs")):
        assert "playout_problems" not in p.read_text(), p
    GS_text = (ROOT / "catspace/incubator/analyst/discovery/game_store.py").read_text()
    assert "glob(\"*.json\")" in GS_text                          # lemmas are read from learned/<game>/lemmas/ only: a file one level up is never a lemma


def test_the_rust_playout_piece_names_no_chess_object_and_no_decision_crate_names_a_teacher():
    code = (ROOT / "rust/engine/decision/src/lemma_prior.rs").read_text()
    code = "\n".join(ln.split("//")[0] for ln in code.splitlines())
    assert "RulesFirst" in code and "Teacher" not in code and "cs_learn" not in code
    assert not re.search(r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|mate|stalemate|uci|fen)\b", code, flags=re.I)


# ---------------------------------------------------------------------------------------------------------------------------------------- the cycle, with a fake judge

def test_the_cycle_with_playouts_reports_match_share_and_problems(tmp_path, monkeypatch):
    """run_size with playouts=True on a tiny sandbox: the helpers are called, the problem file lands next to the lemmas, the report carries match shares and counts."""
    store = tmp_path / "store"
    judge = FakeJudge(lambda m: (0.1, 0.8, 0.1))
    monkeypatch.setattr(IN.paths, "experiment", lambda name: str(tmp_path / name))
    lines = []
    from catspace.incubator.analyst.discovery import sleep as SL

    class Src:                                                        # a rank source that ranks every legal move equal: no regret, no seeds
        source, training_only = "fake", True

        def rank(self, variant, moves, cap=None, start=None):
            _f, _m, legal, _o, _k = CE.state(variant, start, list(moves))
            return [(m, 0) for m in sorted(legal)]

        def grade(self, s):
            return s

        def __call__(self, variant, moves, start=None):
            return None

        def close(self):
            pass
    rep = IN.run_size("chess", 3, source=Src(), store_root=store, games_per_start=1, seconds=0.02, max_plies=8, limit=1, vocabulary="relational", tag="t28", cycles=1,
                      sleep_settings={"depth": 2, "threads": 1}, playouts=True, playout_judge=judge, playout_settings={"positions": 3, "per_position": 1, "max_plies": 20, "check_seconds": 0.02},
                      say=lines.append, sample=5)
    cyc = rep["cycles"][0]["playouts"]
    assert set(cyc) >= {"match_before", "match_after", "problems", "problem_file"} and (store / "learned" / "chess" / "playout_problems.json").exists()
    assert any(l.startswith("OBS playout_sleep problems: total") for l in lines) and any(l.startswith("OBS playout_sleep cycle chess: match share before") for l in lines)
    assert cyc["problems"]["total"] == len(PS.load_problems(cyc["problem_file"]))


@pytest.mark.skipif(not Path(STOCKFISH).exists(), reason="no Stockfish")
def test_the_uci_judge_gives_a_class():
    j = WS.UciWdlJudge(STOCKFISH, 6)
    try:
        v = PS.Verdicts(j, "chess")
        assert v.of(None, ["e2e4", "e7e5"]) in PS.CLASSES
    finally:
        j.close()
