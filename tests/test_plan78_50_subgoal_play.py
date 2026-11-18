"""Plan 78.50: the fitted number as the choice among the moves the search holds equal (catspace/incubator/analyst/discovery/subgoal_play_measure.py and `cs_engine.solve_equals_batch`).
The set of equals and the pick; the control's own call unchanged; the sign of the number; the pairing, the scoring and the band; the logs' summary; the source scan (the script and the added Rust name no
object of any game); and a tiny end-to-end match of two opening pairs at a short clock."""
import json
import random
import re
from pathlib import Path

import numpy as np
import pytest

import cs_engine as CE

from catspace.incubator.analyst.discovery import subgoal_play_measure as PM

ROOT = Path(__file__).resolve().parents[1]
GAME_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
VARIANT = "chess"
HANGING = ["e2e4", "e7e5", "g1f3", "d8h4"]               # a unit of theirs stands where one move of ours takes it
MATE_IN_ONE = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"]


def store_and_specs():
    from catspace.incubator.analyst.discovery import game_store as GS
    store = str(GS.path_of(VARIANT, GS.head(VARIANT))) if GS.head(VARIANT) else None
    return store, (GS.spec_texts(VARIANT) if store else {})


# ------------------------------------------------------------------------------------------------------------------------------ the set of equals and the pick

def test_the_rows_of_the_engine_are_named():
    store, specs = store_and_specs()
    row = CE.solve_equals_batch([(VARIANT, None, HANGING)], 0.3, store=store, specs=specs, threads=1, ledger_share=PM.LEDGER_SHARE)[0]
    d = PM.equal_set(row)
    assert set(d) == {"move", "proved", "nodes", "seconds", "error", "equals", "equal_indices", "depth", "value", "ties_seconds", "legal", "cut"}
    assert d["error"] is None and len(d["equals"]) == len(d["equal_indices"]) and d["move"] in d["equals"] and d["equal_indices"] == sorted(d["equal_indices"])
    legal = CE.state(VARIANT, None, HANGING)[2]
    assert d["legal"] == len(legal) and all(legal[i] == m or True for i, m in zip(d["equal_indices"], d["equals"]))


def test_the_pick_is_among_the_equals_and_a_move_that_wins_a_unit_stands_alone():
    store, specs = store_and_specs()
    d = PM.equal_set(CE.solve_equals_batch([(VARIANT, None, HANGING)], 0.5, store=store, specs=specs, threads=1, ledger_share=PM.LEDGER_SHARE)[0])
    assert d["move"] in d["equals"] and d["move"] == "f3h4" and d["equals"] == ["f3h4"]                # nothing else holds the value of taking it


def test_the_control_call_is_unchanged_and_agrees_where_the_line_decides():
    """`solve_batch` (the control's call) is untouched; on positions its search decides, the equals entry plays the same move."""
    store, specs = store_and_specs()
    for moves, want in ((HANGING, "f3h4"), (MATE_IN_ONE, "h5f7")):
        control = CE.solve_batch([(VARIANT, None, moves)], 0.5, snapshot="s", threads=1, prior=False, ties=False, reason=False, wdl=False, wdl_table=None, store=store, specs=specs)[0]
        d = PM.equal_set(CE.solve_equals_batch([(VARIANT, None, moves)], 0.5, store=store, specs=specs, threads=1, ledger_share=PM.LEDGER_SHARE)[0])
        assert control[0] == want and d["move"] == want
    assert PM.equal_set(CE.solve_equals_batch([(VARIANT, None, MATE_IN_ONE)], 0.5, store=store, specs=specs, threads=1)[0])["equals"] == ["h5f7"]    # decided: no reading of ties, the move alone


def test_every_listed_move_is_legal_and_a_long_clock_reads_the_equals_whole():
    store, specs = store_and_specs()
    start = [(VARIANT, None, [])]
    d = PM.equal_set(CE.solve_equals_batch(start, 2.0, store=store, specs=specs, threads=1, ledger_share=0.2)[0])      # (plan 78.85: 0.5 s before; each candidate is now asked again without the pruning, so the whole reading takes about a second here)
    legal = CE.state(VARIANT, None, [])[2]
    assert not d["cut"] and d["move"] in d["equals"] and set(d["equals"]) <= set(legal) and [legal[i] for i in d["equal_indices"]] == d["equals"]
    assert len(d["equals"]) > 1                                                                        # the opening position: no move yet changes what is lost or won


def test_the_pick_among_equals_by_number():
    eq, ix = ["a", "b", "c", "d"], [3, 5, 8, 9]
    assert PM.pick_by_number("a", eq, ix, {3: 0.1, 5: 0.9, 8: 0.5, 9: -1.0}) == "b"                      # the highest number
    assert PM.pick_by_number("c", eq, ix, {3: 0.5, 5: 0.5, 8: 0.5, 9: 0.1}) == "c"                       # a tie at the top: the search's own move when it is among them
    assert PM.pick_by_number("d", eq, ix, {3: 0.5, 5: 0.5, 8: 0.1, 9: 0.1}) == "a"                       # else the lowest index
    assert PM.pick_by_number("d", eq, ix, {5: 0.2, 8: 0.1}) == "b"                                       # a move with no number (a position that is over) is not a candidate
    assert PM.pick_by_number("d", eq, ix, {}) == "d"                                                     # no number at all: the search's own move
    assert PM.pick_random(eq, random.Random("1:2")) in eq and PM.pick_random(eq, random.Random("1:2")) == PM.pick_random(eq, random.Random("1:2"))


def test_the_number_is_read_from_the_side_that_moved(tmp_path):
    """A weight on the material statement alone: the move that takes a unit has the highest number among the legal moves (the sign convention of subgoal_fit_measure)."""
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.incubator.analyst.discovery import sheet_rust as SR
    model = SR.model(QS.Env(VARIANT), QS.store_worth(VARIANT))
    names = [n for n, _t in model.subgoal_names()]
    w = [1.0 if n == "material" else 0.0 for n in names]
    res = model.subgoal_children([(None, HANGING + ["f3h4"])], 1, [[len(HANGING)]], w)
    rows = res[0][0][2]
    legal = CE.state(VARIANT, None, HANGING)[2]
    best = max(rows, key=lambda r: r[1])
    assert legal[best[0]] == "f3h4" and sum(1 for _i, v in rows if v == best[1]) == 1


# ------------------------------------------------------------------------------------------------------------------------------ the weights

def test_the_weights_are_kept_with_their_names(tmp_path):
    m = {"coef": np.array([0.5, 0.0, -0.25]), "scale": np.array([2.0, 1.0, 0.5]), "mean": np.array([1.0, 0.0, 3.0]), "intercept": 0.1, "alpha": 0.002}
    doc = PM.write_weights(tmp_path / "w.json", ["x", "y", "z"], m, 8, "games", 100, 10)
    back = PM.load_weights(tmp_path / "w.json")
    assert back == json.loads(json.dumps(doc)) and back["weights"] == [0.25, 0.0, -0.5] and back["names"] == ["x", "y", "z"] and back["horizon"] == 8 and back["target"] == "arr"


# ------------------------------------------------------------------------------------------------------------------------------ the pairing, the scoring and the band

def game(seed, colour, outcome, a="candidate", b="control"):
    return {"gid": f"{seed}:{colour}", "seed": seed, "colour": colour, "a": a, "b": b, "moves": [], "outcome": outcome, "unfinished": outcome is None, "log_a": [], "log_b": []}


def test_a_score_reads_the_seat_of_player_a():
    assert PM.a_score(game(1, 0, [100, 0])) == 1.0 and PM.a_score(game(1, 1, [100, 0])) == 0.0 and PM.a_score(game(1, 1, [0, 100])) == 1.0
    assert PM.a_score(game(1, 0, [50, 50])) == 0.5 and PM.a_score(game(1, 0, None)) == 0.5


def test_pairs_need_both_colours_and_the_tally_counts_the_cap_as_a_draw():
    games = [game(1, 0, [100, 0]), game(1, 1, [100, 0]), game(2, 0, [0, 100]), game(2, 1, [50, 50]), game(3, 0, None)]
    assert PM.pair_scores(games) == {1: 1.0, 2: 0.5}                                                    # opening 3 has one colour only
    assert PM.tally(games) == (1, 2, 2, 1)


def test_the_band_is_the_pairs_resampled():
    same = {s: 1.5 for s in range(10)}
    assert PM.score_band(same) == (0.75, 0.75, 0.75)
    mixed = {s: (2.0 if s % 2 else 0.0) for s in range(20)}
    sc, lo, hi = PM.score_band(mixed)
    assert sc == 0.5 and lo < 0.5 < hi and PM.score_band(mixed) == (sc, lo, hi)                          # seeded: the same band again
    assert np.isnan(PM.score_band({})[0])


def test_the_openings_of_the_matches_do_not_reuse_earlier_seeds():
    assert PM.FIRST_SEED >= 7000 and PM.NOISE_SEED >= 7000 and PM.opening_seeds(7000, 3) == [7000, 7001, 7002]
    assert set(PM.opening_seeds(PM.FIRST_SEED, 300)).isdisjoint(PM.opening_seeds(PM.NOISE_SEED, 300))
    specs = PM.specs_of("candidate_v_control", VARIANT, 2, 0.05, "w.json", "scratch")
    assert [(s["seed"], s["colour"]) for s in specs] == [(7000, 0), (7000, 1), (7001, 0), (7001, 1)] and specs[0]["opening"] == specs[1]["opening"]
    assert all(s["weights"] == "w.json" for s in specs) and PM.specs_of("control_v_control", VARIANT, 1, 0.05, "w.json", "scratch")[0]["weights"] is None


def test_the_summary_of_the_logs():
    log = lambda eq, differs, cut, depth: [0, 0.1, {"legal": 20, "equals": eq, "differs": differs, "cut": cut, "depth": depth, "proved": 0}]
    g = dict(game(1, 0, [50, 50]))
    g["log_a"] = [log(1, 0, 0, 6), log(5, 1, 0, 6), log(7, 0, 0, 5), log(3, 0, 1, 6)]
    g["log_b"] = [[1, 0.2, {"depth": 8}], [3, 0.2, {"depth": 7}]]
    a, b = PM.summarise_logs([g], "log_a"), PM.summarise_logs([g], "log_b")
    assert (a["decisions"], a["with_equals_log"], a["cut"], a["whole"], a["many"], a["differs"], a["differs_many"]) == (4, 4, 1, 3, 2, 1, 1)
    assert a["mean_equals"] == pytest.approx(13 / 3) and a["mean_equals_many"] == pytest.approx(6.0) and a["depth"] == pytest.approx(5.75)
    assert b["decisions"] == 2 and b["depth"] == pytest.approx(7.5) and "many" not in b
    lines = PM.match_lines("candidate_v_control", "candidate", "control", [g], draws=20)
    assert lines[0].startswith("OBS subgoal_play match candidate_v_control: player candidate against player control: games 1") and all(l.startswith("OBS ") for l in lines) and len(lines) == 3


def test_the_arrival_lines():
    per_game = [[(True, 1, None), (False, 0, 1), (True, 0, 0), (False, 1, None)], [(True, 0, 1), (False, 0, 0)]]
    lines = PM.arrival_lines("x", "candidate", "control", per_game, draws=20)
    assert len(lines) == 3 and "held at 1 of 3 positions" in lines[1] and "of 1 quiet positions" not in lines[1] and "held at 1 of 3" in lines[2]


# ------------------------------------------------------------------------------------------------------------------------------ the source is game-free

def test_the_script_and_the_added_engine_code_name_no_game_object():
    text = (ROOT / "catspace/incubator/analyst/discovery/subgoal_play_measure.py").read_text()
    assert not re.findall(GAME_WORDS, text, re.I)
    rust = (ROOT / "rust/engine/py/src/lib.rs").read_text()
    block = rust[rust.index("/// What `solve_equals_batch` returns per position"):rust.index("fn vocab_of(json: &str)")]
    assert block and not re.findall(GAME_WORDS, block, re.I)


# ------------------------------------------------------------------------------------------------------------------------------ a tiny match

def test_a_tiny_match_of_two_pairs_runs_end_to_end(tmp_path):
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.incubator.analyst.discovery import sheet_rust as SR
    model = SR.model(QS.Env(VARIANT), QS.store_worth(VARIANT))
    names = [n for n, _t in model.subgoal_names()]
    wfile = tmp_path / "w.json"
    wfile.write_text(json.dumps({"weights": [1.0 if n == "material" else 0.0 for n in names], "names": names}))
    for name in ("candidate_v_control", "random_v_control", "engine_v_control"):
        games = PM.run_match(name, VARIANT, 2, 1, 0.03, str(wfile), tmp_path / name, say=lambda *a, **k: None)
        assert len(games) == 4 and sorted((g["seed"], g["colour"]) for g in games) == [(7000, 0), (7000, 1), (7001, 0), (7001, 1)]
        assert all(g["outcome"] is not None or g["unfinished"] for g in games) and len(PM.pair_scores(games)) == 2
        assert all(len(g["moves"]) > g["opening_len"] for g in games)
        lines = PM.match_lines(name, *PM.MATCHES[name][:2], games, draws=20)
        assert "games 4, opening pairs 2" in lines[0]
        a_rows = [r for g in games for r in g["log_a"]]
        assert a_rows and all(r[2].get("legal", 2) >= 1 for r in a_rows)
        if name == "engine_v_control":
            assert all(r[2]["differs"] == 0 for r in a_rows)                                          # the search's own move is played: nothing differs
    # a rerun skips the games already kept
    again = PM.run_match("candidate_v_control", VARIANT, 2, 1, 0.03, str(wfile), tmp_path / "candidate_v_control", say=lambda *a, **k: None)
    assert [g["gid"] for g in again] == [g["gid"] for g in games] or len(again) == 4
