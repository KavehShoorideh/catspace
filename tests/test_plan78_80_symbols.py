"""Plan 78.80: symbols that go with winning, fact patterns first (catspace/incubator/analyst/discovery/symbols.py; the stages symbols_measure.py; the cases in ladder_print.py). SYSTEM_SPEC 7 and 9.

What a move does to the pattern counts, read against the brute-force matcher of plan 78.53 on built positions; the pick stays inside the tie set and ties go to the search's first move; the two tests on built tables;
the rule reads no confirm row; FIND and CONFIRM share no game and neither shares one with the ladder's 400 positions; no pick source reads answers; the layer and the measure name no object of any game."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import symbols as SY
from catspace.incubator.analyst.discovery import symbols_measure as SM

ROOT = Path(__file__).resolve().parents[1]
DISC = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"


# ------------------------------------------------------------------------------------------------------------------------------ the deltas on built positions (against the brute-force matcher)

@pytest.fixture(scope="module")
def engine():
    import cs_engine as CE
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    try:
        model = GL.model(HD.GAME, HD.VARIANT)
        goal = G2M.goal_kind_of(HD.VARIANT)
    except FileNotFoundError as e:
        pytest.skip(f"data absent from this checkout: {e}")
    return CE, model, goal, HD


POSITIONS = [["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],
             ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"],
             ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"]]


def test_deltas_equal_the_brute_force_counts_of_the_position_before_and_after_each_tied_move(engine):
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import graph_layer2 as L2
    from catspace.incubator.analyst.discovery import graph_layer2_match as GM
    CE, model, goal, HD = engine
    static = model.static_part()
    kinds = [n[len("kind:"):] for ty, n in static["nodes"] if ty == "kind"]
    wires = [(None, m[:k]) for m in POSITIONS for k in (len(m) - 2, len(m))]
    rows = [[len(w[1])] for w in wires]
    keys = SY.mine(model, wires, rows, goal, 1, (1, 2), (0, 1, 3), 2)
    keys = [k for k in keys if k["support"] >= 2][:60:2] + [k for k in keys if k["size"] == 2 and k["level"] == 3 and k["support"] >= 2][:20]
    assert len(keys) >= 30
    counter = SY.Counter(model, [k["key"] for k in keys], goal, 2, (0, 1, 3), 1)
    pats = [L2.decode(k["bytes"]) for k in keys]
    seen_up = seen_down = 0
    for moves in POSITIONS:
        legal = CE.state(HD.VARIANT, None, moves)[2]
        tied = [legal[0], legal[len(legal) // 2], legal[-1]]
        got = SY.deltas_of({"moves": moves}, tied, counter)
        assert set(got) == set(tied)
        before = CE.state(HD.VARIANT, None, moves)[0]
        items, units = GM.graph_items(GL.graph(model, before, static), goal)
        b = [GM.match_counts(items, units, p, kinds) for p in pats]                      # (the mover's count, the other side's)
        for m in tied:
            after = CE.state(HD.VARIANT, None, moves + [m])[0]
            items2, units2 = GM.graph_items(GL.graph(model, after, static), goal)
            a = [GM.match_counts(items2, units2, p, kinds) for p in pats]                # (the other side's, the mover's): the side to move after the move is the other side
            want = set()
            for c in range(len(pats)):
                for s, now, was in (("mover", a[c][1], b[c][0]), ("other", a[c][0], b[c][1])):
                    if now > was:
                        want.add((c, s, "up"))
                    elif now < was:
                        want.add((c, s, "down"))
            assert got[m] == want, m
            seen_up += sum(1 for e in want if e[2] == "up")
            seen_down += sum(1 for e in want if e[2] == "down")
    assert seen_up > 0 and seen_down > 0                                                 # the comparison was not between empty sets


def test_the_matrix_form_is_the_dict_form_and_has_the_event_layout(engine):
    CE, model, goal, HD = engine
    moves = POSITIONS[0]
    wires = [(None, moves)]
    keys = SY.mine(model, wires, [[len(moves)]], goal, 1, (1,), (0,), 1)
    counter = SY.Counter(model, [k["key"] for k in keys], goal, 1, (0,), 1)
    legal = CE.state(HD.VARIANT, None, moves)[2][:3]
    M = SY.deltas_matrix([{"moves": moves}], [legal], counter)
    d = SY.deltas_of({"moves": moves}, legal, counter)
    assert M.shape == (3, 4 * len(keys))
    for r, m in enumerate(legal):
        assert SY.events_of_row(M.tocsr(), r, len(keys)) == d[m]
    for i in range(4 * len(keys)):
        assert SY.event_index(*SY.event_of(i, len(keys)), len(keys)) == i


def test_shut_patterns_are_left_out_of_the_keys(engine):
    from catspace.incubator.analyst.discovery import graph_layer2 as L2
    CE, model, goal, HD = engine
    wires = [(None, m) for m in POSITIONS]
    keys = SY.mine(model, wires, [[len(m)] for m in POSITIONS], goal, 1, (1, 2), (0,), 1)
    assert keys and all(a[0] != "shut" for k in keys for a in L2.decode(k["bytes"]).atoms)


# ------------------------------------------------------------------------------------------------------------------------------ the pick

def table_of(ev: dict) -> dict:
    return {m: frozenset(e) for m, e in ev.items()}


PATS = [{"column": 0, "side": "mover", "dir": "up", "weight": 1.0}, {"column": 1, "side": "other", "dir": "down", "weight": 0.5}, {"column": 2, "side": "mover", "dir": "down", "weight": -2.0}]


def test_the_pick_is_in_the_tie_set_and_follows_the_weights():
    tied = ["a", "b", "c", "d"]
    tab = table_of({"a": [], "b": [(0, "mover", "up")], "c": [(0, "mover", "up"), (1, "other", "down")], "d": [(2, "mover", "down")]})
    assert SY.symbol_pick({"moves": []}, tied, PATS, None, tab) == "c"
    assert SY.symbol_pick({"moves": []}, tied, [], None, tab) == "a"
    for t in (["b", "c"], ["d", "a"], ["a", "d", "b"]):
        assert SY.symbol_pick({"moves": []}, t, PATS, None, {m: tab[m] for m in t}) in t


def test_ties_go_to_the_search_s_first_move():
    tied = ["first", "x", "y"]
    assert SY.symbol_pick({"moves": []}, tied, PATS, None, table_of({m: [] for m in tied})) == "first"             # silent: nothing carried
    same = table_of({"first": [(0, "mover", "up")], "x": [(0, "mover", "up")], "y": []})
    assert SY.symbol_pick({"moves": []}, tied, PATS, None, same) == "first"
    lead = table_of({"first": [], "x": [(0, "mover", "up")], "y": [(0, "mover", "up")]})
    assert SY.symbol_pick({"moves": []}, tied, PATS, None, lead) == "x"                                            # a tie below the top: the earlier of the tied moves
    assert SY.symbol_pick({"moves": []}, ["only"], PATS, None, table_of({"only": []})) == "only"


def test_the_pick_with_the_engine_stays_inside_the_tie_set(engine):
    CE, model, goal, HD = engine
    moves = POSITIONS[1]
    keys = SY.mine(model, [(None, moves)], [[len(moves)]], goal, 1, (1, 2), (0, 3), 1)
    counter = SY.Counter(model, [k["key"] for k in keys], goal, 2, (0, 3), 1)
    legal = CE.state(HD.VARIANT, None, moves)[2]
    tied = legal[5:11]
    pats = [{"column": c, "side": s, "dir": d, "weight": 0.1 * (c % 7 + 1)} for c in range(0, len(keys), 3) for s in SY.SIDES for d in SY.DIRS]
    assert SY.symbol_pick({"moves": moves}, tied, pats, counter) in tied


# ------------------------------------------------------------------------------------------------------------------------------ the two tests on built tables

def built_rows(n_pos: int = 400, seed: int = 1):
    """positions with 4 tied moves, the first played; event 0 made true by the played move far more often than by the others, event 1 by all alike, event 2 never."""
    rng = np.random.default_rng(seed)
    game, played, r, c = [], [], [], []
    k = 0
    for p in range(n_pos):
        for j in range(4):
            game.append(p // 2)
            played.append(j == 0)
            if rng.random() < (0.7 if j == 0 else 0.1):
                r.append(k)
                c.append(0)
            if rng.random() < 0.3:
                r.append(k)
                c.append(1)
            k += 1
    from scipy import sparse
    E = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(k, 3))
    return E, np.array(game), np.array(played)


def test_test_two_log_ratio_and_interval_on_built_tables():
    E, game, played = built_rows()
    two = SY.rate_table(E, game, played)
    kp, ko = two["kp"], two["ko"]
    assert kp[0] > 200 and ko[0] < 120 and kp[2] == 0 and ko[2] == 0
    want = np.log(((kp + 0.5) / (two["np"] + 1)) / ((ko + 0.5) / (two["no"] + 1)))
    assert np.allclose(two["lr"], want)
    assert two["lo"][0] > 1.0 and two["hi"][0] < 3.0                                     # about log(0.7 / 0.1) = 1.95
    assert two["lo"][1] < 0 < two["hi"][1]                                               # the event every tied move shares: no difference
    assert two["lo"][2] < 0 < two["hi"][2]


def test_test_one_term_has_the_sign_of_what_goes_with_winning_and_a_null_pattern_is_not_picked():
    from scipy import sparse
    rng = np.random.default_rng(3)
    n = 6000
    game = np.arange(n) // 5
    y = np.zeros(n, int)
    off = np.zeros((n, 3))
    good = rng.integers(0, 3, n)                                                         # the mover's count of a pattern that goes with winning
    z = np.stack([0.45 * good, np.zeros(n), -0.45 * good], axis=1)                       # win against loss: 0.9 per instance
    pr = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=q) for q in pr])
    null = rng.integers(0, 3, n)
    off = np.tile(np.log(np.bincount(y, minlength=3) / n), (n, 1))                       # the held head: calibrated on the rows, no pattern in it
    us = sparse.csr_matrix(np.stack([good, null], axis=1).astype(float))
    them = sparse.csr_matrix(np.zeros((n, 2)))
    one = SY.term_table(off, y, game, us, them, [0, 1], 60, 5)
    assert 0.6 < one[0]["term"][0] < 1.2 and one[0]["lo"][0] > 0                         # log-odds of a win against a loss per extra instance: 0.9 in this build
    assert one[1]["lo"][0] < 0 < one[1]["hi"][0]
    assert one[0]["rows"] == int((good > 0).sum())
    f = np.stack([good, np.zeros(n)], axis=1).astype(float)
    a8 = SY._fit_addon(off, y, f, np.ones(n), 8)
    a30 = SY._fit_addon(off, y, f, np.ones(n), 30)
    assert np.allclose(a8, a30, atol=1e-4)                                               # the Newton steps have converged


def built_tables(sign_one: int = 1):
    n_cols = 3
    two = {"kp": np.zeros(4 * n_cols), "ko": np.zeros(4 * n_cols), "np": 100.0, "no": 300.0, "sp": np.full(4 * n_cols, .1), "so": np.full(4 * n_cols, .1), "lr": np.zeros(4 * n_cols),
           "se": np.ones(4 * n_cols), "lo": np.full(4 * n_cols, -1.0), "hi": np.full(4 * n_cols, 1.0)}
    i = SY.event_index(1, "mover", "up", n_cols)
    two["kp"][i], two["ko"][i], two["lr"][i], two["lo"][i], two["hi"][i] = 40, 60, 0.8, 0.3, 1.3
    j = SY.event_index(2, "other", "down", n_cols)                                      # a made-false event that played moves avoid (log ratio below zero) and whose pattern goes with the other side's win
    two["kp"][j], two["ko"][j], two["lr"][j], two["lo"][j], two["hi"][j] = 20, 80, -0.7, -1.2, -0.2
    one = {1: {"rows": 500, "term": [0.05 * sign_one, 0.0], "lo": [0.01, -1], "hi": [0.09, 1]}, 2: {"rows": 500, "term": [0.0, 0.04], "lo": [-1, 0.01], "hi": [1, 0.07]},
           0: {"rows": 500, "term": [0.5, 0.5], "lo": [0.1, 0.1], "hi": [0.9, 0.9]}}
    return n_cols, two, one


def test_the_rule_picks_by_floors_both_tests_and_the_signs():
    n_cols, two, one = built_tables()
    picked, info = SY.find(two, one, n_cols)
    got = {(p["column"], p["side"], p["dir"]) for p in picked}
    assert (1, "mover", "up") in got                                                     # up, log ratio +, term + : agree
    assert (0, "mover", "up") not in got                                                 # no events over the floors for that pattern
    n_cols, two, one = built_tables(sign_one=-1)
    picked2, _ = SY.find(two, one, n_cols)
    assert (1, "mover", "up") not in {(p["column"], p["side"], p["dir"]) for p in picked2}                       # the term of the wrong sign
    assert info["tried"] >= 2


def test_the_rule_for_a_down_event_follows_the_sign_convention():
    n_cols, two, one = built_tables()
    picked, _ = SY.find(two, one, n_cols)
    assert (2, "other", "down") in {(p["column"], p["side"], p["dir"]) for p in picked}


def test_the_floors_are_enforced():
    n_cols, two, one = built_tables()
    i = SY.event_index(1, "mover", "up", n_cols)
    two["kp"][i] = SY.FLOOR_EACH - 1
    assert (1, "mover", "up") not in {(p["column"], p["side"], p["dir"]) for p in SY.find(two, one, n_cols)[0]}
    n_cols, two, one = built_tables()
    one[1]["rows"] = SY.FLOOR_ROWS - 1
    assert (1, "mover", "up") not in {(p["column"], p["side"], p["dir"]) for p in SY.find(two, one, n_cols)[0]}


def test_confirm_passes_only_when_both_hold_with_the_find_signs():
    n_cols, two_f, one_f = built_tables()
    picked, _ = SY.find(two_f, one_f, n_cols)
    two_c = {"kp": np.array([30.0] * len(picked)), "ko": np.array([50.0] * len(picked)), "np": 100.0, "no": 300.0, "sp": np.full(len(picked), .3), "so": np.full(len(picked), .1),
             "lr": np.array([0.6 * p["sign_two"] for p in picked]), "se": np.ones(len(picked)), "lo": np.array([0.2 if p["sign_two"] > 0 else -1.0 for p in picked]),
             "hi": np.array([1.0 if p["sign_two"] > 0 else -0.2 for p in picked])}
    one_c = {p["column"]: {"rows": 400, "term": [0.04 * p["sign_one"]] * 2, "lo": [0.01 if p["sign_one"] > 0 else -0.1] * 2, "hi": [0.1 if p["sign_one"] > 0 else -0.01] * 2} for p in picked}
    done = SY.confirm(picked, two_c, one_c)
    assert done and all(d["passes"] for d in done)
    one_bad = {k: dict(v, lo=[-0.1, -0.1], hi=[0.1, 0.1]) for k, v in one_c.items()}
    assert not any(d["passes"] for d in SY.confirm(picked, two_c, one_bad))               # test one's band holds zero: no pass


# ------------------------------------------------------------------------------------------------------------------------------ the split and the walls

def test_the_split_is_by_game_and_stable_and_the_ladder_s_games_are_excluded():
    games = [{"ord": i, "fit": i % 3 == 0, "arrival": i % 7 == 0, "moves": ["m"], "y": 0, "rm": 1, "ro": 1} for i in range(300)]
    held = SM.held_out(games, {1, 2, 4, 5})
    assert held and all(not g["fit"] and not g["arrival"] and g["ord"] not in {1, 2, 4, 5} for g in held)
    parts = {g["ord"]: SM.part_of(g["ord"]) for g in held}
    assert set(parts.values()) == {"find", "confirm"} and all(SM.part_of(o) == p for o, p in parts.items())
    plies = SM.plies_of(7, 80)
    assert plies == SM.plies_of(7, 80) and len(plies) == SM.PER_GAME and all(p >= SM.MIN_PLY and p < 80 for p in plies) and plies == sorted(plies)
    assert SM.plies_of(7, 10) == [] and len(SM.plies_of(7, 13)) == 3


def test_find_and_confirm_share_no_game_and_neither_shares_one_with_the_ladder_s_400_on_the_files():
    root = SM.out_root("full") if (SM.out_root("full") / "sample.jsonl").exists() else (SM.out_root("short") if (SM.out_root("short") / "sample.jsonl").exists() else None)
    if root is None:
        pytest.skip("no sample file in this checkout")
    try:
        ladder = SM.ladder_ordinals()
    except FileNotFoundError as e:
        pytest.skip(f"the ladder's files are absent: {e}")
    rows = SM.jsonl(root / "sample.jsonl")
    find = {r["game"] for r in rows if r["part"] == "find"}
    confirm = {r["game"] for r in rows if r["part"] == "confirm"}
    assert find and confirm and not (find & confirm)
    assert not ((find | confirm) & ladder)
    assert all(r["played"] in r["tied"] and len(r["tied"]) >= 2 and r["tied"][0] == r["move"] for r in rows)
    head = np.load(root / "head.npz") if (root / "head.npz").exists() else None
    if head is not None:
        games = set(int(x) for x in head["game"])
        assert not (games & ladder)
        for g, f in zip(head["game"][:2000], head["find"][:2000]):
            assert (SM.part_of(int(g)) == "find") == bool(f)


def test_the_choosing_rule_reads_no_confirm_row():
    assert list(inspect.signature(SY.find).parameters) == ["two", "one", "n_cols"]
    src = inspect.getsource(SM.stage_rule)
    assert src.index("SY.find(") < src.index('event_rows(sample, M, "confirm")')              # the rule has run before the confirm rows are cut
    assert src.count('"confirm"') == 1
    body = re.sub(r'""".*?"""', "", inspect.getsource(SY.find), flags=re.S)
    assert "confirm" not in body.lower()


def test_no_pick_source_reads_answers_and_the_ladder_s_are_read_in_one_function():
    for fn in (SY.symbol_pick, SY.deltas, SY.deltas_matrix, SY.deltas_of, SY.find, SY.confirm, SM.score_confirm, SM.ladder_tables, SM.patterns_of, SM.stage_rule, SM.stage_deltas):
        src = inspect.getsource(fn)
        assert not re.search(r"answers|accepted|oracle|ladder_oracle", src, re.I), fn.__name__
    assert not re.search(r"answers|accepted|oracle", (DISC / "symbols.py").read_text(), re.I)
    assert "answers" in inspect.getsource(SM.score_ladder)
    assert "answers.jsonl" not in (DISC / "symbols_measure.py").read_text().split("def score_ladder")[0]
    import sys
    assert "ladder_oracle" not in sys.modules


@pytest.mark.parametrize("name", ["symbols.py", "symbols_measure.py"])
def test_the_layer_and_the_measure_name_no_chess_object(name):
    hit = re.search(CHESS_WORDS, (DISC / name).read_text(), flags=re.I)
    assert hit is None, f"{name}: {hit.group(0)!r}"


def test_the_printing_script_is_the_only_file_that_spells_the_game():
    text = (DISC / "ladder_print.py").read_text()
    assert "import chess" in text and "def symbols_cases" in text
    for name in ("symbols.py", "symbols_measure.py"):
        assert "import chess" not in (DISC / name).read_text()


def test_nothing_is_written_to_the_real_store():
    for name in ("symbols.py", "symbols_measure.py"):
        src = (DISC / name).read_text()
        assert not re.search(r"rules/learned|data/derived|snapshots", src.split('"""', 2)[2]), name
    assert SM.out_root("short").parts[-2:] == ("symbols_78_80", "short")
