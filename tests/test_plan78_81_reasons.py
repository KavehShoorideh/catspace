"""Plan 78.81: the symbols made into fewer, cleaner reasons, the endings filled, a first reading of layer one (catspace/incubator/analyst/discovery/symbols.py, symbols_reasons_measure.py; the cases in ladder_print.py).
SYSTEM_SPEC 7 (subgoals as symbols, reachable has two layers), 9 (the score), 10 (the endings risk).

The choice fit recovers planted weights on built data and drops a planted duplicate; the controls absorb a planted confound; the fit and the duplicate drop read no CONFIRM row and no generated row; ahead() on constructed
positions against an independent walk; one_short against the brute-force matcher; the picks stay in the tie set; the layer and the measure name no object of any game."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from catspace.incubator.analyst.discovery import symbols as SY
from catspace.incubator.analyst.discovery import symbols_reasons_measure as SR

ROOT = Path(__file__).resolve().parents[1]
DISC = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"


# ------------------------------------------------------------------------------------------------------------------------------ built choice data

def planted(P: int = 2500, T: int = 8, F: int = 30, weights: "dict | None" = None, seed: int = 1, confound: bool = False, part: str = "find", source: str = "human") -> "tuple[SY.Choice, np.ndarray]":
    """P positions of T tied moves; F events carried at random (15 pct); the played move drawn from the softmax of the planted weights. With `confound`: a control c with weight 2.0 and an event 0 that is c most of the time
    (and no weight of its own); events 1 and 2 carry weights 1 and -1."""
    rng = np.random.default_rng(seed)
    N = P * T
    E = (rng.random((N, F)) < 0.15).astype(np.float64)
    c = (rng.random(N) < 0.2).astype(np.float64)
    w = np.zeros(F)
    wc = 0.0
    if confound:
        E[:, 0] = np.where(rng.random(N) < 0.9, c, E[:, 0])
        w[1], w[2], wc = 1.0, -1.0, 2.0
    else:
        for k, v in (weights or {3: 1.5, 7: -1.2, 11: 1.0, 19: -1.5}).items():
            w[k] = v
    s = E @ w + wc * c
    pos = np.repeat(np.arange(P), T)
    played = np.zeros(N, bool)
    for p in range(P):
        z = s[p * T:(p + 1) * T]
        pr = np.exp(z - z.max())
        pr /= pr.sum()
        played[p * T + rng.choice(T, p=pr)] = True
    game = pos // 3
    rows = SY.Choice(sparse.csr_matrix(E), c[:, None], ["takes"], pos, played, game, np.array([part] * N), np.array([source] * N))
    return rows, w


def test_the_choice_fit_recovers_planted_weights():
    rows, w = planted()
    fit = SY.fit_choice(rows, controls=(), seed=3)
    got = dict(zip(fit["events"], fit["weights"]))
    for k in (3, 7, 11, 19):
        assert k in got and np.sign(got[k]) == np.sign(w[k]), (k, got)
        assert abs(got[k] - w[k]) < 0.5
    assert len(got) <= 4 + 6                                                       # the penalty leaves few false reasons among 26 events with no weight
    assert fit["lambda"] > 0 and len(fit["cv"]) == SY.LAMBDA_STEPS


def test_a_planted_duplicate_is_dropped_and_the_smaller_pattern_kept():
    rows, _w = planted(P=600, F=12, weights={3: 1.5, 7: -1.2})
    E = rows.E.toarray()
    E = np.hstack([E, E[:, [3]], E[:, [5]]])                                         # column 12 = column 3 exactly, 13 = column 5 with a few flips
    flip = np.random.default_rng(0).choice(len(E), 6, replace=False)
    E[flip, 13] = 1 - E[flip, 13]
    rows = SY.Choice(sparse.csr_matrix(E), rows.C, rows.control_names, rows.pos, rows.played, rows.game, rows.part, rows.source)
    order = [(1, 0, j) for j in range(12)] + [(2, 0, 12), (2, 0, 13)]                # the copies are the larger patterns
    kept, info = SY.drop_duplicates(rows, order, 0.9)
    assert 12 not in kept and 13 not in kept and all(j in kept for j in range(12))
    assert info["exact"] == 1 and info["near"] == 1 and info["in"] == 14 and info["kept"] == 12
    order2 = [(2, 0, j) for j in range(12)] + [(1, 0, 12), (1, 0, 13)]               # the copies are the smaller patterns: they stay, the originals go
    kept2, _ = SY.drop_duplicates(rows, order2, 0.9)
    assert 12 in kept2 and 3 not in kept2 and 13 in kept2 and 5 not in kept2
    zero = sparse.csr_matrix(np.hstack([E, np.zeros((len(E), 1))]))
    kept3, info3 = SY.drop_duplicates(SY.Choice(zero, rows.C, rows.control_names, rows.pos, rows.played, rows.game, rows.part, rows.source), order + [(3, 0, 14)], 0.9)
    assert 14 not in kept3 and info3["empty"] == 1


def test_a_control_absorbs_a_planted_confound():
    rows, _w = planted(P=3000, F=20, confound=True, seed=5)
    without = SY.fit_choice(rows, controls=(), seed=1)
    both = SY.fit_choice(rows, controls=("takes",), seed=1)
    w0 = dict(zip(without["events"], without["weights"]))
    w1 = dict(zip(both["events"], both["weights"]))
    assert w0.get(0, 0.0) > 0.5                                                      # without the control the event that only stands for it carries a weight
    assert abs(w1.get(0, 0.0)) < 0.25                                                # with it, none (or hardly any)
    assert both["controls"]["takes"] > 1.2
    assert w1.get(1, 0) > 0.5 and w1.get(2, 0) < -0.5                                # the real ones stay
    alone = SY.fit_choice(rows, controls=("takes",), events=[], seed=1)
    assert alone["events"] == [] and alone["controls"]["takes"] > 1.0


def test_the_fit_and_the_duplicate_drop_read_no_confirm_row_and_no_generated_row():
    rows, _w = planted(P=100, F=6, weights={3: 1.5})
    bad = SY.Choice(rows.E, rows.C, rows.control_names, rows.pos, rows.played, rows.game, np.array(["find"] * (len(rows.pos) - 1) + ["confirm"]), rows.source)
    for fn in (lambda r: SY.fit_choice(r), lambda r: SY.drop_duplicates(r, [(1, 0, j) for j in range(6)]), SY.require_find):
        with pytest.raises(ValueError):
            fn(bad)
    gen = SY.Choice(rows.E, rows.C, rows.control_names, rows.pos, rows.played, rows.game, rows.part, np.array(["human"] * (len(rows.pos) - 1) + ["generated"]))
    for fn in (lambda r: SY.fit_choice(r), lambda r: SY.drop_duplicates(r, [(1, 0, j) for j in range(6)]), SY.require_find):
        with pytest.raises(ValueError):
            fn(gen)
    for fn in (SY.fit_choice, SY.drop_duplicates):
        body = re.sub(r'""".*?"""', "", inspect.getsource(fn), count=1, flags=re.S)
        assert body.index("require_find(rows)") < body.index("rows.E") if "rows.E" in body else True
        assert body.split("\n", 2)[1].strip().startswith("require_find(rows)") or "require_find(rows)" in body.split(":\n", 1)[1][:80]
        assert "confirm" not in body.lower()


def test_the_measure_builds_the_choice_rows_from_human_find_positions_only():
    sample = [{"id": f"h{i}", "game": i, "ply": 12, "moves": ["m"], "tied": ["a", "b", "c"], "played": "b", "part": "find" if i % 2 == 0 else "confirm", "move": "a", "units": 20} for i in range(10)]
    M = sparse.csr_matrix(np.ones((30, 8)))
    ctl = {"takes": np.zeros(30), "kind": np.zeros(30, int), "kinds": ["x"]}
    rows = SR.choice_rows(sample, M, list(range(8)), ctl, "find")
    assert rows.n_positions() == 5 and set(rows.part) == {"find"} and set(rows.source) == {"human"}
    assert rows.played.sum() == 5 and all(rows.played[3 * p + 1] for p in range(5))
    src = inspect.getsource(SR.choice_rows)
    assert "generated" not in src.split('"""')[2]                                  # no generated row can come in by this path
    assert "fill_head_rows" not in inspect.getsource(SR.stage_choice) and "generated" not in inspect.getsource(SR.stage_choice).split("\n", 1)[1]


def test_picks_stay_in_the_tie_set_and_silence_is_the_search_s_first_move():
    rows, _w = planted(P=200, F=10, weights={3: 1.5, 7: -1.2})
    fit = SY.fit_choice(rows, seed=2)
    pick = SY.score_choice(rows, fit)
    assert pick.shape == (200,) and pick.min() >= 0 and pick.max() < 8
    silent = {"events": [], "weights": [], "controls": {}}
    assert not SY.score_choice(rows, silent).any()                                   # nothing separates the moves: the first, the search's own
    tie = SY.Choice(sparse.csr_matrix(np.ones((6, 1))), np.zeros((6, 1)), ["takes"], np.repeat([0, 1], 3), np.array([0, 1, 0, 0, 0, 1], bool), np.array([0] * 6), np.array(["find"] * 6), np.array(["human"] * 6))
    assert SY.score_choice(tie, {"events": [0], "weights": [1.0], "controls": {}}).tolist() == [0, 0]       # every move carries it: a tie goes to the first


def test_test_one_with_a_shift_of_its_own_for_generated_rows_stays_finite_and_recovers_a_planted_term():
    rng = np.random.default_rng(7)
    n_h, n_g = 3000, 2000
    f = np.concatenate([rng.poisson(1.0, n_h), rng.poisson(3.0, n_g)]).astype(float)
    off = np.zeros((n_h + n_g, 3))
    off[n_h:, 0] += 5.0                                                                  # the head's scores are far from the generated rows' results
    true_term = 0.3
    gen = np.r_[np.zeros(n_h), np.ones(n_g)]
    z = off.copy()
    z[:, 0] += true_term * f - 3.0 * gen                                                 # the generated rows' own shift
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=r) for r in p])
    us = sparse.csr_matrix(f[:, None])
    them = sparse.csr_matrix(np.zeros((len(f), 1)))
    game = np.arange(len(f)) // 4
    two = SY.term_table(off, y, game, us, them, [0], 20, 1, free=gen[:, None])
    assert np.isfinite(two[0]["term"]).all() and abs(two[0]["term"][0] - true_term) < 0.1 and abs(two[0]["term"][1]) < 0.1
    base = SY.term_table(off[:n_h], y[:n_h], game[:n_h], us[:n_h], them[:n_h], [0], 20, 1)                 # without free rows the call is the 78.80 one
    assert abs(base[0]["term"][0] - true_term) < 0.15


# ------------------------------------------------------------------------------------------------------------------------------ layer one on built positions

@pytest.fixture(scope="module")
def env():
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    from catspace.incubator.analyst.discovery import subgoal_goals as SG
    try:
        model = GL.model(HD.GAME, HD.VARIANT)
        goal = G2M.goal_kind_of(HD.VARIANT)
    except FileNotFoundError as e:
        pytest.skip(f"data absent from this checkout: {e}")
    kinds = [n[len("kind:"):] for ty, n in model.static_part()["nodes"] if ty == "kind"]
    return SY.Reach(model, HD.VARIANT, goal, kinds, SG.Facts(HD.VARIANT)), HD


GAME = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5", "c1g5", "h7h6", "g5h4", "d7d6", "b1c3", "c8g4"]


def mined(env_, upto: int = 6):
    e, _HD = env_
    wires = [(None, GAME[:k]) for k in range(2, len(GAME) + 1)]
    keys = SY.mine(e.model, wires, [[len(w[1])] for w in wires], e.goal_kind, 3, (1, 2), (0, 1, 3), 2)
    return keys


def test_ahead_equals_an_independent_walk_back_on_a_constructed_game(env):
    e, HD = env
    keys = [k for k in mined(env) if k["size"] <= 2][:200]
    counter = SY.Counter(e.model, [k["key"] for k in keys], e.goal_kind, 2, (0, 1, 3), 1)
    done, longer = 0, 0
    for n in range(7, len(GAME)):                                                    # the position GAME[:n] and the move GAME[n]: the mover's own earlier positions are n - 2, n - 4, n - 6
        legal = e.legal(GAME[:n])
        assert GAME[n] in legal
        table = SY.deltas_of({"moves": GAME[:n]}, legal, counter)
        taken = 0
        for j, k in enumerate(keys):
            if (j, "mover", "up") not in table[GAME[n]] or taken >= 4:                # four reasons a position: each ahead() is its own engine calls
                continue
            taken += 1
            reason = {"key": k["key"], "side": "mover", "dir": "up", "bytes": k["bytes"]}
            run = 0
            for kk in (1, 2, 3):
                at = n - 2 * kk
                if at < 0:
                    break
                lg = e.legal(GAME[:at])
                tab = SY.deltas_of({"moves": GAME[:at]}, lg, counter)
                if not any((j, "mover", "up") in tab[m] for m in lg):
                    break
                run += 1
            got = SY.ahead({"moves": GAME[:n]}, reason, 3, e)
            assert got == run, (n, j, got, run)
            longer += run > 0
            done += 1
            if done <= 6:
                assert SY.ahead({"moves": GAME[:n]}, dict(reason, dir="down"), 3, e) is None
                assert SY.ahead({"moves": GAME[:n]}, dict(reason, side="other"), 3, e) is None
    assert done >= 20 and longer >= 1                                                # the walk was not trivially zero


def test_one_short_agrees_with_the_brute_force_matcher(env):
    from catspace.incubator.analyst.discovery import graph_layer2 as L2
    from catspace.incubator.analyst.discovery import graph_layer2_match as GM
    e, HD = env
    keys = [k for k in mined(env) if k["size"] == 2 and k["level"] == 0][:40]
    assert keys
    saw_full = saw_short = 0
    for k in range(6, len(GAME) + 1, 2):
        items, units = e.items(e.text(GAME[:k]))
        for kk in keys:
            pat = L2.decode(kk["bytes"])
            c0, c1 = GM.match_counts(items, units, pat, e.kinds)
            short, full = SY.one_short(items, units, pat, e.kinds, 0)
            assert not (short & full)
            if c0 > 0:
                assert full, "a pattern with an instance has a complete key"
                saw_full += 1
            else:
                assert not full
            saw_short += bool(short)
    assert saw_full and saw_short
    one = [k for k in mined(env) if k["size"] == 1][0]
    assert SY.one_short(items, units, L2.decode(one["bytes"]), e.kinds, 0) == (set(), set())


# ------------------------------------------------------------------------------------------------------------------------------ the walls

@pytest.mark.parametrize("name", ["symbols.py", "symbols_reasons_measure.py"])
def test_the_layer_and_the_measure_name_no_chess_object(name):
    hit = re.search(CHESS_WORDS, (DISC / name).read_text(), flags=re.I)
    assert hit is None, f"{name}: {hit.group(0)!r}"


def test_no_pick_source_reads_answers_and_the_ladder_s_are_read_in_the_one_scoring_function():
    for name in ("symbols.py",):
        assert not re.search(r"answers|accepted|oracle", (DISC / name).read_text(), re.I)
    text = (DISC / "symbols_reasons_measure.py").read_text()
    assert "answers" in inspect.getsource(SR.score_ladder_reasons)
    assert "answers.jsonl" not in text.split("def score_ladder_reasons")[0]


def test_the_printing_script_is_the_only_file_that_spells_the_game():
    assert "def reasons_cases" in (DISC / "ladder_print.py").read_text()
    for name in ("symbols.py", "symbols_reasons_measure.py"):
        assert "import chess" not in (DISC / name).read_text()


def test_nothing_is_written_to_the_real_store_and_no_player_id_is_written():
    for name in ("symbols.py", "symbols_reasons_measure.py"):
        src = (DISC / name).read_text()
        assert not re.search(r"rules/learned|data/derived|snapshots", src.split('"""', 2)[2]), name
    assert SR.out_root("short").parts[-2:] == ("symbols_78_81", "short")
    assert not re.search(r"player_id|\"white_id\"|\"black_id\"", (DISC / "symbols_reasons_measure.py").read_text())
