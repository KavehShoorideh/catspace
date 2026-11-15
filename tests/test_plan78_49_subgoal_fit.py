"""Plan 78.49 stages B, C, D: the game-free statements of a position (rust/engine/tactics/src/sheet/subgoal.rs, `SheetModel.subgoal_games` / `subgoal_children`), the sparse fit on the even games with
the penalty chosen by cross-validation grouped by game, the split by game, stage A's counts and rows_of_game, the table of quiet rows and stage D's sign convention.  Positions of two variants are made
through the engine's own entry points (the second variant reads the first's medium of movement, as its pieces move alike)."""
import random
import re
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import forcing_arrival_measure as FA
from catspace.incubator.analyst.discovery import question_sheet as QS
from catspace.incubator.analyst.discovery import sheet_rust as SR
from catspace.incubator.analyst.discovery import subgoal_fit_measure as SG

ROOT = Path(__file__).resolve().parents[1]
WORTH = {"k": 0, "q": 14, "r": 8, "b": 5, "n": 4, "p": 2}
LOOSE = "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"                 # a unit of ours bears on a loose unit of theirs; theirs bears on ours, which our other unit guards
STRUCK = "4k3/8/8/4q3/8/8/3R4/4K3 w - - 0 1"                # their unit bears on our goal unit's place
RACE_VARIANT = "racingkings"
RACE_CAPTURE = "8/8/3q4/8/8/3R4/k7/7K w - - 0 1"
SCAN_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"


def sheet_of(variant: str):
    """The sheet model of a variant through the engine's entry point; a variant with no description of its own reads the first game's medium of movement."""
    env = QS.Env("chess")
    if variant == "chess":
        return SR.model(env, WORTH)
    import cs_engine as CE
    med = env.medium()
    hops = [(k, s, f, t, bool(take)) for (k, s, f), row in med.hop.items() for t, take in row.items()]
    overs = [(k, s, f, t, o) for (k, s, f, t), places in med.over_of.items() for o in places]
    wf = env.worth_fn(WORTH)
    return CE.SheetModel(variant, hops, overs, sorted(med.takes), {k: wf(k) for k in env.kind_keys_all()}, "king", len(env.relations["can_go"]), {k: int(v) for k, v in WORTH.items()}, dict(env.kind_keys_all()))


@pytest.fixture(scope="module")
def models():
    return {"chess": sheet_of("chess"), RACE_VARIANT: sheet_of(RACE_VARIANT)}


def named(model, fen: str) -> "tuple[dict, dict]":
    """({statement: number} of the side to move, the same of the other side) at a position."""
    names = [n for n, _t in model.subgoal_names()]
    row = model.subgoal_games([(fen, [])], 1)[0][0]
    n = len(names)
    return dict(zip(names, row[:n])), dict(zip(names, row[n:]))


# ------------------------------------------------------------------------------------------------------------------------------ B: the statements

def test_each_statement_has_a_name_and_a_line_of_text(models):
    names = models["chess"].subgoal_names()
    assert len(names) >= 15 and len({n for n, _t in names}) == len(names) and all(t.strip() and not t.endswith(".") for _n, t in names)


def test_statements_on_a_position_where_one_side_bears_on_a_loose_unit(models):
    us, them = named(models["chess"], LOOSE)
    assert (us["units"], them["units"], us["material"], them["material"]) == (2, 2, 8, 14)
    assert (us["attackers"], us["attacks"], us["attack_worth"], us["cheap_attacks"]) == (1, 1, 14, 1)           # our cheaper unit bears on their dearer one
    assert (us["loose_targets"], us["loose_worth"], us["overloaded"], us["favourable_targets"]) == (1, 14, 1, 1)
    assert (them["attackers"], them["attacks"], them["attack_worth"], them["cheap_attacks"]) == (1, 1, 8, 0)
    assert (them["loose_targets"], them["overloaded"], them["favourable_targets"]) == (0, 0, 0)               # the unit theirs bears on is guarded
    assert (us["guarded"], us["unguarded"], them["guarded"], them["unguarded"]) == (1, 1, 0, 2)
    assert us["mutual_pairs"] == them["mutual_pairs"] == 0
    assert (us["goal_struck"], them["goal_struck"], us["ring_struck"]) == (0, 0, 0)
    assert (us["cheapest_units"], them["cheapest_units"]) == (1, 0)                                              # the least worth of any unit that is not a goal unit
    assert (us["goal_steps"], us["goal_free_steps"], them["goal_steps"], them["goal_free_steps"]) == (4, 4, 5, 5)


def test_statements_on_a_position_where_a_goal_unit_is_borne_on(models):
    us, them = named(models["chess"], STRUCK)
    assert (them["goal_struck"], us["goal_struck"]) == (1, 0)
    assert (us["goal_steps"], us["goal_free_steps"]) == (4, 3)                                                   # one of its steps is borne on by the other side
    assert (us["ring_struck"], them["ring_struck"]) == (2, 1)                                                    # the places beside a goal unit that the other side bears on
    assert (them["loose_targets"], them["overloaded"]) == (1, 1)                                                 # theirs bears on our goal unit, which nothing of theirs guards... and of ours nothing else bears on it


@pytest.mark.parametrize("variant,fen", [("chess", LOOSE), ("chess", "4k3/p7/8/3q4/8/8/P2R4/4K3 w - - 0 1"), (RACE_VARIANT, RACE_CAPTURE), ("chess", "r1bq1rk1/ppp2ppp/2n2n2/3pp3/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 7")])
def test_the_other_side_to_move_swaps_the_two_lists(models, variant, fen):
    """The statements read the board and the two sides alike: the same board with the other side to move gives the lists swapped."""
    flipped = fen.replace(" w ", " b ")
    a, b = named(models[variant], fen), named(models[variant], flipped)
    assert a[0] == b[1] and a[1] == b[0]


@pytest.mark.parametrize("variant", ["chess", RACE_VARIANT])
def test_invariants_over_the_positions_of_a_random_game_and_the_batch_equals_one_by_one(models, variant):
    import cs_engine as CE
    rng = random.Random(49)
    moves = []
    for _ in range(40):
        state = CE.state(variant, None, moves)
        if state[3] is not None or not state[2]:
            break
        moves.append(rng.choice(sorted(state[2])))
    model = models[variant]
    n = len(model.subgoal_names())
    plies = model.subgoal_games([(None, moves)], 2)[0]
    assert len(plies) == len(moves) + 1
    for t, row in enumerate(plies):
        if row is None:
            continue
        for side in (row[:n], row[n:]):
            s = dict(zip([x for x, _t in model.subgoal_names()], side))
            assert all(v >= 0 for v in side)
            assert s["controlled_empty"] <= s["controlled"] <= s["bearing_total"] and s["only_controlled"] <= s["controlled"] and s["widest_bearing"] <= s["bearing_total"]
            assert s["attackers"] <= s["attacks"] and s["loose_targets"] <= s["units"] + 1 and s["movers"] <= s["units"] and s["goal_free_steps"] <= s["goal_steps"]
            assert s["guarded"] + s["unguarded"] == s["units"]
        one = model.subgoal_games([(CE.state(variant, None, moves[:t])[0], [])], 1)[0][0]
        assert one == row


def test_a_finished_position_has_no_statements(models):
    last = models["chess"].subgoal_games([(None, ["f2f3", "e7e5", "g2g4", "d8h4"])], 1)[0]
    assert len(last) == 5 and last[-1] is None and all(x is not None for x in last[:-1])


def test_a_game_with_an_illegal_move_is_refused(models):
    with pytest.raises(ValueError):
        models["chess"].subgoal_games([(None, ["e2e5"])], 1)


# ------------------------------------------------------------------------------------------------------------------------------ D: the number after a move, and its sign

@pytest.mark.parametrize("variant,fen,capture", [("chess", LOOSE, "d2d5"), (RACE_VARIANT, RACE_CAPTURE, "d3d6")])
def test_the_number_after_a_move_is_read_from_the_side_that_moved(models, variant, fen, capture):
    """With the weight on the material statement alone, the number after a move is the material of the side that moved less the other's on the position after it: taking the unit raises it."""
    model = models[variant]
    names = [n for n, _t in model.subgoal_names()]
    w = [1.0 if n == "material" else 0.0 for n in names]
    (res,) = model.subgoal_children([(fen, [capture])], 1, [[0]], w)
    (ply, played, vals) = res[0]
    assert ply == 0 and played is not None
    by_move = dict(vals)
    assert by_move[played] == max(by_move.values()) and by_move[played] > 0                                      # 8 held, nothing left of theirs worth anything
    others = [v for ix, v in vals if ix != played]
    assert others and max(others) < by_move[played]
    us, them = named(model, fen)
    assert by_move[played] == pytest.approx(us["material"])                                                     # their dearer unit is gone: the balance is our material less nothing
    assert max(others) == pytest.approx(us["material"] - them["material"])                                       # a move that takes nothing leaves the balance as it was


def test_children_skip_positions_not_asked_and_check_the_shape(models):
    model = models["chess"]
    n = len(model.subgoal_names())
    assert model.subgoal_children([(None, ["e2e4", "e7e5"])], 1, [[]], [0.0] * n) == [[]]
    with pytest.raises(ValueError):
        model.subgoal_children([(None, ["e2e4"])], 1, [[0]], [0.0] * (n - 1))
    ((ply, played, vals),) = model.subgoal_children([(None, ["e2e4", "e7e5"])], 1, [[1]], [0.0] * n)[0]
    assert ply == 1 and played is not None and len(vals) == 20 and all(v == 0.0 for _i, v in vals)


# ------------------------------------------------------------------------------------------------------------------------------ stage A: counts and rows_of_game

def threat(verdict):
    return (verdict, 1.0, 0, [("takes", 1.0)], 0.0, 2)


def read(*verdicts):
    return {"threats": [threat(v) for v in verdicts], "goal_stake": 20.0, "cut": False, "seconds": 0.0, "nodes": 1}


def ply(ours=(), theirs=None):
    return {"mover": "a", "ours": read(*ours), "theirs": None if theirs is None else read(*theirs)}


def test_counts_of_a_read():
    assert FA.counts(None) == (0, 0, 0)
    assert FA.counts(read("CONFIRMED", "UNREAD", "NOT SETTLED", "NO CLAIM")) == (4, 2, 1)
    assert FA.counts(read()) == (0, 0, 0)


def test_quiet_positions_and_the_arrival_target_of_a_game():
    plies = [ply(["NO CLAIM"], ["NO CLAIM"]), ply([], []), ply(["NOT SETTLED"], []), ply(["CONFIRMED"], []), ply([], ["NO CLAIM"]), ply([], []), None]
    own, oth = FA.counts_of_game(plies)
    assert FA.quiet_indices(plies, own, oth) == [0, 1, 2, 4, 5]                                                   # 3 holds a CONFIRMED threat move; 6 is over
    assert FA.arrival_targets(own, len(plies), 1) == [(1, 0), None, None]                                         # the mover's own turn 3 holds one; h = 8 runs past the game
    assert FA.arrival_targets(own, len(plies), 0) == [(0, 1), None, None]                                         # at 0: the other side's turn 3 holds one, the mover's turns 2 and 4 none
    rows = FA.rows_of_game(plies, random.Random(0))
    assert [r[1][0] for r in rows] == [FA.arrival_targets(own, len(plies), t)[0] for t in (0, 1, 2, 4, 5)]
    assert [r[0][2] for r in rows] == [0, 0, 1, -1, 0]                                                            # the count difference at each quiet position


# ------------------------------------------------------------------------------------------------------------------------------ C: the table, the split and the fit

def synthetic_games(n_games=12, plies_each=9, seed=3):
    rng = random.Random(seed)
    records = [{"game": 100 + g, "moves": []} for g in range(n_games)]
    reads, stats = {}, []
    for r in records:
        ps = []
        for t in range(plies_each):
            confirmed = rng.random() < 0.2
            ps.append(ply(["CONFIRMED"] if confirmed else rng.choice([[], ["NO CLAIM"], ["NOT SETTLED", "NO CLAIM"]]), rng.choice([[], ["NO CLAIM"]])))
        ps.append(None)
        reads[r["game"]] = ps
        stats.append(np.array([[rng.random() for _ in range(6)] if p else [np.nan] * 6 for p in ps]))
    a_opp = {"game": np.array([g for g in range(n_games) for t in range(plies_each)], float), "ply": np.array([t for g in range(n_games) for t in range(plies_each)], float)}
    for h in SG.HORIZONS:
        for k in ("out", "won", "lost"):
            a_opp[f"{k}{h}"] = np.array([rng.random() for _ in range(n_games * plies_each)])
    return records, reads, stats, a_opp


def test_the_table_has_a_row_per_quiet_position_with_stage_a_targets_and_the_split_is_by_game_parity():
    records, reads, stats, a_opp = synthetic_games()
    t = SG.table_of(records, reads, stats, a_opp)
    for g, r in enumerate(records):
        own, oth = FA.counts_of_game(reads[r["game"]])
        quiet = FA.quiet_indices(reads[r["game"]], own, oth)
        rows = np.flatnonzero(t["game"] == g)
        assert t["ply"][rows].tolist() == quiet
        for h_ix, h in enumerate(SG.HORIZONS):
            want = [np.nan if x[h_ix] is None else x[h_ix][0] - x[h_ix][1] for x in (FA.arrival_targets(own, 10, q) for q in quiet)]
            assert np.allclose(t[f"arr{h}"][rows], want, equal_nan=True)
    fit, test = SG.split_rows(t)
    assert set(t["id"][fit] % 2) == {0} and set(t["id"][test] % 2) == {1} and len(fit) + len(test) == len(t["ply"])
    assert set(t["game"][fit]).isdisjoint(set(t["game"][test]))
    assert t["P"].shape == (len(t["ply"]), 3) and t["F"].shape == (len(t["ply"]), 3)
    assert np.allclose(t["P"][0], SG.placed_diff(stats[int(t["game"][0])])[int(t["ply"][0])])                                     # the statement differences are those of the rows' positions


def test_a_difference_of_the_two_sides_statements():
    assert SG.placed_diff(np.array([[3.0, 1.0, 5.0, 2.0]])).tolist() == [[-2.0, -1.0]]


def test_standardise_leaves_a_column_that_does_not_vary_at_zero():
    x = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    mean, scale = SG.standardise(x)
    assert scale[1] == 1.0 and np.allclose(((x - mean) / scale)[:, 1], 0) and np.allclose(((x - mean) / scale)[:, 0].std(), 1)


def planted(n_games=40, per=30, seed=1):
    """y depends on columns 0 and 3 of six; the rest is noise."""
    rng = np.random.default_rng(seed)
    games = np.repeat(np.arange(n_games), per).astype(float)
    x = rng.normal(size=(n_games * per, 6))
    return x, 2.0 * x[:, 0] - 1.0 * x[:, 3] + 0.3 * rng.normal(size=len(x)), games


def test_the_sparse_fit_keeps_the_planted_columns_and_drops_the_noise():
    x, y, games = planted()
    m = SG.fit_sparse(x, y, games, jobs=1)
    kept = set(np.flatnonzero(m["coef"] != 0).tolist())
    assert {0, 3} <= kept and len(kept) <= 4 and m["alpha"] > 0
    assert np.sign(m["coef"][0]) == 1 and np.sign(m["coef"][3]) == -1
    assert np.corrcoef(SG.predict(m, x), y)[0, 1] > 0.95
    assert np.allclose(SG.raw_weights(m) * m["scale"], m["coef"])


def test_the_penalty_is_chosen_by_folds_that_keep_a_game_whole():
    from sklearn.model_selection import GroupKFold
    _x, _y, games = planted()
    for tr, te in GroupKFold(n_splits=SG.FOLDS).split(np.zeros(len(games)), groups=games):
        assert set(games[tr]).isdisjoint(set(games[te]))


def test_the_fit_reads_the_fit_half_only():
    records, reads, stats, a_opp = synthetic_games(n_games=40, plies_each=14, seed=5)
    t = SG.table_of(records, reads, stats, a_opp)
    fit, test = SG.split_rows(t)
    m1 = SG.fit_all(t, fit, "P", "arr", 4, jobs=1)
    t2 = {k: v.copy() for k, v in t.items()}
    t2["arr4"][test] = 99.0                                                                                       # the test half's target is not read by the fit
    t2["P"][test] = 99.0
    m2 = SG.fit_all(t2, fit, "P", "arr", 4, jobs=1)
    assert m1 is not None and np.allclose(m1["coef"], m2["coef"]) and m1["alpha"] == m2["alpha"]


def test_a_shuffle_within_a_game_keeps_each_games_values():
    v = np.arange(12, dtype=float)
    g = np.repeat([0.0, 1.0, 2.0], 4)
    s = SG.shuffled_within_game(v, g, 1)
    for k in (0.0, 1.0, 2.0):
        assert sorted(s[g == k].tolist()) == sorted(v[g == k].tolist())
    assert not np.array_equal(s, v)


def test_the_bands_of_a_reading_and_the_gate_of_stage_d():
    x, y, games = planted(n_games=20, per=20)
    a = {"game": games, "pred_P_arr4": x[:, 0], "arr4": y, "count": np.random.default_rng(0).normal(size=len(y))}
    lines, vals = SG.band_lines("t", a, 20, [("fit", "pred_P_arr4", "arr4"), ("count", "count", "arr4")])
    lo, hi = vals[("pred_P_arr4", "arr4")][1:]
    assert vals[("pred_P_arr4", "arr4")][0] > 0.5 and lo <= vals[("pred_P_arr4", "arr4")][0] <= hi and abs(vals[("count", "arr4")][0]) < 0.2
    assert len(lines) == 2 and all(l.startswith("OBS subgoal t:") for l in lines)
    assert SG.clear_of_count(vals, "P", 4) is True
    vals[("count", "arr4")] = (0.9, 0.8, 0.95)
    assert SG.clear_of_count(vals, "P", 4) is False


def test_the_control_of_stage_d_counts_the_played_move_against_the_mean():
    st = SG.control_stats([(2.0, [0.0, 1.0, 2.0, 3.0]), (0.0, [0.0, 0.0]), (None, [1.0, 2.0])], 1)
    assert st["n"] == 2 and st["mean_diff"] == pytest.approx(0.25) and st["above"] == 1 and st["below"] == 0
    assert st["pct"] == pytest.approx((0.625 + 0.5) / 2)                                                          # beats two of four and ties one (half); ties both of two


def test_the_run_end_to_end_on_synthetic_games():
    records, reads, stats, a_opp = synthetic_games(n_games=30, plies_each=14, seed=9)
    names = [(f"s{i}", f"statement {i}") for i in range(3)]
    stats3 = [np.hstack([s[:, :3], s[:, 3:]]) for s in stats]
    lines, data = SG.run("chess", records, reads, a_opp, stats3, names, 1, 10, "t", False, jobs=1)
    text = "\n".join(lines)
    assert all(l.startswith("OBS subgoal t:") for l in lines)
    assert "fit P arrival h=4" in text and "count difference of stage A" in text and "shuffled within the game" in text and "stage D" not in text


# ------------------------------------------------------------------------------------------------------------------------------ the layer stays game-free

def test_the_statements_and_the_measure_name_no_game_object():
    for p in (ROOT / "rust" / "engine" / "tactics" / "src" / "sheet" / "subgoal.rs", ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "subgoal_fit_measure.py", Path(__file__)):
        text = re.sub(r'"(?:\\.|[^"\\])*"', '""', p.read_text())
        text = re.sub(r"'(?:\\.|[^'\\])*'", "''", text) if p.suffix == ".py" else text
        hit = re.search(SCAN_WORDS, text, flags=re.I)
        assert hit is None, (p.name, hit.group(0) if hit else None)
