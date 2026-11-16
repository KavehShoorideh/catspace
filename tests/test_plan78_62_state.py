"""PLAN 78.62 A, B, C: the tactical state per ply (Rust `questions_state`), the colour tables, the sequence description and the two chance heads. Pure functions on synthetic arrays, the engine call on a short
game where the seed concepts of 78.54 are present (skipped without them)."""
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import tactical_chances as TC
from catspace.incubator.analyst.discovery import tactical_sequence as TQ
from catspace.incubator.analyst.discovery import tactical_tables as TT
from catspace.io import paths

OPENING = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5", "d7d5", "e4d5", "f6d5", "g5f7", "e8f7", "d1f3", "f7e6", "b1c3"]


def test_edge_distance_is_read_from_the_place_names():
    ed = TT.edge_distances(["a1", "d4", "h8", "a5", "e5"])
    assert list(ed) == [0, 3, 0, 0, 3]
    print(f"OBS 78.62 edge distances of five places: {list(ed)}")


def _state(rows):
    """A state dict of rows: each (game, ply, mover, units [(own, kind, class, basis, trap, fate)], board [(role, kind, place)])."""
    uoff, boff, units, board = [0], [0], [], []
    for _g, _p, _m, u, b in rows:
        units += u
        board += b
        uoff.append(len(units))
        boff.append(len(board))
    return {"game": np.array([r[0] for r in rows]), "ply": np.array([r[1] for r in rows]), "mover": np.array([r[2] for r in rows]), "uoff": np.array(uoff), "boff": np.array(boff),
            "units": np.array(units, np.uint8).reshape(-1, 6), "board": np.array(board, np.uint8).reshape(-1, 3), "meta": {"kinds": ["k0", "k1", "k2"], "roles": ["r0", "r1"], "places": ["a1", "a2", "b1", "b2"]}}


def test_colour_tables_count_a_loss_by_estimate_for_the_colour_that_owns_the_unit():
    # row 0: role 0 to move; its own unit (own=1) is safe; the other side's unit (own=0) is lost by the estimate and taken for nothing
    # row 1: role 1 to move; its own unit is lost resting on a bound; the other's unit is trapped
    st = _state([(0, 10, 0, [(1, 1, 1, 0, 0, 0), (0, 2, 0, 2, 0, 1)], [(0, 1, 0), (1, 2, 3)]),
                 (0, 11, 1, [(1, 0, 0, 3, 0, 0), (0, 1, 1, 2, 1, 0)], [(0, 1, 0), (1, 2, 3)])])
    tb = TT.colour_tables(st)
    assert tb["lost_est"][0].tolist() == [0, 1] and tb["lost_for_nothing"][0].tolist() == [0, 1] and tb["lost_est_k"][0, 1, 2] == 1
    assert tb["lost_bound"][1].tolist() == [0, 1] and tb["lost_est"][1].tolist() == [0, 0]
    assert tb["trapped"][1].tolist() == [1, 0] and tb["safe_est"][1].tolist() == [1, 0]
    assert tb["edge_k"][0, 1, 2] == 0
    print("OBS 78.62 colour tables: a loss by the estimate is counted for the owner's colour, one resting on a bound apart, trapped and safe counted")


def test_stretches_and_onsets_on_a_made_up_sequence():
    flag = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], bool)
    ended, cut = TQ.stretches(flag, np.array([0]), np.array([9]))
    assert ended == [2, 1] and cut == [3]
    n = 14
    lost = np.zeros((n, 2), np.int16)
    lost[3, 0] = 1                                              # a standing loss for colour 0 at row 3, then taken
    lost[6, 1] = 1                                              # a counter-chance for colour 1 at row 6
    tb = {"lost_est": lost, "lost_taken": np.zeros((n, 2), np.int16), "lost_for_nothing": np.zeros((n, 2), np.int16)}
    tb["lost_taken"][3, 0] = 1
    tb["lost_for_nothing"][3, 0] = 1
    st = {"mover": np.arange(n) % 2}
    on = TQ.onsets(tb, st, np.array([0]), np.array([n]))
    assert on.shape == (2, 7) or on.shape == (1, 7)
    row = on[on[:, 1] == 0][0]
    assert row[3] == 1 and row[4] == 1 and row[6] == 1 and row[5] == 0                 # it happens, for nothing, a counter-chance appears, not parried
    print("OBS 78.62 sequence: runs of a quiet flag split into ended and cut; an onset carries happened, for nothing, parried and counter")


def test_the_logistic_head_ranks_a_planted_signal_and_is_calibrated_on_it():
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(size=(n, 4))
    p = 1 / (1 + np.exp(-(1.5 * x[:, 0] - 0.5)))
    y = (rng.random(n) < p).astype(float)
    g = np.repeat(np.arange(n // 10), 10)
    m = TC.fit_logit(x[:2000], y[:2000], g[:2000])
    q = TC.predict_logit(m, x[2000:])
    from catspace.incubator.analyst.discovery.opportunity_measure import spearman
    rc = spearman(q, y[2000:])
    base = TC.row_loss(np.full(1000, y[:2000].mean()), y[2000:]).mean()
    assert rc > 0.3 and TC.row_loss(q, y[2000:]).mean() < base
    assert abs(q.mean() - y[2000:].mean()) < 0.05
    assert m["coef"][0] != 0 and abs(m["coef"][1]) < abs(m["coef"][0])
    print(f"OBS 78.62 logistic head: planted signal rank correlation {rc:+.3f}, log loss {TC.row_loss(q, y[2000:]).mean():.4f} against base {base:.4f}")


def test_the_engine_reads_the_units_of_frames_at_every_ply_of_a_short_game():
    from catspace.incubator.analyst.discovery import frames_measure as OLD
    from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    from catspace.incubator.analyst.discovery import tactical_state as TS
    cdir = Path(paths.experiment("unit_fates_78_54", "games_s5016_n500"))
    if not (cdir / "concept_keys.json").exists() or not Path(paths.experiment(*TS.HEADS_DIR, "estimator_heads_human.json")).exists():
        pytest.skip("the 78.54 concepts or the 78.59 human record are not in this checkout")
    rec = TS.human_kind_record()
    games = [{"moves": OPENING}]
    model = TS.make_model(games, rec, 0)
    keys = [k for _d, k, _x, _z in OLD.pick_concepts(cdir, TS.SEEDS)]
    part = TS.read_chunk(model, games, 4, 0, keys, G2M.goal_kind_of(HD.VARIANT), 100, 2)
    assert len(part["ply"]) == len(OPENING) + 1 - 4 and list(part["ply"]) == list(range(4, len(OPENING) + 1))
    assert part["meta"]["kinds"] and part["meta"]["roles"] == ["white", "black"] and part["uoff"][-1] == len(part["units"])
    assert int(part["frames"].sum()) > 0 and set(np.unique(part["units"][:, 2])) <= set(range(7))
    # the board of a row lists every unit: 32 at the start, one fewer after the first taking (ply 9: e4d5 is the ninth ply played)
    sizes = np.diff(part["boff"])
    assert sizes[0] == 32 and sizes[-1] < 32
    print(f"OBS 78.62 state read: {len(part['ply'])} rows, {len(part['units'])} unit records, class counts {np.bincount(part['units'][:, 2], minlength=7).tolist()}, board sizes {sizes.tolist()}")
