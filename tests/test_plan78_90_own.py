"""Plan 78.90: the mover's own loose unit, and one fit on both samples (catspace/incubator/analyst/discovery/symbols_own_measure.py, symbols.split_borne side="own", loose_units, left_loose). SYSTEM_SPEC 7, 9.

The split of the other side's bearing into cells leaves the mover's side alone and the two splits may follow each other in either order; a unit is loose when a unit of the other side bears on it and none of its own side does; a move
leaves a unit loose only when it was not so before (the moved unit followed); a candidate with no parent reason has a sign of its own in the grouping; the candidates are added for each bear-form pattern and direction no reason holds;
the union of two samples refuses a game that is in both and never lets a generated row into a fit; the other side's cell counts agree with the engine's own event on real positions; the measure names no game object."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from catspace.incubator.analyst.discovery import graph_layer2 as L2
from catspace.incubator.analyst.discovery import symbols as SY
from catspace.incubator.analyst.discovery import symbols_own_measure as O

HAVE = (O.SGM.src81() / "reasons.json").exists() and (O.SS.src88() / "fits.pkl").exists()
needs_run = pytest.mark.skipif(not HAVE, reason="needs the 78.81 and 78.88 runs' files")


def planted_pattern() -> L2.Pattern:
    return L2.Pattern(0, [("bear", 0, "u", 1, None, None)], [True, False])


def raw(p: L2.Pattern) -> str:
    return bytes([p.level, 1, 0, 0, 0, 1, 255, 255, 0, 1]).hex()


def reason(side: str, d: str, weight: float, key: int = 1) -> dict:
    return {"side": side, "dir": d, "weight": weight, "bytes": raw(planted_pattern()), "words": "w", "key": key, "event": key, "size": 1, "level": 0}


def unit(side: str, mine: bool) -> dict:
    return dict(mine=mine, side=side, goal=False, edge=False, kind="k", place="p")


def bear(u: str, t: str) -> tuple:
    return ("bear", u, ("u", t), None, None)


def test_the_own_split_takes_the_other_side_s_reasons_and_leaves_the_mover_s():
    mover, other = reason("mover", "up", 0.7), reason("other", "down", 0.4)
    own = SY.split_borne([mover, other], [], side="own")
    assert [(r["parent"], r["cell"], r["side"]) for r in own] == [(0, None, "mover"), (1, 0, "other"), (1, 1, "other"), (1, 2, "other"), (1, 3, "other")]
    assert all(r["weight"] == 0.4 and "mover's" in r["words"] for r in own[1:])
    theirs = SY.split_borne([mover, other], [])
    assert [(r["parent"], r["cell"]) for r in theirs] == [(0, 0), (0, 1), (0, 2), (0, 3), (1, None)]          # the default is as it was
    with pytest.raises(ValueError):
        SY.split_borne([mover], [], side="both")


def test_the_two_splits_in_either_order_give_the_same_reasons_and_keep_the_parent_index():
    rs = [reason("mover", "up", 0.7), reason("mover", "down", -0.2, key=2), reason("other", "down", 0.4), reason("other", "up", 0.0)]
    a = SY.split_borne(SY.split_borne(rs, [], side="own"), [], side="theirs")
    b = SY.split_borne(SY.split_borne(rs, [], side="theirs"), [], side="own")
    key = lambda r: (r["parent"], r["cell"], r["side"], r["dir"])
    assert sorted(map(key, a)) == sorted(map(key, b))
    assert len(a) == 16 and [r["parent"] for r in a if r["cell"] is None] == []          # all four are bear-form here: nothing is left unsplit
    assert sorted({r["parent"] for r in a}) == [0, 1, 2, 3]
    assert all(r["cell"] is not None for r in a)


def test_a_unit_is_loose_when_the_other_side_bears_on_it_and_its_own_side_does_not_and_a_move_leaves_it_so_only_when_it_was_not():
    units = {"u:a": unit("x", True), "u:b": unit("x", True), "u:c": unit("y", False), "u:d": unit("y", False)}
    items = [bear("u:c", "u:a"), bear("u:d", "u:b"), bear("u:a", "u:b")]
    assert SY.loose_units(items, units, 0) == {"u:a"}                      # b is borne on by d but guarded by a; a is borne on by c and guarded by none
    assert SY.loose_units(items, units, 1) == set()
    assert SY.left_loose({"u:a"}, {"u:a", "u:b"}, "zz", "yy") == {"u:b"}
    assert SY.left_loose({"u:a"}, {"u:a"}, "zz", "yy") == set()
    assert SY.left_loose({"unit:x:k:a"}, {"unit:x:k:q"}, "a", "q") == set()        # the loose unit moved from a to q and is loose there: not new
    assert SY.left_loose(set(), {"unit:x:k:q"}, "a", "q") == {"unit:x:k:q"}


def test_a_candidate_with_no_parent_has_a_sign_of_its_own_in_the_grouping():
    E = np.zeros((20, 3))
    E[:10, :] = 1
    n = len(E)
    rows = SY.Choice(sparse.csr_matrix(E), np.zeros((n, 1)), ["takes"], np.arange(n), np.ones(n, bool), np.arange(n) // 2, np.array(["find"] * n), np.array(["human"] * n))
    rs = [{"side": "other", "dir": "up", "weight": 0.3}, {"side": "other", "dir": "up", "weight": 0.0}, {"side": "other", "dir": "up", "weight": -0.3}]
    assert SY.groups_of(rs, rows) == [[0], [1], [2]]
    assert SY.groups_of([dict(r, weight=abs(r["weight"]) + 0.1) for r in rs], rows) == [[0, 1, 2]]


def test_the_candidates_are_added_for_every_pattern_and_direction_no_reason_holds():
    rs = [reason("mover", "up", 0.7, key=1), reason("other", "down", 0.4, key=2), reason("mover", "down", 0.1, key=2)]
    out = O.with_candidates(rs)
    assert len(out) == 3 + (2 * 2 - 1)                                      # two patterns, two directions each, minus the one other-side reason that exists
    fresh = out[3:]
    assert all(r["side"] == "other" and r["weight"] == 0.0 and r["event"] is None for r in fresh)
    assert {(int(r["key"]), r["dir"]) for r in fresh} == {(1, "up"), (1, "down"), (2, "up")}
    assert not any(r["event"] is None for r in out[:3])


@needs_run
def test_the_reasons_of_variant_a_replace_four_parents_and_add_the_candidates():
    R = json.loads((O.SGM.src81() / "reasons.json").read_text())
    keys, _raws = O.bear_patterns(R)
    A = O.reasons_a(R)
    parents_split = 3 + 1                                                   # the three of the mover's and the one of the other side that 78.88 / 78.90 split
    fresh = len(O.with_candidates(R)) - len(R)
    assert fresh == 2 * len(keys) - 1
    assert len(A) == len(R) - parents_split + 4 * (parents_split + fresh)
    assert sum(1 for r in A if r["cell"] is None) == len(R) - parents_split
    assert all(r["weight"] == 0.0 for r in A if r["parent"] >= len(R))              # a candidate has no parent reason and so no weight to inherit


def choice(game: np.ndarray, n_pos: int, tag: str = "find", src: str = "human") -> SY.Choice:
    pos = np.repeat(np.arange(n_pos), 2)
    n = len(pos)
    return SY.Choice(sparse.csr_matrix(np.ones((n, 2))), np.zeros((n, 1)), ["takes"], pos, np.tile([True, False], n_pos), np.repeat(game, 2), np.array([tag] * n), np.array([src] * n), np.full(n, 20))


def test_the_union_of_two_samples_numbers_the_positions_on_and_refuses_a_shared_game():
    a, b = choice(np.array([1, 2]), 2), choice(np.array([3, 4, 5]), 3)
    u = O.concat_choice(a, b)
    assert u.n_positions() == 5 and u.pos.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] and not u.fill_ok
    SY.require_find(u)
    with pytest.raises(ValueError):
        O.concat_choice(a, choice(np.array([2, 9]), 2))
    with pytest.raises(ValueError):
        SY.require_find(O.concat_choice(a, choice(np.array([7]), 1, src="generated")))


@needs_run
def test_the_other_side_s_cell_counts_agree_in_total_with_the_engine_s_event_on_real_positions():
    import cs_engine as CE
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    R = json.loads((O.SGM.src81() / "reasons.json").read_text())
    keys, raws = O.bear_patterns(R)
    sample = O.SS.load_sample(O.SS.src80())[:10]
    counter = SY.Counter(GL.model(HD.GAME, HD.VARIANT), [keys[0]], G2M.goal_kind_of(HD.VARIANT), 2, (0, 1, 2, 3), 1)
    jobs = O._own_job(([(r["moves"], r["tied"]) for r in sample], raws))
    table = [SY.deltas_of({"moves": r["moves"]}, r["tied"], counter) for r in sample]
    i, bad, seen = 0, 0, 0
    for r, t in zip(sample, table):
        for m in r["tied"]:
            cols = set(jobs[i][0])
            up = any(c // 8 == 0 and c % 2 == 0 for c in cols)               # a cell of pattern 0 rose
            down = any(c // 8 == 0 and c % 2 == 1 for c in cols)
            eng_up, eng_down = (0, "other", "up") in t[m], (0, "other", "down") in t[m]
            bad += int(eng_up and not up) + int(eng_down and not down) + int((not up and not down) and (eng_up or eng_down))
            seen += int(eng_up or eng_down)
            i += 1
    assert i > 0 and bad == 0 and seen > 0


@needs_run
def test_the_own_cells_hold_one_row_for_each_tied_move_of_a_sample_and_the_flag_is_a_bit():
    sample = O.SS.load_sample(O.SS.src80())
    R = json.loads((O.SGM.src81() / "reasons.json").read_text())
    keys, raws = O.bear_patterns(R)
    out = O._own_job(([(r["moves"], r["tied"]) for r in sample[:6]], raws))
    assert len(out) == sum(len(r["tied"]) for r in sample[:6])
    assert all(f in (0, 1) and u in (0, 1) for _c, f, u in out)
    assert all(0 <= c < 8 * len(keys) for cols, _f, _u in out for c in cols)


def test_the_measure_names_no_object_of_any_game():
    for f in ("symbols_own_measure.py", "symbols_own_play.py"):
        text = (Path(O.__file__).parent / f).read_text().lower()
        assert not re.search(r"\b[a-h][1-8]\b", text), f
        for w in ("fen", "san", "uci", "check", "mate", "promotion", "pawn", "bishop", "knight", "rook", "queen", "king", "castle"):
            assert not re.search(rf"\b{w}\b", text), (f, w)


def test_no_player_id_is_written_and_the_outputs_go_to_the_experiment_folder_only():
    text = (Path(O.__file__).parent / "symbols_own_measure.py").read_text()
    assert "game_id" not in text and 'g["id"]' not in text
    assert "paths.experiment" in text and "paths.derived" not in text and "write_text" in text
    assert O.OUT == ("symbols_78_90",)
