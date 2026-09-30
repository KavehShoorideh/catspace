"""The depth decider (catspace/incubator/analyst/depth_decider.py): the zone search's learned stop-or-continue call."""
import json

import chess

from catspace.incubator.analyst import depth_decider as DD
from catspace.incubator.analyst.fates import clear_caches


class _Const:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        import numpy as np
        return np.array([[1 - self.p, self.p]] * len(X))


def _deepest(p, threshold=0.5, price=None):
    from catspace.incubator.analyst.zone_search import ZoneSearch, TacticalSpec
    clear_caches()
    DD._MODEL["m"] = _Const(p); DD._MODEL["c"] = _Const(p); DD._MODEL["g"] = {4: 5.0}
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p1B1/4P3/2N5/PPPP1PPP/R2QKBNR b KQkq - 0 4")
    ms = [b.parse_san(s) for s in ("Be7", "h6", "d6")]
    rungs = (4, 6, 8)
    zs = ZoneSearch(spec=TacticalSpec(depths=rungs, budget=3000))
    seen = []
    zs.rank(b, ms, stop=DD.stopper(b, rungs, threshold, price, log=seen))
    DD._MODEL.clear()
    return max(d for h in zs.history.values() for d, _, _ in h), seen


def test_the_decider_stops_or_goes_on():
    d, seen = _deepest(0.0)
    assert d == 4 and seen == [(4, 0.0)]                              # no chance of a change: stop after the first rung
    d, seen = _deepest(1.0)
    assert d == 8 and [x for x, _ in seen] == [4, 6]                  # a sure change: every rung (the deepest needs no call)


def test_features_read_ties_changes_and_mates():
    b = chess.Board()
    h = {"e2e4": [(4, 1.0, 10), (6, 0.0, 40)], "d2d4": [(4, 1.0, 10), (6, 2.0, 50)], "g1f3": [(4, -9.0, 5), (6, -9.0, 5)]}
    x = dict(zip(DD.FEATURES, DD.features(b, 6, h, (4, 6, 8))))
    assert x["top_size"] == 1 and x["changed_last"] is False          # {e4, d4} -> {d4}: they share d4, no change
    assert x["gap"] == 2.0 and x["swing"] == 1.0
    h["c2c4"] = []                                                    # a move that mates at once has no history
    x = dict(zip(DD.FEATURES, DD.features(b, 6, h, (4, 6, 8))))
    assert x["mate"] and x["top_size"] == 1 and x["gap"] == DD.GAP_CAP


def test_ladder_rows_label_a_changed_decision(tmp_path):
    rec = dict(pid="x", step=0, fen=chess.STARTING_FEN, moves=["e2e4", "d2d4"], sol="d2d4", nodes=0, secs=0,
               history={"e2e4": [[4, 1.0, 1], [6, 1.0, 1], [8, 0.0, 1]], "d2d4": [[4, 0.0, 1], [6, 0.0, 1], [8, 2.0, 1]]})
    p = tmp_path / "ladder.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    rs = DD.rows(str(p))
    assert [(r["d"], r["y"]) for r in rs] == [(4, True), (6, True)]   # e4 on top at 4 and 6, d4 at the deepest


def test_a_price_on_search_stops_where_the_next_rung_is_dear():
    d, _ = _deepest(0.5, price=1.0)                                   # a chance of 0.5 is worth less than any rung's nodes
    assert d == 4
    d, _ = _deepest(0.5, price=0.0)                                   # search is free: go on to the deepest rung
    assert d == 8
    assert DD.next_cost([100], 5.0) == 500 and DD.next_cost([100, 400], 5.0) == 1600
