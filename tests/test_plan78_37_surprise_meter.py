"""Plan 78.37 stage 1: the surprise meter. Pure pieces with a fake judge and hand-made readings; one small real run (skipped without the outside engine)."""
import json
import os
import shutil

import pytest

from catspace.incubator.analyst.discovery import surprise_meter as SM

ENGINE = shutil.which("stockfish") or SM.DEFAULT_ENGINE


def rd(value=None, proved=False, claim=None):
    return {"value": value, "proved": proved, "claim": claim}


def wdl(e):
    """A judge reading whose expected result is e (all mass on win and loss)."""
    return {"cp": int(round((e - 0.5) * 1000)), "wdl": (e, 0.0, 1 - e)}


class FakeReader:
    def __init__(self, table):
        self.table, self.calls = table, 0

    def read(self, variant, start, moves):
        self.calls += 1
        return self.table[tuple(moves)]


def game(ours, moves=("a", "b", "c", "d"), **kw):
    return {"variant": "v", "game": 0, "start": None, "moves": list(moves), "outcome": None, "ours": ours, **kw}


def test_jump_flips_the_reader_after_the_move():
    assert SM.jump(0.5, 0.5) == 0.0
    assert SM.jump(0.5, 0.2) == pytest.approx(0.3)       # the other side fell to 0.2: the mover rose
    assert SM.jump(0.5, 0.9) == pytest.approx(-0.4)


def test_each_surprise_kind_and_weight():
    assert SM.classify(0.0, -0.6, 0.15) == (SM.MISSED, 0.6)
    assert SM.classify(0.4, 0.05, 0.15) == (SM.FALSE, 0.4)
    assert SM.classify(0.3, -0.5, 0.15) == (SM.OPPOSITE, 0.5)
    assert SM.classify(0.3, 0.5, 0.15) is None            # both jump the same way
    assert SM.classify(0.1, 0.1, 0.15) is None
    assert SM.classify(0.0, 0.15, 0.15) == (SM.MISSED, 0.15)      # at the threshold counts
    assert SM.classify(0.0, 0.5, 0.6) is None             # the threshold is a parameter


def test_standing_gap_is_apart_from_surprise():
    assert SM.standing_gap(0.9, 0.5, 0.0, 0.0)
    assert not SM.standing_gap(0.9, 0.5, 0.3, 0.0)        # ours jumps: not a standing gap
    assert not SM.standing_gap(0.6, 0.5, 0.0, 0.0)
    assert SM.classify(0.0, 0.0) is None


def test_expected_ours_proof_claim_value():
    assert SM.expected_ours(rd(0, proved=True), 3.0) == 1.0
    assert [SM.expected_ours(rd(7, claim=c), 3.0) for c in ("win", "draw", "loss")] == [1.0, 0.5, 0.0]
    assert SM.expected_ours(rd(0), 3.0) == 0.5
    assert SM.expected_ours(rd(3), 3.0) > 0.7 > 0.3 > SM.expected_ours(rd(-3), 3.0)
    assert SM.expected_ours(rd(1 << 40), 3.0) == 1.0 and SM.expected_ours(rd(-(1 << 40)), 3.0) == 0.0
    assert not SM.uses_value(rd(1 << 40)) and not SM.uses_value(rd(2, claim="win")) and SM.uses_value(rd(2))


def test_claim_class_and_trace_row():
    f = lambda *cls: [{"relation": "root_class", "args": [str(i), c, "", "0"]} for i, c in enumerate(cls)]  # noqa: E731
    assert SM.claim_class(f("L", "W")) == "win"
    assert SM.claim_class(f("L", "D")) == "draw"
    assert SM.claim_class(f("L", "L")) == "loss"
    assert SM.claim_class(f("D", "U")) is None and SM.claim_class([]) is None
    row = {"kind": "position", "proved": False, "error": None, "facts": f("D"),
           "steps": [{"operator": "forcing", "value": None}, {"operator": "ledger", "value": -2}]}
    assert SM.ours_reading(row) == {"value": -2, "proved": False, "claim": "draw"}
    assert SM.ours_reading({"kind": "position", "proved": False, "error": None, "facts": [], "steps": [{"operator": "ledger", "value": None}]}) is None
    assert SM.ours_reading({"kind": "position", "error": "bad", "steps": []}) is None


def test_fit_scale_recovers_the_curve():
    true = 4.0
    pairs = [(v, SM._logistic(v, true)) for v in range(-12, 13)]
    assert SM.fit_scale(pairs) == pytest.approx(true, rel=0.05)
    assert SM.fit_scale([]) == 1.0


def table():
    # moves a, b, c, d: positions 0..4. The judge reads (for the side to move): 0.5, 0.5 (after a: the other side at 0.5), then b blunders: next side at 0.9
    return {(): wdl(0.5), ("a",): wdl(0.5), ("a", "b"): wdl(0.9), ("a", "b", "c"): wdl(0.9), ("a", "b", "c", "d"): wdl(0.1)}


def test_measure_counts_rates_and_weights():
    # ours: flat at 0.5 everywhere (value 0), so every judge jump is MISSED; judge: move b falls from 0.5 to 1-0.9=0.1 (jump -0.4), move d from 0.9 to 0.9 (0)
    g = game({p: rd(0) for p in range(5)})
    m = SM.measure([g], FakeReader(table()), scale=1.0, threshold=0.15)
    assert m["positions"] == 4
    assert m["counts"][SM.MISSED] == 2 and m["counts"][SM.FALSE] == 0
    assert m["by_kind"][SM.MISSED] == pytest.approx(50.0)
    # move b: judge jump (1-0.9)-0.5 = -0.4; move c: (1-0.9)-0.9 = -0.8; moves a and d: no jump
    assert sorted(round(r["weight"], 3) for r in SM.ranked(m)) == [0.4, 0.8]
    ws = sorted(r["weight"] for r in SM.ranked(m))
    assert ws == pytest.approx(sorted(ws)) and len(ws) == 2
    assert m["weighted_per_100"] == pytest.approx(100 * sum(ws) / 4)
    assert m["mean_loss_cp"] is not None and m["mean_loss_e"] > 0


def test_false_opposite_and_standing_gap_from_readings():
    t = {(): wdl(0.5), ("a",): wdl(0.5), ("a", "b"): wdl(0.5)}
    # move a: ours falls hard (value 5 -> ours says the next side is at 0 after), judge flat: FALSE
    g = {"variant": "v", "game": 0, "start": None, "moves": ["a", "b"], "ours": {0: rd(0, claim="draw"), 1: rd(0, claim="loss"), 2: rd(0, claim="draw")}}
    m = SM.measure([g], FakeReader(t), scale=1.0)
    assert [r["kind"] for r in m["records"]] == [SM.FALSE, SM.FALSE] and m["records"][0]["weight"] == pytest.approx(0.5)
    # opposite: ours says the mover gained (next side at loss), judge says the mover lost
    t2 = {(): wdl(0.5), ("a",): wdl(1.0)}
    g2 = {"variant": "v", "game": 0, "start": None, "moves": ["a"], "ours": {0: rd(0, claim="draw"), 1: rd(0, claim="loss")}}
    m2 = SM.measure([g2], FakeReader(t2), scale=1.0)
    assert m2["records"][0]["kind"] == SM.OPPOSITE and m2["records"][0]["weight"] == pytest.approx(0.5)
    # standing gap: ours says win throughout, judge says draw throughout
    t3 = {(): wdl(0.5), ("a",): wdl(0.5)}
    g3 = {"variant": "v", "game": 0, "start": None, "moves": ["a"], "ours": {0: rd(0, claim="win"), 1: rd(0, claim="loss")}}
    m3 = SM.measure([g3], FakeReader(t3), scale=1.0)
    assert m3["records"][0]["kind"] is None and m3["gaps_per_100"] == pytest.approx(100.0)


def test_ranking_order_and_write(tmp_path):
    t = {(): wdl(0.5), ("a",): wdl(0.5), ("a", "b"): wdl(0.9), ("a", "b", "c"): wdl(0.9), ("a", "b", "c", "d"): wdl(0.1)}
    g = game({p: rd(0) for p in range(5)})
    m = SM.measure([g], FakeReader(t), scale=1.0)
    rows = SM.ranked(m)
    assert [r["weight"] for r in rows] == sorted((r["weight"] for r in rows), reverse=True)
    assert set(rows[0]) >= {"start", "moves", "move", "kind", "weight", "ours", "judge"} and rows[0]["moves"] == rows[0]["moves"][:len(rows[0]["moves"])]
    files = SM.write_ranked(m, str(tmp_path / "out"))
    lines = [json.loads(x) for x in open(files["ranked"])]
    assert len(lines) == len(rows)
    assert "records" not in json.load(open(files["measure"]))
    obs = SM.format_obs(m)
    assert obs and all(x.startswith("OBS ") for x in obs)
    assert any("scale" in x for x in obs)


def test_fitted_scale_is_reported_and_given_scale_kept():
    t = {(): wdl(0.5), ("a",): wdl(0.5), ("a", "b"): wdl(0.5)}
    g = {"variant": "v", "game": 0, "start": None, "moves": ["a", "b"], "ours": {0: rd(0), 1: rd(3), 2: rd(-3)}}
    fit = SM.measure([g], FakeReader(t))
    assert fit["scale_fitted"] and fit["scale"] > 0
    fixed = SM.measure([g], FakeReader(t), scale=2.5)
    assert fixed["scale"] == 2.5 and not fixed["scale_fitted"]


def test_side_filter():
    t = {(): wdl(0.5), ("a",): wdl(0.5), ("a", "b"): wdl(0.9), ("a", "b", "c"): wdl(0.9), ("a", "b", "c", "d"): wdl(0.1)}
    g = game({p: rd(0) for p in range(5)}, ours_parity=1)
    both = SM.measure([g], FakeReader(t), scale=1.0)
    even = SM.measure([g], FakeReader(t), scale=1.0, side=0)
    odd = SM.measure([g], FakeReader(t), scale=1.0, side=1)
    ours = SM.measure([g], FakeReader(t), scale=1.0, side="ours")
    assert both["positions"] == 4 and even["positions"] == 2 and odd["positions"] == 2
    assert [r["parity"] for r in ours["records"]] == [1, 1]


def test_missing_readings_leave_moves_out():
    t = {(): wdl(0.5), ("a",): wdl(0.5), ("a", "b"): wdl(0.5)}
    g = game({0: rd(0), 2: rd(0)}, moves=("a", "b"))       # no reading at ply 1: neither move is measured
    r = FakeReader(t)
    assert SM.measure([g], r, scale=1.0)["positions"] == 0 and r.calls == 0


def test_cached_reader_repeats_and_persists(tmp_path):
    inner = FakeReader({("a",): wdl(0.7)})
    path = tmp_path / "c.json"
    c = SM.CachedReader(inner, str(path), tag="j1")
    first = c.read("v", None, ["a"])
    assert c.read("v", None, ["a"]) == first and inner.calls == 1 and c.hits == 1
    c.save()
    c2 = SM.CachedReader(FakeReader({}), str(path), tag="j1")           # a fresh object: answered from the file, the inner reader is never asked
    assert c2.read("v", None, ["a"]) == first and c2.hits == 1
    c3 = SM.CachedReader(FakeReader({("a",): wdl(0.1)}), str(path), tag="j2")      # another judge's tag is not read
    assert c3.read("v", None, ["a"]) != first


def test_read_log_labels(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [{"kind": "position", "label": "r:3:7:1", "proved": True, "error": None, "facts": [], "steps": []},
            {"kind": "game", "moves": []}, {"kind": "position", "label": "r:3:8:0", "error": "x", "steps": []}]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert SM.read_log(str(p)) == {("r", 3, 7): {"value": None, "proved": True, "claim": None}}


@pytest.mark.skipif(not os.path.exists(ENGINE), reason="the outside engine is absent")
def test_small_real_run(tmp_path):
    from catspace.incubator.analyst.discovery import game_store as GS
    from catspace.incubator.analyst.discovery import wdl_sleep as WS
    store = str(GS.path_of("chess", GS.head("chess"))) if GS.head("chess") else None
    src = SM.SelfPlaySource("chess", seconds=0.05, store=store, specs=GS.spec_texts("chess") if store else None, log=str(tmp_path / "trace.jsonl"), random_plies=2, max_plies=30, threads=2)
    games = src.games(2, 3)
    judge = WS.UciWdlJudge(ENGINE, depth=6)
    reader = SM.CachedReader(judge, str(tmp_path / "cache.json"), tag="d6")
    try:
        m = SM.measure(games, reader, clock=0.05, snapshot=src.snapshot, judge={"depth": 6})
    finally:
        reader.close()
    assert m["positions"] > 0
    obs = SM.format_obs(m)
    assert all(x.startswith("OBS ") for x in obs)
    for x in obs:
        print(x)
    files = SM.write_ranked(m, str(tmp_path / "out"))
    assert os.path.exists(files["ranked"])


def test_our_level_is_the_balance_plus_the_net_gain_read_from_there():
    from catspace.incubator.analyst.discovery import surprise_meter as SM
    assert SM.net_of(2, 1000) == 2 and SM.net_of(-1000, 1000) == -1 and SM.net_of(-1999, 1000) == -1 and SM.net_of(-3, 1) == -3
    assert SM.net_of(-2000 + 1, 1000) == -1          # one unit won, two lost
    r = {"value": 1, "proved": False, "claim": None, "balance": -3}
    assert SM.level_of(r, 1000) == -2
    assert SM.expected_ours(r, 1.0, 1000) < 0.5 < SM.expected_ours({**r, "balance": 3}, 1.0, 1000)
    games = [{"variant": "v", "start": None, "moves": ["a", "b"], "ours": {0: {"value": 0, "proved": False, "claim": None}}}]
    SM.add_balance(games, lambda variant, start, moves: 7 - len(moves))
    assert games[0]["ours"][0]["balance"] == 7


def test_the_balance_from_the_codec_counts_the_movers_units_less_the_others():
    from catspace.incubator.analyst.discovery import surprise_meter as SM
    b = SM.codec_balance("chess", {"q": 9, "p": 1})
    assert b("chess", "4k3/8/8/8/8/8/P7/3QK3 w - - 0 1", []) == 10 and b("chess", "4k3/8/8/8/8/8/P7/3QK3 b - - 0 1", []) == -10
