"""Plan 78.27: the sleep cycle of the win / draw / loss estimate (wdl_sleep.py, the table sources of wdl_table.py, the cycle in initialize.py). Fast: tiny clocks, tmp_path, a fake judge.
The UCI judge is the one test that needs an outside engine and skips without it."""
import ast
import json
import os
import shutil
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import initialize as IN
from catspace.incubator.analyst.discovery import wake as WK
from catspace.incubator.analyst.discovery import wdl_sleep as WS
from catspace.incubator.analyst.discovery import wdl_table as WT

ROOT = Path(__file__).resolve().parents[1]
STOCKFISH = "/opt/homebrew/bin/stockfish"
FAST = WS.ReadSetting("fast", 0.03, 8)


class FakeJudge(WS.WdlJudge):
    """A judge with no engine: a fixed reading; records what it was asked."""

    def __init__(self, reading=(0.5, 0.5, 0.0)):
        self.reading, self.asked, self.closed = reading, [], False

    def wdl(self, variant, moves, start=None):
        self.asked.append((variant, tuple(moves), start))
        return self.reading

    def close(self):
        self.closed = True


def wake_log(tmp_path, name="w.jsonl", *, games=2, max_plies=10, table=None, seconds=0.1, run="t"):
    log = str(tmp_path / name)
    out = WK.play_self("chess", games, seconds=seconds, random_plies=2, max_plies=max_plies, seed=1, wdl=True, wdl_table=table, log=log, run=run)
    return log, out


def prow(label, *, wdl=(0.2, 0.7, 0.1, 100.0), word="quiet", hit=None, prefer=None, fired=False):
    """A synthetic wake `position` row with the structured fields the gaps read."""
    facts = [{"relation": "wdl", "args": [str(wdl[0]), str(wdl[1]), str(wdl[2]), str(wdl[3]), "playouts"]}, {"relation": "sharpness", "args": [str(wdl[0] + wdl[2]), word]}]
    if hit:
        facts.append({"relation": "table_hit", "args": [hit[0], *map(str, hit[1])]})
    steps = [{"operator": "wdl", "goal": "estimate(0)", "status": "Read"}]
    if prefer:
        steps.append({"operator": "wdl", "goal": "prefer(0,2 moves,0 defects)", "status": "Read"})
        facts += [{"relation": "wdl", "args": ["0.1", "0.8", "0.1", "9", "playouts"]}, {"relation": "sharpness", "args": ["0.2", "unknown"]}]
        facts += [{"relation": "move_wdl", "args": [m, str(w), str(d), str(l), "5"]} for m, (w, d, l) in prefer.items()]
    if fired:
        facts.append({"relation": "fired", "args": ["lem", "1", "1"]})
    return {"kind": "position", "label": label, "move": "e2e4", "error": None, "steps": steps, "facts": facts, "trace": []}


def synthetic_log(tmp_path, rows, game_outcome=(100, 0), unfinished=False):
    p = tmp_path / "syn.jsonl"
    g = {"kind": "game", "run": "s", "game": 0, "variant": "chess", "start": None, "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"], "outcome": None if unfinished else list(game_outcome),
         "unfinished": unfinished, "snapshot": ""}
    p.write_text("\n".join(json.dumps(r) for r in [*rows, g]) + "\n")
    return str(p)


# ------------------------------------------------------------------------------------------------------------------------------------ gaps

def test_gap_kinds_from_structured_rows(tmp_path):
    exact = "exact:" + "0" * 8
    rows = [prow("s:0:2:0", word="quiet"),                                                           # no table, no lemma
            prow("s:0:3:1", word="quiet", hit=(exact, (30, 0, 0)), fired=True),                      # exact hit, lemma: no gap but the surprise below
            prow("s:0:4:0", word="unknown", hit=("units:p:white", (30, 0, 0)), fired=True),          # coarse + unknown
            prow("s:0:5:1", word="quiet", hit=(exact, (30, 0, 0)), fired=True, prefer={"e2e4": (0.3, 0.6, 0.1), "d2d4": (0.3, 0.59, 0.11)})]   # thin pick (0.6 vs 0.6)
    g = WS.gaps(synthetic_log(tmp_path, rows, game_outcome=(100, 0)))
    c = g["counts"]
    assert g["decisions"] == 4 and g["table"] == {"none": 1, "exact": 2, "coarse": 1}
    assert c["no_table"] == 1 and c["coarse_table"] == 1 and c["unknown"] == 1 and c["no_lemma"] == 1 and c["thin_pick"] == 1
    # surprise: score 0.55 for white (role 0) vs result 1.0 -> 0.45 under 0.5; for black (role 1) 0.55 vs 0.0 -> 0.55
    assert c["outcome_surprise"] >= 1
    by = {(e["ply"]): e for e in g["worklist"]}
    assert by[2]["kinds"][0] == "no_table" and by[2]["moves"] == ["e2e4", "e7e5"] and by[2]["start"] is None and by[2]["run"] == "s" and by[2]["game"] == 0
    assert "thin_pick" in by[5]["kinds"] and "coarse_table" in by[4]["kinds"] and "unknown" in by[4]["kinds"]
    assert g["picked"] == 1 and g["with_lemma"] == 3


def test_unfinished_games_have_no_outcome_surprise(tmp_path):
    g = WS.gaps(synthetic_log(tmp_path, [prow("s:0:2:0")], unfinished=True))
    assert g["counts"]["outcome_surprise"] == 0 and g["counts"]["no_table"] == 1


def test_gaps_of_a_real_wake_log(tmp_path):
    log, games = wake_log(tmp_path)
    g = WS.gaps(log)
    assert g["decisions"] > 0 and g["counts"]["no_table"] == g["decisions"] and g["table"]["none"] == g["decisions"]
    assert set(g["counts"]) == set(WS.GAP_KINDS) and all(set(e["kinds"]) <= set(WS.GAP_KINDS) for e in g["worklist"])
    for e in g["worklist"][:3]:
        assert len(e["moves"]) == e["ply"] and CE.state("chess", e["start"], e["moves"])[1] is not None


# ------------------------------------------------------------------------------------------------------------------------------------ the judge

def test_judge_interface_is_training_only():
    assert WS.WdlJudge.training_only is True and WS.UciWdlJudge.training_only is True
    with pytest.raises(NotImplementedError):
        WS.WdlJudge().wdl("chess", [])


@pytest.mark.skipif(not os.path.exists(STOCKFISH), reason="no stockfish binary")
def test_uci_judge_reads_wdl_caches_and_closes(tmp_path):
    cache = tmp_path / "c.json"
    j = WS.UciWdlJudge(STOCKFISH, depth=4, cache=str(cache))
    try:
        w, d, l = j.wdl("chess", ["e2e4"])
        assert abs(w + d + l - 1.0) < 1e-6 and j.asked == 1 and j.hits == 0
        assert j.wdl("chess", ["e2e4"]) == (w, d, l) and j.hits == 1
    finally:
        j.close()
    assert j._engine is None and cache.exists()
    j2 = WS.UciWdlJudge(STOCKFISH, depth=4, cache=str(cache))
    try:
        assert j2.wdl("chess", ["e2e4"]) == (w, d, l) and j2.hits == 1 and j2._engine is None      # answered from the file: no engine was started
    finally:
        j2.close()


# ------------------------------------------------------------------------------------------------------------------------------------ reading settings

def test_default_menu():
    m = WS.default_menu(2.0)
    assert [(s.name, s.seconds, s.max_playout) for s in m] == [("wake", 0.03, 8), ("long", 2.0, 8), ("far", 2.0, 40), ("end", 2.0, 400)]


def test_choose_reading_takes_the_smallest_error_among_the_deep_entries(monkeypatch):
    scripted = {"wake": 0.75, "long": 0.6, "far": 0.7, "end": 0.4}                  # the judge's score is 0.75: the wake is exact but excluded

    def fake_read(setting, variant, start, moves, **kw):
        p = scripted[setting.name]
        return {"win": p, "draw": 0.0, "loss": 1 - p, "seconds": 0.0}
    monkeypatch.setattr(WS, "read", fake_read)
    judge = FakeJudge((0.5, 0.5, 0.0))
    out = WS.choose_reading([(None, ["e2e4"]), (None, ["d2d4"]), (None, [])], judge, WS.default_menu(), variant="chess", sample=2, seed=3)
    assert out["setting"].name == "far" and out["sample"] == 2 and len(judge.asked) == 2
    assert set(out["scores"]) == {"wake", "long", "far", "end"} and out["scores"]["far"]["mae"] == pytest.approx(0.05) and "far" in out["reason"]
    assert set(out) == {"setting", "reason", "scores", "sample"}


def test_choose_reading_without_a_judge_takes_long():
    out = WS.choose_reading([(None, [])], None, WS.default_menu(), variant="chess")
    assert out["setting"].name == "long" and out["reason"] == "no judge of the readings was given"


def test_choose_reading_with_the_real_engine_runs():
    menu = [WS.ReadSetting("wake", 0.02), WS.ReadSetting("long", 0.04), WS.ReadSetting("far", 0.04, 40)]
    out = WS.choose_reading([(None, ["e2e4"]), (None, ["d2d4", "d7d5"])], FakeJudge(), menu, variant="chess", sample=5)
    assert out["setting"].name in ("long", "far") and all(s["n"] == 2 for s in out["scores"].values())


# ------------------------------------------------------------------------------------------------------------------------------------ settle, deepen

def test_settle_reads_the_last_position_and_marks_it():
    games = [{"start": None, "moves": ["e2e4", "e7e5"]}, {"start": None, "moves": ["d2d4"]}]
    out = WS.settle(games, FAST, variant="chess", say=lambda *_: None)
    assert len(out) == 2 and all(g["settled"] for g in out)
    assert out[0]["mover"] == "white" and out[1]["mover"] == "black"
    assert all(abs(sum(g["wdl"]) - 1.0) < 1e-2 for g in out)


def test_deepen_files_exact_at_full_mass_and_coarse_as_a_vote_spread_over_games():
    work = [{"log": 0, "run": "r", "game": g, "ply": p, "start": None, "moves": ["e2e4", "e7e5", "g1f3", "b8c6"][:p], "kinds": ["no_table"]} for g in range(3) for p in (1, 2)]
    # (the same prefixes in every game are one exact position each: deduped)
    work = [dict(w, moves=(["d2d4", "g8f6", "c2c4", "e7e6"] if w["game"] == 1 else ["e2e4", "c7c5", "g1f3", "d7d6"])[:w["ply"]]) for w in work]
    d = WS.deepen(work, FAST, variant="chess", max_positions=3, say=lambda *_: None)
    assert d["read"] == 3 and len({(e["game"]) for e in d["touched"]}) >= 2
    exact = {k: v for k, v in d["rows"].items() if k.startswith("exact:")}
    assert len(exact) == 3 and all(sum(v) == pytest.approx(WS.DEEP_EXACT_MASS, rel=1e-3) for v in exact.values())
    units = {k: v for k, v in d["rows"].items() if k.startswith("units:")}
    assert units and sum(sum(v) for v in units.values()) == pytest.approx(3 * WS.COARSE_VOTE, rel=1e-3)
    assert WS.DEEP_EXACT_MASS == 200.0 and WS.COARSE_VOTE == 1.0


# ------------------------------------------------------------------------------------------------------------------------------------ the table sources

def test_deep_exact_rows_survive_min_exact_and_are_counted_apart():
    key = CE.wdl_keys("chess", None, ["e2e4"])
    t = WT.build([(None, ["e2e4", "e7e5"], [100, 0])], "chess", min_exact=2, deep_rows={key[0]: [150.0, 50.0, 0.0]})
    assert t["rows"][key[0]] == [150.0, 50.0, 0.0]                                    # (the game's own exact rows are dropped; the deep one is not)
    st = t["stats"]
    assert st["dropped_exact"] == 2 and st["deep"] == {"rows": 1, "mass": 200.0} and st["settled"] == {"games": 0, "positions": 0} and st["carried_rows"] == 0
    assert t["provenance"] == "own games + own deep playouts"
    assert WT.build([], "chess")["provenance"] == "own games"


def test_settled_games_count_softly_from_each_movers_view():
    g = {"start": None, "moves": ["e2e4", "e7e5"], "mover": "white", "wdl": [0.6, 0.3, 0.1], "settled": True}
    t = WT.build([], "chess", settled=[g])
    assert t["stats"]["settled"] == {"games": 1, "positions": 3} and t["stats"]["games"] == 0
    units_white = CE.wdl_keys("chess", None, [])[1]
    units_black = CE.wdl_keys("chess", None, ["e2e4"])[1]
    assert t["rows"][units_white][0] > 0
    assert t["rows"][units_black][0] == pytest.approx(0.1) and t["rows"][units_black][2] == pytest.approx(0.6)
    assert "own settled games" in t["provenance"]


def test_a_prior_table_is_carried_forward(tmp_path):
    key = CE.wdl_keys("chess", None, ["e2e4"])
    prior = WT.build([], "chess", deep_rows={key[0]: [10.0, 0.0, 0.0]})
    path = WT.write(prior, str(tmp_path / "p.json"))
    t = WT.build([], "chess", deep_rows={key[0]: [0.0, 5.0, 0.0]}, prior_table=path)
    assert t["rows"][key[0]] == [10.0, 5.0, 0.0] and t["stats"]["carried_rows"] == 1 and t["provenance"].endswith("prior own table")
    with pytest.raises(ValueError):
        WT.build([], "chess", prior_table={"format": "x", "game": "chess", "rows": {}})


def test_the_table_builder_imports_nothing_of_judges_or_a_chess_library():
    src = (ROOT / "catspace/incubator/analyst/discovery/wdl_table.py").read_text()
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names |= {a.name for a in node.names}
    assert not any(any(w in n for w in ("teacher", "judges", "chess", "sleep", "wdl_sleep")) for n in names), names


# ------------------------------------------------------------------------------------------------------------------------------------ calibration and the sleep

def test_calibrate_counts_and_uses_the_table(tmp_path):
    pos = [(None, ["e2e4"]), (None, ["d2d4"])]
    judge = FakeJudge((0.0, 1.0, 0.0))
    none = WS.calibrate(pos, judge, variant="chess", seconds=0.03)
    assert none["positions"] == 2 and none["table_hits"] == 0 and sum(none["calls"].values()) == 2 and 0 <= none["mae"] <= 1
    keys = [CE.wdl_keys("chess", None, m) for _s, m in pos]
    table = WT.build([], "chess", deep_rows={k[0]: [0.0, 200.0, 0.0] for k in keys})
    path = WT.write(table, str(tmp_path / "t.json"))
    with_t = WS.calibrate(pos, judge, variant="chess", table=path, seconds=0.03)
    assert with_t["table_hits"] == 2 and with_t["mae"] <= none["mae"] + 1e-9


def _deterministic(monkeypatch):
    def fake_read(setting, variant, start, moves, **kw):
        h = (len(moves) * 7 + sum(map(ord, "".join(moves)))) % 10 / 10.0
        return {"win": h, "draw": 1 - h, "loss": 0.0, "seconds": 0.0, "sharp": False, "table_key": None}
    monkeypatch.setattr(WS, "read", fake_read)


def test_no_judge_number_reaches_the_table(tmp_path, monkeypatch):
    _deterministic(monkeypatch)
    log, _ = wake_log(tmp_path, max_plies=7)
    spy = FakeJudge((1.0, 0.0, 0.0))
    quiet = []
    a = WS.sleep_wdl("chess", log, judge=spy, out=str(tmp_path / "a.json"), setting=FAST, max_deep=5, say=quiet.append)
    b = WS.sleep_wdl("chess", log, judge=None, out=str(tmp_path / "b.json"), setting=FAST, max_deep=5, say=quiet.append)
    ta, tb = json.loads((tmp_path / "a.json").read_text()), json.loads((tmp_path / "b.json").read_text())
    assert ta == tb and spy.asked and a["calibration"] is not None and b["calibration"] is None
    assert all("judge" not in k for k in ta) and "own" in ta["provenance"] and "uci" not in ta["provenance"].lower()


def test_sleep_wdl_reports_held_out_and_touched_and_prints_obs(tmp_path):
    log, _ = wake_log(tmp_path, games=2, max_plies=8)
    lines = []
    prior = WT.write(WT.build([], "chess"), str(tmp_path / "prior.json"))
    out = tmp_path / "new.json"
    rep = WS.sleep_wdl("chess", [log], prior_table=prior, judge=FakeJudge(), out=str(out), deep_seconds=0.03, max_deep=3, sample=3, max_settle=2, say=lines.append)
    assert rep["table"] == str(out) and out.exists() and rep["deep"]["read"] <= 3 and rep["choice"]["setting"]["name"] in ("long", "far", "end")
    cal = rep["calibration"]
    assert set(cal) == {"held_out_before", "held_out_after", "touched_before", "touched_after"}
    assert cal["touched_after"]["table_hits"] >= 1 and cal["touched_before"]["table_hits"] == 0
    text = "\n".join(lines)
    for frag in ("OBS wdl_sleep gaps", "OBS wdl_sleep reading", "OBS wdl_sleep deepen", "OBS wdl_sleep table", "held_out before", "touched after"):
        assert frag in text
    t = json.loads(out.read_text())
    assert t["stats"]["deep"]["rows"] > 0 and t["stats"]["settled"]["games"] <= 2


def test_the_cycle_closes_a_no_table_gap_becomes_a_table_hit(tmp_path):
    log1, g1 = wake_log(tmp_path, "w1.jsonl", games=1, max_plies=6, seconds=0.1, run="a")
    before = {(d["game"], d["ply"]): d["reading"] for d in WS.decision_readings(log1)}
    assert before and all(r["table"] == "none" for r in before.values())
    table = str(tmp_path / "t.json")
    WS.sleep_wdl("chess", log1, out=table, setting=FAST, max_deep=50, say=lambda *_: None)
    log2, _ = wake_log(tmp_path, "w2.jsonl", games=1, max_plies=6, seconds=0.1, run="b", table=table)
    after = {(d["game"], d["ply"]): d["reading"] for d in WS.decision_readings(log2)}
    first = min(before)                                                                  # (the seeded random opening makes the first decided position the same)
    assert after[first]["table"] == "exact" and after[first]["table_mass"] >= WS.DEEP_EXACT_MASS - 1


def test_sleep_overwriting_its_own_prior_table_still_measures_before(tmp_path):
    log, _ = wake_log(tmp_path, max_plies=6)
    out = str(tmp_path / "t.json")
    WS.sleep_wdl("chess", log, out=out, setting=FAST, max_deep=5, say=lambda *_: None)
    rep = WS.sleep_wdl("chess", log, prior_table=out, out=out, judge=FakeJudge(), setting=FAST, max_deep=5, sample=3, say=lambda *_: None)
    assert rep["stats"]["carried_rows"] > 0 and rep["calibration"]["touched_before"]["table_hits"] >= 1 and not os.path.exists(out + ".prior")


def test_cli_without_a_judge(tmp_path, capsys):
    log, _ = wake_log(tmp_path, max_plies=6)
    out = tmp_path / "c.json"
    assert WS.main(["chess", "--log", log, "--out", str(out), "--deep-seconds", "0.03", "--max-deep", "2", "--sample", "2"]) == 0
    assert out.exists() and "OBS wdl_sleep table written" in capsys.readouterr().out
    assert Path(WS.default_out("chess")).is_absolute()


# ------------------------------------------------------------------------------------------------------------------------------------ the cycle in initialize.py

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(GS, "LEARNED_ROOT", tmp_path / "learned")
    monkeypatch.setattr(GS, "SNAPSHOT_ROOT", tmp_path / "snapshots")
    shutil.copytree(ROOT / "rules" / "learned" / "chess", tmp_path / "learned" / "chess")
    for p in (tmp_path / "learned" / "chess" / "lemmas").glob("*.json"):
        p.unlink()
    return tmp_path


def centre(move: str) -> int:
    f, r = ord(move[2]) - ord("a"), int(move[3]) - 1
    return max(abs(2 * f - 7), abs(2 * r - 7))


class Prefers:
    """A fake rank source with a real preference: destinations near the centre (as in the 78.24 loop test)."""
    source, training_only, gap = "prefers", True, 100

    def rank(self, variant, moves, cap=None, start=None):
        legal = CE.state(variant, start, moves)[2]
        return sorted(((m, -50 * centre(m)) for m in legal), key=lambda x: (-x[1], x[0]))

    def grade(self, score):
        return score // self.gap


def test_run_size_with_wdl_writes_the_table_inside_the_sandbox_and_reports_cycles(tmp_path, store):
    lines = []
    rep = IN.run_size("chess", 3, source=Prefers(), store_root=tmp_path / "sb", limit=1, games_per_start=1, seconds=0.05, max_plies=6, tag="w27", wdl=True, cycles=1,
                      wdl_judge=FakeJudge(), wdl_settings={"deep_seconds": 0.03, "max_deep": 3, "sample": 3, "max_settle": 2}, say=lines.append)
    table = tmp_path / "sb" / "learned" / "chess" / "wdl_table.json"
    assert table.exists() and json.loads(table.read_text())["format"] == "wdl-table-1"
    assert len(rep["cycles"]) == 1 and set(rep["cycles"][0]) >= {"gaps_before", "sleep", "wdl_sleep", "gaps_after"} and rep["cycles"][0]["wdl_sleep"]["table"] == str(table)
    text = "\n".join(lines)
    assert "OBS init gaps chess wake a" in text and "OBS init gaps chess wake b" in text and "OBS wdl_sleep table" in text
    assert rep["wake_b"]["wdl_table"] == str(table) and rep["wake_a"]["wdl_table"] is None
    with pytest.raises(ValueError):
        IN.run_size("chess", 3, source=Prefers(), store_root=tmp_path / "sb2", cycles=2, say=lambda *_: None)
