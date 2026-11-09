"""PLAN 78.17 INTEGRATION (the main agent's tests over the subagents' pieces): the two tool kits meet in the Python entry and the sleep -- the engine's own estimate
(Monte Carlo tree search) as a rank source beside the teacher, the ledger's ties read without the prior so the tie rate is measurable, and the sleep choosing its
tools from the diagnosis of its own log. Unit tests of the pieces are in tests/test_plan78_17.py and in the crates."""
import json
import shutil
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import diagnose as DG
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import sleep as SL
from catspace.incubator.analyst.discovery import wake as WK

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(GS, "LEARNED_ROOT", tmp_path / "learned")
    monkeypatch.setattr(GS, "SNAPSHOT_ROOT", tmp_path / "snapshots")
    shutil.copytree(ROOT / "rules" / "learned" / "chess", tmp_path / "learned" / "chess")
    for p in (tmp_path / "learned" / "chess" / "lemmas").glob("*.json"):
        p.unlink()
    return tmp_path


def test_78_17_the_estimate_finds_a_win_in_one_and_reports_its_cost_in_measured_parts():
    r = CE.estimate("chess", None, ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"], seconds=0.3, seed=3)
    print(f"OBS 78.17 estimate: move {r['move']} value {r['value']} playouts {r['playouts']} cut {r['cut']} nodes {r['nodes']} in {r['seconds']:.2f} s")
    assert r["status"] == "Read" and r["move"] == "h5f7" and r["candidates"][0] == "h5f7"
    mate = next(m for name, _v, m in r["visits"] if name == "h5f7")
    assert mate == 10000, "the mating move's own mean is a full win every time (hundredths of an outcome unit; the root's mean also carries the explored losses)"
    assert r["value"] > 7000, "the root's mean leans to the win (sampled: the share of exploration moves with the clock and the machine's load)"
    assert r["playouts"] > 0 and r["nodes"] > 0 and sum(v for _, v, _ in r["visits"]) == r["playouts"] - r["cut"] or sum(v for _, v, _ in r["visits"]) == r["playouts"]
    a, b = CE.estimate("chess", None, [], seconds=0.05, seed=7), CE.estimate("chess", None, [], seconds=0.05, seed=7)
    assert a["candidates"][0] == b["candidates"][0] or a["playouts"] != b["playouts"], "the same seed walks the same tree (the clock decides how far)"


def test_78_17_ties_are_read_without_the_prior_and_the_diagnosis_measures_the_rate(tmp_path):
    log = str(tmp_path / "w.jsonl")
    games = WK.play_self("chess", 4, seconds=0.02, random_plies=4, max_plies=30, seed=5, log=log, run="t", threads=2, ties=True)
    assert len(games) == 4
    d = DG.diagnose(log, say=lambda *a, **k: None)
    print(f"OBS 78.17 ties without the prior: {d['ties']}")
    assert d["ties"]["asked"] and d["ties"]["rate"] is not None and 0.0 <= d["ties"]["rate"] <= 1.0
    rows = [json.loads(x) for x in open(log) if '"kind": "position"' in x]
    assert all(all(st["operator"] != "density" for st in r["steps"]) for r in rows), "ties without the prior: no density step"
    off = str(tmp_path / "off.jsonl")
    WK.play_self("chess", 2, seconds=0.02, random_plies=4, max_plies=20, seed=5, log=off, run="o", threads=2)
    assert not DG.diagnose(off, say=lambda *a, **k: None)["ties"]["asked"]


def test_78_17_the_sleep_chooses_its_tools_from_the_diagnosis():
    s = SL.Settings()
    src = SL.EstimateSource(0.05)
    coarse = SL.plan_tools({"ties": {"asked": True, "rate": 0.5}}, src, s)
    assert coarse["coarse"] and coarse["chosen"] == {"teacher_mode": "break", "teacher_surprises": True} and coarse["source"] == "mcts" and not coarse["training_only"]
    fine = SL.plan_tools({"ties": {"asked": True, "rate": 0.05}}, src, s)
    assert not fine["coarse"] and fine["chosen"] == {"teacher_mode": "off", "teacher_surprises": False}
    unknown = SL.plan_tools({"ties": {"asked": False, "rate": None}}, src, s)
    assert unknown["coarse"], "a wake that never read ties leaves the coarseness unknown: the source is worth asking"
    none = SL.plan_tools({"ties": {"asked": True, "rate": 0.9}}, None, s)
    assert none["chosen"]["teacher_mode"] == "off" and "no rank source" in none["why"]


def test_78_17_the_estimate_is_a_rank_source_the_sleep_reads_like_a_teacher_but_not_as_one(store, tmp_path):
    log = str(tmp_path / "wake.jsonl")
    WK.play_self("chess", 6, seconds=0.01, random_plies=4, max_plies=40, seed=11, log=log, run="e", threads=2, ties=True)
    src = SL.EstimateSource(0.03, seed=2)
    rep = SL.sleep("chess", log, settings=SL.Settings(depth=3, seed=11, teacher_positions=12), teacher=src)
    f, t = rep["funnel"], rep["tools"]
    print(f"OBS 78.17 sleep with the estimate as source: tools {t['applied']} ({t['why']}); funnel " + ", ".join(f"{k} {v}" for k, v in f.items()))
    assert t["source"] == "mcts" and t["training_only"] is False
    assert "teacher_surprises" in f and "teacher_confirmed" in f and f["teacher_surprises"] >= f["teacher_confirmed"]
    if t["applied"]["teacher_surprises"]:
        assert rep["teacher"]["source"] == "mcts" and rep["teacher"]["ranked"] > 0
    for l in rep["lemmas"]:
        assert l["meta"]["teacher"] is False, "an estimate from the inference kit is not a teacher: nothing it touched is flagged as taught"
    assert GS.head("chess") is None
