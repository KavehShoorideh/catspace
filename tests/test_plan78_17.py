"""PLAN 78.17: the diagnostics over a wake log, the rank-based second kind of surprise (the teacher's ordering against a card's), the tie-breaker, the MultiPV parser.
Pieces only; no outside engine runs here (a real one is smoked by hand)."""
import json

import pytest

from catspace.incubator.analyst.discovery import diagnose as DG
from catspace.incubator.analyst.discovery import sleep as SL


def row(label, move, steps=None, proved=False):
    return {"kind": "position", "label": label, "move": move, "steps": steps or [], "error": None, "proved": proved}


def ledger_step(role, losses, depth=4, value=0, candidates=0, status="Read", parts=None, nodes=10, seconds=0.5):
    return {"operator": "ledger", "status": status, "depth": depth, "value": value, "candidates": candidates, "nodes": nodes, "seconds": seconds,
            "parts": parts or [], "events": [[role, f"lose {k}"] for k in losses], "line": []}


def test_78_17_diagnose_reports_ties_surprises_cost_depth_and_gaps(tmp_path):
    log = tmp_path / "w.jsonl"
    rows = [
        row("r:0:4:0", "a2a3", [ledger_step(0, [], depth=4, candidates=3, parts=[{"name": "ties", "nodes": 5, "seconds": 0.1}])]),
        row("r:0:5:1", "b7b5", [ledger_step(0, ["p"], depth=4, candidates=1)]),
        row("r:0:6:0", "h2h3", [{"operator": "forcing", "status": "Proved", "depth": 2, "events": [], "nodes": 3, "seconds": 0.2, "parts": []}], proved=True),
        row("r:0:7:1", "g7g6", [ledger_step(0, ["q"], depth=3, candidates=2, status="Gap")]),
    ]
    game = {"kind": "game", "run": "r", "game": 0, "variant": "g", "moves": ["x"] * 4 + ["a", "b", "c", "d"], "outcome": None, "unfinished": True}
    log.write_text("\n".join(json.dumps(r) for r in rows + [game]) + "\n")
    out = []
    rep = DG.diagnose(str(log), say=out.append)
    assert rep["decisions"] == 4 and rep["ledger_steps"] == 3
    assert rep["ties"]["with_ties"] == 2 and rep["ties"]["ties_read"] == 3 and abs(rep["ties"]["rate"] - 2 / 3) < 1e-9 and rep["ties"]["mean_count"] == 2.0
    assert rep["surprises"] == {"unforeseen loss": 1, "proved line": 1, "teacher disagreement": 0}
    assert rep["cost_by_operator"]["ledger"]["steps"] == 3 and rep["cost_by_operator"]["ledger"]["nodes"] == 30
    assert rep["cost_by_part"]["ledger.ties"]["nodes"] == 5
    assert rep["ledger_depth"] == {3: 1, 4: 2} and rep["gap"]["share"] == pytest.approx(1 / 3)
    assert any(o.startswith("OBS diagnose ties") for o in out)


def test_78_17_diagnose_says_so_when_the_wake_never_asked_for_ties(tmp_path):
    log = tmp_path / "w.jsonl"
    log.write_text(json.dumps(row("r:0:4:0", "a", [ledger_step(0, [])])) + "\n")
    rep = DG.diagnose(str(log), say=lambda *_: None)
    assert rep["ties"]["asked"] is False and rep["ties"]["with_ties"] == 0


def card(values):
    return {"moves": [{"name": n, "value": v, "lost": 0} for n, v in values.items()]}


def test_78_17_the_detector_flags_a_top_move_below_our_best_and_a_top_k_miss():
    c = card({"a": 5, "b": 5, "c": 0, "d": -3, "e": -3})
    d = SL.disagreement(c, [("c", 90), ("a", 50), ("b", 10)], "fake")
    assert d["reason"] == "top move below our best" and d["source_top"] == "c" and d["source"] == "fake" and d["source_order"] == ["c", "a", "b"]
    assert d["our_order"][:2] == ["a", "b"] and d["source_top_value"] == 0 and d["best"] == 5
    assert SL.disagreement(c, [("a", 90), ("c", 50)]) is None                       # the teacher's top is one of our optimal moves
    # the top is optimal here, so agreement; a top-k miss needs the top itself not optimal, never a tie at the top
    d2 = SL.disagreement(card({"a": 5, "b": 0, "c": 0, "d": 0}), [("b", 9), ("c", 8), ("d", 7), ("a", 1)], top_k=3)
    assert d2 is not None
    assert SL.disagreement(c, [("zz", 1)]) is None                                   # moves the card does not have are ignored


def test_78_17_the_tie_breaker_prefers_the_teachers_highest_ranked_saving_move():
    saving = [{"name": n, "value": 5, "lost": 0} for n in ("a", "b", "c")]
    out = SL.break_ties(saving, [("z", 100), ("c", 50), ("a", 10)])
    assert [m["name"] for m in out] == ["c", "a", "b"]                                # b is unranked: last
    assert [m["name"] for m in SL.break_ties(saving, [])] == ["a", "b", "c"]


INFO = """info depth 5 seldepth 6 multipv 1 score cp 31 nodes 100 pv e2e4 e7e5
info depth 5 seldepth 6 multipv 2 score cp 20 nodes 100 pv d2d4 d7d5
info depth 6 seldepth 7 multipv 1 score cp 35 nodes 300 pv e2e4 c7c5
info depth 6 seldepth 7 multipv 3 score mate -2 nodes 300 pv a2a3
info string noise
bestmove e2e4
"""


def test_78_17_multipv_lines_parse_to_a_ranking_with_mates_as_large_scores():
    r = SL.parse_multipv(INFO)
    assert r == [("e2e4", 35), ("d2d4", 20), ("a2a3", -(SL.MATE_SCORE - 2))]          # the deeper line of a rank wins; best first by rank
    assert SL.parse_multipv("info depth 1 score mate 3 pv h5h7")[0] == ("h5h7", SL.MATE_SCORE - 3)
    assert SL.parse_multipv("bestmove e2e4") == []
