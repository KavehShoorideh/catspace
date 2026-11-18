"""Plan 78.92: does the search's own choice carry the pruning fault (SYSTEM_SPEC 5, the tactical reasoner's ledger search; section 12 row "the search's equal reading and the line reading's verdict").

The instrument is rust/engine/tactics/src/choice_audit.rs (the Rust test is tactics/tests/choice_audit.rs), reached through cs_engine.choice_audit; the two prunings reach `solve_equals_batch` as keywords that default to the
search as it is (both on). The measure (choice_measure.py), the match (choice_play.py), the new Rust sources and the binding name no game object. The measure's arithmetic is tested on a synthetic audit.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cs_engine as CE
import numpy as np
import pytest

from catspace.incubator.analyst.discovery import choice_measure as CM
from catspace.incubator.analyst.discovery import choice_play as CP

ROOT = Path(__file__).resolve().parents[1]
WORTH = {"q": 4343, "b": 1878, "n": 1661, "r": 1605, "p": 1045}
NAMES = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
OPENING = ["e2e4", "e7e5", "g1f3", "b8c6"]


def _audit(**kw) -> dict:
    from catspace.incubator.analyst.discovery import odds_measure as OM
    return CE.choice_audit(OM.VARIANT, list(kw.pop("moves", OPENING)), WORTH, kw.pop("run_nodes", 8000), 64, kw.pop("big", 3_000_000), 2, True, kw.pop("other", None))


def test_the_audit_repeats_and_reads_the_reference_at_two_depths():
    a, b = _audit(), _audit()
    assert a == b                                                         # a node limit and fresh tables: no clock in the numbers
    assert a["depth"] >= 3 and a["played_name"]
    names = [(r["name"], r["depth"]) for r in a["readings"]]
    assert names == [(n, d) for n in ("plain", "no_null", "no_late", "as_played") for d in (a["depth"], a["depth"] + 2)]
    for r in a["readings"]:
        if r["best"] is not None and r["played"] is not None:
            assert r["played"] <= r["best"]                                # the played move is never above the best of the same reading
    assert a["verify_probe"] is not None and a["verify_nodes"] > 0


def test_a_second_move_is_valued_on_the_same_tables():
    a = _audit()
    other = next(r["best_name"] for r in a["readings"] if r["name"] == "plain" and r["best_name"])
    b = _audit(other=other)
    plain = next(r for r in b["readings"] if r["name"] == "plain" and r["depth"] == b["depth"])
    assert plain["other"] == plain["best"]                                 # the reference's own best move, asked as `other`, is the best
    assert next(r for r in a["readings"] if r["name"] == "plain")["other"] is None


def test_a_reading_the_fuse_cuts_has_no_numbers():
    a = _audit(big=200)
    plain = [r for r in a["readings"] if r["name"] == "plain"]
    assert all(r["best"] is None and r["played"] is None for r in plain)


def test_the_two_prunings_are_keywords_of_the_play_entry_and_default_to_the_search_as_it_is():
    pos = [(__import__("catspace.incubator.analyst.discovery.odds_measure", fromlist=["VARIANT"]).VARIANT, None, OPENING)]
    kw = dict(worth=WORTH, threads=1, ledger_share=0.4, ledger_depth=5)
    default = CE.solve_equals_batch(pos, 30.0, **kw)[0]
    explicit = CE.solve_equals_batch(pos, 30.0, null_move=True, late_reduce=True, **kw)[0]
    assert default[0] == explicit[0] and set(default[5]) == set(explicit[5])        # the same move, the same equal set (the nodes of a clocked equal reading vary a little)
    off = CE.solve_equals_batch(pos, 30.0, null_move=False, late_reduce=False, **kw)[0]
    assert off[2] > default[2]                                             # without the prunings the same depth costs more nodes


def test_the_match_players_differ_in_the_two_switches_alone():
    specs = CP.specs_of("late_off", "A", 2, 0.2, Path("/nonexistent"))
    assert len(specs) == 4
    for s in specs:
        now = s["players"][0 if s["colour"] == 0 else 1]
        other = s["players"][1 if s["colour"] == 0 else 0]
        assert (now["null_move"], now["late_reduce"]) == (True, True) and (other["null_move"], other["late_reduce"]) == (True, False)
        flags = ("null_move", "late_reduce", "verify_choice")
        assert {k: v for k, v in now.items() if k not in flags} == {k: v for k, v in other.items() if k not in flags}
        assert not now["verify_choice"] and not other.get("verify_choice", False)
    assert set(CP.VARIANTS) == {"late_off", "null_off", "both_off", "verify"}
    assert CP.VARIANTS["verify"] == {"null_move": True, "late_reduce": True, "verify_choice": True}


def test_the_verification_of_the_choice_is_a_keyword_that_defaults_to_off_and_its_state_is_in_the_row():
    from catspace.incubator.analyst.discovery import odds_measure as OM
    from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
    pos = [(OM.VARIANT, None, OPENING)]
    kw = dict(worth=WORTH, threads=1, ledger_share=0.4, ledger_depth=5)
    default = CE.solve_equals_batch(pos, 30.0, **kw)[0]
    off = CE.solve_equals_batch(pos, 30.0, verify_choice=False, **kw)[0]
    on = CE.solve_equals_batch(pos, 30.0, verify_choice=True, **kw)[0]
    assert len(default) == 15 and default[12] == 0 and default[13] == 0 and default[14] == 0.0 and off[12] == 0 and default[0] == off[0]
    assert on[12] in (1, 2, 3, 4) and on[13] > 0                          # asked: held, replaced, failed and kept, or cut; it cost nodes
    assert set(PM.equal_set(on)) == {"move", "proved", "nodes", "seconds", "error", "equals", "equal_indices", "depth", "value", "ties_seconds", "legal", "cut"}      # (the named row is the twelve it was)
    if on[12] == 1:
        assert on[0] == default[0]                                         # a held choice is the choice
    assert CE.best_move(OM.VARIANT, None, OPENING, 0.5, WORTH, verify_choice=True)[0] is not None


def _row(set_name, band, pid, d, best, played, other=None, nodes=1000, probe=0, value=0):
    rd = lambda name, depth, b, p: {"name": name, "depth": depth, "best": b, "best_index": 0, "best_name": "m", "played": p, "other": other, "nodes": nodes}
    rs = []
    for n in ("plain", "no_null", "no_late", "as_played"):
        rs += [rd(n, d, best, played), rd(n, d + 2, best, played)]
    return {"set": set_name, "pid": pid, "band": band, "units": 10, "id": f"x{pid}", "other": "m", "audit": {"depth": d, "value": value, "played": 0, "played_name": "m", "run_nodes": 100, "readings": rs,
                                                                                                     "verify_probe": probe, "verify_nodes": 50}, "wall": 0.0}


def test_the_shortfall_and_the_share_are_the_reference_best_less_the_played_value():
    a = _row("p600", "few", 0, 5, 1045, 0)["audit"]
    assert CM.short_of(a, "plain", False) == 1045 and CM.short_of(a, "plain", True) == 1045
    cut = _row("p600", "few", 1, 5, None, None)["audit"]
    assert CM.short_of(cut, "plain", False) is None
    rows = [_row("p600", "few", i, 5, 100, 100 if i % 4 else 0) for i in range(40)]
    txt, st = CM.share_line(rows, "plain", False)
    assert st["n"] == 40 and st["k"] == 10 and abs(st["share"] - 0.25) < 1e-9 and st["lo"] < 0.25 < st["hi"]
    assert "shortfall in the judge's units mean 100" in txt


def test_the_report_prints_the_condition_and_the_split_of_the_flagged_first_moves(tmp_path):
    rows = [_row("p600", b, i, 5, 100, 0 if i % 5 == 0 else 100) for i, b in enumerate(["few", "mid", "many"] * 20)]
    rows += [_row("flag", "mid", 100 + i, 5, 100, 100, other=(0 if i < 3 else 100)) for i in range(10)]
    f = tmp_path / "audit.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    out = []
    CM.stage_report(tmp_path, out.append, 0, "audit.jsonl", 20000, 8_000_000)
    text = "\n".join(out)
    assert "CONDITION part (1)" in text and "MET" in text                                 # 20 pct under on 60 positions: above 4.6 pct, band above 2.3 pct
    assert re.search(r"\(c\) flagged first moves, mid: 10 positions; the reference finished for 10; the first move is UNDER the plain best at the search's depth \(a real loss\) in 3 \(30\.0 pct\), LEVEL with it", text)
    assert json.loads((tmp_path / "condition_audit.json").read_text())["met_share"] is True


def test_the_paired_stage_counts_the_positions_that_leave_and_enter_the_state(tmp_path):
    before = [_row("p600", "few", i, 5, 100, 0 if i < 6 else 100) for i in range(10)]
    after = [_row("p600", "few", i, 5, 100, 100 if i < 4 else (0 if i == 9 else 100) if i >= 6 else 0) for i in range(10)]
    for r in after:
        r["audit"]["repair_state"], r["audit"]["repair_nodes"] = ("replaced" if r["pid"] < 4 else "held"), 50
    for r in before:
        r["audit"]["repair_state"], r["audit"]["repair_nodes"] = "off", 0
    (tmp_path / "b.jsonl").write_text("\n".join(json.dumps(r) for r in before))
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(r) for r in after))
    out = []
    CM.stage_paired(tmp_path, out.append, 0, "b.jsonl", "a.jsonl")
    text = "\n".join(out)
    assert "PAIRED p600 few: 10 positions with both runs, the same deepening in 10" in text and "left the state 4, entered it 1, stayed in it 2" in text
    assert "replaced the choice 4" in text


def test_the_bootstrap_band_is_a_band():
    m, lo, hi = CM.boot(np.array([1.0] * 5 + [0.0] * 95))
    assert abs(m - 0.05) < 1e-9 and lo <= m <= hi and hi < 0.2
    assert np.isnan(CM.boot(np.array([]))[0])


def test_the_twenty_cases_print_through_ladder_print(tmp_path, monkeypatch):
    from catspace.incubator.analyst.discovery import ladder_print as LP
    case = {"pid": 0, "id": "x", "band": "few", "units": 6, "moves": OPENING, "depth": 6, "value": 0, "played": "d2d3", "best0": 100, "best_move0": "d2d4", "played0": 0, "best2": None, "best_move2": None, "played2": None,
            "no_null0": [0, 0], "no_late0": [100, 0], "verify_probe": -5}
    root = tmp_path / "reasoner_choice_78_92" / "t"
    root.mkdir(parents=True)
    (root / "cases92.json").write_text(json.dumps({"rule": "R", "cases": [case]}))
    monkeypatch.setattr(LP.paths, "experiment", lambda *a: tmp_path.joinpath(*a))
    LP.choice_cases("t")
    text = (root / "cases.txt").read_text()
    assert "POSITION 1" in text and "shortfall 100" in text and "not finished within the fuse" in text


def test_the_new_sources_name_no_game_object():
    files = [ROOT / "rust/engine/tactics/src/choice_audit.rs", ROOT / "rust/engine/py/src/audit_py.rs", ROOT / "catspace/incubator/analyst/discovery/choice_measure.py", ROOT / "catspace/incubator/analyst/discovery/choice_play.py",
             ROOT / "rust/engine/tactics/tests/choice_audit.rs"]
    for p in files:
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if p.name == "choice_audit.rs" and "from_fen" in line or p.name == "choice_audit.rs" and p.parent.name == "tests" and ("const POSITIONS" in line):
                continue                                                      # (the Rust test builds positions from texts, as the other tests of this crate do)
            assert not re.search(NAMES, line, re.I), f"{p.name}:{i}: {line.strip()[:100]}"
