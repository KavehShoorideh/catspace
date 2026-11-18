"""Plan 78.85: the two debts of the reasoner (SYSTEM_SPEC 5, the tactical reasoner: its equal reading and its line verdict).

Part one: the equal reading is asked again on a fresh table with the late-move reduction and the null move off (rust/engine/tactics/src/ledger.rs `ties` / `verify`; the Rust test is
rust/engine/tactics/tests/equal_reading.rs, from the traced position); the audit instrument (equal_audit.rs, py/src/audit_py.rs) answers the same question in separate readings. Part two: the verdict's net counts a unit
changing kind, and a line is read on through the captures its horizon resolves (lines.rs `settle_tail`) and is UNSETTLED, not REFUTED, when a fuse stops that. The measure (debts_measure.py) and the new Rust sources name no game object."""
from __future__ import annotations

import re
from pathlib import Path

import cs_engine as CE
import pytest

from catspace.incubator.analyst.discovery import threat_concrete as TC

ROOT = Path(__file__).resolve().parents[1]
WORTH = {"q": 4343, "b": 1878, "n": 1661, "r": 1605, "p": 1045}
HALF = {"q": 120.0, "b": 606.0, "n": 628.0, "r": 534.0, "p": 508.0}
REC = TC.Record(WORTH, HALF, "synthetic")
NAMES = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"


def _line(events, value, changes=(), end=0, cut=False, depth=6, **kw):
    return dict({"events": events, "changes": list(changes), "value": value, "end": end, "cut": cut, "depth": depth, "names": ["m1", "m2", "m3"], "settled": 3, "nodes": 10, "first": (1, 2), "lead": None, "prefix": []}, **kw)


# ------------------------------------------------------------------------------------------------------------------------------ part two: the verdict

def test_the_net_counts_a_unit_changing_kind_for_the_side_that_changed_it():
    lost_pawn = [(True, "p")]
    assert TC.verdict_of(_line(lost_pawn, -1045), REC)["verdict"] == "REFUTED"
    # the same loss while a unit of ours becomes a queen: the line is not behind, and the events now agree with the line search's own number
    v = TC.verdict_of(_line(lost_pawn, -1045 + (4343 - 1045), changes=[(True, "p", "q")]), REC)
    assert v["net"] == -1045 + 4343 - 1045 and v["kind_net"] == 4343 - 1045 and v["verdict"] != "REFUTED" and not v["pv_disagrees"]
    # the other side's change counts against the side the line was read for
    w = TC.verdict_of(_line([], -(4343 - 1045), changes=[(False, "p", "q")]), REC)
    assert w["net"] == -(4343 - 1045) and w["verdict"] == "REFUTED" and not w["pv_disagrees"]
    # a unit that arrives from nothing is the worth of the new kind
    assert TC.kind_change_net(_line([], 0, changes=[(True, "", "n")]), REC) == 1661


def test_the_net_leaves_the_kind_change_out_where_the_read_was_made_without_the_event():
    r = _line([(True, "p")], -1045, changes=[(True, "p", "q")], kind_change=False)
    assert TC.kind_change_net(r, REC) == 0 and not TC.verdict_of(r, REC)["pv_disagrees"]


def test_a_line_still_inside_an_exchange_when_the_tail_stopped_is_unsettled_not_refuted():
    assert TC.verdict_of(_line([(True, "q"), (False, "p")], -(4343 - 1045)), REC)["verdict"] == "REFUTED"
    v = TC.verdict_of(_line([(True, "q"), (False, "p")], -(4343 - 1045), unsettled=True), REC)
    assert v["verdict"] == "UNSETTLED" and v["unsettled"] and "UNSETTLED" in TC.VERDICTS
    assert TC.verdict_of(_line([(False, "p")], 1045, unsettled=True), REC)["verdict"] == "CONFIRMED"       # a gain is still a gain


def test_mirror_of_the_judges_change():
    assert REC.change_of("p", "q") == 4343 - 1045 and REC.change_of("", "n") == 1661


# ------------------------------------------------------------------------------------------------------------------------------ part two: the line read on (Rust, through the binding)

EXCHANGE = "4k3/8/2p5/3p4/4P3/2N5/8/4K3 w - - 0 1"


def test_a_line_read_to_one_reply_is_read_on_through_the_captures_the_horizon_resolves():
    res = CE.read_lines("chess", [(EXCHANGE, [])], [(0, 2, 0, 1, 0, 0, 0)], WORTH, 1, 40000, 7, 100000, 1)
    reads = res[0]["reads"]
    assert reads and all("unsettled" in r for r in reads)
    for r in reads:
        mine = sum(WORTH[k] for me, k in r["events"] if not me) - sum(WORTH[k] for me, k in r["events"] if me)
        assert mine == r["value"], (r["names"], r["events"], r["value"])               # what the line shows is what the search counted
    take = next(r for r in reads if r["names"][0] == "e4d5")
    assert len(take["events"]) >= 2 and not take["unsettled"]


# ------------------------------------------------------------------------------------------------------------------------------ part one: the audit instrument

def test_the_audit_answers_in_separate_readings_and_repeats():
    a = CE.equal_audit("chess", [], WORTH, 20000, 20000, 6, 400000)
    b = CE.equal_audit("chess", [], WORTH, 20000, 20000, 6, 400000)
    assert a is not None and a["depth"] >= 1 and len(a["rows"]) == 20
    assert a == b                                                                       # a node limit and no clock: the result repeats
    for x in a["rows"]:
        assert {"warm_probe", "cold_probe", "no_null_probe", "no_late_probe", "plain_probe", "exact"} <= set(x)
    assert set(a["verified"]) <= set(a["first"])                                        # the second reading only asks the candidates of the first


def test_value_at_reads_a_node_with_a_window_and_the_switches():
    v, nodes, names = CE.value_at("chess", [], 0, WORTH, 3, 1, -10**9, 10**9, True, True, 100000)
    w, _, _ = CE.value_at("chess", [], 0, WORTH, 3, 1, -10**9, 10**9, False, False, 100000)
    assert nodes > 0 and len(names) == 20 and v == w == 0


# ------------------------------------------------------------------------------------------------------------------------------ the wall

def test_the_new_sources_name_no_game_object():
    files = [ROOT / "rust/engine/tactics/src/equal_audit.rs", ROOT / "rust/engine/py/src/audit_py.rs", ROOT / "catspace/incubator/analyst/discovery/debts_measure.py", ROOT / "catspace/incubator/analyst/discovery/debts_verify.py"]
    for p in files:
        for i, line in enumerate(p.read_text().splitlines(), 1):
            assert not re.search(NAMES, line, re.I), f"{p.name}:{i}: {line.strip()[:100]}"


def test_the_switch_of_the_second_reading_reaches_the_binding_and_off_is_the_old_reading():
    """`verify_ties` False is the equal reading as it was: one reading, so no more nodes than on; the sets with it on are within the sets with it off is tested in Rust (tactics/tests/equal_reading.rs)."""
    from catspace.incubator.analyst.discovery import odds_measure as OM
    pos = [(OM.VARIANT, None, ["e2e4", "e7e5", "g1f3", "b8c6"])]
    off = CE.solve_equals_batch(pos, 1.0, worth=WORTH, threads=1, ledger_share=0.4, verify_ties=False)[0]
    on = CE.solve_equals_batch(pos, 1.0, worth=WORTH, threads=1, ledger_share=0.4)[0]           # the default is ON
    assert off[10] == on[10] and off[5] and on[5] and not off[11]                              # both read; the one-reading equal set is not cut by the clock here (a clock-free comparison of the sets is the Rust test)
