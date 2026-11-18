"""Plan 78.93: the verified choice a second time and what the late reduction buys (SYSTEM_SPEC 5, the tactical reasoner's ledger search and its verification of the choice).

Tested: the opening sets C and D (disjoint from A and B and from each other, A and B unchanged), the mode of the verification (a keyword of the play entry and of `best_move`, default "slot" = 78.92's form, off unless
asked), the players of the matches, the fix of `best_move(verify_ties=True)` (the switch reaches the search), the counting of what the verification did, and that the new sources name no game object.
The Rust test of the mode is rust/engine/tactics/tests/choice_mode.rs (and decision/tests/choice_mode_default.rs).
"""
from __future__ import annotations

import re
from pathlib import Path

import cs_engine as CE
import pytest

from catspace.incubator.analyst.discovery import choice_play93 as C93
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import tiebreak_measure as TM

ROOT = Path(__file__).resolve().parents[1]
WORTH = {"q": 4343, "b": 1878, "n": 1661, "r": 1605, "p": 1045}
NAMES = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
OPENING = ["e2e4", "e7e5", "g1f3", "b8c6"]


def test_the_sets_c_and_d_are_disjoint_from_a_and_b_and_from_each_other_and_a_and_b_are_as_they_were():
    sets = {n: TM.openings(n, 100) for n in "ABCD"}
    for n in "ABCD":
        assert len(sets[n]) == 100 and len({tuple(o) for _s, o in sets[n]}) == 100
    for i, x in enumerate("ABCD"):
        for y in "ABCD"[i + 1:]:
            assert not ({tuple(o) for _s, o in sets[x]} & {tuple(o) for _s, o in sets[y]}), (x, y)
            assert not ({s for s, _o in sets[x]} & {s for s, _o in sets[y]}), (x, y)
    assert sets["A"][0][0] == OM.MATCH_FIRST_SEED and sets["B"][0][0] == 41000      # (the rule of A and B is untouched)
    assert TM.openings("C", 40) == sets["C"][:40] and TM.openings("D", 40) == sets["D"][:40]      # (a run of 40 pairs plays the first 40)
    assert TM.openings("C", 100) == TM.openings("C", 100)                          # (deterministic)
    assert all("0" == l.split("openings in both ")[1][0] for l in TM.overlap_lines() if "overlap" in l)


def test_the_mode_is_a_keyword_that_defaults_to_slot_and_to_off():
    pos = [(OM.VARIANT, None, OPENING)]
    kw = dict(worth=WORTH, threads=1, ledger_share=0.4, ledger_depth=5)
    default = CE.solve_equals_batch(pos, 30.0, **kw)[0]
    slot = CE.solve_equals_batch(pos, 30.0, choice_mode="slot", **kw)[0]
    assert default[12] == 0 and slot[12] == 0                                      # the mode alone asks nothing: the switch is off
    assert default[0] == slot[0] and set(default[5]) == set(slot[5])
    with pytest.raises(ValueError):
        CE.solve_equals_batch(pos, 30.0, choice_mode="other", **kw)
    with pytest.raises(ValueError):
        CE.best_move(OM.VARIANT, None, OPENING, 0.2, WORTH, choice_mode="other")


def test_both_modes_ask_the_verification_and_the_same_choice_stands_when_it_holds():
    pos = [(OM.VARIANT, None, OPENING)]
    kw = dict(worth=WORTH, threads=1, ledger_share=0.4, ledger_depth=5, verify_choice=True)
    a = CE.solve_equals_batch(pos, 30.0, choice_mode="slot", **kw)[0]
    b = CE.solve_equals_batch(pos, 30.0, choice_mode="main", **kw)[0]
    assert a[12] in (1, 2, 3, 4) and b[12] in (1, 2, 3, 4) and a[13] > 0 and b[13] > 0
    assert a[12] == b[12] and a[0] == b[0]                                         # (a clock that does not bind: the mode changes where the time comes from, not the answer)
    assert CE.best_move(OM.VARIANT, None, OPENING, 0.5, WORTH, verify_choice=True, choice_mode="main")[0] is not None


def test_best_move_reaches_the_search_with_verify_ties():
    """The fault of 78.85: `best_move(verify_ties=True)` with nothing else on went to a path that built its own parameters and dropped the switch, so it played as with the switch off. In the opening
    (many equal moves) the second reading of the equal reading does not finish at 0.5 s, so the density prior has no equal moves to read: with the switch reaching the search, its line is absent."""
    from catspace.incubator.analyst.discovery import lemma_store as LS
    judge = LS.judge_of(OM.VARIANT, "material")
    off = CE.best_move(OM.VARIANT, None, [], 0.5, **judge, prior=True, prior_depth=3, verify_ties=False)[2]
    on = CE.best_move(OM.VARIANT, None, [], 0.5, **judge, prior=True, prior_depth=3, verify_ties=True)[2]
    assert any("planning density" in x for x in off)
    assert not any("planning density" in x for x in on), on


def test_the_players_differ_in_the_switches_alone_and_the_hint_match_has_both_sides_verify(tmp_path):
    specs = C93.specs_of("verify_main", "verify", "C", 2, 0.2, Path("/nonexistent"))
    assert len(specs) == 4
    for s in specs:
        a = s["players"][0 if s["colour"] == 0 else 1]
        b = s["players"][1 if s["colour"] == 0 else 0]
        assert a["choice_mode"] == "main" and b["choice_mode"] == "slot" and a["verify_choice"] and b["verify_choice"]
        flags = ("null_move", "late_reduce", "verify_choice", "choice_mode")
        assert {k: v for k, v in a.items() if k not in flags} == {k: v for k, v in b.items() if k not in flags}
    now = C93.specs_of("now", "late_off", "D", 1, 0.2, Path("/nonexistent"))[0]["players"][0]
    assert not now["verify_choice"] and now["null_move"] and now["late_reduce"]
    assert set(C93.PLAYERS) == {"now", "verify", "verify_main", "late_off"}
    hs = C93.hint_specs("A", 1, 0.2, tmp_path)
    assert all(p["verify_choice"] for s in hs for p in s["players"]) and {p["kind"] for s in hs for p in s["players"]} == {"sy_reasons", "tb_control"}


def test_the_movers_count_what_the_verification_did():
    class M(TM.Mover67):
        def __init__(self):
            self.verify_choice, self.stats = True, {}
    m = M()
    m.note_choice([None] * 12 + [2, 100, 0.5])
    m.note_choice([None] * 12 + [4, 10, 0.25])
    m.note_choice([None] * 12 + [0, 0, 0.0])
    assert (m.stats["vc_asked"], m.stats["vc_replaced"], m.stats["vc_cut"], m.stats["vc_held"], m.stats["vc_nodes"]) == (2, 1, 1, 0, 110)
    m.verify_choice = False
    assert m.choice_kw() == {}
    m.verify_choice, m.choice_mode = True, "main"
    assert m.choice_kw() == {"verify_choice": True, "choice_mode": "main"}


def test_the_new_sources_name_no_game_object():
    files = [ROOT / "rust/engine/tactics/tests/choice_mode.rs", ROOT / "rust/engine/decision/tests/choice_mode_default.rs", ROOT / "catspace/incubator/analyst/discovery/choice_play93.py"]
    for p in files:
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "from_fen" in line or '"chess"' in line or "4k3/" in line:
                continue                                                      # (the Rust test builds a position from a text, as the other tests of this crate do)
            assert not re.search(NAMES, line, re.I), f"{p.name}:{i}: {line.strip()[:100]}"
    for p in [ROOT / "rust/engine/tactics/src/ledger.rs", ROOT / "rust/engine/tactics/src/operators.rs", ROOT / "rust/engine/decision/src/lib.rs"]:
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "78.93" in line:
                assert not re.search(NAMES, line, re.I), f"{p.name}:{i}: {line.strip()[:100]}"
