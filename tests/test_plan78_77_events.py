"""PLAN 78.77, SYSTEM_SPEC 4 (how a line ends), 5 (worth) and 6 (the judge at the line's end): the event "a unit changed kind" is in the ledger (it has been since plan 78.5) and can be taken out as an ablation; the
best line printed follows the search's own move. Tests: a position where the mover can change a unit's kind next move is scored above the same position where it cannot; the default reproduces the numbers of the engine
before 78.77 on a fixed set (captured from the build of main at 5fe38ece); the Python mirror of the judge's change agrees with Rust; the print of the ladder reads the line after the search's move; the new files name no
object of the game."""
import re
from pathlib import Path

import pytest

import cs_engine as CE

from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.io import paths

ROOT = Path(__file__).resolve().parents[1]
DISC = ROOT / "catspace/incubator/analyst/discovery"


def _rebuilt() -> bool:
    try:
        CE.read_lines(OM.VARIANT, [(None, [])], [(0, 2, 0, 1, 0, 0, 0)], {"p": 1}, 1, 1000, 1, 1000, 1, kind_change=True)
        return True
    except TypeError:
        return False


needs_ext = pytest.mark.skipif(not _rebuilt(), reason="the extension is the build before plan 78.77 (no kind_change parameter)")

W = {"q": 4343, "b": 1878, "n": 1661, "r": 1605, "p": 1045}               # the generation-2 record's worths, as printed by the 78.76 run (a fixed table for the tests)
CAN = "4k3/8/P7/8/8/8/8/4K3 w - - 0 1"                                      # the mover's unit stands one step from changing kind
CANNOT = "4k3/8/8/8/8/8/P7/4K3 w - - 0 1"                                    # the same unit far from it


def _reads(fen: str, depth: int = 4, **kw) -> dict:
    r = CE.read_lines(OM.VARIANT, [(fen, [])], [(0, 2, 0, 1, 0, 0, 0)], W, depth, 200000, 7, 1500000, 1, **kw)[0]["reads"]
    return {x["names"][0]: x for x in r}


@needs_ext
def test_a_position_where_the_mover_can_change_a_units_kind_scores_above_the_same_position_where_it_cannot():
    can, cannot = _reads(CAN), _reads(CANNOT)
    assert max(x["value"] for x in can.values()) == W["q"] - W["p"]
    assert max(x["value"] for x in cannot.values()) == 0
    assert can["a6a7"]["changes"] == [(True, "p", "q")] and cannot["a2a3"]["changes"] == []
    # the search holds the same: the move scores the new kind less the old one
    r = CE.solve_equals_batch([(OM.VARIANT, CAN, [])], 0.3, worth=dict(W), valuation="r", threads=1, ledger_share=0.6)[0]
    assert r[8] == W["q"] - W["p"]


@needs_ext
def test_the_ablation_takes_the_event_out_and_only_the_event():
    on, off = _reads(CAN), _reads(CAN, kind_change=False)
    assert max(x["value"] for x in off.values()) == 0 and max(x["value"] for x in on.values()) == W["q"] - W["p"]
    assert {k: x["changes"] for k, x in on.items()} == {k: x["changes"] for k, x in off.items()}          # the same lines, the event is still listed
    r = CE.solve_equals_batch([(OM.VARIANT, CAN, [])], 0.3, worth=dict(W), valuation="r", threads=1, ledger_share=0.6, kind_change=False)[0]
    assert r[8] == 0
    # a taking is still scored with the event off: a capture by a move that also changes a unit's kind
    fen = "r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1"
    on, off = _reads(fen), _reads(fen, kind_change=False)
    assert on["b7a8q"]["value"] - off["b7a8q"]["value"] == W["q"] - W["p"] and off["b7a8q"]["value"] == W["r"]


# the numbers of the engine before plan 78.77, captured from the build of main at 5fe38ece (read_lines mode 2, depth 4, fuse 200000, the generation-2 worths above)
BEFORE = {
    "4k3/8/P7/8/8/8/8/4K3 w - - 0 1": [("a6a7", 3298), ("e1d1", 3298), ("e1d2", 3298), ("e1e2", 3298), ("e1f1", 3298), ("e1f2", 3298)],
    "r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1": [("b7a8b", 2438), ("b7a8n", 2221), ("b7a8q", 4903), ("b7a8r", 2165), ("b7b8b", -1045), ("b7b8n", -1045), ("b7b8q", -1045), ("b7b8r", -1045),
                                          ("e1d1", -1045), ("e1d2", -1045), ("e1e2", -1045), ("e1f1", -1045), ("e1f2", -1045)],
    "4k3/8/8/8/8/2q5/1P6/R3K3 w - - 0 1": [("b2c3", 4343), ("e1d1", -1045), ("e1e2", -2650), ("e1f1", -1045), ("e1f2", -2650)],
}


@needs_ext
def test_the_default_reproduces_the_numbers_before_the_event_was_a_parameter():
    for fen, want in BEFORE.items():
        got = sorted((n, x["value"]) for n, x in _reads(fen).items())
        assert got == sorted(want), fen
        assert sorted((n, x["value"]) for n, x in _reads(fen, kind_change=True).items()) == sorted(want)


@needs_ext
def test_the_python_mirror_of_the_judges_change_agrees_with_rust():
    rec = TC.Record(dict(W), {}, "test")
    reads = _reads("r3k3/8/P7/8/8/8/8/4K3 w - - 0 1", 1)
    for to in ("q", "r", "b", "n"):
        v = _reads("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", 1)[f"a7a8{to}"]["value"]
        assert v == rec.change_of("p", to) == W[to] - W["p"]
    assert rec.change_of("", "n") == W["n"] and rec.change_of("p", "unknown") == -W["p"]
    assert reads                                                                     # (the position reads)


@needs_ext
def test_mode_nine_reads_the_line_that_starts_with_the_games_last_move():
    ours = CE.read_lines(OM.VARIANT, [(CAN, ["a6a7"])], [(0, 9, 0, 1, 0, 0, 0)], W, 4, 200000, 7, 1500000, 1)[0]["reads"][0]
    assert ours["names"][0] == "a6a7" and ours["value"] == W["q"] - W["p"] and ours["changes"][0] == (True, "p", "q")
    same = _reads(CAN)["a6a7"]
    assert same["value"] == ours["value"] and same["names"][:3] == ours["names"][:3]
    assert CE.read_lines(OM.VARIANT, [(CAN, [])], [(0, 9, 0, 1, 0, 0, 0)], W, 4, 200000, 7, 1500000, 1)[0]["ok"] is False          # no move played: nothing to read


def test_the_ladders_print_reads_the_line_after_the_searchs_move(monkeypatch):
    t = LM.Tools()
    asked = []

    def fake(self, games, queries, depth=None, fuse=None):
        asked.append((games, queries, depth))
        return [{"ok": True, "reads": [{"names": ["x"], "events": [], "end": 0, "depth": 3, "cut": False, "nodes": 1, "changes": []}]}]
    monkeypatch.setattr(TC.Lines, "ask", fake)
    t.last_move = "e2e4"
    t.line({"moves": ["a2a3"]}, 1.0)
    assert asked[-1][0] == [(None, ["a2a3", "e2e4"])] and asked[-1][1][0][1] == 9
    t.line_from = "root"
    t.line({"moves": ["a2a3"]}, 1.0)
    assert asked[-1][0] == [(None, ["a2a3"])] and asked[-1][1][0][1] == 8
    t.line_from, t.last_move = "search", None
    t.line({"moves": ["a2a3"]}, 1.0)
    assert asked[-1][1][0][1] == 8                                                     # no search yet: the position's own best line


def test_the_event_is_a_parameter_that_defaults_to_the_old_behaviour(monkeypatch):
    assert LM.split_run(None) == (None, True) and LM.split_run("gen2") == ("gen2", True)
    assert LM.split_run("gen2@off") == ("gen2", False) and LM.split_run("selfplay_78_58@on") == ("selfplay_78_58", True)
    with pytest.raises(ValueError):
        LM.split_run("gen2@maybe")
    assert str(LM.out_root("full", "gen2@off")).endswith("ladder_78_77/gen2_off") and str(LM.out_root("short", "gen2@on")).endswith("ladder_78_77/gen2_on/short")
    assert str(LM.out_root("full", "gen2")).endswith("ladder_78_76/gen2")
    seen = []
    monkeypatch.setattr(CE, "read_lines", lambda *a, **k: seen.append(k) or [])
    TC.Lines(TC.Record({"p": 1}, {}, "t")).ask([(None, [])], [(0, 8, 0, 1, 0, 0, 0)])
    TC.Lines(TC.Record({"p": 1}, {}, "t"), kind_change=False).ask([(None, [])], [(0, 8, 0, 1, 0, 0, 0)])
    assert seen == [{}, {"kind_change": False}]                                        # the default call is the call of every earlier run
    assert LM.Tools().kind_change is True and LM.Tools().line_from == "search"


def test_the_new_files_name_no_object_of_the_game():
    pat = re.compile(r"\b(fen|san|uci|check|mate|pawn|knight|bishop|rook|queen|king)\b|[a-h][1-8]", re.I)
    for f in (DISC / "line_root_measure.py", ROOT / "rust/engine/py/src/lines_py.rs", ROOT / "rust/engine/tactics/src/lines.rs", ROOT / "rust/engine/tactics/src/ledger.rs"):
        for n, ln in enumerate(f.read_text().splitlines(), 1):
            if "78.77" in ln or "kind_change" in ln or "NoChange" in ln or f.name == "line_root_measure.py":
                assert not pat.search(ln), (f.name, n, ln)
