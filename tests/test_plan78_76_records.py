"""PLAN 78.76, SYSTEM_SPEC 5 and 9: the ladder under each measured record of the kinds. The default record is unchanged; a named record reaches the search AND the line read; the two new record files are made from the stored fits and
their kinds map through the game's codec; the wall of the ladder holds in the new results; the new files name no object of the game."""
import json
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import ladder as LD
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import ladder_records as LR
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.io import paths

ROOT = Path(__file__).resolve().parents[1]
DISC = ROOT / "catspace/incubator/analyst/discovery"
FILES = [DISC / n for n in ("ladder_records.py", "ladder_compare.py", "ladder_measure.py", "ladder.py", "threat_concrete.py")]
NAMES = tuple(LR.SOURCES)


def _have_inputs() -> bool:
    return Path(paths.experiment("odds_78_61", "full", "gen2", "record.json")).exists() and all(Path(paths.experiment(*s[0])).exists() for s in LR.SOURCES.values())


pytestmark = pytest.mark.skipif(not _have_inputs(), reason="the stored fits are not on disk")


def test_the_default_record_is_unchanged():
    default = TC.Record.load()
    assert default.source == str(Path(paths.experiment(*TC.RECORD)))
    assert TC.Record.load("gen2").worth == default.worth and TC.Record.load("gen2").half == default.half
    assert default.worth == OM.players_of(3, OM.out_root() / "full")["worth"]               # the worths the search was always handed
    t = LM.Tools()
    assert t.worth == default.worth and t.lines.record.worth == default.worth


def test_a_named_record_reaches_the_search_and_the_line_read(monkeypatch):
    import cs_engine as CE
    seen = {}

    def fake_solve(batch, seconds, worth=None, **kw):
        seen["search"] = dict(worth)
        return [{"move": "m", "equals": ["m"], "nodes": 1, "depth": 1, "proved": False}]

    def fake_lines(variant, games, queries, worth, *a):
        seen["lines"] = dict(worth)
        return [{"ok": True, "reads": [{}]}]
    monkeypatch.setattr(CE, "solve_equals_batch", fake_solve)
    monkeypatch.setattr(CE, "read_lines", fake_lines)
    monkeypatch.setattr(LM.PM, "equal_set", lambda row: dict(row))
    for name in NAMES:
        LR.make(name)
        rec = TC.Record.load(name)
        t = LM.Tools(name)
        assert t.record.source == rec.source and t.worth == rec.worth
        t.search({"moves": []}, 0.1)
        t.line({"moves": []}, 1.0)
        t.forced({"moves": []})
        assert seen["search"] == rec.worth and seen["lines"] == rec.worth
        assert rec.worth != TC.Record.load().worth                                          # a different record, so a different judge


def test_a_position_where_the_records_must_differ():
    """On real positions of the ladder's sample the two judges are read through the engine: the sets the search holds equal, or the settled end of the best line, differ for at least one."""
    pos = ROOT / "artifacts/experiments/ladder_78_68/full/positions.jsonl"
    if not pos.exists():
        pytest.skip("no sample on disk")
    rows = [json.loads(x) for x in pos.read_text().splitlines() if x.strip()][:60]
    for name in NAMES:
        LR.make(name)
    base, other = LM.Tools(), LM.Tools(NAMES[0])
    differ = []
    for r in rows:
        p = {"moves": r["moves"]}
        a, b = base.search(p, 0.1), other.search(p, 0.1)
        la, lb = base.line(p, 1.0), other.line(p, 1.0)
        if sorted(a["equals"]) != sorted(b["equals"]) or la["events"] != lb["events"] or la["names"] != lb["names"]:
            differ.append(r["id"])
        if len(differ) >= 1:
            break
    assert differ, "no position among the first 60 reads differently under the two records"


def test_the_record_files_are_made_from_the_stored_fits_and_their_kinds_map_through_the_codec():
    gen2 = TC.Record.load()
    for name in NAMES:
        out = LR.make(name)
        rec = json.loads(out.read_text())
        src, keys = LR.SOURCES[name]
        order = LR._dig(json.loads(Path(paths.experiment(*src)).read_text()), keys)
        assert rec["heads"]["odds_effect"]["kind_values"] == {o["kind"]: int(round(1000 * o["term"])) for o in order}
        loaded = TC.Record.load(name)
        assert set(loaded.worth) == set(gen2.worth) and set(loaded.half) == set(gen2.half)      # the game's own words for the same kinds
        assert set(loaded.worth) <= set(OM.kernel_names().values())
        assert all(v > 0 for v in loaded.worth.values()) and all(h > 0 for h in loaded.half.values())
        # the half width is the printed band, as the generation-2 record's is
        for o in order:
            word = OM.kernel_names()[o["kind"]]
            assert abs(loaded.half[word] - 500.0 * (o["hi"] - o["lo"])) < 1e-9
    assert TC.named("gen2") == str(Path(paths.experiment(*TC.RECORD)))
    with pytest.raises(FileNotFoundError):
        TC.named("no_such_record")


def test_the_record_orders_the_kinds_as_stored():
    """Only the order the stored fit prints: pawn below knight below bishop below rook below queen in both new records (the words of the kinds come from the fit, never from this file)."""
    for name in NAMES:
        src, keys = LR.SOURCES[name]
        order = LR._dig(json.loads(Path(paths.experiment(*src)).read_text()), keys)
        terms = [o["term"] for o in order]
        assert terms == sorted(terms)                                                       # the stored order of kinds is ascending, as printed


def test_the_wall_holds_in_the_new_results():
    found = False
    for name in NAMES + ("gen2",):
        root = LM.out_root("full", name)
        res = root / "results.jsonl"
        if not res.exists():
            continue
        found = True
        assert not re.search(r"oracle|centipawn|multipv|\"score\"|accepted", res.read_text(), re.I)
        lst = root / "surprise_list.jsonl"
        rows = [json.loads(x) for x in lst.read_text().splitlines() if x.strip()]
        assert all(LD.wall_violations(r) == [] for r in rows)
    if not found:
        pytest.skip("no full run on disk")


def test_the_new_files_name_no_object_of_the_game():
    words = re.compile(r"\b(fen|san|uci|check|mate|king|queen|rook|bishop|knight|pawn|chess)s?\b", re.I)
    square = re.compile(r"\b[a-h][1-8]\b")
    for p in FILES:
        text = p.read_text()
        assert words.search(text) is None, (p.name, words.search(text).group(0) if words.search(text) else None)
        assert square.search(text) is None, (p.name, square.search(text).group(0) if square.search(text) else None)
