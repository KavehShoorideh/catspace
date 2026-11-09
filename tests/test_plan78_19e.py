"""Plan 78.19 stage E: the human-games index and HumanSource (training only). Everything in tmp_path; no lichess shard needed."""
import sqlite3
from pathlib import Path

import cs_engine as CE
import pytest

from catspace.incubator.analyst.discovery import human_index as HI
from catspace.incubator.analyst.discovery import sleep as SL
from catspace.io import paths

G1 = "e2e4 e7e5 g1f3".split()
G2 = "e2e4 c7c5".split()
G3 = "e2e4 e7e5 f1c4".split()
G4 = "d2d4 d7d5".split()
RECORDS = [(G1, 1), (G2, -1), (G3, [50, 50]), (G4, 0)]       # a result from white's view, or a points pair


def key(moves):
    return HI.signed(CE.state("chess", None, moves)[4])


@pytest.fixture
def index(tmp_path):
    out = str(tmp_path / "idx.sqlite")
    meta = HI.build("chess", "chess", iter(RECORDS), max_ply=40, out_path=out)
    return out, meta


def test_counts_and_mover_points(index):
    out, meta = index
    assert meta["games"] == "4" and meta["max_ply"] == "40"
    con = sqlite3.connect(out)
    row = lambda moves, mv: con.execute("SELECT n, points FROM moves WHERE key=? AND move=?", (key(moves), mv)).fetchone()
    assert row([], "e2e4") == (3, 100 + 0 + 50)          # white won G1, lost G2, drew G3
    assert row(["e2e4"], "e7e5") == (2, 0 + 50)          # black lost G1, drew G3
    assert row(["e2e4"], "c7c5") == (1, 100)             # black won G2
    assert row([], "d2d4") == (1, 50)
    assert int(meta["rows"]) == con.execute("SELECT COUNT(*) FROM moves").fetchone()[0]


def test_incremental_and_min_count(tmp_path):
    out = str(tmp_path / "i.sqlite")
    HI.build("chess", "chess", iter(RECORDS[:2]), out_path=out)
    m = HI.build("chess", "chess", iter(RECORDS[2:]), out_path=out)
    assert m["games"] == "4"
    assert sqlite3.connect(out).execute("SELECT n FROM moves WHERE key=? AND move='e2e4'", (key([]),)).fetchone() == (3,)
    out2 = str(tmp_path / "j.sqlite")
    HI.build("chess", "chess", iter(RECORDS), min_count=2, out_path=out2)
    assert {r[0] for r in sqlite3.connect(out2).execute("SELECT move FROM moves")} == {"e2e4", "e7e5"}


def test_max_ply(tmp_path):
    out = str(tmp_path / "p.sqlite")
    HI.build("chess", "chess", iter(RECORDS), max_ply=1, out_path=out)
    assert sqlite3.connect(out).execute("SELECT COUNT(*) FROM moves").fetchone()[0] == 2     # only e2e4 and d2d4 at the start


def test_human_source_rank_and_filter(index):
    out, _ = index
    h = HI.HumanSource(out)
    assert h.training_only and h.source == "human"
    assert h.rank("chess", []) == [("e2e4", 3), ("d2d4", 1)]
    assert h.rank("chess", ["e2e4"]) == [("e7e5", 2), ("c7c5", 1)]
    assert h.rank("chess", [], cap=1) == [("e2e4", 3)]
    assert h("chess", []) == "e2e4"
    d = h.detail("chess", ["e2e4"])
    assert d["e7e5"] == {"n": 2, "mean_points": 25.0}
    assert h.rank("chess", ["a2a3"]) == [] and h("chess", ["a2a3"]) is None           # unknown position
    # a stored move that is not legal here is dropped
    con = sqlite3.connect(out)
    con.execute("INSERT INTO moves VALUES (?, 'a1a8', 5, 0)", (key([]),))
    con.commit()
    assert "a1a8" not in dict(HI.HumanSource(out).rank("chess", []))
    h.close()


class Fake:
    def __init__(self, name, training_only=True):
        self.source, self.training_only = name, training_only


def test_plan_tools_prefers_human(index):
    out, _ = index
    s = SL.Settings()
    human, mcts = HI.HumanSource(out), Fake("mcts", False)
    coarse = {"ties": {"asked": True, "rate": 0.9}}
    fine = {"ties": {"asked": True, "rate": 0.0}}
    t = SL.plan_tools(coarse, [mcts, human], s)
    assert t["sources"] == ["human", "mcts"] and t["source"] == "human" and t["chosen"]["teacher_mode"] == "break" and "human" in t["why"]
    t = SL.plan_tools(fine, [mcts, human], s)
    assert t["coarse"] is False and t["sources"] == ["mcts", "human"] and t["chosen"]["teacher_mode"] == s.teacher_mode
    assert "sources" not in SL.plan_tools(coarse, mcts, s) and SL.plan_tools(coarse, mcts, s)["source"] == "mcts"
    assert "source" not in SL.plan_tools(coarse, [], s)


def test_source_of_and_disagreement(index):
    out, _ = index
    h = SL.source_of("human:" + out)
    assert isinstance(h, HI.HumanSource) and h.path == out and h.training_only
    card = {"moves": [{"name": "e2e4", "value": 0, "lost": 0}, {"name": "d2d4", "value": -1, "lost": 1}]}
    assert SL.disagreement(card, [("d2d4", 5), ("e2e4", 1)], h.source)["source"] == "human"
    assert SL.break_ties(card["moves"], h.rank("chess", []))[0]["name"] == "e2e4"
    from catspace.incubator.analyst.discovery import rank_bench as RB
    assert isinstance(RB.source_of("human:" + out, 0.1), HI.HumanSource)


def test_lichess_reader_shape():
    if not Path(paths.records("lichess_2019-01", "records_00000.parquet")).exists():
        pytest.skip("no lichess shard")
    moves, pts = next(HI.lichess_records_iter([0]))
    assert isinstance(moves, list) and len(pts) == 2
