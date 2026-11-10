"""Plan 78.22 stages 1 and 2: start positions through the whole pipeline (wake --starts, the log, the sleep's items, the rank sources), the material curriculum (a codec at
the boundary, `required` found by test, `starts`), and the surprise detectors as modules (Hindsight, Disagreement, Regret). Fast and deterministic: fake rank sources;
the one Stockfish test is skipped when the engine is missing."""
import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import curriculum as CU
from catspace.incubator.analyst.discovery import detectors as D
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import human_index as HI
from catspace.incubator.analyst.discovery import sleep as SL
from catspace.incubator.analyst.discovery import wake as WK

ROOT = Path(__file__).resolve().parents[1]
STOCKFISH = "/opt/homebrew/bin/stockfish"
INITIAL = CE.state("chess", None, [])[0]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(GS, "LEARNED_ROOT", tmp_path / "learned")
    monkeypatch.setattr(GS, "SNAPSHOT_ROOT", tmp_path / "snapshots")
    shutil.copytree(ROOT / "rules" / "learned" / "chess", tmp_path / "learned" / "chess")
    for p in (tmp_path / "learned" / "chess" / "lemmas").glob("*.json"):
        p.unlink()
    return tmp_path


# ------------------------------------------------------------------------------------------------------------------------------------ the codec and the curriculum

def test_78_22_the_chess_codec_round_trips_and_keeps_castling_only_where_the_pieces_stand():
    c = CU.ChessCodec()
    units = c.units(INITIAL)
    assert len(units) == 32 and units[0] == "a1" and units == sorted(units, key=lambda u: (int(u[1]), u[0])), "a stable order: the squares' own"
    assert c.keep(INITIAL, units) == INITIAL
    base = [u for u in units if u not in ("a1", "h1", "a8", "h8")]
    assert c.keep(INITIAL, base).split()[2] == "-", "kings without rooks castle nowhere"
    assert c.keep(INITIAL, ["e1", "e8", "h1"]).split()[2] == "K"
    assert c.keep(INITIAL, ["e1", "e8", "a1", "h8"]).split()[2] == "Qk"
    assert c.keep(INITIAL, ["d1", "e1", "e8", "h1"]).split()[2] == "K"
    assert c.keep(INITIAL, ["e8", "h1", "h8"]).split()[2] == "k", "no white king: no white right"
    # en passant is dropped and the clocks reset
    after = CE.state("chess", None, ["e2e4"])[0]
    out = c.keep(after, c.units(after))
    assert out.split()[3] == "-" and out.split()[4:] == ["0", "1"] and out.split()[1] == "b"


def test_78_22_the_required_units_are_found_by_test():
    assert CU.required("chess") == ["e1", "e8"]
    # not by name: a codec whose handles are opaque still gives the answer by the kernel's verdict
    class Indexed(CU.ChessCodec):
        def units(self, position):
            return list(range(len(super().units(position))))

        def keep(self, position, units):
            names = super().units(position)
            return super().keep(position, [names[u] for u in units])
    CU.CODECS["chess_indexed"] = Indexed()
    try:
        assert CU.required("chess_indexed", "chess") == [4, 28], "the two kings, by the kernel's verdict on a codec whose handles say nothing"
    finally:
        del CU.CODECS["chess_indexed"]
    with pytest.raises(ValueError):
        CU.required("no-such-game")


def test_78_22_starts_have_the_size_are_legal_and_deterministic():
    s3 = CU.starts("chess", 3)
    assert len(s3) == len(set(s3)) == 22, "30 third units, minus the 8 lone knights and bishops (insufficient material: the kernel says over)"
    for f in s3:
        fen, mover, legal, outcome, _ = CE.state("chess", f, [])
        assert outcome is None and legal and mover == "white"
        assert sum(ch.isalpha() for ch in f.split()[0]) == 3
    assert s3 == CU.starts("chess", 3)
    s5 = CU.starts("chess", 5, limit=10, seed=7)
    assert len(s5) == 10 and s5 == CU.starts("chess", 5, limit=10, seed=7) and s5 != CU.starts("chess", 5, limit=10, seed=8)
    assert all(sum(ch.isalpha() for ch in f.split()[0]) == 5 for f in s5)
    big = CU.survey("chess", 12, limit=5, seed=1)                 # 30 million combinations: sampled lazily, never enumerated
    assert big["total"] > 10**7 and big["kept"] == 5 and big["candidates"] < 100
    with pytest.raises(ValueError):
        CU.starts("chess", 12)
    with pytest.raises(ValueError):
        CU.starts("chess", 1)
    assert [CU.unrank(5, 3, i) for i in range(10)] == [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]


def test_78_22_the_curriculum_cli_writes_the_file_wake_reads(tmp_path, capsys):
    out = str(tmp_path / "s.txt")
    assert CU.main(["chess", "--size", "3", "--out", out]) == 0
    line = capsys.readouterr().out
    assert line.startswith("OBS curriculum chess") and "candidates 30" in line and "kept 22" in line and "'over': 8" in line
    assert len(WK.read_starts(out)) == 22


# ------------------------------------------------------------------------------------------------------------------------------------ wake and the log

def test_78_22_wake_plays_from_starts_and_the_log_keeps_them(tmp_path):
    starts = CU.starts("chess", 3, limit=2, seed=1)
    log = str(tmp_path / "w.jsonl")
    games = WK.play_self("chess", 3, seconds=0.01, random_plies=0, max_plies=4, seed=1, log=log, run="s", threads=2, starts=starts)
    assert [g["start"] for g in games] == [starts[0], starts[1], starts[0]]
    for g in games:
        assert g["moves"] and CE.state("chess", g["start"], g["moves"])[2] is not None
    d, g = SL.read_log(log)
    assert {k: v["start"] for k, v in g.items()} == {("s", 0): starts[0], ("s", 1): starts[1], ("s", 2): starts[0]} and d
    plain = WK.play_self("chess", 1, seconds=0.01, random_plies=2, max_plies=6, seed=1, log=str(tmp_path / "p.jsonl"), run="p")
    assert plain[0]["start"] is None
    # an old log (no `start` key) reads as the game's own start
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"kind": "game", "run": "o", "game": 0, "variant": "chess", "moves": [], "outcome": None, "unfinished": True}) + "\n")
    assert SL.read_log(str(old))[1][("o", 0)]["start"] is None
    # the starts file: blank lines and comments ignored
    f = tmp_path / "starts.txt"
    f.write_text("# a comment\n\n" + starts[0] + "\n  \n" + starts[1] + "\n")
    assert WK.read_starts(str(f)) == starts[:2]


# ------------------------------------------------------------------------------------------------------------------------------------ the rank sources carry the start

class Seq:
    """A fake rank source over the kernel's legal moves: the sorted names, scored by position, the best first; records the calls. `gap` is the source's own."""
    source, training_only = "fake", True

    def __init__(self, gap=None, best=None):
        self.gap, self.best, self.calls = gap, best or {}, []

    def rank(self, variant, moves, cap=None, start=None):
        self.calls.append((tuple(moves), start))
        legal = sorted(CE.state(variant, start, moves)[2])
        top = self.best.get((start, tuple(moves)))
        if top in legal:
            legal.remove(top)
            legal.insert(0, top)
        return [(m, -10 * i) for i, m in enumerate(legal)]


class NoStart:
    """A third-party source from before the keyword."""
    source, training_only = "old", True

    def rank(self, variant, moves, cap=None):
        return [(m, -i) for i, m in enumerate(sorted(CE.state(variant, None, moves)[2]))]


def test_78_22_a_start_is_passed_only_when_there_is_one():
    ctx = D.Context("chess", SL.Settings(), {}, {}, positions={"a": (None, ["e2e4"]), "b": ("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1", [])})
    old = NoStart()
    assert len(ctx.ranking(old, "a")) == 20, "a source without the keyword is asked on an ordinary game"
    fake = Seq()
    ctx.ranking(fake, "b")
    assert fake.calls == [((), "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")]
    assert ctx.ranking(fake, "b") is ctx.ranking(fake, "b") and len(fake.calls) == 1, "one source call serves every detector"


def test_78_22_the_human_source_answers_a_start_by_the_kernel_key(tmp_path):
    out = str(tmp_path / "h.sqlite")
    HI.build("chess", "chess", iter([(["e2e4", "e7e5"], 1), (["e2e4", "c7c5"], -1)]), out_path=out)
    h = HI.HumanSource(out)
    after = CE.state("chess", None, ["e2e4"])[0]
    assert h.rank("chess", [], start=after) == h.rank("chess", ["e2e4"]) == [("c7c5", 1), ("e7e5", 1)]
    assert h.rank("chess", [], start="4k3/8/8/8/8/8/8/R3K3 w Q - 0 1") == []
    other = HI.HumanSource(out, key_of=lambda v, m: 0)
    assert other.rank("chess", [], start=after) == [], "a source keyed another way knows only the game's own start"
    h.close()
    other.close()


@pytest.mark.skipif(not os.path.exists(STOCKFISH), reason="no Stockfish")
def test_78_22_the_uci_teacher_ranks_from_a_start_position():
    t = SL.UciTeacher(STOCKFISH, 30)
    try:
        start = "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"
        r = t.rank("chess", [], start=start)
        legal = set(CE.state("chess", start, [])[2])
        assert r and {m for m, _ in r} <= legal and len(r) == len(legal) and t.gap == 100
        r2 = t.rank("chess", ["a1a8"], start=start)
        assert {m for m, _ in r2} == set(CE.state("chess", start, ["a1a8"])[2])
        assert t._position(["a1a8"], start) == "position fen " + start + " moves a1a8" and t._position([]) == "position startpos"
    finally:
        t.close()


# ------------------------------------------------------------------------------------------------------------------------------------ the detectors

def row(label, move, steps=None, proved=False):
    return {"kind": "position", "label": label, "move": move, "steps": steps or [], "error": None, "proved": proved}


def ledger_step(role, losses, depth=4):
    return {"operator": "ledger", "status": "Read", "depth": depth, "value": 0, "events": [[role, f"lose {k}"] for k in losses], "line": []}


def test_78_22_hindsight_is_the_surprises_rule(tmp_path):
    log = tmp_path / "w.jsonl"
    rows = [row("r:0:4:0", "a2a3", [ledger_step(0, [])]), row("r:0:5:1", "b7b5", [ledger_step(0, ["p"])]), row("r:0:6:0", "a3a4", [ledger_step(0, ["p"])]),
            row("r:0:7:1", "b5a4", [ledger_step(0, ["p"])])]
    game = {"kind": "game", "run": "r", "game": 0, "variant": "chess", "start": "somewhere", "moves": ["x"] * 4 + ["a2a3", "b7b5", "a3a4", "b5a4"], "outcome": None, "unfinished": True}
    log.write_text("\n".join(json.dumps(r) for r in rows + [game]) + "\n")
    d, g = SL.read_log(str(log))
    ctx = D.Context("chess", SL.Settings(min_units=1), d, g)
    h = D.detector_of("hindsight")
    assert h.detect(ctx) == SL.surprises(d, g, 1) and len(h.detect(ctx)) == 1 and not h.training_only and h.name == "hindsight"
    assert h.detect(ctx)[0]["start"] == "somewhere"
    ctx.settings = SL.Settings(min_units=2)
    assert h.detect(ctx) == SL.surprises(d, g, 2) == []


def card(values, lost=None, cut=False, cid="c"):
    return {"id": cid, "group": "g", "cut": cut, "moves": [{"name": n, "value": v, "lost": (lost or {}).get(n, 0), "removed": [], "added": [], "loss_removed": [], "loss_added": []} for n, v in values.items()]}


def test_78_22_disagreement_detector_is_the_disagreement_rule(store, tmp_path):
    items = [("chess", None, [], "p0", "g0")]
    cards = CE.build_cards(items, GS.spec_texts("chess"), json.dumps(GS.vocabulary("chess")), 2)
    c = json.loads(cards.card(0))
    best = max(c["moves"], key=lambda m: m["value"])["name"]
    worst = min(c["moves"], key=lambda m: (m["value"], m["name"]))["name"]
    src = Seq(best={(None, ()): worst})
    settings = SL.Settings(depth=2, teacher_positions=5)
    ctx = D.Context("chess", settings, {}, {}, cards=cards, dev_idx=[0], positions={"p0": (None, [])}, played={"p0": best}, rep={})
    det = D.detector_of("disagreement", src)
    got = det.detect(ctx)
    expect = SL.disagreement(c, src.rank("chess", []), "fake", settings.teacher_top_k)
    if best == worst:
        assert expect is None and got == []
    else:
        assert len(got) == 1 and got[0]["disagreement"] == expect and got[0]["kind"] == "teacher" and got[0]["start"] is None and got[0]["source_top"] == worst
    assert ctx.rep["teacher"]["ranked"] == 1 and det.training_only


def test_78_22_regret_gap_rule_order_fallback_and_mates():
    rank = [("a", 120), ("b", 60), ("c", 30), ("d", -50), ("e", -SL.MATE_SCORE)]
    gap = D.Regret(Seq(gap=100))
    assert gap.rule(rank, "a", 3) is None
    assert gap.rule(rank, "b", 3) is None, "60 below the best is under the source's own gap"
    ev = gap.rule(rank, "d", 3)
    assert ev["rule"] == "gap" and ev["source_top"] == "a" and ev["played_score"] == -50 and ev["top_score"] == 120 and ev["gap"] == 100
    assert gap.rule(rank, "e", 3)["rule"] == "gap", "a missed mate always counts"
    assert gap.rule(rank, "zz", 3) is None, "a move the source did not score cannot be measured in its units"
    assert D.Regret(Seq(gap=10)).rule(rank, "b", 3)["rule"] == "gap"
    order = D.Regret(Seq())                                            # no gap declared: order alone
    assert order.rule(rank, "c", 3) is None and order.rule(rank, "d", 3)["rule"] == "order" and order.rule(rank, "d", 3)["gap"] is None
    assert order.rule(rank, "c", 2)["rule"] == "order"
    assert order.rule([], "a", 3) is None and order.rule(rank, None, 3) is None
    assert SL.UciTeacher.gap == 100 and SL.Settings().regret is True


def test_78_22_regret_confirmed_tied_unconfirmed_and_cut_on_hand_made_cards():
    settings = SL.Settings(depth=3)
    sur = {"id": "x", "group": "g", "start": None, "moves_before": [], "played": "p", "source_top": "t", "kind": "regret", "source": "fake",
           "regret": {"source_top": "t", "top_score": 300, "played_score": 0, "gap": 100, "rule": "gap"}}

    def verdict(dc):
        ctx = D.Context("chess", settings, {}, {}, cards=[0], dev_idx=[0], rep={"regret": {"confirmed": [], "tied": [], "unconfirmed": [], "cut": [], "open": []}})
        st, parts = D.Regret(Seq()).verdict(sur, dc, ctx)
        return st, parts, ctx.rep["regret"]
    st, parts, r = verdict(card({"t": 2, "p": 1, "o": 0}))
    assert st == "confirmed" and r["confirmed"] == ["x"]
    x, dc, top, tm, played = parts
    assert x["kind"] == "regret" and x["played"] == "p" and tm == "t" and top["name"] == "t" and played["name"] == "p"
    st, parts, r = verdict(card({"t": 2, "p": 2, "o": 0}))
    assert st == "tied" and parts is None and r["tied"] == ["x"]
    assert r["open"] == [{"id": "x", "played": "p", "source_top": "t", "played_score": 0, "top_score": 300, "gap": 100, "rule": "gap", "source": "fake"}]
    st, _, r = verdict(card({"t": 1, "p": 0, "o": 3}))
    assert st == "unconfirmed" and r["unconfirmed"] == ["x"] and not r["tied"], "the source's top is not optimal for us: nothing to learn"
    st, _, r = verdict(card({"p": 1, "o": 0}))
    assert st == "unconfirmed"
    st, _, r = verdict(card({"t": 2, "p": 1}, cut=True))
    assert st == "cut" and r["cut"] == ["x"]


def test_78_22_the_registry():
    assert isinstance(D.detector_of("hindsight"), D.Hindsight)
    src = Seq()
    assert isinstance(D.detector_of("regret", src), D.Regret) and isinstance(D.detector_of("disagreement", src), D.Disagreement) and D.detector_of("teacher", src).name == "teacher"
    assert D.detector_of("regret", src).name == "regret" and D.detector_of("regret", src).training_only is True
    assert D.detector_of("regret", SL.EstimateSource()).training_only is False
    with pytest.raises(ValueError):
        D.detector_of("regret")
    with pytest.raises(ValueError):
        D.detector_of("nope")


# ------------------------------------------------------------------------------------------------------------------------------------ the sleep, end to end

def test_78_22_a_sleep_over_games_from_three_unit_starts_reports_the_regret_funnel(store, tmp_path):
    starts = CU.starts("chess", 3, limit=4, seed=2)
    log = str(tmp_path / "w.jsonl")
    games = WK.play_self("chess", 8, seconds=0.01, random_plies=0, max_plies=40, seed=5, log=log, run="c3", threads=2, starts=starts, ties=True)
    assert {g["start"] for g in games} == set(starts)
    src = Seq(gap=10)
    rep = SL.sleep("chess", log, settings=SL.Settings(depth=3, seed=5, teacher_positions=12, auto_tools=False), teacher=src)
    f, r = rep["funnel"], rep["regret"]
    print("OBS 78.22 sleep from 3-unit starts: " + ", ".join(f"{k} {v}" for k, v in f.items()), "open", len(r["open"]))
    for k in ("regret_surprises", "regret_confirmed", "regret_tied", "regret_unconfirmed", "teacher_surprises", "teacher_confirmed", "decisions", "cards", "seeds"):
        assert k in f
    assert f["games"] == 8 and f["cards"] > 0 and r["source"] == "fake" and r["ranked"] > 0
    assert f["regret_surprises"] == r["surprises"] >= len(r["confirmed"]) + len(r["tied"]) + len(r["unconfirmed"]) + len(r["cut"])
    assert f["regret_confirmed"] == len(r["confirmed"]) and f["regret_tied"] == len(r["tied"]) == len(r["open"])
    assert all(o["played"] != o["source_top"] and o["played_score"] is not None for o in r["open"])
    assert all(s is None or s in starts for _, s in src.calls) and any(s is not None for _, s in src.calls), "the source was asked from the games' own starts"
    assert "teacher" not in rep, "the disagreement detector stays off unless the tools say so"
    for l in rep["lemmas"]:
        assert l["meta"]["teacher"] is True and l["meta"]["source"] == "fake"
    # an ordinary game with a source that has no `start` keyword still works
    ordinary = str(tmp_path / "o.jsonl")
    WK.play_self("chess", 4, seconds=0.01, random_plies=4, max_plies=24, seed=6, log=ordinary, run="o", threads=2, ties=True)
    rep2 = SL.sleep("chess", ordinary, settings=SL.Settings(depth=3, seed=6, teacher_positions=6, auto_tools=False), teacher=NoStart())
    assert "regret_surprises" in rep2["funnel"] and rep2["regret"]["source"] == "old"
    assert GS.head("chess") is None


def test_78_22_regret_off_and_no_source_leave_the_funnel_at_zero(store, tmp_path):
    log = str(tmp_path / "w.jsonl")
    WK.play_self("chess", 4, seconds=0.01, random_plies=4, max_plies=20, seed=7, log=log, run="z", threads=2)
    for kw in ({"teacher": None}, {"teacher": Seq(), "settings": SL.Settings(depth=3, seed=7, regret=False, auto_tools=False)}):
        kw.setdefault("settings", SL.Settings(depth=3, seed=7))
        rep = SL.sleep("chess", log, **kw)
        assert rep["funnel"]["regret_surprises"] == rep["funnel"]["regret_tied"] == 0 and "regret" not in rep
