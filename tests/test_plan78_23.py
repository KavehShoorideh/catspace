"""Plan 78.23: a rank source as the JUDGE of the cards. CardSet.revalued (Rust), UciTeacher.grade, judges.revalue, and the sleep with judge_by="source". Fast and deterministic:
fake sources only (no Stockfish)."""
import json
import shutil
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import curriculum as CU
from catspace.incubator.analyst.discovery import detectors as D
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import judges as JG
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


def centre(move: str) -> int:
    """How far the destination square is from the centre (twice the Chebyshev distance to d4/e5's corner), computed from the move name alone."""
    f, r = ord(move[2]) - ord("a"), int(move[3]) - 1
    return max(abs(2 * f - 7), abs(2 * r - 7))


class Prefers:
    """A fake source with a real preference: moves that bring the mover's piece to the centre. Scores are 50 per step; gap 100; records the calls."""
    source, training_only, gap = "prefers", True, 100
    movetime = 5

    def __init__(self):
        self.calls = []

    def rank(self, variant, moves, cap=None, start=None):
        self.calls.append((tuple(moves), start))
        legal = CE.state(variant, start, moves)[2]
        return sorted(((m, -50 * centre(m)) for m in legal), key=lambda x: (-x[1], x[0]))

    def grade(self, score):
        return score // self.gap


class Plain:
    """No grade, no gap: the default grading applies."""
    source, training_only = "plain", False

    def rank(self, variant, moves, cap=None, start=None):
        legal = sorted(CE.state(variant, start, moves)[2])
        return [(m, float(len(legal) - i) + 0.4) for i, m in enumerate(legal)]


def cards_of(items, depth=1):
    return CE.build_cards(items, GS.spec_texts("chess"), json.dumps(GS.vocabulary("chess")), depth)


# ------------------------------------------------------------------------------------------------------------------------------------ Rust

def test_78_23_revalued_replaces_values_keeps_order_cuts_missing_and_keeps_none():
    items = [("chess", None, [], "a", "g0"), ("chess", None, ["e2e4"], "b", "g1"), ("chess", None, ["d2d4"], "c", "g2")]
    cs = cards_of(items)
    a, b, c = (json.loads(cs.card(i)) for i in range(3))
    va = {m["name"]: i * 7 for i, m in enumerate(a["moves"])}
    vb = {m["name"]: 1 for m in b["moves"]}
    del vb[b["moves"][0]["name"]]                                         # a missing move
    out = cs.revalued([va, vb, None])
    assert len(out) == 3 and len(cs) == 3
    ra, rb, rc = (json.loads(out.card(i)) for i in range(3))
    assert [x["id"] for x in (ra, rb, rc)] == ["a", "b", "c"]
    assert {m["name"]: m["value"] for m in ra["moves"]} == va and ra["cut"] is False
    assert [m["name"] for m in ra["moves"]] == [m["name"] for m in a["moves"]] and ra["moves"][1]["added"] == a["moves"][1]["added"] and ra["facts"] == a["facts"]
    assert rb["cut"] is True, "a move without a value makes the card unusable"
    assert rc == c, "None keeps the card"
    assert json.loads(cs.card(0)) == a, "the original set is not changed"
    with pytest.raises(ValueError):
        cs.revalued([None])
    with pytest.raises(ValueError):
        cs.revalued([None] * 4)


def test_78_23_eval_on_a_revalued_set_reads_the_new_values():
    cs = cards_of([("chess", None, [], "a", "g0")])
    a = json.loads(cs.card(0))
    names = [m["name"] for m in a["moves"]]
    target = names[3]
    # the engine's own lemma shape is read from a real one: the seed of the card's move
    from catspace.incubator.analyst.discovery import abstraction as AB
    seed = AB.seed_from(a, next(m for m in a["moves"] if m["name"] == target), AB.Gates(), None)
    only = cs.revalued([{n: (5 if n == target else 0) for n in names}])
    every = cs.revalued([{n: 0 for n in names}])
    e1, e2 = only.eval(json.dumps(seed), None, 64), every.eval(json.dumps(seed), None, 64)
    assert e1["fires"] == e2["fires"] == 1 and e1["holds"] == 1
    assert e1["expected"] == pytest.approx(1 / len(names)) and e2["expected"] == pytest.approx(1.0), "one optimal move against all of them"


# ------------------------------------------------------------------------------------------------------------------------------------ grade

def stub():
    return object.__new__(SL.UciTeacher)


def test_78_23_uci_teacher_grade_mates_keep_their_order_and_other_scores_share_gap_buckets():
    g = stub().grade
    M = SL.MATE_SCORE
    assert g(M - 1) > g(M - 3) > g(5000), "a faster win is strictly better"
    assert g(-(M - 3)) > g(-(M - 1)), "a slower loss is strictly better"
    assert g(M - 2) == M - 2 and g(-(M - 2)) == -(M - 2)
    assert g(0) == g(99) == 0 and g(100) == g(199) == 1 and g(-1) == g(-100) == -1 and g(-101) == -2, "buckets of the gap, floor on negatives"
    assert g(-(M - 2)) < g(-5000) and g(5000) < g(M - 40)
    assert SL.UciTeacher.grade(SL.UciTeacher, 250) == 2, "callable on the class (gap is its attribute)"
    assert JG.grader(stub()) == stub().grade or JG.grader(stub()).__name__ == "grade"


def test_78_23_the_default_grade_and_describe():
    assert JG.default_grade(7) == 7 and JG.default_grade(2.6) == 3 and JG.default_grade(-2.6) == -3
    assert JG.grader(Plain()) is JG.default_grade and JG.grader(Prefers())(250) == 2
    d = JG.describe(Prefers())
    assert d == {"source": "prefers", "training_only": True, "movetime": 5, "gap": 100}
    assert JG.describe(Plain()) == {"source": "plain", "training_only": False}


# ------------------------------------------------------------------------------------------------------------------------------------ revalue

def test_78_23_revalue_asks_once_grades_reports_the_chance_share_and_shares_the_cache(capsys):
    items = [("chess", None, [], "a", "g0"), ("chess", None, ["e2e4"], "b", "g1"), ("chess", None, ["d2d4"], "c", "g2")]
    cs = cards_of(items)
    pos = {"a": (None, []), "b": (None, ["e2e4"]), "c": (None, ["d2d4"])}
    src = Prefers()
    ctx = D.Context("chess", SL.Settings(), {}, {}, positions=pos)
    out, rep = JG.revalue(cs, pos, src, "chess", [0, 1], context=ctx, say=print)
    assert len(src.calls) == 2 and rep["asked"] == 2 and rep["revalued"] == 2 and rep["cut"] == 0 and rep["source"] == "prefers"
    line = capsys.readouterr().out
    assert line.startswith("OBS sleep judge prefers:") and "optimal share" in line and str(rep["optimal_share"]) in line
    for i in (0, 1):
        card = json.loads(out.card(i))
        got = {m["name"]: m["value"] for m in card["moves"]}
        assert got == {m: s // 100 for m, s in src.rank("chess", pos[card["id"]][1])}
    assert json.loads(out.card(2)) == json.loads(cs.card(2)), "outside the indices: unchanged"
    best = [max(m["value"] for m in json.loads(out.card(i))["moves"]) for i in (0, 1)]
    shares = [sum(1 for m in json.loads(out.card(i))["moves"] if m["value"] == best[i]) / len(json.loads(out.card(i))["moves"]) for i in (0, 1)]
    assert rep["optimal_share"] == round(sum(shares) / 2, 4) and 0 < rep["optimal_share"] <= 1
    # the cache is shared: a second pass, and a detector, ask nothing more
    n = len(src.calls)
    _, rep2 = JG.revalue(cs, pos, src, "chess", [0, 1], context=ctx)
    assert len(src.calls) == n and rep2["asked"] == 0 and rep2["revalued"] == 2
    assert ctx.ranking(src, "a") is ctx.rankings[(id(src), "a")] and len(src.calls) == n


def test_78_23_revalue_cuts_what_the_source_cannot_rank_completely_and_skips_cut_cards():
    items = [("chess", None, [], "a", "g0"), ("chess", None, ["e2e4"], "b", "g1"), ("chess", None, ["d2d4"], "c", "g2")]
    cs = cards_of(items)

    class Partial(Prefers):
        def rank(self, variant, moves, cap=None, start=None):
            r = super().rank(variant, moves, cap, start)
            return r[:-1] if moves == ["e2e4"] else r
    pos = {"a": (None, []), "b": (None, ["e2e4"])}                       # "c" has no known position
    out, rep = JG.revalue(cs, pos, Partial(), "chess")
    assert (rep["revalued"], rep["cut"]) == (1, 2) and json.loads(out.card(0))["cut"] is False
    assert json.loads(out.card(1))["cut"] and json.loads(out.card(2))["cut"]
    # a plain source (floats, no grade) is graded by the default
    out2, rep2 = JG.revalue(cs, {"a": (None, [])}, Plain(), "chess", [0])
    assert all(isinstance(m["value"], int) for m in json.loads(out2.card(0))["moves"]) and rep2["revalued"] == 1
    # an already cut card is not asked
    cut = cs.revalued([{}, None, None])
    src = Prefers()
    _, rep3 = JG.revalue(cut, pos, src, "chess", [0])
    assert rep3["skipped_cut"] == 1 and rep3["asked"] == 0 and src.calls == []


# ------------------------------------------------------------------------------------------------------------------------------------ the sleep

def make_log(tmp_path):
    starts = CU.starts("chess", 3, limit=4, seed=2)
    log = str(tmp_path / "w.jsonl")
    WK.play_self("chess", 8, seconds=0.01, random_plies=0, max_plies=40, seed=5, log=log, run="c3", threads=2, starts=starts, ties=True)
    return log, starts


def test_78_23_a_sleep_judged_by_the_source_and_the_store_funnel_is_unchanged(store, tmp_path, capsys):
    log, starts = make_log(tmp_path)
    src = Prefers()
    lines = []
    st = SL.Settings(depth=2, seed=5, teacher_positions=40, auto_tools=False, judge_by="source", teacher_surprises=True)
    rep = SL.sleep("chess", log, settings=st, teacher=src, say=lines.append)
    f, j = rep["funnel"], rep["judge_report"]
    print("OBS 78.23 sleep judged by the source: " + ", ".join(f"{k} {v}" for k, v in f.items()), "| judge", {k: v for k, v in j.items() if k != "describe"})
    assert rep["judge"] == "source:prefers"
    assert any(l.startswith("OBS sleep judge prefers:") for l in lines) and any("judge source:prefers" in l for l in lines if "funnel" in l)
    for k in ("regret_surprises", "regret_confirmed", "regret_tied", "regret_unconfirmed", "teacher_surprises", "teacher_confirmed", "decisions", "cards", "seeds", "confirmed", "verified_held"):
        assert k in f
    assert j["revalued"] > 0 and j["asked"] == len(src.calls) and 0 < j["optimal_share"] <= 1 and "skipped" in j["disagreement"]
    assert "teacher" not in rep and f["teacher_surprises"] == f["teacher_confirmed"] == 0, "Disagreement is skipped when the source is the judge"
    r = rep["regret"]
    assert f["regret_confirmed"] == len(r["confirmed"]) and f["regret_tied"] == len(r["tied"]) and f["regret_surprises"] == r["surprises"] >= len(r["confirmed"]) + len(r["tied"])
    assert len(src.calls) == j["asked"], "the regret detector asked nothing the judge had not"
    for l in rep["lemmas"]:
        assert l["stats"]["judge"].startswith("source:") and l["meta"]["teacher"] is True and l["meta"]["judge_source"]["source"] == "prefers"
        GS.check_lemma(l)
    assert GS.head("chess") is None, "nothing is written"
    # a cap on the cards revalued
    capped = SL.sleep("chess", log, settings=SL.Settings(depth=2, seed=5, teacher_positions=40, auto_tools=False, judge_by="source", judge_positions=5), teacher=Prefers(), say=lambda *_: None)
    assert capped["judge_report"]["cards"] == 5 and capped["judge_report"]["revalued"] + capped["judge_report"]["cut"] + capped["judge_report"]["skipped_cut"] == 5
    assert capped["funnel"]["cards_dev"] + capped["funnel"]["cards_held"] == 5, "cards outside the cap are left out of dev and held"
    # the store judge: the same funnel as it was (no judge report, regret through the deeper card), run twice
    a = SL.sleep("chess", log, settings=SL.Settings(depth=3, seed=5, teacher_positions=12, auto_tools=False), teacher=Prefers(), say=lambda *_: None)
    b = SL.sleep("chess", log, settings=SL.Settings(depth=3, seed=5, teacher_positions=12, auto_tools=False, judge_by="store"), teacher=Prefers(), say=lambda *_: None)
    assert a["funnel"] == b["funnel"] and "judge_report" not in a and a["judge"] == b["judge"] != "source:prefers"
    assert a["regret"]["deeper_depth"] == 5
    for l in a["lemmas"]:
        assert not l["stats"]["judge"].startswith("source:") and "judge_source" not in l["meta"]


def test_78_23_judge_by_source_needs_a_source_and_a_known_mode(store, tmp_path):
    log, _ = make_log(tmp_path)
    with pytest.raises(ValueError, match="rank source"):
        SL.sleep("chess", log, settings=SL.Settings(depth=2, judge_by="source"))
    with pytest.raises(ValueError, match="judge_by"):
        SL.sleep("chess", log, settings=SL.Settings(depth=2, judge_by="nope"), teacher=Prefers())
    with pytest.raises(SystemExit):
        SL.main(["chess", "--log", log, "--judge-by", "source"])


# ------------------------------------------------------------------------------------------------------------------------------------ D: the source's continuations

CANNED = """info depth 5 multipv 1 score cp 30 pv e2e4 e7e5 g1f3
info depth 5 multipv 2 score cp -20 pv d2d4 d7d5
info depth 9 multipv 2 score mate -3 pv d2d4 e7e5 c2c3
info depth 9 multipv 1 score cp 41 pv e2e4 c7c5 g1f3 d7d6
info string ignored
info depth 9 multipv 3 score mate 2 pv a2a3
bestmove e2e4
"""


def test_78_23_the_line_parser_orders_by_rank_the_later_line_wins_and_mates_score():
    out = SL.parse_multipv_lines(CANNED)
    assert [m for m, _, _ in out] == ["e2e4", "d2d4", "a2a3"]
    assert out[0] == ("e2e4", 41, ["e2e4", "c7c5", "g1f3", "d7d6"]), "the later (deeper) line for rank 1 wins, whole"
    assert out[1] == ("d2d4", -(SL.MATE_SCORE - 3), ["d2d4", "e7e5", "c2c3"]) and out[2][1] == SL.MATE_SCORE - 2
    assert [(m, s) for m, s, _ in out] == SL.parse_multipv(CANNED), "parse_multipv is unchanged and agrees"


def test_78_23_one_engine_analysis_serves_rank_and_lines():
    t = stub()
    t._engine, t.movetime, calls = None, 1, []
    t._analyse_uncached = lambda variant, moves, cap, start: (calls.append(1), [("a", 5, ["a", "b"]), ("c", 1, ["c"])])[1]
    assert t.rank("chess", ["x"]) == [("a", 5), ("c", 1)] and t.lines("chess", ["x"]) == {"a": ["a", "b"], "c": ["c"]} and len(calls) == 1
    t.lines("chess", ["y"])
    assert len(calls) == 2, "another position is another analysis"


def test_78_23_consequence_replays_a_real_line_and_stops_at_an_illegal_move():
    start = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
    c = JG.consequence("chess", start, [], ["a7a8q", "e8d7", "a8a1"])
    print("OBS 78.23 consequence of a7a8q e8d7 a8a1 from a lone pawn on a7:", c)
    assert c == {"plies": 3, "outcome": None, "stopped": "line end", "units_before": 3, "units_end": 3, "first_change": {"ply": 0, "move": "a7a8q", "units": 3}}
    cap = JG.consequence("chess", "4k3/8/8/8/8/8/p7/R3K3 w - - 0 1", [], ["a1a2", "e8d7"])
    assert cap["first_change"] == {"ply": 0, "move": "a1a2", "units": 3} and cap["units_before"] == 4 and cap["units_end"] == 3
    quiet = JG.consequence("chess", start, [], ["e1e2"])
    assert quiet["first_change"] is None and quiet["plies"] == 1
    bad = JG.consequence("chess", start, [], ["e1e2", "e8e5", "e2e3"])
    assert bad["plies"] == 1 and bad["stopped"] == "illegal"
    mate = JG.consequence("chess", "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", [], ["f7f8", "h8h7"])
    assert mate["outcome"] == [100, 0] and mate["stopped"] == "game over" and mate["plies"] == 1
    assert JG.consequence("chess", start, [], [])["plies"] == 0


class PrefersWithLines(Prefers):
    """The fake source also offers its principal line after each move: the move, then its own best reply (the closest-to-centre legal one), then nothing more."""
    source = "prefers_lines"

    def lines(self, variant, moves, cap=None, start=None):
        out, n = {}, len(self.calls)
        ranked = self.rank(variant, moves, start=start)
        del self.calls[n:]                                                  # (the line comes from the same analysis: it is not a second ranking call)
        for m, _ in ranked:
            after = CE.state(variant, start, list(moves) + [m])
            reply = sorted(after[2], key=lambda x: (centre(x), x))[:1]
            out[m] = [m] + reply
        return out


def test_78_23_confirmed_regrets_keep_the_sources_lines_as_evidence(store, tmp_path):
    log, _ = make_log(tmp_path)
    src, lines = PrefersWithLines(), []
    rep = SL.sleep("chess", log, settings=SL.Settings(depth=2, seed=5, teacher_positions=40, auto_tools=False, judge_by="source"), teacher=src, say=lines.append)
    r = rep["regret"]
    rows = r["confirmed_rows"]
    assert rows and len(rows) == len(r["confirmed"]) == rep["funnel"]["regret_confirmed"]
    w = rows[0]
    assert set(w) == {"id", "played", "refutation", "refutation_end", "best", "best_line", "best_end"} and w["refutation"][0] == w["played"] and w["best_line"][0] == w["best"]
    assert w["refutation_end"]["plies"] >= 1 and w["best_end"]["stopped"] in ("line end", "game over")
    why = [l for l in lines if l.startswith("OBS sleep why ")]
    assert 1 <= len(why) <= 5 and " leads to " in why[0] and "; best " in why[0]
    assert len(src.calls) == rep["judge_report"]["asked"], "the lines add no ranking calls"
    for l in rep["lemmas"]:
        assert l["meta"]["why"]["refutation"][0] == l["meta"]["why"]["played"]
        GS.check_lemma(l)
    # a source without `lines` is simply skipped
    plain = SL.sleep("chess", log, settings=SL.Settings(depth=2, seed=5, teacher_positions=40, auto_tools=False, judge_by="source"), teacher=Prefers(), say=lambda *_: None)
    assert "confirmed_rows" not in plain["regret"] and all("why" not in l["meta"] for l in plain["lemmas"])
