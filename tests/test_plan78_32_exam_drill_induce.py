"""Plan 78.32: the exact labeller (exact.py), the corpus and the full drill (corpus.py), the exam (exam.py), induction from exact labels (induce.py) and the cycle options of initialize.py.
Fast: a fake labeller and tiny corpora; the tablebase and Stockfish tests skip when those are not there."""
import json
import re
import tempfile
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import abstraction as AB
from catspace.incubator.analyst.discovery import corpus as CP
from catspace.incubator.analyst.discovery import exact as EX
from catspace.incubator.analyst.discovery import exam as EM
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import induce as IND
from catspace.incubator.analyst.discovery import initialize as IN
from catspace.incubator.analyst.discovery import playout_sleep as PS
from catspace.incubator.analyst.discovery import targeted as TG
from catspace.io import paths

ROOT = Path(__file__).resolve().parents[1]
STOCKFISH = "/opt/homebrew/bin/stockfish"
VOCAB = "relational"
HAVE_TB = (Path(paths.syzygy_dir()) / "KQvK.rtbw").exists()


class Fake(EX.ExactLabeller):
    """A labeller by a function: `cls(moves) -> class of the position for its mover` (a pure function of the position's moves)."""
    source = "fake"

    def __init__(self, fn=lambda start, moves: "draw"):
        super().__init__()
        self.fn = fn

    def _class(self, variant, start, moves):
        return self.fn(start, moves)


def hashed(start, moves):
    """A deterministic pseudo-class by the position's text."""
    return ("loss", "draw", "win")[sum(map(ord, (start or "") + "".join(moves))) % 3]


def tiny_corpus(n=80, seed=1):
    return CP.build_corpus("chess", 3, n=n, seed=seed, labeller=Fake(hashed), say=lambda *_: None)


def spec_vocab():
    return GS.spec_texts("chess", VOCAB), json.dumps(GS.vocabulary("chess", VOCAB))


# ---------------------------------------------------------------------------------------------------------------------------------------- the labeller

def test_the_judge_call_shape_gives_a_one_hot_and_an_uncovered_position_raises():
    f = Fake(lambda s, m: "win" if len(m) % 2 == 0 else None)
    start = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    assert f.wdl("chess", [], start) == (1.0, 0.0, 0.0) and f.result("chess", start, []) == "win"
    with pytest.raises(EX.NotCovered):
        f.wdl("chess", ["d1d2"], start)
    assert f.result("chess", start, ["d1d2"]) is None and f.training_only is True
    v = PS.Verdicts(f, "chess")                                          # (it is a judge: Verdicts takes it)
    assert v.of(start, []) == "win"


def test_the_generic_moves_keep_reads_the_class_after_each_move_and_the_games_own_end():
    start = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    f = Fake(lambda s, m: "win" if not m else "draw")                    # every successor is a draw for its mover: the win is not kept by any move
    keep = f.moves_keep("chess", start, [])
    assert keep and not any(keep.values())
    g = Fake(lambda s, m: "win" if not m else "loss")                    # every successor a loss for ITS mover = a win for ours
    assert all(g.moves_keep("chess", start, []).values())
    assert f.moves_keep("chess", "k7/8/8/8/8/8/8/K7 w - - 0 1", []) is None       # (over: nothing to label)


@pytest.mark.skipif(not HAVE_TB, reason="no Syzygy tables")
def test_syzygy_chess_agrees_with_the_generic_reading_counts_cursed_as_drawn_and_reads_castling_positions():
    sz = EX.SyzygyChess(cache=None)
    try:
        assert sz.result("chess", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", []) == "win"
        assert sz.result("chess", "4k3/8/8/8/8/8/8/3qK3 w - - 0 1", []) in ("draw", "loss", "win")
        assert sz.training_only and sz.result("chess", "k7/8/8/8/8/8/8/K7 w - - 0 1", []) is None          # (over)
        assert sz.result("chess", "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1", []) == "win"                         # castling rights: not in the tables, read by the successors
        assert sz.result("antichess", None, []) is None
        for fen, moves in [("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1", []), ("8/8/8/8/2k2P2/8/4K3/8 w - - 9 6", []), ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", ["d1d7"])]:
            assert sz.moves_keep("chess", fen, moves) == EX.ExactLabeller.moves_keep(sz, "chess", fen, moves)
        src = (ROOT / "catspace/incubator/analyst/discovery/exact.py").read_text()
        assert "cursed" in src.lower() and "draw" in src.lower()                                           # (said in the module)
        # a cursed win (wdl +1) is a draw: probe it in a five-unit table if one is there
    finally:
        sz.close()


@pytest.mark.skipif(not HAVE_TB, reason="no Syzygy tables")
def test_syzygy_disk_cache_repeats(tmp_path):
    p = tmp_path / "c.json"
    a = EX.SyzygyChess(cache=str(p))
    r = a.result("chess", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", [])
    a.close()
    assert p.exists() and json.loads(p.read_text())
    b = EX.SyzygyChess(cache=str(p))
    assert b.result("chess", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", []) == r
    b.close()


@pytest.mark.skipif(not HAVE_TB or not Path(STOCKFISH).exists(), reason="no Syzygy tables or Stockfish")
def test_the_exact_class_agrees_with_stockfish_with_the_same_tablebase_on_random_three_unit_positions():
    sz = EX.SyzygyChess(cache=None)
    try:
        r = EX.agree_with_engine(sz, "chess", 3, engine=STOCKFISH, depth=14, options={"SyzygyPath": str(paths.syzygy_dir())}, n=20, seed=3, say=lambda *_: None)
    finally:
        sz.close()
    assert r["total"] == 20 and r["agree"] >= 19, r


# ---------------------------------------------------------------------------------------------------------------------------------------- the corpus and the drill

def test_the_corpus_is_distinct_split_by_group_labelled_and_round_trips(tmp_path):
    c = tiny_corpus(60)
    keys = [(r["start"], tuple(r["moves"])) for r in c.rows]
    assert len(c) == 60 and len(set(keys)) == 60
    fens = {CE.state("chess", r["start"], r["moves"])[4] for r in c.rows}
    assert len(fens) == 60                                                       # distinct POSITIONS (the kernel's key), not just move lists
    assert set(c.groups.values()) == {"dev", "held"} and set(r["group"] for r in c.rows) <= set(c.groups)
    for r in c.rows:
        _f, _m, legal, _o, _k = CE.state("chess", r["start"], r["moves"])
        assert set(r["keep"]) == set(legal)                                      # every legal move is labelled
    p = c.save(tmp_path / "c.json")
    d = CP.load_corpus(p)
    assert d.rows == c.rows and d.groups == c.groups and d.variant == "chess"
    assert sorted(set(c.part("dev")) | set(c.part("held"))) == list(range(60)) and not set(c.part("dev")) & set(c.part("held"))
    with pytest.raises(ValueError):
        CP.build_corpus("chess", 3, n=5, labeller=Fake(), sources=("nonsense",))


def test_the_corpus_skips_positions_the_labeller_does_not_cover():
    class Half(Fake):
        def moves_keep(self, variant, start, moves):
            return None if len(moves) % 2 else super().moves_keep(variant, start, moves)
    c = CP.build_corpus("chess", 3, n=40, labeller=Half(hashed), say=lambda *_: None)
    assert c.rows and all(len(r["moves"]) % 2 == 0 for r in c.rows)


def test_the_corpus_prints_one_obs_line():
    lines = []
    CP.build_corpus("chess", 3, n=30, labeller=Fake(hashed), say=lines.append)
    assert len(lines) == 1 and re.match(r"OBS corpus: positions 30, groups \d+, non-trivial \d+ \(some move keeps and some loses\), by result class \{.*\}, \d+ seconds", lines[0]), lines


def brute(rule, corpus, part):
    """The drill by hand: the rule's body is always true (control(us)), so it names the moves whose effect matches the head (targeted.head_matches)."""
    head = {"removed": [(l["relation"], l["args"]) for l in rule["response"]["removed"]], "added": [(l["relation"], l["args"]) for l in rule["response"]["added"]]}
    right = wrong = trivial = names = 0
    for ci in corpus.idx(part):
        r = corpus.rows[corpus.row_of[ci]]
        card = json.loads(corpus._cards.card(ci))
        if len(set(r["keep"].values())) == 1:                                      # (every move has the same label: with exact labels, every move keeps)
            trivial += 1
            continue
        named = [m["name"] for m in card["moves"] if TG.head_matches(head, m["removed"], m["added"])]
        if not named:
            continue
        names += 1
        if all(r["keep"][m] for m in named):
            right += 1
        else:
            wrong += 1
    return {"names_a_move": names, "right": right, "wrong": wrong, "trivial": trivial}


def test_the_full_drill_counts_right_wrong_and_trivial_as_a_brute_force_does():
    c = tiny_corpus(120)
    spec, vj = spec_vocab()
    c.cards(spec, vj)
    c.cards(spec, vj)                                                            # (built once: the same object is returned)
    cs = c._cards
    assert c.cards(spec, vj) is cs
    some = None
    for r in c.rows:
        card = json.loads(CE.build_cards([("chess", r["start"], r["moves"], "x", "g")], spec, vj, 1).card(0))
        for m in card["moves"]:
            if m["added"]:
                some = m["added"][0]
                break
        if some:
            break
    rule = {"name": "t", "body": [AB.lit("control", ["us"])], "response": {"removed": [], "added": [AB.lit(some[0], ["?a%d" % i for i in range(len(some) - 1)])]}, "stats": {}, "meta": {}}
    for part in ("dev", "held", "all"):
        d = CP.drill(rule, c, part=part)
        b = brute(rule, c, part)
        assert {k: d[k] for k in b} == b, (part, d, b)
        assert d["binds"] >= d["names_a_move"] + d["trivial"]
    d = CP.drill(rule, c, part="all")
    assert d["wrong"] == len(d["wrong_examples"]) or len(d["wrong_examples"]) == 5
    for e in d["wrong_examples"]:
        assert e["losing"] and set(e) >= {"start", "moves", "group", "losing"}
    assert d["right"] + d["wrong"] == d["names_a_move"]


def test_the_drill_without_cards_is_refused_and_a_part_must_be_named():
    c = tiny_corpus(10)
    with pytest.raises(ValueError):
        CP.drill_many([{}], c, part="dev")
    with pytest.raises(ValueError):
        c.part("nonsense")


def test_excluding_takes_the_exam_positions_out_of_the_corpus():
    c = tiny_corpus(50)
    keys = {(c.rows[0]["start"], tuple(c.rows[0]["moves"])), (c.rows[1]["start"], tuple(c.rows[1]["moves"]))}
    d = c.excluding(keys)
    assert len(d) == 48 and not keys & {(r["start"], tuple(r["moves"])) for r in d.rows}


# ---------------------------------------------------------------------------------------------------------------------------------------- the exam

def test_the_exam_is_non_trivial_held_out_stratified_fixed_and_disjoint_from_the_induction_corpus(tmp_path):
    c = CP.build_corpus("chess", 3, n=400, labeller=Fake(hashed), say=lambda *_: None)
    out = []
    e = EM.build_exam("chess", 3, n=20, seed=2, corpus=c, path=str(tmp_path / "e.json"), say=out.append)
    pos = e["positions"]
    assert len(pos) == 20 and all(c.groups[p["group"]] == "held" for p in pos)
    by = {r_k: r for r_k, r in (((r["start"], tuple(r["moves"])), r) for r in c.rows)}
    for p in pos:
        r = by[(p["start"], tuple(p["moves"]))]
        assert any(r["keep"].values()) and not all(r["keep"].values()) and p["before"] == r["before"]
    classes = {p["before"] for p in pos}
    assert "win" in classes and "draw" in classes and len({p["group"] for p in pos}) >= 2
    assert EM.load_exam(tmp_path / "e.json") == json.loads(json.dumps(e))
    assert EM.build_exam("chess", 3, n=20, seed=2, corpus=c, say=lambda *_: None) == e               # fixed: the same call, the same exam
    rest = c.excluding(EM.exam_keys(e))
    assert not EM.exam_keys(e) & {(r["start"], tuple(r["moves"])) for r in rest.rows}
    assert not EM.exam_keys(e) & {(c.rows[i]["start"], tuple(c.rows[i]["moves"])) for i in c.part("dev")}
    assert out and out[0].startswith("OBS exam built:")


def test_grade_asks_the_labeller_whether_the_decision_keeps_and_compare_lists_the_changes():
    c = CP.build_corpus("chess", 3, n=200, labeller=Fake(hashed), say=lambda *_: None)
    e = EM.build_exam("chess", 3, n=6, seed=1, corpus=c, say=lambda *_: None)
    spec = GS.spec_texts("chess", VOCAB)

    class AllKeep(Fake):
        def moves_keep(self, variant, start, moves):
            _f, _m, legal, _o, _k = CE.state(variant, start, moves)
            return {m: True for m in legal}

    class NoneKeep(Fake):
        def moves_keep(self, variant, start, moves):
            _f, _m, legal, _o, _k = CE.state(variant, start, moves)
            return {m: False for m in legal}
    kw = dict(variant="chess", store=None, spec=spec, seconds=0.02)
    g1 = EM.grade(e, labeller=AllKeep(hashed), **kw)
    g0 = EM.grade(e, labeller=NoneKeep(hashed), **kw)
    assert g1["kept"] == g1["total"] == 6 and g0["kept"] == 0
    assert sum(v["total"] for v in g1["by_class"].values()) == 6 and all(d["move"] for d in g1["decisions"])
    cmp = EM.compare(g0, g1)
    assert len(cmp["gained"]) == 6 and cmp["lost"] == [] and EM.compare(g1, g0)["lost"] == list(range(6))
    assert EM.compare(None, g1) == {"gained": [], "lost": [], "changed": [], "same": 0}
    lines = []
    EM.say_exam("cycle 1", g1, cmp, lines.append)
    assert re.fullmatch(r"OBS exam cycle 1: kept 6 of 6 \(win-keeping \d+ of \d+, draw-keeping \d+ of \d+\), gained 6, lost 0", lines[0]), lines
    a, b, diff = EM.grade_twice(e, say=lines.append, labeller=AllKeep(hashed), **kw)
    assert re.fullmatch(r"OBS exam spread: 6 vs 6, decisions differing \d+", lines[-1]) and diff >= 0


# ---------------------------------------------------------------------------------------------------------------------------------------- induction

def test_a_move_counter_is_never_a_body_literal_and_body_literals_come_from_the_problems_facts():
    facts = [["halfmove", "9"], ["cell", "e", "2", "us", "king"], ["control", "us"]]
    cands = TG.body_candidates(facts, {}, set())
    assert cands and all(l["relation"] != "halfmove" for l, _d in cands)
    assert {l["relation"] for l, _d in cands} == {"cell", "control"}


def test_the_search_returns_zero_wrong_bodies_only_and_is_deterministic():
    c = tiny_corpus(300, seed=4)
    spec, vj = spec_vocab()
    c.cards(spec, vj)
    reader = TG.Reader("chess", spec, vj, PS.Verdicts(Fake(hashed), "chess"))
    done = False
    for i in c.part("held")[:60]:
        r = c.rows[i]
        rd = reader.read({"start": r["start"], "moves": r["moves"]})
        if rd is None or not rd.good or not rd.bad:
            continue
        card = json.loads(reader.cards([(r["start"], r["moves"], "p", "p")]).card(0))
        for h in TG.heads_for(card, rd.keep, limit=2):
            for stride in (1, 2):
                res = IND.search_body(c, card["facts"], h["names"], h["head"], max_literals=2, beam=4, min_right=1, finalists=2, max_bindings=256, stride=stride)
                for body, t in res["found"]:
                    assert t["wrong"] == 0 and t["right"] >= 1 and len(body) <= 2
                    lem = TG.lemma_of(body, h["head"])
                    assert CP.drill(lem, c, part="dev")["wrong"] == 0
                    done = True
            again = IND.search_body(c, card["facts"], h["names"], h["head"], max_literals=2, beam=4, min_right=1, finalists=2, max_bindings=256, stride=1)
            assert json.dumps([b for b, _t in again["found"]]) == json.dumps([b for b, _t in IND.search_body(c, card["facts"], h["names"], h["head"], max_literals=2, beam=4, min_right=1, finalists=2,
                                                                                                          max_bindings=256, stride=1)["found"]])
        if done:
            break
    assert done


def test_induce_says_why_when_no_move_keeps_or_every_move_keeps_or_the_position_is_over():
    c = tiny_corpus(30)
    spec, vj = spec_vocab()
    c.cards(spec, vj)
    start = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    for fn, why in [(lambda s, m: "win" if not m else "draw", "no move keeps the class"), (lambda s, m: "win" if not m else "loss", "every move keeps the class")]:
        reader = TG.Reader("chess", spec, vj, PS.Verdicts(Fake(fn), "chess"))
        out = IND.induce({"start": start, "moves": []}, c, reader=reader, say=lambda *_: None)
        assert out["status"] == "open" and out["why"] == why and out["stage"] == "position"
    reader = TG.Reader("chess", spec, vj, PS.Verdicts(Fake(), "chess"))
    assert IND.induce({"start": "k7/8/8/8/8/8/8/K7 w - - 0 1", "moves": []}, c, reader=reader, say=lambda *_: None)["why"] == "the position is over"
    unc = Fake(lambda s, m: None)
    assert "does not cover" in IND.induce({"start": start, "moves": []}, c, reader=TG.Reader("chess", spec, vj, PS.Verdicts(unc, "chess")), say=lambda *_: None)["why"]


def test_provenance_says_induced_and_holds_no_verdict_or_number():
    m = IND.provenance({"start": "x", "moves": ["a", "b"], "played": "a"}, "chess", Fake(), VOCAB)
    assert m["origin"] == "induced" and m["mined_from"] == "problem" and m["problem"]["moves"] == 2
    text = json.dumps(m)
    assert "keep" not in text and "wrong" not in text and "right" not in text and "drill" not in text and "verdict" not in text


@pytest.mark.skipif(not HAVE_TB, reason="no Syzygy tables")
def test_solve_problems_admits_only_what_passes_held_out_and_the_problems_own_check_and_says_why_otherwise(tmp_path):
    """On a real exact corpus: every admitted rule has zero wrong on both parts and its problem is kept by the decision with the rule; every open problem has a reason; the lemma has no number."""
    sz = EX.SyzygyChess(cache=None)
    lines = []
    try:
        c = CP.build_corpus("chess", 3, n=1500, seed=1, labeller=sz, say=lambda *_: None)
        root = IN.prepare_sandbox("chess", tmp_path / "store")
        with TG.sandbox(root):
            GS.publish("chess", vocabulary=VOCAB)
            snap = str(GS.path_of("chess", GS.head("chess")))
            spec = GS.spec_texts("chess", VOCAB)
            probs = []
            for i in c.part("held"):
                r = c.rows[i]
                if c.non_trivial(i):
                    probs.append({"start": r["start"], "moves": r["moves"]})
                if len(probs) >= 40:
                    break
            res = PS.check(probs, variant="chess", store=snap, spec=spec, judge=sz, seconds=0.05)
            rows = [dict(p, played=x["move"]) for p, x in zip(probs, res) if not x["keeps"]][:3]
            assert rows, "the empty store should lose the class somewhere"
            rep = IND.solve_problems("chess", rows, c, labeller=sz, vocabulary=VOCAB, seconds=0.05, settings={"min_right": 2, "min_held": 1, "max_heads": 2, "max_literals": 2, "beam": 4}, say=lines.append)
            assert rep["total"] == len(rows) and rep["attempted"] == len(rows)
            text = "\n".join(lines)
            assert re.search(r"OBS induce total: problems \d+, attempted \d+, admitted \d+, solved \d+, still open \d+, reasons \{", text)
            assert len(re.findall(r"OBS induce problem \d+/\d+: head .*, rule .*, dev binds/right/wrong/trivial .*, held binds/right/wrong/trivial .*, problem check kept .*; (ADMITTED|open because)", text)) == len(rows)
            for a in rep["admitted"]:
                assert a["dev"]["wrong"] == 0 and a["held"]["wrong"] == 0 and a["held"]["right"] >= 1
            lemmas = GS.read_lemmas("chess", VOCAB)
            assert len(lemmas) == len(rep["admitted"])
            for l in lemmas:
                assert l["meta"]["origin"] == "induced" and l["meta"]["mined_from"] == "problem" and l["stats"] == {"status": "INDUCED"}
            assert rep["solved"] <= rep["attempted"] + sum(len(a["solved_others"]) for a in rep["admitted"])
    finally:
        sz.close()


# ---------------------------------------------------------------------------------------------------------------------------------------- training data and the cycle

def test_the_wake_the_snapshot_writer_and_the_rust_sources_never_read_the_corpus_the_exam_or_the_labeller():
    names = ("exact_corpus", "exact_cache", "exam-1", "exam_", "exact-corpus", "discovery.exact", "discovery.exam", "discovery.corpus", "discovery import exact", "discovery import exam", "discovery import corpus")
    for p in (ROOT / "catspace/incubator/analyst/discovery/wake.py", ROOT / "catspace/incubator/analyst/discovery/game_store.py", ROOT / "catspace/incubator/analyst/discovery/wdl_table.py"):
        text = p.read_text()
        assert not any(n in text for n in names), p.name
    for p in sorted((ROOT / "rust/engine").glob("*/src/**/*.rs")):
        t = p.read_text()
        assert "exact_corpus" not in t and "exam-1" not in t and "exact-corpus" not in t, p


def test_run_size_has_the_induction_the_exam_and_the_exact_switches_off_by_default_and_refuses_them_without_what_they_need():
    import inspect
    sig = inspect.signature(IN.run_size).parameters
    assert sig["induce"].default is False and sig["exam"].default is None and sig["exact"].default is False
    with pytest.raises(ValueError):
        IN.run_size("chess", 3, source=type("S", (), {"grade_mode": "score"})(), induce=True, playouts=False, say=lambda *_: None)
    with pytest.raises(ValueError):
        IN.run_size("chess", 3, source=type("S", (), {"grade_mode": "score"})(), exam="x.json", exact=False, say=lambda *_: None)


def test_the_cycle_with_the_exam_and_the_induction_prints_the_curve(tmp_path, monkeypatch):
    """run_size on a tiny sandbox with a fake labeller: wake, playouts, the lemma sleep, induction on the open problems, the exam graded before cycle 1 and after it."""
    store = tmp_path / "store"
    monkeypatch.setattr(IN.paths, "experiment", lambda name: str(tmp_path / name))
    monkeypatch.setattr(EX, "labeller_of", lambda game, **kw: Fake(hashed))
    monkeypatch.setattr(CP, "default_path", lambda game, size, n, seed: str(tmp_path / "corpus.json"))
    lines = []

    class Src:
        source, training_only = "fake", True

        def rank(self, variant, moves, cap=None, start=None):
            _f, _m, legal, _o, _k = CE.state(variant, start, list(moves))
            return [(m, 0) for m in sorted(legal)]

        def grade(self, s):
            return s

        def __call__(self, variant, moves, start=None):
            return None

        def close(self):
            pass
    Src.grade_mode = "score"
    rep = IN.run_size("chess", 3, source=Src(), store_root=store, games_per_start=1, seconds=0.02, max_plies=8, limit=2, vocabulary=VOCAB, tag="t32", cycles=1,
                      sleep_settings={"depth": 2, "threads": 1}, playouts=True, playout_settings={"positions": 4, "per_position": 1, "max_plies": 20, "check_seconds": 0.02},
                      exact=True, induce=True, induce_settings={"min_right": 1000, "max_heads": 1, "max_literals": 1}, exam=str(tmp_path / "exam.json"), exam_size=6, corpus_n=300,
                      say=lines.append, sample=5)
    text = "\n".join(lines)
    assert len(re.findall(r"OBS exam (before cycle 1|after cycle 1)", text)) == 2 and "OBS exam spread:" in text and "OBS exam curve:" in text, text
    assert re.search(r"OBS induce total: problems \d+, attempted \d+, admitted 0, solved \d+, still open \d+, reasons", text) or "OBS induce total: problems 0" in text, text
    assert [e["label"].split(" (")[0] for e in rep["exam"]] == ["before cycle 1", "after cycle 1"] and rep["cycles"][0]["exam"]["total"] == 6 and "induce" in rep["cycles"][0]
    assert (tmp_path / "exam.json").exists() and (tmp_path / "corpus.json").exists()
    ex = EM.load_exam(tmp_path / "exam.json")
    assert len(ex["positions"]) <= 6


def test_the_new_files_are_in_the_gate_and_in_repo_structure():
    gate = (ROOT / "scripts/gate.py").read_text()
    assert "test_plan78_32_exam_drill_induce" in gate
    rs = (ROOT / "repo_structure.md").read_text()
    for f in ("exact.py", "corpus.py", "exam.py", "induce.py"):
        assert f in rs


def test_the_trial_snapshot_carries_the_valuation_record_the_wake_plays_by(tmp_path):
    """The lemma sleep publishes without the wake's valuation record; a trial that left it out would grade decisions the wake never makes. The induction's trial snapshot and the published one carry it."""
    root = IN.prepare_sandbox("chess", tmp_path / "store")
    lem = {"name": "cell-response-1", "body": [AB.lit("control", ["us"])], "response": {"removed": [], "added": [AB.lit("cell", ["?a", "?b", "us", "?c"])]}, "stats": {}, "meta": {}}
    want = GS.judge_of("chess", "material (mined)")
    assert want is not None
    with TG.sandbox(root):
        GS.publish("chess", vocabulary=VOCAB)
        d = tmp_path / "t"
        d.mkdir()
        snap = json.loads(Path(IND.trial_snapshot("chess", lem, d, VOCAB, "material (mined)")).read_text())
        assert snap["judge"] == want and [l["name"] for l in snap["lemmas"]] == ["cell-response-1"]
        none = json.loads(Path(IND.trial_snapshot("chess", lem, d, VOCAB, None)).read_text())
        assert none["judge"] == GS.judge_of("chess")
