"""Plan 78.24 C: the audit of application (apply_audit) and the loop for one size on a sandbox store (initialize). Fast and deterministic: a fake rank source with a real preference."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import apply_audit as AU
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import initialize as IN
from catspace.incubator.analyst.discovery import sleep as SL
from catspace.incubator.analyst.discovery import wake as WK
from catspace.io import paths

ROOT = Path(__file__).resolve().parents[1]
FEN = "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"
TAKE = {"name": "take-the-attacker", "body": [{"relation": "cell", "args": ["?x", "?y", "us", "?k"], "neg": False}, {"relation": "attack", "args": ["them", "?a", "?b", "?x", "?y"], "neg": False},
                                               {"relation": "cell", "args": ["?a", "?b", "them", "?j"], "neg": False}],
        "response": {"removed": [{"relation": "cell", "args": ["?a", "?b", "them", "?j"], "neg": False}], "added": [{"relation": "cell", "args": ["?a", "?b", "us", "?m"], "neg": False}]},
        "stats": {}, "meta": {}}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(GS, "LEARNED_ROOT", tmp_path / "learned")
    monkeypatch.setattr(GS, "SNAPSHOT_ROOT", tmp_path / "snapshots")
    shutil.copytree(ROOT / "rules" / "learned" / "chess", tmp_path / "learned" / "chess")
    for p in (tmp_path / "learned" / "chess" / "lemmas").glob("*.json"):
        p.unlink()
    return tmp_path


def centre(move: str) -> int:
    f, r = ord(move[2]) - ord("a"), int(move[3]) - 1
    return max(abs(2 * f - 7), abs(2 * r - 7))


class Prefers:
    """A fake source with a real preference: destinations near the centre; gap 100."""
    source, training_only, gap = "prefers", True, 100

    def rank(self, variant, moves, cap=None, start=None):
        legal = CE.state(variant, start, moves)[2]
        return sorted(((m, -50 * centre(m)) for m in legal), key=lambda x: (-x[1], x[0]))

    def grade(self, score):
        return score // self.gap


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode() + p.read_bytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------------ render

def test_78_24_render_a_lemma_with_and_without_why():
    plain = AU.render(TAKE)
    assert plain.startswith("IF cell(?x, ?y, us, ?k) AND attack(them, ?a, ?b, ?x, ?y) AND cell(?a, ?b, them, ?j) THEN a best move removes [cell(?a, ?b, them, ?j)] adds [cell(?a, ?b, us, ?m)]")
    assert "why" not in plain and "\n" not in plain
    end = {"plies": 3, "outcome": None, "stopped": "line end", "units_before": 3, "units_end": 2, "first_change": {"ply": 1, "move": "e8d7", "units": 2}}
    full = {**TAKE, "body": TAKE["body"] + [{"relation": "cell", "args": ["?q", "?r", "them", "?t"], "neg": True}],
            "stats": {"dev": {"fires": 5, "holds": 5, "precision": 1.0, "excess": 0.5}, "held": {"fires": 4, "holds": 3, "precision": 0.75, "excess": 0.2}, "judge": "source:prefers"},
            "meta": {"why": {"played": "e4e5", "refutation": ["e4e5", "d5d4"], "refutation_end": end, "best": "e4d5", "best_line": ["e4d5", "e8d7"], "best_end": end}}}
    text = AU.render(full)
    assert "AND not cell(?q, ?r, them, ?t)" in text and "dev: 5 fires, 5 held, precision 1.0, excess 0.5" in text and "held: 4 fires, 3 held" in text
    assert "judge source:prefers" in text and "why: played e4e5; the refutation e4e5 d5d4" in text and "best e4d5: e4d5 e8d7" in text


def test_78_24_mined_renders_every_lemma_in_the_store(store):
    assert AU.mined("chess") == []
    GS.write_lemma("chess", TAKE)
    out = AU.mined("chess")
    assert len(out) == 1 and out[0].startswith("take-the-attacker: IF ")


# ------------------------------------------------------------------------------------------------------------------------------------ audit

def tiny_wake(tmp_path):
    GS.write_lemma("chess", TAKE)
    sid = GS.publish("chess")
    log = str(tmp_path / "w.jsonl")
    games = WK.play_self("chess", 2, seconds=0.1, random_plies=0, max_plies=2, store=str(GS.path_of("chess", sid)), specs=GS.spec_texts("chess"), log=log, run="t", starts=[FEN])
    return log, games


def test_78_24_audit_recovers_the_proposed_moves_with_the_engines_matcher(store, tmp_path):
    log, games = tiny_wake(tmp_path)
    lines = []
    rep = AU.audit("chess", log, source=Prefers(), say=lines.append)
    assert rep["overall"]["with_lemma"] > 0 and rep["overall"]["games_touched"] == 2 and rep["overall"]["mismatch"] == 0
    for d in rep["decisions"]:
        legal = CE.state("chess", d["start"], d["moves"])[2]
        assert d["lemmas"], d
        for item in d["lemmas"]:
            assert item["name"] == "take-the-attacker" and item["bindings"] >= 1
            assert item["proposed"] and set(item["proposed"]) <= set(legal) and len(item["proposed"]) == item["engine_count"]
            assert item["followed"] == (d["played"] in item["proposed"])
            assert item["proposal_optimal"] == bool(set(item["proposed"]) & set(d["source_best"]))
        assert d["played"] in legal and d["played_optimal"] == (d["played"] in d["source_best"])
    s = rep["lemmas"]["take-the-attacker"]
    assert s["fired"] == len(rep["decisions"]) and s["games"] == 2 and s["proposed"] == sum(len(i["proposed"]) for d in rep["decisions"] for i in d["lemmas"])
    assert s["followed"] == sum(i["followed"] for d in rep["decisions"] for i in d["lemmas"])
    assert s["scored"] == s["fired"] and s["followed_n"] + s["not_followed_n"] == s["scored"] and 0 <= s["proposal_optimal"] <= s["scored"]
    assert any(l.startswith("OBS audit chess:") for l in lines) and any(l.startswith("OBS audit lemma take-the-attacker: fired") and "claim held" in l for l in lines)
    assert 1 <= sum(l.startswith("OBS audit example") for l in lines) <= 3
    # the recomputed names are the engine's: with the lemma ablated no move is proposed, and `e4d5` (the capture) is the one this lemma proposes
    assert rep["decisions"][0]["lemmas"][0]["proposed"] == ["e4d5"]


def test_78_24_audit_without_a_source_and_with_no_lemma(store, tmp_path):
    log, _ = tiny_wake(tmp_path)
    rep = AU.audit("chess", log, say=lambda *_: None)
    assert rep["overall"]["with_lemma"] > 0 and rep["decisions"][0]["source_best"] is None and rep["lemmas"]["take-the-attacker"]["scored"] == 0
    # a wake with a snapshot that holds no lemma
    for p in (store / "learned" / "chess" / "lemmas").glob("*.json"):
        p.unlink()
    sid = GS.publish("chess")
    log2 = str(tmp_path / "w2.jsonl")
    WK.play_self("chess", 1, seconds=0.1, random_plies=0, max_plies=2, store=str(GS.path_of("chess", sid)), specs=GS.spec_texts("chess"), log=log2, run="u", starts=[FEN])
    lines = []
    rep2 = AU.audit("chess", log2, say=lines.append)
    assert rep2["overall"]["with_lemma"] == 0 and any("no lemma in the store" in l for l in lines)


# ------------------------------------------------------------------------------------------------------------------------------------ the loop

def test_78_24_settings_are_typed_from_the_fields_and_unknown_names_are_errors():
    d = IN.settings_pairs(["depth=3", "dev_share=0.25", "regret=false", "judge_positions=10", "judge=material"])
    assert d == {"depth": 3, "dev_share": 0.25, "regret": False, "judge_positions": 10, "judge": "material"}
    assert IN.settings_from(d).depth == 3
    with pytest.raises(ValueError):
        IN.settings_pairs(["nope=1"])
    with pytest.raises(ValueError):
        IN.settings_from({"nope": 1})


def test_78_24_the_sandbox_guard(tmp_path):
    with pytest.raises(AssertionError):
        IN.inside(ROOT / "rules" / "learned", tmp_path)
    assert IN.inside(tmp_path / "a" / "b", tmp_path) == (tmp_path / "a" / "b").resolve()


def test_78_24_run_size_never_touches_the_real_store_and_restores_the_roots_even_on_error(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "experiment", lambda *p: str(tmp_path / ("exp_" + "_".join(p))))
    real = ROOT / "rules" / "learned"
    before = tree_hash(real)
    roots = GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT
    snaps = ROOT / "data" / "derived" / "snapshots"
    snaps_before = tree_hash(snaps) if snaps.is_dir() else None
    lines = []
    rep = IN.run_size("chess", 3, source=Prefers(), store_root=tmp_path / "sb", limit=2, games_per_start=1, seconds=0.05, max_plies=6, tag="t", say=lines.append)
    assert (GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT) == roots and tree_hash(real) == before
    assert (tree_hash(snaps) if snaps.is_dir() else None) == snaps_before, "the real snapshots are untouched"
    assert (tmp_path / "sb" / "learned" / "chess" / "vocabulary.json").is_file() and (tmp_path / "sb" / "snapshots" / "chess" / "HEAD").is_file()
    text = "\n".join(lines)
    for key in ("OBS init sandbox", "OBS init starts", "OBS init wake a", "OBS init sleep", "OBS init wake b", "OBS init compare", "OBS init report"):
        assert key in text, key
    assert "OBS audit chess:" in text and ("no lemma in the store" in text or "OBS mined" in text)
    assert {"wake_a", "wake_b", "sleep", "audit", "mined", "compare"} <= set(rep) and rep["sleep"]["funnel"]["games"] == 2 and rep["wake_b"]["games"] == 2
    assert json.load(open(rep["report"]))["size"] == 3
    # a step that raises: the roots are restored all the same
    def boom(*a, **k):
        raise RuntimeError("sleep failed")
    monkeypatch.setattr(SL, "sleep", boom)
    with pytest.raises(RuntimeError):
        IN.run_size("chess", 3, source=Prefers(), store_root=tmp_path / "sb2", limit=1, games_per_start=1, seconds=0.05, max_plies=4, tag="e", say=lambda *_: None)
    assert (GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT) == roots and tree_hash(real) == before


def test_cached_source_answers_once_and_survives_a_reload(tmp_path):
    """judges.Cached: the wrapped source is asked once per position; a second object on the same file answers without asking; the source's own attributes show through."""
    from catspace.incubator.analyst.discovery import judges as JG

    class Src:
        source, training_only, gap = "counting", True, 7
        def __init__(self):
            self.calls = 0
        def rank(self, variant, moves, cap=None, start=None):
            self.calls += 1
            return [("a", 3 + self.calls), ("b", 1)]
        def lines(self, variant, moves, cap=None, start=None):
            return {"a": ["a", "x"], "b": ["b"]}
        def grade(self, score):
            return score // self.gap

    f = tmp_path / "cache.json"
    s = Src()
    c = JG.Cached(s, f)
    first = c.rank("g", ["m1"], start="S")
    assert c.rank("g", ["m1"], start="S") == first and s.calls == 1 and (c.asked, c.hits) == (2, 1)
    assert c.lines("g", ["m1"], start="S") == {"a": ["a", "x"], "b": ["b"]} and s.calls == 1
    assert c.rank("g", ["m2"], start="S") != first and s.calls == 2                       # (another position is another question)
    assert (c.source, c.gap, c.grade(15)) == ("counting", 7, 2) and JG.describe(c)["source"] == "counting"
    c.close()
    s2 = Src()
    c2 = JG.Cached(s2, f)
    assert c2.rank("g", ["m1"], start="S") == first and s2.calls == 0

    class NoLines:
        source = "plain"
        def rank(self, variant, moves, cap=None, start=None):
            return [("a", 1)]
    assert not hasattr(JG.Cached(NoLines(), tmp_path / "c2.json"), "lines")
