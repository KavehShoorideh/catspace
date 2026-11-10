"""PLAN 78.24 A: A LEMMA LANGUAGE THAT CAN SAY SOMETHING RELATIVE. (1) named lemma vocabularies (rules/learned/<game>/vocabulary.<name>.json: concept deltas composed onto the
description, lemmas stamped with and filtered by the vocabulary they were written in); (2) a move's EFFECT read on derived relations (the Rust helper cs_memory::op::Effects, tested
in rust/engine/decision/tests/lemmas.rs; here through the Python entry); (3) the climb may drop a response literal, and the selectivity gate still stops a response from becoming
"any move"; (4) the sleep's gate overrides and the UCI teacher's mate bucket."""
import json
import shutil
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import abstraction as AB
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import judges as JG
from catspace.incubator.analyst.discovery import sleep as SL

import test_plan78_14 as T14

ROOT = Path(__file__).resolve().parents[1]
ROOK_TAKES = "4k3/8/8/8/8/8/p7/R3K3 w - - 0 1"          # the rook takes a pawn that nothing defends


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(GS, "LEARNED_ROOT", tmp_path / "learned")
    monkeypatch.setattr(GS, "SNAPSHOT_ROOT", tmp_path / "snapshots")
    shutil.copytree(ROOT / "rules" / "learned" / "chess", tmp_path / "learned" / "chess")
    d = tmp_path / "learned" / "chess"
    for p in (d / "lemmas").glob("*.json"):
        p.unlink()
    for p in d.glob("vocabulary.*.json"):             # (a vocabulary another plan filed is not this test's business)
        p.unlink()
    (d / "vocabulary.rel.json").write_text(json.dumps({"spec": "chess.gdl", "variants": {"chess": None}, "derived": ["attack", "defended", "hanging"],
                                                       "delta": ["cell", "hanging"], "units": None, "concepts": ["concepts/chess.gdl"]}))
    return tmp_path


def lemma(name, vocab=None, rel="cell", derived=False):
    lit = lambda r, a: {"relation": r, "args": a, "neg": False}      # noqa: E731
    resp = {"removed": [lit("hanging", ["them", "?a", "?b"])], "added": []} if derived else {"removed": [lit(rel, ["?x", "?y", "us", "pawn"])], "added": [lit(rel, ["?u", "?w", "us", "pawn"])]}
    body = [lit("hanging", ["them", "?a", "?b"])] if derived else [lit(rel, ["?x", "?y", "us", "pawn"])]
    return {"name": name, "body": body, "response": resp, "stats": {}, "meta": ({"vocabulary": vocab} if vocab else {})}


# ------------------------------------------------------------------------------------------------------------------------------------ named vocabularies

def test_78_24_the_default_vocabulary_is_byte_for_byte_what_it_was(store):
    GS.write_lemma("chess", lemma("pawn-moves"))
    old = {"format": 1, "game": "chess", "parent": None, "vocabulary": {"derived": ["attack"], "delta": ["cell"], "units": None}, "judge": None, "teacher_used": False,
           "lemmas": GS.read_lemmas("chess")}
    expect = json.dumps(old, sort_keys=True, separators=(",", ":")).encode()
    assert GS.build("chess", judge=None) == GS.build("chess", judge=None, vocabulary=None) == expect
    assert CE.content_id(GS.build("chess", judge=None)) == CE.content_id(expect)
    assert GS.spec_texts("chess") == GS.spec_texts("chess", None)


def test_78_24_a_named_vocabulary_composes_its_concepts_and_reads_its_own_file(store):
    plain, named = GS.spec_texts("chess"), GS.spec_texts("chess", "rel")
    assert "hanging" not in plain["chess"].split("(signature")[0] and "(hanging " not in plain["chess"] and "(<= (hanging " in named["chess"]
    expect = CE.compose(plain["chess"], (ROOT / "rules" / "concepts" / "chess.gdl").read_text())
    assert named["chess"] == expect
    assert GS.vocabulary("chess", "rel") == {"derived": ["attack", "defended", "hanging"], "delta": ["cell", "hanging"], "units": None}
    assert GS.vocabulary("chess") == {"derived": ["attack"], "delta": ["cell"], "units": None}
    with pytest.raises(FileNotFoundError):
        GS.vocabulary("chess", "nope")
    # one path or a list of them: the same composition
    p = store / "learned" / "chess" / "vocabulary.one.json"
    v = json.loads((store / "learned" / "chess" / "vocabulary.rel.json").read_text())
    p.write_text(json.dumps({**v, "concepts": "concepts/chess.gdl"}))
    assert GS.spec_texts("chess", "one") == named


def test_78_24_a_snapshot_holds_only_the_lemmas_of_its_vocabulary(store):
    GS.write_lemma("chess", lemma("pawn-moves"))
    GS.write_lemma("chess", lemma("hanging-goes", vocab="rel", derived=True))
    assert [l["name"] for l in GS.read_lemmas("chess")] == ["pawn-moves"]                  # the default call: the default vocabulary's lemmas
    assert [l["name"] for l in GS.read_lemmas("chess", "rel")] == ["hanging-goes"]
    assert sorted(l["name"] for l in GS.read_lemmas("chess", GS.ANY)) == ["hanging-goes", "pawn-moves"]
    d = GS.publish("chess", judge=None)
    assert CE.snapshot_info(str(GS.path_of("chess", d)))["lemmas"] == ["pawn-moves"]
    r = GS.publish("chess", judge=None, vocabulary="rel")
    info = CE.snapshot_info(str(GS.path_of("chess", r)))
    assert info["lemmas"] == ["hanging-goes"] and r != d
    assert json.loads(GS.path_of("chess", r).read_bytes())["vocabulary"]["delta"] == ["cell", "hanging"]


# ------------------------------------------------------------------------------------------------------------------------------------ a move's effect on derived relations

def card_of_start(vocab, specs):
    cs = CE.build_cards([("chess", ROOK_TAKES, [], "p", "g")], specs, json.dumps(vocab), depth=2)
    return json.loads(cs.card(0))


def test_78_24_a_derived_delta_relation_shows_in_the_cards_moves(store):
    card = card_of_start(GS.vocabulary("chess", "rel"), GS.spec_texts("chess", "rel"))
    take = next(m for m in card["moves"] if m["name"] == "a1a2")
    assert ["hanging", "them", "a", "2"] in take["removed"], take
    assert not any(f[0] == "hanging" for f in take["added"])
    quiet = next(m for m in card["moves"] if m["name"] == "e1d1")
    assert not any(f[0] == "hanging" for f in quiet["removed"])
    # the base-only delta gives the cards of old: cell facts only, same effect on them
    base = card_of_start(GS.vocabulary("chess"), GS.spec_texts("chess"))
    btake = next(m for m in base["moves"] if m["name"] == "a1a2")
    assert {f[0] for f in btake["removed"] + btake["added"]} == {"cell"}
    assert [f for f in take["removed"] if f[0] == "cell"] == btake["removed"] and [f for f in take["added"] if f[0] == "cell"] == btake["added"]


def test_78_24_a_lemma_over_a_derived_response_is_read_by_the_cards_as_by_the_engine(store):
    cs = CE.build_cards([("chess", ROOK_TAKES, [], "p", "g")], GS.spec_texts("chess", "rel"), json.dumps(GS.vocabulary("chess", "rel")), depth=2)
    ev = cs.eval(json.dumps(lemma("hanging-goes", vocab="rel", derived=True)), None, 64)
    assert ev["fires"] == 1 and ev["holds"] == 1 and ev["refuted"] == 0
    assert ev["selectivity"] < 0.5


# ------------------------------------------------------------------------------------------------------------------------------------ the response may be generalised

def motif_cards():
    fens = T14.motif((3, 4)) + T14.decoys()
    items = [("chess", f, [], f"p{i}", f"g{i}") for i, f in enumerate(fens)]
    return CE.build_cards(items, GS.spec_texts("chess", "rel"), json.dumps(GS.vocabulary("chess", "rel")), depth=3)


def lit(r, a):
    return {"relation": r, "args": a, "neg": False}


def capture_lemma():
    return {"name": "x", "body": [lit("cell", ["?x", "?y", "us", "pawn"]), lit("attack", ["them", "?a", "?b", "?x", "?y"]), lit("cell", ["?a", "?b", "them", "pawn"])],
            "response": {"removed": [lit("cell", ["?a", "?b", "them", "pawn"]), lit("hanging", ["them", "?a", "?b"])], "added": [lit("cell", ["?a", "?b", "us", "pawn"])]}, "stats": {}, "meta": {}}


def test_78_24_size_counts_the_response_and_a_dropped_response_literal_is_a_smaller_lemma(store):
    l = capture_lemma()
    assert AB.size(l) == (3 + 3, 1)
    d = AB.drop_response(l, "removed", 1)
    assert AB.size(d) == (5, 1) and [x["relation"] for x in d["response"]["removed"]] == ["cell"] and d["response"]["added"] == l["response"]["added"]
    assert AB.canonical(d) != AB.canonical(l) and AB.name_of(d) != AB.name_of(l)
    onlyadd = {**l, "response": {"removed": [], "added": l["response"]["added"]}}
    assert AB.name_of(onlyadd).startswith("cell-response-")
    assert AB.constants(l) == ["pawn"] and AB.variables(l) == {"?x", "?y", "?a", "?b"}


def test_78_24_the_climb_drops_response_literals_and_never_the_last_one(store):
    cs = motif_cards()
    g = AB.Gates(min_fires=3, min_groups=3)
    c = AB.generalize(capture_lemma(), cs, list(range(len(cs))), g)
    assert c is not None
    assert any(s.startswith("drop response") for s in c.steps), c.steps
    r = c.lemma["response"]
    assert len(r["removed"]) + len(r["added"]) >= 1
    assert AB.size(c.lemma) < AB.size(capture_lemma())
    one = {**capture_lemma(), "response": {"removed": [lit("cell", ["?a", "?b", "them", "pawn"])], "added": []}}
    c1 = AB.generalize(one, cs, list(range(len(cs))), g)
    assert c1 is not None and not any(s.startswith("drop response") for s in c1.steps) and len(c1.lemma["response"]["removed"]) == 1


def test_78_24_a_response_dropped_to_what_every_move_matches_is_refused(store):
    cs = motif_cards()
    g = AB.Gates(min_fires=3, min_groups=3)
    idx = list(range(len(cs)))
    seed = {**capture_lemma(), "response": {"removed": [lit("cell", ["?a", "?b", "them", "pawn"])], "added": [lit("cell", ["?u", "?w", "us", "?k"])]}}
    ev = AB.evaluate(cs, seed, idx, g)
    assert g.admissible(ev), ev
    anymove = AB.drop_response(seed, "removed", 0)             # "some unit of ours arrives somewhere": every move says that
    ea = AB.evaluate(cs, anymove, idx, g)
    assert ea["selectivity"] > g.max_selectivity and not g.admissible(ea)
    c = AB.generalize(seed, cs, idx, g)
    assert c is not None and not any("drop response removed" in s for s in c.steps)
    assert c.lemma["response"]["removed"], "the literal that made the claim selective stayed"


# ------------------------------------------------------------------------------------------------------------------------------------ small knobs

def test_78_24_gate_overrides_are_typed_by_the_field_and_the_held_gates_are_separate():
    g = AB.Gates().override(["min_fires=2", "min_lower=0.5", "max_selectivity=0.4", "max_body=12"])
    assert (g.min_fires, g.min_lower, g.max_selectivity, g.max_body) == (2, 0.5, 0.4, 12) and isinstance(g.min_fires, int) and isinstance(g.min_lower, float)
    assert AB.Gates().override([]) == AB.Gates() and AB.Gates().override(None) == AB.Gates()
    for bad in ("nope=1", "min_fires", "min_fires=x"):
        with pytest.raises(ValueError):
            AB.Gates().override([bad])


def test_78_24_the_cli_puts_the_gates_in_force_on_both_sides(monkeypatch):
    seen = {}

    def fake(game, log, **kw):
        seen["s"] = kw["settings"]
        return {}
    monkeypatch.setattr(SL, "sleep", fake)
    monkeypatch.setattr(SL.json, "dump", lambda *a, **k: None)
    SL.main(["chess", "--log", "x.jsonl", "--gate", "min_fires=2", "--held-gate", "min_groups=5", "--vocabulary", "rel"])
    s = seen["s"]
    assert (s.gates.min_fires, s.held.min_fires, s.gates.min_groups, s.held.min_groups, s.vocabulary) == (2, 2, 2, 5, "rel")
    assert SL.Settings().vocabulary is None and SL.Settings().gates == AB.Gates()


def test_78_24_the_mate_bucket_ties_close_mates_and_keeps_the_order():
    M = SL.MATE_SCORE
    t1, t4 = SL.UciTeacher.__new__(SL.UciTeacher), SL.UciTeacher.__new__(SL.UciTeacher)
    t1.gap = t4.gap = 100
    t1.mate_bucket, t4.mate_bucket = 1, 4
    assert SL.UciTeacher.mate_bucket == 1                                                         # the default is the exact reading
    wins = [M - n for n in (1, 2, 3, 4, 5, 9)]
    assert [t1.grade(w) for w in wins] == wins                                                    # 1: today's behaviour
    g4 = [t4.grade(w) for w in wins]
    assert g4[0] == g4[1] == g4[2] and g4[3] == g4[4] and g4[0] > g4[3] > g4[5]                   # distances 1..3 floor to 0, 4..5 to 4, 9 to 8
    assert all(a >= b for a, b in zip(g4, g4[1:]))                                                # a faster win is never below a slower one
    losses = [-(M - n) for n in (1, 2, 3, 4, 5, 9)]
    l4 = [t4.grade(x) for x in losses]
    assert all(a <= b for a, b in zip(l4, l4[1:]))                                                # a slower loss is never below a faster one
    assert min(g4) > t4.grade(5000) and max(l4) < t4.grade(-5000) and t4.grade(5000) == 50       # any mate beats every non-mate for the mover, and loses to none
    assert JG.describe(t4)["mate_bucket"] == 4 and JG.describe(t1)["mate_bucket"] == 1
