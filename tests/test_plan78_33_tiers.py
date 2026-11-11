"""Plan 78.33: relative position first, then absolute position as the board's structure (board_structure.py), the tiers of the body literals and the tiered search of induce.py.
Fast: tiny corpora and a fake labeller; the search is stubbed where the order of the tiers is the thing under test."""
import json
import sys
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import board_structure as BS
from catspace.incubator.analyst.discovery import corpus as CP
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import induce as IND
from catspace.incubator.analyst.discovery import playout_sleep as PS
from catspace.incubator.analyst.discovery import targeted as TG
from catspace.io import paths

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plan78_32_exam_drill_induce import Fake, hashed, tiny_corpus          # noqa: E402

RULES = paths.REPO_ROOT / "rules"
KEYS = "keys"
FILES, RANKS = "abcdefgh", "12345678"
START = "8/8/4k3/8/3K4/8/5P2/8 w - - 0 1"


def keys_spec():
    return GS.spec_texts("chess", KEYS), json.dumps(GS.vocabulary("chess", KEYS))


def structure():
    return BS.from_vocabulary(GS.spec_texts("chess", KEYS)["chess"], GS.vocabulary_file("chess", KEYS))


# ------------------------------------------------------------------------------------------------------------------------------ the generated structure

def test_signatures_and_sorts_are_read_from_the_declarations():
    text = "(sort file a b c) (signature at file file) (signature own file) (at a b)"
    assert BS.signatures_of(text) == {"at": ("file", "file"), "own": ("file",)}
    assert BS.sorts_of(text) == {"file": ["a", "b", "c"]}


def test_structure_facts_of_chess_equal_an_independent_computation():
    st = structure()
    edge_file, edge_rank, goal = st.tables["edge_gap_file"], st.tables["edge_gap_rank"], st.tables["goal_gap_rank"]
    for i, f in enumerate(FILES):
        assert edge_file[(f,)] == f"d{min(i, 7 - i)}"
    for i, r in enumerate(RANKS):
        assert edge_rank[(r,)] == f"d{min(i, 7 - i)}"
        assert goal[("white", r)] == f"d{7 - i}" and goal[("black", r)] == f"d{i}"       # convention: |steps| to the side's last rank (white 8, black 1)
    assert (edge_file[("a",)], edge_file[("h",)], edge_file[("b",)], edge_file[("g",)]) == ("d0", "d0", "d1", "d1")
    assert goal[("white", "7")] == "d1" and goal[("black", "7")] == "d6"
    assert len(edge_file) == 8 and len(edge_rank) == 8 and len(goal) == 16
    assert st.edge_gap("file", "a") == "d0" and st.goal_gap("rank", ("white",), "7") == "d1"


def test_the_delta_is_additive_with_signatures_and_the_engine_reads_the_facts_in_the_movers_perspective():
    st = structure()
    d = st.delta()
    for line in ("(signature edge_gap_file file distance)", "(signature edge_gap_rank rank distance)", "(signature goal_gap_rank side rank distance)", "(goal_gap_rank white 7 d1)", "(edge_gap_file h d0)"):
        assert line in d
    assert "<=" not in d                                                          # ground facts only
    specs, vj = keys_spec()
    text = specs["chess"]
    got = {tuple(r) for r in CE.facts("chess", text, vj, START, []) if r[0] in ("goal_gap_rank", "edge_gap_file")}
    assert ("goal_gap_rank", "us", "2", "d6") in got and ("goal_gap_rank", "them", "2", "d1") in got      # white (us) promotes on rank 8
    assert ("edge_gap_file", "a", "d0") in got
    black = {tuple(r) for r in CE.facts("chess", text, vj, START.replace(" w ", " b "), []) if r[0] == "goal_gap_rank"}
    assert ("goal_gap_rank", "us", "2", "d1") in black                                # black to move: us is black, whose last rank is 1


def test_the_vocabulary_lists_the_generated_relations_as_derived_not_as_delta():
    v = GS.vocabulary("chess", KEYS)
    for r in ("edge_gap_file", "edge_gap_rank", "goal_gap_rank"):
        assert r in v["derived"] and r not in v["delta"]
    assert BS.relation_names(GS.vocabulary_file("chess", KEYS)) == ["edge_gap_file", "edge_gap_rank", "goal_gap_rank"]


def test_a_vocabulary_without_axes_is_unchanged():
    for name in ("relational", None):
        vf = GS.vocabulary_file("chess", name)
        assert "axes" not in vf and BS.relation_names(vf) == [] and BS.structure_delta(GS.spec_texts("chess", name)["chess"], vf) == ""
        v = GS.vocabulary("chess", name)
        assert v == {"derived": vf["derived"], "delta": vf["delta"], "units": vf["units"]}
        assert not any("edge_gap" in r or "goal_gap" in r for r in v["derived"] + v["delta"])
    vf = GS.vocabulary_file("chess", "relational")
    base = (RULES / vf["spec"]).read_text()
    by_hand = {}
    for variant, delta in vf["variants"].items():
        t = base if delta is None else CE.compose(base, (RULES / delta).read_text())
        for c in vf["concepts"]:
            t = CE.compose(t, (RULES / c).read_text())
        by_hand[variant] = t
    assert GS.spec_texts("chess", "relational") == by_hand
    assert "edge_gap" not in GS.spec_texts("chess", "relational")["chess"]


def test_the_keys_vocabulary_is_the_same_description_plus_the_generated_delta():
    vf = GS.vocabulary_file("chess", KEYS)
    base = (RULES / vf["spec"]).read_text()
    t = base
    for c in vf["concepts"]:
        t = CE.compose(t, (RULES / c).read_text())
    assert GS.spec_texts("chess", KEYS)["chess"] == CE.compose(t, BS.structure_delta(t, vf))


def test_a_bad_axis_is_refused():
    t = GS.spec_texts("chess", "relational")["chess"]
    with pytest.raises(ValueError):
        BS.Structure(t, [{"sort": "file", "succ": "file_succ", "goal": "pawn_start"}]).delta()      # a goal relation must end in the axis sort
    with pytest.raises(ValueError):
        BS.Structure(t, [{"sort": "file", "succ": "file_succ"}], distance_sort="nosuch")


def test_the_hand_rules_still_pass_the_engines_check_under_the_changed_vocabulary_file():
    rules = json.loads((RULES / "concepts/chess_keys.proposed.json").read_text())["rules"]
    assert len(rules) >= 3
    for r in rules:
        GS.check_lemma(r)


# ------------------------------------------------------------------------------------------------------------------------------ the tiers

def hand_card():
    spec, vj = keys_spec()
    reader = TG.Reader("chess", spec, vj, PS.Verdicts(Fake(hashed), "chess"))
    card = json.loads(reader.cards([(START, [], "p", "p")]).card(0))
    return card, {"e": "?h0", "6": "?h1", "d3": "?h2"}


def test_tiers_of_the_candidate_literals_on_a_hand_position():
    st = structure()
    card, names = hand_card()
    assert any(f[0] == "edge_gap_file" for f in card["facts"])                    # the engine reads the structure at the position
    c = IND.tiered_candidates(card["facts"], names, st)
    coord, sigs = set(st.coordinate_sorts), st.signatures

    def coordinate_constants(lit):
        sig = sigs.get(lit["relation"]) or ()
        return [a for a, s in zip(lit["args"], sig) if s in coord and not a.startswith("?")]
    assert c["relative"] and c["structure"] and c["named"]
    for lit, _d in c["relative"]:
        assert not coordinate_constants(lit)
    for lit, _d, _n in c["structure"]:
        assert lit["relation"] in st.relations() and not coordinate_constants(lit)
        assert lit["args"][-1] in st.sorts["distance"]                           # the distance constant is kept
    for lit, _d in c["named"]:
        assert coordinate_constants(lit)
    assert any(l["relation"] == "apart" and l["args"][4].startswith("?") for l, _d in c["relative"])      # a relation between units, coordinates all variables
    assert any(l["relation"] == "apart" and l["args"][-1] == "d2" for l, _d in c["relative"])             # ... with a distance constant kept
    for lit, _d, needs in c["structure"]:                                        # attached to the coordinate variable, which occurs among the other literals or the head
        (v,) = [a for a in lit["args"] if a.startswith("?")][-1:]
        assert needs == {v}
        assert v in names.values() or any(v in l["args"] for l, _d2 in c["relative"] + c["named"])
    passes = IND.tier_passes(card["facts"], names, st)
    assert [p[0] for p in passes] == ["relative", "structure", "named"]
    assert [len(p[1]) for p in passes] == [len(c["relative"]), len(c["relative"]) + len(c["structure"]), len(c["relative"]) + len(c["structure"]) + len(c["named"])]
    assert [p[0] for p in IND.tier_passes(card["facts"], names, st, ["relative", "structure"])] == ["relative", "structure"]
    assert [p[0] for p in IND.tier_passes(card["facts"], names, None)] == ["untiered"]


def test_without_a_structure_the_candidates_are_the_old_ones():
    card, names = hand_card()
    (tier, cands, needs), = IND.tier_passes(card["facts"], names, None)
    assert tier == "untiered" and needs == {} and cands == TG.body_candidates(card["facts"], names, set())


def test_a_structure_literal_joins_a_body_only_where_its_variable_occurs(monkeypatch):
    c = tiny_corpus(60)
    spec, vj = keys_spec()
    c.cards(spec, vj)
    card, names = hand_card()
    head = {"removed": [("cell", ["?h0", "?h1", "us", "king"])], "added": []}
    rel_lit = IND.AB.lit("flight", ["us", "?k1", "?k2"])
    on_head = IND.AB.lit("edge_gap_file", ["?h0", "d0"])
    elsewhere = IND.AB.lit("edge_gap_file", ["?k1", "d0"])
    nowhere = IND.AB.lit("edge_gap_file", ["?nowhere", "d0"])
    cands = [(rel_lit, "r"), (on_head, "h"), (elsewhere, "e"), (nowhere, "n")]
    needs = {json.dumps(on_head, sort_keys=True): {"?h0"}, json.dumps(elsewhere, sort_keys=True): {"?k1"}, json.dumps(nowhere, sort_keys=True): {"?nowhere"}}
    seen = []

    def spy(lemmas, corpus, **kw):
        seen.extend(l["body"] for l in lemmas)
        return [{"truncated": False, "right": 1, "wrong": 1, "trivial": 0, "right_at": [0], "wrong_at": [], "binds": 1, "names_a_move": 1} for _ in lemmas]
    monkeypatch.setattr(CP, "drill_many", spy)
    IND.search_body(c, card["facts"], names, head, max_literals=2, beam=8, min_right=1, finalists=1, max_bindings=8, cands=cands, needs=needs)
    flat = [json.dumps(l, sort_keys=True) for b in seen for l in b]
    assert json.dumps(nowhere, sort_keys=True) not in flat
    assert [b for b in seen if b == [on_head]]                                  # on a head variable: allowed alone
    assert not [b for b in seen if b == [elsewhere]]                            # on a variable no literal has yet: not alone
    assert [b for b in seen if b == [rel_lit, elsewhere]]                       # but after the literal that has it


# ------------------------------------------------------------------------------------------------------------------------------ the order of the search

def induce_on_a_problem(monkeypatch, stub, **kw):
    """induce() on a position of a tiny corpus with search_body replaced by `stub(cands...)` and the admission drill by a fixed good tally."""
    c = tiny_corpus(300, seed=4)
    spec, vj = keys_spec()
    c.cards(spec, vj)
    reader = TG.Reader("chess", spec, vj, PS.Verdicts(Fake(hashed), "chess"))
    good = {"binds": 3, "names_a_move": 3, "right": 99, "wrong": 0, "trivial": 0, "truncated": False}
    monkeypatch.setattr(IND, "search_body", stub)
    monkeypatch.setattr(IND.CP, "drill", lambda lem, corpus, **k: dict(good))
    for i in c.part("held")[:80]:
        r = c.rows[i]
        rd = reader.read({"start": r["start"], "moves": r["moves"]})
        if rd is None or not rd.good or not rd.bad:
            continue
        out = IND.induce({"start": r["start"], "moves": r["moves"]}, c, reader=reader, say=lambda *_: None, structure=structure(), max_heads=1, **kw)
        if out["heads_tried"] or out["stage"] != "position":
            return out
    pytest.skip("no problem position with a head in the tiny corpus")


def test_the_search_widens_only_when_the_narrower_tier_fails(monkeypatch):
    calls = []

    def stub(corpus, facts, names, head, *, cands=None, needs=None, **kw):
        n = len(cands)
        has_structure = any(l["relation"].startswith(("edge_gap", "goal_gap")) for l, _d in cands)
        has_named = any(not a.startswith("?") and a not in ("us", "them") and a in "abcdefgh12345678" for l, _d in cands for a in l["args"])
        calls.append(("named" if has_named else "structure" if has_structure else "relative", n))
        tier = calls[-1][0]
        if tier == "structure":
            return {"found": [([cands[0][0]], {"wrong": 0, "right": 99})], "best": None, "levels": []}
        return {"found": [], "best": None, "levels": []}
    out = induce_on_a_problem(monkeypatch, stub)
    assert [t for t, _n in calls[:2]] == ["relative", "structure"] and len(calls) == 2           # relative failed, structure gave a rule, named never ran
    assert out["status"] == "found" and out["tier"] == "structure"
    assert out["tiers"]["relative"]["stage"] == "dev" and out["candidates"][0]["tier"] == "structure"
    assert calls[0][1] < calls[1][1]


def test_the_narrowest_tier_that_gives_a_rule_stops_the_search_and_the_setting_stops_a_run_early(monkeypatch):
    calls = []

    def first_wins(corpus, facts, names, head, *, cands=None, needs=None, **kw):
        calls.append(len(cands))
        return {"found": [([cands[0][0]], {"wrong": 0, "right": 99})], "best": None, "levels": []}
    out = induce_on_a_problem(monkeypatch, first_wins)
    assert out["tier"] == "relative" and len(calls) == 1

    calls.clear()

    def none(corpus, facts, names, head, *, cands=None, needs=None, **kw):
        calls.append(len(cands))
        return {"found": [], "best": None, "levels": []}
    out = induce_on_a_problem(monkeypatch, none, tiers=["relative", "structure"])
    assert out["status"] == "open" and set(out["tiers"]) == {"relative", "structure"} and out["tier"] is None
    assert len(calls) == 2 * out["heads_tried"]


def test_a_near_miss_is_a_rule_that_passed_development_and_failed_the_held_out_part(monkeypatch):
    def one(corpus, facts, names, head, *, cands=None, needs=None, **kw):
        return {"found": [([cands[0][0]], {"wrong": 0, "right": 99})], "best": None, "levels": []}
    c = tiny_corpus(300, seed=4)
    bad_held = {"binds": 3, "names_a_move": 3, "right": 9, "wrong": 2, "trivial": 0, "truncated": False}
    good = {"binds": 3, "names_a_move": 3, "right": 99, "wrong": 0, "trivial": 0, "truncated": False}
    monkeypatch.setattr(IND, "search_body", one)
    monkeypatch.setattr(IND.CP, "drill", lambda lem, corpus, part="dev", **k: dict(good if part == "dev" else bad_held))
    spec, vj = keys_spec()
    c.cards(spec, vj)
    reader = TG.Reader("chess", spec, vj, PS.Verdicts(Fake(hashed), "chess"))
    for i in c.part("held")[:80]:
        r = c.rows[i]
        rd = reader.read({"start": r["start"], "moves": r["moves"]})
        if rd is None or not rd.good or not rd.bad:
            continue
        out = IND.induce({"start": r["start"], "moves": r["moves"]}, c, reader=reader, say=lambda *_: None, structure=structure(), max_heads=1)
        if out["heads_tried"]:
            assert out["status"] == "open" and out["stage"] == "held"
            assert all(pg["near"] and pg["stage"] == "held" for pg in out["tiers"].values()) and set(out["tiers"]) == {"relative", "structure", "named"}
            return
    pytest.skip("no problem position with a head in the tiny corpus")


def test_each_admitted_lemma_records_its_tier_and_the_settings_name_the_tiers():
    assert IND.SETTINGS["tiers"] == ["relative", "structure", "named"] and IND.TIERS == ("relative", "structure", "named")
    m = IND.provenance({"start": "x", "moves": ["a"], "played": "a"}, "chess", Fake(), KEYS, "structure")
    assert m["tier"] == "structure" and m["plan"] == "78.32"
    assert "tier" not in IND.provenance({"start": "x", "moves": ["a"]}, "chess", Fake(), KEYS)
    lem = {"name": "t", "body": [IND.AB.lit("edge_gap_file", ["?h0", "d0"])], "response": {"removed": [], "added": [IND.AB.lit("cell", ["?h0", "?h1", "us", "king"])]}, "stats": {"status": "INDUCED"},
           "meta": m}
    GS.check_lemma(lem)                                                          # a lemma with a structure literal and meta.tier is a well formed lemma
