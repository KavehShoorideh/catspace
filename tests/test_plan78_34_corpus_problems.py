"""Plan 78.34: problems from the corpus, a corpus that grows with the games and withdraws a rule, heads in tiers (relative before named), and the cycle options of initialize.py.
Fast: a fake labeller, tiny corpora, the wake's decision replaced by a fake where a real wake would be slow; one small run_size at the end."""
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import abstraction as AB
from catspace.incubator.analyst.discovery import board_structure as BS
from catspace.incubator.analyst.discovery import corpus as CP
from catspace.incubator.analyst.discovery import corpus_problems as CPB
from catspace.incubator.analyst.discovery import exact as EX
from catspace.incubator.analyst.discovery import exam as EM
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import induce as IND
from catspace.incubator.analyst.discovery import initialize as IN
from catspace.incubator.analyst.discovery import playout_sleep as PS
from catspace.incubator.analyst.discovery import targeted as TG

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plan78_32_exam_drill_induce import Fake, hashed, tiny_corpus          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VOCAB = "relational"
KEYS = "keys"
START = "8/8/4k3/8/3K4/8/5P2/8 w - - 0 1"
QUIET = lambda *_: None  # noqa: E731


def mute(*_):
    pass


class DBad(Fake):
    """Every move of a position after the first is labelled `loses` when it starts from the a, b or c file: a rule that names every move is right on the start positions and wrong on any position of a game."""

    def moves_keep(self, variant, start, moves):
        _f, mover, legal, outcome, _k = CE.state(variant, start, list(moves))
        if mover is None or outcome is not None or not legal:
            return None
        return {m: not (moves and m.startswith(("a", "b", "c"))) for m in legal}


def starts_corpus(n=400):
    """A corpus of the start positions only (every position of a game after its first move is new to it)."""
    return CP.build_corpus("chess", 3, n=n, seed=1, labeller=DBad(hashed), sources=("starts",), say=QUIET)


def game_of(start, plies=6):
    moves = []
    for _ in range(plies):
        _f, mover, legal, outcome, _k = CE.state("chess", start, moves)
        if mover is None or outcome is not None or not legal:
            break
        moves.append(sorted(legal)[0])
    return moves


def write_log(path, games, rule="rule-a"):
    """A wake log of games [{start, moves}]: a `fired` fact at every decision from ply 1 on."""
    lines = []
    for g, row in enumerate(games):
        lines.append({"kind": "game", "run": "r", "game": g, "start": row["start"], "moves": row["moves"], "outcome": 0, "unfinished": False})
        for ply in range(len(row["moves"])):
            facts = [{"relation": "fired", "args": [rule, "1", "3"]}] if ply >= 1 else []
            lines.append({"kind": "position", "label": f"r:{g}:{ply}:0", "move": row["moves"][ply], "steps": [], "facts": facts})
    Path(path).write_text("\n".join(json.dumps(l) for l in lines) + "\n")


ALL_MOVES_RULE = {"name": "all-moves", "body": [AB.lit("control", ["us"])], "response": {"removed": [], "added": [AB.lit("cell", ["?a", "?b", "?c", "?d"])]}, "stats": {}, "meta": {"vocabulary": VOCAB}}
NEVER_RULE = {"name": "never", "body": [AB.lit("control", ["them"])], "response": {"removed": [], "added": [AB.lit("cell", ["?a", "?b", "?c", "?d"])]}, "stats": {}, "meta": {"vocabulary": VOCAB}}


# ------------------------------------------------------------------------------------------------------------------------------ 1: problems from the corpus

def test_the_sample_is_development_only_non_trivial_spread_over_groups_deterministic_and_different_each_cycle():
    c = tiny_corpus(600, seed=3)
    a = CPB.sample_dev(c, 30, seed=1, cycle=1)
    assert a and len(a) == len(set(a)) <= 30
    assert all(c.part_of(i) == "dev" and c.non_trivial(i) for i in a)                     # never the held-out part, never a trivial position
    assert a == CPB.sample_dev(c, 30, seed=1, cycle=1)                                     # deterministic for a seed and a cycle
    assert a != CPB.sample_dev(c, 30, seed=1, cycle=2) and a != CPB.sample_dev(c, 30, seed=2, cycle=1)
    dev_groups = {c.rows[i]["group"] for i in c.part("dev") if c.non_trivial(i)}
    assert len({c.rows[i]["group"] for i in a}) >= min(len(dev_groups), 30) // 2           # spread over the groups, not drawn from one
    per = {}
    for i in a:
        per[c.rows[i]["group"]] = per.get(c.rows[i]["group"], 0) + 1
    full = CPB.sample_dev(c, 10 ** 6, seed=1, cycle=1)
    if len(dev_groups) <= 30:                                                                # when every group can have one, every group has one
        assert set(per) == dev_groups or len(a) < len(dev_groups)
    assert CPB.sample_dev(c, 0, seed=1, cycle=1) == []
    assert len(CPB.sample_dev(c, 10 ** 6, seed=1, cycle=1)) == sum(1 for i in c.part("dev") if c.non_trivial(i))      # (the cap is a cap)


def test_corpus_problems_are_the_decisions_that_lose_the_class_and_carry_their_source(monkeypatch):
    c = tiny_corpus(400, seed=3)
    asked = []

    def fake_check(problems, **kw):
        out = []
        for p in problems:
            r = next(x for x in c.rows if (x["start"], x["moves"]) == (p["start"], p["moves"]))
            asked.append(r)
            lose = sorted(m for m, k in r["keep"].items() if not k)
            keep = sorted(m for m, k in r["keep"].items() if k)
            move = lose[0] if len(asked) % 2 else keep[0]                                     # every other decision is wrong
            out.append({"move": move, "keeps": True, "before": "draw", "after": "draw", "proved": False, "steps": [], "error": None})
        return out
    monkeypatch.setattr(CPB.PS, "check", fake_check)
    lines = []
    old = [{"start": "x", "moves": ["a1a2"], "played": "a1a2", "first_seen": 1, "history": []}]
    r = CPB.corpus_problems(c, old, cycle=2, seed=1, n=20, variant="chess", store=None, spec={}, judge=None, seconds=0.01, say=lines.append)
    assert r["asked"] == 20 and r["wrong"] == 10 and r["new"] == 10 and r["known"] == 0 and len(r["rows"]) == 11
    assert all(c.part_of(c.rows.index(x)) == "dev" and c.non_trivial(c.rows.index(x)) for x in asked)
    for row in r["new_rows"]:
        assert row["source"] == "corpus" and row["first_seen"] == 2 and row["history"][-1]["phase"] == "before" and row["history"][-1]["status"] == "open"
        src = next(x for x in c.rows if (x["start"], tuple(x["moves"])) == PS.key_of(row))
        assert src["keep"][row["played"]] is False                                            # the chosen move loses the exact class
    assert CPB.source_of(old[0]) == "playout" and CPB.source_of(r["new_rows"][0]) == "corpus"
    assert re.fullmatch(r"OBS corpus problems cycle 2: asked 20, wrong 10, new problems 10 \(already known 0\)", lines[0]), lines
    asked.clear()                                                                              # the same sample again: every wrong one is now known
    again = CPB.corpus_problems(c, r["rows"], cycle=2, seed=1, n=20, variant="chess", store=None, spec={}, judge=None, seconds=0.01, say=lines.append)
    assert again["new"] == 0 and again["known"] == 10 and len(again["rows"]) == 11
    counts = CPB.source_counts(r["rows"])
    assert counts["playout"]["total"] == 1 and counts["corpus"]["total"] == 10 and counts["corpus"]["open"] == 10 and counts["withdrawn"]["total"] == 0
    out = []
    CPB.say_sources(r["rows"], 2, out.append)
    assert out == ["OBS problems by source cycle 2: playout total 1 open 0, corpus total 10 open 10, withdrawn total 0 open 0"]


def test_a_problem_row_keeps_a_source_only_when_it_has_one():
    rows, new = PS.merge_problems([], [{"start": "s", "moves": ["a"], "played": "a"}, {"start": "s", "moves": ["b"], "played": "b", "source": "corpus"}], 1)
    assert "source" not in new[0] and new[1]["source"] == "corpus"


# ------------------------------------------------------------------------------------------------------------------------------ 2: the corpus grows, a rule is withdrawn

def test_fired_positions_are_read_from_the_fired_facts_of_the_log(tmp_path):
    s = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    m = game_of(s, 4)
    write_log(tmp_path / "w.jsonl", [{"start": s, "moves": m}])
    got = CPB.fired_positions(str(tmp_path / "w.jsonl"))
    assert got == [{"start": s, "moves": m[:k]} for k in (1, 2, 3)]                           # (ply 0 has no firing)


def test_the_growth_adds_only_new_exactly_labelled_positions_keeps_the_split_by_group_and_never_adds_an_exam_position(tmp_path):
    c = starts_corpus()
    n0, groups0 = len(c), dict(c.groups)
    s0, s1 = c.rows[0]["start"], c.rows[1]["start"]
    g0 = game_of(s0, 5)
    other = "7k/8/8/8/8/8/8/KQ6 w - - 0 1"
    assert other not in {r["start"] for r in c.rows}
    g2 = game_of(other, 3)
    positions = [{"start": s0, "moves": g0[:k]} for k in range(0, 6)] + [{"start": s1, "moves": game_of(s1, 3)[:2]}, {"start": other, "moves": g2[:2]}]
    positions.append(dict(positions[1]))                                                       # a duplicate in the list
    exam_pos = {(s0, tuple(g0[:3]))}
    lab = DBad(hashed)
    c.cards(*spec_vocab())
    old_cards = len(c._cards)
    g = CPB.grow(c, positions, lab, exclude=exam_pos, say=QUIET)
    assert g["known"] == 2 and g["exam"] == 1 and g["added"] == len(c) - n0 == len(g["rows"])        # the start (ply 0) is in the corpus; the duplicate is not added twice
    assert (s0, tuple(g0[:3])) not in {(r["start"], tuple(r["moves"])) for r in g["rows"]}
    keys = [CE.state("chess", r["start"], r["moves"])[4] for r in c.rows]
    assert len(set(keys)) == len(keys)                                                         # still distinct positions
    for r in g["rows"]:
        assert r["keep"] == lab.moves_keep("chess", r["start"], r["moves"]) and r["before"] == lab.result("chess", r["start"], r["moves"])      # the exact labels
        if r["start"] != other:
            assert r["group"] == next(x["group"] for x in c.rows[:n0] if x["start"] == r["start"])    # its game's start's group: the split by group holds
    assert {k: v for k, v in c.groups.items() if k in groups0} == groups0                      # no group moved
    assert set(r["group"] for r in c.rows) <= set(c.groups)
    new_group = next(r["group"] for r in g["rows"] if r["start"] == other)
    assert new_group == "grown0" and c.groups["grown0"] == CPB.part_of_new_start(other) == ("dev" if int(hashlib.sha1(other.encode()).hexdigest(), 16) % 2 == 0 else "held")
    assert g["new_groups"] == ["grown0"] and g["dev"] + g["held"] == g["added"]
    assert len(c._cards) == len(c.rows) == old_cards + g["added"] and len(c.row_of) == len(c.rows)       # the cards were extended, not rebuilt
    fresh = CP.Corpus([dict(r) for r in c.rows], dict(c.groups), game="chess", variant="chess", size=3)
    fresh.cards(*spec_vocab())
    for rule in (ALL_MOVES_RULE, NEVER_RULE):
        for part in ("dev", "held"):
            a, b = CP.drill(rule, c, part=part), CP.drill(rule, fresh, part=part)
            assert {k: a[k] for k in ("binds", "names_a_move", "right", "wrong", "trivial")} == {k: b[k] for k in ("binds", "names_a_move", "right", "wrong", "trivial")}
    with pytest.raises(ValueError):
        c.extend([{"start": s0, "moves": ["x"], "group": "nosuch", "before": "draw", "keep": {}}])
    class Half(DBad):
        def moves_keep(self, variant, start, moves):
            return None
    n = len(c)
    assert CPB.grow(c, [{"start": s1, "moves": game_of(s1, 5)}], Half(hashed), say=QUIET)["uncovered"] == 1 and len(c) == n          # (not covered: not added)


def spec_vocab():
    return GS.spec_texts("chess", VOCAB), json.dumps(GS.vocabulary("chess", VOCAB))


def fake_decisions(corpus, wrong_first=True):
    """A stand-in for the wake's decision: a losing move at the first position asked, a keeping move at the next, and so on alternately."""
    seen = []

    def check(problems, **kw):
        out = []
        for p in problems:
            r = next(x for x in corpus.rows if (x["start"], x["moves"]) == (p["start"], p["moves"]))
            lose = sorted(m for m, k in r["keep"].items() if not k)
            keep = sorted(m for m, k in r["keep"].items() if k)
            seen.append(r)
            bad = (len(seen) % 2 == 1) == wrong_first
            out.append({"move": (lose[0] if bad else keep[0]), "keeps": not bad, "before": "draw", "after": "draw", "proved": False, "steps": [], "error": None})
        return out
    return check


def sandbox_with(tmp_path, rules):
    root = IN.prepare_sandbox("chess", tmp_path / "store")
    return root


def test_a_rule_with_a_wrong_firing_on_the_grown_corpus_is_withdrawn_and_one_without_stays(tmp_path, monkeypatch):
    c = starts_corpus()
    s0 = c.rows[0]["start"]
    g0 = game_of(s0, 6)
    write_log(tmp_path / "w.jsonl", [{"start": s0, "moves": g0}])
    shared = tmp_path / "shared_corpus.json"
    c.save(shared)
    before_bytes = shared.read_bytes()
    root = IN.prepare_sandbox("chess", tmp_path / "store")
    pf = root / "learned" / "chess" / "playout_problems.json"
    PS.save_problems(pf, [{"start": "x", "moves": [], "played": None, "first_seen": 1, "history": []}])
    lines = []
    want = GS.judge_of("chess", "material (mined)")
    with TG.sandbox(root):
        for l in (ALL_MOVES_RULE, NEVER_RULE):
            GS.write_lemma("chess", l)
        sid0 = GS.publish("chess", judge=want, vocabulary=VOCAB)
        assert {l["name"] for l in GS.read_lemmas("chess", VOCAB)} == {"all-moves", "never"}
        spec, vj = spec_vocab()
        c.cards(spec, vj)
        assert all(CP.drill(l, c, part=p)["wrong"] == 0 for l in (ALL_MOVES_RULE, NEVER_RULE) for p in ("dev", "held"))      # both are fine on the corpus before the growth
        monkeypatch.setattr(CPB.PS, "check", fake_decisions(c))
        rep = CPB.grow_and_withdraw("chess", c, str(tmp_path / "w.jsonl"), variant="chess", labeller=DBad(hashed), spec=spec, vocabulary=VOCAB, judge="material (mined)", seconds=0.01,
                                    problems_file=pf, cycle=3, grown_path=root / "learned" / "chess" / "corpus_grown.json", max_check=2, say=lines.append)
        assert [w["name"] for w in rep["withdrawn"]] == ["all-moves"] and rep["drilled"] == 2 and rep["added"] >= 4
        assert [l["name"] for l in GS.read_lemmas("chess", VOCAB)] == ["never"]                # the lemma file is gone, the other stays
        assert not (GS.lemma_dir("chess") / "all-moves.json").exists() and (GS.lemma_dir("chess") / "never.json").exists()
        sid1 = GS.head("chess")
        assert sid1 != sid0 and rep["snapshot"] == sid1
        snap = json.loads(GS.path_of("chess", sid1).read_text())
        assert [l["name"] for l in snap["lemmas"]] == ["never"] and snap["judge"] == want        # republished with the wake's valuation record
        assert GS.remove_lemma("chess", "all-moves") is False
    w = rep["withdrawn"][0]
    assert w["wrong"] > 0 and w["dev"] + w["held"] == w["wrong"] and w["wrong"] >= 2 and w["asked"] == 2 and w["wrong_without"] == 1 and w["new_problems"] == 1 and len(w["recorded"]) == 1
    rows = PS.load_problems(pf)
    withdrawn_rows = [r for r in rows if r.get("source") == "withdrawn"]
    assert len(rows) == 2 and len(withdrawn_rows) == 1 and all(r["first_seen"] == 3 and r["history"][-1]["status"] == "open" for r in withdrawn_rows)
    assert rows[0].get("source") is None
    text = "\n".join(lines)
    assert re.search(r"OBS corpus grow cycle 3: positions with a firing 5, added 5 \(dev \d+, held \d+\), rules drilled 2, withdrawn 1 \['all-moves'\]", text), text
    assert re.search(r"OBS corpus withdrawn cycle 3: rule all-moves, wrong firings \d+ \(dev \d+, held \d+\), e\.g\. .* after \d+ move\(s\) .*\(losing moves \[.*\]\); the decision without the rule is wrong at 1 of the 2 "
                     r"positions asked, new problems 1", text), text
    assert re.search(r"OBS corpus grow detail cycle 3: already in the corpus 0, exam positions left out 0, game over 0, not covered by the labeller 0, new start groups \[\], corpus size now \d+", text), text
    assert shared.read_bytes() == before_bytes                                                 # the shared cached corpus file is never written
    grown = CP.load_corpus(root / "learned" / "chess" / "corpus_grown.json")
    assert len(grown) == len(c) > len(CP.load_corpus(shared))


def test_a_rule_with_no_wrong_firing_changes_nothing_and_no_snapshot_is_published(tmp_path, monkeypatch):
    c = starts_corpus()
    s0 = c.rows[0]["start"]
    write_log(tmp_path / "w.jsonl", [{"start": s0, "moves": game_of(s0, 4)}])
    root = IN.prepare_sandbox("chess", tmp_path / "store")
    lines = []
    with TG.sandbox(root):
        GS.write_lemma("chess", NEVER_RULE)
        sid0 = GS.publish("chess", vocabulary=VOCAB)
        monkeypatch.setattr(CPB.PS, "check", lambda *a, **k: pytest.fail("no rule was withdrawn: the decision is not asked"))
        rep = CPB.grow_and_withdraw("chess", c, str(tmp_path / "w.jsonl"), variant="chess", labeller=DBad(hashed), spec=spec_vocab()[0], vocabulary=VOCAB, say=lines.append)
        assert rep["withdrawn"] == [] and GS.head("chess") == sid0 and [l["name"] for l in GS.read_lemmas("chess", VOCAB)] == ["never"]
    assert any(re.fullmatch(r"OBS corpus grow cycle 1: positions with a firing 3, added 3 \(dev \d+, held \d+\), rules drilled 1, withdrawn 0 \[\]", l) for l in lines), lines


def test_an_exam_position_where_a_rule_fired_is_not_added(tmp_path):
    c = starts_corpus()
    s0 = c.rows[0]["start"]
    m = game_of(s0, 4)
    write_log(tmp_path / "w.jsonl", [{"start": s0, "moves": m}])
    root = IN.prepare_sandbox("chess", tmp_path / "store")
    with TG.sandbox(root):
        rep = CPB.grow_and_withdraw("chess", c, str(tmp_path / "w.jsonl"), variant="chess", labeller=DBad(hashed), spec=spec_vocab()[0], vocabulary=VOCAB, exclude={(s0, tuple(m[:2]))}, say=QUIET)
    assert rep["added"] == 2 and rep["exam"] == 1 and (s0, tuple(m[:2])) not in {(r["start"], tuple(r["moves"])) for r in c.rows}


# ------------------------------------------------------------------------------------------------------------------------------ 3: heads in tiers

def keys_structure():
    return BS.from_vocabulary(GS.spec_texts("chess", KEYS)["chess"], GS.vocabulary_file("chess", KEYS))


def push_only(start, moves):
    """A class function under which only the pawn push f2f4 keeps the win's class (the other moves hand the win to the opponent)."""
    return "draw" if not moves else ("loss" if moves == ["f2f4"] else "win")


def pawn_problem():
    spec, vj = GS.spec_texts("chess", KEYS), json.dumps(GS.vocabulary("chess", KEYS))
    reader = TG.Reader("chess", spec, vj, PS.Verdicts(Fake(push_only), "chess"))
    rd = reader.read({"start": START, "moves": []})
    card = json.loads(reader.cards([(START, [], "p", "p")]).card(0))
    return reader, rd, card


def test_relative_heads_come_before_named_ones_and_each_head_has_its_tier():
    st = keys_structure()
    _reader, rd, card = pawn_problem()
    assert rd.good == ["f2f4"] and rd.bad
    heads = IND.tiered_heads(card, rd.keep, st, max_heads=3)
    tiers = [h["head_tier"] for h in heads]
    assert "relative" in tiers and "named" in tiers and tiers == sorted(tiers, key=("relative", "named").index)          # every relative head before any named head
    for h in heads:
        coords = IND.head_coordinates(h["head"], st)
        assert (not coords) if h["head_tier"] == "relative" else bool(coords)
        moves = {m["name"]: m for m in card["moves"]}
        assert h["matched"] and all(rd.keep[x] for x in h["matched"])                         # a head names keeping moves only
        assert sorted(x for x, mv in moves.items() if TG.head_matches(h["head"], mv["removed"], mv["added"])) == h["matched"]
    assert len({IND._alpha_key(h["head"]) for h in heads if h["head_tier"] == "relative"}) == sum(1 for t in tiers if t == "relative")      # no duplicate relative head
    assert max(tiers.count("relative"), tiers.count("named")) <= 3
    old = TG.heads_for(card, rd.keep, limit=16)
    assert any(IND.head_coordinates(h["head"], st) for h in old)                              # (heads_for itself still gives named heads: only induce.py orders them)
    assert [h["head_tier"] for h in IND.tiered_heads(card, rd.keep, st, max_heads=3, head_tiers=["named"])] == ["named"] * tiers.count("named")
    assert [h["head_tier"] for h in IND.tiered_heads(card, rd.keep, st, max_heads=3, head_tiers=["named", "relative"])][0] == "named"
    with pytest.raises(ValueError):
        IND.tiered_heads(card, rd.keep, st, max_heads=3, head_tiers=["nosuch"])


def test_a_named_head_gives_its_relative_form_once_and_never_a_head_that_names_a_losing_move():
    st = keys_structure()
    _reader, rd, card = pawn_problem()
    named = next(h for h in TG.heads_for(card, rd.keep, limit=16) if IND.head_coordinates(h["head"], st))
    rf = IND.relative_form(named, card, rd.keep, st)
    assert rf is not None and not IND.head_coordinates(rf["head"], st) and rf["matched"] and all(rd.keep[x] for x in rf["matched"])
    assert all(c in rf["names"] for c in IND.head_coordinates(named["head"], st))              # the lifted coordinates are variables the body may share
    assert IND.relative_form(rf, card, rd.keep, st) is None                                    # nothing left to lift
    nothing_keeps = dict.fromkeys(rd.keep, False)
    assert IND.relative_form(named, card, nothing_keeps, st) is None


def test_without_axes_the_heads_are_the_old_ones_and_have_no_tier():
    _reader, rd, card = pawn_problem()
    for structure in (None, BS.Structure(GS.spec_texts("chess", VOCAB)["chess"], [])):
        got = IND.tiered_heads(card, rd.keep, structure, max_heads=4)
        assert got == TG.heads_for(card, rd.keep, limit=4) and not any("head_tier" in h for h in got)


def test_induce_tries_every_relative_head_before_a_named_one_and_records_the_head_tier(monkeypatch):
    reader, rd, card = pawn_problem()
    c = tiny_corpus(60)
    seen = []
    good = {"binds": 3, "names_a_move": 3, "right": 99, "wrong": 0, "trivial": 0, "truncated": False}
    monkeypatch.setattr(IND.CP, "drill", lambda lem, corpus, **k: dict(good))

    def named_wins(corpus, facts, names, head, *, cands=None, needs=None, **kw):
        coords = IND.head_coordinates(head, keys_structure())
        seen.append("named" if coords else "relative")
        return {"found": [([cands[0][0]], {"wrong": 0, "right": 99})] if coords else [], "best": None, "levels": []}
    monkeypatch.setattr(IND, "search_body", named_wins)
    lines = []
    out = IND.induce({"start": START, "moves": []}, c, reader=reader, say=lines.append, structure=keys_structure(), max_heads=2)
    assert out["status"] == "found" and out["head_tier"] == "named" and out["candidates"][0]["head_tier"] == "named"
    first_named = seen.index("named")
    assert first_named >= 1 and set(seen[:first_named]) == {"relative"} and out["heads_tried"] >= 2       # every relative head failed (every tier of its body) before the named one
    assert any("[relative head]" in l for l in lines) and any(re.search(r"OBS induce .* head \d+: named head removes", l) for l in lines)
    seen.clear()

    def relative_wins(corpus, facts, names, head, **kw):
        coords = IND.head_coordinates(head, keys_structure())
        seen.append("named" if coords else "relative")
        return {"found": [([kw["cands"][0][0]], {"wrong": 0, "right": 99})], "best": None, "levels": []}
    monkeypatch.setattr(IND, "search_body", relative_wins)
    out = IND.induce({"start": START, "moves": []}, c, reader=reader, say=QUIET, structure=keys_structure(), max_heads=2)
    assert out["head_tier"] == "relative" and seen == ["relative"]                              # a relative head that works: no named head is ever tried


def test_the_admitted_lemma_records_the_head_tier_and_the_setting_defaults_are_as_stated():
    assert IND.SETTINGS["head_tiers"] == ["relative", "named"] and IND.SETTINGS["max_problems"] is None and IND.HEAD_TIERS == ("relative", "named")
    m = IND.provenance({"start": "x", "moves": ["a"], "played": "a"}, "chess", Fake(), KEYS, "relative", "named")
    assert m["head_tier"] == "named" and m["tier"] == "relative"
    assert "head_tier" not in IND.provenance({"start": "x", "moves": ["a"]}, "chess", Fake(), KEYS, "relative")


# ------------------------------------------------------------------------------------------------------------------------------ the cap on the problems attempted

def test_max_problems_caps_the_attempts_oldest_first_and_says_how_many_are_left(tmp_path, monkeypatch):
    c = tiny_corpus(80)
    root = IN.prepare_sandbox("chess", tmp_path / "store")
    tried = []
    monkeypatch.setattr(IND, "induce", lambda problem, corpus, **kw: (tried.append(problem["moves"]), {"status": "open", "why": "stub", "stage": "head", "lemma": None, "head": None, "dev": None, "held": None,
                                                                                                         "tiers": {}, "heads_tried": 0, "candidates": []})[1])

    monkeypatch.setattr(IND.PS, "check", lambda ps, **kw: [{"move": "x", "keeps": False, "before": None, "after": None, "proved": False, "steps": [], "error": None} for _ in ps])
    start = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    rows = [{"start": start, "moves": ["a"] * i, "first_seen": 5 - i, "history": []} for i in range(5)]      # the oldest (smallest first_seen) is the LAST row
    lines = []
    with TG.sandbox(root):
        GS.publish("chess", vocabulary=VOCAB)
        rep = IND.solve_problems("chess", rows, c, labeller=Fake(hashed), vocabulary=VOCAB, spec=GS.spec_texts("chess", VOCAB), settings={"max_problems": 2}, say=lines.append)
        assert rep["attempted"] == 2 and rep["left"] == 3 and tried == [["a"] * 4, ["a"] * 3]
        tried.clear()
        rep = IND.solve_problems("chess", rows, c, labeller=Fake(hashed), vocabulary=VOCAB, spec=GS.spec_texts("chess", VOCAB), say=lines.append)
        assert rep["attempted"] == 5 and rep["left"] == 0 and tried == [["a"] * i for i in range(5)]           # default: all, in file order
    assert "OBS induce deferred: max_problems 2, open at the start 5, attempted 2, left for the next cycle 3" in lines
    assert sum(1 for l in lines if l.startswith("OBS induce deferred")) == 1                                    # (not printed without a cap)


# ------------------------------------------------------------------------------------------------------------------------------ the switches and the cycle

def test_the_switches_are_off_by_default_and_run_size_refuses_them_without_the_corpus():
    sig = inspect.signature(IN.run_size).parameters
    assert sig["corpus_problems"].default == 0 and sig["grow_corpus"].default is False
    src = type("S", (), {"grade_mode": "score"})()
    with pytest.raises(ValueError, match="exact corpus"):
        IN.run_size("chess", 3, source=src, corpus_problems=5, playouts=True, say=QUIET)
    with pytest.raises(ValueError, match="exact corpus"):
        IN.run_size("chess", 3, source=src, grow_corpus=True, say=QUIET)
    with pytest.raises(ValueError, match="playouts"):
        IN.run_size("chess", 3, source=src, corpus_problems=5, exact=True, say=QUIET)
    with pytest.raises(ValueError):
        IN.run_size("chess", 3, source=src, corpus_problems=-1, exact=True, playouts=True, say=QUIET)


def test_the_cli_has_the_flags_and_reads_a_list_setting():
    text = (ROOT / "catspace/incubator/analyst/discovery/initialize.py").read_text()
    assert '"--corpus-problems"' in text and '"--grow-corpus"' in text and "corpus_problems=a.corpus_problems, grow_corpus=a.grow_corpus" in text
    assert "head_tiers=relative,named" in text


def test_the_wake_the_snapshot_writer_and_the_rust_sources_never_read_the_new_training_module():
    names = ("corpus_problems", "discovery.corpus_problems", "discovery import corpus_problems", "corpus_grown")
    for p in (ROOT / "catspace/incubator/analyst/discovery/wake.py", ROOT / "catspace/incubator/analyst/discovery/game_store.py", ROOT / "catspace/incubator/analyst/discovery/wdl_table.py"):
        assert not any(n in p.read_text() for n in names), p.name
    for p in sorted((ROOT / "rust/engine").glob("*/src/**/*.rs")):
        t = p.read_text()
        assert "corpus_problems" not in t and "corpus_grown" not in t, p


def test_the_new_files_are_in_the_gate_and_in_repo_structure():
    assert "test_plan78_34_corpus_problems" in (ROOT / "scripts/gate.py").read_text()
    rs = (ROOT / "repo_structure.md").read_text()
    for f in ("corpus_problems.py", "remove_lemma", "tiered_heads", "Corpus.extend"):
        assert f in rs


def test_one_small_cycle_with_the_corpus_problems_and_the_growth_on(tmp_path, monkeypatch):
    """run_size on a tiny sandbox with a fake labeller and a rule already in the store: corpus problems before induction, the corpus grown from the wake's firings, the rule withdrawn, the exam graded
    after the withdrawal, the grown corpus saved beside the sandbox and the shared corpus file untouched."""
    store = tmp_path / "store"
    monkeypatch.setattr(IN.paths, "experiment", lambda name: str(tmp_path / name))
    monkeypatch.setattr(EX, "labeller_of", lambda game, **kw: DBad(hashed))
    monkeypatch.setattr(CP, "default_path", lambda game, size, n, seed: str(tmp_path / "corpus.json"))
    root = IN.prepare_sandbox("chess", store)
    with TG.sandbox(root):
        GS.write_lemma("chess", ALL_MOVES_RULE)
    lines = []

    class Src:
        source, training_only, grade_mode = "fake", True, "score"

        def rank(self, variant, moves, cap=None, start=None):
            _f, _m, legal, _o, _k = CE.state(variant, start, list(moves))
            return [(m, 0) for m in sorted(legal)]

        def grade(self, s):
            return s

        def __call__(self, variant, moves, start=None):
            return None

        def close(self):
            pass
    rep = IN.run_size("chess", 3, source=Src(), store_root=store, games_per_start=1, seconds=0.02, max_plies=8, limit=2, vocabulary=VOCAB, tag="t34", cycles=1, sleep_settings={"depth": 2, "threads": 1},
                      playouts=True, playout_settings={"positions": 4, "per_position": 1, "max_plies": 20, "check_seconds": 0.02}, exact=True, induce=True,
                      induce_settings={"min_right": 1000, "max_heads": 1, "max_literals": 1, "max_problems": 2}, exam=str(tmp_path / "exam.json"), exam_size=6, corpus_n=300,
                      corpus_problems=6, grow_corpus=True, say=lines.append, sample=5)
    text = "\n".join(lines)
    cyc = rep["cycles"][0]
    assert re.search(r"OBS corpus problems cycle 1: asked \d+, wrong \d+, new problems \d+ \(already known \d+\)", text), text
    assert re.search(r"OBS problems by source cycle 1: playout total \d+ open \d+, corpus total \d+ open \d+, withdrawn total 0 open 0", text), text
    assert re.search(r"OBS corpus grow cycle 1: positions with a firing \d+, added \d+ \(dev \d+, held \d+\), rules drilled 1, withdrawn", text), text
    assert cyc["corpus_problems"]["asked"] <= 6 and cyc["corpus_grow"]["drilled"] == 1 and cyc["corpus_grow"]["positions"] > 0
    assert text.index("OBS induce total") < text.index("OBS corpus grow cycle 1") < text.index("OBS exam after cycle 1")       # induction, then the wake and the growth, then the exam
    assert text.index("OBS init wake b") < text.index("OBS corpus grow cycle 1")
    w = cyc["corpus_grow"]["withdrawn"]
    if w:                                                                                       # (the rule names every move, so it is wrong on the games' positions whenever one loses)
        assert w[0]["name"] == "all-moves" and "OBS corpus withdrawn cycle 1: rule all-moves" in text
        assert not (root / "learned" / "chess" / "lemmas" / "all-moves.json").exists() and rep["wake_b"]["survivor"] == cyc["corpus_grow"]["snapshot"]
        assert cyc["exam"]["label"] == "after cycle 1"
    assert (root / "learned" / "chess" / "corpus_grown.json").exists()
    grown = CP.load_corpus(root / "learned" / "chess" / "corpus_grown.json")
    assert len(CP.load_corpus(tmp_path / "corpus.json")) <= 300 and len(grown) >= len(CP.load_corpus(tmp_path / "corpus.json")) - 6
    ex = EM.exam_keys(EM.load_exam(tmp_path / "exam.json"))
    assert not ex & {(r["start"], tuple(r["moves"])) for r in grown.rows}                      # no exam position in the grown corpus
