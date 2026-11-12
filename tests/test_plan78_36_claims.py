"""Plan 78.36: a rule that CLAIMS a result and a search that works toward it. The claim rule in the engine's own language (check_lemma), its drill on the exact corpus (right where the row's exact
class is the claimed one, wrong otherwise), its admission, the decision's claim step (present with claims in the snapshot, absent with the switch off, and a snapshot of response rules only decides
as it did), and the conversion exam (fake labeller, fake decision). Everything in tmp sandboxes; the real store is never written; no tablebase and no Stockfish."""
import json
import re
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import conversion as CV
from catspace.incubator.analyst.discovery import corpus as CP
from catspace.incubator.analyst.discovery import exact as EX
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import induce as IND
from catspace.incubator.analyst.discovery import initialize as IN
from catspace.incubator.analyst.discovery import playout_sleep as PS

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "rules"
KEYS = "keys"
START = "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"          # king and pawn against king: three units
QUIET = lambda *_: None  # noqa: E731


def lit(rel, *args, neg=False):
    return {"relation": rel, "args": list(args), "neg": neg}


def claim(name, body, cls, **extra):
    return {"name": name, "body": body, "claim": {"class": cls}, "stats": {}, "meta": extra}


KING = [lit("cell", "?f", "?r", "us", "king")]                       # binds wherever the side to move has a king (everywhere)
NOBODY = [lit("cell", "?f", "?r", "us", "dragon")]                    # binds nowhere


class Fake(EX.ExactLabeller):
    source = "fake"

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def _class(self, variant, start, moves):
        return self.fn(start, moves)


# ---------------------------------------------------------------------------------------------------------------------------------------- the rule in the engine's language

def test_check_lemma_accepts_a_claim_and_rejects_both_and_neither():
    CE.check_lemma(json.dumps(claim("k", KING, "win")))
    for cls in ("draw", "loss"):
        CE.check_lemma(json.dumps(claim("k", KING, cls)))
    both = claim("k", KING, "win")
    both["response"] = {"removed": [lit("cell", "?f", "?r", "us", "king")], "added": []}
    with pytest.raises(ValueError, match="not both"):
        CE.check_lemma(json.dumps(both))
    neither = claim("k", KING, "win")
    del neither["claim"]
    with pytest.raises(ValueError, match="neither"):
        CE.check_lemma(json.dumps(neither))
    with pytest.raises(ValueError, match="win, draw or loss"):
        CE.check_lemma(json.dumps(claim("k", KING, "maybe")))
    with pytest.raises(ValueError, match="positive literal"):
        CE.check_lemma(json.dumps(claim("k", [], "win")))
    resp = {"name": "r", "body": KING, "response": {"removed": [lit("cell", "?f", "?r", "us", "king")], "added": []}, "stats": {}, "meta": {}}
    CE.check_lemma(json.dumps(resp))                                   # (a response rule is what it was)


# ---------------------------------------------------------------------------------------------------------------------------------------- the drill of a claim

def corpus_with(fn, n=60):
    c = CP.build_corpus("chess", 3, n=n, seed=1, labeller=Fake(fn), say=QUIET)
    spec, vocab = GS.spec_texts("chess", "relational"), json.dumps(GS.vocabulary("chess", "relational"))
    c.cards(spec, vocab)
    return c, spec, vocab


def hashed(start, moves):
    return ("loss", "draw", "win")[sum(map(ord, (start or "") + "".join(moves))) % 3]


def test_the_claim_drill_counts_right_and_wrong_by_the_exact_class_and_zero_binds_is_zero():
    c, _spec, _vocab = corpus_with(hashed)
    for part in ("dev", "held"):
        want = {k: 0 for k in ("win", "draw", "loss")}
        for ci in c.idx(part):
            want[c.rows[c.row_of[ci]]["before"]] += 1
        n = sum(want.values())
        for cls in ("win", "draw", "loss"):
            d = CP.drill(claim("k", KING, cls), c, part=part)
            print(f"claim {cls} on {part}: binds/right/wrong {d['binds']}/{d['right']}/{d['wrong']}")
            assert d["binds"] == n and d["right"] == want[cls] and d["wrong"] == n - want[cls] and d["right"] + d["wrong"] == d["binds"]
            assert d["trivial"] == 0 and len(d["wrong_examples"]) == min(5, d["wrong"])
            assert all(e["exact_class"] != cls for e in d["wrong_examples"])
        z = CP.drill(claim("none", NOBODY, "win"), c, part=part)
        assert (z["binds"], z["right"], z["wrong"]) == (0, 0, 0)
    # the many-rule call keeps the order and mixes the kinds
    resp = {"name": "r", "body": KING, "response": {"removed": [lit("cell", "?f", "?r", "us", "king")], "added": []}, "stats": {}, "meta": {}}
    many = CP.drill_many([claim("a", KING, "win"), resp, claim("b", NOBODY, "draw")], c, part="dev")
    assert many[0]["claim"] == "win" and "claim" not in many[1] and many[2]["binds"] == 0


def test_drill_admit_keeps_a_claim_that_is_never_wrong_and_drops_a_wrong_one_and_one_that_never_binds():
    c, spec, vocab = corpus_with(lambda s, m: "win")                   # every position is a win for its mover
    lines = []
    resp = {"name": "r", "body": KING, "response": {"removed": [lit("cell", "?f", "?r", "us", "king")], "added": []}, "stats": {}, "meta": {}}
    kept, dropped = IND.drill_admit([claim("is-win", KING, "win"), claim("is-draw", KING, "draw"), claim("never", NOBODY, "win"), resp], c, spec, vocab, say=lines.append)
    assert [k["name"] for k in kept if "claim" in k] == ["is-win"]
    assert sorted(d["name"] for d in dropped if "claim" in d) == ["is-draw", "never"]
    pat = re.compile(r"OBS drill admit (\S+) \(claim (win|draw|loss)\): dev binds/right/wrong \d+/\d+/\d+, held \d+/\d+/\d+; (KEPT|DROPPED)")
    by = {m.group(1): m.group(3) for m in map(pat.match, lines) if m}
    assert by == {"is-win": "KEPT", "is-draw": "DROPPED", "never": "DROPPED"}
    assert any(l.startswith("OBS drill admit r: dev binds/right/wrong/trivial") for l in lines), "a response rule's line is the line it was"
    assert lines[-1] == "OBS drill admit total: rules 4, kept " + str(len(kept)) + ", dropped " + str(len(dropped))


CORPUS_FILE = Path(CP.default_path("chess", 3, 30000, 1))


@pytest.mark.skipif(not CORPUS_FILE.exists(), reason="the cached exact corpus is not there")
def test_a_hand_claim_on_the_real_corpus_has_consistent_counts():
    full = CP.load_corpus(CORPUS_FILE)
    sub = CP.Corpus(full.rows[::10], dict(full.groups), game=full.game, variant=full.variant, size=full.size, seed=full.seed)
    spec, vocab = GS.spec_texts("chess", KEYS), json.dumps(GS.vocabulary("chess", KEYS))
    sub.cards(spec, vocab)
    rule = claim("hand-key-gap-d0-wins", [lit("key_gap", "us", "?f", "?r", "d0")], "win")
    for part in ("dev", "held"):
        d = CP.drill(rule, sub, part=part)
        print(f"hand claim (key_gap us d0 -> win) on {len(sub.part(part))} {part} rows: binds/right/wrong {d['binds']}/{d['right']}/{d['wrong']}")
        assert d["right"] + d["wrong"] == d["binds"] and d["trivial"] == 0 and not d["truncated"]
    kept, dropped = IND.drill_admit([rule], sub, spec, vocab, say=print)
    assert len(kept) + len(dropped) == 1


# ---------------------------------------------------------------------------------------------------------------------------------------- the decision

@pytest.fixture
def store(tmp_path):
    old = GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT
    d = tmp_path / "learned" / "chess"
    d.mkdir(parents=True)
    (d / "vocabulary.keys.json").write_text((RULES / "learned/chess/vocabulary.keys.json").read_text())
    GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT = tmp_path / "learned", tmp_path / "snapshots"
    try:
        yield tmp_path
    finally:
        GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT = old


def decide(sid, pos, *, seconds=0.5, **kw):
    """The batch solver's decision on positions [(fen, moves)] with the sandbox snapshot -> (rows, steps per position)."""
    import os
    import tempfile
    specs = GS.spec_texts("chess", KEYS)
    with tempfile.TemporaryDirectory() as d:
        trace = os.path.join(d, "t.jsonl")
        rows = CE.solve_batch([("chess", fen, mv) for fen, mv in pos], seconds, snapshot="t", trace_path=trace, threads=1, store=str(GS.path_of("chess", sid)), specs=specs,
                              labels=[f"p{i}" for i in range(len(pos))], ties=True, **kw)
        recs = [json.loads(l) for l in open(trace) if l.strip()]
    return rows, [[s["operator"] for s in r["steps"]] for r in recs], recs


POSITIONS = [(START, []), ("8/8/8/4k3/8/8/4P3/4K3 b - - 0 1", []), ("8/8/4k3/8/3PK3/8/8/8 w - - 0 1", []), ("8/8/8/8/3k4/8/3P4/3K4 w - - 0 1", [])]


def test_a_snapshot_of_response_rules_only_decides_and_traces_as_it_did(store):
    rules = json.loads((RULES / "concepts/chess_keys.proposed.json").read_text())["rules"]
    GS.write_lemma("chess", {**rules[0], "meta": {**rules[0].get("meta", {}), "vocabulary": KEYS}})
    sid = GS.publish("chess", vocabulary=KEYS)
    raw = json.loads(GS.path_of("chess", sid).read_bytes())
    assert [l["name"] for l in raw["lemmas"]] == [rules[0]["name"]] and "claims" not in raw and all("claim" not in l for l in raw["lemmas"])
    info = CE.snapshot_info(str(GS.path_of("chess", sid)))
    assert info["lemmas"] == [rules[0]["name"]] and info["claims"] == []
    on, steps_on, _ = decide(sid, POSITIONS, claims=True)
    off, steps_off, _ = decide(sid, POSITIONS, claims=False)
    assert steps_on == steps_off and all("claim" not in s for s in steps_on)
    assert [r[0] for r in on] == [r[0] for r in off], "the same moves"
    print("response-only snapshot, steps per position:", steps_on)


def test_a_claim_in_the_snapshot_gives_the_claim_step_and_the_switch_removes_it(store):
    rule = claim("hand-key-gap-wins", [lit("key_gap", "us", "?f", "?r", "?d")], "win", vocabulary=KEYS)
    GS.write_lemma("chess", rule)
    sid = GS.publish("chess", vocabulary=KEYS)
    info = CE.snapshot_info(str(GS.path_of("chess", sid)))
    assert info["lemmas"] == [] and info["claims"] == ["hand-key-gap-wins"]
    assert GS.read_lemmas("chess", KEYS) == [] and [c["name"] for c in GS.read_claims("chess", KEYS)] == ["hand-key-gap-wins"], "readers of responses do not meet a claim"
    rows, steps, recs = decide(sid, POSITIONS, claims=True)
    assert any("claim" in s for s in steps), steps
    assert all(s[s.index("claim") - 1] == "forcing" for s in steps if "claim" in s), "the claim step follows the proof"
    for r in recs:
        for t in r["trace"]:
            if t.startswith("claim toward"):
                print(t)
    assert any(t.startswith("claim toward: depth") for r in recs for t in r["trace"])
    off_rows, off_steps, off_recs = decide(sid, POSITIONS, claims=False)
    assert all("claim" not in s for s in off_steps)
    assert not any("claim" in t for r in off_recs for t in r["trace"])
    # the search on its own: depth, nodes, claim evaluations and the class of every root move
    d = CE.claim_toward("chess", str(GS.path_of("chess", sid)), GS.spec_texts("chess", KEYS), fen=START, seconds=0.3)
    print("claim_toward:", {k: d[k] for k in ("status", "depth", "nodes", "claim_evaluations", "seconds", "refuted", "candidates", "spoke")})
    assert d["depth"] >= 1 and d["nodes"] > 0 and d["claim_evaluations"] > 0 and d["evidence"][0].startswith("claim toward: depth")


def test_a_claim_is_published_with_the_response_rules_and_changes_the_address(store):
    rules = json.loads((RULES / "concepts/chess_keys.proposed.json").read_text())["rules"]
    GS.write_lemma("chess", {**rules[0], "meta": {**rules[0].get("meta", {}), "vocabulary": KEYS}})
    base = GS.publish("chess", vocabulary=KEYS)
    GS.write_lemma("chess", claim("hand-key-gap-wins", [lit("key_gap", "us", "?f", "?r", "?d")], "win", vocabulary=KEYS))
    both = GS.publish("chess", vocabulary=KEYS)
    assert both != base
    raw = json.loads(GS.path_of("chess", both).read_bytes())
    assert len(raw["lemmas"]) == 2 and sum("claim" in l for l in raw["lemmas"]) == 1
    i = CE.snapshot_info(str(GS.path_of("chess", both)))
    assert len(i["lemmas"]) == 1 and i["claims"] == ["hand-key-gap-wins"]
    # (the response rule's JSON is byte for byte what it was in the first snapshot)
    first = json.loads(GS.path_of("chess", base).read_bytes())["lemmas"][0]
    assert [l for l in raw["lemmas"] if "claim" not in l][0] == first


# ---------------------------------------------------------------------------------------------------------------------------------------- the conversion exam

WIN_MATE = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"             # Ra8 mates
WIN_DROP = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
DRAW_HOLD = "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"
DRAW_DROP = "4k3/8/8/8/8/8/8/R2K4 w - - 0 1"


def fake_class(start, moves):
    if start == WIN_MATE:
        return "win"
    if start == DRAW_DROP:
        return "win" if len(moves) == 3 else "draw"          # the engine's second move hands the opponent a win
    return "draw"


def first_legal(problems):
    return [{"move": sorted(CE.state("chess", p["start"], p["moves"])[2])[0], "error": None} for p in problems]


def fake_decide(problems):
    out = first_legal(problems)
    for o, p in zip(out, problems):
        if p["start"] == WIN_MATE:
            o["move"] = "a1a8"                                  # mate: the game's own win
    return out


class DropFirst(Fake):
    """WIN_DROP's engine move hands the class to the opponent at once."""

    def _class(self, variant, start, moves):
        if start == WIN_DROP and moves:
            return "win"                                         # (for the side to move after the engine's move: the opponent) -> the engine's class dropped to loss
        return fake_class(start, moves)


def exam_of(positions):
    return {"variant": "chess", "positions": [{"start": s, "moves": [], "group": "g", "before": b} for s, b in positions]}


def test_conversion_counts_kept_dropped_finished_and_max_plies_with_a_fake_labeller_and_decision():
    exam = exam_of([(WIN_MATE, "win"), (WIN_DROP, "win"), (DRAW_HOLD, "draw"), (DRAW_DROP, "draw")])
    r = CV.convert(exam, labeller=DropFirst(fake_class), decide=fake_decide, n=4, max_plies=4)
    rows = {x["i"]: x for x in r["rows"]}
    assert (r["games"], r["win_games"], r["draw_games"]) == (4, 2, 2)
    assert rows[0]["status"] == "finished" and rows[0]["outcome"] == "win" and rows[0]["plies"] == 1
    assert rows[1]["status"] == "dropped" and rows[1]["drop"]["ply"] == 1 and rows[1]["drop"]["after"] == "loss"
    assert rows[2]["status"] == "max_plies" and rows[2]["plies"] == 4 and "drop" not in rows[2]
    assert rows[3]["status"] == "dropped" and rows[3]["drop"]["ply"] == 3 and rows[3]["drop"]["after"] == "loss"
    assert (r["kept"], r["kept_win"], r["kept_draw"], r["finished"]) == (2, 1, 1, 1)
    assert r["median_drop_ply"] == 2.0 and len(r["drops"]) == 2
    assert r["drops"][1]["moves"] and len(r["drops"][1]["moves"]) == 2, "the position before the move that dropped the class"
    lines = []
    CV.say_conversion("test", r, lines.append)
    assert lines[0] == ("OBS conversion test: games 4 (win-class 2, draw-class 2); never dropped the class 2 (win-class 1 of 2, draw-class 1 of 2); wins finished 1 of 2 within 4 plies; "
                        "first drop at ply (median) 2")
    assert len(lines) == 3 and all(l.startswith("OBS conversion example:") for l in lines[1:])
    print("\n".join(lines))


def test_the_games_are_picked_half_and_half_deterministically():
    pos = [(f"fen{i}", "win") for i in range(5)] + [(f"fen{i}", "draw") for i in range(5, 8)]
    exam = exam_of(pos)
    assert CV.pick(exam, 4) == [0, 1, 5, 6]
    assert CV.pick(exam, 4) == CV.pick(exam, 4)
    assert CV.pick(exam, 8) == list(range(8))
    assert CV.pick(exam_of([("a", "win")] * 3), 4) == [0, 1, 2], "a short class leaves its share"
    assert CV.pick(exam_of([]), 4) == []
    # at most n, whatever the exam holds
    assert len(CV.pick(exam, 3)) == 3


def test_a_position_the_labeller_does_not_cover_ends_the_game_without_a_drop():
    exam = exam_of([(DRAW_HOLD, "draw")])
    lab = Fake(lambda s, m: "draw" if len(m) < 2 else None)
    r = CV.convert(exam, labeller=lab, decide=first_legal, n=1, max_plies=10)
    assert r["rows"][0]["status"] == "uncovered" and r["kept"] == 1 and r["drops"] == []


def test_the_default_decision_is_the_checks_call_with_the_given_clock_and_switches(monkeypatch):
    seen = {}

    def fake_check(problems, **kw):
        seen.update(kw)
        return first_legal(problems)
    monkeypatch.setattr(PS, "check", fake_check)
    CV.convert(exam_of([(DRAW_HOLD, "draw")]), labeller=Fake(lambda s, m: "draw"), store="S", spec={"chess": "T"}, seconds=0.7, wake_kw={"claims": False}, n=1, max_plies=2)
    assert seen["store"] == "S" and seen["seconds"] == 0.7 and seen["wake_kw"] == {"claims": False} and seen["variant"] == "chess"


# ---------------------------------------------------------------------------------------------------------------------------------------- the flags and the purity

def test_the_flags_parse_and_the_conversion_needs_an_exam():
    text = (ROOT / "catspace/incubator/analyst/discovery/initialize.py").read_text()
    assert '"--conversion"' in text and "conversion=a.conversion" in text and "conversion: int = 0" in text
    with pytest.raises(ValueError, match="needs an exam"):
        IN.run_size("chess", 3, source=None, conversion=5, say=QUIET)


def test_the_wake_the_snapshot_writer_and_the_rust_sources_never_read_the_training_modules():
    for p in (ROOT / "catspace/incubator/analyst/discovery/wake.py", ROOT / "catspace/incubator/analyst/discovery/game_store.py"):
        t = p.read_text()
        assert "import conversion" not in t and "discovery.conversion" not in t and "discovery import corpus" not in t and "discovery import exact" not in t, p.name
    for p in sorted((ROOT / "rust/engine").glob("*/src/**/*.rs")):
        t = p.read_text().lower()
        assert "conversion.py" not in t and "syzygy" not in t and "stockfish" not in t, p


def test_the_new_files_are_in_the_gate_and_in_repo_structure():
    assert "test_plan78_36_claims" in (ROOT / "scripts/gate.py").read_text()
    rs = (ROOT / "repo_structure.md").read_text()
    for f in ("conversion.py", "claim.rs", "toward.rs", "claim_op.rs", "CLAIM_SHARE", "claim_toward"):
        assert f in rs, f


def test_the_usage_split_reads_only_the_rules_that_name_a_move():
    """A snapshot's  list holds claim rules too (plan 78.36); usage_split.split must leave them out (a claim has no response to match)."""
    import inspect
    from catspace.incubator.analyst.discovery import usage_split as US
    src = inspect.getsource(US.split)
    assert '"claim" not in l' in src


# ---------------------------------------------------------------------------------------------------------------------------------------- suggestions

def test_check_lemma_takes_suggest_on_a_response_rule_only():
    resp = {"name": "r", "body": KING, "response": {"removed": [lit("cell", "?f", "?r", "us", "king")], "added": []}, "stats": {}, "meta": {}}
    CE.check_lemma(json.dumps({**resp, "suggest": True}))
    CE.check_lemma(json.dumps({**resp, "suggest": False}))
    with pytest.raises(ValueError, match="suggest"):
        CE.check_lemma(json.dumps({**claim("k", KING, "win"), "suggest": True}))
    with pytest.raises(ValueError, match="suggest"):
        CE.check_lemma(json.dumps({**resp, "suggest": "yes"}))


def test_drill_admit_makes_a_suggestion_of_a_rule_that_is_mostly_right_only_when_asked(monkeypatch):
    class Stub:
        def cards(self, *a):
            return None
    counts = {"mostly": ((15, 1), (10, 0)), "half": ((5, 5), (0, 0)), "clean": ((3, 0), (2, 0)), "few": ((8, 1), (4, 0))}

    def fake_drill(lem, corpus, part="dev", **k):
        r, w = counts[lem["name"]][0 if part == "dev" else 1]
        return {"binds": r + w, "names_a_move": r + w, "right": r, "wrong": w, "trivial": 0, "truncated": False, "wrong_examples": []}
    monkeypatch.setattr(IND.CP, "drill", fake_drill)
    rules = [{"name": n, "body": KING, "response": {"removed": [lit("cell", "?f", "?r", "us", "king")], "added": []}, "stats": {}, "meta": {}} for n in counts]
    lines = []
    out = IND.drill_admit(rules, Stub(), {}, "{}", say=lines.append)
    assert len(out) == 2, "without a bar the return is the 2-tuple it always was"
    assert sorted(l["name"] for l in out[0]) == ["clean"] and sorted(l["name"] for l in out[1]) == ["few", "half", "mostly"]
    assert lines[-1] == "OBS drill admit total: rules 4, kept 1, dropped 3"
    lines = []
    kept, dropped, sug = IND.drill_admit(rules, Stub(), {}, "{}", say=lines.append, suggest_bar=0.9)
    assert [l["name"] for l in kept] == ["clean"] and sorted(l["name"] for l in dropped) == ["few", "half"]
    assert [l["name"] for l in sug] == ["mostly"] and sug[0]["suggest"] is True and "suggest" not in rules[0]
    assert any(l.endswith("SUGGESTION (right 25, wrong 1, share 0.962)") and "drill admit mostly" in l for l in lines), lines
    assert lines[-1] == "OBS drill admit total: rules 4, kept 1, dropped 2, suggestions 1"
    k2, d2, s2 = IND.drill_admit(rules, Stub(), {}, "{}", say=QUIET, suggest_bar=0.99)
    assert s2 == [] and len(d2) == 3
    k3, d3, s3 = IND.drill_admit(rules, Stub(), {}, "{}", say=QUIET, suggest_bar=0.9, suggest_min_right=5)
    assert sorted(l["name"] for l in s3) == ["few", "mostly"]
    for l in sug:
        CE.check_lemma(json.dumps(l))


def test_a_suggestion_is_in_the_snapshot_and_changes_the_address_and_usage_split_counts_it_apart(store):
    from catspace.incubator.analyst.discovery import usage_split as US
    rules = json.loads((RULES / "concepts/chess_keys.proposed.json").read_text())["rules"]
    base = {**rules[0], "meta": {**rules[0].get("meta", {}), "vocabulary": KEYS}}
    GS.write_lemma("chess", base)
    plain = GS.publish("chess", vocabulary=KEYS)
    GS.write_lemma("chess", {**base, "suggest": True})
    sug = GS.publish("chess", vocabulary=KEYS)
    assert plain != sug
    i = CE.snapshot_info(str(GS.path_of("chess", sug)))
    assert i["lemmas"] == [base["name"]] and i["suggestions"] == [base["name"]]
    assert CE.snapshot_info(str(GS.path_of("chess", plain)))["suggestions"] == []
    pos = "8/8/4k3/8/3K4/8/5P2/8 w - - 0 1"          # (the position the hand rule names d4e4 at, as test_plan78_35_words)
    legal = CE.state("chess", pos, [])[2]

    class Keep(EX.ExactLabeller):
        def moves_keep(self, variant, start, moves):
            return {m: True for m in legal}
    exam = {"positions": [{"start": pos, "moves": [], "group": "g", "before": "win"}]}
    graded = {"decisions": [{"i": 0, "move": "d4e4", "keeps": True, "before": "win"}]}
    specs = GS.spec_texts("chess", KEYS)
    r_plain = US.split(exam, graded, variant="chess", snapshot_path=str(GS.path_of("chess", plain)), specs=specs, labeller=Keep())
    r_sug = US.split(exam, graded, variant="chess", snapshot_path=str(GS.path_of("chess", sug)), specs=specs, labeller=Keep())
    assert r_plain["kept"]["named_played"] == 1 and r_plain["suggestion_rules"] == 0
    assert r_sug["kept"]["named_played"] == 0 and r_sug["kept"]["no_rule"] == 1, "a suggestion is not a rule that named the move played"
    assert r_sug["suggestion_rules"] == 1 and r_sug["suggestions_named_played"] == 1
    lines, lines_plain = [], []
    US.say_split("t", r_sug, None, lines.append)
    US.say_split("t", r_plain, None, lines_plain.append)
    assert lines[0].endswith("; suggestions named the move played 1") and "suggestions named" not in lines_plain[0]


def test_the_suggest_bar_flag_is_in_the_cli():
    text = (ROOT / "catspace/incubator/analyst/discovery/initialize.py").read_text()
    assert '"--suggest-bar"' in text and "suggest_bar=a.suggest_bar" in text
