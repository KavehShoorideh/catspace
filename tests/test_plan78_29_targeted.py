"""Plan 78.29: the targeted miner (discovery/targeted.py): one problem, the simplest rule in the store's lemma format, drilled against counter-positions, admitted only when the
problem's own check passes. Fast: a fake judge, tmp sandbox, the engine's own decision; the one Stockfish test skips without it."""
import json
import os
import re
import tempfile
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import initialize as IN
from catspace.incubator.analyst.discovery import playout_sleep as PS
from catspace.incubator.analyst.discovery import targeted as TG
from catspace.incubator.analyst.discovery import wdl_sleep as WS

STOCKFISH = "/opt/homebrew/bin/stockfish"
FEN = "8/8/8/8/2k2P2/8/4K3/8 w - - 9 6"
VOCAB = "relational"


class FakeJudge(WS.WdlJudge):
    """Verdicts by a function of the position (start, moves): `rule(start, moves) -> (w, d, l)` for the side to move."""
    source = "fake"

    def __init__(self, rule):
        self.rule, self.asked = rule, 0

    def wdl(self, variant, moves, start=None):
        self.asked += 1
        return self.rule(start, list(moves))


def sandbox_with_head(tmp: str) -> Path:
    root = IN.prepare_sandbox("chess", Path(tmp) / "store")
    with TG.sandbox(root):
        GS.publish("chess", vocabulary=VOCAB)
    return root


def problems_file(tmp: str, rows: list) -> str:
    p = Path(tmp) / "problems.json"
    PS.save_problems(p, [{"start": r["start"], "moves": r["moves"], "played": r.get("played"), "first_seen": 1, "history": []} for r in rows])
    return str(p)


def judge_where_only(bad_first: str):
    """A fake outside engine for the problem FEN with no prefix: the position is a win for the mover; after the first move `bad_first` the opponent is shown a draw (the mover's win was
    thrown away), after any other first move the opponent is shown a loss (the win is kept); every other position is a loss for its mover (nothing can drop)."""
    def rule(start, moves):
        if not moves:
            return (1.0, 0.0, 0.0)
        if len(moves) == 1:
            return (0.0, 1.0, 0.0) if moves[0] == bad_first else (0.0, 0.0, 1.0)
        return (0.0, 0.0, 1.0)
    return FakeJudge(rule)


# ---------------------------------------------------------------------------------------------------------------------------------------- pure pieces

def test_head_matches_reads_a_response_the_way_the_engine_does():
    removed = [["cell", "e", "2", "us", "king"]]
    added = [["cell", "f", "3", "us", "king"], ["flight", "us", "g", "4"]]
    head = {"removed": [("cell", ["?a", "?b", "us", "king"])], "added": [("cell", ["?c", "?d", "us", "king"])]}
    assert TG.head_matches(head, removed, added)
    assert not TG.head_matches({"removed": [], "added": [("cell", ["f", "3", "us", "pawn"])]}, removed, added)
    same = {"removed": [("cell", ["?a", "?b", "us", "king"])], "added": [("cell", ["?a", "?d", "us", "king"])]}          # the same variable must take the same value
    assert not TG.head_matches(same, removed, added)


def test_lift_head_makes_one_variable_per_constant_and_never_lifts_the_sides():
    h, names = TG.lift_head({"removed": [("cell", ["e", "2", "us", "king"])], "added": [("cell", ["e", "3", "us", "king"])]})
    assert h["removed"][0][1][0] == h["added"][0][1][0] and h["removed"][0][1][2] == "us"
    assert set(names) == {"e", "2", "3", "king"}
    h2, names2 = TG.lift_head({"removed": [], "added": [("cell", ["e", "3", "us", "king"])]}, frozenset({"king"}))
    assert h2["added"][0][1][3] == "king" and "king" not in names2


def test_heads_name_only_moves_that_keep_and_the_fewest_first():
    card = {"facts": [], "moves": [
        {"name": "m1", "removed": [["a", "x"]], "added": [["b", "x"], ["c"]]},
        {"name": "m2", "removed": [["a", "y"]], "added": [["b", "y"]]},
        {"name": "m3", "removed": [["a", "z"]], "added": [["b", "z"], ["c"]]}]}
    hs = TG.heads_for(card, {"m1": True, "m2": False, "m3": True}, limit=50)
    assert hs and all(set(h["matched"]) <= {"m1", "m3"} for h in hs)
    assert [len(h["matched"]) for h in hs] == sorted(len(h["matched"]) for h in hs)
    assert not any(TG.head_matches(h["head"], card["moves"][1]["removed"], card["moves"][1]["added"]) for h in hs)
    assert TG.heads_for(card, {"m1": True, "m2": True, "m3": True}) == []          # nothing to separate
    assert TG.heads_for(card, {"m1": False, "m2": False, "m3": False}) == []          # nothing keeps


def test_body_literals_share_a_variable_for_a_constant_and_use_the_heads_variables():
    facts = [["attack", "us", "e", "2", "f", "3"], ["cell", "f", "3", "us", "pawn"]]
    c = dict((d, l) for l, d in TG.body_candidates(facts, {"f": "?h0"}, set()))
    assert c["attack(us, ?k1, ?k2, ?h0, ?k3)"]["args"][3] == "?h0"
    lits = [l for l, _ in TG.body_candidates(facts, {}, set())]
    lift = [l for l in lits if l["relation"] == "attack" and not any(a in ("e", "2", "f", "3") for a in l["args"])][0]
    assert lift["args"][1] != lift["args"][3]
    assert not [l for l, _ in TG.body_candidates(facts, {}, {json.dumps(lits[0], sort_keys=True)}) if l == lits[0]]


def test_family_key_groups_by_make_up_and_effect():
    a = {"facts": [["cell", "a", "1", "us", "king"], ["closer", "d0", "d1"]], "moves": [{"name": "x", "removed": [["cell", "a", "1", "us", "king"]], "added": []}]}
    b = {"facts": [["cell", "b", "2", "us", "king"], ["closer", "d0", "d2"], ["closer", "d1", "d2"]], "moves": [{"name": "x", "removed": [["cell", "b", "2", "us", "king"]], "added": []}]}
    assert TG.family_key(a, "x") == TG.family_key(b, "x")
    assert TG.family_key(a, "x") != TG.family_key(a, "nothing")


def test_provenance_says_it_was_mined_from_a_problem_and_holds_no_verdict():
    meta = TG.provenance({"start": FEN, "moves": [], "played": "f4f5"}, {}, [1, 2], 40, {"fires": 5, "right": 5, "wrong": 0}, FakeJudge(lambda s, m: (1, 0, 0)), "chess", VOCAB)
    assert meta["origin"] == "targeted" and meta["mined_from"] == "problem" and meta["vocabulary"] == VOCAB and meta["drill"]["wrong"] == 0
    text = json.dumps(meta)
    assert not re.search(r"\b(win|draw|loss)\b", text) and "wdl" not in text


# ---------------------------------------------------------------------------------------------------------------------------------------- the drill and the rule, with the engine

def test_the_drill_counts_a_firing_that_names_a_losing_move_as_wrong():
    with tempfile.TemporaryDirectory() as tmp:
        root = sandbox_with_head(tmp)
        with TG.sandbox(root):
            spec, vocab = GS.spec_texts("chess", VOCAB), json.dumps(GS.vocabulary("chess", VOCAB))
            v = PS.Verdicts(judge_where_only("f4f5"), "chess")
            reader = TG.Reader("chess", spec, vocab, v)
            rd = reader.read({"start": FEN, "moves": []}, "problem")
            assert rd.bad == ["f4f5"] and "e2f3" in rd.good and rd.before == "win"
            drill = TG.Drill(reader, [rd])
            pawn_move = TG.lemma_of([{"relation": "control", "args": ["us"], "neg": False}], {"removed": [("cell", ["f", "4", "us", "pawn"])], "added": []})
            king_move = TG.lemma_of([{"relation": "control", "args": ["us"], "neg": False}], {"removed": [("cell", ["e", "2", "us", "king"])], "added": []})
            a, b = drill.read(pawn_move), drill.read(king_move)
            assert (a["fires"], a["wrong"], a["right"]) == (1, 1, 0)
            assert (b["fires"], b["wrong"], b["right"]) == (1, 0, 1)


def test_solve_admits_a_rule_only_when_the_problems_own_check_passes_and_records_where_it_came_from():
    with tempfile.TemporaryDirectory() as tmp:
        root = sandbox_with_head(tmp)
        spec = GS.spec_texts("chess", VOCAB)
        with TG.sandbox(root):
            snap = str(GS.path_of("chess", GS.head("chess")))
            chosen = PS.check([{"start": FEN, "moves": []}], variant="chess", store=snap, spec=spec, judge=judge_where_only("zzzz"), seconds=0.1)[0]["move"]
        pf = problems_file(tmp, [{"start": FEN, "moves": [], "played": chosen}])
        out = []
        rep = TG.solve_problems("chess", pf, store_root=root, judge=judge_where_only(chosen), vocabulary=VOCAB, seconds=0.1, logs=[], say=out.append,
                                settings={"min_right": 0, "min_fresh": 0, "wild_n": 0, "drill_cap": 40, "similar": 0, "neighbours": 6, "max_heads": 4, "verify_cap": 20, "verify_similar": 0, "verify_neighbours": 4})
        text = "\n".join(out)
        assert rep["total"] == 1 and rep["attempted"] == 1, text
        assert re.search(r"OBS targeted problems: total 1, attempted 1, solved 1, still open 0, rules admitted 1", text), text
        assert rep["admitted"] and rep["admitted"][0]["problem"] == 0
        with TG.sandbox(root):
            lemmas = GS.read_lemmas("chess", VOCAB)
            assert len(lemmas) == 1 and lemmas[0]["meta"]["origin"] == "targeted" and lemmas[0]["meta"]["mined_from"] == "problem"
            assert GS.head("chess") == rep["snapshot"] and len(GS.info("chess")["lemmas"]) == 1
            again = PS.check([{"start": FEN, "moves": []}], variant="chess", store=str(GS.path_of("chess", GS.head("chess"))), spec=spec, judge=judge_where_only(chosen), seconds=0.1)[0]
            assert again["keeps"] and again["move"] != chosen


def test_solve_leaves_a_problem_open_with_its_reason_when_no_move_keeps_the_verdict():
    with tempfile.TemporaryDirectory() as tmp:
        root = sandbox_with_head(tmp)
        pf = problems_file(tmp, [{"start": FEN, "moves": [], "played": "f4f5"}])
        every_move_loses = FakeJudge(lambda s, m: (1.0, 0.0, 0.0) if not m else (0.0, 1.0, 0.0))
        out = []
        rep = TG.solve_problems("chess", pf, store_root=root, judge=every_move_loses, vocabulary=VOCAB, seconds=0.1, say=out.append, settings={"drill_cap": 10, "similar": 0, "neighbours": 2, "wild_n": 0})
        assert rep["solved"] == 0 and rep["still_open"] == 1 and rep["admitted"] == []
        assert "still open, because no move keeps the verdict" in "\n".join(out)
        with TG.sandbox(root):
            assert GS.read_lemmas("chess", VOCAB) == []


def test_the_bar_is_not_lowered_too_few_right_firings_leave_the_problem_open():
    with tempfile.TemporaryDirectory() as tmp:
        root = sandbox_with_head(tmp)
        spec = GS.spec_texts("chess", VOCAB)
        with TG.sandbox(root):
            snap = str(GS.path_of("chess", GS.head("chess")))
            chosen = PS.check([{"start": FEN, "moves": []}], variant="chess", store=snap, spec=spec, judge=judge_where_only("zzzz"), seconds=0.1)[0]["move"]
        pf = problems_file(tmp, [{"start": FEN, "moves": [], "played": chosen}])
        out = []
        rep = TG.solve_problems("chess", pf, store_root=root, judge=judge_where_only(chosen), vocabulary=VOCAB, seconds=0.1, say=out.append,
                                settings={"min_right": 50, "min_fresh": 0, "wild_n": 0, "drill_cap": 30, "similar": 0, "neighbours": 4, "max_heads": 2, "verify_cap": 10, "verify_similar": 0, "verify_neighbours": 2})
        assert rep["solved"] == 0 and rep["admitted"] == []
        assert "(need 50 in all and 0 on fresh positions)" in "\n".join(out)


def test_the_store_paths_are_restored_after_a_sandbox_block():
    before = GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        root = IN.prepare_sandbox("chess", Path(tmp) / "s")
        with TG.sandbox(root):
            assert GS.LEARNED_ROOT == (root / "learned").resolve()
    assert (GS.LEARNED_ROOT, GS.SNAPSHOT_ROOT) == before


# ---------------------------------------------------------------------------------------------------------------------------------------- the cycle option and the outside engine

def test_run_size_has_the_targeted_switch_off_by_default():
    import inspect
    sig = inspect.signature(IN.run_size)
    assert sig.parameters["targeted"].default is False
    assert "--targeted" in open(IN.__file__).read()


@pytest.mark.skipif(not os.path.exists(STOCKFISH), reason="no Stockfish")
def test_stockfish_names_the_one_keeping_move_of_a_king_and_pawn_problem():
    judge = WS.UciWdlJudge(STOCKFISH, 12)
    try:
        rd = TG.Reader("chess", GS.spec_texts("chess", VOCAB), json.dumps(GS.vocabulary("chess", VOCAB)), PS.Verdicts(judge, "chess")).read({"start": "5k2/8/4K3/5P2/8/8/8/8 w - - 21 18", "moves": []})
        assert rd.before == "win" and rd.good == ["e6f6"] and "f5f6" in rd.bad
    finally:
        judge.close()


# ---------------------------------------------------------------------------------------------------------------------------------------- the wild drill (plan 78.29, the audit's standard)

class FakeRank:
    """A rank source: the first legal move (sorted) is the only best one."""
    source = "fake-rank"
    training_only = True

    def rank(self, variant, moves, cap=None, start=None):
        legal = sorted(CE.state(variant, start, list(moves))[2])
        return [(m, 100 if i == 0 else 0) for i, m in enumerate(legal)]


def fake_log(tmp: str, games: int = 12, plies: int = 8) -> str:
    import random
    rng = random.Random(3)
    path = Path(tmp) / "wake.jsonl"
    with open(path, "w") as f:
        for g in range(games):
            moves = []
            for _ in range(plies):
                legal = CE.state("chess", None, moves)[2]
                if not legal:
                    break
                moves.append(rng.choice(sorted(legal)))
            f.write(json.dumps({"kind": "game", "run": "r", "game": g, "moves": moves, "start": None}) + "\n")
            for ply, m in enumerate(moves):
                f.write(json.dumps({"kind": "position", "label": f"r:{g}:{ply}:{ply % 2}", "steps": [], "move": m}) + "\n")
    return str(path)


def test_the_standards_are_reconciled_in_one_printed_text():
    assert "ANY named move" in TG.STANDARD and "CardSet.eval" in TG.STANDARD and "abstraction.Gates" in TG.STANDARD


def test_the_wild_held_out_half_applies_the_lemma_sleeps_own_gates_and_says_why_it_rejects():
    with tempfile.TemporaryDirectory() as tmp:
        root = sandbox_with_head(tmp)
        with TG.sandbox(root):
            spec, vocab = GS.spec_texts("chess", VOCAB), json.dumps(GS.vocabulary("chess", VOCAB))
            reader = TG.Reader("chess", spec, vocab, PS.Verdicts(judge_where_only("zzzz"), "chess"))
            out = []
            wild = TG.Wild(reader, [fake_log(tmp)], FakeRank(), n=80, say=out.append)
            assert wild.dev_idx and wild.held_idx and not set(wild.dev_idx) & set(wild.held_idx)
            assert {wild.groups[i] for i in wild.dev_idx}.isdisjoint({wild.groups[i] for i in wild.held_idx})          # split by game
            assert wild.gates.min_precision == 1.0 and wild.gates.min_groups == 2                                       # the sleep's held gates
            everything = TG.lemma_of([{"relation": "control", "args": ["us"], "neg": False}], {"removed": [], "added": [("cell", ["?a", "?b", "us", "?c"])]})
            w = wild.held(everything)
            assert not w["ok"] and w["fails"] and w["summary"]["expected"] is not None
            TG.say_wild("probe", w, out.append)
            assert any("REJECTED" in l and "chance level" in l and "excess" in l for l in out)


def test_solve_with_a_wild_drill_admits_only_what_passes_the_held_out_gates_or_says_why_not():
    with tempfile.TemporaryDirectory() as tmp:
        root = sandbox_with_head(tmp)
        spec = GS.spec_texts("chess", VOCAB)
        with TG.sandbox(root):
            snap = str(GS.path_of("chess", GS.head("chess")))
            chosen = PS.check([{"start": FEN, "moves": []}], variant="chess", store=snap, spec=spec, judge=judge_where_only("zzzz"), seconds=0.1)[0]["move"]
        pf = problems_file(tmp, [{"start": FEN, "moves": [], "played": chosen}])
        out = []
        rep = TG.solve_problems("chess", pf, store_root=root, judge=judge_where_only(chosen), teacher=FakeRank(), logs=[fake_log(tmp)], vocabulary=VOCAB, seconds=0.1, say=out.append,
                                settings={"min_right": 0, "min_fresh": 0, "drill_cap": 30, "similar": 0, "neighbours": 4, "max_heads": 3, "verify_cap": 10, "verify_similar": 0,
                                          "verify_neighbours": 2, "wild_n": 80})
        text = "\n".join(out)
        assert "OBS targeted standard:" in text and "OBS targeted wild:" in text and "wild held-out:" in text
        for a in rep["admitted"]:
            assert a["wild_held"]["precision"] == 1.0 and a["wild_held"]["fires"] >= 3
            assert re.search(r"OBS targeted rule .*wild held-out: fires \d+, right \d+, precision", text)
        if not rep["admitted"]:
            assert "wild held-out half" in rep["rows"][0]["why"] or "wild" in rep["rows"][0]["why"], rep["rows"]


def test_solve_refuses_to_go_without_a_rank_source_unless_the_wild_drill_is_switched_off():
    with tempfile.TemporaryDirectory() as tmp:
        root = sandbox_with_head(tmp)
        pf = problems_file(tmp, [{"start": FEN, "moves": [], "played": "f4f5"}])
        with pytest.raises(ValueError):
            TG.solve_problems("chess", pf, store_root=root, judge=judge_where_only("f4f5"), vocabulary=VOCAB, seconds=0.1, say=lambda *_a: None)


# ---------------------------------------------------------------------------------------------------------------------------------------- forced seeds of the lemma sleep (second route)

def test_forced_positions_are_the_open_problems_carried_ones_too_and_replace_duplicate_turnings():
    rows = [{"start": FEN, "moves": [], "played": "f4f5", "history": [{"status": "open", "move": "f4f5"}]},
            {"start": FEN, "moves": ["e2e3"], "played": "x", "history": [{"status": "solved", "move": "y"}]},
            {"start": FEN, "moves": ["e2d3"], "played": "p", "history": [{"status": "open", "move": None}]}]
    turning = [{"id": "playout:0:0", "start": FEN, "moves": [], "played": "f4f5", "group": "playout:0"}, {"id": "playout:1:1", "start": FEN, "moves": ["e2f3"], "played": "z", "group": "playout:1"}]
    forced, rest = IN.forced_positions(rows, turning)
    assert [f["id"] for f in forced] == ["problem:0:0", "problem:2:1"] and all(f["force"] for f in forced)
    assert forced[1]["played"] == "p"                                  # (no decision move recorded: the playout's)
    assert [e["id"] for e in rest] == ["playout:1:1"]


def test_a_forced_card_is_always_shown_to_the_rank_source_and_a_regret_is_raised_for_it():
    from catspace.incubator.analyst.discovery import detectors as D
    from catspace.incubator.analyst.discovery import sleep as SL
    spec, vocab = GS.spec_texts("chess", VOCAB), json.dumps(GS.vocabulary("chess", VOCAB))
    poss = [(None, [], "c0", "g0"), (None, ["e2e4"], "c1", "g1"), (None, ["d2d4"], "c2", "g2")]
    cards = CE.build_cards([("chess", s, m, i, g) for s, m, i, g in poss], spec, vocab, 1)
    legal = sorted(CE.state("chess", None, [])[2])
    played = {"c0": legal[1], "c1": sorted(CE.state("chess", None, ["e2e4"])[2])[1], "c2": sorted(CE.state("chess", None, ["d2d4"])[2])[1]}          # the source's second best everywhere
    def ctx(force):
        return D.Context("chess", SL.Settings(teacher_positions=1), {}, {}, cards=cards, dev_idx=[0, 1, 2], positions={i: (s, m) for s, m, i, _g in poss}, played=played,
                         spec=spec, vocab=vocab, rep={}, force_ids=force)
    assert [c["id"] for _i, c in D._sampled(ctx(set()))] == ["c0"]
    c = ctx({"c2"})
    assert [x["id"] for _i, x in D._sampled(c)] == ["c0", "c2"]
    found = D.Regret(FakeRank()).detect(c)
    assert [f["id"] for f in found] == ["c2"] and found[0]["regret"]["rule"] == "forced"          # the other is within the source's top three: no regret
    assert D.Regret(FakeRank()).detect(ctx(set())) == []


def test_sleep_puts_a_forced_seed_in_the_development_half_and_reports_where_it_stops():
    from catspace.incubator.analyst.discovery import sleep as SL
    with tempfile.TemporaryDirectory() as tmp:
        root = sandbox_with_head(tmp)
        log = fake_log(tmp, games=10, plies=6)
        legal = sorted(CE.state("chess", None, ["e2e4"])[2])
        extra = [{"id": "problem:0:1", "start": None, "moves": ["e2e4"], "played": legal[1], "group": "problem:0", "force": True}]
        out = []
        st = SL.Settings(judge_by="source", vocabulary=VOCAB, depth=1, max_cards=24, max_seeds=1, max_saving=1)
        with TG.sandbox(root):
            rep = SL.sleep("chess", log, settings=st, teacher=FakeRank(), write=False, say=out.append, extra_positions=extra)
            plain = SL.sleep("chess", log, settings=st, teacher=FakeRank(), write=False, say=lambda *_a, **_k: None)
        assert "forced" not in plain and "forced" not in plain["funnel"]                                                   # (default behaviour unchanged)
        assert rep["forced"][0]["id"] == "problem:0:1" and rep["forced"][0]["stage"]
        assert rep["funnel"]["cards_held"] == plain["funnel"]["cards_held"]                                                # the forced card is never held out
        assert any(l.startswith("OBS sleep forced seeds: given 1, confirmed") for l in out)
        assert rep["forced"][0]["confirmed"] or any("died" in l for l in out)


def test_a_move_counter_fact_is_never_a_body_literal():
    facts = [["halfmove", "21"], ["cell", "f", "5", "us", "pawn"]]
    assert TG.is_counter(facts[0]) and not TG.is_counter(facts[1])
    assert all(not l["relation"] == "halfmove" for l, _d in TG.body_candidates(facts, {}, set()))
