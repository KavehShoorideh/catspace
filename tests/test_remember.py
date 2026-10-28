"""ANALYST_PLAN 50: the remembered state (relations stored, recalled by relations, the recalled family asked first)."""
import re

import pytest

from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import remember as RM


def _puzzle(pid):
    return next(p for p in PA.load_set("dev_20") if p["id"] == pid)


def _record(pid, family, **kw):
    p = _puzzle(pid)
    us = "w" if p["fen"].split()[1] == "w" else "b"
    thr = RM.threat_token(p["fen"], us)
    return p, us, RM.make_record(p, "dev_20", p["fen"], us, p["answer"], family, 100, True, thr, **kw)


def test_a_record_round_trips_and_a_stored_id_is_skipped(tmp_path):
    f = str(tmp_path / "s.jsonl")
    _, _, rec = _record("01IFU", "mate_in_two")
    assert RM.store(rec, f) is True
    assert RM.store(rec, f) is False                                  # the per-item checkpoint: a rerun skips the stored id
    assert RM.load(f) == [rec] and RM.stored_ids(f) == {"01IFU"}
    assert rec["source"] == "puzzle" and rec["family"] == "mate_in_two"


def test_a_signature_has_no_square_names():
    _, _, rec = _record("01IFU", "mate_in_two")
    sig = RM.sig_of_record(rec)
    assert sig and not [t for t in sig if re.search(r"\b[a-h][1-8]\b", t)]
    print("OBS signature of 01IFU:", sorted(sig))


def test_jaccard_of_identical_is_one_and_of_disjoint_is_zero():
    assert RM.jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert RM.jaccard({"a"}, {"b"}) == 0.0


def test_recall_never_returns_the_puzzles_own_id(tmp_path):
    f = str(tmp_path / "s.jsonl")
    _, _, r1 = _record("01IFU", "mate_in_two")
    _, _, r2 = _record("03B3o", "mate_in_two")
    RM.store(r1, f), RM.store(r2, f)
    got = RM.recall(RM.sig_of_record(r1), 3, exclude_id="01IFU", path=f)
    assert [r["id"] for _, r in got] == ["03B3o"]
    assert RM.recall(RM.sig_of_record(r1), 3, path=f)[0][0] == 1.0       # without the exclusion the identical record comes back at 1.0


def test_propose_on_03B3o_after_storing_01IFU_names_mate_in_two(tmp_path):
    f = str(tmp_path / "s.jsonl")
    _, _, rec = _record("01IFU", "mate_in_two")
    RM.store(rec, f)
    p = _puzzle("03B3o")
    us = "w" if p["fen"].split()[1] == "w" else "b"
    got = RM.propose(p["fen"], us, exclude_id="03B3o", path=f)
    print("OBS propose on 03B3o after storing 01IFU:", got)
    assert len(got) == 1
    fam, mover, victim, mk, sim, rid = got[0]
    assert (fam, mover, victim, mk, rid) == ("mate_in_two", "N", "none", "check", "01IFU")
    assert 0.0 < sim <= 1.0
    assert RM.ladder_name(fam) == "mate2"


# ---------------------------------------------------------------------------------------------------------------- 52.9: memory orders the agenda

def _agenda(pid, memory=None):
    from catspace.incubator.analyst.discovery import base_vocabulary as BV
    from catspace.incubator.analyst.discovery.agenda import Agenda
    from catspace.incubator.analyst.discovery.state import state_of
    BV.install_terms()
    p = _puzzle(pid)
    us = "w" if p["fen"].split()[1] == "w" else "b"
    ag = Agenda(state_of(p["fen"], us), memory=memory)
    ag.ask()
    return p, ag


def test_a_record_carries_the_goal_and_fix_kinds():
    p, ag = _agenda("000Sa")
    d = ag.decision()
    assert d.goal and isinstance(d.fixes, tuple)
    us = "w" if p["fen"].split()[1] == "w" else "b"
    rec = RM.make_record(p, "dev_20", p["fen"], us, d.move, d.lemma, d.value, True, RM.threat_token(p["fen"], us), goal=d.goal, fixes=d.fixes, lemma=d.lemma)
    assert rec["goal"] == d.goal and rec["fixes"] == list(d.fixes) and rec["lemma"] == d.lemma and rec["pieces"] and rec["roles"]
    print("OBS record of 000Sa:", rec["goal"], rec["fixes"], rec["lemma"])


def test_the_key_reorders_equal_stakes_only():
    p, plain = _agenda("000Sa")
    stakes = [q.stake for q in plain.questions]
    goals = {q.goal for q in plain.questions}
    # a memory that loves 'win' roots: a win root may overtake an equal-stake root of another goal, never a higher stake
    mem = [(0.9, dict(id="x", goal="win", lemma="L", fixes=[]))]
    _, ag = _agenda("000Sa", mem)
    order = [q.stake for q in ag.open_questions(ordered=True)]           # (plan 74 (b): the heap is the worklist, read in its order key)
    assert order == sorted(order, reverse=True)                         # still ordered by stake first
    assert sorted(q.stake for q in ag.all) == sorted(q.stake for q in plain.all)   # nothing added or removed
    assert ag.asked <= ag.budget and goals


def test_decisions_are_identical_on_000Sa_with_and_without_memory():
    p, plain = _agenda("000Sa")
    for goal in ("mate", "win", "save", "promote"):
        _, ag = _agenda("000Sa", [(0.8, dict(id="x", goal=goal, lemma="L", fixes=[]))])
        assert ag.decision().move == plain.decision().move
    print("OBS 000Sa decision with and without memory:", plain.decision().what)
