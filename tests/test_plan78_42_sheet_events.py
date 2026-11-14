"""Plan 78.42, the sheet's events: listeners over the fact store, the agenda, the drain under a budget, nothing stale served, superseded facts dropped. The unit tests use a stub codec and made-up
facts (no game); two tests read real positions."""
import math
import re
import time
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import question_sheet as Q
from catspace.incubator.analyst.discovery import sheet_events as SE

WORTH = {"k": 0, "q": 14, "r": 8, "b": 5, "n": 4, "p": 2}
OWN_IN_WAY = "4k3/8/8/3b4/8/8/2P5/N3K3 w - - 0 1"
ROOT = Path(__file__).resolve().parents[1]
EVENTS_SOURCE = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "sheet_events.py"
MEASURE_SOURCE = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "sheet_events_measure.py"


# ---------------------------------------------------------------------------------------------------------------------------------------------- a stub game: a position is "place=side+kind,..."

class StubCodec:
    """Places are letters; a position is 'p=uK,q=oR' (place = side letter u/o and a kind letter). The reader gives (place, side, kind, 0, 0); the context is the same for every position."""

    def _read(self, position):
        out = {}
        for part in filter(None, position.split(",")):
            place, what = part.split("=")
            out[place] = ("us" if what[0] == "u" else "them", what[1])
        return out

    def cell_reader(self, position):
        d = self._read(position)
        return lambda place: (place, *(d.get(place, (None, None))), 0, 0)

    def context(self, position):
        return ("us", None, None)

    def changed_places(self, before, after):
        a, b = self._read(before), self._read(after)
        return {p for p in set(a) | set(b) if a.get(p) != b.get(p)}


def _verdict(word, reads, **kw):
    return {"verdict": word, "value": kw.pop("value", 5.0), "reads": frozenset(reads), "route": None, "alts": kw.pop("alts", []), "breaks": kw.pop("breaks", []), "stops": kw.pop("stops", []),
            "answer": "", "nodes": 1, **kw}


def _ctx(stake=4.0, route=("p", "q"), answer=("z",), d=1, owner="them"):
    return {"t": (owner, "N", ("t",)), "unit": ("us", "R", ("u",)), "asker": "us", "way": (d, ("q",), "strike"), "routes": [], "stake": stake, "group": "clear", "route_places": frozenset(route),
            "answer_places": frozenset(answer)}


def _file(pool, name, word, reads, position="p=uK,q=oR,z=oB", **kw):
    ctx = kw.pop("ctx", None)
    if ctx is None:
        ctx = _ctx()
    return pool.add((name, "them", "u", "v", "strike", ()), (name, "them", "us"), _verdict(word, reads, **kw), position, extra=frozenset(reads), ctx=ctx)


def _pool():
    pool = Q.Pool(StubCodec())
    pool.position, pool.ply = "p=uK,q=oR,z=oB", 0
    return pool


class _Table:
    def chance(self, key, horizon=4):
        return {"waiting": 0.1, "confirmed": 0.9}.get(key[1], 0.5)


# ---------------------------------------------------------------------------------------------------------------------------------------------- listeners

def test_a_fact_wakes_when_a_place_in_its_read_set_changes_and_not_otherwise():
    pool = _pool()
    f = _file(pool, "a", "confirmed", {"p", "q"})
    pool.advance("p=uK,q=oR,z=oB,y=oN")                                   # a place the fact did not read changes
    assert f.id not in pool.store.stale and len(pool.agenda) == 0 and not pool.wake_log
    flagged = pool.advance("p=uK,z=oB,y=oN")                              # a place it read (q) is emptied
    assert f.id in flagged and f.id in pool.store.stale
    assert [(w["fid"], w["role"]) for w in pool.wake_log] == [(f.id, "route")]      # a confirmed way is subscribed by its route: the strongest role of the places it names
    assert len(pool.agenda) == 1


def test_a_waiting_record_wakes_on_its_condition_and_is_killed_by_its_break():
    pool = _pool()
    wait = Q.Stop("delay", "zq", 1, 1, "place_taken", ("q",), "the place is taken", ("z", "q"))
    f = _file(pool, "w", "waiting", {"p", "q", "z"}, alts=[("vacated", ("q",), "the place q is vacated")], breaks=[wait])
    pool.advance("p=uK,q=oR,z=oB,y=oN")
    assert not pool.wake_log
    pool.advance("p=uK,q=oR,z=oB,x=oP")                                   # nothing it reads
    pool.advance("p=uK,q=uQ,z=oB,x=oP")                                   # q still occupied: the condition is not true; q is read, so the fact is stale (touched), not woken by its condition
    assert pool.wake_log[-1]["role"] != "wake" and f.id in pool.store.stale
    pool2 = _pool()
    g = _file(pool2, "w", "waiting", {"p", "q", "z"}, alts=[("vacated", ("q",), "the place q is vacated")], breaks=[wait])
    pool2.advance("p=uK,z=oB")                                            # q is vacated: the condition is true now
    assert [(w["fid"], w["role"], w["why"]) for w in pool2.wake_log] == [(g.id, "wake", "vacated")]
    pool3 = _pool()
    k = _file(pool3, "w", "waiting", {"p", "q", "z"}, alts=[("vacated", ("q",), "the place q is vacated")], breaks=[wait])
    pool3.advance("p=uK,q=oB")                                            # their unit went from z to q: the answer the record names, played for real
    assert pool3.killed == 1 and k.id not in pool3.store.facts and k.id not in pool3.agenda._best          # killed and dropped
    assert [(w["role"]) for w in pool3.wake_log] == ["break"]


def test_a_condition_given_as_places_plus_a_test_is_subscribable_and_new_kinds_need_no_new_machinery():
    L = SE.Listeners()
    duty_lifted = SE.Cond("duty lifted", ("g", "h"), lambda cell: cell("g")[2] is None and cell("h")[2] == "R", "the duty on g is lifted")
    L.subscribe(7, "wake", {"x"}, duty_lifted)
    reader = lambda d: (lambda place: (place, *(d.get(place, (None, None))), 0, 0))
    assert L.notify({"g"}, reader({"g": ("us", "B"), "h": ("us", "R")})) == []                  # a place changed, the test is false
    got = L.notify({"h"}, reader({"h": ("us", "R")}))                                              # the test is true now
    assert [(w.fid, w.role, w.why) for w in got] == [(7, "wake", "duty lifted")]
    assert L.notify({"zz"}, reader({})) == []
    L.drop(7)
    assert L.notify({"g", "h"}, reader({"h": ("us", "R")})) == []
    with pytest.raises(ValueError):
        L.subscribe(1, "no such role", {"a"})


def test_the_ready_made_tests_read_the_codecs_cells():
    cell = lambda d: (lambda place: (place, *(d.get(place, (None, None))), 0, 0))
    assert SE.all_empty(("a", "b"))(cell({})) and not SE.all_empty(("a",))(cell({"a": ("us", "R")}))
    filed = {"a": cell({"a": ("us", "R")})("a")}
    assert not SE.changed_from(("a",), filed)(cell({"a": ("us", "R")})) and SE.changed_from(("a",), filed)(cell({}))
    assert SE.occupant_changed(("a",), filed)(cell({"a": ("them", "R")})) and not SE.occupant_changed(("a",), filed)(cell({"a": ("us", "R")}))
    assert SE.answer_made("a", "b", "them")(cell({"b": ("them", "N")})) and not SE.answer_made("a", "b", "them")(cell({"a": ("them", "N"), "b": ("them", "N")}))


# ---------------------------------------------------------------------------------------------------------------------------------------------- the agenda

def test_the_agenda_orders_by_priority_and_a_fact_woken_twice_is_on_it_once_with_the_strongest_role():
    A = SE.Agenda()
    A.push(SE.Wake(1, "reads", "a"), 3.0)
    A.push(SE.Wake(2, "wake", "b"), 9.0)
    A.push(SE.Wake(3, "route", "c"), 3.0)
    A.push(SE.Wake(1, "wake", "d"), 1.0)                                  # woken again, stronger role, lower priority: one entry, the highest priority, the strongest role
    assert len(A) == 3
    got = [(w.fid, w.role) for w in A.left()]
    assert got == [(2, "wake"), (1, "wake"), (3, "route")]
    assert [A.pop().fid for _ in range(3)] == [2, 1, 3] and A.pop() is None


def test_the_agenda_is_ordered_by_chance_times_stake_when_there_is_a_table_else_by_stake():
    pool = _pool()
    pool.table = _Table()
    hi_stake_waiting = _file(pool, "a", "waiting", {"p", "q"}, ctx=_ctx(stake=10.0))      # 10 x 0.1 = 1.0
    lo_stake_confirmed = _file(pool, "b", "confirmed", {"p", "q"}, ctx=_ctx(stake=2.0))   # 2 x 0.9 = 1.8
    pool.advance("p=uK,z=oB")                                             # q emptied: both are woken
    assert [w.fid for w in pool.agenda.left()] == [lo_stake_confirmed.id, hi_stake_waiting.id]
    pool2 = _pool()
    a = _file(pool2, "a", "waiting", {"p", "q"}, ctx=_ctx(stake=10.0))
    b = _file(pool2, "b", "confirmed", {"p", "q"}, ctx=_ctx(stake=2.0))
    pool2.advance("p=uK,z=oB")
    assert [w.fid for w in pool2.agenda.left()] == [a.id, b.id]           # no table: by stake


# ---------------------------------------------------------------------------------------------------------------------------------------------- the drain

class _R:
    position = "p=uK,z=oB"


def test_refresh_honours_the_budget_and_reports_what_is_left_as_stale_not_re_read():
    pool = _pool()
    facts = [_file(pool, n, "confirmed", {"p", "q"}, ctx=_ctx(stake=s)) for n, s in (("a", 3.0), ("b", 2.0), ("c", 1.0))]
    pool.advance("p=uK,z=oB")
    asked = []

    def slow(f, r, params):
        asked.append(f.id)
        time.sleep(0.03)
        pool._supersede(f)
        return "unit gone"
    pool._reask = slow
    got = pool.refresh(_R(), Q.Params(), budget=0.0)
    assert got["asked"] == 0 and got["left"] == 3 and not asked
    report = pool.stale_report()
    assert [x["last"] for x in report] == ["confirmed"] * 3 and all(x["status"] == "stale, not re-read" for x in report)
    got = pool.refresh(_R(), Q.Params(), budget=0.01)                     # one ask begun is finished: the first (highest stake), then the budget is used up
    assert asked == [facts[0].id] and got["asked"] == 1 and got["left"] == 2
    assert [x["key"][0] for x in pool.left_stale] == ["b", "c"]
    got = pool.refresh(_R(), Q.Params(), budget=None)
    assert asked == [f.id for f in facts] and got["left"] == 0 and pool.stale_report() == []


def test_nothing_stale_is_served_as_current_and_a_flagged_fact_with_the_same_cells_is_served_after_the_comparison():
    pool = _pool()
    f = _file(pool, "a", "confirmed", {"p", "q"})
    key = f.key
    pool.advance("p=uK,q=oR,z=oB,y=oN")
    assert pool.lookup(key, pool.position)[0] is f and pool.kept == 1                              # not flagged: served as filed
    pool.advance("p=uK,z=oB")                                                                       # a place it read is emptied: flagged, its cells no longer match
    got, state = pool.lookup(key, pool.position)
    assert got is None and state == "reread" and pool.reread == 1                                   # never served
    assert pool.stale_report() and pool.stale_report()[0]["last"] == "confirmed"                    # reported as stale, not re-read, with its last verdict
    pool.advance("p=uK,q=oR,z=oB")                                                                  # the cells are as they were: flagged, but equal on comparison
    got, state = pool.lookup(key, pool.position)
    assert got is f and state == "revalidated" and f.id not in pool.store.stale and f.id not in pool.agenda._best


def test_superseded_and_dead_facts_are_dropped_from_the_store_the_listeners_and_the_agenda():
    pool = _pool()
    f = _file(pool, "a", "confirmed", {"p", "q"})
    g = _file(pool, "b", "confirmed", {"p", "q"})
    pool.advance("p=uK,z=oB")
    assert len(pool.agenda) == 2
    pool._supersede(f)
    pool.collect()
    assert f.id not in pool.store.facts and f.id not in pool.events.subs and f.id not in pool.agenda._best and g.id in pool.store.facts
    assert all(fid != f.id for ids in pool.events.by_place.values() for fid, _i in ids)
    assert pool.dropped == 1 and not any(x.superseded for x in pool.store.facts.values())


# ---------------------------------------------------------------------------------------------------------------------------------------------- on real positions

def test_a_waiting_way_wakes_when_the_place_it_waits_on_is_vacated_and_the_drain_asks_it_again_before_the_questions(env_chess):
    env = env_chess
    pool = Q.Pool(env.codec)
    pool.advance(OWN_IN_WAY)
    Q.run(OWN_IN_WAY, WORTH, env=env, max_steps=60, pool=pool, light=True)
    wait = [f for f in pool.ways() if f.payload["verdict"]["verdict"] == Q.WAITING_V and f.key[2] == "a1"]
    assert wait and wait[0].payload["waits_on"] == ["the place c2 is vacated"]
    after = "4k3/8/8/3b4/8/2P5/8/N3K3 b - - 0 1"                         # the pawn on c2 steps to c3
    pool.advance(after)
    assert any(w["role"] == "wake" and w["why"] == "vacated" for w in pool.wake_log)
    sheet = Q.run(after, WORTH, env=env, max_steps=60, pool=pool, light=True, refresh=math.inf)
    asked = [x for x in pool.refreshed if x["role"] == "wake"]
    assert asked and asked[0]["old"] == Q.WAITING_V and asked[0]["new"] != Q.WAITING_V               # asked again at once, before any question
    assert sheet.refreshed["asked"] >= 1 and sheet.stale == [] and pool.stats()["stale_now"] <= len([f for f in pool.ways() if f.id in pool.store.stale])


def test_the_drain_under_a_budget_leaves_stale_facts_on_the_sheet_with_their_last_verdict(env_chess):
    env = env_chess
    pool = Q.Pool(env.codec)
    pool.advance(OWN_IN_WAY)
    Q.run(OWN_IN_WAY, WORTH, env=env, max_steps=60, pool=pool, light=True)
    pool.advance("4k3/8/8/3b4/8/2P5/8/N3K3 b - - 0 1")
    n = len(pool.agenda)
    sheet = Q.run("4k3/8/8/3b4/8/2P5/8/N3K3 b - - 0 1", WORTH, env=env, max_steps=60, pool=pool, light=True, refresh=0.0)
    assert n > 0 and sheet.refreshed["asked"] == 0 and len(sheet.stale) == n
    assert all(x["status"] == "stale, not re-read" and x["last"] in (Q.CONFIRMED_V, Q.REFUTED_V, Q.WAITING_V, Q.UNSETTLED_V) for x in sheet.stale)


def test_the_graph_of_a_question_is_one_fact_superseded_when_the_question_is_filed_again(env_chess):
    env = env_chess
    pool = Q.Pool(env.codec)
    pool.advance(OWN_IN_WAY)
    sheet = Q.read_position(env, WORTH, Q.Params(), OWN_IN_WAY, pool, light=True)
    n = pool.file_graph(sheet)
    qs = [f for f in pool.store.facts.values() if f.payload.get("question_graph")]
    assert n == len(qs) == len([q for q in sheet.questions if q.node is not None and q.target is not None]) < len(sheet.graph.nodes)      # fewer facts than nodes
    assert all(f.payload["nodes"] and "edges" in f.payload for f in qs)
    first = {f.id for f in qs}
    pool.file_graph(sheet)
    pool.collect()
    qs2 = [f for f in pool.store.facts.values() if f.payload.get("question_graph")]
    assert len(qs2) == n and not ({f.id for f in qs2} & first)           # the earlier ones were superseded and dropped
    legacy = Q.Pool(env.codec)
    assert legacy.file_graph(sheet, legacy=True) == len(sheet.graph.nodes)


def test_the_read_set_of_a_way_holds_the_route_what_bears_on_it_the_goal_units_and_the_answerers(env_chess):
    env = env_chess
    pool = Q.Pool(env.codec)
    pool.advance(OWN_IN_WAY)
    Q.run(OWN_IN_WAY, WORTH, env=env, max_steps=60, pool=pool, light=True)
    f = next(f for f in pool.ways() if f.key[2] == "a1")
    owner_units = set(env.codec.unit_places(OWN_IN_WAY, "black"))
    assert {"a1", "c2", "e1", "e8", "d5"} <= f.reads and any(p.startswith(Q.CONTEXT) for p in f.reads)
    allp = Q.Pool(env.codec, answerers="all")
    allp.advance(OWN_IN_WAY)
    Q.run(OWN_IN_WAY, WORTH, env=env, max_steps=60, pool=allp, light=True)
    g = next(g for g in allp.ways() if g.key == f.key)
    assert owner_units <= g.reads and f.reads <= g.reads


def test_the_modules_name_no_object_of_any_one_game():
    words = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check\w*|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
    for src in (EVENTS_SOURCE,):
        assert not re.findall(words, src.read_text(), flags=re.I), src.name


@pytest.fixture(scope="module")
def env_chess():
    return Q.Env("chess")


# ---------------------------------------------------------------------------------------------------------------------------------------------- the single shared agenda

KING_ROUTE = "8/6Np/8/1P6/2p1P2K/8/8/4k3 w - - 0 1"
KING_AFTER = "8/6Np/1P6/8/2p1P2K/8/8/4k3 b - - 0 1"           # my pawn b5 -> b6: only a unit of the target's side moved, far from the routes


def _woken_pool(env, fen=OWN_IN_WAY, after="4k3/8/8/3b4/8/2P5/8/N3K3 b - - 0 1", answerers="near"):
    pool = Q.Pool(env.codec, answerers=answerers)
    pool.advance(fen)
    Q.run(fen, WORTH, env=env, max_steps=60, pool=pool, light=True)
    pool.advance(after)
    assert len(pool.agenda) > 0
    return pool, after


def _order(monkeypatch, pool, after, env, recheck_p, step_p, **kw):
    seq = []
    orig = pool.recheck
    monkeypatch.setattr(pool, "recheck_priority", lambda w, c, p: recheck_p)
    monkeypatch.setattr(Q, "chance_priority", lambda q, c, p, t, r: step_p)
    monkeypatch.setattr(pool, "recheck", lambda w, r, p: (seq.append("recheck"), orig(w, r, p))[1])
    sheet = Q.run(after, WORTH, env=env, pool=pool, light=True, shared=True, on_step=lambda sh, st, el: seq.append("step"), **kw)
    return seq, sheet


def test_one_agenda_mixes_rechecks_and_new_steps_and_comes_out_in_priority_order(env_chess, monkeypatch):
    pool, after = _woken_pool(env_chess)
    n = len(pool.agenda)
    seq, _sheet = _order(monkeypatch, pool, after, env_chess, 9.0, 1.0, max_steps=3)
    assert seq[:n] == ["recheck"] * n and "step" in seq[n:]                  # every recheck outranks every step
    pool2, after = _woken_pool(env_chess)
    seq2, _ = _order(monkeypatch, pool2, after, env_chess, 1.0, 9.0, max_steps=3)
    assert seq2[:3] == ["step"] * 3 and "recheck" not in seq2[:3]            # every step outranks every recheck: the three steps first


def test_the_budget_is_shared_a_high_stake_step_before_a_low_stake_recheck_and_the_reverse(env_chess, monkeypatch):
    pool, after = _woken_pool(env_chess)
    seq, sheet = _order(monkeypatch, pool, after, env_chess, 1.0, 9.0, budget_cost=1e-9)          # one action fits the budget
    assert seq == ["step"] and sheet.stale and all(x["status"] == "stale, not re-read" for x in sheet.stale)
    pool2, after = _woken_pool(env_chess)
    seq2, sheet2 = _order(monkeypatch, pool2, after, env_chess, 9.0, 1.0, budget_cost=1e-9)
    assert seq2 == ["recheck"] and len(sheet2.stale) == len(pool2.agenda) and sheet2.steps == []


def test_a_woken_fact_under_the_stake_floor_is_not_rechecked_and_is_reported_stale(env_chess):
    pool, after = _woken_pool(env_chess)
    n = len(pool.agenda)
    sheet = Q.run(after, WORTH, env=env_chess, max_steps=20, pool=pool, light=True, shared=True, params=Q.Params(recheck_floor=1000.0))
    assert sheet.refreshed["asked"] == 0 and len(sheet.stale) == n and pool.queue_reread == 0
    assert all(x["last"] in (Q.CONFIRMED_V, Q.REFUTED_V, Q.WAITING_V, Q.UNSETTLED_V) for x in sheet.stale)


def test_a_refuted_way_whose_own_places_are_untouched_is_settled_by_comparing_cells_not_asked_again(env_chess):
    env = env_chess
    pool = Q.Pool(env.codec, answerers="all")
    pool.advance(KING_ROUTE)
    Q.run(KING_ROUTE, WORTH, env=env, max_steps=60, pool=pool)
    pool.advance(KING_AFTER)                                                # only a unit of the target's side moved: read by the conservative rule, not by the refutation
    assert any(w["role"] == "reads" for w in pool.wake_log)
    full0 = pool.full_reasked
    sheet = Q.run(KING_AFTER, WORTH, env=env, max_steps=60, pool=pool, shared=True)
    cheap = [x for x in pool.refreshed if x.get("cheap")]
    assert pool.cheap_settled >= 1 and cheap and all(x["old"] == Q.REFUTED_V and x["new"] == Q.REFUTED_V for x in cheap)
    assert not any(f.superseded for f in pool.store.facts.values()) and sheet.stale == []
    off = Q.Pool(env.codec, answerers="all")
    off.cheap = False
    off.advance(KING_ROUTE)
    Q.run(KING_ROUTE, WORTH, env=env, max_steps=60, pool=off)
    off.advance(KING_AFTER)
    Q.run(KING_AFTER, WORTH, env=env, max_steps=60, pool=off, shared=True)
    assert off.cheap_settled == 0 and off.full_reasked >= pool.cheap_settled and pool.full_reasked - full0 < off.full_reasked
