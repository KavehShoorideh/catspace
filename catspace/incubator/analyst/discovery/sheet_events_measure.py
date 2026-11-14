"""PLAN 78.42, MEASURE I2: the sheet's memory with listeners (discovery/sheet_events.py), on the light reading, at every ply of whole games. Two passes, both from a real script file:

  AUDIT pass (`--events N`, across the cores): per game, a kept sheet (the pool carried from ply to ply, events and the agenda drained at each move) against a fresh light sheet of the same position.
      Every fact the kept sheet served is compared with the fresh sheet's verdict for the same way: the differences are listed with what changed since the fact was filed. The categories of
      asks, the waiting records (woken, killed), the store's size, the agenda's length after each move, the frames' cache simulated on the fresh readings.
  TIME pass (`--events-time N`, ONE process, nothing else running): ms a position of the fresh sheet, of the kept sheet without and with the drain of the agenda, and under budgets of step cost.

  .venv/bin/python -m catspace.incubator.analyst.discovery.sheet_events_measure chess --games-dir DIR --events 16 [--answerers near|all|bearers] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import sys
import time
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import question_sheet as QS

DANGER = QS.DANGER
HORIZON = 8
BUDGETS = (0.05, 0.2)
DECIDED = ("confirmed", "lost", "waiting", "unsettled", "refuted")


def _games(directory: str, n: int) -> list:
    return [json.loads(x) for x in (Path(directory) / "games.jsonl").read_text().splitlines() if x.strip()][:n]


def _kinds(sheet) -> dict:
    """{(place, owner): verdict kind} of the sheet's questions of units."""
    return {(QS.place_text(q.target.at), q.owner): QS.verdict_kind(q) for q in sheet.questions if q.target is not None and not q.quiet}


class FrameCache:
    """The simulation of a cache of single exchange-frame results by their read set (the place, the units that bear on it and on one another, the goal units and what bears on them, the side to
    move), across the positions of a game. The real solve is always run; a hit is counted when the key was seen, with the time the solve took (what a hit would have saved), and the value is compared
    with the cached one. The bookkeeping (building the key, the lookups) is timed on every call."""

    def __init__(self, codec, goal_kind):
        self.codec, self.goal_kind = codec, goal_kind
        self.cache: dict = {}
        self.calls = self.hits_same = self.hits_earlier = self.wrong = 0
        self.t_solve = self.t_saved = self.t_book = 0.0
        self.ply = 0

    def key(self, position: str, place: str) -> tuple:
        c = self.codec
        places = {place} | c.bearers(position, place)
        if self.goal_kind is not None:
            for side, kind, at in c.cells(position):
                if kind == self.goal_kind:
                    gp = QS.place_text(at)
                    places |= {gp} | c.bearers(position, gp)
        cell = c.cell_reader(position)
        return (place, tuple(cell(p) for p in sorted(places)), c.context(position))

    def wrap(self, orig):
        def solve(r, frame, state, label, params, source):
            place = getattr(frame, "place", None)
            t0 = time.perf_counter()
            res = orig(r, frame, state, label, params, source)
            real = time.perf_counter() - t0
            if place is None or not isinstance(state, str):
                return res
            t1 = time.perf_counter()
            k = self.key(state, place)
            got = self.cache.get(k)
            self.calls += 1
            self.t_solve += real
            if got is not None:
                if got[0] == self.ply:
                    self.hits_same += 1
                else:
                    self.hits_earlier += 1
                self.t_saved += real
                self.wrong += abs(got[1] - res["value"]) > 1e-9
            else:
                self.cache[k] = (self.ply, res["value"])
            self.t_book += time.perf_counter() - t1
            return res
        return solve


def _changed_roles(codec, prev_pos: str, pos: str, places, owner: str, asker: str) -> list:
    """The changed places named by what stood there in the later position (for the cause of a difference)."""
    cell = codec.cell_reader(pos)
    out = []
    for p in sorted(places):
        c = cell(p)
        role = "empty" if c[2] is None else ("owner's unit" if c[1] == owner else "asker's unit")
        out.append(f"{p} ({role})")
    return out


def explain_difference(env, worth, params, light: bool, pos_filed: str, pos_now: str, f) -> str:
    """THE CAUSE of a wrong serve under the first form's read set: which rule, in order, would have flagged the fact between the position it was filed on and the one it was served on. context: what no place
    holds (the asker's base position: the side to move, the rights, whether the pass is given); routes: the same position read along another set of routes; then the places a way depends on, in widening
    sets (`way_reads` with answerers 'bearers', 'near', 'all'); none: no read set of places flags it."""
    import dataclasses
    codec, ctx = env.codec, f.payload["ctx"]
    r = QS.Reading(env, pos_filed, worth, light=light)
    sides = {r.absolute(QS.SELF): QS.SELF, r.absolute(QS.OTHER): QS.OTHER}
    mk = lambda spec: QS.Unit(sides[spec[0]], spec[1], spec[2])
    t, u = mk(ctx["t"]), mk(ctx["unit"])
    way = (ctx["way"][0], u, ctx["way"][1], ctx["way"][2])
    routes = [dataclasses.replace(rt, unit=QS.Unit(sides[a], rt.unit.kind, rt.unit.at)) for a, rt in ctx["routes"]]
    a_pool, b_pool = QS.Pool(codec), QS.Pool(codec)
    a_pool.variant = b_pool.variant = env.variant
    if a_pool.cell(pos_filed)(QS.CONTEXT + ctx["asker"]) != b_pool.cell(pos_now)(QS.CONTEXT + ctx["asker"]):
        return "context: the asker's base position differs (the side to move, the rights, or the pass refused)"
    ca, cb = codec.cell_reader(pos_filed), codec.cell_reader(pos_now)
    if pos_filed == pos_now:
        return "routes: the same position, the way read along another set of routes (the old key did not name them)"
    _all, own, _a = QS.way_reads(r, t, way, routes, params, "bearers")
    if any(ca(p) != cb(p) for p in own):
        return "places: a place the route stands on or passes over (a line shut or opened between the steps)"
    for mode, label in (("bearers", "units that bear on a place of the route, the target or a goal unit (a pin, an exchange's supply)"),
                        ("near", "a unit of the target's side that could move onto a place read"), ("all", "another unit of the target's side")):
        reads, _p, _a = QS.way_reads(r, t, way, routes, params, mode)
        if any(ca(p) != cb(p) for p in reads):
            return f"places: {label}"
    return "none: no read set of places flags it"


def audit_game(spec: dict) -> dict:
    """One game: the kept sheet at every ply against the fresh sheet. -> a dict of counts and records (everything printed later is read from it)."""
    env = QS.Env(spec["game"])
    light = spec.get("light", True)
    worth, params, codec = QS.store_worth(spec["game"]), QS.Params(), env.codec
    g = spec["record"]
    v, start, moves = g.get("variant") or env.variant, g.get("start"), g["moves"]
    by_ply, taken = codec.tracks(start, moves)
    events = codec.events(start, moves)
    r0 = QS.Reading(env, CE.state(v, start, [])[0], worth, light=light)
    worth_of = r0.worth_of
    lowest = r0.lowest_worth()
    goal_kind = r0.goal_kind()
    pool = QS.Pool(codec, answerers=spec["answerers"])
    fcache = FrameCache(codec, goal_kind)
    orig_solve = QS._solve
    wrapped = fcache.wrap(orig_solve)
    out = {"game": spec["index"], "plies": 0, "diffs": [], "served": 0, "differ_word": 0, "differ_value": 0, "agenda": [], "left": [], "stale_served": 0, "asked_by_event": 0, "used_by_question": 0,
           "kept_untouched": 0, "kept_revalidated": 0, "event_reasked": 0, "question_reread": 0, "new": 0, "other_turn": 0, "positions": 0, "unreadable": 0,
           "graph_nodes": 0, "ways_filed": 0, "size_live": [], "agree": 0, "agree_n": 0, "agree2": 0, "agree2_n": 0, "refreshed": [], "kills": [], "wake_roles": {}, "dropped": 0, "flagged": 0}
    prev_pos = None
    diffs_by_ply: dict = {}
    try:
        for ply in range(len(moves)):
            pos = CE.state(v, start, moves[:ply])[0]
            try:
                fresh_pool = QS.Pool(codec, answerers=spec["answerers"])
                fcache.ply = ply
                fresh_pool.advance(pos)
                QS._solve = wrapped                                     # the frames' cache is simulated on the fresh readings only
                try:
                    fresh = QS.read_position(env, worth, params, pos, fresh_pool, light=light)
                finally:
                    QS._solve = orig_solve
            except ValueError:
                out["unreadable"] += 1
                prev_pos = pos
                continue
            if prev_pos is not None:
                diffs_by_ply[ply] = codec.changed_places(prev_pos, pos)
            prev_pos = pos
            n_served0, n_ref0, n_wake0 = len(pool.served_log), len(pool.refreshed), len(pool.wake_log)
            before = (pool.kept, pool.revalidated, pool.reread, pool.new, pool.other_turn)
            pool.advance(pos)
            out["agenda"].append(len(pool.agenda))
            sheet = QS.read_position(env, worth, params, pos, pool, light=light, refresh=math.inf)
            out["positions"] += 1
            out["graph_nodes"] += sheet.graph.size()[0]
            pool.file_graph(sheet)
            out["size_live"].append(len(pool.store.facts))
            out["left"].append(len(sheet.stale))
            ref_now = pool.refreshed[n_ref0:]
            new_fids = {x["fid"] for x in ref_now if x.get("fid") is not None}
            out["asked_by_event"] += sum(1 for x in ref_now if x["new"] != "already current")
            served = pool.served_log[n_served0:]
            for s in served:
                if s["state"] == "revalidated":
                    out["kept_revalidated"] += 1
                elif s["fid"] in new_fids:
                    out["used_by_question"] += 1
                else:
                    out["kept_untouched"] += 1
            out["event_reasked"] += sum(1 for x in ref_now if x["new"] not in ("already current",))
            out["question_reread"] += pool.reread - before[2]
            out["new"] += pool.new - before[3]
            out["other_turn"] += pool.other_turn - before[4]
            out["flagged"] = pool.flagged
            for x in ref_now:
                x = dict(x)
                x["ply"] = ply
                out["refreshed"].append({k: x[k] for k in ("ply", "key", "role", "old", "new", "value") if k in x})
            for w in pool.wake_log[n_wake0:]:
                out["wake_roles"][w["role"]] = out["wake_roles"].get(w["role"], 0) + 1
            # the audit: every served fact against the fresh sheet's verdict for the same way
            fresh_by_key = {f.key: f for f in fresh_pool.ways()}
            cell_now = codec.cell_reader(pos)
            for s in served:
                f = pool.store.facts.get(s["fid"])
                if f is None:
                    continue
                out["served"] += 1
                if f.id in pool.store.stale:
                    out["stale_served"] += 1
                fr = fresh_by_key.get(s["key"])
                if fr is None:
                    continue
                a, b = f.payload["verdict"], fr.payload["verdict"]
                word_differs = a["verdict"] != b["verdict"]
                value_differs = (not word_differs) and abs(float(a["value"]) - float(b["value"])) > 1e-9
                out["differ_word"] += word_differs
                out["differ_value"] += value_differs
                if word_differs or value_differs:
                    filed = s["filed"] if s["filed"] is not None else ply
                    since = set()
                    for i in range(filed + 1, ply + 1):
                        since |= diffs_by_ply.get(i, set())
                    owner = pool.store.facts[s["fid"]].payload["ctx"]["t"][0]
                    asker = pool.store.facts[s["fid"]].payload["ctx"]["asker"]
                    out["diffs"].append({"game": spec["index"], "ply": ply, "filed": filed, "state": s["state"], "key": list(map(str, s["key"][:5])), "stored": [a["verdict"], a["value"], a.get("answer", "")[:90]],
                                         "fresh": [b["verdict"], b["value"], b.get("answer", "")[:90]], "changed_since": _changed_roles(codec, None, pos, since, owner, asker),
                                         "read_places": sorted(p for p in f.reads if not p.startswith(QS.CONTEXT)), "in_reads": sorted(since & set(f.reads)),
                                         "cause": explain_difference(env, worth, params, light, CE.state(v, start, moves[:filed])[0], pos, f)})
            # the sheet against the fresh sheet: the same verdict kinds
            ka, kb = _kinds(sheet), _kinds(fresh)
            for k, val in kb.items():
                out["agree_n"] += 1
                out["agree"] += ka.get(k) == val
                if val in DECIDED and ka.get(k) in DECIDED:                # both sheets reached level 2 on the question (the schedule is by measured cost, so which questions get there varies)
                    out["agree2_n"] += 1
                    out["agree2"] += ka.get(k) == val
    finally:
        QS._solve = orig_solve
    out["plies"] = len(moves)
    out["ways_filed"] = len(pool.born_log)
    out["waiting_filed"] = sum(1 for x in pool.born_log if x[2] == QS.WAITING_V)
    out["size_final"] = len(pool.store.facts)
    out["size_max"] = pool.size_max
    out["size_ever"] = pool.store.next_id - 1
    out["killed"] = pool.killed
    out["kills"] = pool.kill_log
    out["dropped"] = pool.dropped
    out["frames"] = {"calls": fcache.calls, "hits_same": fcache.hits_same, "hits_earlier": fcache.hits_earlier, "wrong": fcache.wrong, "t_solve": fcache.t_solve, "t_saved": fcache.t_saved, "t_book": fcache.t_book}
    # the outcomes of woken and confirmed records: the unit LOST within the horizon
    weighty = lambda kind: worth_of(kind) > lowest
    lost = lambda uid, ply: (lambda res: res is not None and res["status"] == QS.LOST)(QS.unit_loss(events, uid, ply, HORIZON, worth_of))
    out["outcomes"] = {"woken_confirmed": [0, 0], "woken_other": [0, 0], "asked_confirmed": [0, 0], "all_weighty": [0, 0], "filed_confirmed": [0, 0]}
    for x in out["refreshed"]:
        place = x["key"][0]
        got = by_ply[x["ply"]].get(place) if x["ply"] < len(by_ply) else None
        if got is None or x["ply"] + HORIZON > len(moves) or x["new"] in ("unit gone", "no base", "no context", "already current"):
            continue
        slot = "woken_confirmed" if (x["role"] == "wake" and x["new"] == QS.CONFIRMED_V) else None
        if slot:
            out["outcomes"][slot][0] += 1
            out["outcomes"][slot][1] += lost(got[0], x["ply"])
        if x["new"] == QS.CONFIRMED_V:
            out["outcomes"]["asked_confirmed"][0] += 1
            out["outcomes"]["asked_confirmed"][1] += lost(got[0], x["ply"])
    for (ply, key, word, _val) in pool.born_log:
        got = by_ply[ply].get(key[0]) if 0 <= ply < len(by_ply) else None
        if got is not None and word == QS.CONFIRMED_V and ply + HORIZON <= len(moves):
            out["outcomes"]["filed_confirmed"][0] += 1
            out["outcomes"]["filed_confirmed"][1] += lost(got[0], ply)
    for ply in range(0, max(0, len(moves) - HORIZON)):
        for place, (uid, _side, kind) in by_ply[ply].items():
            if weighty(kind):
                out["outcomes"]["all_weighty"][0] += 1
                out["outcomes"]["all_weighty"][1] += lost(uid, ply)
    return out


def _pct(a, b) -> str:
    return "n/a" if not b else f"{100.0 * a / b:.1f}%"


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def format_audit(results: list, answerers: str, seconds: float, light: bool = True) -> list:
    sm = lambda k: sum(r[k] for r in results)
    out = [f"OBS (I2) {'light' if light else 'full'} reading, {len(results)} games, {sm('positions')} positions read (every ply; {sm('unreadable')} unreadable), read sets: answerers '{answerers}', {seconds:.0f} s on all cores"]
    asks = sm("kept_untouched") + sm("kept_revalidated") + sm("used_by_question") + sm("question_reread") + sm("new")
    out.append(f"OBS (I2) ways asked for by the questions {asks}: served untouched (filed on an earlier ply, no flag) {sm('kept_untouched')} ({_pct(sm('kept_untouched'), asks)}); served after the cheap comparison (flagged, cells equal) "
               f"{sm('kept_revalidated')} ({_pct(sm('kept_revalidated'), asks)}); served from a fact the move's event had asked again {sm('used_by_question')} ({_pct(sm('used_by_question'), asks)}); "
               f"asked again because a question asked (flagged, cells changed, the agenda had not) {sm('question_reread')} ({_pct(sm('question_reread'), asks)}); new {sm('new')} ({_pct(sm('new'), asks)}, of them a fact stood for the other side to move {sm('other_turn')})")
    out.append(f"OBS (I2) the agenda: facts asked again from it by events {sm('event_reasked')}; facts flagged by moves {sm('flagged')}; unbudgeted drain, so left stale not re-read after the drain: {sum(sum(r['left']) for r in results)}")
    ag = [x for r in results for x in r["agenda"]]
    out.append(f"OBS (I2) the agenda's length after each move (before the drain): mean {_mean(ag):.1f}, max {max(ag) if ag else 0}; wakes by role: "
               + ", ".join(f"{k} {sum(r['wake_roles'].get(k, 0) for r in results)}" for k in ("break", "wake", "route", "refuter", "reads")))
    out.append(f"OBS (I2) AUDIT every served fact against a fresh light sheet of the same position: served {sm('served')}, verdict differs in {sm('differ_word')} ({_pct(sm('differ_word'), sm('served'))}), same verdict but another value in "
               f"{sm('differ_value')}; served while flagged stale {sm('stale_served')}")
    diffs = [x for r in results for x in r["diffs"]]
    causes: dict = {}
    for d in diffs:
        causes[d["cause"]] = causes.get(d["cause"], 0) + 1
    for c, n in sorted(causes.items(), key=lambda x: -x[1]):
        out.append(f"OBS (I2) AUDIT CAUSE {n} differences: {c}")
    out.append(f"OBS (I2) AUDIT of the differences, with a changed place inside the read set: {sum(1 for d in diffs if d['in_reads'])} (a fact whose read set held a changed place was flagged, so was never served kept)")
    for d in diffs[:12]:
        out.append(f"OBS (I2) AUDIT DIFFERENCE game {d['game']} ply {d['ply']} (filed ply {d['filed']}, served {d['state']}) way {d['key']}: stored {d['stored']} against fresh {d['fresh']}; cause: {d['cause']}; changed since filed: "
                   f"{', '.join(d['changed_since'][:14])}; of them in the read set: {d['in_reads']}; read places {d['read_places']}")
    ev = {}
    for r in results:
        for x in r["refreshed"]:
            if x["role"] == "wake":
                ev[x["new"]] = ev.get(x["new"], 0) + 1
    nw = sum(r["wake_roles"].get("wake", 0) for r in results)
    out.append(f"OBS (I2) waiting records: woken by their condition {nw}; asked again from the agenda {sum(ev.values())}, answers: "
               + (", ".join(f"{k} {v}" for k, v in sorted(ev.items())) or "none") + f"; killed by their breaks {sm('killed')}; waiting facts filed {sm('waiting_filed')}")
    oc = {k: [sum(r['outcomes'][k][0] for r in results), sum(r['outcomes'][k][1] for r in results)] for k in results[0]["outcomes"]}
    for k, label in (("woken_confirmed", "woken by their condition and confirmed when asked again"), ("asked_confirmed", "any fact asked again from the agenda and confirmed"), ("filed_confirmed", "every confirmed way filed"),
                     ("all_weighty", "the base rate: every unit above the lowest worth at every ply")):
        out.append(f"OBS (I2) the unit LOST within {HORIZON} plies (unit_loss), {label}: {oc[k][1]} of {oc[k][0]} ({_pct(oc[k][1], oc[k][0])})")
    for r in results:
        out.append(f"OBS (I2) store size, game {r['game']} ({r['plies']} plies): the first form would hold {r['ways_filed'] + r['graph_nodes']} facts (ways {r['ways_filed']} + graph nodes {r['graph_nodes']}, nothing dropped); "
                   f"now filed in all {r['size_ever']}, live at the end {r['size_final']}, most live at one time {r['size_max']}, dropped {r['dropped']}")
    fr = {k: sm_f for k, sm_f in ((k, sum(r['frames'][k] for r in results)) for k in results[0]["frames"])}
    out.append(f"OBS (I2) frames cache (simulated, the real solve always run): solves {fr['calls']}, key seen on the same ply {fr['hits_same']} ({_pct(fr['hits_same'], fr['calls'])}), seen on an earlier ply {fr['hits_earlier']} "
               f"({_pct(fr['hits_earlier'], fr['calls'])}), cached value differs from the real one {fr['wrong']}; solve time {fr['t_solve']:.2f} s, a hit would have saved {fr['t_saved']:.2f} s "
               f"(of which on earlier plies' keys not counted apart), the bookkeeping (keys and lookups on every solve) cost {fr['t_book']:.2f} s")
    out.append(f"OBS (I2) the kept sheet's verdict kinds against the fresh sheet's, per question: equal in {sm('agree')} of {sm('agree_n')} ({_pct(sm('agree'), sm('agree_n'))})")
    out.append(f"OBS (I2) the same, on the questions both sheets took to a verdict at level 2 or deeper (the schedule is by measured cost, so the two sheets do not always reach the same questions): equal in {sm('agree2')} of {sm('agree2_n')} ({_pct(sm('agree2'), sm('agree2_n'))})")
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- the time pass

def time_game(spec: dict) -> dict:
    env = QS.Env(spec["game"])
    light = spec.get("light", True)
    worth, params, codec = QS.store_worth(spec["game"]), QS.Params(), env.codec
    g = spec["record"]
    v, start, moves = g.get("variant") or env.variant, g.get("start"), g["moves"]
    pools = {"nodrain": QS.Pool(codec, answerers=spec["answerers"]), "drain": QS.Pool(codec, answerers=spec["answerers"]), "b50": QS.Pool(codec, answerers=spec["answerers"]),
             "b200": QS.Pool(codec, answerers=spec["answerers"]), "f50": None, "f200": None}
    res = {k: [] for k in ("fresh", "nodrain", "drain", "b50", "b200", "f50", "f200")}
    danger = {k: [0, 0] for k in ("b50", "b200", "f50", "f200")}           # [fresh danger verdicts, of them held]
    held_by_pool = {"b50": 0, "b200": 0}
    left = {"b50": [], "b200": []}
    agree = [0, 0]
    for ply in range(len(moves)):
        pos = CE.state(v, start, moves[:ply])[0]
        t0 = time.perf_counter()
        try:
            fresh = QS.read_position(env, worth, params, pos, None, light=light)
        except ValueError:
            continue
        res["fresh"].append(time.perf_counter() - t0)
        kf = _kinds(fresh)
        fd = {k for k, x in kf.items() if x in DANGER}
        for name, refresh in (("nodrain", None), ("drain", math.inf)):
            t0 = time.perf_counter()
            pools[name].advance(pos)
            sh = QS.read_position(env, worth, params, pos, pools[name], light=light, refresh=refresh)
            res[name].append(time.perf_counter() - t0)
            if name == "drain":
                kd = _kinds(sh)
                for k, x in kf.items():
                    if x in DECIDED and kd.get(k) in DECIDED:
                        agree[0] += 1
                        agree[1] += kd.get(k) == x
        for name, b in (("b50", 0.05), ("b200", 0.2)):
            t0 = time.perf_counter()
            pools[name].advance(pos)
            sh = QS.run(pos, worth, env=env, params=params, pool=pools[name], light=light, refresh=b, budget_cost=b)
            res[name].append(time.perf_counter() - t0)
            kk = _kinds(sh)
            mine = {k for k, x in kk.items() if x in DANGER}
            danger[name][0] += len(fd)
            danger[name][1] += len(fd & mine)
            left[name].append(len(sh.stale))
            valid = {f.key[0] for f in pools[name].valid_ways(pos) if f.kind == "claim"}
            held_by_pool[name] += len({k for k in fd - mine if k[0] in valid})
        for name, b in (("f50", 0.05), ("f200", 0.2)):                      # the fresh sheet under the same budget: no memory
            t0 = time.perf_counter()
            sh = QS.run(pos, worth, env=env, params=params, light=light, budget_cost=b)
            res[name].append(time.perf_counter() - t0)
            mine = {k for k, x in _kinds(sh).items() if x in DANGER}
            danger[name][0] += len(fd)
            danger[name][1] += len(fd & mine)
    r0 = QS.Reading(env, CE.state(v, start, [])[0], worth, light=light)
    fcache = FrameCache(codec, r0.goal_kind())                          # the frames' cache, simulated again with nothing else running (the solves' own times are then quiet times)
    orig = QS._solve
    QS._solve = fcache.wrap(orig)
    try:
        for ply in range(len(moves)):
            fcache.ply = ply
            try:
                QS.read_position(env, worth, params, CE.state(v, start, moves[:ply])[0], None, light=light)
            except ValueError:
                pass
    finally:
        QS._solve = orig
    return {"game": spec["index"], "times": res, "danger": danger, "held": held_by_pool, "left": left, "agree": agree,
            "frames": {"calls": fcache.calls, "hits_same": fcache.hits_same, "hits_earlier": fcache.hits_earlier, "wrong": fcache.wrong, "t_solve": fcache.t_solve, "t_saved": fcache.t_saved, "t_book": fcache.t_book}}


def format_time(results: list, answerers: str, light: bool = True) -> list:
    t = {k: [x for r in results for x in r["times"][k]] for k in results[0]["times"]}
    out = [f"OBS (I2) TIME one process, nothing else running, {'light' if light else 'full'} reading, {sum(len(r['times']['fresh']) for r in results)} positions, answerers '{answerers}'"]
    for k, label in (("fresh", "a fresh light sheet at every position"), ("nodrain", "the kept sheet (events flag, questions ask for what they need)"), ("drain", "the kept sheet with the agenda drained after each move"),
                     ("b50", "kept, budget 50 ms of step cost (the drain included)"), ("b200", "kept, budget 200 ms"), ("f50", "fresh, budget 50 ms (no memory)"), ("f200", "fresh, budget 200 ms (no memory)")):
        out.append(f"OBS (I2) ms a position, {label}: mean {1000 * _mean(t[k]):.1f}, max {1000 * max(t[k]):.1f}")
    fr = {k: sum(r['frames'][k] for r in results) for k in results[0]["frames"]}
    out.append(f"OBS (I2) TIME frames cache (simulated, the real solve always run, nothing else running): solves {fr['calls']}, key seen on the same ply {fr['hits_same']} ({_pct(fr['hits_same'], fr['calls'])}), seen on an earlier ply "
               f"{fr['hits_earlier']} ({_pct(fr['hits_earlier'], fr['calls'])}), cached value differs from the real one {fr['wrong']}; the solves took {fr['t_solve']:.2f} s, the hits would have saved {fr['t_saved']:.2f} s, "
               f"the bookkeeping (building the key and the lookups, on every solve) cost {fr['t_book']:.2f} s")
    a = [sum(r["agree"][0] for r in results), sum(r["agree"][1] for r in results)]
    out.append(f"OBS (I2) the kept sheet with the drain against the fresh sheet: the same verdict kind for {a[1]} of {a[0]} questions both took to a verdict at level 2 or deeper ({_pct(a[1], a[0])})")
    for name in ("b50", "b200", "f50", "f200"):
        d = [sum(r["danger"][name][0] for r in results), sum(r["danger"][name][1] for r in results)]
        extra = ""
        if name in ("b50", "b200"):
            h = sum(r["held"][name] for r in results)
            extra = f"; the pool also holds a confirmed way, valid now, for {h} more ({_pct(d[1] + h, d[0])} together); left stale not re-read after the drain: mean {_mean(x for r in results for x in r['left'][name]):.1f} a position, max {max(x for r in results for x in r['left'][name])}"
        out.append(f"OBS (I2) danger verdicts (live at level 0, lost, confirmed) of the fresh full sheet that the sheet under budget {name} holds: {d[1]} of {d[0]} ({_pct(d[1], d[0])}){extra}")
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- the shared agenda against drain-first against fresh

SHARED_BUDGETS = (0.02, 0.05, 0.1, 0.2, None)
ROWS = ("a fresh", "b kept, drain first", "c kept, one agenda")


def _danger(sheet) -> dict:
    return {(QS.place_text(q.target.at), q.owner): float(q.stake) for q in sheet.questions if q.target is not None and not q.quiet and QS.verdict_kind(q) in DANGER}


def shared_game(spec: dict) -> dict:
    """One game, every ply: the fresh sheet, the kept sheet with the drain first, the kept sheet with the single shared agenda, each under each budget of step cost (and unbudgeted), against the fresh
    unbudgeted sheet. One process, nothing else running (the milliseconds are real)."""
    env = QS.Env(spec["game"])
    light = spec.get("light", True)
    worth, codec = QS.store_worth(spec["game"]), env.codec
    chances = spec.get("chances")
    p_a, p_b, p_c = QS.Params(), QS.Params(chances=chances), QS.Params(chances=chances, clock="chance")
    g = spec["record"]
    v, start, moves = g.get("variant") or env.variant, g.get("start"), g["moves"]
    by_ply, _taken = codec.tracks(start, moves)
    events = codec.events(start, moves)
    r0 = QS.Reading(env, CE.state(v, start, [])[0], worth, light=light)
    worth_of = r0.worth_of
    pools = {(row, b): QS.Pool(codec, answerers=spec["answerers"]) for row in ("b", "c") for b in SHARED_BUDGETS}
    acc = {(row, b): {"held": 0, "held_n": 0, "w_held": 0.0, "w_n": 0.0, "pool_held": 0, "pool_w": 0.0, "stale": 0, "stale_stake": 0.0, "ms": [], "served": 0, "wrong": 0,
                      "lost": {"mine": [0, 0, 0], "theirs": [0, 0, 0]}} for row in ROWS for b in SHARED_BUDGETS for row in (row[0],)}
    for ply in range(len(moves)):
        pos = CE.state(v, start, moves[:ply])[0]
        try:
            fpool = QS.Pool(codec, answerers=spec["answerers"])
            fpool.advance(pos)
            ref = QS.read_position(env, worth, p_a, pos, fpool, light=light)
        except ValueError:
            continue
        D = _danger(ref)
        fresh_by_key = {f.key: f.payload["verdict"]["verdict"] for f in fpool.ways()}
        mover = codec.context(pos)[0]
        lost = {}
        for q in ref.questions:
            if q.target is None or q.quiet or q.goal or not ref.reading.weighty(q.target):
                continue
            got = by_ply[ply].get(QS.place_text(q.target.at))
            if got is None:
                continue
            res = QS.unit_loss(events, got[0], ply, 4, worth_of)
            lost[(QS.place_text(q.target.at), q.owner)] = bool(res and res["status"] == QS.LOST)
        for b in SHARED_BUDGETS:
            for row in ("a", "b", "c"):
                pool = pools.get((row, b))
                t0 = time.perf_counter()
                if pool is not None:
                    pool.advance(pos)
                n0 = len(pool.served_log) if pool is not None else 0
                if row == "a":
                    sh = QS.read_position(env, worth, p_a, pos, None, light=light) if b is None else QS.run(pos, worth, env=env, params=p_a, light=light, budget_cost=b)
                elif row == "b":
                    sh = (QS.read_position(env, worth, p_b, pos, pool, light=light, refresh=math.inf) if b is None
                          else QS.run(pos, worth, env=env, params=p_b, pool=pool, light=light, refresh=b, budget_cost=b))
                else:
                    sh = QS.run(pos, worth, env=env, params=p_c, pool=pool, light=light, shared=True, budget_cost=b)
                    if b is None:
                        QS.finish_levels(sh, 1, p_c)
                        QS.settle_flagged(sh, p_c)
                ms = time.perf_counter() - t0
                a = acc[(row, b)]
                a["ms"].append(ms)
                DR = _danger(sh)
                a["held"] += len(set(D) & set(DR))
                a["held_n"] += len(D)
                a["w_held"] += sum(D[k] for k in set(D) & set(DR))
                a["w_n"] += sum(D.values())
                if pool is not None:
                    valid = {(f.key[0], "us" if f.key[1] == mover else "them") for f in pool.valid_ways(pos) if f.kind == "claim"}
                    extra = {k for k in D if k not in DR and k in valid}
                    a["pool_held"] += len(extra)
                    a["pool_w"] += sum(D[k] for k in extra)
                    a["stale"] += len(sh.stale)
                    a["stale_stake"] += sum(float(x.get("stake") or 0.0) for x in sh.stale)
                    for sv in pool.served_log[n0:]:
                        a["served"] += 1
                        fr = fresh_by_key.get(sv["key"])
                        f = pool.store.facts.get(sv["fid"])
                        if fr is not None and f is not None:
                            a["wrong"] += f.payload["verdict"]["verdict"] != fr
                for k, was_lost in lost.items():
                    slot = a["lost"]["mine" if k[1] == QS.SELF else "theirs"]
                    if was_lost:
                        slot[0] += 1
                        slot[1] += k in DR
                        slot[2] += (k in DR) or (pool is not None and k in {(f.key[0], "us" if f.key[1] == mover else "them") for f in pool.valid_ways(pos) if f.kind == "claim"})
    pc = pools[("c", None)]
    pb = pools[("b", None)]
    out = {"game": spec["index"], "acc": {f"{k[0]}|{k[1]}": v_ for k, v_ in acc.items()}, "cheap": [pc.cheap_settled, pc.full_reasked], "cheap_b": [pb.cheap_settled, pb.full_reasked], "wakes": [], "outcome": [0, 0]}
    for w in pb.wake_log:
        if w["role"] == "wake" and w["ply"] + 8 <= len(moves):
            got = by_ply[w["ply"]].get(w["key"][0])
            if got is not None:
                res = QS.unit_loss(events, got[0], w["ply"], 8, worth_of)
                out["outcome"][0] += 1
                out["outcome"][1] += bool(res and res["status"] == QS.LOST)
    return out


def format_shared(results: list) -> list:
    out = [f"OBS (I2) SHARED AGENDA, one process, nothing else running, {len(results)} games, every ply; rows: a = a fresh light sheet in the present order, b = the kept sheet with the drain first, c = the kept sheet with one agenda "
           f"(chance x stake at risk over cost for rechecks and steps, the table fit1112 for b and c)"]
    for b in SHARED_BUDGETS:
        for row, label in (("a", "fresh"), ("b", "kept, drain first"), ("c", "kept, one agenda")):
            acc = [r["acc"][f"{row}|{b}"] for r in results]
            sm = lambda k: sum(x[k] for x in acc)
            ms = [m for x in acc for m in x["ms"]]
            name = "unbudgeted" if b is None else f"{int(b * 1000)} ms"
            line = (f"OBS (I2) budget {name}, {label}: danger verdicts of the fresh unbudgeted sheet held as current {sm('held')} of {sm('held_n')} ({_pct(sm('held'), sm('held_n'))}), weighted by stake "
                    f"{sm('w_held'):.0f} of {sm('w_n'):.0f} ({_pct(sm('w_held'), sm('w_n'))})")
            if row != "a":
                line += f"; with the pool's valid confirmed ways {sm('held') + sm('pool_held')} ({_pct(sm('held') + sm('pool_held'), sm('held_n'))}), weighted {sm('w_held') + sm('pool_w'):.0f} ({_pct(sm('w_held') + sm('pool_w'), sm('w_n'))})"
            lm = [sum(x["lost"][side][0] for x in acc), sum(x["lost"][side][1] for x in acc), sum(x["lost"][side][2] for x in acc)] if False else None
            for side in ("mine", "theirs"):
                n_ = sum(x["lost"][side][0] for x in acc)
                h_ = sum(x["lost"][side][1] for x in acc)
                hp = sum(x["lost"][side][2] for x in acc)
                line += f"; weighty units LOST within 4 plies, {side}: {h_} of {n_} with a current danger verdict ({_pct(h_, n_)})" + (f", {hp} with the pool's ways ({_pct(hp, n_)})" if row != "a" else "")
            if row != "a":
                line += f"; facts left stale {sm('stale')}, summed stake {sm('stale_stake'):.0f}; served {sm('served')}, wrong against the fresh unbudgeted sheet {sm('wrong')}"
            line += f"; real ms a position: mean {1000 * _mean(ms):.1f}, max {1000 * max(ms):.1f}"
            out.append(line)
    ch = [sum(r["cheap"][0] for r in results), sum(r["cheap"][1] for r in results)]
    out.append(f"OBS (I2) rechecks (kept sheet, one agenda, unbudgeted): settled cheaply (the recorded reason stands, cells compared) {ch[0]}, asked again in full {ch[1]}")
    cb = [sum(r["cheap_b"][0] for r in results), sum(r["cheap_b"][1] for r in results)]
    out.append(f"OBS (I2) rechecks (kept sheet, drain first, unbudgeted): settled cheaply {cb[0]}, asked again in full {cb[1]}")
    oc = [sum(r["outcome"][0] for r in results), sum(r["outcome"][1] for r in results)]
    out.append(f"OBS (I2) a waiting record woken because its waits_on condition came true (the kind 'waiting, condition true', no cell in the table): the unit LOST within 8 plies in {oc[1]} of {oc[0]} ({_pct(oc[1], oc[0])}); until fitted it takes the confirmed kind's chance")
    return out


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--events", type=int, default=0, help="the audit pass over the first N games, across the cores")
    ap.add_argument("--events-time", type=int, default=0, help="the time pass over the first N games, one process")
    ap.add_argument("--events-shared", type=int, default=0, help="the shared-agenda measure over the first N games, one process")
    ap.add_argument("--chances", default=None, help="the table of chances the rows b and c read")
    ap.add_argument("--answerers", default="near", choices=("near", "all", "bearers", "old"))
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--full", action="store_true", help="the full reading (the fact view) instead of the light one")
    ap.add_argument("--out", default=None, help="write the OBS lines here")
    a = ap.parse_args(argv)
    lines = []
    if a.events:
        specs = [{"game": a.game, "index": i, "record": g, "answerers": a.answerers, "light": not a.full} for i, g in enumerate(_games(a.games_dir, a.events))]
        t0 = time.perf_counter()
        with multiprocessing.get_context("spawn").Pool(a.jobs) as p:
            results = []
            for r in p.imap_unordered(audit_game, specs):
                results.append(r)
                print(f"OBS progress: game {r['game']} done, {len(results)} of {len(specs)}, {time.perf_counter() - t0:.0f} s", flush=True)
        results.sort(key=lambda r: r["game"])
        lines += format_audit(results, a.answerers, time.perf_counter() - t0, not a.full)
    if a.events_time:
        specs = [{"game": a.game, "index": i, "record": g, "answerers": a.answerers, "light": not a.full} for i, g in enumerate(_games(a.games_dir, a.events_time))]
        results = []
        for s in specs:
            results.append(time_game(s))
            print(f"OBS progress: timed game {s['index']}, {len(results)} of {len(specs)}", flush=True)
        lines += format_time(results, a.answerers, not a.full)
    if a.events_shared:
        specs = [{"game": a.game, "index": i, "record": g, "answerers": a.answerers, "light": not a.full, "chances": a.chances} for i, g in enumerate(_games(a.games_dir, a.events_shared))]
        results = []
        for sp in specs:
            results.append(shared_game(sp))
            print(f"OBS progress: shared game {sp['index']}, {len(results)} of {len(specs)}", flush=True)
        lines += format_shared(results)
    for ln in lines:
        print(ln, flush=True)
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
