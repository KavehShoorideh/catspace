"""PLAN 78.45 THE SHEET ON PUZZLES: does the route reasoning see a prepared attack that is there? A MEASURE: nothing is fitted, no player is changed, nothing of a puzzle reaches the engine or a fit.

A puzzle set is read through `puzzle_apply.load_set`: the position, the answer (the first move), the themes. Per puzzle (the side to move is the solver):
  the Rust sheet under the clock present at the schedule's default budget (40 steps, run.rs) and again at five times (200 steps), seconds recorded;
  SEES   the sheet holds, for the side to move, a claim in danger (the sheet's own DANGER kinds: live0, lost, confirmed) against a unit of the other side (its goal unit included) with a way
         whose FIRST MOVE is the answer. A way's first move is read from the sheet's data: a deep entry's `move`, else the first word of its `line` that is a legal move, else the first step of
         its `route` (the unit's place and the route's first place, as a legal move); a way of one move has the unit's place to its arrival; a way of two or more moves that the sheet lists
         without routes (the scenario ways of a goal question) has the first steps its geometry allows, derived here from the game's medium of movement (units ignored, the same notion as
         the sheet's own routes) and marked as derived; a search-mode finding gives it in the result of the way at the same index; a way not yet settled at a deep
         level has the first steps of its level-1 routes. `confirmed`: a deep way with status confirmed whose first move is the answer. `lost` (a unit standing lost, a way of no moves): the answer
         is a taking on the place of that unit (the first level already shows the unit lost, so the answer that takes it is that claim's way). `live0`: a level-1 route of a way of the claim begins
         with the answer.
  PICKS  a plain chooser fixed before the results: the first move of the top such claim by stake, ties by the sheet's own order (a claim in danger that has a first move to name: its first
         confirmed way's move, for a unit standing lost the first move of its line, for live0 the first step of the first route of its first way). It names the answer or not.
  the move question (`move_sheet`): the verdict on the answer and the number of legal moves called SAFE.
  BASELINE the plain search (the wake's call, `PlainSearch`) at 0.2 s and at the seconds the default sheet took (rounded up to 0.01 s, at least 0.02 s).
  THE LINE the set files hold no solution line, so one is made by the plain search: the answer, then its best replies and the solver's best moves, `--line-ply` plies at `--line-seconds` each.
  THE TARGET from the line: their goal unit when the line ends the game in a decision after the solver's move, else the unit of the other side of the highest worth the solver takes in it
  (earliest on a tie), else none (class `other`).
THE MISSES (the sheet does not SEE at the default budget), each in one class:
  T not a question on the sheet at all; R the target is on the sheet but no way begins with the answer (no route of any of its ways starts with it; sub-kinds named); X such a way exists and is
  refuted; W waiting; N not settled or not reached; other. O is seen but outranked (SEES, not PICKS); it is not a miss.
PLAN 78.46 adds a second column, the FORCING READ (`SheetModel.forcing`, rust/engine/tactics/src/sheet/forcing.rs), read once more per puzzle (seconds recorded):
  SEES2   the sheet SEES (as above) OR the forcing read holds a CONFIRMED threat move equal to the answer.
  PICKS2  a plain chooser fixed before results: the CONFIRMED threat move of highest value (ties: the engine's own order, by the move's name), else the sheet's own pick; it names the answer or not.
  BASELINE at 0.2 s and at the seconds the sheet and the forcing read together took (rounded up to 0.01 s).
  THE MISSES LEFT (not SEES2): the classes above for the sheet, and F = the forcing read read the answer as a threat move and did not confirm it (NO CLAIM: the reply that closed it; NOT SETTLED:
  the weakest reply so far; UNREAD: the fuse stopped before it), with whether the plain-search line plays that reply.
Run: python -m catspace.incubator.analyst.discovery.puzzle_sheet_measure --variant <game> --sets dev_300 [--limit 20] [--fuse N]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import random
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_STEPS = None            # the schedule's own default (40 steps)
BIG_STEPS = 200                 # five times the default
FIRST_STEPS = None              # set per process by `_init`: (kind, place, arrival, moves) -> the places one move of the unit can go to that are still `moves - 1` moves from the arrival (units ignored)
STATUS_RANK ={"confirmed": 0, "waiting": 1, "open": 2, "unreached": 2, "unsettled": 2, "not settled": 2, "refuted": 3}
LENGTH_TAGS = ("oneMove", "short", "long", "veryLong")


def danger_kinds() -> tuple:
    from catspace.incubator.analyst.discovery import question_sheet as QS
    return tuple(QS.DANGER)


# ------------------------------------------------------------------------------------------------------------------------------- reading the sheet's data (pure functions)

def _step_moves(at: str, place: str, legal: list, ends) -> list:
    return [m for m in legal if ends(m) == (at, place)]


def _first_move(e: dict, legal: list, ends) -> "str | None":
    """The first move of a frame entry's way: its `move`, else the first word of its line that is a legal move, else the unit's place and the route's first place."""
    if e.get("move") in legal:
        return e["move"]
    line = e.get("line") or []
    if line and line[0] in legal:
        return line[0]
    way, route = e.get("way"), e.get("route") or []
    if way is not None and route:
        got = _step_moves(way[2], route[0], legal, ends)
        if got:
            return got[0]
    return None


def _deep(q: dict) -> "dict | None":
    return next((f for f in reversed(q["findings"]) if f["level"] >= 2), None)


def _level1(q: dict) -> "dict | None":
    f = next((f for f in q["findings"] if f["level"] == 1), None)
    return None if f is None else f["detail"]


def way_table(q: dict, legal: list, ends) -> dict:
    """The ways of one question as plain data: {ways: [{way, firsts (the first steps of its level-1 routes), places (the places of those routes), deep: None | {status, move, stops, ...}}],
    stand (the exchange where the unit stands, a way of no moves): None | {status, line, moves, first}, lost (the same when confirmed), mode}. A way is named by the tuple the sheet gives it."""
    lv1 = _level1(q) or {}
    table: dict = {}
    for way, routes in lv1.get("routes") or []:
        way = tuple(way)
        rec = table.setdefault(way, {"way": way, "firsts": set(), "places": set(), "deep": None})
        for r in routes:
            if r[2]:
                rec["firsts"].update(_step_moves(r[1], r[2][0], legal, ends))
                rec["places"].update(r[2])
    for way in lv1.get("ways") or []:
        table.setdefault(tuple(way), {"way": tuple(way), "firsts": set(), "places": set(), "deep": None})
    d = _deep(q)
    lost, stand = None, None
    if d is not None:
        det = d["detail"]
        if d["mode"] == "frame":
            for e in det.get("results") or []:
                if e["way"] is None:
                    target_place = q["target"][2] if q.get("target") else None
                    taking = [m for m in legal if ends(m)[1] == target_place]
                    stand = {"status": e["status"], "line": list(e.get("line") or []), "moves": taking, "first": next((m for m in (e.get("line") or []) if m in legal), None)}
                    if e["status"] == "confirmed":
                        lost = stand
                    continue
                rec = table.setdefault(tuple(e["way"]), {"way": tuple(e["way"]), "firsts": set(), "places": set(), "deep": None})
                rec["deep"] = {"status": e["status"], "move": _first_move(e, legal, ends), "stops": list(e.get("stops") or []), "time": e.get("time"), "answer": e.get("answer"),
                               "line": list(e.get("line") or []), "obstacles": list(e.get("obstacles") or []), "without_stopper": e.get("without_stopper"), "route": e.get("route")}
        elif d["mode"] == "search":
            ways, results = det.get("ways") or [], det.get("results") or []
            for w, r in zip(ways, results):
                rec = table.setdefault(tuple(w), {"way": tuple(w), "firsts": set(), "places": set(), "deep": None})
                mv, status, value, reply = r
                rec["deep"] = {"status": status, "move": mv if mv in legal else None, "stops": [(True, reply, 0, None, "search_refuted", [], None)] if status == "refuted" else [], "time": None,
                               "answer": reply, "line": [], "obstacles": [], "without_stopper": None, "route": None}
    for rec in table.values():       # a way of one move has one first step, the unit's place to the arrival, whether or not level 1 listed its routes
        w = rec["way"]
        if w[0] == 1:
            rec["firsts"].update(_step_moves(w[2], w[3], legal, ends))
            rec["places"].add(w[3])
        if w[0] >= 2 and not rec["firsts"] and FIRST_STEPS is not None:       # a way the sheet listed without routes (the scenario ways of a goal question): the first steps its geometry allows
            for place in FIRST_STEPS(w[1], w[2], w[3], w[0]):
                rec["firsts"].update(_step_moves(w[2], place, legal, ends))
            rec["derived"] = bool(rec["firsts"])
            rec["routes_read"] = True                  # the geometry was asked: a way none of whose first steps is a legal move does not begin with any move
            continue
        rec["routes_read"] = bool(rec["firsts"]) or w[0] <= 1
    return {"ways": list(table.values()), "lost": lost, "stand": stand, "mode": None if d is None else d["mode"]}


def claim_moves(q: dict, legal: list, ends, kinds: tuple) -> list:
    """The first moves a claim in danger names, in the sheet's order of its ways; [] when it is no claim in danger or names none."""
    kind = q["key"][1]
    if q["asker"] != "us" or kind not in kinds or not q.get("target") or q["target"][0] != "them":
        return []
    t = way_table(q, legal, ends)
    if kind == "lost":
        return list(t["lost"]["moves"]) if t["lost"] else []
    if kind == "confirmed":
        return [w["deep"]["move"] for w in t["ways"] if w["deep"] and w["deep"]["status"] == "confirmed" and w["deep"]["move"]]
    return sorted({m for w in t["ways"][:1] for m in w["firsts"]})[:1] if t["ways"] else []


def seeing(questions: list, answer: str, legal: list, ends, kinds: "tuple | None" = None) -> dict:
    """SEES and PICKS on one sheet. -> {sees, via (the kinds of the claims that see), claims (targets), pick, picks, picks_any}."""
    kinds = kinds or danger_kinds()
    seen, via, pool = [], [], []
    for i, q in enumerate(questions):
        mv = claim_moves(q, legal, ends, kinds)
        if not mv:
            continue
        kind = q["key"][1]
        if kind == "lost":
            first = (way_table(q, legal, ends)["lost"] or {}).get("first")
            pool.append((-q["stake"], i, first or mv[0]))
        elif kind == "confirmed":
            pool.append((-q["stake"], i, mv[0]))
        else:
            pool.append((-q["stake"], i, mv[0]))
        if kind == "live0":
            t = way_table(q, legal, ends)
            hit = any(answer in w["firsts"] for w in t["ways"])
        else:
            hit = answer in mv
        if hit:
            seen.append(tuple(q["target"]))
            via.append(kind)
    pool.sort()
    pick = pool[0][2] if pool else None
    return {"sees": bool(seen), "via": via, "claims": seen, "pick": pick, "picks": bool(seen) and pick == answer, "picks_any": bool(pool)}


def forcing_view(fr: "dict | None", answer: str) -> "dict | None":
    """What the forcing read says about a puzzle: the confirmed threat moves (in the engine's order: by value, then by name), the answer's own account (its verdict, value, the reply that held it)."""
    if fr is None:
        return None
    conf = [(t["move"], t["value"]) for t in fr["threats"] if t["verdict"] == "CONFIRMED"]
    view = {"confirmed_moves": conf, "pick": conf[0][0] if conf else None, "answer_confirmed": any(m == answer for m, _v in conf), "answer": None,
            **{k: fr[k] for k in ("found", "confirmed", "unsettled", "no_claim", "unread", "nodes", "cut", "level")}}
    t = next((t for t in fr["threats"] if t["move"] == answer), None)
    if t is not None:
        held = None
        if t["verdict"] == "NO CLAIM" and t["replies"]:
            held = t["replies"][-1]                                     # the reply that closed it (the read stops at it)
        elif t["verdict"] == "NOT SETTLED" and t["replies"]:
            held = min(t["replies"], key=lambda r: (r["value"], not r["open"]))
        view["answer"] = {"verdict": t["verdict"], "value": t["value"], "level": t["level"], "replies": len(t["replies"]), "claims": [(c["target"], c["kind"], c["value"]) for c in t["claims"]],
                          "held_by": held["reply"] if held else None, "held_value": held["value"] if held else None, "held_open": held["open"] if held else None,
                          "held_line": held["line"] if held else None, "held_took": held["took"] if held else None}
    return view


def seeing2(seen: dict, view: "dict | None", answer: str) -> dict:
    """SEES2 and PICKS2 from the sheet's own `seeing` result and the forcing view."""
    by_forcing = bool(view and view["answer_confirmed"])
    pick = view["pick"] if view and view["pick"] is not None else seen["pick"]
    return {"sees2": bool(seen["sees"]) or by_forcing, "by_forcing": by_forcing, "pick2": pick, "picks2": pick == answer}


def forcing_class(view: "dict | None", sheet_class: str, sees2: bool) -> str:
    """The class of a miss left after the forcing read: SEEN2 | F (the answer was read as a threat move and not confirmed) | the sheet's class (T R X W N other)."""
    if sees2:
        return "SEEN2"
    if view and view["answer"] is not None:
        return "F"
    return sheet_class


def forcing_lines(view: "dict | None") -> list:
    if view is None:
        return ["forcing read: not read"]
    out = [f"forcing read: threat moves found {view['found']}, confirmed {view['confirmed']} {view['confirmed_moves'][:6]}, not settled {view['unsettled']}, no claim {view['no_claim']}, unread {view['unread']}; "
           f"positions read {view['nodes']}, fuse stopped it {view['cut']}, deepest level {view['level']}"]
    a = view["answer"]
    out.append("the answer is not a threat move of the forcing read" if a is None else
               f"the answer: {a['verdict']} value {a['value']} level {a['level']} claims {a['claims']}; held by {a['held_by']} (value {a['held_value']}, open {a['held_open']}, line {a['held_line']}, took {a['held_took']})")
    return out


def _stop_dict(s) -> dict:
    hard, ans, after, step, reason, places, mover = s
    return {"hard": hard, "answer": ans, "after": after, "step": step, "reason": reason, "places": list(places), "mover": mover}


def classify(questions: list, answer: str, target: "tuple | None", legal: list, ends, took: list, line: list, kinds: "tuple | None" = None) -> dict:
    """The class of a miss (the sheet did not SEE): T R X W N or other, with the sheet's own account. `target`: (side, kind, place) as the sheet names units; `took`: the units the answer takes
    (the move question's own list); `line`: the solution line (the answer first)."""
    if target is None:
        return {"cls": "other", "why": "no target derived from the line (it takes nothing and ends no game in a decision)"}
    qs = [q for q in questions if q["asker"] == "us" and q.get("target") and tuple(q["target"]) == tuple(target)]
    if not qs:
        return {"cls": "T", "why": f"no question about {target} on the sheet"}
    q = qs[0]
    t = way_table(q, legal, ends)
    own = [w for w in t["ways"] if answer in w["firsts"] or (w["deep"] and w["deep"]["move"] == answer)]
    st = t["stand"]
    if st is not None and answer in st["moves"] and not any(w["deep"] and w["deep"]["move"] == answer for w in own):
        own = own + [{"way": (0, "stand", target[2], target[2], "strike"), "firsts": {answer}, "places": set(), "stand": True,
                      "deep": {"status": st["status"], "move": answer, "stops": [], "time": None, "answer": None, "line": st["line"], "obstacles": [], "without_stopper": None, "route": None}}]
    base = {"target": list(target), "verdict": q["key"][1], "key": list(q["key"]), "level": q["level"], "state": q["state"], "stake": q["stake"], "ways": len(t["ways"])}
    if not own:
        blind = [w for w in t["ways"] if not w["routes_read"]]
        if blind:
            return dict(base, cls="other", sub="ways of two or more moves whose routes the sheet did not list (level 1 gave no first steps), so it cannot be said whether one begins with the answer",
                        why=f"{len(blind)} way(s) of {target} with no routes listed")
        if not t["ways"]:
            sub = "the target's question lists no way at all (closed or idle before any way was kept)"
        elif any(tuple(u[:3]) != tuple(target) for u in took):
            sub = "takes another unit (not toward the target)"
        elif any(ends(answer)[1] in w["places"] for w in t["ways"]):
            sub = "lands on a place of some route but begins none"
        else:
            sub = "a quiet move that lands on no route place of the target's ways (a clearing, deflecting or threatening move the routes cannot express)"
        return dict(base, cls="R", sub=sub, why=f"no way of {target} has a route beginning with {answer}")
    own.sort(key=lambda w: STATUS_RANK.get(w["deep"]["status"] if w["deep"] else "unreached", 2))
    w = own[0]
    d = w["deep"]
    base.update(way=list(w["way"]), way_status=d["status"] if d else "unreached", way_first_move=d["move"] if d else None, route_differs=bool(d and d["move"] != answer), first_step_derived=bool(w.get("derived")))
    if d is None:
        return dict(base, cls="N", why="a route begins with the answer but the way has no deep reading (not reached)")
    stops = [_stop_dict(s) for s in d["stops"]]
    opp = line[1::2]
    for s in stops:
        s["answer_played_in_line"] = s["answer"] in opp if s["answer"] else None
    base["stops"] = stops
    if w.get("stand"):
        base["line_of_the_exchange"] = d["line"]
    if d["status"] == "refuted":
        return dict(base, cls="X", why="the way is refuted" + (" (by another route than the answer's)" if base["route_differs"] else ""), line_reply_matches=[s["answer_played_in_line"] for s in stops])
    if d["status"] == "waiting":
        return dict(base, cls="W", why="the way waits", time=d["time"], obstacles=[str(o)[:200] for o in d["obstacles"]][:3])
    if d["status"] in ("open", "unsettled", "not settled"):
        return dict(base, cls="N", why="the way is not settled")
    return dict(base, cls="other", why=f"the way is {d['status']} by a first move other than the answer ({d['move']}) and the question is not seen" if base["route_differs"] else f"the way is {d['status']} but the question is not in danger")


# ------------------------------------------------------------------------------------------------------------------------------- the line and its target

def line_target(units0: list, line: list, ends, units_of_line: "list | None" = None, decided_after_solver: bool = False, goal_target: "tuple | None" = None) -> "tuple | None":
    """The intended target from a line (the answer first, the solver's moves at even places). `units0`: the other side's units before the line as (kind, place, worth). The highest worth taken
    by the solver (earliest on a tie), followed through the other side's moves; else `goal_target` when the line ends the game in a decision after a solver move; else None."""
    live = [[k, p, w, p] for k, p, w in units0]           # kind, place now, worth, place at the start
    best = None
    for i, m in enumerate(line):
        origin, dest = ends(m)
        if i % 2 == 0:
            hit = next((u for u in live if u[1] == dest), None)
            if hit is not None:
                live.remove(hit)
                if best is None or hit[2] > best[2]:
                    best = hit
        else:
            mover = next((u for u in live if u[1] == origin), None)
            if mover is not None:
                mover[1] = dest
    if decided_after_solver and goal_target is not None:
        return goal_target                                    # the game is won by the line: the win outranks the material it takes on the way
    return ("them", best[0], best[3]) if best is not None else None


# ------------------------------------------------------------------------------------------------------------------------------- one puzzle

_W: dict = {}


def _init(variant: str) -> None:
    from catspace.incubator.analyst.discovery import game_store as GS
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.incubator.analyst.discovery import sheet_player as SP
    from catspace.incubator.analyst.discovery import sheet_rust as SR
    env = QS.Env(variant)
    worth = QS.store_worth(variant)
    head = GS.head(variant)
    store = str(GS.path_of(variant, head)) if head else None
    specs = GS.spec_texts(variant) if store else None
    trace = str(Path(tempfile.gettempdir()) / f"puzzle_sheet_trace_{os.getpid()}.jsonl")
    _W.update(variant=variant, env=env, QS=QS, SR=SR, model=SR.model(env, worth), search=SP.PlainSearch(variant, store=store, specs=specs, trace_path=trace), mover=None)
    global FIRST_STEPS
    med = env.medium()

    def first_steps(kind: str, place: str, arrival: str, moves: int) -> list:
        return [x for x in med.hops(kind, _W["mover"], place) if med.distances(kind, _W["mover"], x).get(arrival) == moves - 1]
    FIRST_STEPS = first_steps


def _theme_length(themes: list) -> str:
    for t in themes:
        if t.startswith("mateIn"):
            return t[len("mateIn"):] + " moves (a mate)"
    return next((t for t in LENGTH_TAGS if t in themes), "unknown")


def _units(pos: str) -> tuple:
    r = _W["SR"].read(_W["model"], _W["env"], pos)
    return r["mover"], [(u[1], u[2], u[3]) for u in r["units"] if u[0] == "them"], r["units"]


def derive_line(pos: str, answer: str, plies: int, seconds: float, ends) -> tuple:
    """(line, decided_after_solver): the answer, then the plain search's moves for both sides, up to `plies` plies; stops when the game ends."""
    import cs_engine as CE
    variant, search = _W["variant"], _W["search"]
    line = [answer]
    while len(line) < plies:
        _p, _m, legal, outcome, _k = CE.state(variant, pos, line)
        if outcome is not None or not legal:
            break
        mv = search.decide(pos, list(line), seconds)["move"]
        if not mv:
            break
        line.append(mv)
    _p, _m, legal, outcome, _k = CE.state(variant, pos, line)
    decided = outcome is not None and len(line) % 2 == 1 and len(set(outcome)) > 1
    return line, decided


def measure(rec: dict, opts: dict) -> dict:
    import cs_engine as CE
    QS, SR, env, model, search, variant = _W["QS"], _W["SR"], _W["env"], _W["model"], _W["search"], _W["variant"]
    pos, answer = rec["pos"], rec["answer"]
    row = {"id": rec["id"], "set": rec["set"], "answer": answer, "themes": rec["themes"], "length": _theme_length(rec["themes"])}
    ends = env.codec.move_ends
    legal = list(CE.state(variant, pos, [])[2])
    row["legal"] = len(legal)
    _W["mover"] = _units(pos)[0]
    params = QS.Params(compare_old=False, clock="present")
    sheets = {}
    for name, steps in (("default", DEFAULT_STEPS), ("big", BIG_STEPS)):
        t0 = time.perf_counter()
        try:
            sheets[name] = SR.sheet(model, env, pos, params, max_steps=steps)
        except ValueError as e:
            row["error"] = str(e)
            return row
        row[f"seconds_{name}"] = time.perf_counter() - t0
    row["seen"] = {name: seeing(s["questions"], answer, legal, ends) for name, s in sheets.items()}
    # the forcing read (78.46)
    t0 = time.perf_counter()
    try:
        fr = model.forcing(pos, opts.get("fuse"))
    except ValueError:
        fr = None
    row["seconds_forcing"] = time.perf_counter() - t0
    row["forcing"] = forcing_view(fr, answer)
    row["seen2"] = seeing2(row["seen"]["default"], row["forcing"], answer)
    # the move question
    try:
        mq = model.move_question(pos, answer, params.reach_k, params.max_open_exits)
        ms = model.move_sheet(pos, params.reach_k, params.max_open_exits, None)
        row["answer_verdict"] = mq["verdict"]
        row["safe_moves"] = sum(r["verdict"] == "SAFE" for r in ms["results"])
        row["moves_read"] = len(ms["results"])
        took = mq["took_units"]
    except ValueError:
        row["answer_verdict"], row["safe_moves"], took = None, None, []
    # the search baselines
    sec_b = max(0.02, math.ceil(row["seconds_default"] * 100) / 100)
    row["search_seconds_b"] = sec_b
    row["search_a"] = search.decide(pos, [], 0.2)["move"] == answer
    row["search_b"] = search.decide(pos, [], sec_b)["move"] == answer
    sec_c = max(0.02, math.ceil((row["seconds_default"] + row["seconds_forcing"]) * 100) / 100)
    row["search_seconds_c"] = sec_c
    row["search_c"] = search.decide(pos, [], sec_c)["move"] == answer
    # the line and the target
    line = list(rec.get("solution") or [])
    row["line_source"] = "set" if line else "plain search"
    decided = False
    if not line:
        line, decided = derive_line(pos, answer, opts["line_ply"], opts["line_seconds"], ends)
    mover0, other0, _units_all = _units(pos)
    goal_target = next((tuple(q["target"]) for q in sheets["default"]["questions"] if q["asker"] == "us" and q.get("goal") and q.get("target")), None)
    target = line_target(other0, line, ends, None, decided, goal_target)
    row["line"], row["target"], row["line_decided"] = line, list(target) if target else None, decided
    # the classes, at the default and at five times
    for name in ("default", "big"):
        qs = sheets[name]["questions"]
        row[f"class_{name}"] = None if row["seen"][name]["sees"] else classify(qs, answer, target, legal, ends, took, line)
    row["class"] = ("O" if not row["seen"]["default"]["picks"] else "SEEN-PICKED") if row["seen"]["default"]["sees"] else row["class_default"]["cls"]
    row["account"] = account(sheets["default"]["questions"], target)
    a = (row["forcing"] or {}).get("answer")
    if a is not None and a["held_by"]:
        a["held_is_first_reply_of_line"] = len(line) > 1 and a["held_by"] == line[1]
        a["held_played_in_line"] = a["held_by"] in line[1::2]
    row["class2"] = forcing_class(row["forcing"], row["class"], row["seen2"]["sees2"])
    row["forcing_account"] = forcing_lines(row["forcing"])
    return row


def account(questions: list, target: "tuple | None") -> list:
    """The sheet's own account of the target, as plain text lines (its questions' verdicts, ways and stops)."""
    out = []
    for q in questions:
        if q["asker"] != "us" or not q.get("target") or (target is not None and tuple(q["target"]) != tuple(target)):
            continue
        out.append(f"question {tuple(q['target'])} verdict {q['key'][1]} key {q['key']} stake {q['stake']} level {q['level']} state {q['state']}")
        d = _deep(q)
        if d is None:
            lv1 = _level1(q) or {}
            for way, routes in lv1.get("routes") or []:
                out.append(f"   level-1 way {tuple(way)} routes {[r[2] for r in routes]}")
            continue
        if d["mode"] == "frame":
            for e in d["detail"].get("results") or []:
                out.append(f"   way {e['way']} {e['status']} move {e.get('move')} route {e.get('route')} line {e.get('line')} stops {[_stop_dict(s) for s in e.get('stops') or []]}")
        else:
            out.append(f"   search ways {d['detail'].get('ways')} results {d['detail'].get('results')}")
    return out


def run_one(job: tuple) -> dict:
    rec, opts = job
    try:
        return measure(rec, opts)
    except Exception as e:                              # a position the sheet or the search refuses is a row, not a crash
        return {"id": rec["id"], "set": rec["set"], "error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------------------------------------------------------------------- tables

def class_of(r: dict) -> "str | None":
    return r.get("class") if "class" in r else None


def tabulate(rows: list) -> dict:
    rows = [r for r in rows if "class" in r]
    classes = ("T", "R", "X", "W", "N", "other", "O")
    out: dict = {"n": len(rows), "classes": Counter(("O" if r["class"] == "O" else r["class"]) for r in rows if r["class"] != "SEEN-PICKED")}
    out["seen"] = sum(r["seen"]["default"]["sees"] for r in rows)
    out["picked"] = sum(r["seen"]["default"]["picks"] for r in rows)
    out["picks_any"] = sum(r["seen"]["default"]["picks_any"] for r in rows)
    out["seen_big"] = sum(r["seen"]["big"]["sees"] for r in rows)
    out["picked_big"] = sum(r["seen"]["big"]["picks"] for r in rows)
    out["by_theme"], out["by_length"] = defaultdict(Counter), defaultdict(Counter)
    for r in rows:
        key = "SEEN" if r["class"] == "SEEN-PICKED" else r["class"]
        for t in r["themes"]:
            out["by_theme"][t][key] += 1
        out["by_length"][r["length"]][key] += 1
    four = {}
    for b in ("a", "b"):
        four[b] = Counter((bool(r["seen"]["default"]["sees"]), bool(r[f"search_{b}"])) for r in rows)
    out["four"] = four
    out["sheet_yes_search_a_no"] = [r["id"] for r in rows if r["seen"]["default"]["sees"] and not r["search_a"]]
    out["sheet_yes_search_b_no"] = [r["id"] for r in rows if r["seen"]["default"]["sees"] and not r["search_b"]]
    out["verdicts"] = Counter(str(r.get("answer_verdict")) for r in rows)
    out["safe_moves_mean"] = sum(r["safe_moves"] or 0 for r in rows) / max(1, len(rows))
    out["via"] = Counter(v for r in rows for v in set(r["seen"]["default"]["via"]))
    out["n_big_turns"] = sum(1 for r in rows if not r["seen"]["default"]["sees"] and r["seen"]["big"]["sees"])
    out["n_search_a"] = sum(r["search_a"] for r in rows)
    out["n_search_b"] = sum(r["search_b"] for r in rows)
    out["seconds_default"] = sum(r["seconds_default"] for r in rows) / max(1, len(rows))
    out["seconds_big"] = sum(r["seconds_big"] for r in rows) / max(1, len(rows))
    out["classes_all"] = classes
    rows2 = [r for r in rows if "seen2" in r]
    if rows2:
        two = {"n": len(rows2), "sees2": sum(r["seen2"]["sees2"] for r in rows2), "by_forcing": sum(r["seen2"]["by_forcing"] and not r["seen"]["default"]["sees"] for r in rows2),
               "picks2": sum(r["seen2"]["picks2"] for r in rows2), "n_search_c": sum(r["search_c"] for r in rows2),
               "classes2": Counter(r["class2"] for r in rows2 if r["class2"] != "SEEN2"),
               "with_confirmed": sum(1 for r in rows2 if r["forcing"] and r["forcing"]["confirmed"]),
               "confirmed_other": sum(1 for r in rows2 if r["forcing"] and any(m != r["answer"] for m, _v in r["forcing"]["confirmed_moves"])),
               "pick_wrong_confirmed": sum(1 for r in rows2 if r["forcing"] and r["forcing"]["pick"] is not None and r["forcing"]["pick"] != r["answer"]),
               "cut": sum(1 for r in rows2 if r["forcing"] and r["forcing"]["cut"]),
               "seconds_forcing": sum(r["seconds_forcing"] for r in rows2) / len(rows2), "seconds_default": sum(r["seconds_default"] for r in rows2) / len(rows2),
               "four_c": Counter((bool(r["seen2"]["sees2"]), bool(r["search_c"])) for r in rows2), "four_a": Counter((bool(r["seen2"]["sees2"]), bool(r["search_a"])) for r in rows2),
               "F_sub": Counter(), "F_held": Counter(), "sheet_class_of_F": Counter()}
        for r in rows2:
            if r["class2"] != "F":
                continue
            a = r["forcing"]["answer"]
            two["F_sub"][a["verdict"]] += 1
            two["sheet_class_of_F"][r["class"]] += 1
            if a["held_by"]:
                two["F_held"]["the plain-search line's first reply is that reply" if a.get("held_is_first_reply_of_line") else ("the plain-search line plays that reply later" if a.get("held_played_in_line") else "the plain-search line does not play that reply")] += 1
            else:
                two["F_held"]["no reply held it (unread, or not borne out by the search)"] += 1
        out["two"] = two
    out["detail"] = defaultdict(Counter)
    for r in rows:
        c = r.get("class_default")
        if c is None:
            continue
        cl = c["cls"]
        out["detail"][cl][("sub: " + c["sub"][:70]) if "sub" in c else ("way status: " + str(c.get("way_status")) + (", by another route" if c.get("route_differs") else "") + (", first step derived from the medium" if c.get("first_step_derived") else "") if "way_status" in c else "why: " + c["why"][:70])] += 1
        out["detail"][cl]["target verdict: " + str(c.get("verdict"))] += 1
        for s in c.get("stops") or []:
            out["detail"][cl]["stop: " + str(s["reason"]) + (", the reply is played in the line" if s.get("answer_played_in_line") else ", the reply is not in the line")] += 1
    out["turns_by_class"] = Counter(r["class_default"]["cls"] for r in rows if r.get("class_default") and r["seen"]["big"]["sees"])
    out["moves"] = Counter((r["class_default"]["cls"], "SEES" if r["seen"]["big"]["sees"] else (r.get("class_big") or {}).get("cls")) for r in rows if r.get("class_default"))
    return out


def obs_lines(label: str, tab: dict) -> list:
    n = tab["n"]
    pct = lambda x: f"{x} ({100 * x / max(1, n):.1f} pct)"
    L = [f"OBS puzzle sheet {label}: puzzles read {n}; sheet SEES {pct(tab['seen'])} at the default budget, {pct(tab['seen_big'])} at five times; PICKS {pct(tab['picked'])} (names anything for {pct(tab['picks_any'])}); at five times PICKS {pct(tab['picked_big'])}",
         f"OBS puzzle sheet {label}: SEES by kind of claim {dict(tab['via'])}; sheet seconds mean {tab['seconds_default']:.3f} default, {tab['seconds_big']:.3f} five times",
         f"OBS puzzle sheet {label}: the move question on the answer {dict(tab['verdicts'])}; legal moves called SAFE per puzzle (mean) {tab['safe_moves_mean']:.1f}",
         f"OBS puzzle sheet {label}: plain search finds {pct(tab['n_search_a'])} at 0.2 s and {pct(tab['n_search_b'])} at the seconds the sheet took",
         f"OBS puzzle sheet {label}: classes (misses T R X W N other; seen outranked O; seen and picked not listed): " + " ".join(f"{c} {tab['classes'].get(c, 0)}" for c in ("T", "R", "X", "W", "N", "other", "O")),
         f"OBS puzzle sheet {label}: misses the five-times budget turns into SEES {tab['n_big_turns']} (by class {dict(tab['turns_by_class'])})"]
    L.append(f"OBS puzzle sheet {label}: class at the default budget -> at five times: " + "; ".join(f"{a}->{b} {v}" for (a, b), v in sorted(tab["moves"].items(), key=lambda kv: (str(kv[0]), kv[1]))))
    for cl in ("R", "X", "W", "N", "other"):
        for k, v in sorted(tab["detail"].get(cl, {}).items(), key=lambda kv: -kv[1]):
            L.append(f"OBS puzzle sheet {label}: class {cl}: {k}: {v}")
    for b, name in (("a", "0.2 s"), ("b", "sheet seconds")):
        f = tab["four"][b]
        L.append(f"OBS puzzle sheet {label}: four-way (sheet sees / search {name} finds): yes/yes {f[(True, True)]}, yes/no {f[(True, False)]}, no/yes {f[(False, True)]}, no/no {f[(False, False)]}")
    t2 = tab.get("two")
    if t2:
        p2 = lambda x: f"{x} ({100 * x / max(1, t2['n']):.1f} pct)"
        L.append(f"OBS puzzle sheet {label}: SEES {pct(tab['seen'])} SEES2 {p2(t2['sees2'])} (the forcing read adds {t2['by_forcing']}); PICKS {pct(tab['picked'])} PICKS2 {p2(t2['picks2'])}; plain search {pct(tab['n_search_a'])} at 0.2 s, "
                 f"{p2(t2['n_search_c'])} at the seconds the sheet and the forcing read took")
        L.append(f"OBS puzzle sheet {label}: forcing read: puzzles with a confirmed threat move {t2['with_confirmed']}, with one that is not the answer {t2['confirmed_other']}, whose chosen move (highest value) is not the answer {t2['pick_wrong_confirmed']}; "
                 f"the fuse stopped the read in {t2['cut']}")
        L.append(f"OBS puzzle sheet {label}: seconds per puzzle: sheet {t2['seconds_default']:.3f}, forcing read {t2['seconds_forcing']:.3f}, both {t2['seconds_default'] + t2['seconds_forcing']:.3f}")
        for b, name in (("a", "0.2 s"), ("c", "the sheet and forcing seconds")):
            f = t2["four_" + b]
            L.append(f"OBS puzzle sheet {label}: four-way SEES2 (sheet or forcing sees / search {name} finds): yes/yes {f[(True, True)]}, yes/no {f[(True, False)]}, no/yes {f[(False, True)]}, no/no {f[(False, False)]}")
        L.append(f"OBS puzzle sheet {label}: misses left after the forcing read (T R X W N other, F = the forcing read read the answer and did not confirm it): " + " ".join(f"{c} {t2['classes2'].get(c, 0)}" for c in ("T", "R", "X", "W", "N", "other", "F")))
        for k, v in sorted(t2["F_sub"].items(), key=lambda kv: -kv[1]):
            L.append(f"OBS puzzle sheet {label}: class F: the answer is {k}: {v}")
        for k, v in sorted(t2["F_held"].items(), key=lambda kv: -kv[1]):
            L.append(f"OBS puzzle sheet {label}: class F: {k}: {v}")
        for k, v in sorted(t2["sheet_class_of_F"].items(), key=lambda kv: -kv[1]):
            L.append(f"OBS puzzle sheet {label}: class F: the sheet's class of these (before the forcing read): {k}: {v}")
    L.append(f"OBS puzzle sheet {label}: sheet sees and the 0.2 s search does not: {tab['sheet_yes_search_a_no']}")
    for t, c in sorted(tab["by_theme"].items(), key=lambda kv: -sum(kv[1].values())):
        L.append(f"OBS puzzle sheet {label}: theme {t}: n {sum(c.values())} " + " ".join(f"{k} {v}" for k, v in sorted(c.items())))
    for t, c in sorted(tab["by_length"].items()):
        L.append(f"OBS puzzle sheet {label}: solution length {t}: n {sum(c.values())} " + " ".join(f"{k} {v}" for k, v in sorted(c.items())))
    return L


def pick_misses(rows: list, n: int, seed: int) -> list:
    """About n misses spread over the classes: round-robin over the classes present (in a fixed order), a seeded draw inside each."""
    rng = random.Random(seed)
    pools = {c: [r for r in rows if r.get("class2", r.get("class")) == c] for c in ("F", "T", "R", "X", "W", "N", "other", "O")}
    for p in pools.values():
        rng.shuffle(p)
    out = []
    while len(out) < n and any(pools.values()):
        for c in pools:
            if pools[c] and len(out) < n:
                out.append(pools[c].pop())
    return out


def write_misses(rows: list, path: str, n: int, seed: int) -> None:
    with open(path, "w") as f:
        for r in pick_misses(rows, n, seed):
            f.write(f"=== {r['set']} {r['id']}  class {r.get('class2', r['class'])} (the sheet's {r['class']})  themes {r['themes']}  solution length {r['length']}\n")
            f.write(f"position {r.get('pos', '')}\nanswer {r['answer']}   line (plain search after the answer, {r['line_source']}) {r['line']}   target {r['target']}\n")
            f.write(f"sheet SEES {r['seen']['default']['sees']} (five times {r['seen']['big']['sees']}); PICKS {r['seen']['default']['pick']}; the move question on the answer {r.get('answer_verdict')}; safe moves {r.get('safe_moves')} of {r.get('moves_read')}\n")
            c = r.get("class_default")
            if c:
                f.write("reading of the miss: " + json.dumps(c, default=str) + "\n")
            f.write("the forcing read: " + "\n    ".join(r.get("forcing_account", [])) + "\n")
            f.write("the sheet's account of the target:\n" + "\n".join("  " + x for x in r["account"]) + "\n\n")


# ------------------------------------------------------------------------------------------------------------------------------- main

def main(argv: "list | None" = None) -> int:
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    from catspace.io import paths
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", required=True, help="the game's name as the engine knows it")
    ap.add_argument("--sets", default="dev_300")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--seed", type=int, default=4545)
    ap.add_argument("--line-ply", type=int, default=6)
    ap.add_argument("--line-seconds", type=float, default=0.5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fuse", type=int, default=None, help="positions the forcing read may read (default: the engine's FUSE_POSITIONS)")
    a = ap.parse_args(argv)
    opts = {"line_ply": a.line_ply, "line_seconds": a.line_seconds, "fuse": a.fuse}
    t0 = time.perf_counter()
    for name in a.sets.split(","):
        recs = [dict(p, set=name, pos=p["fen"]) for p in PA.load_set(name)]
        if a.limit:
            recs = recs[:a.limit]
        with multiprocessing.get_context("spawn").Pool(a.jobs, initializer=_init, initargs=(a.variant,)) as pool:
            rows = pool.map(run_one, [(r, opts) for r in recs], chunksize=1)
        for r, rec in zip(rows, recs):
            r.setdefault("pos", rec["pos"])
        tag = a.tag or f"{name}{'_' + str(a.limit) if a.limit else ''}"
        out = Path(paths.experiment("puzzle_sheet_78_45", f"{tag}.json"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1, default=str))
        good = [r for r in rows if "class" in r]
        tab = tabulate(good)
        lines = [f"OBS puzzle sheet {tag}: {len(rows)} puzzles, {len(rows) - len(good)} not read (errors); wall {time.perf_counter() - t0:.0f} s, jobs {a.jobs}; line {a.line_ply} plies at {a.line_seconds} s; danger kinds {','.join(danger_kinds())}"]
        lines += obs_lines(tag, tab)
        for line in lines:
            print(line, flush=True)
        for r in rows:
            if "error" in r:
                print(f"OBS puzzle sheet {tag}: not read {r['id']}: {r['error']}", flush=True)
        misses = Path(paths.experiment("puzzle_sheet_78_45", f"{tag}_misses.txt"))
        write_misses(good, str(misses), a.show, a.seed)
        Path(paths.experiment("puzzle_sheet_78_45", f"{tag}_obs.txt")).write_text("\n".join(lines) + "\n")
        print(f"OBS puzzle sheet {tag}: table {out}; {a.show} misses written out {misses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
