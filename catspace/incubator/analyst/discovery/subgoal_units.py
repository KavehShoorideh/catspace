"""PLAN 78.74 A: THE LINE READ FOR EVERY UNIT OF THE OTHER SIDE. SYSTEM_SPEC section 7 "Choosing a subgoal, and getting there" ("this is the material that's the most weak": a goal for every target, not only the targets the threat list names); the piece is the
OPPORTUNITY POLICY'S PROPOSER. A sandbox: nothing calls it in play, no player is changed, nothing is written to rules/learned or data/derived. Game-free: a kind, a role, a place is a word read from the engine at run time; no unit, place or kind is named here.
No engine evaluation, no external model, no assumed worth: a taking is said to work or fail ONLY by a line read by the existing reader (threat_concrete.Lines.ask, the same modes and the same judge, the measured record), never by counting the units that bear.

INTERFACE (what this module takes, what it returns, who calls it).
  every_unit(items, sides, concs, facts, lines, stake_fns)
                  items       as for threat_concrete.concrete_lists: {"start", "moves", "raw", "names", "list"} per position.
                  sides       per item, the absolute sides (0 or 1) whose goals are wanted (the targets are the units of the OTHER side).
                  concs       per item, the concrete list that concrete_lists returned for it. CHANGED IN PLACE: for every unit of the other side that has no row in it (the threat list found no claim on that unit), one row is added, read the way
                              concrete_lists reads a row: the taking lines of the unit (mode 0, the best of them judged by the record); where no taking is legal, the line of a quiet first move that makes a taking stand (mode 3); a row is put where
                              concrete_lists would put it (a taking that works or is cut: the side's list; an exchange: `exchange`; a taking that fails or a quiet move that does not end ahead: `dropped_rows` with the code of why). Each such row has
                              "every": True. The unit of the goal kind gets no row (it cannot be taken: subgoal_goals.goal_unit_goal is its goal).
                  facts       a subgoal_goals.Facts. lines  a threat_concrete.Lines (one per process).
                  stake_fns   per item, callable(kind word, owner is the side to move) -> the stake of the kind (the kinds reading of plan 78.69; None for a kind it does not have), or None.
                  Returns {"units": the units that were read, "reads": the line queries asked, "wall": seconds}.
  Called by: subgoal_measure.py (the grade) and subgoal_print.py (the print) now; the wake's choice of move later. subgoal_goals.goals takes the augmented list as it takes any list: its interface is unchanged.
"""
from __future__ import annotations

import time

import cs_engine as CE

from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import threat_list as TL


def _covered(c: dict) -> set:
    """The (owner, place) pairs the concrete list has a row for (a row of the goal is about the game's end, not about a unit)."""
    out = set()
    for pool in (c["mover"], c["other"], c["exchange"], c["dropped_rows"]):
        for r in pool:
            if not r.get("goal") and r.get("place") is not None:
                out.add((r["owner"], r["place"]))
    return out


def every_unit(items: list, sides: list, concs: list, facts: SG.Facts, lines: TC.Lines, stake_fns: list) -> dict:
    t0 = time.perf_counter()
    if facts.goal_name is None:
        from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
        facts.goal_name = G2M.goal_kind_of(facts.variant)
    rec = lines.record
    games = [(it["start"], list(it["moves"])) for it in items]
    rows = []
    for i, (it, c) in enumerate(zip(items, concs)):
        names = it["names"]
        roles, places = names["roles"], names["places"]
        mover = it["raw"]["mover"]
        text = CE.state(facts.variant, it.get("start"), list(it["moves"]))[0]
        f0 = facts.of(text)
        have = _covered(c)
        for s in sides[i]:
            owner = 1 - s
            for uid, (sd, kind, place) in sorted(f0.units.items()):
                if sd != roles[owner] or kind == facts.goal_name or (owner, place) in have:
                    continue
                st = stake_fns[i](kind, owner == mover) if stake_fns[i] else None
                ix = places.index(place)
                by = [(f0.units[u][1], f0.units[u][2]) for u in f0.bearers(place, roles[s])]
                row = {"side": "mover" if owner == mover else "other", "owner": owner, "goal": False, "kind": kind, "place": place, "place_ix": ix, "class": "unit", "by": by, "stake": st, "response": [], "response_ix": [], "frames": [],
                       "resp_read": 0, "resp_unread": 0, "every": True, "verdict": None, "two_moves": False, "_i": i, "_taker_p": 1 if owner == mover else 0}
                rows.append(row)
    nq = 0
    q1 = [(r["_i"], 0, r["_taker_p"], 1, r["place_ix"], 0, 0) for r in rows]
    r1 = lines.ask(games, q1)
    nq += len(q1)
    two = []
    for row, res in zip(rows, r1):
        if not res["ok"]:
            row.update(verdict="UNREAD", ended=TL.ENDED[3], reason="the side to move must answer: the game gives no pass, so what the other side would take cannot be read", net=None, line=[], lead=None, outcome=None, two_moves=False)
            continue
        best = TC.best_taking(res["reads"], rec) if res["reads"] else None
        if best is None:
            two.append(row)
        else:
            TC.fill_row(row, best, False)
    q2 = [(r["_i"], 3, r["_taker_p"], 1, r["place_ix"], 0, 0) for r in two]
    r2 = lines.ask(games, q2)
    nq += len(q2)
    for row, res in zip(two, r2):
        best = TC.best_taking(res["reads"], rec) if res["ok"] and res["reads"] else None
        if best is not None:
            TC.fill_row(row, best, True)
            if best["verdict"] not in ("CONFIRMED", "UNREAD"):
                row["dropped_verdict"] = row["verdict"]
                row["verdict"] = "DROPPED"
                row["why"] = "two_move_not_ahead"
        else:
            row.update(verdict="DROPPED", why=("no_bearer_no_move" if not row["by"] else "no_taking_legal"), line=[], net=None, outcome=None)
    for row in rows:
        c = concs[row["_i"]]
        v = row["verdict"]
        row = {k: x for k, x in row.items() if not k.startswith("_")}
        if v == "DROPPED":
            c["dropped_rows"].append(row)
        elif v == "REFUTED":
            c["dropped_rows"].append(dict(row, why="refuted"))
        elif v == "EXCHANGE":
            c["exchange"].append(row)
        else:
            c[row["side"]].append(row)
    wall = time.perf_counter() - t0
    for c in concs:
        c["every_wall"] = wall / max(1, len(concs))
    return {"units": len(rows), "reads": nq, "wall": wall}


def stake_closure(stake_row, kind_cols: list):
    """The stake of a kind for one position from the kinds reading's table row (2 owners x kinds; owner index 1 is the side to move): the callable `every_unit` takes. A kind the table does not have gives None."""
    def stake_of(kind, own):
        return float(stake_row[1 if own else 0, kind_cols.index(kind)]) if kind in kind_cols else None
    return stake_of
