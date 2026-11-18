"""PLAN 78.94 (SYSTEM_SPEC 7 "the future imagination layer": what a goal is, DECIDED 2026-11-19; 8 the threat list; 5 the line reading): A GOAL, READ FROM THE CONCRETE THREAT LIST. Game-free: a kind, a place or a move is a word or a number read from the
engine at run time; nothing of any game is named here. No oracle, no worth assumed: a stake is what the line reading says is won (threat_concrete's net, in the measured record's worths).

WHAT A GOAL IS (Kaveh 2026-11-19): two or more threats of one side that no one reply meets. A threat is a row of the concrete threat list (threat_concrete.concrete_lists) of that side: CONFIRMED by the line reading, not cut by a fuse, not a threat that needs a
quiet first move (those are counted apart). The replies that meet a threat are its kept responses (the frame read's answers and the sweep of every legal move). A pair of threats on two different units whose sets of replies have nothing in common is a
goal. KIND: "open" when at least one of the two has no reply at all (a unit that is simply won, with another threat beside it), "fork" when each has replies and no reply is in both.

INTERFACE (written before the bodies).
  Goal                      a dict {"first": the move (a pair of place numbers) of the taking line of the threat with the larger stake, "threats": [{"kind": the game's word for the unit's kind, "place": its place number, "net": the line's net, "replies": the number of replies}, two or more],
                            "kind": "open" | "fork", "stake": the least net of the pair, the greatest over the pairs, "shape": the kind words of the pair ordered by the record's worth, the larger first, "n_pairs": how many pairs qualify, "questions": 1}
  goals_of_list(c, side, record)   `c` one position's concrete list (threat_concrete.concrete_lists), `side` "mover" or "other": the side whose goals are asked (the threats AGAINST the other side's units). Returns {"goals": [Goal, best first], "threats": n of rows that count, "unread": n of rows cut or
                            needing a quiet move first, "forced": True when the forcing read finds a forced end of the game for that side, "deferred": n of threats in two moves}.
  goals_of(items, lines, record)   items as threat_concrete.concrete_lists takes them; returns per item {"mover": ..., "other": ..., "list": c} (the call that does the reading; called by future_labels and by the explicit run against the future judge).
  Called by: future_labels.py (the labels), future_measure.py (the explicit run), tests.
"""
from __future__ import annotations

from itertools import combinations

from catspace.incubator.analyst.discovery import threat_concrete as TC


def _replies(row: dict) -> frozenset:
    return frozenset(tuple(x["first"]) for x in row.get("response", []))


def counting_rows(c: dict, side: str) -> "tuple[list, int, int]":
    """The rows against the OTHER side's units that count as threats of `side`: (rows, unread, deferred). `side` is "mover" or "other"; the rows against the units of the opposite list."""
    against = "other" if side == "mover" else "mover"
    rows, unread, deferred = [], 0, 0
    for r in c[against]:
        if r.get("goal"):
            continue
        if r.get("two_moves") or r.get("deferred"):
            deferred += 1
        elif r["verdict"] == "CONFIRMED":
            rows.append(r)
        else:
            unread += 1
    return rows, unread, deferred


def goals_of_list(c: dict, side: str, record: TC.Record) -> dict:
    rows, unread, deferred = counting_rows(c, side)
    against = "other" if side == "mover" else "mover"
    forced = any(r.get("goal") for r in c[against])
    pairs = []
    for a, b in combinations(rows, 2):
        if a["place_ix"] == b["place_ix"]:
            continue
        ra, rb = _replies(a), _replies(b)
        if ra & rb:
            continue
        net_a = a["net"] if a["net"] is not None else 0
        net_b = b["net"] if b["net"] is not None else 0
        big = a if net_a >= net_b else b
        pairs.append({"first": tuple(big["first"]) if big.get("first") is not None else None,
                      "threats": [{"kind": x["kind"], "place": x["place_ix"], "net": x["net"], "replies": len(_replies(x))} for x in (a, b)],
                      "kind": "open" if (not ra or not rb) else "fork", "stake": min(net_a, net_b),
                      "shape": sorted((a["kind"], b["kind"]), key=lambda k: (-record.worth.get(k, 0), k)), "questions": 1})
    pairs.sort(key=lambda g: (-g["stake"], g["kind"]))
    for g in pairs:
        g["n_pairs"] = len(pairs)
    return {"goals": pairs, "threats": len(rows), "unread": unread, "forced": forced, "deferred": deferred}


def goals_of(items: list, lines: TC.Lines, record: "TC.Record | None" = None) -> list:
    rec = record or lines.record
    cc = TC.concrete_lists(items, lines, rec)
    return [{"mover": goals_of_list(c, "mover", rec), "other": goals_of_list(c, "other", rec), "list": c} for c in cc]
