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


def goals_of(items: list, lines: TC.Lines, record: "TC.Record | None" = None, complete: bool = False, ctx: "Complete | None" = None, scope: str = "kept") -> list:
    if complete:
        return goals_of_complete(items, lines, record, ctx, scope)
    rec = record or lines.record
    cc = TC.concrete_lists(items, lines, rec)
    return [{"mover": goals_of_list(c, "mover", rec), "other": goals_of_list(c, "other", rec), "list": c} for c in cc]


# ------------------------------------------------------------------------------------------------------------------------------ PLAN 78.95 PART ONE (a): THE REPLIES, EVERY LEGAL ONE
#
# SYSTEM_SPEC 7 (what a goal is: two threats that no one reply meets), 8 (the threat list), 5 (the line reading). Why: the kept list of replies (the frame read's answers and the sweep's, kept only when they lose nothing larger) is not every
# reply; a reply it does not keep can still meet a threat, so the goals counted from it are an upper count (the main agent read one false alarm in 78.94). HOW IT IS READ HERE, fixed before any count was looked at: for a threat of one side
# (the taker) against a unit of the other (the owner), at the position where the owner is to move with the threat standing (the taker's pass for a goal of the side to move; the position itself for a goal of the other side), EVERY legal move of the
# owner is tried; after it the position is read with the same threat list and line reading, and the reply MEETS the threat when the line reading no longer confirms a taking of that unit (where the unit stands after the reply: it moves with the
# reply when the reply is its own move). A reply that ends the game meets everything. The reply set of a threat is the set of replies that meet it; a pair of threats is a goal when the two reply sets have nothing in common (kind "open" when one set
# is empty, "fork" otherwise). A reply that meets a threat by losing more elsewhere still meets it here: the stake of the loss is the line reading's business, not this test's.
# SCOPE: the pairs the kept-list test already passes (a reply the kept list holds is a reply after which the taking is not confirmed, so every kept reply is in the complete set and a pair the kept test rejects is rejected here too); scope "all"
# tries every pair of confirmed threats on different units, to count the goals the kept test missed (it should be none).
#
# INTERFACE (written before the bodies).
#   Complete(variant, reader, lines, record)   per process: the position reader (threat_list.ThreatReader), the line reader and the record.
#     .reply_sets(item, side, rows) -> ({row index: frozenset of reply words that meet the row's threat}, n_replies)   `item` as threat_concrete.concrete_lists takes it (start, moves, ...), `side` "mover" or "other" (whose goal),
#                                      `rows` the confirmed rows of the threats (counting_rows). One batch read of the position after every legal reply of the owner.
#   complete_of_list(c, side, record, sets, n_replies, rows) -> as goals_of_list plus {"kept_pairs": n, "fell": n of the kept test's pairs that are no goal now, "new": n goals the kept test did not have, "replies": n tried}
#   goals_of(items, lines, record=None, complete=False, ctx=None, scope="kept") -> as before; with complete=True each item's two sides are the complete reading and "kept" holds the kept-list reading of the same item.
#   Called by: future_frames.py (frames), future_measure.py (stage fair), tests.

def move_cells(word: str, places: list) -> "tuple[int, int] | None":
    """The (from, to) place numbers of a move word, read from the place names the engine gave (all of one length): None when the word does not start with two names."""
    if not places:
        return None
    n = len(places[0])
    index = {p: i for i, p in enumerate(places)}
    a, b = index.get(word[:n]), index.get(word[n:2 * n])
    return None if a is None or b is None else (a, b)


class Complete:
    """See the section's doc."""

    def __init__(self, variant: str, reader, lines: TC.Lines, record: TC.Record):
        self.variant, self.reader, self.lines, self.record = variant, reader, lines, record
        self.tried = 0

    def base_text(self, item: dict, side: str) -> "str | None":
        import cs_engine as CE
        text = CE.state(self.variant, item["start"], list(item["moves"]))[0]
        if side == "other":
            return text
        p = CE.passed(self.variant, text, [])
        return None if p is None else p[0]

    def reply_sets(self, item: dict, side: str, rows: list) -> "tuple[dict, int]":
        import cs_engine as CE
        from catspace.incubator.analyst.discovery import threat_list as TL
        base = self.base_text(item, side)
        if base is None or not rows:
            return {i: frozenset() for i in range(len(rows))}, 0
        legal = list(CE.state(self.variant, base, [])[2])
        kids = [CE.state(self.variant, base, [m])[0] for m in legal]
        names = item["names"]
        places = names["places"]
        confirmed: list = [None] * len(kids)
        if kids:
            res = self.reader.read([(t, []) for t in kids], [[0]] * len(kids))
            its, at = [], []
            for r in res["rows"]:
                its.append({"start": kids[r["game"]], "moves": [], "raw": r, "names": names, "list": TL.threat_rows(r, names, None)})
                at.append(r["game"])
            cc = TC.concrete_lists(its, self.lines, self.record) if its else []
            for g, c in zip(at, cc):
                confirmed[g] = {r["place_ix"] for r in counting_rows(c, "mover")[0]}
        self.tried += len(kids)
        out = {}
        for i, row in enumerate(rows):
            p = row["place_ix"]
            met = set()
            for m, conf in zip(legal, confirmed):
                cells = move_cells(m, places)
                q = cells[1] if cells and cells[0] == p else p
                if conf is None or q not in conf:                      # no reading (the game is over after the reply) or no confirmed taking of the unit where it stands
                    met.add(m)
            out[i] = frozenset(met)
        return out, len(legal)


def complete_of_list(c: dict, side: str, record: TC.Record, sets: dict, n_replies: int, rows: list) -> dict:
    """The goals of one side from the complete reply sets; `rows` the confirmed rows that `sets` is indexed by."""
    old = goals_of_list(c, side, record)
    old_keys = {tuple(sorted(t["place"] for t in g["threats"])) for g in old["goals"]}
    pairs, fallen = [], []
    for i, j in combinations(range(len(rows)), 2):
        a, b = rows[i], rows[j]
        if a["place_ix"] == b["place_ix"]:
            continue
        ra, rb = sets[i], sets[j]
        if ra & rb:
            if tuple(sorted((a["place_ix"], b["place_ix"]))) in old_keys:        # a pair the kept test passed and a reply meets both: which reply, for the reader
                fallen.append({"threats": [{"kind": x["kind"], "place": x["place_ix"], "net": x["net"]} for x in (a, b)], "common": sorted(ra & rb)[:8], "n_common": len(ra & rb), "first": [list(x["first"]) if x.get("first") is not None else None for x in (a, b)]})
            continue
        net_a = a["net"] if a["net"] is not None else 0
        net_b = b["net"] if b["net"] is not None else 0
        big = a if net_a >= net_b else b
        pairs.append({"first": tuple(big["first"]) if big.get("first") is not None else None,
                      "threats": [{"kind": x["kind"], "place": x["place_ix"], "net": x["net"], "replies": len(s)} for x, s in ((a, ra), (b, rb))],
                      "kind": "open" if (not ra or not rb) else "fork", "stake": min(net_a, net_b),
                      "shape": sorted((a["kind"], b["kind"]), key=lambda k: (-record.worth.get(k, 0), k)), "questions": 1,
                      "met_by": {"a": sorted(ra)[:6], "b": sorted(rb)[:6]}})
    pairs.sort(key=lambda g: (-g["stake"], g["kind"]))
    for g in pairs:
        g["n_pairs"] = len(pairs)
    now_keys = {tuple(sorted(t["place"] for t in g["threats"])) for g in pairs}
    return {"goals": pairs, "threats": old["threats"], "unread": old["unread"], "forced": old["forced"], "deferred": old["deferred"], "kept_pairs": len(old_keys), "fell": len(old_keys - now_keys), "new": len(now_keys - old_keys), "replies": n_replies,
            "fallen": fallen, "kept_top": old["goals"][0] if old["goals"] else None}


def goals_of_complete(items: list, lines: TC.Lines, record: "TC.Record | None", ctx: Complete, scope: str = "kept") -> list:
    rec = record or lines.record
    cc = TC.concrete_lists(items, lines, rec)
    out = []
    for it, c in zip(items, cc):
        d = {"list": c, "kept": {}}
        for side in ("mover", "other"):
            old = goals_of_list(c, side, rec)
            d["kept"][side] = old
            rows = counting_rows(c, side)[0]
            different = len({r["place_ix"] for r in rows}) >= 2
            if (scope == "kept" and not old["goals"]) or not different:
                d[side] = dict(old, kept_pairs=len(old["goals"]), fell=0, new=0, replies=0)
                continue
            sets, n = ctx.reply_sets(it, side, rows)
            d[side] = complete_of_list(c, side, rec, sets, n, rows)
        out.append(d)
    return out
