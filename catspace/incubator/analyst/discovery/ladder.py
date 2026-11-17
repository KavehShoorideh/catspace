"""PLAN 78.68 (first run: rungs 1 to 3 as a measurement, and the list of open surprises with its groups). SYSTEM_SPEC 9, pieces: THE LADDER ON A POSITION and THE LIST OF SURPRISES ("a rule is scored by how many
surprises it resolves"). A sandbox: no player is changed, nothing is written to the real store. Game-free: a move, a kind, a class is a word that arrives as data; no object of any game is named here.

INTERFACE (what this module takes, what it returns, who calls it).
  ladder(position, best_move, tools, limits) -> dict
      position   {"moves": [move names played from the start]} (the position after them).
      best_move  the set of moves the oracle accepts at the position (the answers file's row). It is read ONLY to decide which rung stops the ladder; it is never copied into the result.
      tools      an object with the four readers of OUR OWN reasoning (the oracle is not among them):
                   search(position, seconds) -> {"move", "equals": [moves the search holds equal to its best], "nodes", "depth", "proved"}   the search with the measured record at a clock
                   forced(position)          -> {"first": a move or None, "kind": "proved" | "suspected" | "none" | "gap"}              the forcing read
                   heads(position, equals)   -> {head name: the move it picks among `equals`}                                          the heads as tie-break (the arrival number first)
                   line(position, scale)     -> {"names": [moves], "events": [(mine, kind)], "end", "depth", "cut", "nodes"}            our deepest reading's best line, at a stated scale of the clock
                   features(position, line)  -> {family: [tokens]}                                                                      what the surprise has in common with others
      limits     {"wake_seconds": the wake's clock (rung 1 and 2), "ladder": the increasing clocks of rung 3, "head": the head that decides rung 2 (default "arrival")}.
      returns    {"rung": 1 | 2 | 3 | 0, "cost": {rung: {"seconds", "nodes", "queries"}}, "chose": {what each rung chose, in moves of OUR tools and the booleans below},
                  "equals": the size of the wake's equal set, "tie": {"eligible", "random_hit", "hits": {head: bool}, "control_hit"}, "first_hit": the search's own first move at the wake's clock is accepted, "found_at": the clock at which rung 3 picked the move, or None,
                  "deep": what the deep reading found (the line of `tools.line` at the last clock reached), "features": the grouping features (rungs 3 and 0 only)}.
      RUNG 1 SHARP: our reasoning alone reaches the move: every move the search holds equal at the wake's clock is accepted, or the forcing read proves a first move that is accepted.
      RUNG 2 NOT SHARP: the search holds several moves equal, the accepted move is among them, and the head named in `limits` picks an accepted one. The other heads and the search's own first move are measured beside it
             (`tie`), and so is the chance of a random pick; they do not decide the rung.
      RUNG 3 A SURPRISE REMAINS BUT DEEPER READING FINDS IT: the search at the next clock of the ladder, in turn, until every move it holds equal is accepted.
      RUNG 0 STILL OPEN at the last clock of the ladder.
  surprise_row(pid, source, moves, result) / write_surprise_list(path, rows)    the list of open surprises (rungs 3 and 0): one row each with the position (its moves), the rung, the costs and the features. A row has only the fields of
      ROW_FIELDS: THE WALL: no oracle move, score, evaluation or depth is in a row (`wall_violations` names any field or text that does).
  groups(rows) -> [{"key": [tokens], "size", "ids"}] largest first: rows grouped by the pair (or, failing a pair, the single token) of features most rows share, greedily.
Called by: ladder_measure.py (the run) and ladder_print.py (the cases). Nothing in play calls it.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

LIMITS = {"wake_seconds": 0.2, "ladder": (1.0, 4.0, 16.0, 64.0), "head": "arrival"}
ROW_FIELDS = ("id", "source", "moves", "rung", "found_at", "cost", "equals", "features", "deep")
DEEP_FIELDS = ("names", "events", "end", "depth", "cut", "nodes")
# any key or word of an oracle's reading: the wall's grep (a key of a row, or a word in the file's text)
FORBIDDEN = re.compile(r"oracle|stockfish|centipawn|\bcp\b|score|eval|accepted|best_move|answer|label|multipv|pv\b", re.I)


def _zero() -> dict:
    return {"seconds": 0.0, "nodes": 0, "queries": 0}


def _timed(cost: dict, fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    cost["seconds"] += time.perf_counter() - t0
    cost["queries"] += 1
    cost["nodes"] += int((r or {}).get("nodes", 0) or 0) if isinstance(r, dict) else 0
    return r


def ladder(position: dict, best_move, tools, limits: "dict | None" = None) -> dict:
    lim = dict(LIMITS, **(limits or {}))
    acc = set(best_move)
    cost = {1: _zero(), 2: _zero(), 3: _zero()}
    chose: dict = {}
    out = {"rung": 0, "cost": cost, "chose": chose, "equals": 0, "tie": {"eligible": False, "random_hit": None, "hits": {}, "control_hit": None}, "found_at": None, "deep": None, "features": None,
           "first_hit": None}

    def done(rung: int) -> dict:
        out["rung"] = rung
        return out

    # rung 1
    s = _timed(cost[1], tools.search, position, lim["wake_seconds"])
    eq = list(s["equals"])
    out["equals"] = len(eq)
    chose["1 search"] = {"move": s["move"], "equals": len(eq), "proved": bool(s.get("proved"))}
    out["first_hit"] = s["move"] in acc                          # PLAN 78.76: the search's own first move at the wake's clock is an accepted move (a measurement; it decides no rung by itself)
    if eq and set(eq) <= acc:
        return done(1)
    f = _timed(cost[1], tools.forced, position)
    chose["1 forced"] = {"kind": f.get("kind"), "first": f.get("first")}
    if f.get("kind") == "proved" and f.get("first") in acc:
        return done(1)
    # rung 2
    if len(eq) >= 2 and acc & set(eq):
        picks = _timed(cost[2], tools.heads, position, eq)
        tie = out["tie"]
        tie["eligible"] = True
        tie["random_hit"] = len(acc & set(eq)) / len(eq)
        tie["hits"] = {h: (m in acc) for h, m in picks.items()}
        tie["control_hit"] = s["move"] in acc
        chose["2 heads"] = dict(picks)
        if tie["hits"].get(lim["head"]):
            return done(2)
    # rung 3
    found = None
    last = None
    for clock in lim["ladder"]:
        r = _timed(cost[3], tools.search, position, clock)
        last = clock
        chose[f"3 search {clock:g} s"] = {"move": r["move"], "equals": len(r["equals"]), "depth": r.get("depth")}
        if r["equals"] and set(r["equals"]) <= acc:
            found = clock
            break
    out["found_at"] = found
    deep = _timed(cost[3], tools.line, position, last)
    out["deep"] = {k: deep.get(k) for k in DEEP_FIELDS}
    out["features"] = tools.features(position, deep)
    return done(3 if found is not None else 0)


# ------------------------------------------------------------------------------------------------------------------------------ the list of open surprises

def surprise_row(pid: str, source: str, moves: list, result: dict) -> dict:
    """The row of the list for a position left open at rungs 1 and 2 (rung 3 or 0). Only ROW_FIELDS."""
    return {"id": pid, "source": source, "moves": list(moves), "rung": result["rung"], "found_at": result["found_at"],
            "cost": {str(k): v for k, v in result["cost"].items() if k == 3 or v["queries"]}, "equals": result["equals"],
            "features": result["features"], "deep": result["deep"]}


def write_surprise_list(path: "str | Path", rows: list) -> None:
    bad = [v for r in rows for v in wall_violations(r)]
    if bad:
        raise ValueError(f"the wall: {bad[:3]}")
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def wall_violations(row: dict) -> list:
    """Fields of a row outside ROW_FIELDS, and any key (at any depth) or string value that reads as the oracle's reading."""
    bad = [f"field {k}" for k in row if k not in ROW_FIELDS]

    def walk(x, where):
        if isinstance(x, dict):
            for k, v in x.items():
                if FORBIDDEN.search(str(k)):
                    bad.append(f"key {where}/{k}")
                walk(v, f"{where}/{k}")
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v, where)
        elif isinstance(x, str) and FORBIDDEN.search(x):
            bad.append(f"text {where}: {x[:40]}")
    walk({k: v for k, v in row.items() if k != "moves"}, "")
    return bad


def tokens_of(row: dict) -> list:
    f = row.get("features") or {}
    toks = [f"rung:{row['rung']}"]
    for fam in sorted(f):
        toks += [f"{fam}:{t}" for t in f[fam]]
    return toks


def groups(rows: list, min_size: int = 2) -> list:
    """Greedy: take the pair of tokens (of different families) that most of the remaining rows share, else the single token; the rows holding it are a group; repeat. Largest first. Rows sharing nothing are one
    group with the key []."""
    left = {r["id"]: set(tokens_of(r)) for r in rows}
    if len(rows) >= 8:                                           # a feature nearly every row has separates nothing: it is not a key (a stated cut: more than three rows in four)
        freq = Counter(t for toks in left.values() for t in toks)
        common = {t for t, n in freq.items() if n > 0.75 * len(rows)}
        left = {i: toks - common for i, toks in left.items()}
    out = []
    while left:
        cnt = Counter()
        for toks in left.values():
            for a, b in combinations(sorted(toks), 2):
                if a.split(":")[0] != b.split(":")[0]:
                    cnt[(a, b)] += 1
        key = None
        if cnt and cnt.most_common(1)[0][1] >= min_size:
            key = cnt.most_common(1)[0][0]
        else:
            single = Counter(t for toks in left.values() for t in toks)
            if single and single.most_common(1)[0][1] >= min_size:
                key = (single.most_common(1)[0][0],)
        if key is None:
            out.append({"key": [], "size": len(left), "ids": sorted(left)})
            break
        ids = sorted(i for i, t in left.items() if set(key) <= t)
        out.append({"key": list(key), "size": len(ids), "ids": ids})
        for i in ids:
            del left[i]
    return sorted(out, key=lambda g: -g["size"])
