"""PLAN 78.53: the readable matcher -- the mirror the engine's pattern counts (rust/engine/tactics/src/sheet/patterns.rs) are tested against. Brute force over a layer-1 `Graph` (graph_layer1.Graph) as the
Python mirror reads it; slow and plain on purpose. Game-free: a kind is the word the graph's `kind:<word>` node carries.

    items, units = graph_items(graph, goal_kind)
    match_counts(items, units, pattern, kinds)        # (count for the side to move, count for the other side)
"""
from __future__ import annotations

from itertools import permutations

from catspace.incubator.analyst.discovery.graph_layer2 import Pattern


def graph_items(g, goal_kind: "str | None") -> "tuple[list, dict]":
    """The atoms of a layer-1 graph of a position and the attributes of its units. -> (items, units): item = (type, actor id, ('u', id) | ('p', place id), line id | None, between id | None);
    units[id] = dict(mine, goal, edge, kind word, place, side). Shut lines with one unit between only, as the engine reads them."""
    mover = g.mover
    units: dict = {}
    for nid in g.of_type("unit"):
        ls = g.neighbours(nid)
        side = [o for lt, _v, _x, o, _f in ls if lt == "unit_side"][0].split(":", 1)[1]
        kind = [o for lt, _v, _x, o, _f in ls if lt == "unit_kind"][0].split(":", 1)[1]
        place = [o for lt, _v, _x, o, _f in ls if lt == "unit_on"][0]
        edge = any(lt == "on_edge" for lt, *_ in g.neighbours(place))
        units[nid] = dict(mine=side == mover, kind=kind, edge=edge, goal=False, place=place, side=side)
    if goal_kind is not None:
        for side in {u["side"] for u in units.values()}:
            of = [i for i, u in units.items() if u["side"] == side and u["kind"] == goal_kind]
            if len(of) == 1:
                units[of[0]]["goal"] = True
    at = {u["place"]: i for i, u in units.items()}
    items = []
    for ty, name in (("bear", "bear"), ("move", "may_move"), ("shut", "shut"), ("through", "through")):
        for nid in g.of_type(name):
            ls = [(lt, o) for lt, _v, _x, o, fwd in g.neighbours(nid) if fwd]
            actor = [o for lt, o in ls if lt == "by"][0]
            place = [o for lt, o in ls if lt == "at_place"][0]
            along = [o for lt, o in ls if lt == "along"]
            between = [o for lt, o in ls if lt == "between"]
            target = ("u", at[place]) if place in at else ("p", place)
            if ty in ("shut", "through"):
                if len(between) != 1 or not along:
                    continue
                items.append((ty, actor, target, along[0], between[0]))
            else:
                items.append((ty, actor, target, along[0] if along else None, None))
    return items, units


def _bind(atoms: list, chosen: tuple) -> "dict | None":
    """The bindings (unit, place, line variables to objects) of the pattern's atoms in this order onto these items, or None; injective both ways."""
    bound: dict = {}
    back: dict = {}

    def put(var: tuple, obj) -> bool:
        if bound.setdefault(var, obj) != obj:
            return False
        return back.setdefault((var[0], obj), var) == var
    for (ty, a, ts, t, ln, b), it in zip(atoms, chosen):
        ity, iactor, itarget, iline, ibetween = it
        if ty != ity or (ln is None) != (iline is None) or (b is None) != (ibetween is None) or itarget[0] != ts:
            return None
        if not put(("u", a), iactor) or not put((ts, t), itarget[1]):
            return None
        if ln is not None and not put(("l", ln), iline):
            return None
        if b is not None and not put(("u", b), ibetween):
            return None
    return bound


def match_counts(items: list, units: dict, p: Pattern, kinds: list) -> "tuple[int, int]":
    """(count for the side to move, count for the other) of the pattern at a position: the connected sets of as many atoms as the pattern has, each matched in some order."""
    k = p.size
    objs = [{v for v in (it[1], it[2][1], it[3], it[4]) if v is not None} for it in items]
    adj = [[j for j in range(len(items)) if j != i and objs[i] & objs[j]] for i in range(len(items))]
    sets: set = set()

    def grow(cur: tuple, cand: set) -> None:
        if len(cur) == k:
            sets.add(tuple(sorted(cur)))
            return
        for j in sorted(cand):
            grow(cur + (j,), (cand | set(adj[j])) - set(cur) - {j})
    for i in range(len(items)):
        grow((i,), set(adj[i]))
    out = [0, 0]
    for s in sets:
        for side in (0, 1):                  # 0: read for the side to move, 1: for the other
            for perm in permutations(s):
                bound = _bind(p.atoms, tuple(items[j] for j in perm))
                if not bound:
                    continue
                uvars = {v[1]: o for v, o in bound.items() if v[0] == "u"}
                ok = True
                for u in range(p.n_units()):
                    un = units[uvars[u]]
                    mine = un["mine"] if side == 0 else not un["mine"]
                    if mine != p.sides[u] or (p.level >= 1 and un["goal"] != p.goal[u]) or (p.level == 2 and un["edge"] != p.edge[u]) or (p.level == 3 and kinds[p.kind[u]] != un["kind"]):
                        ok = False
                        break
                if ok:
                    out[side] += 1
                    break
    return out[0], out[1]


def match_binds(items: list, units: dict, p: Pattern, kinds: list) -> "tuple[list, list]":
    """PLAN 78.54: like `match_counts`, but the BINDINGS: per side (0: read for the side to move, 1: for the other) one entry per matched set of atoms, the set of distinct tuples of unit ids by unit variable
    (one tuple where the set matches one way; more than one where the pattern has a symmetry that lets two variables swap their units)."""
    k = p.size
    objs = [{v for v in (it[1], it[2][1], it[3], it[4]) if v is not None} for it in items]
    adj = [[j for j in range(len(items)) if j != i and objs[i] & objs[j]] for i in range(len(items))]
    sets: set = set()

    def grow(cur: tuple, cand: set) -> None:
        if len(cur) == k:
            sets.add(tuple(sorted(cur)))
            return
        for j in sorted(cand):
            grow(cur + (j,), (cand | set(adj[j])) - set(cur) - {j})
    for i in range(len(items)):
        grow((i,), set(adj[i]))
    out: "list[list]" = [[], []]
    for s in sets:
        for side in (0, 1):
            found = set()
            for perm in permutations(s):
                bound = _bind(p.atoms, tuple(items[j] for j in perm))
                if not bound:
                    continue
                uvars = {v[1]: o for v, o in bound.items() if v[0] == "u"}
                ok = True
                for u in range(p.n_units()):
                    un = units[uvars[u]]
                    mine = un["mine"] if side == 0 else not un["mine"]
                    if mine != p.sides[u] or (p.level >= 1 and un["goal"] != p.goal[u]) or (p.level == 2 and un["edge"] != p.edge[u]) or (p.level == 3 and kinds[p.kind[u]] != un["kind"]):
                        ok = False
                        break
                if ok:
                    found.add(tuple(uvars[u] for u in range(p.n_units())))
            if found:
                out[side].append(found)
    return out[0], out[1]
