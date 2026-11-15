"""PLAN 78.54: the readable mirror of the unit tracker (rust/engine/tactics/src/sheet/fates.rs) and of the line through one unit (rust/engine/tactics/src/sheet/base_graph.rs, the `through` relation node), and
the helpers the measure uses. Game-free: a unit is (owner word, kind word, place text) as the layer-1 graph's `unit:<owner>:<kind>:<place>` node gives it; no object of any game is named here.

THE TRACKER between two consecutive positions (the difference of their unit lists):
    same (owner, kind, place) in both                          STAYED -> alive at that place
    gone, and a unit of the same owner and kind has appeared    the unit MOVED there (several gone and appeared together are paired in list order and COUNTED)
    gone, and a unit of ANOTHER owner stands on its place now   TAKEN, by the kind that stands there
    gone otherwise (nobody there; an owner and kind not matched: a change of kind)      OTHER
A fate at a horizon follows the chain; alive after h steps is STAYED when it stands where it started (whatever it did between), else MOVED.

THE LINE THROUGH ONE UNIT (`through_walk`): from the layer-1 graph alone. A bear node of a unit on a unit b along a line l, where the unit's kind has been seen to continue along l's family in that heading
(profile: a bear two or more steps away along it), and the next place on l beyond b that holds a unit.
"""
from __future__ import annotations

STAYED, MOVED, OTHER = 0, 1, 2
NO_FATE = 255


def class_names(kinds: list) -> list:
    """The fate classes: stayed, moved, other, then taken by each kind (the kind words come from the graph)."""
    return ["stayed", "moved", "other"] + [f"taken by {k}" for k in kinds]


def units_of(g) -> list:
    """The units of a layer-1 `Graph` as (owner word, kind word, place text), sorted as the engine sorts them (owner, kind, place by their indices): here by the graph's own node order, which is the
    engine's order. -> list of tuples in node order."""
    out = []
    for nid in g.of_type("unit"):
        _tag, owner, kind, place = nid.split(":", 3)
        out.append((owner, kind, place))
    return out


def steps_between(prev: list, nxt: list) -> "tuple[list, int]":
    """-> (per unit of `prev`: ('alive', index in nxt) | ('taken', taker kind) | ('other',), the number of units paired among several of one owner and kind)."""
    out: list = [None] * len(prev)
    pos = {c: j for j, c in enumerate(nxt)}
    used = [False] * len(nxt)
    gone = []
    for i, c in enumerate(prev):
        if c in pos:
            out[i] = ("alive", pos[c])
            used[pos[c]] = True
        else:
            gone.append(i)
    appeared = [j for j in range(len(nxt)) if not used[j]]
    taken_pair = set()
    ambiguous = 0
    for i in gone:
        o, k, at = prev[i]
        same = [j for j in appeared if nxt[j][0] == o and nxt[j][1] == k]
        if sum(1 for g in gone if prev[g][0] == o and prev[g][1] == k) > 1 or len(same) > 1:
            ambiguous += 1
        free = [j for j in same if j not in taken_pair]
        if free:
            taken_pair.add(free[0])
            out[i] = ("alive", free[0])
            continue
        taker = [c for c in nxt if c[2] == at and c[0] != o]
        out[i] = ("taken", taker[0][1]) if taker else ("other",)
    return out, ambiguous


def fate(plies: list, kinds: list, t: int, i: int, h: int) -> int:
    """The class number of the fate of unit i of ply t after h plies (`plies`: per ply the sorted unit list); NO_FATE past the end of the game. 0 stayed, 1 moved, 2 other, 3 + k taken by kind k."""
    if t + h >= len(plies):
        return NO_FATE
    cur = i
    for s in range(t, t + h):
        st = steps_between(plies[s], plies[s + 1])[0][cur]
        if st[0] == "alive":
            cur = st[1]
        elif st[0] == "taken":
            return 3 + kinds.index(st[1])
        else:
            return OTHER
    return STAYED if plies[t + h][cur][2] == plies[t][i][2] else MOVED


def history(plies: list, kinds: list, t: int, i: int, h: int) -> "tuple[list, str, int | None]":
    """PLAN 78.55 STAGE 0, the mirror of `fates::fate_history`: what unit i of ply t took and how it ended within h plies. -> (the kinds it took, in order; the end: 'stayed' | 'moved' | 'taken' | 'other'
    | 'nofate'; the kind that took it, when taken). A unit takes when it moves onto a place where a unit of the other owner stood and was taken in that step."""
    if t + h >= len(plies):
        return [], "nofate", None
    cur, took = i, []
    for s in range(t, t + h):
        st = steps_between(plies[s], plies[s + 1])[0][cur]
        if st[0] == "alive":
            j = st[1]
            a, b = plies[s][cur], plies[s + 1][j]
            if a[2] != b[2]:
                for c in plies[s]:
                    if c[2] == b[2] and c[0] != a[0]:
                        took.append(c[1])
                        break
            cur = j
        elif st[0] == "taken":
            return took, "taken", st[1]
        else:
            return took, "other", None
    return took, ("stayed" if plies[t + h][cur][2] == plies[t][i][2] else "moved"), None


def history_class(h: "tuple", kinds: list) -> int:
    """The class number of a history, as `fates::history_class`: 0 stayed, 1 moved, 2 other (nothing taken), 3 took and is alive, 4 + k taken by kind k having taken nothing (lost for nothing),
    4 + nk + k taken by kind k after having taken (an exchange); NO_FATE past the end."""
    took, end, by = h
    if end == "nofate":
        return NO_FATE
    if end == "other":
        return 2
    if end in ("stayed", "moved"):
        return 3 if took else (0 if end == "stayed" else 1)
    return 4 + (len(kinds) if took else 0) + kinds.index(by)


def through_walk(g, profile: set) -> set:
    """THE BRUTE WALK of the line through one unit, over a layer-1 `Graph` `g` and a profile {(kind word, line direction word, forward)}: {(unit id, id of the place beyond)} for every bear node of a unit
    on a unit along a line where the unit's kind continues, the next place along the line beyond the unit borne on that holds a unit, with exactly one unit between the actor and that place."""
    line_places: dict = {}
    line_dir: dict = {}
    for t, a, b, via, value in g.links:
        if t == "on_line":
            line_places.setdefault(g.nodes[b][1], {})[value] = g.nodes[a][1]
        elif t == "line_along":
            line_dir[g.nodes[a][1]] = g.nodes[b][1][len("direction:"):]
    at = {}
    kind_of = {}
    for nid in g.of_type("unit"):
        ls = g.neighbours(nid)
        place = [o for lt, _v, _x, o, _f in ls if lt == "unit_on"][0]
        at[place] = nid
        kind_of[nid] = [o for lt, _v, _x, o, _f in ls if lt == "unit_kind"][0][len("kind:"):]
    out = set()
    for nid in g.of_type("bear"):
        ls = [(lt, o) for lt, _v, _x, o, fwd in g.neighbours(nid) if fwd]
        actor = [o for lt, o in ls if lt == "by"][0]
        place = [o for lt, o in ls if lt == "at_place"][0]
        if place not in at:
            continue
        actor_place = [o for lt, _v, _x, o, _f in g.neighbours(actor) if lt == "unit_on"][0]
        for line in [o for lt, o in ls if lt == "along"]:
            order = line_places[line]
            index = {p: k for k, p in order.items()}
            i, j = index[actor_place], index[place]
            if abs(j - i) < 1 or (kind_of[actor], line_dir[line], j > i) not in profile:
                continue
            step = 1 if j > i else -1
            k = j + step
            while k in order:
                q = order[k]
                if q in at:
                    between = [p for p in (order[x] for x in range(min(i, k) + 1, max(i, k))) if p in at]
                    if len(between) == 1:
                        out.add((actor, q))
                    break
                k += step
    return out
