"""PLAN 78.33 THE BOARD'S STRUCTURE FROM THE DESCRIPTION (game-free): where a coordinate stands on its axis, written as a rules-delta, so that a rule can say "the pawn is on an edge file" or
"one step from the promotion rank" without naming a square.

The vocabulary file's `axes` entry lists the game's AXES: [{"sort": S, "succ": R, "goal": G (optional)}]. `sort` is a coordinate sort of the description, `succ` its successor relation (a binary
relation over that sort), `goal` an optional relation whose LAST argument is a coordinate of the axis (the side's last coordinate: `last_rank ?side ?rank`; its other arguments are kept in the
generated relation). The optional `distance_sort` (default "distance") names the sort whose members, in the order of its declaration, are the distances 0, 1, 2 ... (its constants are never
the coordinates' own). Nothing here names a game; the facts are computed in Python from the description's own ground facts and written as ground facts of a delta (ADDITIVE ONLY, with
`signature` lines), composed after the concept deltas by game_store.spec_texts:

  edge_gap_<S> ?c ?d            the coordinate ?c of axis S is ?d steps from the nearest end of the axis (an end is a coordinate with no successor or no predecessor; ?d is the smaller of the
                                steps to either end). Chess: files a and h d0, b and g d1 ... ; ranks 1 and 8 d0.
  goal_gap_<S> ?x ... ?c ?d     (only for an axis with `goal`) ?c is ?d steps from the coordinate G names for the arguments ?x ... (the absolute number of steps, in either direction).
                                Chess: goal_gap_rank white 7 d1 (white promotes on rank 8), goal_gap_rank black 7 d6 (black promotes on rank 1).

  Structure(text, axes, distance_sort)   .tables, .coordinate_sorts, .signatures, .sorts, .delta(), .relations(), .edge_gap(sort, c), .goal_gap(sort, prefix, c)
  structure_delta(text, vocabulary_file) -> the delta text ("" when the vocabulary has no `axes`);  relation_names(vocabulary_file) -> [the generated relation names]
  signatures_of(text) -> {relation: (argument sort, ...)};  sorts_of(text) -> {sort: [member, ...]}
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import gdl as GDL


def signatures_of(text: str) -> dict:
    """{relation: (argument sorts)} from the `(signature name sort ...)` declarations of a description text."""
    return GDL.split_declarations(GDL.parse(text))[1]


def sorts_of(text: str) -> dict:
    """{sort: [member, ...]} from the `(sort name member ...)` declarations of a description text."""
    return GDL.split_declarations(GDL.parse(text))[0]


def edge_name(sort: str) -> str:
    return f"edge_gap_{sort}"


def goal_name(sort: str) -> str:
    return f"goal_gap_{sort}"


def axes_of(vocabulary_file: dict) -> list:
    return list(vocabulary_file.get("axes") or [])


def relation_names(vocabulary_file: dict) -> list:
    """The relations structure_delta defines for a vocabulary file (in the order of the delta); [] without `axes`."""
    out = []
    for a in axes_of(vocabulary_file):
        out.append(edge_name(a["sort"]))
        if a.get("goal"):
            out.append(goal_name(a["sort"]))
    return out


def _ground_facts(sexprs: list, rel: str) -> list:
    return [x[1:] for x in sexprs if isinstance(x, list) and x and x[0] == rel and all(isinstance(t, str) and not t.startswith("?") for t in x[1:])]


class Structure:
    """The structure of the axes of one composed description. `axes` as in the vocabulary file; with none, the structure is empty (no coordinate sort, nothing generated)."""

    def __init__(self, text: str, axes: "list | None" = None, distance_sort: str = "distance"):
        self.axes = list(axes or [])
        self.sorts, self.signatures, rest = GDL.split_declarations(GDL.parse(text))
        self.coordinate_sorts = [a["sort"] for a in self.axes]
        self.distance_sort = distance_sort
        self.order: dict = {}          # sort -> coordinates from the first end to the last
        self.tables: dict = {}         # relation name -> {args tuple (without the distance): distance constant}
        if not self.axes:
            return
        dist = self.sorts.get(distance_sort)
        if not dist:
            raise ValueError(f"the distance sort {distance_sort} is not declared in the description")
        for a in self.axes:
            s = a["sort"]
            pairs = _ground_facts(rest, a["succ"])
            nxt = {p[0]: p[1] for p in pairs if len(p) == 2}
            if len(nxt) != len(pairs) or len(set(nxt.values())) != len(nxt):
                raise ValueError(f"the successor relation {a['succ']} of axis {s} is not a chain")
            starts = [c for c in nxt if c not in set(nxt.values())]
            if len(starts) != 1:
                raise ValueError(f"the successor relation {a['succ']} of axis {s} has {len(starts)} starts, not one")
            seq = [starts[0]]
            while seq[-1] in nxt:
                seq.append(nxt[seq[-1]])
            self.order[s] = seq
            n = len(seq)
            if len(dist) < n:
                raise ValueError(f"the distance sort {distance_sort} has {len(dist)} members, axis {s} needs {n}")
            self.tables[edge_name(s)] = {(c,): dist[min(i, n - 1 - i)] for i, c in enumerate(seq)}
            if a.get("goal"):
                tab = {}
                for g in _ground_facts(rest, a["goal"]):
                    j = seq.index(g[-1])
                    for i, c in enumerate(seq):
                        tab[tuple(g[:-1]) + (c,)] = dist[abs(i - j)]
                self.tables[goal_name(s)] = tab

    def relations(self) -> list:
        return list(self.tables)

    def edge_gap(self, sort: str, c: str) -> "str | None":
        return self.tables.get(edge_name(sort), {}).get((c,))

    def goal_gap(self, sort: str, prefix: tuple, c: str) -> "str | None":
        return self.tables.get(goal_name(sort), {}).get(tuple(prefix) + (c,))

    def delta(self) -> str:
        """The rules-delta text: signature lines and ground facts, additive only ("" with no axes)."""
        if not self.axes:
            return ""
        lines = [";;; generated by catspace/incubator/analyst/discovery/board_structure.py (plan 78.33): where a coordinate stands on its axis; ADDITIVE ONLY."]
        for a in self.axes:
            s = a["sort"]
            lines.append(f"(signature {edge_name(s)} {s} {self.distance_sort})")
            if a.get("goal"):
                sig = self.signatures.get(a["goal"])
                if not sig or sig[-1] != s:
                    raise ValueError(f"the goal relation {a['goal']} must be declared with the axis sort {s} as its last argument")
                lines.append(f"(signature {goal_name(s)} {' '.join(sig)} {self.distance_sort})")
        for name, tab in self.tables.items():
            lines.append("")
            lines += [f"({name} {' '.join(k)} {d})" for k, d in tab.items()]
        return "\n".join(lines) + "\n"


def from_vocabulary(text: str, vocabulary_file: dict) -> Structure:
    """The structure of the composed description `text` for a vocabulary file's `axes` (and `distance_sort`)."""
    return Structure(text, axes_of(vocabulary_file), vocabulary_file.get("distance_sort", "distance"))


def structure_delta(text: str, vocabulary_file: dict) -> str:
    """The delta text game_store.spec_texts composes after the concept deltas; "" for a vocabulary with no `axes`."""
    return from_vocabulary(text, vocabulary_file).delta()
