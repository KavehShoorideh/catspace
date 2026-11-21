"""PLAN 78.52: THE GRAPH, LAYER 1 -- the small Python mirror for reading and inspecting what the Rust engine builds (rust/engine/tactics/src/sheet/base_graph.rs).

The engine derives the graph from the rules: the places, the steps by direction, the lines (maximal chains under a direction and its reverse), the edges and each place's edge distance, then,
per position, the units and the relation nodes (a unit bears on a place, may move to a place, runs along a shut line). This module hands the engine the rules' description and the bound
names of the game's graph vocabulary, and wraps the node and link lists it returns. It names no object of any game.

    m = model(game)                # the engine's `BaseGraph` of a game (the static part, built once)
    g = graph(m, position)              # a `Graph`: the static part and the position's, one node list
    g.counts()                          # {node type: n}, {link type: n}
    g.neighbours("place:e4")            # [(link type, direction or None, value, other node id)]
    print(g.text())                     # the position's nodes with their links, one per line
"""
from __future__ import annotations

import cs_engine as CE

from catspace.incubator.analyst.discovery import game_store as GS


def model(game: str, variant: "str | None" = None):
    """The engine's `BaseGraph` for `variant` of `game`: the description's text (composed with the variant's delta where the game has a description for it; the game's own text otherwise,
    as the variants without one of their own read the first game's), the static relations and the objects of the game's graph vocabulary."""
    gv = GS.graph_vocabulary(game)
    texts = GS.graph_spec_texts(game)
    variant = variant or game
    text = texts.get(variant) or texts[game]
    objects = [(name, list(parts)) for name, parts in (gv.get("objects") or {}).items()]
    d = gv.get("fact_decl")                                  # plan 79.22: where the game's facts hold its units and its tempo (the game's own data), so that a game at hand keeps its position between decisions
    decl = None if not d else (d["units"]["relation"], d["units"]["owner"], d["units"].get("kind"), d.get("tempo"))
    return CE.BaseGraph(variant, text, list(gv["static"]), objects, decl) if decl else CE.BaseGraph(variant, text, list(gv["static"]), objects)


class Graph:
    """The static part and one position's part of the graph as one node list (indices as the engine gives them: the position's nodes continue after the static ones)."""

    def __init__(self, static: dict, position: "dict | None" = None):
        self.nodes: list = list(static["nodes"])
        self.links: list = list(static["links"])
        self.mover = None
        self.offset = len(self.nodes)
        if position is not None:
            self.mover = position["mover"]
            self.nodes += position["nodes"]
            self.links += position["links"]
        self.index = {nid: i for i, (_, nid) in enumerate(self.nodes)}
        self._out: "dict | None" = None

    def counts(self) -> tuple:
        """({node type: count}, {link type: count})."""
        nodes: dict = {}
        links: dict = {}
        for t, _ in self.nodes:
            nodes[t] = nodes.get(t, 0) + 1
        for t, *_ in self.links:
            links[t] = links.get(t, 0) + 1
        return nodes, links

    def _adjacent(self) -> dict:
        if self._out is None:
            out: dict = {}
            for t, a, b, via, value in self.links:
                out.setdefault(a, []).append((t, b, via, value, True))
                out.setdefault(b, []).append((t, a, via, value, False))
            self._out = out
        return self._out

    def neighbours(self, node_id: str) -> list:
        """Every link at a node, as (link type, the node it goes by or None, the number or None, the other node's id, whether the link starts at this node)."""
        out = []
        for t, other, via, value, forward in self._adjacent().get(self.index[node_id], []):
            out.append((t, self.nodes[via][1] if via >= 0 else None, value if value >= 0 else None, self.nodes[other][1], forward))
        return out

    def of_type(self, node_type: str) -> list:
        """The ids of the nodes of one type."""
        return [nid for t, nid in self.nodes if t == node_type]

    def text(self, only_position: bool = True) -> str:
        """One node per line with its outgoing links, the position's nodes only unless asked for all."""
        lines = []
        for i, (t, nid) in enumerate(self.nodes):
            if only_position and i < self.offset:
                continue
            outs = [f"{lt}->{other}" + (f"[by {via}]" if via else "") + (f"[{v}]" if v is not None else "") for lt, via, v, other, fwd in self.neighbours(nid) if fwd]
            lines.append(f"{nid} [{t}]  " + " ".join(outs))
        return "\n".join(lines)


def static_part(m) -> dict:
    return m.static_part()


def graph(m, position: str, static: "dict | None" = None) -> Graph:
    """The graph of one position (pass `static_part(m)` to avoid asking for it again)."""
    return Graph(static or m.static_part(), m.position(position))
