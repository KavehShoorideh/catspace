"""PLAN 78.52: the size of the layer-1 graph and the time to build it per position, on about a thousand positions of the kept games (two per game, at a third and two thirds of its length).
A measurement; nothing is wired into any player. Prints OBS lines.

  .venv/bin/python -m catspace.incubator.analyst.discovery.graph_layer1_measure --game NAME [--games PATH] [--per-game 2]
"""
from __future__ import annotations

import argparse
import json

import cs_engine as CE

from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.io import paths

DEFAULT_GAMES = ("opportunity_78_44", "games_s5016_n500", "games.jsonl")


def positions_of(records: list, per_game: int) -> list:
    """(variant, position text) at fractions i / (per_game + 1) of each game's length."""
    out = []
    for rec in records:
        n = len(rec["moves"])
        for i in range(1, per_game + 1):
            t = n * i // (per_game + 1)
            out.append((rec["variant"], CE.state(rec["variant"], rec.get("start"), rec["moves"][:t])[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--games", default=paths.experiment(*DEFAULT_GAMES))
    ap.add_argument("--per-game", type=int, default=2)
    ap.add_argument("--passes", type=int, default=3)
    a = ap.parse_args()
    records = [json.loads(line) for line in open(a.games)]
    pos = positions_of(records, a.per_game)
    label = f"graph layer1 {a.game}"
    models: dict = {}
    for v in sorted({v for v, _ in pos}):
        models[v] = GL.model(a.game, v)
    for v, m in models.items():
        sp = m.static_part()
        mine = [p for pv, p in pos if pv == v]
        print(f"OBS {label} {v}: static part {len(sp['nodes'])} nodes, {len(sp['links'])} links; lines {len(sp['lines'])}; directions {len(sp['directions'])}")
        res = None
        for k in range(a.passes):
            res = m.measure(mine)
            print(f"OBS {label} {v}: pass {k + 1}: build of the per-position part {res['seconds'] / res['positions'] * 1e6:.1f} microseconds per position over {res['positions']} positions ({res['seconds']:.4f} s)")
        n = res["positions"]
        for t in res["node_types"]:
            if res["static_nodes"].get(t):
                print(f"OBS {label} {v}: static nodes {t}: {res['static_nodes'][t]}")
        for t in res["link_types"]:
            if res["static_links"].get(t):
                print(f"OBS {label} {v}: static links {t}: {res['static_links'][t]}")
        for t in res["node_types"]:
            if res["nodes"].get(t):
                print(f"OBS {label} {v}: position nodes {t}: {res['nodes'][t]} over {n} positions, {res['nodes'][t] / n:.1f} per position")
        for t in res["link_types"]:
            if res["links"].get(t):
                print(f"OBS {label} {v}: position links {t}: {res['links'][t]} over {n} positions, {res['links'][t] / n:.1f} per position")
        tn, tl = sum(res["nodes"].values()), sum(res["links"].values())
        print(f"OBS {label} {v}: position part {tn / n:.1f} nodes and {tl / n:.1f} links per position on average")


if __name__ == "__main__":
    main()
