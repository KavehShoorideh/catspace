"""PLAN 78.26 THE WIN / DRAW / LOSS RESULTS TABLE: counts of how finished games ended from positions LIKE this one, keyed by an abstraction of the position (the rust
engine's `wdl_keys`: the exact position, then the units standing with the side to move). The engine's `wdl` operator pools a hit with its playouts (cs_planning::wdl).

Every position along a FINISHED game adds the game's result, seen from that position's mover, under each of its keys; an outcome that is not 0, 50 or 100 is split by
the same rule as the rust `split` (the score kept exactly). Unfinished games are skipped (and counted). `exact:` rows seen in fewer than `min_exact` games are dropped:
a position met once says nothing about the next one. The file is JSON: {"format": "wdl-table-1", "game", "provenance", "rows": {key: [mover wins, draws, mover losses]},
"stats"}; the engine refuses another format string. The human-games index is NOT read here (that needs the owner's say): the source is the engine's own games (wake logs).
Plan 78.27: the sleep adds two more OWN sources through `build` (unfinished games settled by a deep reading, deep readings of the wake's gap positions) and carries the previous
table forward; each source is counted apart in `stats` and named in `provenance`. Nothing of a teacher or a judge is imported here.

  .venv/bin/python -m catspace.incubator.analyst.discovery.wdl_table chess --log artifacts/experiments/wake_chess_<tag>.jsonl [--log ...] [--out PATH] [--provenance TEXT]
"""
from __future__ import annotations

import argparse
import json
import sys

import cs_engine as CE

from catspace.io import paths

FORMAT = "wdl-table-1"                         # the format string the engine's loader requires
MIN_EXACT = 2                                  # an exact row needs this many games behind it to be kept
ROLES = {"white": 0, "black": 1}               # the chess family's seats (the kernels' roles), as wake.py has them
PROVENANCE = "own games"


def default_path(game: str) -> str:
    """Where a game's table lives by default (the derived-data registry, never a relative path)."""
    return paths.derived(f"wdl_table_{game}.json")


def split(value: float) -> list:
    """One outcome (0..100 for a role) as [win, draw, loss] mass of total 1, the score kept exactly (the rust `wdl::split`)."""
    v = min(1.0, max(0.0, value / 100.0))
    if v > 0.5:
        return [2 * v - 1, 2 - 2 * v, 0.0]
    if v < 0.5:
        return [0.0, 2 * v, 1 - 2 * v]
    return [0.0, 1.0, 0.0]


def family(key: str) -> str:
    return key.split(":", 1)[0]


def flip(mass) -> list:
    """[win, draw, loss] seen from the other side of a two-seat game."""
    return [mass[2], mass[1], mass[0]]


def load(path) -> dict:
    """A table file, checked for the format string (a prior table to carry forward)."""
    with open(path) as f:
        t = json.load(f)
    if t.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} table")
    return t


def build(records, variant: str, min_exact: int = MIN_EXACT, roles: "dict | None" = None, provenance: str = PROVENANCE, *, settled=None, deep_rows=None, prior_table=None) -> dict:
    """The table of `records` -- (start or None, moves, outcome per role 0..100 or None) -- for `variant`. `roles`: the mover's name -> its index in the outcome.

    Plan 78.27 adds three own sources, each counted apart in `stats`:
      `settled`    unfinished games read to a soft result: dicts {start, moves, mover, wdl: [win, draw, loss] of the LAST position's mover}; every position along the game
                   (the last one included) adds that result from its own mover's view (the other seat of a two-seat game sees it flipped). They count as games for `min_exact`.
      `deep_rows`  {key: [w, d, l]} from the sleep's deep readings; never dropped by `min_exact` (a reading is not a single lucky game).
      `prior_table` a table (dict, or a file path) whose rows are carried forward and summed in, so the table accumulates across cycles.
    With none of the three the table is what it was before."""
    seats = roles or ROLES
    rows: dict = {}
    seen: dict = {}
    games = skipped = positions = 0
    for start, moves, outcome in records:
        games += 1
        if outcome is None:
            skipped += 1
            continue
        moves = list(moves)
        along = CE.wdl_keys_along(variant, start, moves)
        for i in range(len(moves)):                              # (the last position is the end: nothing moves there)
            mover = CE.state(variant, start, moves[:i])[1]
            if mover is None:
                break
            mass = split(outcome[seats[mover]])
            positions += 1
            for key in along[i]:
                r = rows.setdefault(key, [0.0, 0.0, 0.0])
                for j in range(3):
                    r[j] += mass[j]
                seen[key] = seen.get(key, 0) + 1
    settled = list(settled or [])
    settled_positions = 0
    for g in settled:
        moves = list(g["moves"])
        along = CE.wdl_keys_along(variant, g.get("start"), moves)
        for i in range(len(moves) + 1):                          # (the last position HAS a mover: it is the one that was read)
            mover = CE.state(variant, g.get("start"), moves[:i])[1]
            if mover is None:
                break
            mass = list(g["wdl"]) if mover == g["mover"] else flip(g["wdl"])
            settled_positions += 1
            for key in along[i]:
                r = rows.setdefault(key, [0.0, 0.0, 0.0])
                for j in range(3):
                    r[j] += mass[j]
                seen[key] = seen.get(key, 0) + 1
    dropped = [k for k in rows if k.startswith("exact:") and seen[k] < min_exact]
    for k in dropped:
        del rows[k]
    deep_mass = 0.0
    deep = dict(deep_rows or {})
    for k, m in deep.items():
        r = rows.setdefault(k, [0.0, 0.0, 0.0])
        for j in range(3):
            r[j] += m[j]
        deep_mass += sum(m)
    carried = 0
    sources = [provenance]
    if settled:
        sources.append("own settled games")
    if deep:
        sources.append("own deep playouts")
    if prior_table is not None:
        prior = load(prior_table) if isinstance(prior_table, (str, bytes)) or hasattr(prior_table, "__fspath__") else prior_table
        if prior.get("format") != FORMAT or prior.get("game") != variant:
            raise ValueError(f"the prior table is not a {FORMAT} table of {variant}")
        for k, m in prior["rows"].items():
            r = rows.setdefault(k, [0.0, 0.0, 0.0])
            for j in range(3):
                r[j] += m[j]
            carried += 1
        sources.append("prior own table")
    kept: dict = {}
    for k in rows:
        kept[family(k)] = kept.get(family(k), 0) + 1
    return {"format": FORMAT, "game": variant, "provenance": " + ".join(sources), "rows": rows,
            "stats": {"games": games, "skipped_unfinished": skipped, "positions": positions, "min_exact": min_exact, "dropped_exact": len(dropped), "rows_kept": kept,
                      "settled": {"games": len(settled), "positions": settled_positions}, "deep": {"rows": len(deep), "mass": deep_mass}, "carried_rows": carried}}


def records_from_wake_logs(log_paths) -> list:
    """The games of wake logs (JSON Lines `game` rows) as (start, moves, outcome) -- the outcome None for an unfinished game."""
    out = []
    for p in log_paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("kind") != "game":
                    continue
                done = not row.get("unfinished") and row.get("outcome") is not None
                out.append((row.get("start"), row.get("moves", []), row["outcome"] if done else None))
    return out


def write(table: dict, path: str) -> str:
    with open(path, "w") as f:
        json.dump(table, f, sort_keys=True)
    return str(path)


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--log", action="append", required=True, help="a wake log (JSON Lines) whose finished games are counted; repeat for several")
    ap.add_argument("--out", default=None, help="where to write the table (default: the game's derived-data path)")
    ap.add_argument("--min-exact", type=int, default=MIN_EXACT, help="exact rows seen in fewer games than this are dropped")
    ap.add_argument("--provenance", default=PROVENANCE)
    a = ap.parse_args(argv)
    variant = a.variant or a.game
    table = build(records_from_wake_logs(a.log), variant, min_exact=a.min_exact, provenance=a.provenance)
    out = write(table, a.out or default_path(a.game))
    st = table["stats"]
    print(f"OBS wdl_table {a.game}: {st['games']} games read, {st['skipped_unfinished']} skipped as unfinished, {st['positions']} positions, "
          f"{st['dropped_exact']} exact rows dropped under {st['min_exact']} games")
    print(f"OBS wdl_table {a.game}: rows kept " + (", ".join(f"{k} {v}" for k, v in sorted(st["rows_kept"].items())) or "none"))
    print(f"OBS wdl_table {a.game}: wrote {out} (provenance: {table['provenance']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
