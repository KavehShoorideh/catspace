"""PLAN 78.62: the tables the later stages read, made from the state of stage A (tactical_state.py): per row, per COLOUR (the two roles of the game, read from the engine), what stands lost, safe, threatened or
trapped, by kind; the board's edge distances; the rules' forcing read where it was kept (78.59); the game's own table (ratings, band, result, fit or test). Game-free: a kind, a role, a place is a word read from
the state file; the edge distance of a place is read from the place's own name (its two coordinates, whatever the board), never from a listed place.

A STANDING LOSS of a colour at a row: a unit of that colour whose class is `lost` or `both` and which rests on the ESTIMATE (the word of the plan: lost by estimate). The units that rest on a bound are counted
apart (a claim on a cut line is worth nothing, 78.60 read 7); forced and pruned ones are counted too (they are almost empty).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import forcing_arrival_measure as FA
from catspace.incubator.analyst.discovery import tactical_state as TS
from catspace.io import paths

PLACE = re.compile(r"^([A-Za-z]+)(\d+)$")


def edge_distances(places: list) -> np.ndarray:
    """The distance of each place to the nearest edge of the board, read from the place names (a letter word and a number): the least of the steps to the lowest and the highest of each coordinate."""
    xy = []
    for nm in places:
        m = PLACE.match(nm)
        xy.append((ord(m.group(1)[0].lower()), int(m.group(2))) if m else (0, 0))
    a = np.array(xy)
    lo, hi = a.min(axis=0), a.max(axis=0)
    return np.minimum(a - lo, hi - a).min(axis=1).astype(np.int16)


def colour_tables(st: dict) -> dict:
    """Per row and colour (0, 1 = the roles in the engine's order): arrays of shape (rows, 2) or (rows, 2, kinds):
    lost_est, lost_bound, lost_firm (forced or pruned), safe_firm, safe_est, threat, trapped, immobile; by kind: lost_est_k, edge_k (sum of the edge distances of the board's units of the colour, by kind)."""
    nk = len(st["meta"]["kinds"])
    n = len(st["ply"])
    z = lambda *s: np.zeros((n,) + s, np.int16)
    out = {k: z(2) for k in ("lost_est", "lost_bound", "lost_firm", "safe_firm", "safe_est", "threat", "trapped", "immobile", "units_read")}
    out["lost_est_k"], out["edge_k"], out["count_k"] = z(2, nk), z(2, nk), z(2, nk)
    out["lost_taken"], out["lost_for_nothing"] = z(2), z(2)
    ed = edge_distances(st["meta"]["places"])
    mover = st["mover"].astype(int)
    for i in range(n):
        m = mover[i]
        u = st["units"][st["uoff"][i]:st["uoff"][i + 1]]
        for own, kind, cls, basis, trap, fate in u:
            c = m if own == 1 else 1 - m
            out["units_read"][i, c] += 1
            if cls in (0, 2):
                if basis == 2:
                    out["lost_est"][i, c] += 1
                    out["lost_est_k"][i, c, kind] += 1
                    out["lost_taken"][i, c] += fate in (1, 2)
                    out["lost_for_nothing"][i, c] += fate == 1
                elif basis == 3:
                    out["lost_bound"][i, c] += 1
                else:
                    out["lost_firm"][i, c] += 1
            if cls in (1, 2):
                if basis in (0, 1):
                    out["safe_firm"][i, c] += 1
                elif basis == 2:
                    out["safe_est"][i, c] += 1
            if cls == 4:
                out["threat"][i, c] += 1
            if trap == 1:
                out["trapped"][i, c] += 1
            if trap == 2:
                out["immobile"][i, c] += 1
        b = st["board"][st["boff"][i]:st["boff"][i + 1]]
        for role, kind, place in b:
            out["edge_k"][i, role, kind] += ed[place]
            out["count_k"][i, role, kind] += 1
    return out


def game_slices(st: dict) -> "tuple[np.ndarray, np.ndarray]":
    """(first row, one past the last row) of each game id that has rows: arrays over the unique game numbers, the rows of a game contiguous and in ply order."""
    g = st["game"]
    ids, first = np.unique(g, return_index=True)
    last = np.append(first[1:], len(g))
    return ids, first, last


def forcing_counts(games: list, st: dict, reads_path: Path) -> np.ndarray:
    """Per row: the confirmed threat moves of the side to move and of the other side by the rules' forcing read of 78.59 (-1 where the position is over or no read is kept)."""
    out = -np.ones((len(st["ply"]), 2), np.int16)
    reads = {}
    with open(reads_path) as f:
        for line in f:
            if line.strip():
                x = json.loads(line)
                reads[x["game"]] = x["plies"]
    counted = {g: FA.counts_of_game(pl) for g, pl in reads.items()}
    for i, (g, t) in enumerate(zip(st["game"], st["ply"])):
        pl = reads.get(int(g))
        if pl is None or t >= len(pl) or pl[t] is None:
            continue
        own, oth = counted[int(g)]
        out[i] = (own[t][2], oth[t][2])
    return out


def load_all(state_dir: Path, heads_dir: "Path | None" = None) -> dict:
    """The state of stage A, its game table and the colour tables, in one dict."""
    st = TS.load(state_dir / "state.npz")
    games = json.loads((state_dir / "games.json").read_text())
    tb = colour_tables(st)
    rd = Path(heads_dir or paths.experiment(*TS.HEADS_DIR)).parent / "full_arrival" / "reads_light.jsonl"
    fc = forcing_counts(games, st, rd) if rd.exists() else None
    return {"st": st, "games": games, "tb": tb, "forcing": fc}
