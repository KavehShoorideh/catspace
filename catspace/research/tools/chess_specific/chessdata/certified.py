"""
data/certified.py — which games carry a TRUSTWORTHY outcome label.

Kaveh 2026-07-19: a flag-fall or concession in a balanced position says nothing
provable about the board, so it must not shape any outcome-conditioned surface
(committor/phead labels, atlas result coloring). Certified outcomes are:

  - DRAW (result 0)                              rule outcome
  - win whose final position is CHECKMATE        board-proven
  - decisive non-mate win (resign/timeout) where the WINNER leads by
    >= resign_material_gap nominal points at the final position ("include
    resignations at 3+ points") -- material-backed concessions are real wins;
    the shards carry no Termination header, and the material gate is the right
    discriminator anyway (a timeout at +3 IS winning; a resignation at +0 is not).

The field GEOMETRY always trains on ALL games -- certification gates only
outcome LABELS. Measured on the 4GB Jan-2019 shards: ~75% of games certify
(draw 4.1% + mate 27.2% + material-backed 43.4%); the ~25% balanced-position
concessions/flag-falls are masked.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed

_PIECE_VAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}   # P N B R Q K nominal points


def collect_certified_games(shard_dir: Path, resign_material_gap: float = 3.0) -> np.ndarray:
    """Boolean array indexed by game_id: True iff the game's outcome is
    certified per the module rules. Needs only the existing shards
    (include_final stored each game's final position) -- NO rebuild. Cached
    beside the shards, keyed by the gap; the board scan is the expensive half."""
    cache = shard_dir / f"certified_games_mg{resign_material_gap:g}.npy"
    if cache.exists():
        return np.load(cache)
    n_games = int(json.loads((shard_dir / "manifest.json").read_text())["games_kept"])
    arr = np.zeros(n_games, dtype=bool)
    for path in sorted(shard_dir.glob("shard_*.npz")):
        npz = np.load(path)
        gid, result = npz["game_id"], npz["result"]
        packed, meta = npz["packed"], npz["meta"]
        last = np.flatnonzero(np.r_[np.diff(gid) != 0, True])   # final row per game in-shard
        for row in last:
            g = int(gid[row]); res = int(result[row])
            if res == 0:                                         # draw: rule-certified
                arr[g] = True; continue
            board = board_from_packed(packed[row], meta[row])
            if board.is_checkmate():                             # win by mate: board-proven
                arr[g] = True; continue
            wp = bp = 0                                          # nominal material at the end
            for pc in board.piece_map().values():
                v = _PIECE_VAL.get(pc.piece_type, 0)
                if pc.color:
                    wp += v
                else:
                    bp += v
            gap = (wp - bp) if res == 1 else (bp - wp)           # winner minus loser
            if gap >= resign_material_gap:                       # material-backed win
                arr[g] = True
    np.save(cache, arr)
    return arr
