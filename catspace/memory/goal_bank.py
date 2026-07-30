"""
goal_bank.py — goal-as-REGION representation: a bank of exemplar B-embeddings
scored by best-over-bank instead of one centroid vector.

2026-07-13 measurement that motivated this (JOURNAL.md): on KRvK positions
with exact tablebase plies-to-mate, distance to ANY single mate centroid
(human-game mates or even same-material KRvK mates) is flat (spearman ~0) --
averaging exemplars into one point destroys the geometry -- while distance to
the NEAREST mate exemplar correlates (+0.17 baseline, +0.25 after endgame-
curriculum training). Kaveh's design requirement made concrete: "corner the
king" must be a region in embedding space, broader than any single exemplar
and never collapsed to a mean.

Banks are harvested from real game data (human shards or self-play shards):
final positions of decisive games that are genuine checkmates, optionally
filtered by material size so a bank matches the regime being played (e.g.
endgame banks for endgame diagnostics).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from catspace.data.encode import board_from_packed


def harvest_mate_finals(shard_dirs: list, want_result: int = 1,
                        max_pieces: int | None = None, cap: int = 512) -> list:
    """Checkmate final BOARDS of decisive games across shard dirs.
    want_result: +1 = white delivered mate, -1 = black did.
    max_pieces: keep only mates with at most this many pieces on the board
    (None = all) -- lets a bank match the material regime under study."""
    boards = []
    for shard_dir in shard_dirs:
        for path in sorted(Path(shard_dir).glob("shard_*.npz")):
            npz = np.load(path)
            gid, result = npz["game_id"], npz["result"]
            packed, meta = npz["packed"], npz["meta"]
            last = np.flatnonzero(np.r_[np.diff(gid) != 0, True])
            for row in last:
                if int(result[row]) != want_result:
                    continue
                if max_pieces is not None:
                    if sum(bin(int(m)).count("1") for m in packed[row]) > max_pieces:
                        continue
                board = board_from_packed(packed[row], meta[row])
                if board.is_checkmate():
                    boards.append(board)
                if len(boards) >= cap:
                    return boards
    return boards


def embed_bank(fb, boards: list, device, near: bool = False) -> np.ndarray:
    """(m, d) B-embeddings of exemplar boards -- pass directly as the `z` of
    FBSearchPolicy/FBBoardPolicy (both accept a 2-D bank and score
    best-over-exemplars). near=True uses the two-horizon NEAR head
    (embed_B_near) instead of the default/far head."""
    import torch

    from catspace.data.encode import encode_meta, encode_packed
    from catspace.nn.features import feature_planes

    packed = np.stack([encode_packed(b) for b in boards])
    meta = np.stack([encode_meta(b) for b in boards])
    planes = torch.from_numpy(feature_planes(packed, meta)).to(device)
    embed = fb.embed_B_near if near else fb.embed_B
    with torch.no_grad():
        return embed(planes).cpu().numpy()
