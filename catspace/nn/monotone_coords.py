"""
nn/monotone_coords.py — OPTION (built 2026-07-17, NOT wired in): fixed monotone
resource coordinates that install chess's irreversibility partial order into the
IQE geometry by construction.

Idea (Kaveh): three resources only ever SHRINK along play -- material, pawn
advancement budget (pawns never retreat), castling rights. They define a DAG of
"eras"; crossing an era boundary is one-way, so the BACKWARD distance across it
is truly infinite. Rather than learning that, append these as fixed (non-learned)
coordinates, scaled by `scale`, to BOTH the F and B embeddings before the IQE
head. IQE charges d(u->v) where v EXCEEDS u; these coordinates DECREASE along
play, so forward moves are free on these axes while any backward era-crossing
costs >= scale per unit of resource -- graded (1 capture back = scale, twenty =
20*scale), unbounded, but only linear: numerically stable "infinity."

To wire (deliberately NOT done): in TorchFB.embed_F / embed_B under a flag,
    e = torch.cat([e, torch.from_numpy(monotone_coords(packed, meta)) * scale], 1)
with iqe components adjusted so d + k_mono divides evenly, and the coordinates
EXCLUDED from var-reg (they are constants w.r.t. training). Discuss before
enabling: it changes the embedding contract (checkpoint compat, d dimensioning).
"""
from __future__ import annotations

import numpy as np

# packed layout: 12 uint64 bitboards (see data/encode.py) -- P N B R Q K per color
_PIECE_VALS = np.array([1, 3, 3, 5, 9, 0, 1, 3, 3, 5, 9, 0], dtype=np.float32)


def monotone_coords(packed: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """(n, 12 uint64) packed bitboards + meta -> (n, 3) monotone-DECREASING
    resource coordinates: [material_total, pawn_advancement_budget,
    castling_rights_count]. Every legal move leaves each coordinate <= its
    previous value; captures/pushes/rights-losses strictly decrease one.

    pawn budget = total ranks-to-promotion remaining over both sides' pawns
    (a pawn push strictly decreases it; a pawn capture-move also decreases it;
    capturing a pawn removes its remaining budget)."""
    n = len(packed)
    out = np.empty((n, 3), dtype=np.float32)
    counts = np.stack([np.bitwise_count(packed[:, i]) for i in range(12)], axis=1)
    out[:, 0] = counts.astype(np.float32) @ _PIECE_VALS          # material
    # pawn advancement budget: white pawns need rank 7-r more steps, black r-...
    wp, bp = packed[:, 0], packed[:, 6]
    budget = np.zeros(n, dtype=np.float32)
    for r in range(8):
        rank_mask = np.uint64(0xFF) << np.uint64(8 * r)
        budget += np.bitwise_count(wp & rank_mask) * float(7 - r)   # white promotes at r=7
        budget += np.bitwise_count(bp & rank_mask) * float(r)       # black promotes at r=0
    out[:, 1] = budget
    # castling rights count: meta slots 1-4 are the four rights flags
    # (encode_meta: WK, WQ, BK, BQ) -- rights only ever disappear.
    out[:, 2] = meta[:, 1:5].astype(np.float32).sum(axis=1)
    return out
