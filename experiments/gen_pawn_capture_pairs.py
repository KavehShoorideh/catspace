#!/usr/bin/env python
"""experiments/gen_pawn_capture_pairs.py — PAWN-CAPTURE one-way transitions from
the near-mate nucleus (Kaveh 2026-07-20: "only for pawn captures make it
irreversible; push the distance to infinite"). A pawn capture is the canonical
no-way-back move: the pawn changes file AND a piece is removed -- the parent is
unreachable from the child. Precomputes (parent, child) pairs so the foundation
fine-tune can push d(child->parent) toward infinity (IQE represents unreachable
pairs as very large) without per-step board reconstruction.

Also emits a random NON-capture 1-ply child per source (a reversible-ish unit
step) so the fine-tune can pin forward~1 and keep reversible symmetric.

Saves p_packed/p_meta (parent), c_packed/c_meta (pawn-capture child),
s_packed/s_meta (a unit-step child). Source = the near-mate nucleus positions.

Usage:
  .venv/bin/python experiments/gen_pawn_capture_pairs.py --workers 9 \
    --in data/derived/lichess_nearmate.npz --out data/derived/pawncap_pairs.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed, encode_meta, encode_packed


def _kills_a_pawn(board, m):
    """A move is ONE-WAY (d(child->parent) = infinite) iff it removes a pawn from the
    board. Total pawn count is monotonically non-increasing -- pawns are never created,
    only captured or promoted away -- so a dead pawn's material is NEVER reachable
    again (Kaveh 2026-07-20). A captured PIECE (Q/R/B/N) is NOT one-way: a pawn can
    promote to restore it. So the irreversible set = {any capture of a pawn (incl. en
    passant), any promotion}, NOT "a pawn makes a capture" (which wrongly kept
    pawn-takes-queen and dropped rook-takes-pawn)."""
    if m.promotion is not None:
        return True                                        # a pawn leaves the board by promoting
    if board.is_en_passant(m):
        return True                                        # en passant captures a pawn
    if board.is_capture(m):
        victim = board.piece_at(m.to_square)
        return victim is not None and victim.piece_type == chess.PAWN
    return False


def _chunk(task):
    packed, meta, seed = task
    rng = np.random.default_rng(seed)
    pp, pm, cp, cm, sp, sm = [], [], [], [], [], []
    for i in range(len(packed)):
        b = board_from_packed(packed[i], meta[i])
        if b.is_game_over():
            continue
        mv = list(b.legal_moves)
        pdeaths = [m for m in mv if _kills_a_pawn(b, m)]    # one-way moves: a pawn leaves the board
        if not pdeaths:
            continue                                       # only keep positions WITH a pawn-death move
        m = pdeaths[int(rng.integers(len(pdeaths)))]
        child = b.copy(stack=False); child.push(m)
        if child.is_game_over():
            continue
        # a REVERSIBLE unit-step child (non-capture, non-pawn, non-promo -> a piece move
        # that can be undone) for scale-pinning; fall back to any non-capture, then any move.
        rev = [x for x in mv if not b.is_capture(x) and x.promotion is None
               and b.piece_at(x.from_square).piece_type != chess.PAWN]
        noncap = [x for x in mv if not b.is_capture(x)]
        pool = rev or noncap or mv
        step = pool[int(rng.integers(len(pool)))]
        sb = b.copy(stack=False); sb.push(step)
        pp.append(packed[i]); pm.append(meta[i])
        cp.append(encode_packed(child)); cm.append(encode_meta(child))
        sp.append(encode_packed(sb)); sm.append(encode_meta(sb))
    if not pp:
        return None
    return (np.stack(pp), np.stack(pm), np.stack(cp), np.stack(cm), np.stack(sp), np.stack(sm))


def main():
    from concurrent.futures import ProcessPoolExecutor
    from catspace.io.paths import newest_shard_dir
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/derived/lichess_nearmate.npz")
    ap.add_argument("--shards", action="store_true",
                    help="scan the Lichess shards (all game phases) for pawn captures "
                         "instead of the near-mate npz -- pawn captures are common in "
                         "full games but rare in <=5-piece endgames")
    ap.add_argument("--cap", type=int, default=40000)
    ap.add_argument("--out", default="data/derived/pawndeath_pairs.npz")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    W = max(1, args.workers)
    if args.shards:
        shards = sorted(newest_shard_dir().glob("shard_*.npz"))
        tasks = []
        for si, sp in enumerate(shards):
            z = np.load(sp)
            per = max(1, args.cap // max(1, len(shards)))
            pk, mt = z["packed"][:per * 3], z["meta"][:per * 3]   # oversample; sparse-ish
            tasks.append((pk, mt, si))
    else:
        dz = np.load(args.inp)
        packed, meta = dz["packed"], dz["meta"]
        n = len(packed)
        b = np.linspace(0, n, W + 1, dtype=int)
        tasks = [(packed[b[i]:b[i + 1]], meta[b[i]:b[i + 1]], i) for i in range(W) if b[i + 1] > b[i]]
    n = sum(len(t[0]) for t in tasks)
    t0 = time.time()
    parts = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for r in ex.map(_chunk, tasks):
            if r is not None:
                parts.append(r)
    keys = ["p_packed", "p_meta", "c_packed", "c_meta", "s_packed", "s_meta"]
    out = {k: np.concatenate([p[j] for p in parts]) for j, k in enumerate(keys)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"[stage] {len(out['p_packed'])} pawn-capture pairs from {n} near-mate "
          f"positions: {time.time()-t0:.1f}s")
    print(f"VERDICT PAWNCAP_PAIRS n={len(out['p_packed'])} -> {args.out}")


if __name__ == "__main__":
    main()
