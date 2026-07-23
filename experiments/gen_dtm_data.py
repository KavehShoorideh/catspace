#!/usr/bin/env python
"""
experiments/gen_dtm_data.py — tablebase DISTANCE-TO-MATE data for the DTM-hinge
fine-tune (Kaveh 2026-07-19). Samples WON positions across the KRRvKBP
conversion tree (KRRvKBP -> KRRvK -> KRvK) and labels each with dtm = plies-to-
mate under Syzygy-optimal play (rollout; Syzygy has DTZ+WDL, not DTM, so we play
the optimal line and count -- monotone toward mate, covers all <=6-piece toy
positions, no Gaviota download). Saves packed/meta/dtm/result for the hinge:
constrain d(F(s), MATE_W) ~ dtm/scale so the metric's gradient points at mate.

Usage:
  .venv/bin/python experiments/gen_dtm_data.py --per 12000 --out data/derived/dtm_endgame.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import encode_meta, encode_packed
from experiments.selfplay_generate import random_endgame_start
from experiments.value_fixed_point import TB, tb_best_move, white_pov_value


# MOVED to catspace/tb.py -- re-export for existing importers.
from catspace.tb import rollout_dtm  # noqa: F401


def gen_chunk(task):
    """One worker: generate `n` WON positions of `material` with its OWN
    tablebase handle + seed (position generation is embarrassingly parallel --
    each rollout is independent). Returns (packed, meta, dtm, material_index)."""
    material, mi, n, seed, syzygy_dir = task
    tb = TB(syzygy_dir)
    rng = np.random.default_rng(seed)
    packs, metas, dtms = [], [], []
    got = tries = 0
    while got < n and tries < n * 300:
        tries += 1
        b = (random_class_start(rng, material) if "v" in material
             else random_endgame_start(rng, material))
        if b is None or b.turn != chess.WHITE:
            continue
        if white_pov_value(b, tb) != 1.0:              # WON only (DTM defined)
            continue
        dtm = rollout_dtm(b, tb)
        if dtm is None or dtm < 1:
            continue
        packs.append(encode_packed(b)); metas.append(encode_meta(b)); dtms.append(dtm)
        got += 1
    tb.close()
    packed = np.stack(packs) if packs else np.zeros((0,), dtype=np.uint8)
    meta = np.stack(metas) if metas else np.zeros((0,), dtype=np.uint8)
    return packed, meta, np.array(dtms, dtype=np.float32), mi


_PT = {"Q": 5, "R": 4, "B": 3, "N": 2, "P": 1}


def random_class_start(rng, name: str):
    """Sample a legal White-to-move position of tablebase class `name` ('KRRvKB').
    Kaveh 2026-07-25: 'why not sample everything in the tablebase, irrespective of
    material class' -- classes come from the TABLE FILES, no hand lists to miss."""
    import chess as _c
    w, bl = name.split("v")
    pieces = ([( _PT[c], _c.WHITE) for c in w[1:]] + [(_PT[c], _c.BLACK) for c in bl[1:]])
    for _ in range(60):
        n = 2 + len(pieces)
        sqs = rng.choice(64, size=n, replace=False)
        b = _c.Board(None)
        b.set_piece_at(int(sqs[0]), _c.Piece(_c.KING, _c.WHITE))
        b.set_piece_at(int(sqs[1]), _c.Piece(_c.KING, _c.BLACK))
        ok = True
        for sq, (pt, col) in zip(sqs[2:], pieces):
            if pt == _c.PAWN and _c.square_rank(int(sq)) in (0, 7):
                ok = False; break
            b.set_piece_at(int(sq), _c.Piece(pt, col))
        if not ok:
            continue
        b.turn = _c.WHITE
        if b.is_valid() and not b.is_game_over(claim_draw=True):
            return b
    return None


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=12000, help="WON positions per material class")
    ap.add_argument("--materials", default="krrkbp,krrvk,krvk")
    ap.add_argument("--out", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="parallel worker processes (each rollout is independent; "
                         "default = cpu_count-2). 1 = the old single-process path.")
    args = ap.parse_args()

    if args.materials == "all":
        materials = sorted({f.stem for f in Path(args.syzygy_dir).glob("*.rtbw")})
        print(f"[gen] ALL tablebase classes on disk: {materials}", flush=True)
    else:
        materials = args.materials.split(",")
    W = max(1, args.workers)
    # split each material's target across W chunks, each with a distinct seed so
    # workers don't regenerate identical positions. materials(M) x W = M*W tasks.
    tasks = []
    for mi, material in enumerate(materials):
        base, rem = divmod(args.per, W)
        for w in range(W):
            n = base + (1 if w < rem else 0)
            if n > 0:
                tasks.append((material, mi, n, args.seed + 1000 * mi + w, args.syzygy_dir))
    t0 = time.time()
    print(f"[stage] DTM gen: {len(materials)} materials x {args.per} = {len(materials)*args.per} "
          f"target across {W} workers ({len(tasks)} chunks)", flush=True)

    packs, metas, dtms, mats = [], [], [], []
    done_by_mat = {mi: 0 for mi in range(len(materials))}
    if W == 1:                                          # single-process (debug / fallback)
        for t in tasks:
            p, m, d, mi = gen_chunk(t)
            if len(d):
                packs.append(p); metas.append(m); dtms.append(d); mats.append(np.full(len(d), mi))
    else:
        with ProcessPoolExecutor(max_workers=W) as ex:
            futs = {ex.submit(gen_chunk, t): t for t in tasks}
            for fut in as_completed(futs):
                p, m, d, mi = fut.result()
                if len(d):
                    packs.append(p); metas.append(m); dtms.append(d); mats.append(np.full(len(d), mi))
                done_by_mat[mi] += len(d)
                tot = sum(done_by_mat.values())
                print(f"  chunk done: {materials[mi]} +{len(d)}  "
                      f"total {tot}/{len(materials)*args.per}  ({tot/(time.time()-t0):.0f}/s)", flush=True)
                if len(packs) % 15 == 0 and packs:      # incremental checkpoint: a long
                    np.savez(str(args.out) + ".partial.npz",   # gen must not be losable
                             packed=np.concatenate(packs), meta=np.concatenate(metas),
                             dtm=np.concatenate(dtms),
                             material=np.concatenate(mats).astype(np.int8))

    packed = np.concatenate(packs, axis=0); meta = np.concatenate(metas, axis=0)
    dtm = np.concatenate(dtms, axis=0); material = np.concatenate(mats, axis=0).astype(np.int8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, packed=packed, meta=meta, dtm=dtm, material=material)
    per_mat = {materials[mi]: int((material == mi).sum()) for mi in range(len(materials))}
    print(f"[stage] {len(dtm)} positions: {time.time()-t0:.1f}s  per-material {per_mat}")
    print(f"VERDICT DTM_DATA n={len(dtm)} dtm[min={dtm.min():.0f} med={np.median(dtm):.0f} "
          f"max={dtm.max():.0f}] workers={W} -> {args.out}")


if __name__ == "__main__":
    main()
