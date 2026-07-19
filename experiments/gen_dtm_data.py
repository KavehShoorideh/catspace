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


def rollout_dtm(board, tb, cap=200):
    """Plies to mate under tablebase-optimal play (both sides), or None if it
    doesn't reach mate within cap (drawn / coverage gap)."""
    b = board.copy(stack=False)
    seen = set()
    plies = 0
    for _ in range(cap):
        if b.is_checkmate():
            return plies
        if b.is_game_over(claim_draw=True):
            return None
        m = tb_best_move(b, tb, seen)
        if m is None:
            return None
        if b.turn == chess.BLACK:
            seen.add(b.board_fen())
        b.push(m)
        plies += 1
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=12000, help="WON positions per material class")
    ap.add_argument("--materials", default="krrkbp,krrvk,krvk")
    ap.add_argument("--out", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tb = TB(args.syzygy_dir)
    rng = np.random.default_rng(args.seed)
    packs, metas, dtms, mats = [], [], [], []
    t0 = time.time()
    for mi, material in enumerate(args.materials.split(",")):
        got = tries = 0
        while got < args.per and tries < args.per * 200:
            tries += 1
            b = random_endgame_start(rng, material)
            if b is None or b.turn != chess.WHITE:
                continue
            if white_pov_value(b, tb) != 1.0:          # WON only (DTM defined)
                continue
            dtm = rollout_dtm(b, tb)
            if dtm is None or dtm < 1:
                continue
            packs.append(encode_packed(b)); metas.append(encode_meta(b))
            dtms.append(dtm); mats.append(mi)
            got += 1
            if got % 2000 == 0:
                print(f"  {material}: {got}/{args.per}  ({got/(time.time()-t0):.0f}/s)", flush=True)
    tb.close()
    packed = np.stack(packs); meta = np.stack(metas)
    dtm = np.array(dtms, dtype=np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, packed=packed, meta=meta, dtm=dtm, material=np.array(mats, dtype=np.int8))
    print(f"[stage] {len(dtm)} positions: {time.time()-t0:.1f}s")
    print(f"VERDICT DTM_DATA n={len(dtm)} dtm[min={dtm.min():.0f} med={np.median(dtm):.0f} "
          f"max={dtm.max():.0f}] -> {args.out}")


if __name__ == "__main__":
    main()
