#!/usr/bin/env python
"""gen_search_labels.py -- search-resolved committor labels for tactical distillation
(2026-08-11, after the external ladder: 94% vs random defense, ~4% vs ANY real tactics --
the static field's tactical blindness is THE binding constraint on absolute strength).

For sampled corpus positions: run OUR OWN shallow search (the engine's play config) and read
the probability head at the END of the best line -- the search-resolved, turn-aware committor.
Labels are frozen at generation (no moving-target bootstrap). The turbulence gap
|E_static - E_resolved| is stored per row so training can weight the positions where the
static field is provably lying to itself. NO oracle evaluations anywhere (Kaveh's rule).

    .venv/bin/python -m ...gen_search_labels --ckpt <field.pt> [--n 30000] [--depth 2]
writes <ckpt>_search_labels.npz {row, probs(3, white-POV), e_static, e_resolved}
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.eval_dtz_gate import (
    row_to_board)
from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=30_000)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--turn-contrast", action="store_true",
                    help="TURN-CONTRAST pairs (Kaveh 2026-08-12: 'side to move matters for "
                         "distance, and for probability'): for each sampled position, also "
                         "search its NULL-MOVE counterpart (same placement, turn flipped, ep "
                         "cleared) and store both resolved committors. The minimal pairs the "
                         "corpus lacks -- the ONLY thing that can give the turn flag gradient.")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    eng = KittyChess(args.ckpt, args.device)
    c = eng.cfg
    tr = T.build(n_human=0, n_sf=c["games"], seed=c["traj_seed"], max_plies=c["max_plies"],
                 n_piecedown=c.get("n_piecedown", 0), verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    game = tr.game_of_row()
    rows = np.flatnonzero(np.isin(game, np.flatnonzero(split == 0)))   # TRAIN rows only
    rng = np.random.default_rng(0)
    rows = rows[rng.choice(len(rows), min(args.n, len(rows)), replace=False)]

    if args.turn_contrast:
        import chess as _ch
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
            tokenize as _tok)

        def resolved(b):
            sr = eng.search(b, depth=args.depth)
            if not sr:
                return None
            bc = b.copy()
            for mv in sr[0]["pv"][:6]:
                bc.push(mv)
            (qW, qD, qB), _ = eng.wdl(bc)
            return qW, qD, qB

        o_row, o_pa, o_pb, o_tokb, o_globb, skipped = [], [], [], [], [], 0
        t0 = time.time()
        for i, r in enumerate(rows):
            b = row_to_board(tr.tok[r], tr.glob[r])
            if not b.is_valid() or b.is_game_over(claim_draw=True):
                continue
            b2 = b.copy()
            b2.turn = not b2.turn
            b2.ep_square = None
            if not b2.is_valid() or b2.is_game_over(claim_draw=True):
                skipped += 1        # counterpart illegal (mover's king already attacked)
                continue
            try:
                pa, pb = resolved(b), resolved(b2)
            except Exception:
                continue
            if pa is None or pb is None:
                continue
            tk2, gb2 = _tok(b2)
            o_row.append(int(r)); o_pa.append(pa); o_pb.append(pb)
            o_tokb.append(np.asarray(tk2)); o_globb.append(np.asarray(gb2))
            if (i + 1) % 2000 == 0:
                dpa = np.array(o_pa); dpb = np.array(o_pb)
                gap = np.abs((dpa[:, 0] + 0.5 * dpa[:, 1]) - (dpb[:, 0] + 0.5 * dpb[:, 1]))
                print(f"[turnpair] {i+1}/{len(rows)}  pairs {len(o_row):,}  "
                      f"mean|dE(turn)| {gap.mean():.3f}  gap>0.1: {(gap > 0.1).mean():.0%}  "
                      f"illegal-skip {skipped:,}  [{(time.time()-t0)/60:.0f}m]", flush=True)
        base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
        dpa = np.array(o_pa, np.float32); dpb = np.array(o_pb, np.float32)
        np.savez(base + "_turn_pairs.npz",
                 row=np.array(o_row, np.int64), probs_a=dpa, probs_b=dpb,
                 tok_b=np.array(o_tokb, np.uint8), glob_b=np.array(o_globb, np.uint8))
        gap = np.abs((dpa[:, 0] + 0.5 * dpa[:, 1]) - (dpb[:, 0] + 0.5 * dpb[:, 1]))
        print(f"[turnpair] DONE {len(o_row):,} pairs  mean|dE(turn)| {gap.mean():.3f}  "
              f"gap>0.1: {(gap > 0.1).mean():.0%}  illegal-skip {skipped:,} "
              f"-> {base}_turn_pairs.npz")
        return

    out_row, out_p, out_es, out_er = [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        b = row_to_board(tr.tok[r], tr.glob[r])
        if not b.is_valid() or b.is_game_over(claim_draw=True):
            continue
        try:
            (pW, pD, pB), _ = eng.wdl(b)
            e_static = pW + 0.5 * pD
            sr = eng.search(b, depth=args.depth)
            if not sr:
                continue
            bc = b.copy()
            for mv in sr[0]["pv"][:6]:
                bc.push(mv)
            (qW, qD, qB), _ = eng.wdl(bc)
            e_res = qW + 0.5 * qD
        except Exception:
            continue
        out_row.append(int(r)); out_p.append((qW, qD, qB))
        out_es.append(e_static); out_er.append(e_res)
        if (i + 1) % 2000 == 0:
            gap = np.abs(np.array(out_es) - np.array(out_er))
            print(f"[labels] {i+1}/{len(rows)}  mean|gap| {gap.mean():.3f}  "
                  f"gap>0.1: {(gap > 0.1).mean():.0%}  [{(time.time()-t0)/60:.0f}m]",
                  flush=True)
    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    np.savez(base + "_search_labels.npz",
             row=np.array(out_row, np.int64), probs=np.array(out_p, np.float32),
             e_static=np.array(out_es, np.float32), e_resolved=np.array(out_er, np.float32))
    gap = np.abs(np.array(out_es) - np.array(out_er))
    print(f"[labels] DONE {len(out_row):,} rows  mean|gap| {gap.mean():.3f}  "
          f"gap>0.1: {(gap > 0.1).mean():.0%} -> {base}_search_labels.npz")


if __name__ == "__main__":
    main()
