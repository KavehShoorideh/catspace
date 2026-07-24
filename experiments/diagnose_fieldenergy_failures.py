#!/usr/bin/env python
"""experiments/diagnose_fieldenergy_failures.py -- WHY is fieldenergy 0.79 and not 1.00 on
KRRvK-central? (Kaveh 2026-07-24: "why only 0.79? what's failing?")

Reruns the exam's EXACT starts (seed 0, same generator) with the exam engine (field value +
energy prior, 800 nodes) but referees every white move with the tablebase (tb = referee only,
never consulted by the engine). Splits failures into:
  BLUNDER   -- a white move flipped tb-WDL from won to not-won (subtypes: stalemate-delivering,
               rook-hang = rook en prise taken next ply, other)
  NO-CLOSE  -- never lost the tb-win but failed to convert (subtypes: 50-move/rep draw claimed,
               ply-budget timeout); records closest-approach |DTZ| = how near mate it got
Per-game lines stream as they finish; aggregate VERDICT at the end.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.tb import TB, tb_best_move, white_pov_value
from experiments.ladder_mate import white_mcts
from experiments.mate_ladder_eval import make_energy_prior, make_field_value, sample_scenarios


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--nodes", type=int, default=800)
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--scenario", default="KRRvK-central")
    ap.add_argument("--energy-ckpt", default="data/derived/sep/opponent_energy_v1.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed); tb = TB()

    starts = dict(sample_scenarios(rng, args.n))[args.scenario]
    print(f"[diag] {len(starts)} starts ({args.scenario}), nodes={args.nodes}", flush=True)
    vfn = make_field_value()
    pfn = make_energy_prior(ckpt=args.energy_ckpt)

    rows = []
    for gi, start in enumerate(starts):
        b = start.copy(stack=False)
        plies = 0
        blunder = None                      # (ply, san, subtype)
        closest = abs(tb.wdl_dtz(b)[1] or 999)
        for _ in range(args.max_plies):
            if b.is_game_over(claim_draw=True):
                break
            if b.turn == chess.WHITE:
                won_before = white_pov_value(b, tb) == 1.0
                mv, _ev = white_mcts(b, args.nodes, vfn, pfn)
                san = b.san(mv)
                b.push(mv)
                if won_before and blunder is None and white_pov_value(b, tb) != 1.0:
                    if b.is_stalemate():
                        sub = "stalemate"
                    else:
                        # rook en prise and taken by tb's reply = hang
                        nb = b.copy(stack=False); nb.push(tb_best_move(nb, tb))
                        rooks_now = len(b.pieces(chess.ROOK, chess.WHITE))
                        rooks_next = len(nb.pieces(chess.ROOK, chess.WHITE))
                        sub = "rook-hang" if rooks_next < rooks_now else "other"
                    blunder = (plies, san, sub)
            else:
                b.push(tb_best_move(b, tb))
            plies += 1
            if white_pov_value(b, tb) == 1.0:
                closest = min(closest, abs(tb.wdl_dtz(b)[1] or 999))
        out = b.outcome(claim_draw=True)
        if out and out.winner == chess.WHITE:
            kind = "mate"
        elif blunder is not None:
            kind = f"BLUNDER:{blunder[2]}"
        elif out is not None:
            kind = f"NO-CLOSE:{out.termination.name.lower()}"
        else:
            kind = "NO-CLOSE:timeout"
        rows.append((kind, plies, closest, blunder))
        det = f" blunder@{blunder[0]} {blunder[1]}" if blunder else f" closest_dtz={closest}"
        print(f"  g{gi:03d} {kind:24s} plies={plies}{'' if kind == 'mate' else det}"
              f"  [{time.time()-t0:.0f}s]", flush=True)

    kinds = [r[0] for r in rows]
    n = len(rows)
    print(f"VERDICT FIELDENERGY_DIAG {args.scenario} n={n} "
          f"mate={kinds.count('mate')/n:.2f}", flush=True)
    for k in sorted(set(kinds)):
        if k != "mate":
            cl = [r[2] for r in rows if r[0] == k and not r[3]]
            extra = f"  closest_dtz median={int(np.median(cl))} min={min(cl)}" if cl else ""
            print(f"    {k:26s} {kinds.count(k):3d}/{n}{extra}", flush=True)
    bl = [r[3] for r in rows if r[3]]
    if bl:
        print(f"    blunder plies: median={int(np.median([x[0] for x in bl]))}  "
              f"moves: {[x[1] for x in bl][:12]}", flush=True)
    tb.close()


if __name__ == "__main__":
    main()
