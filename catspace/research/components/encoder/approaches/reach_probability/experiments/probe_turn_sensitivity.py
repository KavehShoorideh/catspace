#!/usr/bin/env python
"""probe_turn_sensitivity.py -- does the field USE the side-to-move flag?

The scar this measures: sensitivity was 0.000 through v3 (the corpus had no minimal pairs on
glob[0]; the flag's pathway atrophied; tactical distillation could not fix it). The stratified
turn-fork corpus (9,450 game-grounded null-move pairs) exists to move this number.

For sampled legal positions: build the null-move twin (turn flipped, ep cleared, skipped if
illegal), read E = P(W) + 0.5 P(D) from the committor and dA to the decisive poles for BOTH.
  sensitivity_E  = mean |E(s) - E(twin)|          (probability ruler)
  sensitivity_dA = mean |dA(s->W) - dA(twin->W)|  (length ruler, log1p space)
Also printed on TB-labeled <=5-piece positions where the ORACLE says the turn decides
(win-if-move / draw-if-not): the flip agreement -- does E move the RIGHT WAY.

    .venv/bin/python -m ...probe_turn_sensitivity --ckpt <field.pt> [--n 2000]
"""
from __future__ import annotations

import argparse

import chess
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess
    from catspace.research.components.encoder.approaches.reach_probability.experiments.eval_dtz_gate import (
        row_to_board)
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
    from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T

    eng = KittyChess(args.ckpt, args.device)
    c = eng.cfg
    tr = T.build(n_human=0, n_sf=c["games"], seed=c["traj_seed"], max_plies=c["max_plies"],
                 n_piecedown=c.get("n_piecedown", 0), verbose=False)
    rng = np.random.default_rng(0)
    rows = rng.choice(tr.n_positions, args.n * 2, replace=False)

    P3 = eng.poles[[eng.pi["WIN"], eng.pi["DRAW"], eng.pi["LOSS"]]].to(args.device)
    toks_a, globs_a, toks_b, globs_b, boards = [], [], [], [], []
    for r in rows:
        if len(boards) >= args.n:
            break
        b = row_to_board(tr.tok[r], tr.glob[r])
        if not b.is_valid() or b.is_game_over(claim_draw=True):
            continue
        b2 = b.copy(stack=False)
        b2.turn = not b2.turn
        b2.ep_square = None
        if not b2.is_valid() or b2.is_game_over(claim_draw=True):
            continue
        tk, gl = tokenize(b)
        tk2, gl2 = tokenize(b2)
        toks_a.append(np.asarray(tk)); globs_a.append(np.asarray(gl))
        toks_b.append(np.asarray(tk2)); globs_b.append(np.asarray(gl2))
        boards.append(b)

    def read(toks, globs):
        E, dAW = [], []
        for a in range(0, len(toks), 2048):
            with torch.no_grad():
                z = eng._embed(toks[a:a + 2048], globs[a:a + 2048]).float()
                DB = torch.stack([eng.net.dB(z, P3[[k]].expand(len(z), -1))
                                  for k in range(3)], 1)
                DA = torch.stack([eng.net.dA(z, P3[[k]].expand(len(z), -1))
                                  for k in range(3)], 1)
                pr = torch.softmax(-DB / 5.0, 1).cpu().numpy()
            E.append(pr[:, 0] + 0.5 * pr[:, 1])
            dAW.append(np.log1p(np.clip(DA[:, 0].cpu().numpy(), 0, None)))
        return np.concatenate(E), np.concatenate(dAW)

    E_a, dA_a = read(toks_a, globs_a)
    E_b, dA_b = read(toks_b, globs_b)
    sE = np.abs(E_a - E_b)
    sA = np.abs(dA_a - dA_b)
    print(f"[turn] {len(boards):,} legal null-move pairs from the training corpus")
    print(f"[turn] sensitivity_E  mean {sE.mean():.4f}  p90 {np.percentile(sE, 90):.4f}  "
          f"(v3 baseline: 0.000)")
    print(f"[turn] sensitivity_dA mean {sA.mean():.4f}  p90 {np.percentile(sA, 90):.4f}  "
          f"(log1p plies to the WIN pole)")

    # oracle-decided subset: <=5 pieces where TB says the tempo decides
    if eng.tb is not None:
        agree, n_dec = 0, 0
        for i, b in enumerate(boards):
            if len(b.piece_map()) > 5:
                continue
            b2 = b.copy(stack=False); b2.turn = not b2.turn; b2.ep_square = None
            try:
                w1, _ = eng.tb.wdl_dtz(b)
                w2, _ = eng.tb.wdl_dtz(b2)
            except Exception:
                continue
            if w1 is None or w2 is None:
                continue
            # white-POV truth for each variant
            t1 = (w1 if b.turn else -w1)
            t2 = (w2 if b2.turn else -w2)
            if np.sign(t1) == np.sign(t2):
                continue                       # the tempo does not decide here
            n_dec += 1
            agree += ((E_a[i] - E_b[i]) > 0) == (t1 > t2)
        if n_dec:
            print(f"[turn] TB tempo-decisive subset: {n_dec} pairs, "
                  f"E moves the RIGHT way in {agree/n_dec:.0%}  (chance 50%)")
        else:
            print("[turn] TB tempo-decisive subset: none in sample")


if __name__ == "__main__":
    main()
