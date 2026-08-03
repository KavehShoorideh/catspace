#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/reachability_field/experiments/probe_constraint_field.py -- does the field ALREADY encode the cornering
concept (Kaveh 2026-07-22: "find the positions closer to my goal... from my opponent's
pieces' perspective as in I constrain the king")? The concept's exact value is the black
king's escape volume (flood-fill; ladder_mate.escape_volume) -- measured there to recover
most of the oracle's search guidance (0.75 vs 0.85 mate rate @400 nodes). If the field
encodes it, subgoal-finding is FIELD-NATIVE recognition ("this seems familiar"), no
hand-coded flood-fill needed at plan time.

Three read-outs on KRRvK positions (the ladder regime), all VERDICT lines:

  1. REACH   spearman( min d(F(s), B(constrained bank)), escape_volume(s) ) -- does
             quasimetric distance-to-the-constrained-region track the concept? This is
             the navigation signal the planner would actually use.
  2. LINEAR  ridge probe B(s) -> escape_volume, held out. DIAGNOSTIC ONLY (reads the
             representation; does not drive moves) -- is the concept linearly present?
  3. CONTROL same ridge probe on raw feature planes -- if RAW ~ FIELD the field adds no
             organization beyond its input features.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch


from scipy.stats import spearmanr

from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.ladder_mate import escape_volume, random_krrvk
from catspace.io import paths

BOARD_ONLY = (18, 19)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default=paths.sep("iqe_geom_field.pt"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--bank-vol", type=int, default=2, help="escape_volume <= this defines the constrained bank")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]

    # KRRvK sample, anywhere (need the full escape-volume spectrum, corner to center)
    boards = [b for b in (random_krrvk(rng, central=False) for _ in range(args.n * 2)) if b is not None][:args.n]
    vol = np.array([escape_volume(b) for b in boards], dtype=np.float64)
    pk = np.stack([encode_packed(b) for b in boards]); mt = np.stack([encode_meta(b) for b in boards])
    print(f"[probe] {len(boards)} KRRvK positions  escape_volume: min {vol.min():.0f} med {np.median(vol):.0f} "
          f"max {vol.max():.0f}  constrained(<= {args.bank_vol}): {(vol <= args.bank_vol).sum()}", flush=True)

    def bp(pk_, mt_):
        pl = feature_planes(pk_, mt_); pl[:, BOARD_ONLY] = 0.0
        return pl

    F_list, B_list, RAW_list = [], [], []
    for s in range(0, len(pk), 1024):
        pl = bp(pk[s:s + 1024], mt[s:s + 1024])
        with torch.no_grad():
            t = torch.from_numpy(pl).to(dev)
            F_list.append(fb.embed_F(t, torch.from_numpy(np.tile(om, (len(pl), 1))).to(dev)).cpu().numpy())
            B_list.append(fb.embed_B(t).cpu().numpy())
        RAW_list.append(pl.reshape(len(pl), -1))
    F = np.concatenate(F_list); B = np.concatenate(B_list); RAW = np.concatenate(RAW_list)

    # ---- 1. REACH: min quasimetric distance to the constrained bank vs escape volume
    bank = np.flatnonzero(vol <= args.bank_vol)
    if len(bank) < 10:
        print(f"VERDICT CONSTRAINT_FIELD.REACH  SKIPPED (only {len(bank)} constrained positions; raise --n)", flush=True)
    else:
        Bbank = torch.from_numpy(B[bank]).to(dev)
        dmin = []
        for s in range(0, len(F), 512):
            with torch.no_grad():
                dmin.append(fb.distance_matrix(torch.from_numpy(F[s:s + 512]).to(dev), Bbank)
                            .min(1).values.cpu().numpy())
        dmin = np.concatenate(dmin)
        nb = np.flatnonzero(vol > args.bank_vol)          # exclude the bank itself
        rho = spearmanr(dmin[nb], vol[nb]).correlation
        print(f"VERDICT CONSTRAINT_FIELD.REACH field={Path(args.field).stem} bank={len(bank)}  "
              f"spearman(min d(F,B_constrained), escape_vol) = {rho:+.3f}   "
              f"(+ = field distance TRACKS the concept: constrained reads near)", flush=True)

    # ---- 2/3. LINEAR probe (diagnostic read of the representation) vs RAW control
    tr = np.flatnonzero(rng.random(len(vol)) < 0.8); te = np.setdiff1d(np.arange(len(vol)), tr)

    def ridge(X):
        Xc = X - X[tr].mean(0); y = vol - vol[tr].mean()
        lam = 1e-2 * len(tr)
        W = np.linalg.solve(Xc[tr].T @ Xc[tr] + lam * np.eye(X.shape[1]), Xc[tr].T @ y[tr])
        return spearmanr(Xc[te] @ W, vol[te]).correlation

    rB, rF, rRAW = ridge(B), ridge(F), ridge(RAW)
    print(f"VERDICT CONSTRAINT_FIELD.LINEAR  held-out spearman(probe, escape_vol):  "
          f"B {rB:+.3f}  F {rF:+.3f}  RAW-planes {rRAW:+.3f}   "
          f"(B >> RAW = the field ORGANIZES the concept; B ~ RAW = it merely inherits it)  "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
