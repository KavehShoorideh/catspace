#!/usr/bin/env python
"""eval_dtz_gate.py -- the GOLD-STANDARD generalisation gate: learned distances vs tablebase truth.

Kaveh 2026-08-07: the phase-grouped committor exposed that outcome accuracy can be faked by
learning per-phase BASE RATES (every phase sat exactly at its own majority). Tablebase ground
truth is immune to that: for held-out positions with <=5 pieces, Syzygy gives the EXACT minimax
outcome (WDL) and distance (DTZ), no human blunders, no base-rate shortcut. A model that only
knows base rates shows zero within-class DTZ correlation; a model that knows geometry shows it.

Reports, on TEST-GAME positions with <=5 pieces (no dedup -- by Kaveh's call the phase table
carries the memorisation split instead):
  1. TB-WDL committor accuracy vs the class majority of the subset (exact labels, mover POV)
  2. Spearman( d_IQE(s -> own TB-outcome pole), DTZ ) within each TB class -- the distance gate:
     does the field know HOW FAR endings are where reality is computable?
  3. The same for the nearest-pole distance regardless of class (rank-blind variant).
"""
from __future__ import annotations

import argparse

import chess
import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.experiments.plot_strata_figures import (
    embed)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB

# inverse of jepa.tokenize's vocab: 0 empty, 1-6 white PNBRQK, 7-12 black pnbrqk
_INV = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
        7: "p", 8: "n", 9: "b", 10: "r", 11: "q", 12: "k"}


def row_to_board(tok, glob):
    b = chess.Board(None)
    for sq in range(64):
        t = int(tok[sq])
        if t:
            b.set_piece_at(sq, chess.Piece.from_symbol(_INV[t]))
    b.turn = bool(glob[0])
    castle = ""
    if glob[1]: castle += "K"
    if glob[2]: castle += "Q"
    if glob[3]: castle += "k"
    if glob[4]: castle += "q"
    b.set_castling_fen(castle or "-")
    if glob[5]:
        # ep file is stored 1-indexed; rank follows side to move
        f = int(glob[5]) - 1
        b.ep_square = chess.square(f, 5 if b.turn else 2)
    return b


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-pos", type=int, default=2500, help="tb-probed positions (cache-friendly)")
    ap.add_argument("--cond-elo", type=float, default=None,
                    help="evaluate the CONDITIONED field at this Elo (e.g. 3500 = near-minimax "
                         "readout, 1500 = exploitation readout); requires a --dual checkpoint")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    net, pay = load_net(args.ckpt, args.device)

    def _embed_rows(net, tr, rows, device, batch=4096):
        import torch as _t
        if args.cond_elo is None or not getattr(net, "dual", False):
            from catspace.research.components.encoder.approaches.reach_probability.experiments.plot_strata_figures import embed as _e
            return _e(net, tr, rows, device)
        cval = (args.cond_elo - 1500.0) / 500.0
        out = []
        for s0 in range(0, len(rows), batch):
            r = rows[s0:s0 + batch]
            cond = _t.full((len(r), net.qhead.proj_delta.in_features
                            - net.qhead.proj_base.in_features), cval, device=device)
            zb, zc = net.encode_dual(
                _t.from_numpy(tr.tok[r].astype(np.int64)).to(device),
                _t.from_numpy(tr.glob[r].astype(np.float32)).to(device), cond)
            out.append(zc.float().cpu())
        return _t.cat(out)
    c = pay["cfg"]
    tr = T.build(n_human=c["games"] // 2, n_sf=c["games"] // 2, seed=c["traj_seed"],
                 max_plies=c["max_plies"], n_piecedown=c.get("n_piecedown", 0), verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    test_games = np.flatnonzero(split == 2)
    game, pc = tr.game_of_row(), tr.piece_count()
    rows = np.flatnonzero(np.isin(game, test_games) & (pc <= 5) & (pc >= 3))
    rng = np.random.default_rng(0)
    if len(rows) > args.n_pos:
        rows = rng.choice(rows, args.n_pos, replace=False)
    print(f"[dtz] {len(rows):,} held-out-game positions with 3-5 pieces", flush=True)

    tb = TB()
    wdl_cls, dtz_abs, keep = [], [], []
    for r in rows:
        b = row_to_board(tr.tok[r], tr.glob[r])
        if not b.is_valid():
            continue
        try:
            w, d = tb.wdl_dtz(b)
        except Exception:
            continue
        if w is None:
            continue
        # mover POV: w>0 win for side to move, 0 draw, <0 loss -> class 0/1/2 like the poles
        wdl_cls.append(0 if w > 0 else (1 if w == 0 else 2))
        dtz_abs.append(abs(d) if d is not None else 0)
        keep.append(r)
    tb.close()
    keep = np.array(keep); wdl_cls = np.array(wdl_cls); dtz_abs = np.array(dtz_abs, float)
    print(f"[dtz] probed OK: {len(keep):,} | TB classes W {int((wdl_cls==0).sum())} "
          f"D {int((wdl_cls==1).sum())} L {int((wdl_cls==2).sum())}", flush=True)

    Z = _embed_rows(net, tr, keep, args.device)
    iqe = net.qhead.iqe if getattr(net, "dual", False) else net.iqe
    pn = c["pole_names"]; pi = [pn.index(n) for n in ("WIN", "DRAW", "LOSS")]
    sp = net.poles.poles.detach().float()
    with torch.no_grad():
        D = np.stack([iqe(Z.to(args.device), sp[[k]].expand(len(Z), -1).to(args.device))
                      .float().cpu().detach().numpy() for k in pi], 1)

    maj = max(np.bincount(wdl_cls)) / len(wdl_cls)
    acc = (D.argmin(1) == wdl_cls).mean()
    print(f"\n[dtz] TB-WDL committor acc = {acc:.3f} vs subset majority {maj:.3f} "
          f"(edge {acc-maj:+.3f})  <- EXACT labels, no blunders, no base-rate shortcut")
    for k, n in ((0, "TB-win "), (1, "TB-draw"), (2, "TB-loss")):
        m = (wdl_cls == k) & (dtz_abs > 0)
        if m.sum() < 40:
            print(f"[dtz] {n}: n={m.sum()} too few for DTZ corr"); continue
        rho = spearman(D[m, k], dtz_abs[m])
        rho_min = spearman(D[m].min(1), dtz_abs[m])
        print(f"[dtz] {n}: n={m.sum():>5}  Spearman(d->own pole, |DTZ|) = {rho:+.3f}   "
              f"(nearest-pole variant {rho_min:+.3f})")
    print("[dtz] positive rho = the field knows HOW FAR endings are where reality is computable")


if __name__ == "__main__":
    main()
