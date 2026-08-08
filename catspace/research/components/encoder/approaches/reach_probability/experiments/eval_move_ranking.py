#!/usr/bin/env python
"""eval_move_ranking.py -- FAITHFULNESS: does argmin-d pick the moves the oracle picks?

Kaveh 2026-08-07: 'absolute error is not really as important as relative rank... do we properly
rank the moves the same as the engine would at each position?' This is the primary-endpoint
metric (faithfulness), evaluated where an EXACT oracle exists: 3-5 piece positions from held-out
games, oracle = Syzygy.

Per position: enumerate legal moves, embed each child, rank children by d(child -> LOSS pole)
ascending -- the child's mover is the OPPONENT, so 'opponent is close to losing' = good for us.
Oracle ranks by the opponent's WDL after the move (ascending: their loss first), DTZ as
tie-break. Reported:
  top1-optimal  -- how often our #1 move preserves the best achievable TB outcome
  random        -- expected top1 of a random-move baseline (fraction of optimal moves)
  Kendall-tau   -- rank agreement between field ordering and oracle class ordering
"""
from __future__ import annotations

import argparse

import chess
import numpy as np
import torch

from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.eval_dtz_gate import (
    row_to_board)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize


def kendall(a, b):
    n = len(a); c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0: c += 1
            elif s < 0: d += 1
    t = c + d
    return (c - d) / t if t else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-pos", type=int, default=600)
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
                 max_plies=c["max_plies"], verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    test = np.flatnonzero(split == 2)
    game, pc = tr.game_of_row(), tr.piece_count()
    rows = np.flatnonzero(np.isin(game, test) & (pc <= 5) & (pc >= 3))
    rng = np.random.default_rng(0)
    rng.shuffle(rows)

    def _enc_children(net, tok_t, glob_t):
        if args.cond_elo is not None and getattr(net, "dual", False):
            cval = (args.cond_elo - 1500.0) / 500.0
            cond = torch.full((len(tok_t), net.qhead.proj_delta.in_features
                               - net.qhead.proj_base.in_features), cval, device=args.device)
            return net.encode_dual(tok_t, glob_t, cond)[1]
        return net.encode_q(tok_t, glob_t)

    iqe = net.qhead.iqe if getattr(net, "dual", False) else net.iqe
    pn = c["pole_names"]
    pL = net.poles.poles.detach().float()[pn.index("LOSS")]

    tb = TB()
    top1 = rand_base = 0.0
    taus, n_done = [], 0
    for r in rows:
        if n_done >= args.n_pos:
            break
        b = row_to_board(tr.tok[r], tr.glob[r])
        if not b.is_valid():
            continue
        moves = list(b.legal_moves)
        if len(moves) < 3:
            continue
        toks, globs, oracle = [], [], []
        ok = True
        for mv in moves:
            b.push(mv)
            try:
                w, dz = tb.wdl_dtz(b)          # child's mover POV = the opponent
            except Exception:
                w = None
            if w is None:
                ok = False; b.pop(); break
            tk, gl = tokenize(b)
            toks.append(tk); globs.append(gl)
            # oracle score: opponent's wdl ascending (their loss = our win first);
            # tie-break: when they lose, faster is better (small |dtz|); when they win, slower
            oracle.append((w, (abs(dz) if w < 0 else -abs(dz)) if dz is not None else 0))
            b.pop()
        if not ok or len(toks) < 3:
            continue
        with torch.no_grad():
            z = _enc_children(net, torch.from_numpy(np.array(toks).astype(np.int64)).to(args.device),
                             torch.from_numpy(np.array(globs).astype(np.float32)).to(args.device))
            d = iqe(z, pL.expand(len(z), -1).to(args.device)).float().cpu().numpy()
        osc = np.array([o[0] for o in oracle], float)      # opponent wdl: -2 best for us
        best = osc == osc.min()                            # TB-optimal set
        top1 += float(best[int(np.argmin(d))])
        rand_base += best.mean()
        # oracle full ordering for tau: wdl primary, dtz tiebreak
        okey = np.array([o[0] * 1000 + o[1] for o in oracle], float)
        taus.append(kendall(d, okey))
        n_done += 1
    tb.close()

    taus = np.array([t for t in taus if np.isfinite(t)])
    print(f"[rank] {n_done} held-out 3-5 piece positions, oracle = Syzygy")
    print(f"[rank] top1-optimal: {top1/n_done:.3f}   random baseline: {rand_base/n_done:.3f}   "
          f"edge {top1/n_done - rand_base/n_done:+.3f}")
    print(f"[rank] Kendall tau (field order vs oracle order): mean {taus.mean():+.3f} "
          f"(random = 0.000, n {len(taus)})")


if __name__ == "__main__":
    main()
