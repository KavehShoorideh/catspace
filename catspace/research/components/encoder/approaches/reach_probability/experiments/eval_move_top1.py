#!/usr/bin/env python
"""eval_move_top1.py -- held-out picks-SF's-move for a ConceptEvent checkpoint with a move head.

The one number that makes cef6 comparable to cpl1 (0.114) and the graph models: same sibling
cache, same by-game split, same top-1/top-3 readout.

    .venv/bin/python -m ...eval_move_top1 --ckpt artifacts/experiments/cef6_latest.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sib-k", type=int, default=40)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_event_field import ConceptEvent
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import factor_vocab
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    dev = args.device
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    n = len(tr.tok)
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    cr, ce, co, egs = factor_vocab(vocab)
    net = ConceptEvent(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
                       square_codes=A["square_codes"], hidden=A["hidden"],
                       n_res=len(vocab), factored=True, n_eg=len(egs),
                       cls_reach=cr, cls_eg=ce, cls_out=co, move_head=True).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()

    sb = np.load(paths.derived(f"sibling_contrast_{args.sib_k}_{n}.npz"))
    gor = tr.game_of_row()
    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    te = np.flatnonzero(test_g[gor[sb["p_row"]]])
    rng = np.random.default_rng(0)
    probe = te[rng.choice(len(te), min(args.n, len(te)), replace=False)]
    K = sb["neg_tok"].shape[1]
    D = {k: torch.from_numpy(sb[k]) for k in ("pos_tok", "pos_glb", "neg_tok", "neg_glb", "n_neg")}

    h1 = h3 = tot = 0
    with torch.no_grad():
        for a in range(0, len(probe), 96):
            ii = torch.from_numpy(probe[a:a + 96])
            B = len(ii)
            tk = torch.cat([D["pos_tok"][ii][:, None], D["neg_tok"][ii]], 1)
            gb = torch.cat([D["pos_glb"][ii][:, None], D["neg_glb"][ii]], 1)
            zq, _, _, _ = net.encode(tk.reshape(B * (K + 1), 64).long().to(dev),
                                     gb.reshape(B * (K + 1), 6).float().to(dev))
            sc = net.move_score(zq).squeeze(-1).view(B, K + 1)
            valid = torch.cat([torch.ones(B, 1, dtype=torch.bool),
                               torch.arange(K)[None, :].expand(B, K) < D["n_neg"][ii][:, None]],
                              1).to(dev)
            sc = sc.masked_fill(~valid, float("-inf"))
            rk = (sc > sc[:, :1]).sum(1)
            h1 += int((rk == 0).sum()); h3 += int((rk < 3).sum()); tot += B
    nl = float(D["n_neg"][torch.from_numpy(probe)].float().mean()) + 1
    print(f"\n  {args.ckpt.split('/')[-1]} step {ck.get('step')}  |  {tot:,} held-out parents")
    print(f"  picks SF's move: top1 {h1/tot:.3f}   top3 {h3/tot:.3f}   (chance {1/nl:.3f})")


if __name__ == "__main__":
    main()
