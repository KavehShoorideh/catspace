#!/usr/bin/env python
"""eval_jepa.py -- what does the JEPA loss MEAN? (Kaveh 2026-08-13: "how well was the
training loss of the jepa predictor? which concepts? how far out?")

The predictor is one-ply, global-stream, embedding-space: pred(zq_parent, move) vs the
EMA-branch child embedding. A raw MSE is unreadable alone, so this scores it against:
    copy-parent   MSE(zq_parent, zq_child_target) -- metastability makes copying strong;
                  JEPA only earns its keep by beating it ON THE FLIPS
    shuffled-move same predictor, wrong moves -- does the move input actually condition?
And in CODE space (nearest codebook vector per head): flip-prediction accuracy -- of the
heads that DO change codes this ply, how often does the predictor name the new code?

    .venv/bin/python -m ...eval_jepa --ckpt artifacts/experiments/reach_jqt4_latest.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--transitions", default="data/derived/game_transitions_4652261.npz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import os, re
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.jqt import (
        JQTModule)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
        load_net)
    dev = args.device
    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    stem = re.sub(r"_(latest|step\d+)$", "", base)
    model, _ = load_net(args.ckpt, dev)
    model.eval()
    pj = torch.load(next(p for p in (base + "_jqt.pt", stem + "_jqt.pt")
                         if os.path.exists(p)), map_location=dev, weights_only=False)
    jqt = JQTModule(d_model=pj["d_in"], heads=pj["heads"], codes=pj["codes"], d=pj["d"],
                    square_codes=pj.get("square_codes", 0),
                    piece_codes=pj.get("piece_codes", 0)).to(dev)
    jqt.load_state_dict(pj["state_dict"], strict=False)
    jqt.eval()
    H, D = pj["heads"], pj["d_code"] if "d_code" in pj else jqt.d_code

    gt = np.load(args.transitions)
    # the trainer's exact corpus (cache hit): sf-only 4000 + piecedown 45906, seed 0
    tr = T.build(n_human=0, n_sf=4000, seed=0, cache=True, max_plies=400,
                 n_piecedown=45906)
    rng = np.random.default_rng(0)
    ts = rng.choice(len(gt["par"]), size=args.pairs, replace=False)
    par = gt["par"][ts]
    mid = torch.from_numpy(gt["mid"][ts].astype(np.int64)).to(dev)
    rows = np.concatenate([par, par + 1])
    tok = torch.from_numpy(tr.tok[rows].astype(np.int64)).to(dev)
    glob = torch.from_numpy(tr.glob[rows].astype(np.float32)).to(dev)
    n = args.pairs

    with torch.no_grad():
        phi = model.backbone(tok, glob)
        _, zq_par_flat, ids_par, _ = jqt.quantize(phi[:n])
        # exact JEPA target: EMA trunk -> EMA concept encoder -> live-EMA codebooks
        t_enc = getattr(model, "t_enc", None)
        phi_ch_t = t_enc(tok[n:], glob[n:]) if t_enc is not None else phi[n:]
        zq_t, ids_ch = jqt.target_codes(phi_ch_t)
        zq_t = zq_t.reshape(n, -1)
        pred = jqt.predict_child(zq_par_flat, mid)
        mid_sh = mid[torch.randperm(n)]
        pred_sh = jqt.predict_child(zq_par_flat, mid_sh)
        mse_pred = float((pred - zq_t).pow(2).mean())
        mse_copy = float((zq_par_flat - zq_t).pow(2).mean())
        mse_shuf = float((pred_sh - zq_t).pow(2).mean())
        # code-space: nearest codebook vector per head
        ids_pred = torch.zeros(n, H, dtype=torch.long)
        ph = pred.view(n, H, -1)
        for h in range(H):
            cb = jqt.vq[h].codebook            # (C, d_code)
            d2 = (ph[:, h][:, None] - cb[None]).pow(2).sum(-1)
            ids_pred[:, h] = d2.argmin(-1).cpu()
        ids_par = ids_par.cpu(); ids_ch = ids_ch.cpu()
        flip = ids_ch != ids_par                             # heads that change this ply
        stay = ~flip
        acc_all = float((ids_pred == ids_ch).float().mean())
        acc_copy_all = float((ids_par == ids_ch).float().mean())
        acc_flip = float((ids_pred[flip] == ids_ch[flip]).float().mean()) \
            if flip.any() else float("nan")
        acc_stay = float((ids_pred[stay] == ids_ch[stay]).float().mean())
        flip_rate = float(flip.float().mean())

    print(f"\n[jepa-eval] {args.ckpt} | {n} transitions, global stream, ONE ply ahead")
    print(f"  embedding MSE: pred {mse_pred:.4f} | copy-parent {mse_copy:.4f} | "
          f"shuffled-move {mse_shuf:.4f}")
    print(f"  code accuracy: pred {acc_all:.1%} vs copy-parent {acc_copy_all:.1%} "
          f"(flip rate {flip_rate:.1%})")
    print(f"  ON THE FLIPS (the real test): pred names the NEW code {acc_flip:.1%} "
          f"(copy-parent scores 0% here by definition); on stays {acc_stay:.1%}")
    print(f"  VERDICT: move-conditioning gain = shuffled/pred MSE ratio "
          f"{mse_shuf / max(mse_pred, 1e-9):.2f}x")


if __name__ == "__main__":
    main()
