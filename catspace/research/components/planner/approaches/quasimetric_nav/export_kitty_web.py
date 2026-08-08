#!/usr/bin/env python
"""export_kitty_web.py -- export a KittyChess checkpoint to JSON for the in-browser engine.

Ships every tensor the JS forward needs (token/square/glob embeddings, CLS, transformer layers,
proj_b, IQE-B params, W/D/L poles) plus a TEST VECTOR: the startpos three-pole distances computed
by PyTorch. The page re-computes them in JS at load and refuses to play unless they match to
1e-2 -- a silent porting mismatch would otherwise play garbage with a straight face.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
import chess


def arr(t):
    return [round(float(x), 5) for x in t.detach().float().cpu().numpy().ravel()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net, pay = load_net(args.ckpt, "cpu")
    net.eval()
    c = pay["cfg"]
    bk = net.enc
    d = bk.d
    W = {"d": d, "layers": c["layers"], "heads": c["heads"], "head_d": c["d"],
         "piece_emb": arr(bk.piece_emb.weight), "sq_emb": arr(bk.sq_emb.weight),
         "glob_w": arr(bk.glob_proj.weight), "glob_b": arr(bk.glob_proj.bias),
         "cls": arr(bk.cls), "out_ln_w": arr(bk.out.weight), "out_ln_b": arr(bk.out.bias),
         "L": []}
    for lyr in bk.tr.layers:
        W["L"].append({
            "ln1_w": arr(lyr.norm1.weight), "ln1_b": arr(lyr.norm1.bias),
            "ln2_w": arr(lyr.norm2.weight), "ln2_b": arr(lyr.norm2.bias),
            "qkv_w": arr(lyr.self_attn.in_proj_weight), "qkv_b": arr(lyr.self_attn.in_proj_bias),
            "ao_w": arr(lyr.self_attn.out_proj.weight), "ao_b": arr(lyr.self_attn.out_proj.bias),
            "m1_w": arr(lyr.linear1.weight), "m1_b": arr(lyr.linear1.bias),
            "m2_w": arr(lyr.linear2.weight), "m2_b": arr(lyr.linear2.bias)})
    W["proj_w"] = arr(net.proj_b.weight); W["proj_b"] = arr(net.proj_b.bias)
    iqe = net.iqe_b if getattr(net, "split_head", False) else net.iqe
    W["split"] = bool(getattr(net, "split_head", False))
    W["iqe_components"] = iqe.components
    W["iqe_alpha"] = float(torch.sigmoid(iqe.alpha_logit))
    W["iqe_scale"] = float(torch.exp(iqe.log_scale))
    pn = c["pole_names"]
    P = net.poles.poles.detach().float()
    W["poles"] = {n: arr(P[pn.index(n)]) for n in ("WIN", "DRAW", "LOSS")}

    # test vector: startpos three-pole distances through the exact same path the page will use
    tk, gl = tokenize(chess.Board())
    with torch.no_grad():
        phi = bk(torch.from_numpy(tk[None].astype(np.int64)),
                 torch.from_numpy(gl[None].astype(np.float32)))
        z = net.proj_b(phi)
        h = z.shape[-1] // 2
        zb = z[:, h:] if W["split"] else z
        dd = {}
        for n in ("WIN", "DRAW", "LOSS"):
            pv = P[pn.index(n)][None]
            pb = pv[:, h:] if W["split"] else pv
            dd[n] = float(iqe(zb, pb))
    W["test"] = {"tok": tk.tolist(), "glob": gl.tolist(),
                 "d": {n: round(v, 3) for n, v in dd.items()}}
    with open(args.out, "w") as f:
        json.dump(W, f)
    import os
    print(f"[export] {args.out} ({os.path.getsize(args.out)/2**20:.1f} MB) | "
          f"startpos d(W/D/L) = {dd['WIN']:.2f}/{dd['DRAW']:.2f}/{dd['LOSS']:.2f}")


if __name__ == "__main__":
    main()
