#!/usr/bin/env python
"""catspace/research/tools/embeddings/embed.py -- produce a REPRESENTATION FILE (the shared probe convention:
npz with `emb` (N,d) + aligned label columns) from any encoder we own.

Encoders:
  --encoder jepa   : a JEPA T1 checkpoint (catspace/encoder/jepa.py), any step
  --encoder trunk  : the frozen lc0 trunk + IQE head (ReachabilityField)

Sources (positions + labels ride along automatically):
  --source corpus --rows {cx,bd}  : the JEPA T1 corpus npz (contexts / boundary)
  --source fens                   : a text file of FENs (no labels)

Examples:
  catspace/research/tools/embeddings/embed.py --encoder jepa --ckpt artifacts/experiments/jepa_t1_latest.pt \
      --source corpus --rows bd --limit 20000 --out /tmp/jepa_bd.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from catspace.research.tools.training_infra.train.scaffold import resolve_device                      # noqa: E402
from catspace.io import paths


def jepa_encode(ckpt, tok, glob, dev, bs=512):
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import JepaT1
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    model = JepaT1(**{k: ck["cfg"][k] for k in ("d", "layers", "n_class")}).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    out = np.zeros((len(tok), ck["cfg"]["d"]), np.float32)
    with torch.no_grad():
        for i in range(0, len(tok), bs):
            out[i:i+bs] = model.enc(torch.as_tensor(tok[i:i+bs]).to(dev),
                                    torch.as_tensor(glob[i:i+bs]).to(dev)).cpu().numpy()
    return out


def trunk_encode(fens, bs=512):
    from catspace.research.components.encoder.approaches.reachability_field.src import ReachabilityField
    from lczerolens import LczeroBoard
    rf = ReachabilityField()
    out = np.zeros((len(fens), 64), np.float32)
    for i in range(0, len(fens), bs):
        out[i:i+bs] = rf.phi([LczeroBoard(f) for f in fens[i:i+bs]]).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["jepa", "trunk"], required=True)
    ap.add_argument("--ckpt", default=paths.experiment("jepa_t1_latest.pt"))
    ap.add_argument("--source", choices=["corpus", "fens"], default="corpus")
    ap.add_argument("--data", default=paths.derived("checkpoints/jepa_t1_corpus.npz"))
    ap.add_argument("--rows", choices=["cx", "bd"], default="cx")
    ap.add_argument("--fens", default="", help="text file of FENs (--source fens)")
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    dev = resolve_device("auto")

    labels = {}
    if args.source == "corpus":
        d = dict(np.load(args.data, allow_pickle=True))
        p = args.rows + "_"
        n = len(d[p + "tok"])
        sel = np.sort(rng.choice(n, min(args.limit, n), replace=False))
        tok, glob = d[p + "tok"][sel], d[p + "glob"][sel]
        for k, v in d.items():
            if k.startswith(p) and k not in (p + "tok", p + "glob") and len(v) == n:
                labels[k[len(p):]] = v[sel]
        if args.encoder == "trunk":
            raise SystemExit("trunk encoder needs FENs; corpus stores tokens only "
                             "(use the mined checkpoints npz + --source fens)")
        emb = jepa_encode(args.ckpt, tok, glob, dev)
    else:
        fens = [ln.strip() for ln in open(args.fens) if ln.strip()][:args.limit]
        if args.encoder == "jepa":
            import chess
            from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
            tg = [tokenize(chess.Board(f)) for f in fens]
            emb = jepa_encode(args.ckpt, np.stack([t for t, _ in tg]),
                              np.stack([g for _, g in tg]), dev)
        else:
            emb = trunk_encode(fens)
    np.savez_compressed(args.out, emb=emb, **labels,
                        meta_encoder=args.encoder, meta_source=args.source)
    print(f"wrote {args.out}: emb {emb.shape}, labels {sorted(labels)} "
          f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
