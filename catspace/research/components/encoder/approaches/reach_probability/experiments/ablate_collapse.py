#!/usr/bin/env python
"""ablate_collapse.py -- WHY do our codebooks collapse? (Kaveh 2026-08-15)

The same `VectorQuantize` class, same decay 0.9, same commitment 0.25, reaches 100% codebook
usage on a frozen lc0 trunk and collapses to ONE code inside jqt. So the cause is in the
training CONTEXT, not the quantizer. This starts from the configuration we know does NOT
collapse and adds the suspects back one at a time:

  A baseline        frozen features, standardised, recon + outcome        (known healthy)
  B +persistence    pull consecutive-ply latents together (w=3, as jqt)   -- a force whose
                    optimum IS "all latents identical", i.e. collapse
  C +moving trunk   a trainable layer in front of the encoder, so the
                    representation drifts under the codebook the way phi
                    does in jqt (EMA codebooks assume a stationary target)
  D -standardise    feed raw feature scales, as jqt does
  E all three       the jqt-like condition

Whichever row's usage% falls off a cliff is the cause.

    .venv/bin/python -m ...ablate_collapse
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=20000)
    ap.add_argument("--games", type=int, default=900)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--codes", type=int, default=256)
    ap.add_argument("--w-pers", type=float, default=3.0)
    ap.add_argument("--onnx", default="assets/engines/lc0/t1-256x10.onnx")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from vector_quantize_pytorch import VectorQuantize
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    import chess
    from lczerolens import LczeroBoard
    from catspace.research.components.encoder.approaches.reachability_field.src.field import (
        ReachabilityField)

    dev = args.device
    field = ReachabilityField(onnx=args.onnx)

    # ---- CONSECUTIVE position pairs, so persistence has something to act on --------------
    games = T.load_piecedown_games(args.games, seed=0)
    rng = np.random.default_rng(0)
    fens_a, fens_b, ys = [], [], []
    per = max(1, args.positions // max(len(games), 1))
    for g in games:
        res, ucis = g[1], g[2]
        b = chess.Board(g[5]) if len(g) > 5 and g[5] else chess.Board()
        cls = 0 if res == 1 else (2 if res == 0 else 1)
        keep = set(rng.choice(max(len(ucis) - 1, 1), size=min(per, max(len(ucis) - 1, 1)),
                              replace=False).tolist())
        for t, u in enumerate(ucis):
            try:
                mv = chess.Move.from_uci(u)
                if mv not in b.legal_moves:
                    break
                if t in keep:
                    fa = b.fen()
                    b.push(mv)
                    fens_a.append(fa); fens_b.append(b.fen()); ys.append(cls)
                else:
                    b.push(mv)
            except Exception:
                break
        if len(fens_a) >= args.positions:
            break
    fens_a, fens_b, ys = (fens_a[:args.positions], fens_b[:args.positions],
                          ys[:args.positions])
    print(f"[ablate] {len(fens_a):,} consecutive pairs from {len(games)} games", flush=True)

    def encode(fens):
        out = []
        for a in range(0, len(fens), 256):
            lb = [LczeroBoard(f) for f in fens[a:a + 256]]
            planes = [x.to_input_tensor().numpy() for x in lb]
            out.append(field.trunk_feats(planes).mean(dim=(2, 3)).detach().cpu().numpy())
        return torch.from_numpy(np.concatenate(out)).float()

    t0 = time.time()
    XA, XB = encode(fens_a), encode(fens_b)
    Y = torch.from_numpy(np.array(ys, np.int64))
    print(f"[ablate] encoded through the frozen trunk [{time.time()-t0:.0f}s]", flush=True)
    mu, sd = XA.mean(0), XA.std(0).clamp(min=1e-6)
    XAs, XBs = (XA - mu) / sd, (XB - mu) / sd
    maj = float((Y == torch.mode(Y).values).float().mean())

    CONFIGS = [
        ("A baseline",       False, False, True),
        ("B +persistence",   True,  False, True),
        ("C +moving trunk",  False, True,  True),
        ("D -standardise",   False, False, False),
        ("E all three",      True,  True,  False),
    ]
    print(f"\n{'config':>16} {'usage%':>8} {'perplex':>9} {'outcome':>9}  "
          f"(codes {args.codes}, majority {maj:.1%})")
    for label, use_pers, moving, standard in CONFIGS:
        torch.manual_seed(0)
        A, B = (XAs, XBs) if standard else (XA, XB)
        d_in = A.shape[1]
        drift = (nn.Sequential(nn.Linear(d_in, d_in), nn.GELU()) if moving
                 else nn.Identity())
        enc = nn.Sequential(nn.Linear(d_in, 128), nn.GELU(), nn.Linear(128, 8))
        dec = nn.Sequential(nn.Linear(8, 128), nn.GELU(), nn.Linear(128, d_in))
        head = nn.Linear(8, 3)
        q = VectorQuantize(dim=8, codebook_size=args.codes, decay=0.9,
                           commitment_weight=0.25)
        mods = nn.ModuleList([drift, enc, dec, head, q]).to(dev)
        opt = torch.optim.Adam(mods.parameters(), lr=3e-4)
        for _ in range(args.steps):
            i = torch.randint(0, len(A), (args.batch,))
            xa, xb, yb = A[i].to(dev), B[i].to(dev), Y[i].to(dev)
            ha = enc(drift(xa))
            zq, idx, vql = q(ha)
            loss = (nn.functional.mse_loss(dec(zq), xa)
                    + nn.functional.cross_entropy(head(zq), yb)
                    + vql.mean())
            if use_pers:
                # THE SUSPECT: jqt pulls consecutive-ply latents together (w_pers 3). Its
                # optimum is "every latent identical", which is exactly codebook collapse.
                hb = enc(drift(xb))
                loss = loss + args.w_pers * nn.functional.mse_loss(ha, hb)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ids, acc = [], []
            for a in range(0, min(len(A), 16000), 2048):
                xa, yb = A[a:a + 2048].to(dev), Y[a:a + 2048].to(dev)
                zq, idx, _ = q(enc(drift(xa)))
                ids.append(idx.reshape(-1).cpu())
                acc.append(float((head(zq).argmax(-1) == yb).float().mean()))
            ids = torch.cat(ids)
            uq, cnt = torch.unique(ids, return_counts=True)
            p = cnt.float() / cnt.sum()
            perp = float(torch.exp(-(p * p.log()).sum()))
            print(f"{label:>16} {100*len(uq)/args.codes:>7.1f}% {perp:>9.1f} "
                  f"{np.mean(acc):>8.1%}")
    print("\n[ablate] READ: the row whose usage% falls off a cliff is why jqt collapses.")


if __name__ == "__main__":
    main()
