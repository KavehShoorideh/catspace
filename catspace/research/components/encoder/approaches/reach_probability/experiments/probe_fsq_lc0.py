#!/usr/bin/env python
"""probe_fsq_lc0.py -- FSQ vs VQ on a FROZEN lc0 trunk (Kaveh 2026-08-15: "test fsq,
increase codebook size, use the existing lc0 trunk to test it out").

WHY A FROZEN TRUNK. Every collapse we have hit was entangled: the representation, the
codebooks and half a dozen objectives all moved together, so "the codebook collapsed" never
had a single cause. Freezing lc0's trunk removes the representation as a variable entirely.
What is left is exactly the question asked: at a given codebook size, which quantizer keeps
its codes alive, and do those codes carry the outcome?

WHY BIGGER CODEBOOKS. FSQ's own Fig. 3 says it beats VQ only ABOVE ~2^10 = 1024 codes and
that "VQ marginally outperforms FSQ" below. Our heads carried 16-64. So the honest test
sweeps INTO the regime where FSQ is supposed to win.

Reported per configuration:
    usage%      fraction of the codebook ever used   (VQ collapse shows up here)
    perplexity  exp(entropy of code usage)           (effective vocabulary size)
    recon       MSE back to the frozen trunk feature
    outcome     W/D/L accuracy from the CODES ALONE, vs the majority baseline

    .venv/bin/python -m ...probe_fsq_lc0 --positions 40000
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

# (label, kind, spec) -- FSQ specs are the paper's Table 1 level sets
CONFIGS = [
    ("VQ   256", "vq", 256),
    ("FSQ  256", "fsq", [8, 6, 5]),
    ("VQ  1024", "vq", 1024),
    ("FSQ 1024", "fsq", [8, 5, 5, 5]),
    ("VQ  4096", "vq", 4096),
    ("FSQ 4096", "fsq", [7, 5, 5, 5, 5]),
    ("VQ 16384", "vq", 16384),
    ("FSQ 16384", "fsq", [8, 8, 8, 6, 5]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=40000)
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--onnx", default="assets/engines/lc0/t1-256x10.onnx")
    ap.add_argument("--objective", choices=["both", "outcome", "recon"], default="outcome",
                    help="what the CODES are trained to carry. 'recon' makes them encode the "
                         "whole trunk feature -- the same mistake as reconstructing the whole "
                         "board (Kaveh's value-equivalence principle: encode the position only "
                         "insofar as it helps you win). 'outcome' is the honest test.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from vector_quantize_pytorch import VectorQuantize, FSQ
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    import chess
    from lczerolens import LczeroBoard
    from catspace.research.components.encoder.approaches.reachability_field.src.field import (
        ReachabilityField)

    dev = args.device
    # ---- frozen lc0 trunk: the representation is FIXED, so nothing here can drift --------
    field = ReachabilityField(onnx=args.onnx)
    print(f"[fsq] frozen trunk {args.onnx}", flush=True)

    # ---- positions WITH outcomes, replayed so we have real boards for lc0 ---------------
    games = T.load_piecedown_games(args.games, seed=0)
    rng = np.random.default_rng(0)
    boards, ys = [], []
    per = max(1, args.positions // max(len(games), 1))
    for g in games:
        res, ucis = g[1], g[2]
        start = g[5] if len(g) > 5 and g[5] else None
        b = chess.Board(start) if start else chess.Board()
        cls = 0 if res == 1 else (2 if res == 0 else 1)          # W / D / L, white POV
        keep = set(rng.choice(len(ucis), size=min(per, len(ucis)), replace=False).tolist())
        ok = True
        for t, u in enumerate(ucis):
            try:
                mv = chess.Move.from_uci(u)
                if mv not in b.legal_moves:
                    ok = False; break
                if t in keep:
                    boards.append(b.fen()); ys.append(cls)
                b.push(mv)
            except Exception:
                ok = False; break
        if len(boards) >= args.positions:
            break
    boards, ys = boards[:args.positions], ys[:args.positions]
    print(f"[fsq] {len(boards):,} positions replayed from {len(games)} games", flush=True)

    # ---- one pass through the FROZEN trunk ------------------------------------------------
    t0 = time.time()
    feats = []
    B = 256
    for a in range(0, len(boards), B):
        lb = [LczeroBoard(f) for f in boards[a:a + B]]
        planes = [b.to_input_tensor().numpy() for b in lb]
        # RAW trunk features (B, C, 8, 8), mean-pooled over the 64 squares -> (B, C).
        # phi() would instead give a previously-TRAINED 64-d IQE head output, which is a
        # different representation and not what "use the lc0 trunk" means.
        tf = field.trunk_feats(planes)
        feats.append(tf.mean(dim=(2, 3)).detach().cpu().numpy())
        if a == 0:
            print(f"[fsq] trunk feature dim = {feats[0].shape[1]}", flush=True)
    X = torch.from_numpy(np.concatenate(feats)).float()
    Y = torch.from_numpy(np.array(ys[:len(X)], np.int64))
    print(f"[fsq] {len(X):,} positions encoded [{time.time()-t0:.0f}s]", flush=True)
    X = (X - X.mean(0)) / X.std(0).clamp(min=1e-6)     # unit scale: recon MSE ~1, comparable
    d_in = X.shape[1]                                   # to the cross-entropy it competes with
    maj = float((Y == torch.mode(Y).values).float().mean())
    print(f"[fsq] feature dim {d_in}, standardised (recon and CE now the same order)",
          flush=True)

    print(f"\n{'config':>10} {'usage%':>8} {'perplex':>9} {'recon':>9} {'outcome':>9}"
          f"  (majority {maj:.1%})")
    for label, kind, spec in CONFIGS:
        torch.manual_seed(0)
        if kind == "fsq":
            q = FSQ(levels=list(spec))
            d_lat, size = len(spec), int(np.prod(spec))
        else:
            d_lat, size = 8, int(spec)
            q = VectorQuantize(dim=d_lat, codebook_size=size, decay=0.9,
                               commitment_weight=0.25)
        enc = nn.Sequential(nn.Linear(d_in, 128), nn.GELU(), nn.Linear(128, d_lat))
        dec = nn.Sequential(nn.Linear(d_lat, 128), nn.GELU(), nn.Linear(128, d_in))
        head = nn.Linear(d_lat, 3)                    # outcome FROM THE QUANTISED CODE ONLY
        mods = nn.ModuleList([enc, dec, head, q]).to(dev)
        opt = torch.optim.Adam(mods.parameters(), lr=3e-4)
        for st in range(args.steps):
            i = torch.randint(0, len(X), (args.batch,))
            xb, yb = X[i].to(dev), Y[i].to(dev)
            h = enc(xb)
            out = q(h.unsqueeze(1) if kind == "fsq" else h)
            zq, idx = out[0], out[1]
            if kind == "fsq":
                zq = zq.squeeze(1)
            vql = out[2] if (kind == "vq" and len(out) > 2) else torch.zeros((), device=dev)
            _rec = nn.functional.mse_loss(dec(zq), xb)
            _ce = nn.functional.cross_entropy(head(zq), yb)
            loss = ((_rec if args.objective in ("both", "recon") else 0.0 * _rec)
                    + (_ce if args.objective in ("both", "outcome") else 0.0 * _ce)
                    + (vql.mean() if vql.numel() else vql))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ids, rec, acc = [], [], []
            for a in range(0, min(len(X), 20000), 2048):
                xb, yb = X[a:a + 2048].to(dev), Y[a:a + 2048].to(dev)
                h = enc(xb)
                out = q(h.unsqueeze(1) if kind == "fsq" else h)
                zq, idx = out[0], out[1]
                if kind == "fsq":
                    zq = zq.squeeze(1)
                ids.append(idx.reshape(len(xb), -1).cpu())
                rec.append(float(nn.functional.mse_loss(dec(zq), xb)))
                acc.append(float((head(zq).argmax(-1) == yb).float().mean()))
            ids = torch.cat(ids)
            flat = ids[:, 0] if ids.shape[1] == 1 else ids[:, 0]
            uniq, cnt = torch.unique(flat, return_counts=True)
            p = cnt.float() / cnt.sum()
            perp = float(torch.exp(-(p * p.log()).sum()))
            print(f"{label:>10} {100*len(uniq)/size:>7.1f}% {perp:>9.1f} "
                  f"{np.mean(rec):>9.4f} {np.mean(acc):>8.1%}")
    print("\n[fsq] READ: VQ usage% collapsing as size grows is the pathology FSQ removes by "
          "construction. Outcome accuracy says whether the codes carry the thing we care "
          "about at all.")


if __name__ == "__main__":
    main()
