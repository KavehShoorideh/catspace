#!/usr/bin/env python
"""experiments/arch_bakeoff.py -- ARCHITECTURE BAKE-OFF (Kaveh 2026-07-25: 'change the
architecture to get it working... make it smaller (d~16)... transformer like Leela to
encode the board instead of a CNN... represent something other than a scalar... try
different architectures and see which one wins').

The analysis: the field's scalar-distance objective collapsed the representation to ~3
effective dims, so it has no middlegame signal. This bake-off tests the fix on the direct
"does it understand positions" metric -- HELD-OUT MOVE PREDICTION (top-1/top-5) on lichess
move-selection data -- across (backbone x embed-dim x target):
  backbone: transformer (square tokens + attention, Leela-style) vs CNN (current)
  target:   policy (predict played move -- dense) [+ value (WDL) multitask option]
Reports, per config: held-out top1/top5 move acc AND the effective rank of the pooled
board embedding on middlegames (the quantity that was ~5 for the scalar field). A config
that predicts moves well AND keeps high embedding rank = a representation with real
middlegame structure. Small/short by design (fail-fast); scale the winner.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import decode_planes
from catspace.nn.fb import pick_device


# ---- input: packed bitboards -> per-square piece-type tokens (0=empty, 1..12 pieces)
def tokens(packed, meta):
    pl = decode_planes(np.atleast_2d(packed))            # (N,12,8,8)
    n = pl.shape[0]
    ids = np.zeros((n, 64), np.int64)
    flat = pl.reshape(n, 12, 64)
    for c in range(12):
        ids[flat[:, c] > 0.5] = 0                        # placeholder, set below
    # square s gets piece-channel+1 where a piece sits, else 0
    occ = flat.argmax(1) + 1
    any_piece = flat.max(1) > 0.5
    ids = np.where(any_piece, occ, 0)
    stm = np.atleast_2d(meta)[:, 0].astype(np.int64)     # 0=white,1=black to move
    return ids, stm


# ---- backbones -> per-square features (N,64,d) + pooled (N,d)
class TransformerBackbone(nn.Module):
    def __init__(self, d=64, layers=4, heads=4):
        super().__init__()
        self.tok = nn.Embedding(13, d)
        self.pos = nn.Embedding(64, d)
        self.stm = nn.Embedding(2, d)
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.register_buffer("posid", torch.arange(64))

    def forward(self, ids, stm):
        h = self.tok(ids) + self.pos(self.posid)[None] + self.stm(stm)[:, None]
        h = self.enc(h)                                  # (N,64,d)
        return h, h.mean(1)


class CNNBackbone(nn.Module):
    def __init__(self, d=64, blocks=6):
        super().__init__()
        self.emb = nn.Embedding(13, d)
        self.stm = nn.Embedding(2, d)
        self.blocks = nn.ModuleList([nn.Conv2d(d, d, 3, padding=1) for _ in range(blocks)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, d) for _ in range(blocks)])

    def forward(self, ids, stm):
        n = ids.shape[0]
        h = (self.emb(ids).transpose(1, 2).reshape(n, -1, 8, 8)
             + self.stm(stm)[:, :, None, None])
        for c, g in zip(self.blocks, self.norm):
            h = torch.relu(h + g(c(h)))
        feat = h.reshape(n, -1, 64).transpose(1, 2)      # (N,64,d)
        return feat, feat.mean(1)


class Net(nn.Module):
    def __init__(self, backbone, d, value=False):
        super().__init__()
        self.bb = backbone
        self.mv = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.value = nn.Linear(d, 3) if value else None

    def forward(self, ids, stm, mvf, mvt, nm):
        feat, pooled = self.bb(ids, stm)                 # (N,64,d), (N,d)
        N, L = mvf.shape
        ff = torch.gather(feat, 1, mvf[:, :, None].expand(-1, -1, feat.shape[-1]))
        ft = torch.gather(feat, 1, mvt[:, :, None].expand(-1, -1, feat.shape[-1]))
        logit = self.mv(torch.cat([ff, ft], -1))[:, :, 0]     # (N,L)
        mask = torch.arange(L, device=mvf.device)[None] >= nm[:, None]
        logit = logit.masked_fill(mask, -1e9)
        val = self.value(pooled) if self.value is not None else None
        return logit, val, pooled


def eff_rank(E):
    E = E - E.mean(0, keepdims=True)
    s = np.linalg.svd(E, compute_uv=False)
    p = s / s.sum(); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", choices=["xf", "cnn"], default="xf")
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--value", action="store_true")
    ap.add_argument("--data", default="data/derived/move_selection_full_v1.npz")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    z = np.load(args.data)
    ids, stm = tokens(z["packed"], z["meta"])
    mvf, mvt, nm, played = z["mv_from"], z["mv_to"], z["n_moves"], z["played"]
    res = None
    n = len(played)
    te = rng.random(n) < 0.05
    tr = ~te
    idx_tr = np.flatnonzero(tr); idx_te = np.flatnonzero(te)
    tag = f"{args.backbone}-d{args.d}-L{args.layers}{'-val' if args.value else ''}"
    print(f"[bakeoff {tag}] {tr.sum()} train / {te.sum()} test", flush=True)

    bb = (TransformerBackbone(args.d, args.layers) if args.backbone == "xf"
          else CNNBackbone(args.d, args.layers))
    net = Net(bb, args.d, value=args.value).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def batch(idx):
        g = lambda a, dt=torch.int64: torch.from_numpy(a[idx].astype(np.int64)).to(dev)
        return (g(ids), g(stm), g(mvf.astype(np.int64)), g(mvt.astype(np.int64)),
                g(nm), torch.from_numpy(played[idx].astype(np.int64)).to(dev))

    for s in range(args.steps):
        bi = idx_tr[rng.integers(0, len(idx_tr), args.batch)]
        di, ds, df, dt, dn, dy = batch(bi)
        logit, val, _ = net(di, ds, df, dt, dn)
        loss = F.cross_entropy(logit, dy)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0:
            print(f"  step {s} loss {float(loss):.3f} [{time.time()-t0:.0f}s]", flush=True)

    # ---- held-out move accuracy
    net.eval(); top1 = top5 = ntot = 0
    with torch.no_grad():
        for s in range(0, len(idx_te), 1024):
            bi = idx_te[s:s + 1024]
            di, ds, df, dt, dn, dy = batch(bi)
            logit, _, _ = net(di, ds, df, dt, dn)
            top1 += (logit.argmax(1) == dy).sum().item()
            t5 = logit.topk(5, 1).indices
            top5 += (t5 == dy[:, None]).any(1).sum().item()
            ntot += len(bi)
    # ---- embedding effective rank on held-out MIDDLEGAME positions (>=20 pieces)
    npc = decode_planes(z["packed"][idx_te]).reshape(len(idx_te), 12, 64).sum((1, 2))
    mid = idx_te[npc >= 20][:1500]
    with torch.no_grad():
        di, ds = torch.from_numpy(ids[mid]).to(dev), torch.from_numpy(stm[mid]).to(dev)
        _, pooled = net.bb(di, ds)
        er = eff_rank(pooled.cpu().numpy())
    print(f"VERDICT ARCH {tag}: top1 {top1/ntot:.3f} top5 {top5/ntot:.3f} "
          f"| embed_eff_rank(mid) {er:.1f}/{args.d} | {npar/1e6:.2f}M params "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
