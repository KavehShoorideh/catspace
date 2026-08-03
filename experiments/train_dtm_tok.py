#!/usr/bin/env python
"""experiments/train_dtm_tok.py -- PIECE-TOKEN TRANSFORMER for the last-mile DTM
(Kaveh 2026-07-25: 'shouldn't we train a transformer for this instead of a CNN?').

Endgames are SPARSE (3-7 pieces on 64 squares) and DTM is RELATIONAL (opposition,
cutoffs, king distance): tokens = one per piece (type+color emb + square emb) + a
side-to-move CLS token; a small TransformerEncoder computes pairwise relations in one
hop; CLS head regresses dtm/scale. Same npz, same scale, same VERDICT protocol as
train_dtm_cnn -- attribution stays clean: CNN v2->v3 = the data fix, CNN v3 vs TOK v3 =
the architecture. Cross-class relation sharing is the bet for the general nucleus (v4).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
from catspace.research.tools.stats_eval.tracking import track_run

MAX_TOKENS = 8          # <=7 pieces in <=6-man + CLS


class DTMTok(nn.Module):
    def __init__(self, d: int = 128, heads: int = 4, layers: int = 3, seed: int = 0):
        torch.manual_seed(seed)
        super().__init__()
        self.config = dict(d=d, heads=heads, layers=layers, seed=seed)
        self.emb_piece = nn.Embedding(13, d)     # 0 CLS, 1-6 white, 7-12 black
        self.emb_sq = nn.Embedding(65, d)        # 0 CLS slot, 1-64 squares
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                         batch_first=True, dropout=0.0, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, 1))

    def forward(self, piece_ids, sq_ids, pad):
        tok = self.emb_piece(piece_ids) + self.emb_sq(sq_ids)
        h = self.enc(tok, src_key_padding_mask=pad)
        return self.head(h[:, 0, :])[:, 0]       # CLS


def tokenize(P, M):
    n = len(P)
    pid = np.zeros((n, MAX_TOKENS), np.int64)
    sqi = np.zeros((n, MAX_TOKENS), np.int64)
    pad = np.ones((n, MAX_TOKENS), bool)
    for i in range(n):
        b = board_from_packed(P[i], M[i])
        pad[i, 0] = False                        # CLS
        j = 1
        for sq, p in sorted(b.piece_map().items()):
            pid[i, j] = p.piece_type + (0 if p.color else 6)
            sqi[i, j] = sq + 1
            pad[i, j] = False
            j += 1
        if not b.turn:                            # side-to-move in the CLS square slot
            sqi[i, 0] = 0; pid[i, 0] = 0
        # white-to-move CLS keeps ids 0/0; black-to-move flagged via piece id 12+... keep
        # simple: encode stm by giving CLS sq id 64 when black to move
        sqi[i, 0] = 64 if not b.turn else 0
    return pid, sqi, pad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame_v3.npz")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--out", default="data/derived/sep/dtm_tok_v3.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from contextlib import ExitStack
    stack = ExitStack()
    trk = stack.enter_context(track_run("dtm_tok", args, run_name=Path(args.out).stem))
    t0 = time.time(); dev = pick_device(args.device)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    Ps, Ms, Ds = [], [], []
    for path in args.dtm_npz.split(","):
        z = np.load(path.strip())
        Ps.append(np.asarray(z["packed"])); Ms.append(np.asarray(z["meta"]))
        Ds.append(np.asarray(z["dtm"]).astype(np.float32))
    P, M, dtm = np.concatenate(Ps), np.concatenate(Ms), np.concatenate(Ds)
    print(f"[data] {len(P)} positions  [{time.time()-t0:.0f}s]", flush=True)
    pid, sqi, pad = tokenize(P, M)
    print(f"[tok] done  [{time.time()-t0:.0f}s]", flush=True)
    # per-class split for verdicts
    sigs = np.array(["".join(sorted(p.symbol() for p in board_from_packed(P[i], M[i]).piece_map().values()))
                     for i in range(len(P))])
    idx = rng.permutation(len(P))
    n_te = len(P) // 10
    te, tr = idx[:n_te], idx[n_te:]

    net = DTMTok().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    y = dtm / args.scale
    for s in range(args.steps):
        bi = tr[rng.integers(0, len(tr), args.batch)]
        pred = net(torch.from_numpy(pid[bi]).to(dev), torch.from_numpy(sqi[bi]).to(dev),
                   torch.from_numpy(pad[bi]).to(dev))
        loss = nn.functional.smooth_l1_loss(pred, torch.from_numpy(y[bi]).to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0:
            print(f"  step {s} loss {float(loss.detach()):.4f}  [{time.time()-t0:.0f}s]", flush=True)
            trk.metrics(dict(loss=float(loss.detach())), step=s)

    net.eval()
    from scipy.stats import spearmanr
    with torch.no_grad():
        preds = []
        for s in range(0, len(te), 2048):
            bi = te[s:s + 2048]
            preds.append(net(torch.from_numpy(pid[bi]).to(dev), torch.from_numpy(sqi[bi]).to(dev),
                             torch.from_numpy(pad[bi]).to(dev)).cpu().numpy())
        pred = np.concatenate(preds) * args.scale
    true = dtm[te]
    print(f"VERDICT DTM_TOK overall: spearman {spearmanr(pred, true).correlation:+.3f} "
          f"MAE {np.abs(pred-true).mean():.2f} plies (n={len(te)})", flush=True)
    for sig in sorted(set(sigs[te])):
        m = sigs[te] == sig
        if m.sum() > 100:
            sp = spearmanr(pred[m], true[m]).correlation
            print(f"    class {sig:8s}: spearman {sp:+.3f} MAE {np.abs(pred[m]-true[m]).mean():.2f} "
                  f"(n={int(m.sum())})", flush=True)
            trk.metrics({f"sp_{sig}": float(sp)})
    torch.save({"state": net.state_dict(), "config": net.config, "scale": args.scale,
                "args": vars(args)}, args.out)
    print(f"saved {args.out}  [{time.time()-t0:.0f}s]", flush=True)
    stack.close()


if __name__ == "__main__":
    main()
