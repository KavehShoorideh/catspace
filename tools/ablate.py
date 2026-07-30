#!/usr/bin/env python
"""tools/ablate.py -- causal ablation, two granularities. Per the activation-patching
best-practice caveats (Zhang & Nanda 2023): results depend on the corruption choice
and metric, so both are explicit here and printed with every verdict; mean-ablation
is the default corruption (zero-ablation overstates dependence).

MODE dims  : ablate embedding dimensions of a representation file and measure the
             delta of a frozen linear probe's metric (probe trained on clean
             features; --label / --group as in probe_linear). Reports the top-k
             most causally-important dims. Generic -- works on any rep file.

MODE board : minimal-pair ablation on a JEPA checkpoint -- the paper's T2 atom
             certificate primitive: remove a piece (corrupt the BOARD, re-encode)
             and report the deltas of the model's own heads (any-event
             reachability within horizon, destination top-class mass). Removing
             the structure should collapse the predicted risk; if it doesn't,
             the recognition was not causal.

Usage:
  tools/ablate.py dims rep.npz --label wdl [--group gid] [--topk 12]
  tools/ablate.py board --ckpt jepa_t1_latest.pt --fen "..." --square e5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def mode_dims(args):
    rng = np.random.default_rng(0)
    d = np.load(args.rep, allow_pickle=True)
    X = d["emb"].astype(np.float64); y = d[args.label]
    if len(X) > args.sample:
        idx = np.sort(rng.choice(len(X), args.sample, replace=False))
        X, y = X[idx], y[idx]
        groups = d[args.group][idx] if args.group else None
    else:
        groups = d[args.group][:len(X)] if args.group else None
    if groups is not None:
        gs = np.unique(groups)
        te_g = set(rng.choice(gs, int(len(gs) * 0.2), replace=False).tolist())
        te = np.array([g in te_g for g in groups])
    else:
        te = rng.random(len(X)) < 0.2
    tr = ~te
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    base = float(clf.score(X[te], y[te]))
    mu = X[tr].mean(0)
    deltas = []
    for j in range(X.shape[1]):
        Xa = X[te].copy(); Xa[:, j] = mu[j]              # mean-ablation
        deltas.append(base - float(clf.score(Xa, y[te])))
    deltas = np.array(deltas)
    order = np.argsort(-deltas)[:args.topk]
    print(f"VERDICT ablate-dims[{args.label}]: base acc {base:.3f} | corruption "
          f"mean-ablate | top-{args.topk} dims by acc drop: "
          + " ".join(f"d{j}:{deltas[j]:+.3f}" for j in order)
          + f" | median drop {np.median(deltas):+.4f}")


def mode_board(args):
    import chess
    import torch
    from catspace.encoder.jepa import JepaT1, tokenize
    from catspace.train.scaffold import resolve_device
    dev = resolve_device("auto")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = JepaT1(**{k: ck["cfg"][k] for k in ("d", "layers", "n_class")}).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()

    def heads(board):
        t, g = tokenize(board)
        with torch.no_grad():
            phi = model.enc(torch.as_tensor(t[None]).to(dev),
                            torch.as_tensor(g[None]).to(dev))
            lam = torch.sigmoid(model.haz(phi, torch.zeros(1, 2, device=dev)))
            R = float(1 - torch.prod(1 - lam))            # any-event reach in horizon
            dd = torch.softmax(model.dest(phi).flatten(1), -1)[0]
            top = int(dd.argmax()); mass = float(dd[top])
        return R, top, mass

    b = chess.Board(args.fen)
    R0, c0, m0 = heads(b)
    squares = ([chess.parse_square(args.square)] if args.square else
               [sq for sq, pc in b.piece_map().items() if pc.piece_type != chess.KING])
    print(f"base: R_any {R0:.3f} | dest top class {c0} mass {m0:.3f} | "
          f"corruption: piece removal (minimal pair)")
    for sq in squares:
        pc = b.piece_map().get(sq)
        if pc is None:
            continue
        b2 = b.copy(); b2.remove_piece_at(sq)
        if not b2.is_valid():
            continue
        R1, c1, m1 = heads(b2)
        print(f"  -{pc.symbol()}@{chess.square_name(sq)}: dR_any {R1-R0:+.3f} | "
              f"dest {'SAME' if c1 == c0 else f'{c0}->{c1}'} dmass {m1-m0:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dims")
    d.add_argument("rep"); d.add_argument("--label", required=True)
    d.add_argument("--group", default=""); d.add_argument("--topk", type=int, default=12)
    d.add_argument("--sample", type=int, default=20000)
    bd = sub.add_parser("board")
    bd.add_argument("--ckpt", default="artifacts/experiments/jepa_t1_latest.pt")
    bd.add_argument("--fen", required=True)
    bd.add_argument("--square", default="", help="one square; empty = every non-king piece")
    args = ap.parse_args()
    (mode_dims if args.mode == "dims" else mode_board)(args)


if __name__ == "__main__":
    main()
