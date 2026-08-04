#!/usr/bin/env python
"""experiments/sample_line_positions.py -- Kaveh 2026-08-03: pull REAL positions off the dense
p_win == p_draw ridge visible in the ternary basin chart, so the structure can be read as chess
rather than as a density artifact.

The ridge is the symmetry axis hanging off the "mover loses" corner: positions where winning and
drawing are equally (im)probable, so only "am I losing" is determined. Measured composition:
63% terminal positions, 87% true-label loss, median 0 plies to game end. This script maps a
stratified sample back through (source, game, ply) to actual FENs by replaying the source moves,
so each point on the line can be inspected as a board.

Stratified along the ridge by p_loss, not uniform: the interesting question is what changes as
you slide DOWN the axis away from the mate corner, and a uniform sample would return almost
nothing but mates (they dominate the ridge's mass).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from catspace.research.tools.training_infra.losses import WIN, DRAW, LOSS
from catspace.research.tools.embeddings.basin_simplex_chart import load_head

LABEL = {WIN: "mover wins", DRAW: "draw", LOSS: "mover loses"}


def fens_for_sf(moves_tsv, want):
    """{(gid, ply)} -> {(gid, ply): fen} by replaying the SF-vs-SF move lists."""
    import chess
    need = defaultdict(set)
    for g, p in want:
        need[g].add(p)
    out = {}
    with open(moves_tsv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gid = int(parts[0])
            if gid not in need:
                continue
            board = chess.Board()
            for ply, u in enumerate(parts[2].split()):
                board.push(chess.Move.from_uci(u))
                if ply in need[gid]:
                    out[(gid, ply)] = board.fen()
            del need[gid]
            if not need:
                break
    return out


def fens_for_human(records_dir, want):
    import chess
    import pyarrow.parquet as pq
    need = defaultdict(set)
    for g, p in want:
        need[g].add(p)
    out = {}
    for shard in sorted(Path(records_dir).glob("records_*.parquet")):
        d = pq.read_table(shard, columns=["game_id", "moves"]).to_pydict()
        for gid, mv in zip(d["game_id"], d["moves"]):
            gid = int(gid)
            if gid not in need:
                continue
            board = chess.Board()
            for ply, u in enumerate(mv.split()):
                try:
                    board.push(chess.Move.from_uci(u))
                except Exception:
                    break
                if ply in need[gid]:
                    out[(gid, ply)] = board.fen()
            del need[gid]
        if not need:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/iqe_poles_both_latest.pt")
    ap.add_argument("--combined", default="data/derived/field_combined_sub600k.npz")
    ap.add_argument("--sf-moves", default="data/derived/opening_pool_sfsf_moves.tsv")
    ap.add_argument("--human-records", default="data/records/lichess_2019-01")
    ap.add_argument("--tol", type=float, default=0.01, help="|p_win - p_draw| ridge width")
    ap.add_argument("--per-band", type=int, default=2)
    ap.add_argument("--pool", type=int, default=200000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    net = load_head(args.ckpt, args.device)
    T = float(net.temperature.detach())
    z = np.load(args.combined, allow_pickle=True)
    meta = eval(str(z["_meta"][0]))
    mm = np.load(meta["feats"][0], mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(z["y"]), min(args.pool, len(z["y"])), replace=False))

    with torch.no_grad():
        ds = []
        for i in range(0, len(idx), 8192):
            x = torch.from_numpy(np.asarray(mm[z["local_row"][idx[i:i + 8192]]],
                                            dtype=np.float32)).to(args.device)
            ds.append(net.d_poles(net.phi(x)).cpu().numpy())
    d = np.concatenate(ds)
    p = torch.softmax(-torch.log1p(torch.tensor(d)) / T, dim=-1).numpy()

    on_line = np.abs(p[:, 0] - p[:, 1]) < args.tol
    print(f"ridge |p_win-p_draw| < {args.tol}: {on_line.sum():,} of {len(idx):,} "
          f"({100*on_line.mean():.2f}%)  [{time.time()-t0:.0f}s]")

    # walk DOWN the axis: bands of p_loss from the mate corner toward the base
    bands = [(0.95, 1.01), (0.85, 0.95), (0.70, 0.85), (0.55, 0.70), (0.40, 0.55), (0.0, 0.40)]
    picks = []
    for lo, hi in bands:
        m = np.flatnonzero(on_line & (p[:, 2] >= lo) & (p[:, 2] < hi))
        if not len(m):
            continue
        for j in rng.choice(m, min(args.per_band, len(m)), replace=False):
            picks.append((float(lo), float(hi), int(j)))

    off = meta["game_offset"]
    want_sf, want_hu = set(), set()
    for _, _, j in picks:
        r = idx[j]
        g, ply = int(z["game"][r]), int(z["ply"][r])
        (want_sf if z["orig_source"][r] == 1 else want_hu).add((g - off if z["orig_source"][r] == 1 else g, ply))
    fen = {}
    if want_sf:
        fen.update({("sf", g, p_): f for (g, p_), f in fens_for_sf(args.sf_moves, want_sf).items()})
    if want_hu:
        fen.update({("hu", g, p_): f for (g, p_), f in fens_for_human(args.human_records, want_hu).items()})

    print(f"\n{'p_loss band':>12s} {'src':>3s} {'ply':>4s} {'to_end':>6s} {'term':>5s} "
          f"{'true':>11s}  p(win,draw,loss)   FEN")
    for lo, hi, j in picks:
        r = idx[j]
        s = "sf" if z["orig_source"][r] == 1 else "hu"
        g = int(z["game"][r]) - (off if s == "sf" else 0)
        ply = int(z["ply"][r])
        f = fen.get((s, g, ply), "(replay unavailable)")
        print(f"  [{lo:.2f},{hi:.2f})  {s:>3s} {ply:>4d} {int(z['n_to_end'][r]):>6d} "
              f"{str(bool(z['is_terminal'][r])):>5s} {LABEL[int(z['y'][r])]:>11s}  "
              f"({p[j,0]:.3f},{p[j,1]:.3f},{p[j,2]:.3f})  {f}")
    print(f"\n[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
