#!/usr/bin/env python
"""experiments/catalog_mate_directions.py -- Kaveh 2026-07-22: "tb isn't optimal [DTZ hangs
rooks], stockfish is more optimal, humans are even better because they get mated easily. Use
the B field trained on lichess data and see if you can catalog the mate directions."

Harvest REAL human mates from the lichess shards (games whose final stored position is
checkmate, Black mated), classify each by the classic human mate PATTERN (rules-computed,
audit-clean): back-rank, ladder, smothered, queen-kiss, king-support rook, escort, other.
Embed with the lichess-trained B tower (full planes -- the lichess field's convention).
Catalog two things:

  CLUSTERS     do human mate patterns cluster in B? (silhouette + same-pattern cohesion,
               t-SNE panel colored by pattern)
  DIRECTIONS   the approach direction DeltaB = B(mate) - B(t-k) per game, unit-normalized:
               are directions COHERENT within a pattern (median pairwise cosine, within vs
               across)? k-means over directions -> the actual catalog (each entry: size,
               dominant pattern, coherence) -> JSON.

VERDICT lines throughout; PNG + JSON out.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import chess
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device


# ------------------------------------------------------------------ pattern classifier
def mate_pattern(b: chess.Board) -> str:
    """Classic human mate patterns for a Black-mated board. Rules only (audit-clean)."""
    bk = b.king(chess.BLACK); wk = b.king(chess.WHITE)
    checkers = list(b.checkers())
    if len(checkers) > 1:
        return "double"
    chk = checkers[0]; ct = b.piece_type_at(chk)
    f, r = chess.square_file(bk), chess.square_rank(bk)
    adj = [s for s in chess.SquareSet(chess.BB_KING_ATTACKS[bk])]
    own_blocked = [s for s in adj if (p := b.piece_at(s)) is not None and p.color == chess.BLACK]
    if ct == chess.KNIGHT and len(own_blocked) == len(adj):
        return "smothered"
    if ct in (chess.ROOK, chess.QUEEN) and r == 7 and chess.square_rank(chk) == 7:
        front = [s for s in adj if chess.square_rank(s) == 6]
        if front and all(b.piece_at(s) is not None and b.piece_at(s).color == chess.BLACK for s in front):
            return "backrank"
    if ct in (chess.ROOK, chess.QUEEN) and (r in (0, 7) or f in (0, 7)):
        # ladder: a SECOND heavy piece holds the inner line
        heavies = [s for s in list(b.pieces(chess.ROOK, chess.WHITE)) + list(b.pieces(chess.QUEEN, chess.WHITE))
                   if s != chk]
        if r in (0, 7):
            inner = 1 if r == 0 else 6
            if any(chess.square_rank(s) == inner for s in heavies):
                return "ladder"
        if f in (0, 7):
            inner = 1 if f == 0 else 6
            if any(chess.square_file(s) == inner for s in heavies):
                return "ladder"
    if ct == chess.QUEEN and chess.square_distance(chk, bk) == 1:
        return "qkiss"
    if ct in (chess.ROOK, chess.QUEEN) and chess.square_distance(wk, bk) <= 2:
        return "ksupport"
    return {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
            chess.ROOK: "rook-other", chess.QUEEN: "queen-other"}[ct]


# ------------------------------------------------------------------ harvest
def harvest(shard_dir: str, cap: int, tail: int, rng):
    """Games whose FINAL stored position is checkmate with Black mated. Returns
    (mate_rows, tail_rows_per_game) as (packed, meta) arrays + pattern labels + elos."""
    mates_pk, mates_mt, tails, pats, elos = [], [], [], [], []
    for path in sorted(glob.glob(str(Path(shard_dir) / "shard_*.npz"))):
        z = np.load(path)
        gid = z["game_id"]; res = z["result"]; pk = z["packed"]; mt = z["meta"]
        we = z["white_elo"]
        bounds = np.flatnonzero(np.diff(gid)) + 1
        starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(gid)]])
        order = rng.permutation(len(starts))
        for gi in order:
            s, e = int(starts[gi]), int(ends[gi])
            if e - s < tail + 2 or res[s] != 1:            # White won; need some history
                continue
            b = board_from_packed(pk[e - 1], mt[e - 1])
            if not (b.is_checkmate() and b.turn == chess.BLACK):
                continue
            mates_pk.append(pk[e - 1]); mates_mt.append(mt[e - 1])
            tails.append((pk[e - 1 - tail:e - 1], mt[e - 1 - tail:e - 1]))
            pats.append(mate_pattern(b)); elos.append(int(we[s]))
            if len(mates_pk) >= cap:
                return (np.stack(mates_pk), np.stack(mates_mt)), tails, np.array(pats), np.array(elos)
    return (np.stack(mates_pk), np.stack(mates_mt)), tails, np.array(pats), np.array(elos)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="data/derived/sep/lichess_gn_iqeqrl_sf.pt")
    ap.add_argument("--shards", default="data/shards/lichess_db_standard_rated_2019-01.prefix256mb")
    ap.add_argument("--n-mates", type=int, default=1500)
    ap.add_argument("--tail", type=int, default=6, help="approach plies before mate")
    ap.add_argument("--dir-k", type=int, default=8, help="k-means clusters for the direction catalog")
    ap.add_argument("--perplexity", type=float, default=40.0)
    ap.add_argument("--n-arrows", type=int, default=40)
    ap.add_argument("--out", default="artifacts/experiments/mate_directions_lichess.png")
    ap.add_argument("--out-json", default="artifacts/experiments/mate_directions_catalog.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)

    (mpk, mmt), tails, pats, elos = harvest(args.shards, args.n_mates, args.tail, rng)
    uniq, cnt = np.unique(pats, return_counts=True)
    print(f"[harvest] {len(mpk)} human mates (Black mated)  median elo {int(np.median(elos))}  "
          f"patterns: { {u: int(c) for u, c in sorted(zip(uniq, cnt), key=lambda x: -x[1])} }  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    fb, pay = load_ckpt(Path(args.field), dev); fb.eval()

    def embB(pk, mt):
        out = []
        for s in range(0, len(pk), 1024):
            pl = feature_planes(pk[s:s + 1024], mt[s:s + 1024])   # lichess field: full planes
            with torch.no_grad():
                out.append(fb.embed_B(torch.from_numpy(pl).to(dev)).cpu().numpy())
        return np.concatenate(out)

    Bm = embB(mpk, mmt)
    # approach embedding: one pre-mate anchor per game (t - tail) for the direction, plus
    # full tails for a subset of arrow games
    anchor_pk = np.stack([t[0][0] for t in tails]); anchor_mt = np.stack([t[1][0] for t in tails])
    Ba = embB(anchor_pk, anchor_mt)
    D = Bm - Ba
    D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)

    # ---------------- CLUSTERS verdict
    from sklearn.metrics import silhouette_score
    big = [u for u, c in zip(uniq, cnt) if c >= 40]
    sel = np.isin(pats, big)
    sil = float(silhouette_score(Bm[sel], pats[sel])) if len(big) >= 2 else float("nan")

    def cohesion(idx):
        ds, rs = [], []
        for _ in range(6000):
            i, j = idx[rng.integers(len(idx))], idx[rng.integers(len(idx))]
            a, c = rng.integers(len(Bm)), rng.integers(len(Bm))
            if i != j:
                ds.append(np.linalg.norm(Bm[i] - Bm[j]))
            if a != c:
                rs.append(np.linalg.norm(Bm[a] - Bm[c]))
        return float(np.median(ds) / np.median(rs))

    coh = {p: cohesion(np.flatnonzero(pats == p)) for p in big}
    print(f"VERDICT MATE_CLUSTERS_LICHESS field={Path(args.field).stem}  silhouette(pattern)={sil:+.2f}  "
          f"cohesion: " + "  ".join(f"{p} {v:.2f}" for p, v in sorted(coh.items(), key=lambda x: x[1]))
          + "  (<1 = pattern clusters)", flush=True)

    # ---------------- DIRECTIONS verdict + catalog
    # LEVEL 1: the MASTER mate-direction (all approaches share one axis if this is high)
    master = D.mean(0); master /= max(np.linalg.norm(master), 1e-9)
    cos_master = D @ master
    print(f"VERDICT MATE_DIRECTIONS_LICHESS.MASTER  median cos(DeltaB, master)= {np.median(cos_master):+.2f}  "
          f"(high = ONE global 'toward mate' direction exists in B)", flush=True)
    # LEVEL 2: pattern-specific structure in the RESIDUAL (master projected out)
    R = D - np.outer(cos_master, master)
    R = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-9)
    within, across = [], []
    for _ in range(8000):
        i, j = rng.integers(len(R)), rng.integers(len(R))
        if i == j:
            continue
        (within if pats[i] == pats[j] else across).append(float(R[i] @ R[j]))
    print(f"VERDICT MATE_DIRECTIONS_LICHESS.RESIDUAL  cosine within-pattern {np.median(within):+.2f} "
          f"vs across {np.median(across):+.2f}  (within >> across = pattern-specific approach directions)",
          flush=True)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.dir_k, n_init=6, random_state=args.seed).fit(R)
    catalog = []
    for c in range(args.dir_k):
        m_ = km.labels_ == c
        cp, cc = np.unique(pats[m_], return_counts=True)
        dom = cp[np.argmax(cc)]
        cen = km.cluster_centers_[c] / max(np.linalg.norm(km.cluster_centers_[c]), 1e-9)
        catalog.append(dict(cluster=c, n=int(m_.sum()), dominant=str(dom),
                            dominant_frac=float(cc.max() / m_.sum()),
                            coherence=float(np.median(R[m_] @ cen)),
                            patterns={str(p): int(n) for p, n in zip(cp, cc)}))
    catalog.sort(key=lambda x: -x["n"])
    print("  direction catalog (k-means over unit DeltaB):", flush=True)
    for e in catalog:
        print(f"    #{e['cluster']}: n={e['n']:4d}  dominant={e['dominant']:9s} ({e['dominant_frac']:.0%})  "
              f"coherence={e['coherence']:+.2f}", flush=True)
    Path(args.out_json).write_text(json.dumps(dict(field=args.field, n_mates=len(mpk),
                                                   patterns={str(u): int(c) for u, c in zip(uniq, cnt)},
                                                   silhouette=sil, cohesion=coh, catalog=catalog), indent=2))

    # ---------------- figure: t-SNE colored by pattern + arrows + zW
    from sklearn.manifold import TSNE
    arrow_ids = rng.choice(len(tails), min(args.n_arrows, len(tails)), replace=False)
    tail_B = {i: embB(tails[i][0], tails[i][1]) for i in arrow_ids}
    _zw = pay["zgoals"]["MATE_W"]
    zW = (_zw.detach().float().numpy() if torch.is_tensor(_zw) else np.asarray(_zw, np.float32))
    stack = [Bm] + [tail_B[i] for i in arrow_ids] + [zW[None, :]]
    X = np.concatenate(stack, 0)
    XY = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
              early_exaggeration=1.6, random_state=args.seed).fit_transform(X)
    xy_m = XY[:len(Bm)]; off = len(Bm); xy_t = {}
    for i in arrow_ids:
        xy_t[i] = XY[off:off + len(tail_B[i])]; off += len(tail_B[i])
    xy_z = XY[-1]

    fig, ax = plt.subplots(figsize=(13, 11))
    cmap = plt.get_cmap("tab10")
    order = np.argsort(-cnt)
    for k, ui in enumerate(order[:9]):
        p = uniq[ui]; m_ = pats == p
        ax.scatter(xy_m[m_, 0], xy_m[m_, 1], s=16, color=cmap(k), alpha=0.7,
                   label=f"{p} ({m_.sum()})", linewidths=0)
    for i in arrow_ids:
        xy = np.concatenate([xy_t[i], xy_m[i:i + 1]])
        for a, b in zip(xy[:-1], xy[1:]):
            ax.annotate("", xy=b, xytext=a,
                        arrowprops=dict(arrowstyle="->", color="crimson", alpha=0.4, lw=0.9))
    ax.scatter(*xy_z, marker="*", s=420, color="gold", edgecolors="black", zorder=5,
               label="learned MATE_W goal")
    ax.set_title(f"Human mates in lichess-B ({Path(args.field).stem}) -- t-SNE, color=mate pattern\n"
                 f"silhouette {sil:+.2f} | direction cosine within {np.median(within):+.2f} "
                 f"vs across {np.median(across):+.2f}")
    ax.legend(loc="best", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(args.out, dpi=140)
    print(f"saved {args.out} + {args.out_json}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
