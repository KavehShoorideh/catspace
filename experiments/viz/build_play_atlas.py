#!/usr/bin/env python
"""
experiments/viz/build_play_atlas.py — COMPONENT A of the play-atlas app
(FROZEN CONTRACT). Precompute the t-SNE background atlas the play server (B)
serves and the frontend (C) draws:

  1. sample ~4000 holdout positions from the newest Lichess shard dir,
  2. embed F with the INCUMBENT (cert_base_full.pt) on CPU (device="cpu"
     ALWAYS -- the training GPU/MPS is busy),
  3. fit a persisted openTSNE map on the normalized F (out-of-sample
     transformable so B can project NEW boards),
  4. k-means the 2-D t-SNE coords into ~14 clusters (clustering, not
     absolute distance, is the meaningful structure),

then write artifacts/generated/play_atlas/atlas.json plus the persisted map
(tsne_map/normalizer.npz + tsne_map/embedding.pkl) EXACTLY as the DATA
CONTRACT specifies. Per-stage wall-clock timings are printed (project rule).

Per-point:
  reach = F(s) . zW_unit   (zW = payload["zgoals"]["MATE_W"], unit-normalized;
                            F rows are unit, so reach is a cosine similarity)
  winp  = softmax(phead(F(s)))[...,0]   (P(win), White-POV; class 0 == W)

Usage (small smoke run, then the real run the orchestrator launches):
  .venv/bin/python experiments/viz/build_play_atlas.py --n 200 --clusters 6
  .venv/bin/python experiments/viz/build_play_atlas.py            # n=4000, 14 clusters
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from catspace.data.shards import sample_shard_rows
from catspace.io.paths import newest_shard_dir
from catspace.nn.eval_head import EvalHead
from catspace.nn.fb import load_ckpt
from catspace.viz.realboard import board_from_row, embed_positions, fit_projection

# columns pulled from each shard for the sampled rows (contract: packed/meta/
# elo/clock/result/ply; game_id kept for provenance/debug).
COLS = ("packed", "meta", "ply", "clock", "result", "white_elo", "black_elo", "game_id")


def load_rows(shard_dir: Path, picks: list) -> dict:
    """Gather the (shard_file, row) picks from sample_shard_rows into one dict
    of concatenated per-column arrays (same pattern as build_fullboard_viewer)."""
    by_file: dict = {}
    for name, row in picks:
        by_file.setdefault(name, []).append(row)
    out: dict = {k: [] for k in COLS}
    for name, rows in sorted(by_file.items()):
        npz = np.load(shard_dir / name)
        idx = np.array(sorted(rows))
        for k in COLS:
            out[k].append(npz[k][idx])
    return {k: np.concatenate(v) for k, v in out.items()}


def cluster_label(results: np.ndarray, plies: np.ndarray) -> str:
    """Short auto label 'W-heavy | mid' == dominant result + median-ply band."""
    counts = {r: int((results == r).sum()) for r in (1, 0, -1)}
    dom = max(counts, key=counts.get)
    word = {1: "W-heavy", -1: "B-heavy", 0: "Draw"}[dom]
    med = float(np.median(plies)) if len(plies) else 0.0
    band = "open" if med < 30 else ("mid" if med < 70 else "late")
    return f"{word} · {band}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--phead", default="data/derived/sep/cert_base_full_phead.pt")
    ap.add_argument("--n", type=int, default=4000, help="holdout positions to sample")
    ap.add_argument("--clusters", type=int, default=14, help="k-means clusters on 2-D t-SNE")
    ap.add_argument("--pool", type=int, default=40, help="max sampled FENs per cluster")
    ap.add_argument("--perplexity", type=float, default=40.0)
    ap.add_argument("--exaggeration", type=float, default=1.6,
                    help=">1 pulls clusters apart (more separation)")
    ap.add_argument("--tsne-iter", type=int, default=1500,
                    help="gradient iterations (more = more separation, slower)")
    ap.add_argument("--out", default="artifacts/generated/play_atlas")
    ap.add_argument("--shards", default=None, help="shard dir (default: newest)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cpu"  # contract: CPU ALWAYS (training GPU/MPS is busy)
    rng = np.random.default_rng(args.seed)
    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()

    # -------------------------------------------------- load model + phead
    t0 = time.time()
    fb, payload = load_ckpt(Path(args.ckpt), device)
    fb.eval()
    import torch
    hp = torch.load(args.phead, map_location=device, weights_only=False)
    phead = EvalHead(d_in=hp["d_in"]).to(device)
    phead.load_state_dict(hp["state"])
    phead.eval()
    zw = payload["zgoals"]["MATE_W"].numpy().astype(np.float32)
    zw_unit = zw / (np.linalg.norm(zw) + 1e-12)
    step = payload.get("step", "?")
    print(f"[stage] load model+phead: {time.time() - t0:.1f}s  "
          f"ckpt={Path(args.ckpt).name} step={step} shards={shard_dir.name} device={device}")

    # -------------------------------------------------- sample holdout rows
    t0 = time.time()
    picks = sample_shard_rows(shard_dir, args.n, seed=args.seed, holdout_only=True)
    rows = load_rows(shard_dir, picks)
    n = len(rows["packed"])
    print(f"[stage] sample+load {n} holdout rows: {time.time() - t0:.1f}s")

    # -------------------------------------------------- embed F (CPU)
    t0 = time.time()
    F, _ = embed_positions(fb, rows["packed"], rows["meta"], rows["white_elo"],
                           rows["black_elo"], rows["clock"], device)
    reach = (F @ zw_unit).astype(np.float32)
    with torch.no_grad():
        winp = torch.softmax(phead(torch.from_numpy(F).to(device)), dim=1)[:, 0].cpu().numpy()
    print(f"[stage] embed F + reach/winp ({n} rows): {time.time() - t0:.1f}s")

    # -------------------------------------------------- fit persisted t-SNE
    # Tuned for SEPARATION (Kaveh 2026-07-18: "run a bit more"): more gradient
    # iterations + a >1 late exaggeration pulls clusters apart (openTSNE's
    # separation knob). Built directly (not via fit_projection) for the extra
    # params, then wrapped so the server can still out-of-sample transform.
    t0 = time.time()
    from openTSNE import TSNE
    from catspace.viz.projection import Normalizer, TSNEProjection
    from catspace.viz.realboard import _FittedProjection
    normalizer = Normalizer.fit(F)
    Fn = normalizer.apply(F)
    emb = TSNE(perplexity=args.perplexity, initialization="pca", metric="cosine",
               exaggeration=args.exaggeration, n_iter=args.tsne_iter,
               random_state=args.seed, n_jobs=-1).fit(Fn)
    tp = TSNEProjection(perplexity=args.perplexity, seed=args.seed)
    tp._embedding = emb
    proj = _FittedProjection(normalizer, tp)
    xy = np.asarray(emb, dtype=np.float32)  # in-sample 2-D coords
    print(f"[stage] fit t-SNE ({n}x{F.shape[1]} -> 2D, perp={args.perplexity} "
          f"exag={args.exaggeration} iter={args.tsne_iter}): {time.time() - t0:.1f}s")

    # -------------------------------------------------- k-means on 2-D coords
    t0 = time.time()
    from sklearn.cluster import KMeans
    k = min(args.clusters, n)
    km = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit(xy)
    labels = km.labels_.astype(int)
    centers = km.cluster_centers_
    print(f"[stage] k-means ({k} clusters): {time.time() - t0:.1f}s")

    # -------------------------------------------------- assemble atlas.json
    t0 = time.time()
    fens = [board_from_row(rows["packed"][i], rows["meta"][i]).fen() for i in range(n)]
    results = rows["result"].astype(int)
    plies = rows["ply"].astype(int)

    points = [dict(x=round(float(xy[i, 0]), 3), y=round(float(xy[i, 1]), 3),
                   result=int(results[i]), reach=round(float(reach[i]), 4),
                   winp=round(float(winp[i]), 4), ply=int(plies[i]),
                   cluster=int(labels[i]), fen=fens[i]) for i in range(n)]

    clusters = []
    for cid in range(k):
        member = np.flatnonzero(labels == cid)
        pool = member if len(member) <= args.pool else \
            rng.choice(member, size=args.pool, replace=False)
        clusters.append(dict(
            id=int(cid), cx=round(float(centers[cid, 0]), 3),
            cy=round(float(centers[cid, 1]), 3), n=int(len(member)),
            label=cluster_label(results[member], plies[member]),
            fens=[fens[int(i)] for i in pool]))

    bounds = dict(xmin=round(float(xy[:, 0].min()), 3), xmax=round(float(xy[:, 0].max()), 3),
                  ymin=round(float(xy[:, 1].min()), 3), ymax=round(float(xy[:, 1].max()), 3))
    atlas = dict(ckpt=args.ckpt, bounds=bounds, points=points, clusters=clusters)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "atlas.json").write_text(json.dumps(atlas))

    tsne_dir = out_dir / "tsne_map"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    np.savez(tsne_dir / "normalizer.npz", mu=proj.normalizer.mu, sd=proj.normalizer.sd)
    with open(tsne_dir / "embedding.pkl", "wb") as f:
        pickle.dump(proj.projection._embedding, f)
    print(f"[stage] build+write atlas.json ({len(points)} pts, {len(clusters)} clusters) "
          f"+ tsne_map: {time.time() - t0:.1f}s")
    print(f"wrote {out_dir / 'atlas.json'}")
    print(f"wrote {tsne_dir / 'normalizer.npz'}")
    print(f"wrote {tsne_dir / 'embedding.pkl'}")


if __name__ == "__main__":
    main()
