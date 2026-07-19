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
  4. (no clustering -- the field is a continuum; the viz shows raw points),

then write artifacts/generated/play_atlas/atlas.json plus the persisted map
(tsne_map/normalizer.npz + tsne_map/embedding.pkl) EXACTLY as the DATA
CONTRACT specifies. Per-stage wall-clock timings are printed (project rule).

Per-point:
  reach = F(s) . zW_unit   (zW = payload["zgoals"]["MATE_W"], unit-normalized;
                            F rows are unit, so reach is a cosine similarity)
  winp  = softmax(phead(F(s)))[...,0]   (P(win), White-POV; class 0 == W)

Usage:
  .venv/bin/python experiments/viz/build_play_atlas.py            # incumbent, n=4000
  .venv/bin/python experiments/viz/build_play_atlas.py --ckpt <field>.pt --phead <field>_phead.pt \
      --n 4000 --exaggeration 4 --tsne-iter 2000    # view any field's geometry
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

from catspace.data.certified import collect_certified_games
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--phead", default="data/derived/sep/cert_base_full_phead.pt")
    ap.add_argument("--n", type=int, default=10000, help="holdout positions to sample (capped by availability)")
    ap.add_argument("--perplexity", type=float, default=500.0,
                    help="clamped to <= n/3 at fit time (higher = more global structure)")
    ap.add_argument("--exaggeration", type=float, default=1.6,
                    help=">1 pulls clusters apart (more separation)")
    ap.add_argument("--tsne-iter", type=int, default=1500,
                    help="gradient iterations (more = more separation, slower)")
    ap.add_argument("--no-multiscale", dest="multiscale", action="store_false",
                    help="single perplexity instead of multiscale (local+global) affinities")
    ap.set_defaults(multiscale=True)
    ap.add_argument("--out", default="artifacts/generated/play_atlas")
    ap.add_argument("--shards", default=None, help="shard dir (default: newest)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-outcomes", action="store_true",
                    help="skip the certified-outcome filter (default: only positions from "
                         "games whose outcome is board-certified -- mate|draw|winner up "
                         ">=3 pts -- so the result coloring is board-honest; Kaveh "
                         "2026-07-19, exclusion rules apply to the t-SNE too)")
    ap.add_argument("--resign-material-gap", type=float, default=3.0)
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
    # certified filter (default): oversample, keep only rows from games whose
    # outcome is board-certified (mate|draw|material-backed win), trim to n --
    # so the atlas's result coloring never shows a balanced-position flag-fall
    # as a "win region". ~75% of games certify, hence the 1.6x oversample.
    t0 = time.time()
    want = args.n if args.all_outcomes else int(args.n * 1.6)
    picks = sample_shard_rows(shard_dir, want, seed=args.seed, holdout_only=True)
    rows = load_rows(shard_dir, picks)
    if not args.all_outcomes:
        cert = collect_certified_games(shard_dir, args.resign_material_gap)
        keep = cert[rows["game_id"]]
        n_raw = len(keep)
        rows = {k: v[keep][:args.n] for k, v in rows.items()}
        print(f"[stage] certified filter: {int(keep.sum())}/{n_raw} sampled rows "
              f"certified -> using {len(rows['packed'])}")
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
    from openTSNE import TSNE, TSNEEmbedding, affinity
    from openTSNE import initialization as tsne_init
    from catspace.viz.projection import Normalizer, TSNEProjection
    from catspace.viz.realboard import _FittedProjection
    normalizer = Normalizer.fit(F)
    Fn = normalizer.apply(F)
    cap = max(5.0, len(Fn) / 3.0)                          # perplexity must be << n
    perp = min(args.perplexity, cap)
    # openTSNE defaults already do the right thing for large data: learning_rate
    # "auto" (=N/exaggeration), PCA init (global structure + reproducible),
    # negative_gradient_method/neighbors "auto" (FFT + approximate NN at this N).
    # For COMPLEX data (chess), openTSNE + Kobak/Berens recommend a HIGH
    # perplexity (500) and, better, MULTISCALE affinities: a small perplexity
    # keeps local neighbourhoods while the large one carries global layout.
    if args.multiscale and len(Fn) >= 60:
        perps = sorted({int(min(cap, max(5, perp / 10))), int(perp)})
        aff = affinity.Multiscale(Fn, perplexities=perps, metric="cosine",
                                  n_jobs=-1, random_state=args.seed)
        emb = TSNEEmbedding(tsne_init.pca(Fn, random_state=args.seed), aff,
                            negative_gradient_method="auto", learning_rate="auto",
                            random_state=args.seed, n_jobs=-1)
        emb.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)   # early
        emb.optimize(n_iter=args.tsne_iter, exaggeration=args.exaggeration,
                     momentum=0.8, inplace=True)                                # main
        perp_note = f"multiscale{perps}"
    else:
        emb = TSNE(perplexity=perp, initialization="pca", metric="cosine",
                   exaggeration=args.exaggeration, n_iter=args.tsne_iter,
                   learning_rate="auto", random_state=args.seed, n_jobs=-1).fit(Fn)
        perp_note = f"perp={perp:g}"
    tp = TSNEProjection(perplexity=perp, seed=args.seed)
    tp._embedding = emb
    proj = _FittedProjection(normalizer, tp)
    xy = np.asarray(emb, dtype=np.float32)  # in-sample 2-D coords
    print(f"[stage] fit t-SNE ({n}x{F.shape[1]} -> 2D, perp={perp:g} "
          f"exag={args.exaggeration} iter={args.tsne_iter}): {time.time() - t0:.1f}s")

    # -------------------------------------------------- assemble atlas.json
    # (no k-means: the field is a continuum, so imposed cluster boundaries were
    # just illegible clutter -- the viz shows the raw points + colorings.)
    t0 = time.time()
    fens = [board_from_row(rows["packed"][i], rows["meta"][i]).fen() for i in range(n)]
    results = rows["result"].astype(int)
    plies = rows["ply"].astype(int)

    points = [dict(x=round(float(xy[i, 0]), 3), y=round(float(xy[i, 1]), 3),
                   result=int(results[i]), reach=round(float(reach[i]), 4),
                   winp=round(float(winp[i]), 4), ply=int(plies[i]),
                   fen=fens[i]) for i in range(n)]

    bounds = dict(xmin=round(float(xy[:, 0].min()), 3), xmax=round(float(xy[:, 0].max()), 3),
                  ymin=round(float(xy[:, 1].min()), 3), ymax=round(float(xy[:, 1].max()), 3))
    atlas = dict(ckpt=args.ckpt, bounds=bounds, points=points)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _tmp = out_dir / "atlas.json.tmp"          # atomic write: the play server may be
    _tmp.write_text(json.dumps(atlas))         # serving /atlas while we rebuild in place
    _tmp.replace(out_dir / "atlas.json")

    tsne_dir = out_dir / "tsne_map"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    np.savez(tsne_dir / "normalizer.npz", mu=proj.normalizer.mu, sd=proj.normalizer.sd)
    with open(tsne_dir / "embedding.pkl", "wb") as f:
        pickle.dump(proj.projection._embedding, f)
    print(f"[stage] build+write atlas.json ({len(points)} pts) + tsne_map: {time.time() - t0:.1f}s")
    print(f"wrote {out_dir / 'atlas.json'}")
    print(f"wrote {tsne_dir / 'normalizer.npz'}")
    print(f"wrote {tsne_dir / 'embedding.pkl'}")


if __name__ == "__main__":
    main()
