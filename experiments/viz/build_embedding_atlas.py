#!/usr/bin/env python
"""
experiments/viz/build_embedding_atlas.py — populate
catspace/viz/templates/embedding_atlas.html: project F(s) over a shared
holdout sample for TWO ARBITRARY checkpoints (--ckpt-a/--ckpt-b, any two
.pt files under data/derived/ or absolute paths -- defaults to the
step-2000/step-30000 before/after pair, but any pair works) and let the
viewer switch coloring by result/ply/reach/winprob/Elo. Each checkpoint
gets its OWN projection fit (geometry isn't comparable across checkpoints)
but the same point sample, so cluster-shape comparisons are apples-to-
apples. Boards render client-side from FEN, not pre-rendered SVG.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from catspace.data.encode import board_from_packed
from catspace.data.shards import sample_shard_rows
from catspace.io.paths import derived_dir, generated_dir, newest_shard_dir
from catspace.nn.features import elo_bin, winprob_cp
from catspace.nn.fb import load_ckpt, pick_device
from catspace.viz.build_html import build_html
from catspace.viz.realboard import embed_positions, fit_projection

COLS = ("packed", "meta", "ply", "clock", "eval_cp", "result", "white_elo", "black_elo", "game_id")


def load_rows(shard_dir: Path, picks: list) -> dict:
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
    ap.add_argument("--shards", default=None)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--projection", choices=("pca", "tsne"), default="tsne")
    ap.add_argument("--ckpt-a", default="lichess_fb_step2000.pt",
                    help="any .pt checkpoint under data/derived/ (or an absolute path) -- "
                         "not required to be a step-2000 snapshot")
    ap.add_argument("--ckpt-b", default="lichess_fb.pt",
                    help="second checkpoint to compare against --ckpt-a")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    device = pick_device(args.device)

    t0 = time.time()
    picks = sample_shard_rows(shard_dir, args.n, seed=args.seed, holdout_only=True)
    data = load_rows(shard_dir, picks)
    n = len(data["ply"])
    print(f"sample+load {n} holdout rows: {time.time() - t0:.1f}s")

    wp = winprob_cp(data["eval_cp"])
    welo_bin = elo_bin(data["white_elo"])
    fens = [board_from_packed(data["packed"][i], data["meta"][i]).fen() for i in range(n)]

    points = dict(fen=fens, result=[int(r) for r in data["result"]], ply=[int(p) for p in data["ply"]],
                 white_elo_bin=[int(b) for b in welo_bin],
                 winprob=[None if not np.isfinite(v) else round(float(v), 3) for v in wp])

    ckpts_out = []
    for name in (args.ckpt_a, args.ckpt_b):
        path = derived_dir() / name if not Path(name).is_absolute() else Path(name)
        t0 = time.time()
        fb, payload = load_ckpt(path, device)
        fb.eval()
        step = payload.get("step", "?")
        label = f"{path.stem} (step {step})"    # derived from the actual file + its own step, never hardcoded
        zdiff = payload["zgoals"]["MATE_DIFF"].numpy().astype(np.float32)
        z = zdiff / np.linalg.norm(zdiff)
        F, _ = embed_positions(fb, data["packed"], data["meta"], data["white_elo"], data["black_elo"],
                               data["clock"], device)
        proj = fit_projection(F, kind=args.projection, seed=args.seed)
        xy = proj.fit_points()
        reach = F @ z
        ckpts_out.append(dict(name=label,
                              xy=[[round(float(x), 2), round(float(y), 2)] for x, y in xy],
                              reach=[round(float(r), 4) for r in reach]))
        print(f"{label} ({path.name}): embed+project {time.time() - t0:.1f}s")

    data_out = dict(meta=dict(title=f"catspace — embedding atlas  ·  {n} holdout positions  ·  {args.projection}"),
                    points=points, ckpts=ckpts_out)

    out = Path(args.out) if args.out else generated_dir() / "embedding-atlas.html"
    template = Path(__file__).resolve().parents[2] / "catspace" / "viz" / "templates" / "embedding_atlas.html"
    build_html(template, data_out, out)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
