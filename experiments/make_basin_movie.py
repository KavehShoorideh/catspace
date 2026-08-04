#!/usr/bin/env python
"""experiments/make_basin_movie.py -- Kaveh 2026-08-03: render the ternary basin simplex from every
checkpoint in the ladder and stitch them into a movie, so the basins can be watched FORMING rather
than inspected once at the end.

Efficiency note, since this runs over hundreds of checkpoints: the frozen trunk features for a
fixed sample are loaded into device memory ONCE, and each frame only re-runs the tiny 139k-param
head over them. Re-reading features per frame would dominate the cost (that is the same disk-bound
trap the training itself hit). Frames are skipped if already rendered, so this can be run
repeatedly while training is still going and it will only pick up what is new.

Axes are FIXED across frames (the triangle, and a fixed hexbin extent + colour normalization), so
motion in the movie is real motion of the distribution and not a rescaling artifact -- an
auto-scaled colour bar per frame would make a static distribution appear to pulse.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.losses import basin_logp
from experiments.basin_simplex_chart import (bary_to_xy, draw_triangle, VERTS, COLOR_HUMAN,
                                             COLOR_SF, INK, MUTED)
from catspace.encoder.iqe_head import IQEHead

STEP_RE = re.compile(r"_step(\d+)\.pt$")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-glob", default="artifacts/experiments/movie/iqe_poles_30k_step*.pt")
    ap.add_argument("--combined", default="data/derived/field_combined_sub600k.npz")
    ap.add_argument("--n", type=int, default=25000, help="positions per dataset (held fixed)")
    ap.add_argument("--frames", default="artifacts/experiments/movie/frames")
    ap.add_argument("--out", default="artifacts/experiments/movie/basin_simplex_movie.mp4")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    ck = sorted(Path().glob(args.ckpt_glob.replace("artifacts", "artifacts", 1)),
                key=lambda p: int(STEP_RE.search(p.name).group(1)))
    ck = [p for p in ck if STEP_RE.search(p.name)]
    if not ck:
        raise SystemExit(f"no checkpoints matched {args.ckpt_glob}")
    print(f"{len(ck)} checkpoints, steps {int(STEP_RE.search(ck[0].name).group(1))}.."
          f"{int(STEP_RE.search(ck[-1].name).group(1))}")

    z = np.load(args.combined, allow_pickle=True)
    meta = eval(str(z["_meta"][0]))
    mm = np.load(meta["feats"][0], mmap_mode="r")
    split = z["orig_source"] if "orig_source" in z.files else z["source"]
    rng = np.random.default_rng(args.seed)

    # Load the sample's features ONCE onto the device; every frame reuses them.
    feats, names = {}, ["human", "SF-vs-SF"]
    for name, s in zip(names, (0, 1)):
        idx = np.flatnonzero(split == s)
        take = np.sort(rng.choice(idx, min(args.n, len(idx)), replace=False))
        arr = np.asarray(mm[z["local_row"][take]], dtype=np.float32)
        feats[name] = torch.from_numpy(arr).to(args.device)
        print(f"  cached {name}: {tuple(feats[name].shape)} [{time.time()-t0:.0f}s]", flush=True)

    p0 = torch.load(ck[0], map_location=args.device, weights_only=False)
    cfg = p0["cfg"]
    net = IQEHead(in_ch=cfg["in_ch"], d=cfg["d"], components=cfg["components"],
                  adapter_ch=cfg["adapter_ch"]).to(args.device).eval()

    fdir = Path(args.frames); fdir.mkdir(parents=True, exist_ok=True)
    xlim, ylim = (-0.03, 1.03), (-0.05, np.sqrt(3) / 2 + 0.08)
    # Fixed colour scale across frames -- see the module docstring.
    vmax = max(1000.0, args.n / 25.0)
    rendered = 0
    for c in ck:
        step = int(STEP_RE.search(c.name).group(1))
        out_png = fdir / f"frame_{step:06d}.png"
        if out_png.exists():
            continue
        sd = torch.load(c, map_location=args.device, weights_only=False)["state_dict"]
        net.load_state_dict(sd, strict=False)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
        with torch.no_grad():
            T = net.temperature
            for ax, name in zip(axes, names):
                d = net.d_poles(net.phi(feats[name]))
                p = basin_logp(d, T).exp().cpu().numpy()
                xy = bary_to_xy(p)
                ax.hexbin(xy[:, 0], xy[:, 1], gridsize=44, cmap="Blues" if name == "human" else "Reds",
                          norm=LogNorm(vmin=1, vmax=vmax), mincnt=1, linewidths=0,
                          extent=(*xlim, *ylim))
                draw_triangle(ax)
                ax.set_xlim(*xlim); ax.set_ylim(*ylim)
                amb = float((p.max(1) < 0.5).mean())
                ax.set_title(f"{name}   ambiguous {100*amb:.0f}%", fontsize=10, color=INK)
        fig.suptitle(f"W/D/L basin simplex -- step {step:,}", fontsize=13, color=INK)
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        rendered += 1
        if rendered % 25 == 0:
            print(f"  {rendered} frames [{time.time()-t0:.0f}s]", flush=True)
    print(f"rendered {rendered} new frames ({len(list(fdir.glob('frame_*.png')))} total) "
          f"[{time.time()-t0:.0f}s]")

    if not args.no_video:
        import subprocess
        cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-pattern_type", "glob",
               "-i", str(fdir / "frame_*.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", args.out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f"ffmpeg -> {args.out}" if r.returncode == 0 else f"ffmpeg FAILED:\n{r.stderr[-800:]}")
    print(f"DONE [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
