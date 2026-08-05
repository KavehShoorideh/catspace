#!/usr/bin/env python
"""make_tent_movie.py -- the TENT forming over training, from full-game replays.

The simplex movie could reuse the precomputed trunk-feature memmap, but the tent needs full-game
replays (every ply), and those positions are not in that cache. Replaying per checkpoint would
cost ~500s x 600 frames. So: replay ONCE, cache the frozen TRUNK features for those positions in
memory, then per checkpoint run only the 140k-param head over them. The trunk is frozen, so its
features are identical for every checkpoint -- caching them is exact, not an approximation.

Rows are 2 plies = 1 full move: the mover alternates every ply and this field is turn-dependent,
so 1-ply rows stripe by parity. Axes and colour scale are fixed across frames so motion is the
distribution moving, not a rescaling.
"""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path
import numpy as np, torch

from catspace.research.tools.embeddings.basin_tent_fullgames import replay
from catspace.research.tools.embeddings.basin_hazard_flow import uniform_human, uniform_sf
from catspace.research.tools.embeddings.basin_tent import white_pov_x, COLOR_WHITE_WIN, COLOR_BLACK_WIN
from catspace.research.tools.embeddings.basin_simplex_chart import INK, MUTED
from catspace.research.tools.training_infra.losses import basin_logp
from catspace.research.components.encoder.approaches.reachability_field.src.iqe_head import IQEHead

STEP_RE = re.compile(r"_step(\d+)\.pt$")


@torch.no_grad()
def trunk_feats(field, planes, batch=2048):
    """Frozen-trunk features (B,C,8,8) -- identical for every checkpoint, so cache once."""
    out = []
    for i in range(0, len(planes), batch):
        x = torch.as_tensor(np.stack(planes[i:i + batch])).float().to(field.dev)
        field.trunk(x); t = field._f["t"]
        if field.tokens:
            B = x.shape[0]; C = t.shape[-1]
            t = t.reshape(B, 64, C).permute(0, 2, 1).reshape(B, C, 8, 8)
        out.append(t.cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-glob", default="artifacts/experiments/movie4/iqe_4pole_30k_step*.pt")
    ap.add_argument("--onnx", default="assets/engines/lc0/t1-256x10.onnx")
    ap.add_argument("--sf-moves", default="data/derived/opening_pool_sfsf_moves.tsv")
    ap.add_argument("--human-records", default="data/records/lichess_2019-01")
    ap.add_argument("--n-games", type=int, default=350)
    ap.add_argument("--max-ply", type=int, default=100)
    ap.add_argument("--every", type=int, default=1, help="use every Nth checkpoint")
    ap.add_argument("--frames", default="artifacts/experiments/movie4/tent_frames")
    ap.add_argument("--out", default="artifacts/experiments/movie4/tent_movie.mp4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    t0 = time.time()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField

    ck = sorted([p for p in Path().glob(args.ckpt_glob) if STEP_RE.search(p.name)],
                key=lambda p: int(STEP_RE.search(p.name).group(1)))[::args.every]
    print(f"{len(ck)} checkpoints, steps {int(STEP_RE.search(ck[0].name).group(1))}.."
          f"{int(STEP_RE.search(ck[-1].name).group(1))}", flush=True)

    field = ReachabilityField(onnx=args.onnx, head=str(ck[-1]))
    rng = np.random.default_rng(0)
    pools = {"human": uniform_human(args.human_records, args.n_games, rng),
             "SF-vs-SF": uniform_sf(args.sf_moves, args.n_games, rng)}
    cache = {}
    for name, pool in pools.items():
        P, PL = [], []
        for _, _, ucis, _ in pool:
            pl, _, _ = replay(ucis, args.max_ply)
            if pl is None or len(pl) < 4: continue
            P.extend(list(pl.astype(np.float32))); PL.append(np.arange(len(pl)))
        f = trunk_feats(field, P)
        cache[name] = (f.to(args.device), np.concatenate(PL))
        print(f"  cached {name}: {tuple(f.shape)} [{time.time()-t0:.0f}s]", flush=True)

    p0 = torch.load(ck[0], map_location=args.device, weights_only=False); cfg = p0["cfg"]
    net = IQEHead(in_ch=cfg["in_ch"], d=cfg["d"], components=cfg["components"],
                  adapter_ch=cfg["adapter_ch"]).to(args.device).eval()
    gx = np.linspace(-1, 1, 101); gy = np.arange(0, args.max_ply + 2, 2)   # 2 plies = 1 move
    fdir = Path(args.frames); fdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for c in ck:
        step = int(STEP_RE.search(c.name).group(1))
        png = fdir / f"tent_{step:06d}.png"
        if png.exists(): continue
        net.load_state_dict(torch.load(c, map_location=args.device,
                                       weights_only=False)["state_dict"], strict=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True)
        with torch.no_grad():
            for ax, (name, (f, ply)) in zip(axes, cache.items()):
                pr = basin_logp(net.d_poles(net.phi(f)), net.temperature).exp().cpu().numpy()
                x = white_pov_x(pr, ply)
                H, _, _ = np.histogram2d(x, ply, bins=[gx, gy])
                with np.errstate(invalid="ignore"):
                    C = H / np.maximum(H.sum(0, keepdims=True), 1)
                ax.pcolormesh(gx, gy, np.ma.masked_less(C, 2e-3).T,
                              cmap="Blues" if name == "human" else "Reds",
                              norm=LogNorm(vmin=2e-3, vmax=1.0), shading="flat")
                ax.invert_yaxis(); ax.set_xlim(-1.02, 1.02); ax.set_ylim(args.max_ply, 0)
                ax.axvline(0, color=MUTED, lw=.6, ls=":")
                ax.set_xlabel("P(White wins) - P(Black wins)"); ax.set_title(name, color=INK)
                ax.text(-1.0, args.max_ply*.03, "White", fontsize=8, color=COLOR_WHITE_WIN)
                ax.text(1.0, args.max_ply*.03, "Black", fontsize=8, color=COLOR_BLACK_WIN, ha="right")
        axes[0].set_ylabel("ply (start at top)")
        fig.suptitle(f"The tent forming -- step {step:,}   (P(x | move), full-game replays)")
        fig.tight_layout(); fig.savefig(png, dpi=105); plt.close(fig); n += 1
        if n % 50 == 0: print(f"  {n} frames [{time.time()-t0:.0f}s]", flush=True)
    print(f"rendered {n} frames [{time.time()-t0:.0f}s]", flush=True)
    import subprocess
    r = subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps), "-pattern_type", "glob",
                        "-i", str(fdir / "tent_*.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", args.out], capture_output=True, text=True)
    print(f"ffmpeg -> {args.out}" if r.returncode == 0 else f"FAILED\n{r.stderr[-600:]}")
    print(f"DONE [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
