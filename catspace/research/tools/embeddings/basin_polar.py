#!/usr/bin/env python
"""basin_polar.py -- the field's OWN coordinate system, no embedding algorithm.

Kaveh's design intent for the start pole: absorb the dominant ply variance into an explicit radial
coordinate so the remaining directional freedom steers toward one basin or another. That is a
POLAR coordinate system, and the field already computes both halves:

    radius  r      = time elapsed          (the start pole tracks true ply at rho 0.967)
    angle   theta  = which basin           (angular position among the three outcome poles)

So there is nothing to embed. No kNN graph, no force constants, no symmetrising, no Procrustes --
all of which were workarounds for not having a coordinate system. Two attempts at a neighbour-graph
layout failed here for a structural reason worth recording: symmetrised, d(position->pole) ~ 116
against a median position-to-position distance of 91, so the poles are FARTHER than typical points,
acquire almost no kNN edges, and drift to the periphery together. A neighbour-graph layout is the
wrong instrument for points defined by being far from everything.

Both fields use the SAME radius (true ply) so the comparison isolates what the start pole did to
the ANGULAR structure. A separate panel checks the 4-pole field's own radius against true ply.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

from catspace.research.components.encoder.approaches.reachability_field.src.iqe_head import IQEHead
from catspace.research.tools.training_infra.losses import basin_logp, WIN, DRAW, LOSS
from catspace.research.tools.embeddings.basin_simplex_chart import (
    bary_to_xy, VERTS, COLOR_WIN, COLOR_DRAW, COLOR_LOSS, INK, MUTED)

POLE_ANGLE = {k: np.arctan2(*(VERTS[k] - VERTS.mean(0))[::-1]) for k in (WIN, DRAW, LOSS)}
PNAME = {WIN: "mover wins", DRAW: "draw", LOSS: "mover loses"}
PCOL = {WIN: COLOR_WIN, DRAW: COLOR_DRAW, LOSS: COLOR_LOSS}


def load(ck, dev):
    p = torch.load(ck, map_location=dev, weights_only=False); c = p["cfg"]
    n = IQEHead(in_ch=c["in_ch"], d=c["d"], components=c["components"],
                adapter_ch=c["adapter_ch"]).to(dev)
    info, _ = n.load_compat(p["state_dict"]); n.eval()
    # has_start is False when the checkpoint predates the START pole: its outcome poles load fine
    # (padded), but row 3 stays at init, so its radius must come from true ply, not from the field.
    has_start = not any("rows loaded" in m or m == "poles" for m in info)
    return n, has_start


def polar_of(p):
    """(N,3) basin probabilities -> (theta, confidence). theta is the direction in the simplex from
    its centroid; confidence is how far out of the centre it sits (0 = fully undetermined)."""
    xy = bary_to_xy(p) - VERTS.mean(0)
    return np.arctan2(xy[:, 1], xy[:, 0]), np.linalg.norm(xy, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt3", default="artifacts/experiments/movie/iqe_poles_30k_latest.pt")
    ap.add_argument("--ckpt4", default="artifacts/experiments/movie4/iqe_4pole_30k_latest.pt")
    ap.add_argument("--combined", default="data/derived/field_combined_sub600k.npz")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--max-ply", type=int, default=160)
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_polar")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    t0 = time.time()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    z = np.load(args.combined, allow_pickle=True)
    meta = eval(str(z["_meta"][0])); mm = np.load(meta["feats"][0], mmap_mode="r")
    split = z["orig_source"] if "orig_source" in z.files else z["source"]
    rng = np.random.default_rng(0)
    take = np.sort(np.concatenate([
        rng.choice(np.flatnonzero((split == s) & (z["ply"] <= args.max_ply)), args.n, replace=False)
        for s in (0, 1)]))
    source, ply = split[take], z["ply"][take]
    r_true = np.log1p(ply + 1.0)                       # SAME radius for both fields -> fair

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 13.5), subplot_kw={"projection": "polar"})
    stats = []
    for row, (nm, ck) in enumerate([("without start pole", args.ckpt3), ("with start pole", args.ckpt4)]):
        net, ok = load(ck, args.device)
        with torch.no_grad():
            P, R = [], []
            for i in range(0, len(take), 8192):
                x = torch.from_numpy(np.asarray(mm[z["local_row"][take[i:i + 8192]]],
                                                dtype=np.float32)).to(args.device)
                e = net.phi(x)
                P.append(basin_logp(net.d_poles(e), net.temperature).exp().cpu().numpy())
                if ok:
                    R.append(net.d_from_start(e).cpu().numpy())
        p = np.concatenate(P)
        th, conf = polar_of(p)
        if R:
            rf = np.log1p(np.concatenate(R))
            stats.append((nm, float(np.corrcoef(rf, r_true)[0, 1])))
        for col, (sn, m) in enumerate([("human", source == 0), ("SF-vs-SF", source == 1)]):
            ax = axes[row, col]
            h, te, re_ = np.histogram2d(th[m] % (2*np.pi), r_true[m],
                                        bins=[np.linspace(0, 2*np.pi, 145),
                                              np.linspace(0, r_true.max(), 90)])
            T, Rg = np.meshgrid(te, re_, indexing="ij")
            ax.pcolormesh(T, Rg, np.ma.masked_less(h, 1),
                          cmap="Blues" if sn == "human" else "Reds",
                          norm=LogNorm(vmin=1, vmax=max(h.max(), 10)), shading="flat")
            for k in (WIN, DRAW, LOSS):
                a = POLE_ANGLE[k] % (2*np.pi)
                ax.plot([a, a], [0, r_true.max()], color=PCOL[k], lw=1.6, alpha=.75, zorder=5)
                ax.text(a, r_true.max()*1.06, PNAME[k], color=PCOL[k], fontsize=8.5,
                        ha="center", va="center")
            ax.set_title(f"{nm} -- {sn}", color=INK, pad=22)
            ax.set_rlabel_position(100); ax.grid(alpha=.25)
            ax.set_xticklabels([])
    fig.suptitle("The field's own polar coordinates: radius = log1p(ply) (time), angle = which basin\n"
                 "games start at the centre and flow outward; the coloured spokes are the three "
                 "outcome poles", y=0.995)
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}.png", dpi=140)
    for nm, c in stats:
        print(f"  {nm}: corr(field's own radius d(start->s), true log1p(ply)) = {c:.3f}")
    print(f"wrote {args.out_prefix}.png [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
