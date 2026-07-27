#!/usr/bin/env python
"""experiments/engine_vs_human_basins.py -- Kaveh's control: do the 3 W/D/L basins come out SHARP
under (near-)perfect ENGINE play vs the leaky HUMAN jumble? Embed engine (SF-vs-SF) and human
positions with the SAME field phi, UMAP each colored by the ACTUAL game OUTCOME (the near-perfect
committor under perfect play), and quantify BASIN PURITY = 1 - outcome-entropy within phi-microstates
(pure = same outcome for phi-neighbors = sharp basin; mixed = leaky). Also the material-stacked
outcome purity: does the engine commit to a basin at HIGH material where humans are still a jumble?
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_clock_field import ClockField
from catspace.train.scaffold import resolve_device


def load_embed(net, dev, path, n, rng):
    z = np.load(path); planes = z["planes"]; result = z["result"] if "result" in z else \
        np.where(z["ending"] == 0, 1, np.where(z["ending"] == 5, -1, 0)).astype(np.int8)
    sub = rng.integers(0, len(planes), min(n, len(planes)))
    pieces = planes[sub][:, 0:12].reshape(len(sub), 12, -1).sum(axis=(1, 2)).astype(int)
    phi = []
    for i in range(0, len(sub), 4096):
        x = torch.from_numpy(planes[sub[i:i+4096]].astype(np.float32)).to(dev)
        with torch.no_grad():
            phi.append(net.phi(x).cpu().numpy())
    return np.concatenate(phi), result[sub].astype(int), pieces


def basin_purity(phi, outcome, k=120, seed=0):
    """microstate outcome purity: cluster phi, per-cluster max-outcome-fraction (1=pure basin)."""
    from sklearn.cluster import MiniBatchKMeans
    lab = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096).fit_predict(phi)
    purities, ent = [], []
    for c in range(k):
        m = lab == c
        if m.sum() < 10:
            continue
        vals, cnts = np.unique(outcome[m], return_counts=True); p = cnts / cnts.sum()
        purities.append(p.max()); ent.append(-(p * np.log(p + 1e-12)).sum())
    return float(np.mean(purities)), float(np.mean(ent))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/field_fullgame_v3_final.pt")
    ap.add_argument("--engine", default="data/derived/engine_sfsf.npz")
    ap.add_argument("--human", default="data/derived/field_fullgame_v1.npz")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default="artifacts/experiments/engine_vs_human_basins.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); rng = np.random.default_rng(args.seed)
    p = torch.load(args.ckpt, map_location=dev, weights_only=False); cfg = p["cfg"]
    net = ClockField(cfg["d"], ch=cfg["ch"], blocks=cfg["blocks"], in_planes=112).to(dev)
    net.load_state_dict(p["state_dict"]); net.eval()

    eng_phi, eng_out, eng_pc = load_embed(net, dev, args.engine, args.n, rng)
    hum_phi, hum_out, hum_pc = load_embed(net, dev, args.human, args.n, rng)
    ep, ee = basin_purity(eng_phi, eng_out); hp, he = basin_purity(hum_phi, hum_out)
    print(f"BASIN PURITY (phi-microstate mean max-outcome-fraction; 1.0 = perfectly sharp basins):")
    print(f"  ENGINE (SF vs SF): purity {ep:.3f} | outcome-entropy {ee:.3f} | W/D/L "
          f"{(eng_out==1).mean():.0%}/{(eng_out==0).mean():.0%}/{(eng_out==-1).mean():.0%} n={len(eng_out)}")
    print(f"  HUMAN  (lichess):  purity {hp:.3f} | outcome-entropy {he:.3f} | W/D/L "
          f"{(hum_out==1).mean():.0%}/{(hum_out==0).mean():.0%}/{(hum_out==-1).mean():.0%} n={len(hum_out)}")
    # material-stacked purity: sharp basins already at HIGH material for the engine?
    print("\noutcome purity by material bucket (engine vs human):")
    for lo, hi in [(27, 33), (19, 27), (11, 19), (3, 11)]:
        em = (eng_pc >= lo) & (eng_pc < hi); hm = (hum_pc >= lo) & (hum_pc < hi)
        epb = basin_purity(eng_phi[em], eng_out[em], k=40)[0] if em.sum() > 400 else float("nan")
        hpb = basin_purity(hum_phi[hm], hum_out[hm], k=40)[0] if hm.sum() > 400 else float("nan")
        print(f"  {lo}-{hi-1}p: engine {epb:.3f} vs human {hpb:.3f}  (n_eng {int(em.sum())} n_hum {int(hm.sum())})")

    import umap, matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {1: "#3b6fb0", 0: "#9a9a9a", -1: "#c04040"}
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for a, (phi, out, name, pur) in zip(ax, [(eng_phi, eng_out, f"ENGINE SF-vs-SF (near-perfect)", ep),
                                             (hum_phi, hum_out, "HUMAN lichess 1400-1800", hp)]):
        s = rng.choice(len(phi), size=min(9000, len(phi)), replace=False)
        xy = umap.UMAP(n_neighbors=25, min_dist=0.15, random_state=args.seed).fit_transform(phi[s])
        a.scatter(xy[:, 0], xy[:, 1], c=[col[o] for o in out[s]], s=6, alpha=0.5)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{name}\nbasin purity {pur:.2f}  (blue=win grey=draw red=loss)")
    fig.suptitle("Do the 3 basins sharpen under perfect play? Field UMAP colored by actual outcome", fontsize=13)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nVERDICT -> {args.out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
