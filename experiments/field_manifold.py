#!/usr/bin/env python
"""experiments/field_manifold.py -- does the geometry field's embedding space cluster
naturally? Embed the nucleus training positions with iqe_geom (board-only F), then:
  - k-NN MATERIAL PURITY in the raw 512-d space (do a point's nearest neighbors share
    its material class?) + a DTM-neighbor check (are neighbors close in DTM?) -- the
    quantitative "does it cluster" answer, independent of any 2-D projection.
  - UMAP -> 2-D, colored by material class and by DTM, for the visual.

Usage: .venv/bin/python experiments/field_manifold.py --ckpt data/derived/sep/iqe_geom.pt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch, chess, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.data.encode import board_from_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="data/derived/sep/iqe_geom.pt")
ap.add_argument("--data", default="data/derived/lichess_nearmate.npz")
ap.add_argument("--n", type=int, default=12000)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--out", default="artifacts/experiments/field_manifold.png")
args = ap.parse_args()
dev = pick_device("auto")
fb, _ = load_ckpt(Path(args.ckpt), dev); fb.eval()
om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
nz = np.load(args.data); won = np.flatnonzero(nz["dtm"] > 0)
rng = np.random.default_rng(0)
sel = rng.choice(won, min(args.n, len(won)), replace=False)
pk, mt, dtm = nz["packed"][sel], nz["meta"][sel], nz["dtm"][sel].astype(np.float32)
mat = np.array(["".join(sorted(p.symbol() for p in board_from_packed(pk[i], mt[i]).piece_map().values()))
                for i in range(len(sel))])


def embF(pk, mt, bs=2048):
    out = []
    for i in range(0, len(pk), bs):
        pl = feature_planes(pk[i:i+bs], mt[i:i+bs]); pl[:, (18, 19)] = 0.0    # board-only
        o = torch.from_numpy(np.tile(om, (len(pl), 1))).to(dev)
        with torch.no_grad():
            out.append(fb.embed_F(torch.from_numpy(pl).to(dev), o).cpu().numpy())
    return np.concatenate(out)


print(f"[stage] embedding {len(sel)} nucleus positions with {Path(args.ckpt).name} (board-only F)...", flush=True)
E = embF(pk, mt).astype(np.float32)

# k-NN purity in the raw embedding space (cosine, like the store)
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=args.k + 1, metric="cosine").fit(E)
_, idx = nn.kneighbors(E)
idx = idx[:, 1:]                                                    # drop self
mat_purity = np.mean([(mat[idx[i]] == mat[i]).mean() for i in range(len(sel))])
dtm_nbr_gap = np.mean([np.abs(dtm[idx[i]] - dtm[i]).mean() for i in range(len(sel))])
# baselines: random purity = sum(p_c^2); random DTM gap = mean|dtm_i-dtm_j|
_, cnt = np.unique(mat, return_counts=True); rand_purity = float(((cnt / cnt.sum()) ** 2).sum())
rand_gap = float(np.mean(np.abs(dtm[rng.permutation(len(dtm))] - dtm)))
print(f"k-NN (k={args.k}) MATERIAL purity = {mat_purity:.3f}   (random baseline {rand_purity:.3f})")
print(f"k-NN mean |ΔDTM| to neighbors = {dtm_nbr_gap:.2f} plies   (random baseline {rand_gap:.2f})")

print("[stage] UMAP -> 2D...", flush=True)
import umap
XY = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0).fit_transform(E)

top = [m for m, _ in sorted(zip(*np.unique(mat, return_counts=True)), key=lambda x: -x[1])][:8]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.5))
for m in top:
    s = mat == m; a1.scatter(XY[s, 0], XY[s, 1], s=4, alpha=0.5, label=m)
a1.legend(markerscale=3, fontsize=8, loc="best"); a1.set_title(f"colored by MATERIAL (top 8)  -- k-NN purity {mat_purity:.2f} (rand {rand_purity:.2f})")
sc = a2.scatter(XY[:, 0], XY[:, 1], s=4, c=np.minimum(dtm, 30), cmap="viridis_r", alpha=0.6)
plt.colorbar(sc, ax=a2, label="DTM (plies to mate, capped 30)")
a2.set_title(f"colored by DTM  -- neighbors within {dtm_nbr_gap:.1f} plies (rand {rand_gap:.1f})")
for a in (a1, a2): a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"iqe_geom embedding manifold (UMAP, cosine) -- {len(sel)} nucleus positions, board-only F", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(args.out, dpi=115)
print(f"VERDICT FIELD_MANIFOLD mat_purity={mat_purity:.3f}(rand {rand_purity:.3f}) "
      f"dtm_nbr_gap={dtm_nbr_gap:.2f}(rand {rand_gap:.2f}) -> {args.out}")
