"""Tent at 1-ply resolution from full-game replays -- no 6-ply rectangles."""
import sys, time
import numpy as np, torch
from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField
from catspace.research.tools.embeddings.basin_tent_fullgames import tent_density
from catspace.research.tools.embeddings.basin_hazard_flow import uniform_human, uniform_sf
from catspace.research.tools.embeddings.basin_simplex_chart import INK, MUTED
from catspace.research.tools.embeddings.basin_tent import COLOR_WHITE_WIN, COLOR_BLACK_WIN
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CK, N, MAXPLY = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t0=time.time()
field = ReachabilityField(onnx="/Users/kav/code/remote/github/catspace/assets/engines/lc0/t1-256x10.onnx", head=CK)
rng = np.random.default_rng(0)
pools = {"human": uniform_human("/Users/kav/code/remote/github/catspace/data/records/lichess_2019-01", N, rng),
         "SF-vs-SF": uniform_sf("/Users/kav/code/remote/github/catspace/data/derived/opening_pool_sfsf_moves.tsv", N, rng)}
import os
CACHE="artifacts/experiments/basin_tent_1ply_data.npz"
if os.path.exists(CACHE) and os.environ.get("REPLOT"):
    _z=np.load(CACHE); D={k:(_z[k+"_x"],_z[k+"_p"],_z[k+"_q"]) for k in ("human","SF-vs-SF")}
else:
    D = tent_density(field, pools, MAXPLY)
    np.savez(CACHE, **{f"{k}_{n}":v for k,(x,p,q) in D.items()
                       for n,v in (("x",x),("p",p),("q",q))})
for k,(x,p,_q) in D.items(): print(f"  {k}: {len(x):,} positions from {len(pools[k])} games [{time.time()-t0:.0f}s]", flush=True)

gx = np.linspace(-1, 1, 121)                 # 0.0167-wide x bins
# TWO plies = ONE FULL MOVE per row. Not a workaround for the sampler comb (full replays have no
# comb) -- the mover alternates every ply and this field is turn-dependent (measured: lag-1
# autocorrelation 0.42 vs lag-2 0.63), so 1-ply rows alternate mover and stripe by parity. A row
# of one full move contains both movers and cancels it exactly, and a move is the natural unit.
gy = np.arange(0, MAXPLY + 2, 2)
fig, axes = plt.subplots(2, 2, figsize=(13.5, 13), sharey=True)
for col,(name,(x,p,_q)) in enumerate(D.items()):
    H,_,_ = np.histogram2d(x, p, bins=[gx, gy])
    cm = "Blues" if name=="human" else "Reds"
    # ROW 0: raw joint -- shows where games actually SPEND time, and the attrition as they end.
    ax = axes[0,col]
    pc = ax.pcolormesh(gx, gy, np.ma.masked_less(H,1).T, cmap=cm,
                       norm=LogNorm(vmin=1, vmax=max(H.max(),10)), shading="flat")
    fig.colorbar(pc, ax=ax, shrink=0.78, label="positions (log)")
    ax.set_title(f"{name} -- joint density ({len(x):,} positions)", color=INK)
    # ROW 1: P(x | ply). Row-normalizing is legitimate at 1-ply resolution -- it was only a comb
    # WORKAROUND when rows were 6 plies wide. It is what makes the tent's fan visible, because the
    # joint is dominated by how long games last rather than by where they are.
    with np.errstate(invalid="ignore", divide="ignore"):
        C = H / H.sum(0, keepdims=True)
    ax = axes[1,col]
    pc = ax.pcolormesh(gx, gy, np.ma.masked_invalid(np.ma.masked_less(C,1e-3)).T, cmap=cm,
                       norm=LogNorm(vmin=1e-3, vmax=1.0), shading="flat")
    fig.colorbar(pc, ax=ax, shrink=0.78, label="P(x | ply)")
    ax.set_title(f"{name} -- conditional P(x | ply)", color=INK)
for ax in axes.ravel():
    ax.invert_yaxis(); ax.set_xlim(-1.02,1.02); ax.set_ylim(MAXPLY,0)
    ax.axvline(0, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("P(White wins) - P(Black wins)")
    ax.text(-1.0, MAXPLY*0.02, "White wins", fontsize=8, color=COLOR_WHITE_WIN)
    ax.text(1.0, MAXPLY*0.02, "Black wins", fontsize=8, color=COLOR_BLACK_WIN, ha="right")
for r in (0,1):
    axes[r,0].set_ylabel("ply  (start at the top, game descends)")
fig.suptitle("The tent at 1-PLY resolution, from full-game replays -- no stride comb, no row binning\n"
             "top: joint density (where games spend time)   bottom: P(x | ply) (the fan)")
fig.tight_layout(); fig.savefig("artifacts/experiments/basin_tent_1ply.png", dpi=150)
print(f"wrote artifacts/experiments/basin_tent_1ply.png [{time.time()-t0:.0f}s]")
