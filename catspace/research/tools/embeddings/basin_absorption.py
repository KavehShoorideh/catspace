"""1-ply tent + the answer to 'if flow is inward, how does anyone reach mate?': ABSORPTION.

Mean drift is computed over SURVIVORS. A game that converts ends and leaves the population, so
the trajectories still at large |x| are exactly the ones that came back -- the flow field is
survivorship-biased inward by construction. What actually carries games to a result is (a) the
diffusion (spread of per-ply steps), and (b) absorption: the rate at which games simply STOP in
a cell. This measures all three on the same grid.
"""
import sys, time
import numpy as np, torch
from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField
from catspace.research.tools.embeddings.basin_tent_fullgames import replay
from catspace.research.tools.embeddings.basin_hazard_flow import uniform_human, uniform_sf
from catspace.research.tools.embeddings.basin_tent import white_pov_x, COLOR_WHITE_WIN, COLOR_BLACK_WIN
from catspace.research.tools.embeddings.basin_simplex_chart import INK, MUTED
from catspace.research.tools.training_infra.losses import basin_logp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CK, N, MAXPLY = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t0 = time.time()
field = ReachabilityField(onnx="/Users/kav/code/remote/github/catspace/assets/engines/lc0/t1-256x10.onnx", head=CK)
rng = np.random.default_rng(0)
pools = {"human": uniform_human("/Users/kav/code/remote/github/catspace/data/records/lichess_2019-01", N, rng),
         "SF-vs-SF": uniform_sf("/Users/kav/code/remote/github/catspace/data/derived/opening_pool_sfsf_moves.tsv", N, rng)}
G = {}
for name, pool in pools.items():
    xs, ps, last = [], [], []
    for _, _, ucis, _ in pool:
        pl, _, trunc = replay(ucis, MAXPLY)
        if pl is None or len(pl) < 4: continue
        with torch.no_grad():
            o=[]
            for i in range(0,len(pl),4096):
                phi=field.phi_from_planes(list(pl[i:i+4096].astype(np.float32)))
                o.append(basin_logp(field.head.d_poles(phi), field.head.temperature).exp().cpu().numpy())
        pr=np.concatenate(o); ply=np.arange(len(pr))
        xs.append(white_pov_x(pr,ply)); ps.append(ply)
        last.append(bool(not trunc))          # game genuinely ENDED here (not ply-capped)
    G[name]=(xs,ps,last)
    print(f"  {name}: {len(xs)} games, {sum(len(a) for a in xs):,} positions [{time.time()-t0:.0f}s]", flush=True)

gx=np.linspace(-1,1,61); gy=np.arange(0,MAXPLY+1,3)
cx=0.5*(gx[:-1]+gx[1:]); cy=0.5*(gy[:-1]+gy[1:])
def fields(xs,ps,last):
    X=np.concatenate(xs); P=np.concatenate(ps)
    occ,_,_=np.histogram2d(X,P,bins=[gx,gy])                     # occupancy
    # per-ply steps (drift + diffusion), within-game only
    x0=np.concatenate([a[:-1] for a in xs]); p0=np.concatenate([b[:-1] for b in ps])
    dx=np.concatenate([np.diff(a) for a in xs])
    n,_,_=np.histogram2d(x0,p0,bins=[gx,gy]); s1,_,_=np.histogram2d(x0,p0,bins=[gx,gy],weights=dx)
    s2,_,_=np.histogram2d(x0,p0,bins=[gx,gy],weights=dx**2)
    with np.errstate(invalid="ignore",divide="ignore"):
        mu=s1/n; var=s2/n-mu**2
    # ABSORPTION: the final position of each game that genuinely ended
    ex=np.array([a[-1] for a,L in zip(xs,last) if L]); ep=np.array([b[-1] for b,L in zip(ps,last) if L])
    end,_,_=np.histogram2d(ex,ep,bins=[gx,gy])
    with np.errstate(invalid="ignore",divide="ignore"):
        rate=end/occ                                              # P(game ends here | here)
    return occ,mu,np.sqrt(np.maximum(var,0)),rate,end
R={k:fields(*v) for k,v in G.items()}

fig,axes=plt.subplots(2,3,figsize=(18,10),sharey=True)
for r,(name,(occ,mu,sd,rate,end)) in enumerate(R.items()):
    m=occ>=25
    a=axes[r,0]; a.quiver(*np.meshgrid(cx,cy,indexing="ij"),np.where(m,mu,np.nan),
        np.zeros_like(mu),color="#2a78d6" if r==0 else "#e34948",angles="xy",
        scale_units="xy",scale=np.nanmax(np.abs(mu[m]))/0.16,width=0.005)
    a.set_title(f"{name}: MEAN DRIFT (survivors only)",color=INK)
    b=axes[r,1]; pc=b.pcolormesh(gx,gy,np.ma.masked_invalid(np.where(m,sd,np.nan)).T,cmap="viridis")
    fig.colorbar(pc,ax=b,shrink=.8,label="sd of per-ply step"); b.set_title(f"{name}: DIFFUSION",color=INK)
    c=axes[r,2]; pc2=c.pcolormesh(gx,gy,np.ma.masked_invalid(np.where(m,rate,np.nan)).T,cmap="magma_r")
    fig.colorbar(pc2,ax=c,shrink=.8,label="P(game ends here)"); c.set_title(f"{name}: ABSORPTION",color=INK)
    for a_ in axes[r]:
        a_.invert_yaxis(); a_.set_xlim(-1.02,1.02); a_.set_ylim(MAXPLY,0); a_.axvline(0,color=MUTED,lw=.6,ls=":")
        a_.set_xlabel("P(White wins) - P(Black wins)")
    axes[r,0].set_ylabel(f"{name}\nply")
fig.suptitle("Why inward flow still reaches mate: drift is measured over SURVIVORS, but games LEAVE\n"
             "by absorption at the edges -- the converted trajectories are gone from the drift field")
fig.tight_layout(); fig.savefig("artifacts/experiments/basin_absorption.png",dpi=140)

print("\nWHERE DO GAMES ACTUALLY END?  (share of all endings, by |x| band)")
print(f"  {'|x| band':>12s} {'human':>9s} {'SF-vs-SF':>9s}")
for lo,hi in [(0,.2),(.2,.5),(.5,.8),(.8,1.01)]:
    row=[]
    for name,(occ,mu,sd,rate,end) in R.items():
        sel=(np.abs(cx)>=lo)&(np.abs(cx)<hi)
        row.append(100*end[sel].sum()/max(end.sum(),1))
    print(f"  {lo:>4.1f}-{hi:<6.2f} {row[0]:>8.1f}% {row[1]:>8.1f}%")
print("\nMEAN |drift| vs DIFFUSION (median over well-sampled cells)")
for name,(occ,mu,sd,rate,end) in R.items():
    m=occ>=25
    print(f"  {name:9s} |mean drift| {np.nanmedian(np.abs(mu[m])):.4f}   diffusion sd {np.nanmedian(sd[m]):.4f}"
          f"   ratio {np.nanmedian(sd[m])/max(np.nanmedian(np.abs(mu[m])),1e-9):.1f}x")
print(f"wrote artifacts/experiments/basin_absorption.png [{time.time()-t0:.0f}s]")
