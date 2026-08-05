"""Where does the SF-vs-SF quantisation come from: the softmax, the IQE distances, or phi?"""
import numpy as np, torch
from catspace.research.tools.embeddings.basin_simplex_chart import load_head
from catspace.research.tools.training_infra.losses import basin_logp
from catspace.research.tools.embeddings.basin_tent import white_pov_x
z=np.load('data/derived/field_combined_sub600k.npz',allow_pickle=True)
meta=eval(str(z['_meta'][0])); mm=np.load(meta['feats'][0],mmap_mode='r')
net=load_head('artifacts/experiments/movie4/iqe_4pole_30k_latest.pt','mps')
T=float(net.temperature.detach())
rng=np.random.default_rng(0)
res={}
for nm,src in [("SF-vs-SF",1),("human",0)]:
    idx=np.flatnonzero(z['orig_source']==src)
    take=np.sort(rng.choice(idx,120000,replace=False))
    with torch.no_grad():
        ds=[];ph=[]
        for i in range(0,len(take),8192):
            x=torch.from_numpy(np.asarray(mm[z['local_row'][take[i:i+8192]]],dtype=np.float32)).to('mps')
            e=net.phi(x); ph.append(e.cpu().numpy()); ds.append(net.d_poles(e).cpu().numpy())
    d=np.concatenate(ds); phi=np.concatenate(ph)
    p=torch.softmax(-torch.log1p(torch.tensor(d))/T,dim=-1).numpy()
    xx=white_pov_x(p, z['ply'][take])
    res[nm]=(d,p,xx,phi)
    h,e=np.histogram(xx,bins=400,range=(-1,1))
    top=np.argsort(h)[-6:][::-1]
    print(f"\n== {nm} ==  top-6 spikes in x (bin centre, count, share)")
    for i in top:
        c=0.5*(e[i]+e[i+1]); print(f"   x={c:+.3f}  n={h[i]:>6d}  {100*h[i]/len(xx):.2f}%")
    print(f"   median bin count {int(np.median(h))}  -> top spike is {h[top[0]]/max(np.median(h),1):.0f}x the median")
    # is the discreteness already in the DISTANCES?
    print(f"   d(->win)  unique values in 120k samples: {len(np.unique(np.round(d[:,0],4))):>7,}")
    print(f"   d(->draw) unique: {len(np.unique(np.round(d[:,1],4))):>7,}   d(->loss) unique: {len(np.unique(np.round(d[:,2],4))):>7,}")
    print(f"   frac with d(->draw) EXACTLY 0: {100*(d[:,1]==0).mean():.2f}%   d(->win)==0: {100*(d[:,0]==0).mean():.2f}%  d(->loss)==0: {100*(d[:,2]==0).mean():.2f}%")
    print(f"   phi: unique rounded rows {len(np.unique(np.round(phi,3),axis=0)):,} of {len(phi):,}")
