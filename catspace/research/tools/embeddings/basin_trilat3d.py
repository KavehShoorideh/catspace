#!/usr/bin/env python
"""basin_trilat3d.py -- trilateration in 3-D, with an interactive viewer.

2-D trilateration misfit the four pole distances by ~55% (median rms 0.358 against an anchor
radius of 0.65) and showed reflection artifacts: with 4 distances and only 2 unknowns the system is
overdetermined and simply cannot be satisfied. In 3-D it is 4 constraints for 3 unknowns -- nearly
determined -- so the residual should collapse if the geometry is genuinely three-dimensional, and
the left/right ambiguity that produced the doubled lobes largely resolves.

The residual is the test. It is printed against the 2-D value, and if 3-D does not help then the
structure is not simply higher-dimensional and something else is going on.

Anchors form a TETRAHEDRON: START at the apex, the three outcome poles in a plane below it -- which
is the cone-from-the-start picture the whole design has been aiming at.

The viewer is a single self-contained HTML file: no CDN, no libraries, plain canvas + a painter's
algorithm, so it works offline.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, torch

from catspace.research.components.encoder.approaches.reachability_field.src.iqe_head import IQEHead
from catspace.research.tools.training_infra.losses import WIN, DRAW, LOSS

START = 3
PNAME = {WIN: "mover wins", DRAW: "draw", LOSS: "mover loses", START: "START"}
PCOL = {WIN: "#0ca30c", DRAW: "#8a8985", LOSS: "#d03b3b", START: "#7b5cd6"}


def anchors3d(r_out):
    A = np.zeros((4, 3))
    for i, k in enumerate((WIN, DRAW, LOSS)):
        th = np.pi / 2 + i * 2 * np.pi / 3
        A[k] = [r_out * np.cos(th), r_out * np.sin(th), 0.0]
    A[START] = [0.0, 0.0, r_out * 1.2]              # apex: games are pushed down and out from it
    return A


def trilaterate(R, A, iters=140, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    w = 1.0 / np.maximum(R, 1e-6)
    Y = (w[:, :, None] * A[None]).sum(1) / w.sum(1)[:, None]
    Y = Y + rng.normal(0, 1e-3, Y.shape)
    for _ in range(iters):
        d = Y[:, None, :] - A[None]
        n = np.linalg.norm(d, axis=2)
        u = d / np.maximum(n, 1e-9)[:, :, None]
        Y = Y - lr * ((n - R)[:, :, None] * u).mean(1)
    n = np.linalg.norm(Y[:, None, :] - A[None], axis=2)
    return Y, np.sqrt(((n - R) ** 2).mean(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/movie4/iqe_4pole_30k_latest.pt")
    ap.add_argument("--combined", default="data/derived/field_combined_sub600k.npz")
    ap.add_argument("--n", type=int, default=9000, help="positions per source in the viewer")
    ap.add_argument("--out", default="artifacts/experiments/basin_3d.html")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    t0 = time.time()

    p = torch.load(args.ckpt, map_location=args.device, weights_only=False); cfg = p["cfg"]
    net = IQEHead(in_ch=cfg["in_ch"], d=cfg["d"], components=cfg["components"],
                  adapter_ch=cfg["adapter_ch"]).to(args.device)
    net.load_compat(p["state_dict"]); net.eval()
    z = np.load(args.combined, allow_pickle=True)
    meta = eval(str(z["_meta"][0])); mm = np.load(meta["feats"][0], mmap_mode="r")
    split = z["orig_source"] if "orig_source" in z.files else z["source"]
    rng = np.random.default_rng(0)
    take = np.sort(np.concatenate([rng.choice(np.flatnonzero(split == s), args.n, replace=False)
                                   for s in (0, 1)]))
    with torch.no_grad():
        Dl = []
        for i in range(0, len(take), 8192):
            x = torch.from_numpy(np.asarray(mm[z["local_row"][take[i:i + 8192]]],
                                            dtype=np.float32)).to(args.device)
            e = net.phi(x)
            Dl.append(torch.cat([net.d_poles(e), net.d_from_start(e)[:, None]], 1).cpu().numpy())
    R = np.log1p(np.maximum(np.concatenate(Dl), 0))
    R = R / np.median(R[:, START])
    A = anchors3d(float(np.median(R[:, :3])))
    Y, err = trilaterate(R, A)
    print(f"3-D trilateration: median rms error {np.median(err):.3f}, p90 {np.percentile(err,90):.3f}"
          f"   (2-D was 0.358 / 0.505)  -> {100*(1-np.median(err)/0.358):.0f}% reduction")

    Yc = Y - Y.mean(0); s = np.percentile(np.linalg.norm(Yc, axis=1), 99)
    pts = [{"x": round(float(a), 3), "y": round(float(b), 3), "z": round(float(c), 3),
            "o": int(o), "s": int(sc)}
           for (a, b, c), o, sc in zip(Yc / s, z["y"][take], split[take])]
    anc = [{"x": float(v[0]/s), "y": float(v[1]/s), "z": float(v[2]/s),
            "n": PNAME[k], "c": PCOL[k]} for k, v in enumerate(A - Y.mean(0))]
    html = HTML.replace("__PTS__", json.dumps(pts)).replace("__ANC__", json.dumps(anc)) \
               .replace("__MED__", f"{np.median(err):.3f}")
    open(args.out, "w").write(html)
    print(f"wrote {args.out} ({len(html)/1e6:.1f} MB) [{time.time()-t0:.0f}s]")


HTML = r"""<!doctype html><html><head><meta charset=utf-8><title>basin field 3D</title>
<style>html,body{margin:0;background:#0b0b10;color:#e8e8ee;font:13px system-ui,sans-serif;overflow:hidden}
#c{display:block;cursor:grab}#c:active{cursor:grabbing}
#ui{position:fixed;top:10px;left:12px;background:#14141ccc;padding:10px 12px;border-radius:8px;line-height:1.7}
label{margin-right:10px}#hint{position:fixed;bottom:10px;left:12px;color:#8a8a98}</style></head><body>
<canvas id=c></canvas>
<div id=ui><b>basin field — 3D trilateration</b><br>
<label><input type=checkbox id=hu checked> human</label>
<label><input type=checkbox id=sf checked> SF-vs-SF</label><br>
<label><input type=checkbox id=win checked style="accent-color:#0ca30c"> wins</label>
<label><input type=checkbox id=drw checked style="accent-color:#8a8985"> draws</label>
<label><input type=checkbox id=los checked style="accent-color:#d03b3b"> losses</label><br>
<span style="color:#8a8a98">median fit error __MED__ &nbsp;·&nbsp; drag rotate · wheel zoom</span></div>
<div id=hint>each point placed only by its four pole distances</div>
<script>
const PTS=__PTS__, ANC=__ANC__;
const COL=["#0ca30c","#8a8985","#d03b3b"];
const c=document.getElementById('c'),g=c.getContext('2d');
let rx=-0.45,ry=0.6,zoom=1,drag=false,px=0,py=0;
function resize(){c.width=innerWidth;c.height=innerHeight}
addEventListener('resize',()=>{resize();draw()});resize();
c.addEventListener('mousedown',e=>{drag=true;px=e.clientX;py=e.clientY});
addEventListener('mouseup',()=>drag=false);
addEventListener('mousemove',e=>{if(!drag)return;ry+=(e.clientX-px)*0.008;rx+=(e.clientY-py)*0.008;px=e.clientX;py=e.clientY;draw()});
c.addEventListener('wheel',e=>{e.preventDefault();zoom*=Math.exp(-e.deltaY*0.0012);draw()},{passive:false});
function proj(p){const cx=Math.cos(rx),sx=Math.sin(rx),cy=Math.cos(ry),sy=Math.sin(ry);
 let x=p.x*cy-p.z*sy, z=p.x*sy+p.z*cy, y=p.y*cx-z*sx; z=p.y*sx+z*cx;
 const f=1/(1+z*0.35), S=Math.min(c.width,c.height)*0.36*zoom;
 return [c.width/2+x*S*f, c.height/2-y*S*f, z, f]}
function draw(){g.fillStyle='#0b0b10';g.fillRect(0,0,c.width,c.height);
 const on={0:hu.checked,1:sf.checked}, oc=[win.checked,drw.checked,los.checked];
 const vis=PTS.filter(p=>on[p.s]&&oc[p.o]).map(p=>{const q=proj(p);return{q,o:p.o}});
 vis.sort((a,b)=>b.q[2]-a.q[2]);
 for(const v of vis){g.globalAlpha=0.16+0.30*v.q[3];g.fillStyle=COL[v.o];
  g.fillRect(v.q[0],v.q[1],1.9,1.9)}
 g.globalAlpha=1;
 for(const a of ANC){const q=proj(a);g.beginPath();g.arc(q[0],q[1],7,0,7);g.fillStyle=a.c;g.fill();
  g.strokeStyle='#000';g.lineWidth=1.5;g.stroke();
  g.fillStyle='#e8e8ee';g.font='12px system-ui';g.fillText(a.n,q[0]+11,q[1]+4)}}
for(const id of ['hu','sf','win','drw','los'])document.getElementById(id).onchange=draw;
draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
