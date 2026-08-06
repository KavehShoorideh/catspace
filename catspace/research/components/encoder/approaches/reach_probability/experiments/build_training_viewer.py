#!/usr/bin/env python
"""build_training_viewer.py -- render the export_training_umap frames into a standalone 3D viewer.

Kaveh: "each checkpoint, reinsert into viz." The training-step slider scrubs through checkpoints in
ONE shared UMAP (see export_training_umap.py for why that co-fit is the only honest option), so a
cloud that moves between frames really moved. Press play to watch the field organise.

A script rather than a hand-written heredoc so the figure is reproducible from the repo.

    .venv/bin/python .../build_training_viewer.py --data <json> --out <html>
"""
from __future__ import annotations

import argparse
import json
import os

HTML = """<title>%(title)s</title>
<style>
:root{--bg:#f7f6f3;--fg:#1b1a18;--dim:#6f6b63;--line:#d8d4cc;--panel:#fffefb;--acc:#a8541f}
@media (prefers-color-scheme:dark){:root{--bg:#131311;--fg:#eceae4;--dim:#8e8a81;--line:#2f2e2a;--panel:#1c1b18;--acc:#e08a45}}
:root[data-theme=dark]{--bg:#131311;--fg:#eceae4;--dim:#8e8a81;--line:#2f2e2a;--panel:#1c1b18;--acc:#e08a45}
:root[data-theme=light]{--bg:#f7f6f3;--fg:#1b1a18;--dim:#6f6b63;--line:#d8d4cc;--panel:#fffefb;--acc:#a8541f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:18px 22px 12px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:19px;font-weight:640;letter-spacing:-.01em}
h1 small{display:block;font-weight:400;font-size:13px;color:var(--dim);letter-spacing:0;margin-top:3px}
main{display:grid;grid-template-columns:minmax(0,1fr) 250px;gap:0;height:calc(100vh - 78px);min-height:520px}
@media (max-width:820px){main{grid-template-columns:1fr;height:auto}#wrap{height:62vh}}
#wrap{position:relative;min-height:0}
canvas{display:block;width:100%%;height:100%%;cursor:grab;touch-action:none}
canvas:active{cursor:grabbing}
aside{border-left:1px solid var(--line);background:var(--panel);padding:16px 16px 30px;overflow-y:auto;
 display:flex;flex-direction:column;gap:16px}
.grp{display:flex;flex-direction:column;gap:7px}
.lab{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:620}
.val{font-variant-numeric:tabular-nums;color:var(--acc);font-weight:640}
input[type=range]{width:100%%;accent-color:var(--acc)}
button{font:inherit;font-size:12.5px;padding:5px 10px;border:1px solid var(--line);background:var(--bg);
 color:var(--fg);border-radius:5px;cursor:pointer}
button[aria-pressed=true]{background:var(--acc);border-color:var(--acc);color:#fff}
button:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.row{display:flex;gap:5px;flex-wrap:wrap}
.key{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--dim)}
.sw{width:11px;height:11px;border-radius:2px;flex:none}
.note{font-size:11.5px;line-height:1.45;color:var(--dim);border-top:1px solid var(--line);padding-top:12px}
</style>
<header>
<h1>Watching the field organise
<small>%(sub)s &mdash; one shared UMAP co-fit across every checkpoint, so motion between frames is real motion, not a re-fit artefact.</small></h1>
</header>
<main>
<div id="wrap"><canvas id="c"></canvas></div>
<aside>
 <div class="grp"><span class="lab">Training step <span class="val" id="stepv"></span></span>
  <input type="range" id="step" min="0" value="0">
  <div class="row"><button id="play">&#9654; Play</button><button id="poles" aria-pressed="true">Poles</button></div></div>
 <div class="grp"><span class="lab">Ply &le; <span class="val" id="plyv"></span></span>
  <input type="range" id="ply" min="1" value="210"></div>
 <div class="grp"><span class="lab">Colour by</span>
  <div class="row" id="cmode">
   <button data-m="arr" aria-pressed="true">Arrived</button><button data-m="ply">Ply</button>
   <button data-m="pc">Material</button><button data-m="src">Source</button></div>
  <div id="legend" class="grp" style="gap:4px"></div></div>
 <p class="note" id="drift"></p>
 <p class="note">Drag to rotate, scroll to zoom. Points are a fixed sample of held-out positions,
 identical in every frame &mdash; so nothing here moves because a different sample was drawn.</p>
</aside>
</main>
<script>
const D=%(data)s;
const C=document.getElementById('c'),X=C.getContext('2d');
let f=D.frames.length-1,maxPly=210,mode='arr',showP=true,rx=-0.45,ry=0.7,zoom=1,playing=false;
const OUT={'-1':['#3d7fc1','in progress'],'0':['#c0392b','arrived LOSS'],'1':['#8d8880','arrived DRAW'],'2':['#2e8b57','arrived WIN']};
function ramp(t,a,b){const p=[[247,251,255],[8,48,107]];return `hsl(${(1-t)*205+t*20},70%%,${58-t*18}%%)`}
function colOf(i){
 if(mode==='arr')return OUT[D.arr[i]][0];
 if(mode==='ply')return ramp(Math.min(D.ply[i],150)/150);
 if(mode==='pc')return ramp(1-(D.pc[i]-4)/28);
 return D.src[i]===0?'#7b5ea7':'#c98a2b';}
function legend(){const L=document.getElementById('legend');L.innerHTML='';
 const it=mode==='arr'?Object.values(OUT):mode==='src'?[['#7b5ea7','Stockfish'],['#c98a2b','human']]:
  mode==='ply'?[[ramp(0),'ply 0'],[ramp(1),'ply 150+']]:[[ramp(1),'4 pieces'],[ramp(0),'32 pieces']];
 for(const[c,t]of it){const d=document.createElement('div');d.className='key';
  d.innerHTML=`<span class="sw" style="background:${c}"></span>${t}`;L.append(d);}}
function proj(x,y,z){x-=.5;y-=.5;z-=.5;
 let a=x*Math.cos(ry)-z*Math.sin(ry),b=x*Math.sin(ry)+z*Math.cos(ry);
 let c=y*Math.cos(rx)-b*Math.sin(rx),d=y*Math.sin(rx)+b*Math.cos(rx);
 const p=1.9/(1.9+d);return[a*p,c*p,d];}
function draw(){const dpr=devicePixelRatio||1,W=C.clientWidth,H=C.clientHeight;
 C.width=W*dpr;C.height=H*dpr;X.setTransform(dpr,0,0,dpr,0,0);X.clearRect(0,0,W,H);
 const s=Math.min(W,H)*0.82*zoom,cx=W/2,cy=H/2,F=D.frames[f];
 const pts=[];
 for(let i=0;i<D.n;i++){if(D.ply[i]>maxPly)continue;
  const[a,b,d]=proj(F.x[i],F.y[i],F.z[i]);pts.push([a*s+cx,b*s+cy,d,colOf(i),0]);}
 if(showP&&D.pole_frames){const P=D.pole_frames[f];
  for(const p of P){const[a,b,d]=proj(p.x,p.y,p.z);pts.push([a*s+cx,b*s+cy,d,'#111',1,p.name]);}}
 pts.sort((u,v)=>v[2]-u[2]);
 for(const p of pts){if(p[4]){X.beginPath();X.arc(p[0],p[1],6,0,7);X.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--acc');X.fill();
   X.strokeStyle=getComputedStyle(document.body).color;X.lineWidth=1.4;X.stroke();
   X.fillStyle=getComputedStyle(document.body).color;X.font='600 11px ui-sans-serif';X.fillText(p[5],p[0]+9,p[1]+4);}
  else{X.globalAlpha=.62;X.fillStyle=p[3];X.beginPath();X.arc(p[0],p[1],2.1,0,7);X.fill();X.globalAlpha=1;}}}
const stepEl=document.getElementById('step'),plyEl=document.getElementById('ply');
stepEl.max=D.frames.length-1;stepEl.value=f;plyEl.max=Math.max(...D.ply);plyEl.value=plyEl.max;maxPly=+plyEl.max;
function sync(){document.getElementById('stepv').textContent=D.steps[f].toLocaleString();
 document.getElementById('plyv').textContent=maxPly;draw();}
stepEl.oninput=e=>{f=+e.target.value;sync()};
plyEl.oninput=e=>{maxPly=+e.target.value;sync()};
document.getElementById('cmode').onclick=e=>{const b=e.target.closest('button');if(!b)return;
 mode=b.dataset.m;[...e.currentTarget.children].forEach(x=>x.setAttribute('aria-pressed',x===b));legend();draw()};
document.getElementById('poles').onclick=e=>{showP=!showP;e.target.setAttribute('aria-pressed',showP);draw()};
const pb=document.getElementById('play');
pb.onclick=()=>{playing=!playing;pb.setAttribute('aria-pressed',playing);pb.innerHTML=playing?'&#9646;&#9646; Pause':'&#9654; Play';
 if(playing)tick()};
function tick(){if(!playing)return;f=(f+1)%%D.frames.length;stepEl.value=f;sync();setTimeout(tick,420)}
let drag=null;
C.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];C.setPointerCapture(e.pointerId)});
C.addEventListener('pointermove',e=>{if(!drag)return;ry+=(e.clientX-drag[0])*.008;rx+=(e.clientY-drag[1])*.008;
 rx=Math.max(-1.5,Math.min(1.5,rx));drag=[e.clientX,e.clientY];draw()});
addEventListener('pointerup',()=>drag=null);
C.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.4,Math.min(6,zoom*(e.deltaY<0?1.11:.9)));draw()},{passive:false});
addEventListener('resize',draw);
// pole drift: the honest headline of the animation
if(D.pole_frames){const A=D.pole_frames[0],B=D.pole_frames[D.pole_frames.length-1];
 let m=0;for(let i=0;i<A.length;i++)m=Math.max(m,Math.hypot(A[i].x-B[i].x,A[i].y-B[i].y,A[i].z-B[i].z));
 document.getElementById('drift').textContent=
  `Largest pole movement from first frame to last: ${m.toFixed(3)} (UMAP units, box normalised to 1). `+
  (m<0.02?'Effectively zero \\u2014 the learned poles never moved.':'The learned poles migrated during training.');}
legend();sync();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Watching the field organise")
    ap.add_argument("--sub", default="reach_probability checkpoint ladder")
    args = ap.parse_args()
    with open(args.data) as fh:
        data = fh.read()
    with open(args.out, "w") as fh:
        fh.write(HTML % {"data": data, "title": args.title, "sub": args.sub})
    print(f"[viewer] -> {args.out} ({os.path.getsize(args.out)/2**20:.1f} MB)")


if __name__ == "__main__":
    main()
