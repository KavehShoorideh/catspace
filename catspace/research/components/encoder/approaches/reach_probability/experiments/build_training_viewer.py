#!/usr/bin/env python
"""build_training_viewer.py -- render export_training_umap frames into the FAMILIAR ply viewer,
extended with a training-step slider.

Kaveh, on the first standalone version: "the viz changed a bit; i liked it before" -> merge into
the old viewer. So this is the ply_v2 template -- same rail, same colour modes, same follow-a-game
PGN, same keyboard map, same vector zoom -- with:

  * a TRAINING STEP slider ("watching the field organize"): scrubs checkpoints inside ONE shared
    UMAP co-fit, so motion between frames is real motion, not a re-fit artefact;
  * clickable LEGEND chips: toggle each category on/off per colour mode;
  * pan on shift-drag AND middle-drag (the standalone page had lost pan entirely);
  * the pole-label collision logic from the old viewer (the standalone page piled labels up).

    .venv/bin/python .../build_training_viewer.py --data <json> --out <html>
"""
from __future__ import annotations

import argparse
import os

HTML = r"""<title>__TITLE__</title>
<style>
:root{
  --ground:#f5f5f3; --panel:#ffffff; --edge:#e2e2df; --ink:#161b22; --muted:#6b7280;
  --accent:#d13c74; --ghost:rgba(20,25,32,.10); --grid:rgba(20,25,32,.06);
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#0e1216; --panel:#161c22; --edge:#242c35; --ink:#dde3ea; --muted:#7b8794;
  --accent:#e0568a; --ghost:rgba(221,227,234,.10); --grid:rgba(221,227,234,.06);
}}
:root[data-theme="dark"]{
  --ground:#0e1216; --panel:#161c22; --edge:#242c35; --ink:#dde3ea; --muted:#7b8794;
  --accent:#e0568a; --ghost:rgba(221,227,234,.10); --grid:rgba(221,227,234,.06);
}
:root[data-theme="light"]{
  --ground:#f5f5f3; --panel:#ffffff; --edge:#e2e2df; --ink:#161b22; --muted:#6b7280;
  --accent:#d13c74; --ghost:rgba(20,25,32,.10); --grid:rgba(20,25,32,.06);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1240px;margin:0 auto;padding:28px 22px 44px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:14px;
  border-bottom:1px solid var(--edge);padding-bottom:14px;margin-bottom:20px}
h1{font-size:19px;font-weight:620;margin:0;letter-spacing:-.01em}
.sub{font-family:var(--mono);font-size:11.5px;color:var(--muted);letter-spacing:.02em}
.tag{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;
  color:var(--accent);border:1px solid var(--accent);border-radius:2px;padding:2px 7px}
.console{display:grid;grid-template-columns:216px 1fr;gap:20px}
@media(max-width:860px){.console{grid-template-columns:1fr}}
.rail{display:flex;flex-direction:column;gap:18px}
.grp{display:flex;flex-direction:column;gap:7px}
.lbl{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted)}
.seg{display:flex;flex-direction:column;border:1px solid var(--edge);border-radius:3px;overflow:hidden}
.seg button{appearance:none;background:var(--panel);border:0;border-bottom:1px solid var(--edge);
  color:var(--ink);font-family:var(--mono);font-size:12px;padding:7px 10px;text-align:left;
  cursor:pointer}
.seg button:last-child{border-bottom:0}
.seg button[aria-pressed="true"]{background:var(--accent);color:#fff}
.seg button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.stats{border:1px solid var(--edge);border-radius:3px;background:var(--panel)}
.row{display:flex;justify-content:space-between;gap:10px;padding:6px 10px;
  border-bottom:1px solid var(--edge);font-family:var(--mono);font-size:12px;
  font-variant-numeric:tabular-nums}
.row:last-child{border-bottom:0}
.row span:first-child{color:var(--muted)}
.stage{display:flex;flex-direction:column;gap:12px;min-width:0}
.canvasbox{position:relative;border:1px solid var(--edge);border-radius:3px;background:var(--panel);
  aspect-ratio:1.42/1}
canvas{width:100%;height:100%;display:block;border-radius:3px}
.plyread{position:absolute;top:12px;left:14px;font-family:var(--mono);pointer-events:none}
.plynum{font-size:38px;font-weight:600;letter-spacing:-.03em;line-height:1;
  font-variant-numeric:tabular-nums}
.plyword{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);
  margin-top:3px}
.stepread{position:absolute;top:12px;right:14px;font-family:var(--mono);pointer-events:none;
  text-align:right}
.stepnum{font-size:22px;font-weight:600;letter-spacing:-.02em;line-height:1;
  font-variant-numeric:tabular-nums;color:var(--accent)}
.transport{display:flex;align-items:center;gap:14px}
button.play{appearance:none;background:var(--accent);border:0;color:#fff;font-family:var(--mono);
  font-size:12px;padding:8px 15px;border-radius:3px;cursor:pointer;min-width:74px}
button.play:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
button.play.alt{background:var(--panel);color:var(--accent);border:1px solid var(--accent)}
button.step{appearance:none;background:var(--panel);border:1px solid var(--edge);color:var(--ink);
  font-family:var(--mono);font-size:14px;line-height:1;padding:4px 9px;border-radius:3px;
  cursor:pointer}
button.step:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.sliderbox{flex:1;display:flex;flex-direction:column;gap:3px;min-width:0}
input[type=range]{width:100%;accent-color:var(--accent);margin:0}
.hist{width:100%;height:26px;display:block}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-family:var(--mono);font-size:11px;
  color:var(--muted);align-items:center}
.legend .chip{cursor:pointer;user-select:none;padding:2px 6px;border-radius:3px;
  border:1px solid transparent}
.legend .chip:hover{border-color:var(--edge)}
.legend .chip.off{opacity:.32;text-decoration:line-through}
.legend .fixed{opacity:.75}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;
  vertical-align:-1px}
.ramp{width:96px;height:9px;border-radius:2px;display:inline-block;vertical-align:-1px;
  margin:0 6px}
.pgn{border:1px solid var(--edge);border-radius:3px;background:var(--panel);padding:9px 11px;
  font-family:var(--mono);font-size:12px;line-height:1.85;max-height:150px;overflow-y:auto}
.pgn .mv{padding:1px 4px;border-radius:2px;cursor:pointer}
.pgn .mv:hover{background:var(--ghost)}
.pgn .mv.on{background:var(--accent);color:#fff;font-weight:600}
.pgn .mv.start{font-style:italic;color:var(--muted);border:1px solid var(--edge);margin-right:6px}
.pgn .mv.start.on{color:#fff;border-color:transparent}
.pgn .no{color:var(--muted);margin-left:5px}
.pgn .no:first-child{margin-left:0}
.note{border-left:2px solid var(--accent);padding:9px 0 9px 13px;color:var(--muted);
  font-size:12.5px;max-width:66ch}
.note b{color:var(--ink);font-weight:600}
</style>

<div class="wrap">
<header>
  <h1>Watching the field organise</h1>
  <span class="tag">__TAG__</span>
  <span class="sub" id="hdr"></span>
</header>

<div class="console">
  <div class="rail">
    <div class="grp">
      <div class="lbl">Colour by</div>
      <div class="seg" id="colorby">
        <button data-k="arr" aria-pressed="true">arrived at W/D/L</button>
        <button data-k="pc" aria-pressed="false">piece count</button>
        <button data-k="out" aria-pressed="false">eventual outcome</button>
        <button data-k="ph" aria-pressed="false">game phase</button>
        <button data-k="cas" aria-pressed="false">castling rights</button>
        <button data-k="endt" aria-pressed="false">ending type</button>
        <button data-k="src" aria-pressed="false">population</button>
      </div>
    </div>
    <div class="grp">
      <div class="lbl">Cloud sampling</div>
      <div class="seg" id="cohby">
        <button data-c="0" aria-pressed="true">upsample sparse plies</button>
        <button data-c="1" aria-pressed="false">true cohort</button>
      </div>
    </div>
    <div class="grp">
      <div class="lbl">Full cloud</div>
      <div class="seg" id="ghostby">
        <button data-g="1" aria-pressed="true">ghost behind</button>
        <button data-g="0" aria-pressed="false">hide</button>
      </div>
    </div>
    <div class="grp">
      <div class="lbl">What to show</div>
      <div class="seg" id="traceby">
        <button data-t="0" aria-pressed="true">whole sample at this ply</button>
        <button data-t="1" aria-pressed="false">follow one game</button>
      </div>
      <div id="gamepick" style="display:none;flex-direction:column;gap:6px;margin-top:8px">
        <div style="display:flex;gap:6px;align-items:center">
          <button id="gprev" class="step">&#8249;</button>
          <input type="range" id="game" min="0" max="399" value="0" step="1"
                 aria-label="game" style="flex:1">
          <button id="gnext" class="step">&#8250;</button>
        </div>
        <div class="stats">
          <div class="row"><span>game</span><span id="g-idx">&mdash;</span></div>
          <div class="row"><span>population</span><span id="g-pop">&mdash;</span></div>
          <div class="row"><span>plies</span><span id="g-len">&mdash;</span></div>
          <div class="row"><span>ended in</span><span id="g-end">&mdash;</span></div>
          <div class="row"><span>at this ply</span><span id="g-at">&mdash;</span></div>
        </div>
      </div>
    </div>
    <div class="grp">
      <div class="lbl">Point size</div>
      <input type="range" id="ptsize" min="0.7" max="2.5" step="0.1" value="1" aria-label="point size">
    </div>
    <div class="grp">
      <div class="lbl">View</div>
      <div class="seg"><button id="spin">&#9711; auto-rotate</button><button id="reset">reset view</button></div>
      <div class="sub" style="font-size:10.5px">drag rotate (unclamped) &middot; alt-drag roll axis &middot; shift/middle-drag pan &middot; scroll zoom</div>
      <div class="sub" style="font-size:10.5px">&larr; &rarr; ply &middot; &uarr; &darr; ends &middot; space play &middot; [ ] game &middot; , . step</div>
    </div>
    <div class="grp">
      <div class="lbl">This cross-section</div>
      <div class="stats">
        <div class="row"><span>positions</span><span id="s-n">&mdash;</span></div>
        <div class="row"><span>ended here</span><span id="s-term">&mdash;</span></div>
        <div class="row"><span>mean pieces</span><span id="s-pc">&mdash;</span></div>
        <div class="row"><span>open/mid/end</span><span id="s-ph">&mdash;</span></div>
        <div class="row"><span>win / draw / loss</span><span id="s-wdl">&mdash;</span></div>
        <div class="row"><span>human / engine</span><span id="s-src">&mdash;</span></div>
      </div>
    </div>
  </div>

  <div class="stage">
    <div class="canvasbox">
      <canvas id="cv"></canvas>
      <div class="plyread"><div class="plynum" id="plynum">0</div><div class="plyword">ply</div></div>
      <div class="stepread"><div class="stepnum" id="stepnum"></div><div class="plyword">training step</div></div>
    </div>
    <div class="transport">
      <button class="play" id="play">&#9654; play</button>
      <div class="sliderbox">
        <input type="range" id="ply" min="0" max="120" value="0" step="1" aria-label="ply">
        <canvas class="hist" id="hist"></canvas>
      </div>
    </div>
    <div class="transport">
      <button class="play alt" id="playstep">&#9654; train</button>
      <div class="sliderbox">
        <input type="range" id="stepsl" min="0" max="0" value="0" step="1" aria-label="training step">
      </div>
      <button id="sprev" class="step">&#8249;</button>
      <button id="snext" class="step">&#8250;</button>
    </div>
    <div class="legend" id="legend"></div>
    <div class="pgn" id="pgn" style="display:none"></div>
    <p class="note"><b>The training-step slider scrubs checkpoints.</b> Every frame embeds the
      SAME positions under a different checkpoint of the model, and all frames are co-fitted into
      ONE UMAP &mdash; one coordinate system, so a cloud that moves between steps really moved.
      Press &#9654; train to watch the field organise.</p>
    <p class="note"><span id="sepnote"></span></p>
    <p class="note"><b>Cloud sampling:</b> <i>upsample sparse plies</i> draws a per-ply balanced
      sample &mdash; every cross-section equally readable, but dot DENSITY is engineered
      (terminals injected, late plies over-represented). <i>True cohort</i> follows a fixed set of
      games end-to-end: density is real, terminals appear at their natural rate, and thinning at
      late plies is genuine attrition &mdash; the histogram under the slider becomes the survival
      curve.</p>
    <p class="note"><b>Legend chips are click-to-toggle</b> &mdash; hide categories to isolate the
      ones you care about. Poles: gold = fixed W/D/L simplex (the gauge), pink diamond with a halo = START; the small grey markers are
      the learned ending types &mdash; unlabelled, since they cluster at their outcome pole and
      one set of names is enough.</p>
  </div>
</div>
</div>

<script id="payload" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const hist = document.getElementById('hist'), hctx = hist.getContext('2d');
const slider = document.getElementById('ply');
const stepsl = document.getElementById('stepsl');
let mode = 'arr', ghost = true, playing = false, raf = null, traceMode = 0;
let fi = D.frames.length - 1;                       // current checkpoint frame (start at latest)
stepsl.max = D.frames.length - 1; stepsl.value = fi;
let view = {k:1, tx:0, ty:0}, drag=null;
let rot = {ax:-0.45, ay:0.6, az:0}, spin=false, spinRaf=null;
const is3d = () => D.dims===3 && D.frames[fi].z;
function rot3(x,y,z){
  const cx=Math.cos(rot.ax), sx=Math.sin(rot.ax), cy=Math.cos(rot.ay), sy=Math.sin(rot.ay);
  x-=.5; y-=.5; z-=.5;
  let X= x*cy + z*sy, Z= -x*sy + z*cy;
  let Y= y*cx - Z*sx;  Z = y*sx + Z*cx;
  if(rot.az){ const cz=Math.cos(rot.az), sz=Math.sin(rot.az);
    const Xr = X*cz - Y*sz, Yr = X*sz + Y*cz; X=Xr; Y=Yr; }
  return [X+.5, Y+.5, Z+.5];
}
const VIRIDIS = [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
                 [31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]];
function viridis(t){t=Math.max(0,Math.min(1,t));const x=t*(VIRIDIS.length-1),i=Math.floor(x),f=x-i;
  const a=VIRIDIS[i],b=VIRIDIS[Math.min(i+1,VIRIDIS.length-1)];
  return `rgb(${a[0]+(b[0]-a[0])*f|0},${a[1]+(b[1]-a[1])*f|0},${a[2]+(b[2]-a[2])*f|0})`;}
const OUT = ['#3fa66b','#8a93a0','#c94f4f'], OUTN = ['win','draw','loss'];
const ARR = ['#35a76a','#98a1ad','#d0483f'], INPROG = 'rgba(74,124,196,.34)';
const SRC = ['#4f8fd1','#d98a3a'], SRCN = ['human','engine'];
const TERMN = ['mate','resign (loss)','resign (win)','draw agreed','draw adjudicated',
  'draw 50-move','stalemate','insufficient material','threefold'];
const PHC = ['#4f8fd1','#c8913a','#a05fc0'], PHN = ['opening','middlegame','endgame'];
// ending-type palette: reds = losses, greens = wins, cool/neutral = draws, dashed grey = censored
const ENDC = ['#8b1e3f','#d0483f','#2e8b57','#708090','#9aa3ad','#b8860b','#7a5ea8','#4a7fc1','#3fa07f'];
const ENDCEN = '#5b6270';
function endCat(i){ const e=D.endt[i]; return e>=0 ? e : (e===-2 ? 9 : 10); }
const CASC = ['#5b6270','#7a5ea8','#4a7fc1','#3fa07f','#d9a13a'];
function nbits(v){let n=0;while(v){n+=v&1;v>>=1;}return n;}
const pcLo = Math.min(...D.pc), pcHi = Math.max(...D.pc);

// LEGEND TOGGLES (Kaveh: "i wanna be able to toggle each item in the legend as well").
// A hidden category set PER COLOUR MODE; catOf(i) maps a point to its chip in the current mode.
const hidden = {arr:new Set(), out:new Set(), ph:new Set(), cas:new Set(), src:new Set(), pc:new Set(), endt:new Set()};
function catOf(i){
  if(mode==='arr') return D.arr[i]<0 ? 3 : D.arr[i];      // 0 W 1 D 2 L 3 in-progress
  if(mode==='ph')  return D.ph[i];
  if(mode==='cas') return nbits(D.cas[i]);
  if(mode==='out') return D.out[i]<0 ? 3 : D.out[i];      // 3 = censored
  if(mode==='endt') return endCat(i);
  if(mode==='src') return D.src[i];
  return 0;                                               // pc: continuous ramp, no toggles
}
function vis(i){ return !hidden[mode].has(catOf(i)); }
function colOf(i){
  if(mode==='arr') return D.arr[i]<0 ? INPROG : ARR[D.arr[i]];
  if(mode==='ph') return PHC[D.ph[i]];
  if(mode==='cas') return CASC[nbits(D.cas[i])];
  if(mode==='pc') return viridis((D.pc[i]-pcLo)/Math.max(pcHi-pcLo,1));
  if(mode==='out') return D.out[i]<0 ? 'rgba(140,150,160,.5)' : OUT[D.out[i]];
  if(mode==='endt'){ const c=endCat(i); return c<9 ? ENDC[c] : (c===9 ? ENDCEN : 'rgba(140,150,160,.4)'); }
  return SRC[D.src[i]];
}
// TWO cloud samplings: coh=0 per-ply BALANCED (densities engineered), coh=1 fixed COHORT of
// games end-to-end (densities real; late-ply thinning is genuine attrition).
let cohMode = 0, ptScale = 1;
const byPlyBal = new Map(), byPlyCoh = new Map();
D.ply.forEach((p,i)=>{ const m = (D.coh && D.coh[i]) ? byPlyCoh : byPlyBal;
  if(!m.has(p)) m.set(p,[]); m.get(p).push(i); });
const byPlyAt = () => cohMode ? byPlyCoh : byPlyBal;
const plies = [...new Set([...byPlyBal.keys(), ...byPlyCoh.keys()])].sort((a,b)=>a-b);
slider.min = plies[0]; slider.max = plies[plies.length-1]; slider.value = plies[0];
function sepnote(){ if(!D.pole_sep) return;
  document.getElementById('sepnote').innerHTML =
    `In the EMBEDDING space (not the projection) the median pole&ndash;pole distance is `+
    `<b>${D.pole_sep[fi]}&times;</b> the median point&ndash;point distance at this checkpoint `+
    `&mdash; if that number is small, the crowding is real geometry, not a UMAP artefact.`+
    (D.term_gap ? ` Median d(terminal&rarr;its pole) = <b>${D.term_gap[fi]}</b> `+
    `(trained toward 1; falling across frames = the endings are converging onto their poles).` : '')+
    (D.start_gap ? ` START vs the ply-0 point: d(START&rarr;start pos) = `+
    `<b>${D.start_gap[fi][0]}</b> (trained toward 0 = domination, NOT identity &mdash; the pole `+
    `sits behind the start position, so map separation is expected), reverse = `+
    `<b>${D.start_gap[fi][1]}</b>.` : ''); }
function hdr(){ sepnote(); document.getElementById('hdr').textContent =
  `${D.n.toLocaleString()} positions · ${plies.length} plies · ${D.frames.length} checkpoints · one shared UMAP`;
  document.getElementById('stepnum').textContent = D.steps[fi].toLocaleString(); }

function fit(){ const r=cv.getBoundingClientRect(), d=window.devicePixelRatio||1;
  cv.width=r.width*d; cv.height=r.height*d; ctx.setTransform(d,0,0,d,0,0);
  const h=hist.getBoundingClientRect(); hist.width=h.width*d; hist.height=h.height*d;
  hctx.setTransform(d,0,0,d,0,0); draw(); drawHist(); }
function base(x,y,w,h){ return [26+x*(w-52), h-26-y*(h-52)]; }
function proj(i){ const F=D.frames[fi];
  if(!is3d()) return [F.x[i],F.y[i],0.5];
  return rot3(F.x[i],F.y[i],F.z[i]); }
function tx(x,y,w,h){ const [a,b]=base(x,y,w,h);
  return [a*view.k+view.tx, b*view.k+view.ty]; }
function px(i,w,h){ const [a,b]=proj(i); return tx(a,b,w,h); }
function depth(i){ return is3d() ? proj(i)[2] : 0.5; }

function draw(){
  const r=cv.getBoundingClientRect(), w=r.width, h=r.height;
  ctx.clearRect(0,0,w,h);
  ctx.save(); ctx.beginPath(); ctx.rect(0,0,w,h); ctx.clip();
  ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--grid');
  ctx.lineWidth=1;
  for(let k=1;k<4;k++){const gx=26+(w-52)*k/4, gy=26+(h-52)*k/4;
    ctx.beginPath();ctx.moveTo(gx,20);ctx.lineTo(gx,h-20);ctx.stroke();
    ctx.beginPath();ctx.moveTo(20,gy);ctx.lineTo(w-20,gy);ctx.stroke();}
  if(ghost){ ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--ghost');
    for(let i=0;i<D.n;i+=3){ if(D.coh && (D.coh[i]===1)!==(cohMode===1)) continue;
      const[a,b]=px(i,w,h);ctx.fillRect(a,b,1.6,1.6);} }
  const following = traceMode===1 && D.traces;
  if(following){
    const gi=Math.min(+gameSel.value,D.traces.length-1);
    const t=D.traces[gi], tc=D.trace_frames[fi][gi], q = +slider.value - t.p0;
    if(q>=0 && q<tc.x.length){
      const pt = is3d()&&tc.z ? rot3(tc.x[q],tc.y[q],tc.z[q]) : [tc.x[q],tc.y[q]];
      const[a,b]=tx(pt[0],pt[1],w,h), col = t.pop===0?'#4f8fd1':'#d98a3a';
      ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.globalAlpha=.5;
      ctx.beginPath();ctx.arc(a,b,12,0,6.2832);ctx.stroke(); ctx.globalAlpha=1;
      ctx.fillStyle=col; ctx.beginPath();ctx.arc(a,b,5.5,0,6.2832);ctx.fill();
      ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--panel');
      ctx.lineWidth=1.6; ctx.stroke();
    }
  }
  if(D.pole_frames){
    // Label-collision logic from the old viewer: draw a label only if it clears the ones already
    // placed. The marker always draws; zoom in and hidden labels reappear.
    const placed=[];
    // START first: its label is placed before anything else so it can never lose the collision
    // contest (Kaveh: "its hard to see the start pole").
    const rank=n=>(n==='START'?0:(['WIN','DRAW','LOSS'].indexOf(n)>=0?1:2));
    for(const P of [...D.pole_frames[fi]].sort((u,v)=>rank(u.name)-rank(v.name))){
      const pp = is3d()&&P.z!=null ? rot3(P.x,P.y,P.z) : [P.x,P.y];
      const[a,b]=tx(pp[0],pp[1],w,h);
      const outc = ['WIN','DRAW','LOSS'].indexOf(P.name)>=0, st = P.name==='START';
      if(st){
        // START: a small diamond -- distinct shape, no halo (Kaveh: "start is highlighted but
        // it doesn't need to be")
        ctx.fillStyle='#e0568a';
        ctx.beginPath();ctx.moveTo(a,b-6);ctx.lineTo(a+6,b);ctx.lineTo(a,b+6);ctx.lineTo(a-6,b);
        ctx.closePath();ctx.fill();
      } else {
        ctx.fillStyle = outc ? '#d9a13a' : '#9aa3ad';
        ctx.beginPath();ctx.arc(a,b,outc?8:5,0,6.2832);ctx.fill();
      }
      ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--panel');
      ctx.lineWidth=2; ctx.stroke();
      // ONE SET OF NAMES ONLY (Kaveh): the ending-type poles sit right at their outcome pole,
      // so labelling both sets stacks text on itself and says the same thing twice. Only the
      // W/D/L simplex and START get labels; ending poles stay as small unlabelled markers.
      if(!outc && !st) continue;
      // The four labels (W/D/L/START) ALWAYS draw -- dropping a clashing label hid WIN entirely
      // (Kaveh: "i see start and loss but no win"). Collisions are resolved by trying offset
      // positions around the marker instead; the last candidate is used even if it clashes.
      ctx.font='600 10px ui-monospace,monospace';
      const tw=ctx.measureText(P.name).width;
      let lx=a+10, ly=b+3;
      for(const[dx,dy] of [[10,3],[10,16],[10,-10],[-tw-12,3],[10,29],[-tw-12,16]]){
        const cx0=a+dx, cy0=b+dy;
        lx=cx0; ly=cy0;
        if(!placed.some(q=>Math.abs(q.x-cx0)<(q.w+tw)/2+6 && Math.abs(q.y-cy0)<12)) break;
      }
      placed.push({x:lx,y:ly,w:tw});
      ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--panel');
      ctx.fillRect(lx-2, ly-9, tw+4, 12);
      ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--ink');
      ctx.fillText(P.name, lx, ly);
    }
  }
  const cur0 = following ? [] : (byPlyAt().get(+slider.value)||[]);
  const cur = cur0.filter(vis);
  const term = [], prog = [];
  for(const i of cur){ (mode==='arr' && D.arr[i]>=0 ? term : prog).push(i); }
  const order = is3d() ? prog.slice().sort((p,q)=>depth(p)-depth(q)) : prog;
  for(const i of order){ const[a,b]=px(i,w,h); const dz=depth(i);
    ctx.globalAlpha = is3d() ? 0.28+0.72*dz : 1;
    ctx.fillStyle=colOf(i);
    ctx.beginPath();ctx.arc(a,b,(is3d()?(1.7+2.0*dz):2.6)*ptScale,0,6.2832);ctx.fill(); }
  ctx.globalAlpha=1;
  const torder = is3d() ? term.slice().sort((p,q)=>depth(p)-depth(q)) : term;
  for(const i of torder){ const[a,b]=px(i,w,h); const dz=depth(i);
    ctx.globalAlpha = is3d() ? 0.45+0.55*dz : 1; ctx.fillStyle=colOf(i);
    ctx.beginPath();ctx.arc(a,b,(is3d()?(2.6+2.4*dz):3.7)*ptScale,0,6.2832);ctx.fill(); }
  ctx.globalAlpha=1;
  ctx.restore();
}
function drawHist(){
  const r=hist.getBoundingClientRect(), w=r.width, h=r.height;
  hctx.clearRect(0,0,w,h);
  const M=byPlyAt();
  const mx=Math.max(...plies.map(p=>(M.get(p)||[]).length));
  const cs=getComputedStyle(document.documentElement);
  plies.forEach(p=>{ const x=(p-plies[0])/(plies[plies.length-1]-plies[0])*(w-2);
    const bh=((M.get(p)||[]).length/mx)*(h-4);
    hctx.fillStyle = p===+slider.value ? cs.getPropertyValue('--accent') : cs.getPropertyValue('--ghost');
    hctx.fillRect(x, h-bh, Math.max(w/plies.length-0.5,1.2), bh); });
}
function stats(){
  const cur = traceMode===1 ? [] : (byPlyAt().get(+slider.value)||[]);
  document.getElementById('s-n').textContent = cur.length.toLocaleString();
  if(!cur.length){['s-pc','s-wdl','s-src','s-term','s-ph'].forEach(k=>document.getElementById(k).textContent='—');return;}
  const nt = cur.filter(i=>D.arr[i]>=0).length;
  document.getElementById('s-term').textContent = nt ? nt.toLocaleString() : '0';
  const mp = cur.reduce((s,i)=>s+D.pc[i],0)/cur.length;
  document.getElementById('s-pc').textContent = mp.toFixed(1);
  const c=[0,0,0]; let cen=0; cur.forEach(i=>{ D.out[i]<0?cen++:c[D.out[i]]++; });
  const tot=c[0]+c[1]+c[2]||1;
  document.getElementById('s-wdl').textContent =
    `${(100*c[0]/tot).toFixed(0)}/${(100*c[1]/tot).toFixed(0)}/${(100*c[2]/tot).toFixed(0)}%`;
  const ph=[0,0,0]; cur.forEach(i=>ph[D.ph[i]]++);
  document.getElementById('s-ph').textContent =
    `${(100*ph[0]/cur.length).toFixed(0)}/${(100*ph[1]/cur.length).toFixed(0)}/${(100*ph[2]/cur.length).toFixed(0)}%`;
  const hm=cur.filter(i=>D.src[i]===0).length;
  document.getElementById('s-src').textContent =
    `${(100*hm/cur.length).toFixed(0)}/${(100*(cur.length-hm)/cur.length).toFixed(0)}%`;
}
// Legend chips: data-cat carries the category index; click toggles it in hidden[mode].
function chip(cat,col,txt){ const off=hidden[mode].has(cat)?' off':'';
  return `<span class="chip${off}" data-cat="${cat}"><span class="sw" style="background:${col}"></span>${txt}</span>`; }
function legend(){
  const el=document.getElementById('legend');
  if(mode==='arr'){ el.innerHTML=
      chip(0,ARR[0],'arrived: win')+chip(1,ARR[1],'arrived: draw')+chip(2,ARR[2],'arrived: loss')+
      chip(3,INPROG,'game still in progress')+
      (cohMode? `<span class="fixed">cohort: terminal density is REAL</span>` : `<span class="fixed">terminals are one row per game &mdash; deliberately oversampled</span>`);
    return; }
  if(mode==='ph'){ el.innerHTML=PHN.map((n,k)=>chip(k,PHC[k],n)).join('')
      +`<span class="fixed">endgame = non-pawn material &le;10 &middot; ply-balanced, not corpus-representative</span>`;
    return; }
  if(mode==='cas'){ el.innerHTML=[0,1,2,3,4].map(n=>
      chip(n,CASC[n],`${n} right${n===1?'':'s'} left`)).join('')
      +`<span class="fixed">rights are only ever LOST &mdash; irreversible, like material</span>`;
    return; }
  if(mode==='endt'){ el.innerHTML=TERMN.map((n,k)=>chip(k,ENDC[k],n)).join('')
      +chip(9,ENDCEN,'time forfeit (censored)')
      +`<span class="fixed">every ply coloured by how its GAME ends &mdash; click chips to filter</span>`;
    return; }
  if(mode==='pc'){ const g=`linear-gradient(90deg,${VIRIDIS.map(c=>`rgb(${c})`).join(',')})`;
    el.innerHTML=`<span>${pcLo} pieces<span class="ramp" style="background:${g}"></span>${pcHi}</span>`; }
  else if(mode==='out'){ el.innerHTML=OUTN.map((n,i)=>chip(i,OUT[i],n)).join('')
    +chip(3,'rgba(140,150,160,.5)','censored (flagged)'); }
  else { el.innerHTML=SRCN.map((n,i)=>chip(i,SRC[i],n)).join(''); }
}
document.getElementById('legend').addEventListener('click',e=>{
  const c=e.target.closest('.chip'); if(!c)return;
  const cat=+c.dataset.cat;
  if(hidden[mode].has(cat)) hidden[mode].delete(cat); else hidden[mode].add(cat);
  legend(); draw(); });
function update(){ document.getElementById('plynum').textContent=slider.value;
  draw(); drawHist(); stats(); }

slider.addEventListener('input',()=>{stop();update();gameInfo();});
stepsl.addEventListener('input',()=>{stopStep(); fi=+stepsl.value; hdr(); draw();});
document.getElementById('sprev').addEventListener('click',()=>{
  stopStep(); fi=Math.max(0,fi-1); stepsl.value=fi; hdr(); draw();});
document.getElementById('snext').addEventListener('click',()=>{
  stopStep(); fi=Math.min(D.frames.length-1,fi+1); stepsl.value=fi; hdr(); draw();});
document.getElementById('colorby').addEventListener('click',e=>{
  const b=e.target.closest('button'); if(!b)return; mode=b.dataset.k;
  [...e.currentTarget.children].forEach(x=>x.setAttribute('aria-pressed',x===b));
  legend(); draw(); });
const gameSel=document.getElementById('game');
const gamePick=document.getElementById('gamepick');
if(D.traces){ gameSel.max = D.traces.length-1; }
function renderPgn(t){
  const el=document.getElementById('pgn');
  if(traceMode!==1 || !t || !t.san || !t.san.length){ el.style.display='none'; return; }
  el.style.display='block';
  const cur = +slider.value;
  let html=`<span class="mv start${cur===0?' on':''}" data-p="0">start</span> `;
  for(let k=0;k<t.san.length;k++){
    if(k%2===0) html+=`<span class="no">${k/2+1}.</span>`;
    html+=`<span class="mv${k===cur-1?' on':''}" data-p="${k+1}">${t.san[k]}</span> `;
  }
  el.innerHTML=html;
  const on=el.querySelector('.mv.on');
  if(on) on.scrollIntoView({block:'nearest', behavior:'auto'});
}
function gameInfo(){
  if(!D.traces) return; const t=D.traces[Math.min(+gameSel.value,D.traces.length-1)];
  renderPgn(t);
  const tc=D.trace_frames[fi][Math.min(+gameSel.value,D.traces.length-1)];
  document.getElementById('g-idx').textContent = `${+gameSel.value+1} / ${D.traces.length}`;
  document.getElementById('g-pop').textContent = t.pop===0?'human':'engine';
  document.getElementById('g-len').textContent = `${tc.x.length}`;
  document.getElementById('g-end').textContent = t.end>=0 ? TERMN[t.end]
    : (t.end===-2?'time forfeit (censored)':'not recorded');
  const q = +slider.value - t.p0;
  document.getElementById('g-at').textContent =
    q < 0 ? 'not started' : (q >= tc.x.length ? 'already ended' : `ply ${+slider.value}`);
}
document.getElementById('traceby').addEventListener('click',e=>{
  const b=e.target.closest('button'); if(!b)return; traceMode=+b.dataset.t;
  [...e.currentTarget.children].forEach(x=>x.setAttribute('aria-pressed',x===b));
  gamePick.style.display = traceMode===1?'flex':'none';
  gameInfo(); draw(); });
gameSel.addEventListener('input',()=>{gameInfo();draw();});
document.getElementById('pgn').addEventListener('click',e=>{
  const m=e.target.closest('.mv'); if(!m)return;
  stop(); slider.value=m.dataset.p; update(); gameInfo(); });
document.getElementById('gprev').addEventListener('click',()=>{
  gameSel.value=Math.max(0,+gameSel.value-1); gameInfo(); draw(); });
document.getElementById('gnext').addEventListener('click',()=>{
  gameSel.value=Math.min(+gameSel.max,+gameSel.value+1); gameInfo(); draw(); });
function setPly(v){ stop(); slider.value=Math.min(Math.max(v,+slider.min),+slider.max);
  update(); gameInfo(); }
function setStep(v){ stopStep(); fi=Math.min(Math.max(v,0),D.frames.length-1);
  stepsl.value=fi; hdr(); draw(); }
window.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'&&e.target.type==='range'&&![',','.','[',']'].includes(e.key)) return;
  const t = (traceMode===1&&D.traces) ? D.traces[Math.min(+gameSel.value,D.traces.length-1)] : null;
  const tc = t ? D.trace_frames[fi][Math.min(+gameSel.value,D.traces.length-1)] : null;
  switch(e.key){
    case 'ArrowRight': e.preventDefault(); setPly(+slider.value+1); break;
    case 'ArrowLeft':  e.preventDefault(); setPly(+slider.value-1); break;
    case 'ArrowUp':    e.preventDefault(); setPly(t ? t.p0 : +slider.min); break;
    case 'ArrowDown':  e.preventDefault(); setPly(t ? t.p0+tc.x.length-1 : +slider.max); break;
    case ' ':          e.preventDefault(); playBtn.click(); break;
    case ',':          e.preventDefault(); setStep(fi-1); break;
    case '.':          e.preventDefault(); setStep(fi+1); break;
    case '[': if(traceMode===1){gameSel.value=Math.max(0,+gameSel.value-1);gameInfo();draw();} break;
    case ']': if(traceMode===1){gameSel.value=Math.min(+gameSel.max,+gameSel.value+1);gameInfo();draw();} break;
  }});
document.getElementById('reset').addEventListener('click',()=>{
  view={k:1,tx:0,ty:0}; rot={ax:-0.45,ay:0.6,az:0}; draw();});
document.getElementById('spin').addEventListener('click',()=>{
  spin=!spin; if(spin){ const tick=()=>{ if(!spin)return; rot.ay+=0.006; draw();
    spinRaf=requestAnimationFrame(tick); }; tick(); } else if(spinRaf) cancelAnimationFrame(spinRaf); });
cv.addEventListener('wheel',e=>{ e.preventDefault();
  const r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  // DEEP ZOOM (Kaveh: the old cap could not resolve near points). Points render at constant
  // SCREEN radius, so magnification genuinely separates near-identical embeddings instead of
  // growing blobs; 2000x is enough to split anything the 3-decimal coordinates can distinguish.
  const f=Math.exp(-e.deltaY*0.0015), nk=Math.min(Math.max(view.k*f,0.5),2000);
  const rf=nk/view.k;
  view.tx = mx - (mx-view.tx)*rf; view.ty = my - (my-view.ty)*rf; view.k = nk; draw(); },{passive:false});
// PAN: shift-drag OR middle-button drag (button 1). Plain drag rotates in 3D.
cv.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,tx:view.tx,ty:view.ty,
    ax:rot.ax,ay:rot.ay,az:rot.az,pan:e.shiftKey||e.button===1,roll:e.altKey};
  if(e.button===1) e.preventDefault();
  cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointermove',e=>{ if(!drag)return;
  if(is3d() && !drag.pan){
    if(drag.roll){ rot.az=drag.az+(e.clientX-drag.x)*0.008; draw(); return; }
    rot.ay=drag.ay+(e.clientX-drag.x)*0.008;
    rot.ax=drag.ax+(e.clientY-drag.y)*0.008; draw(); return; }
  view.tx=drag.tx+(e.clientX-drag.x); view.ty=drag.ty+(e.clientY-drag.y); draw(); });
cv.addEventListener('pointerup',()=>{drag=null;});
cv.addEventListener('pointercancel',()=>{drag=null;});
cv.addEventListener('auxclick',e=>e.preventDefault());
cv.style.cursor='grab';
document.getElementById('ptsize').addEventListener('input',e=>{ptScale=+e.target.value;draw();});
document.getElementById('cohby').addEventListener('click',e=>{
  const b=e.target.closest('button'); if(!b)return; cohMode=+b.dataset.c;
  [...e.currentTarget.children].forEach(x=>x.setAttribute('aria-pressed',x===b));
  update(); });
// data exported before the cohort feature has no coh array -- hide the toggle rather than
// offering a mode that would draw an empty cloud
if(!D.coh) document.getElementById('cohby').parentElement.style.display='none';
if(!D.endt) document.querySelector('#colorby [data-k=endt]').style.display='none';
document.getElementById('ghostby').addEventListener('click',e=>{
  const b=e.target.closest('button'); if(!b)return; ghost=b.dataset.g==='1';
  [...e.currentTarget.children].forEach(x=>x.setAttribute('aria-pressed',x===b)); draw(); });
const playBtn=document.getElementById('play');
function stop(){ playing=false; playBtn.innerHTML='&#9654; play'; if(raf)cancelAnimationFrame(raf); raf=null; }
let acc=0,last=0;
function loop(t){ if(!playing)return; if(!last)last=t; acc+=t-last; last=t;
  if(acc>90){ acc=0; let v=+slider.value+1; if(v>+slider.max)v=+slider.min; slider.value=v; update(); gameInfo(); }
  raf=requestAnimationFrame(loop); }
playBtn.addEventListener('click',()=>{ if(playing){stop();return;}
  playing=true; last=0; acc=0; playBtn.innerHTML='&#10074;&#10074; pause'; raf=requestAnimationFrame(loop); });
// train play: advance one checkpoint frame every 450ms, loop at the end
const stepBtn=document.getElementById('playstep');
let stepping=false, stepTimer=null;
function stopStep(){ stepping=false; stepBtn.innerHTML='&#9654; train'; if(stepTimer)clearTimeout(stepTimer); stepTimer=null; }
stepBtn.addEventListener('click',()=>{ if(stepping){stopStep();return;}
  stepping=true; stepBtn.innerHTML='&#10074;&#10074; pause';
  const tick=()=>{ if(!stepping)return; fi=(fi+1)%D.frames.length; stepsl.value=fi; hdr(); draw();
    stepTimer=setTimeout(tick,450); }; tick(); });
if(window.matchMedia('(prefers-reduced-motion:reduce)').matches){
  playBtn.style.display='none'; stepBtn.style.display='none'; }
new ResizeObserver(fit).observe(cv);
window.addEventListener('resize',fit);
legend(); gameInfo(); hdr(); fit(); update();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Watching the field organise")
    ap.add_argument("--tag", default="checkpoint ladder")
    args = ap.parse_args()
    with open(args.data) as fh:
        data = fh.read()
    html = HTML.replace("__TITLE__", args.title).replace("__TAG__", args.tag)
    html = html.replace("__DATA__", data)
    with open(args.out, "w") as fh:
        fh.write(html)
    print(f"[viewer] -> {args.out} ({os.path.getsize(args.out)/2**20:.1f} MB)")


if __name__ == "__main__":
    main()
