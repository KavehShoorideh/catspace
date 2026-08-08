#!/usr/bin/env python
"""build_plank_page.py -- the playable ChessPlank artifact: full rules + the real network forward
+ threat-first navigation, all in-browser, weights inlined. The engine's reasoning is shown per
move (three pole-distances + margin) -- people see the geometry choose.

Self-test at load: recomputes the exported startpos distances in JS and refuses to play unless
they match PyTorch to 0.5 -- a silent porting mismatch would otherwise play garbage confidently.
"""
from __future__ import annotations

import argparse
import os

HTML = r"""<title>ChessPlank — navigation on a learned quasimetric</title>
<style>
:root{--bg:#f5f4f1;--panel:#fff;--edge:#ddd9d2;--ink:#1d1c1a;--dim:#75716a;--acc:#b3502e;
--sqd:#b58863;--sql:#f0d9b5;--hl:#7ba05baa;--sel:#e8c35a;--mono:ui-monospace,Menlo,monospace}
@media(prefers-color-scheme:dark){:root{--bg:#151412;--panel:#1f1d1a;--edge:#37332e;--ink:#e8e5df;
--dim:#8f8a82;--acc:#d97550;--sqd:#8a6a4f;--sql:#c0a880}}
:root[data-theme=dark]{--bg:#151412;--panel:#1f1d1a;--edge:#37332e;--ink:#e8e5df;--dim:#8f8a82;
--acc:#d97550;--sqd:#8a6a4f;--sql:#c0a880}
:root[data-theme=light]{--bg:#f5f4f1;--panel:#fff;--edge:#ddd9d2;--ink:#1d1c1a;--dim:#75716a;--acc:#b3502e;
--sqd:#b58863;--sql:#f0d9b5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:24px 18px}
h1{font-size:20px;margin:0 0 2px}
.sub{color:var(--dim);font-size:12.5px;margin-bottom:16px}
.cols{display:grid;grid-template-columns:minmax(300px,480px) 1fr;gap:22px}
@media(max-width:760px){.cols{grid-template-columns:1fr}}
#board{width:100%;max-width:480px;aspect-ratio:1}
</style><style>__CG_CSS__</style><style>
.bar{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap;align-items:center}
button{font:inherit;font-size:13px;padding:6px 12px;border:1px solid var(--edge);background:var(--panel);
color:var(--ink);border-radius:5px;cursor:pointer}
button:focus-visible{outline:2px solid var(--acc)}
#status{font-family:var(--mono);font-size:12.5px;color:var(--dim)}
.panel{background:var(--panel);border:1px solid var(--edge);border-radius:6px;padding:12px 14px}
.panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin:0 0 8px}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:11.5px}
th,td{text-align:right;padding:2.5px 7px;border-bottom:1px solid var(--edge)}
th:first-child,td:first-child{text-align:left}
tr.pick td{color:var(--acc);font-weight:700}
#test{font-family:var(--mono);font-size:11px;padding:3px 8px;border-radius:4px}
#test.ok{background:#2e8b5722;color:#2e8b57}#test.bad{background:#c0392b22;color:#c0392b}
.note{color:var(--dim);font-size:12px;margin-top:10px;max-width:60ch}
</style>
<div class="wrap">
<h1>ChessPlank</h1>
<div class="sub">a chess engine that navigates a learned quasimetric field — no search, no eval
function: every move is chosen by pushing the nearest unwanted ending away and pulling the win
closer. The table shows the geometry deciding.</div>
<div class="cols">
<div>
  <div id="board" class="cg-wrap"></div>
  <div class="bar">
    <button id="new">new game</button>
    <button id="flip">play black</button>
    <span id="test">verifying model…</span>
    <span id="status"></span>
  </div>
  <div class="note">You play by clicking. The engine embeds every legal reply with a 2-layer
  transformer (running in your browser), measures each child's distance to the WIN / DRAW / LOSS
  poles of its field, and plays the move maximising <b>d(nearest&nbsp;threat)&nbsp;−&nbsp;d(win)</b>.
  It is deliberately search-free and endgame-weak (no tablebases in the browser) — an honest floor
  of what the geometry alone knows.</div>
</div>
<div class="panel">
  <h2>the engine's last deliberation — one row per legal move</h2>
  <table id="think"><thead><tr><th>move</th><th>d→win</th><th>d→draw</th><th>d→loss</th>
  <th>margin</th></tr></thead><tbody></tbody></table>
</div>
</div>
</div>
<script>__CG_JS__</script>
<script id="W" type="application/json">__WEIGHTS__</script>
<script>
"use strict";
const W=JSON.parse(document.getElementById('W').textContent);
window.onerror=function(m,src,l,c,e){var el=document.getElementById('status');
  if(el)el.textContent="JS error: "+m+" @"+l+":"+c; return false;};

/* ================= chess rules (ids match the tokenizer: 0 empty, 1-6 PNBRQK, 7-12 pnbrqk) === */
const WP=1,WN=2,WB=3,WR=4,WQ=5,WK=6,BP=7,BN=8,BB=9,BR=10,BQ=11,BK=12;
const isW=p=>p>=1&&p<=6, isB=p=>p>=7;
const GLYPH={1:"♙",2:"♘",3:"♗",4:"♖",5:"♕",6:"♔",7:"♟",8:"♞",9:"♝",10:"♜",11:"♛",12:"♚",0:""};
function startState(){
  const b=new Int8Array(64);
  const back=[WR,WN,WB,WQ,WK,WB,WN,WR];
  for(let f=0;f<8;f++){b[f]=back[f];b[8+f]=WP;b[48+f]=BP;b[56+f]=back[f]+6;}
  return {b, turn:1, cK:1,cQ:1,ck:1,cq:1, ep:0, hist:[]};
}
const KN=[[1,2],[2,1],[2,-1],[1,-2],[-1,-2],[-2,-1],[-2,1],[-1,2]];
const KI=[[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]];
const DIAG=[[1,1],[1,-1],[-1,1],[-1,-1]], ORTH=[[1,0],[-1,0],[0,1],[0,-1]];
const F=s=>s&7, R=s=>s>>3, SQ=(f,r)=>r*8+f, on=(f,r)=>f>=0&&f<8&&r>=0&&r<8;
function attacked(st,sq,byWhite){
  const b=st.b, f0=F(sq), r0=R(sq);
  const P=byWhite?WP:BP, N=byWhite?WN:BN, B_=byWhite?WB:BB, Rk=byWhite?WR:BR, Q=byWhite?WQ:BQ, K=byWhite?WK:BK;
  const dr=byWhite?-1:1;                       // pawn ATTACKING sq sits one rank behind (from its POV)
  for(const df of[-1,1]){const f=f0+df,r=r0+dr;if(on(f,r)&&b[SQ(f,r)]===P)return true;}
  for(const[df,dr2]of KN){const f=f0+df,r=r0+dr2;if(on(f,r)&&b[SQ(f,r)]===N)return true;}
  for(const[df,dr2]of KI){const f=f0+df,r=r0+dr2;if(on(f,r)&&b[SQ(f,r)]===K)return true;}
  for(const[df,dr2]of DIAG){let f=f0+df,r=r0+dr2;while(on(f,r)){const p=b[SQ(f,r)];
    if(p){if(p===B_||p===Q)return true;break;}f+=df;r+=dr2;}}
  for(const[df,dr2]of ORTH){let f=f0+df,r=r0+dr2;while(on(f,r)){const p=b[SQ(f,r)];
    if(p){if(p===Rk||p===Q)return true;break;}f+=df;r+=dr2;}}
  return false;
}
function kingSq(st,white){const K=white?WK:BK;for(let s=0;s<64;s++)if(st.b[s]===K)return s;return -1;}
function makeMove(st,m){                        // m={from,to,promo,ep,castle}
  const n={b:st.b.slice(),turn:st.turn^1,cK:st.cK,cQ:st.cQ,ck:st.ck,cq:st.cq,ep:0,hist:st.hist};
  const p=n.b[m.from]; n.b[m.from]=0;
  n.b[m.to]=m.promo?m.promo:p;
  if(m.ep){ n.b[m.to+(st.turn?-8:8)]=0; }
  if(m.castle==="K"){n.b[st.turn?5:61]=n.b[st.turn?7:63];n.b[st.turn?7:63]=0;}
  if(m.castle==="Q"){n.b[st.turn?3:59]=n.b[st.turn?0:56];n.b[st.turn?0:56]=0;}
  if(p===WK){n.cK=0;n.cQ=0;} if(p===BK){n.ck=0;n.cq=0;}
  if(m.from===7||m.to===7)n.cK=0; if(m.from===0||m.to===0)n.cQ=0;
  if(m.from===63||m.to===63)n.ck=0; if(m.from===56||m.to===56)n.cq=0;
  if((p===WP&&m.to-m.from===16)||(p===BP&&m.from-m.to===16)) n.ep=F(m.from)+1;
  return n;
}
function pseudo(st){
  const b=st.b, white=st.turn===1, out=[];
  const mine=white?isW:isB, theirs=white?isB:isW;
  for(let s=0;s<64;s++){
    const p=b[s]; if(!p||!mine(p))continue;
    const f0=F(s),r0=R(s), t=p>6?p-6:p;
    if(t===1){ const dr=white?1:-1, start=white?1:6, last=white?7:0;
      const one=SQ(f0,r0+dr);
      if(on(f0,r0+dr)&&!b[one]){
        if(R(one)===last){out.push({from:s,to:one,promo:white?WQ:BQ});}
        else{out.push({from:s,to:one});
          if(r0===start&&!b[SQ(f0,r0+2*dr)])out.push({from:s,to:SQ(f0,r0+2*dr)});}}
      for(const df of[-1,1]){const f=f0+df,r=r0+dr;if(!on(f,r))continue;const q=SQ(f,r);
        if(b[q]&&theirs(b[q])){
          if(r===last)out.push({from:s,to:q,promo:white?WQ:BQ});
          else out.push({from:s,to:q});}
        else if(!b[q]&&st.ep&&f===st.ep-1&&r===(white?5:2))out.push({from:s,to:q,ep:1});}}
    else if(t===2){for(const[df,dr]of KN){const f=f0+df,r=r0+dr;if(!on(f,r))continue;
      const q=SQ(f,r);if(!b[q]||theirs(b[q]))out.push({from:s,to:q});}}
    else if(t===6){for(const[df,dr]of KI){const f=f0+df,r=r0+dr;if(!on(f,r))continue;
      const q=SQ(f,r);if(!b[q]||theirs(b[q]))out.push({from:s,to:q});}
      const home=white?4:60;
      if(s===home){
        const rts=white?[st.cK,st.cQ]:[st.ck,st.cq];
        if(rts[0]&&!b[home+1]&&!b[home+2]&&!attacked(st,home,!white)&&!attacked(st,home+1,!white)&&!attacked(st,home+2,!white))
          out.push({from:s,to:home+2,castle:"K"});
        if(rts[1]&&!b[home-1]&&!b[home-2]&&!b[home-3]&&!attacked(st,home,!white)&&!attacked(st,home-1,!white)&&!attacked(st,home-2,!white))
          out.push({from:s,to:home-2,castle:"Q"});}}
    else{const rays=t===3?DIAG:t===4?ORTH:DIAG.concat(ORTH);
      for(const[df,dr]of rays){let f=f0+df,r=r0+dr;
        while(on(f,r)){const q=SQ(f,r);
          if(!b[q])out.push({from:s,to:q});
          else{if(theirs(b[q]))out.push({from:s,to:q});break;}
          f+=df;r+=dr;}}}}
  return out;
}
function legal(st){
  const white=st.turn===1;
  return pseudo(st).filter(m=>{const n=makeMove(st,m);return !attacked(n,kingSq(n,white),!white);});
}
function gameOver(st){
  if(legal(st).length===0){
    const white=st.turn===1;
    if(attacked(st,kingSq(st,white),!white))return white?"black wins (checkmate)":"white wins (checkmate)";
    return "draw (stalemate)";
  }
  let pieces=0;for(let s=0;s<64;s++)if(st.b[s])pieces++;
  if(pieces<=2)return "draw (bare kings)";
  return null;
}
/* ================= model forward ============================================================ */
const D=W.d, H=W.heads, DK=D/H, NT=65, HD=W.head_d, HH=HD/2;
function erf(x){const s=x<0?-1:1;x=Math.abs(x);const t=1/(1+0.3275911*x);
  const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);
  return s*y;}
const gelu=x=>0.5*x*(1+erf(x/Math.SQRT2));
function ln(x,w,b,off,n){let mu=0;for(let i=0;i<n;i++)mu+=x[off+i];mu/=n;
  let v=0;for(let i=0;i<n;i++){const d0=x[off+i]-mu;v+=d0*d0;}v=Math.sqrt(v/n+1e-5);
  const o=new Float32Array(n);for(let i=0;i<n;i++)o[i]=(x[off+i]-mu)/v*w[i]+b[i];return o;}
function forward(tok,glob){
  // x: (65, D) -- row 0 = cls + glob_proj(glob); rows 1..64 = piece_emb[tok]+sq_emb
  const x=new Float32Array(NT*D);
  for(let i=0;i<D;i++){let g=W.glob_b[i];for(let j=0;j<6;j++)g+=W.glob_w[i*6+j]*glob[j];
    x[i]=W.cls[i]+g;}
  for(let s=0;s<64;s++)for(let i=0;i<D;i++)
    x[(s+1)*D+i]=W.piece_emb[tok[s]*D+i]+W.sq_emb[s*D+i];
  for(const L of W.L){
    // attention (pre-norm)
    const q=new Float32Array(NT*D),k=new Float32Array(NT*D),v=new Float32Array(NT*D);
    const xn=new Float32Array(NT*D);
    for(let t=0;t<NT;t++){const r=ln(x,L.ln1_w,L.ln1_b,t*D,D);xn.set(r,t*D);}
    for(let t=0;t<NT;t++)for(let i=0;i<D;i++){
      let sq_=L.qkv_b[i],sk=L.qkv_b[D+i],sv=L.qkv_b[2*D+i];
      for(let j=0;j<D;j++){const xv=xn[t*D+j];
        sq_+=L.qkv_w[i*D+j]*xv; sk+=L.qkv_w[(D+i)*D+j]*xv; sv+=L.qkv_w[(2*D+i)*D+j]*xv;}
      q[t*D+i]=sq_;k[t*D+i]=sk;v[t*D+i]=sv;}
    const ao=new Float32Array(NT*D);
    for(let h=0;h<H;h++){const o0=h*DK;
      for(let t=0;t<NT;t++){
        const sc=new Float32Array(NT);let mx=-1e30;
        for(let u=0;u<NT;u++){let s0=0;
          for(let i=0;i<DK;i++)s0+=q[t*D+o0+i]*k[u*D+o0+i];
          sc[u]=s0/Math.sqrt(DK);if(sc[u]>mx)mx=sc[u];}
        let Z=0;for(let u=0;u<NT;u++){sc[u]=Math.exp(sc[u]-mx);Z+=sc[u];}
        for(let i=0;i<DK;i++){let s0=0;
          for(let u=0;u<NT;u++)s0+=sc[u]*v[u*D+o0+i];
          ao[t*D+o0+i]=s0/Z;}}}
    for(let t=0;t<NT;t++)for(let i=0;i<D;i++){
      let s0=L.ao_b[i];for(let j=0;j<D;j++)s0+=L.ao_w[i*D+j]*ao[t*D+j];
      x[t*D+i]+=s0;}
    // mlp (pre-norm)
    const FF=4*D;
    for(let t=0;t<NT;t++){
      const r=ln(x,L.ln2_w,L.ln2_b,t*D,D);
      const hbuf=new Float32Array(FF);
      for(let i=0;i<FF;i++){let s0=L.m1_b[i];for(let j=0;j<D;j++)s0+=L.m1_w[i*D+j]*r[j];
        hbuf[i]=gelu(s0);}
      for(let i=0;i<D;i++){let s0=L.m2_b[i];for(let j=0;j<FF;j++)s0+=L.m2_w[i*FF+j]*hbuf[j];
        x[t*D+i]+=s0;}}}
  const phi=ln(x,W.out_ln_w,W.out_ln_b,0,D);
  const z=new Float32Array(HD);
  for(let i=0;i<HD;i++){let s0=W.proj_b[i];for(let j=0;j<D;j++)s0+=W.proj_w[i*D+j]*phi[j];z[i]=s0;}
  return z;
}
function iqeB(z,pole){
  // B-block = second half; comps over HH dims, k dims each; union length of [u, max(u,v)]
  const C=W.iqe_components, K=HH/C;
  let mx=-1e30, mean=0;
  for(let c=0;c<C;c++){
    const iv=[];
    for(let i=0;i<K;i++){const u=z[HH+c*K+i], v=pole[HH+c*K+i];
      iv.push([u,Math.max(u,v)]);}
    iv.sort((a,b)=>a[0]-b[0]);
    let len=0,reach=-1e30;
    for(const[lo,hi]of iv){len+=Math.max(0,hi-Math.max(lo,reach));if(hi>reach)reach=hi;}
    mean+=len;if(len>mx)mx=len;}
  mean/=C;
  return W.iqe_scale*(W.iqe_alpha*mx+(1-W.iqe_alpha)*mean);
}
function tokglob(st){const glob=[st.turn,st.cK,st.cQ,st.ck,st.cq,st.ep];return [st.b,glob];}
function poleDists(st){
  const [tk,gl]=tokglob(st); const z=forward(tk,gl);
  return {W:iqeB(z,W.poles.WIN), D:iqeB(z,W.poles.DRAW), L:iqeB(z,W.poles.LOSS)};
}
/* self-test against the PyTorch export */
(function(){
  const z=forward(W.test.tok,W.test.glob);
  const got={WIN:iqeB(z,W.poles.WIN),DRAW:iqeB(z,W.poles.DRAW),LOSS:iqeB(z,W.poles.LOSS)};
  const el=document.getElementById('test');
  let ok=true;
  for(const k of["WIN","DRAW","LOSS"])if(Math.abs(got[k]-W.test.d[k])>0.5)ok=false;
  if(ok){el.textContent="model verified ✓";el.className="ok";}
  else{el.textContent=`MODEL MISMATCH (js ${got.WIN.toFixed(1)}/${got.DRAW.toFixed(1)}/${got.LOSS.toFixed(1)} vs ${W.test.d.WIN}/${W.test.d.DRAW}/${W.test.d.LOSS}) — refusing to play`;
    el.className="bad";document.getElementById("board").style.pointerEvents="none";}
})();

/* threat-first: child's mover = opponent -> our win = d(child->LOSS) etc. */
function engineMove(st){
  const ms=legal(st); if(!ms.length)return null;
  const rows=[];
  for(const m of ms){
    const n=makeMove(st,m);
    const d=poleDists(n);
    const dwin=d.L, ddraw=d.D, dloss=d.W;
    const dbad=Math.min(ddraw,dloss);
    rows.push({m, dwin, ddraw, dloss, margin:dbad-dwin, dbad});
  }
  rows.sort((a,b)=>(b.margin-a.margin)||(b.dbad-a.dbad));
  return rows;
}
/* ================= UI (lichess chessground) ================================================ */
const files="abcdefgh";
const alg=s=>files[F(s)]+(R(s)+1);
const nameOf=m=>m.castle?(m.castle==="K"?"O-O":"O-O-O"):alg(m.from)+alg(m.to)+(m.promo?"=Q":"");
function fenOf(st){
  const SYM={1:"P",2:"N",3:"B",4:"R",5:"Q",6:"K",7:"p",8:"n",9:"b",10:"r",11:"q",12:"k"};
  let rows=[];
  for(let r=7;r>=0;r--){let row="",run=0;
    for(let f=0;f<8;f++){const p=st.b[SQ(f,r)];
      if(!p)run++;else{if(run){row+=run;run=0;}row+=SYM[p];}}
    if(run)row+=run;rows.push(row);}
  let c=(st.cK?"K":"")+(st.cQ?"Q":"")+(st.ck?"k":"")+(st.cq?"q":"");
  const ep=st.ep?files[st.ep-1]+(st.turn?6:3):"-";
  return rows.join("/")+" "+(st.turn?"w":"b")+" "+(c||"-")+" "+ep+" 0 1";
}
function destsOf(st){
  const m=new Map();
  for(const mv of legal(st)){const a=alg(mv.from);
    if(!m.has(a))m.set(a,[]);m.get(a).push(alg(mv.to));}
  return m;
}
let st=startState(), humanWhite=true, thinking=false;
let cg;
try{ cg=window.Chessground(document.getElementById('board'),{fen:fenOf(st),coordinates:true}); }
catch(e){ document.getElementById('status').textContent="board init failed: "+e.message;
  cg={set:()=>{}}; }
function inCheck(st){const w=st.turn===1;return attacked(st,kingSq(st,w),!w);}
function sync(last){
  const over=gameOver(st);
  cg.set({fen:fenOf(st), turnColor:st.turn?"white":"black",
    check:inCheck(st), lastMove:last||undefined,
    movable:{free:false, color:over?undefined:(humanWhite?"white":"black"),
             dests:((st.turn===1)===humanWhite&&!over)?destsOf(st):new Map(),
             events:{after:onUser}}});
  setStatus(over|| (((st.turn===1)===humanWhite)?"your move":"") );
}
function setStatus(t){document.getElementById('status').textContent=t;}
function showThink(rows){
  const tb=document.querySelector('#think tbody');tb.innerHTML="";
  rows.slice(0,18).forEach((r,i)=>{
    const tr=document.createElement('tr');if(i===0)tr.className="pick";
    tr.innerHTML=`<td>${nameOf(r.m)}</td><td>${r.dwin.toFixed(1)}</td>`+
      `<td>${r.ddraw.toFixed(1)}</td><td>${r.dloss.toFixed(1)}</td><td>${r.margin.toFixed(1)}</td>`;
    tb.appendChild(tr);});
}
function onUser(orig,dest){
  const mv=legal(st).find(m=>alg(m.from)===orig&&alg(m.to)===dest);
  if(!mv){sync();return;}
  st=makeMove(st,mv);sync([orig,dest]);maybeEngine();
}
function maybeEngine(){
  if(gameOver(st))return;
  if((st.turn===1)===humanWhite||thinking)return;
  thinking=true;setStatus("thinking…");
  setTimeout(()=>{
    const rows=engineMove(st);
    let last;
    if(rows){const m=rows[0].m;last=[alg(m.from),alg(m.to)];st=makeMove(st,m);showThink(rows);}
    thinking=false;sync(last);
  },30);
}
document.getElementById('new').onclick=()=>{st=startState();sync();maybeEngine();};
document.getElementById('flip').onclick=e=>{humanWhite=!humanWhite;
  e.target.textContent=humanWhite?"play black":"play white";
  cg.set({orientation:humanWhite?"white":"black"});
  st=startState();sync();maybeEngine();};

sync();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import os as _os
    cg = _os.path.join(_os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp/cg")
    cg_js = open(_os.path.join(cg, "bundle.js")).read()
    cg_css = open(_os.path.join(cg, "bundle.css")).read()
    with open(args.weights) as f:
        w = f.read()
    html = HTML.replace("__CG_JS__", cg_js).replace("__CG_CSS__", cg_css)
    with open(args.out, "w") as f:
        f.write(html.replace("__WEIGHTS__", w))
    print(f"[plank-page] -> {args.out} ({os.path.getsize(args.out)/2**20:.1f} MB)")


if __name__ == "__main__":
    main()
