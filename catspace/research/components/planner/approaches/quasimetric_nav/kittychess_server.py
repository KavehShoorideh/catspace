#!/usr/bin/env python
"""kittychess_server.py -- KittyChess over localhost/tailscale: the FULL engine (PyTorch on MPS,
Syzygy lookup at <=5 pieces), lichess chessground UI, zero browser-side model.

    .venv/bin/python -m ...kittychess_server --ckpt <ckpt.pt> [--port 8420]
then open http://<tailscale-ip>:8420

stdlib http.server, single process, single engine instance -- no worker pools (the 2026-07-27
orphaned-viz-workers swap-thrash is why).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess

from catspace.io import paths
from catspace.research.components.planner.approaches.quasimetric_nav.player_profile import (
    PlayerStore, fen_key, expectation_dist, ranked_E, surprisal_bits, entropy_bits,
    SURPRISE_BITS, TAU_E)
from catspace.research.components.planner.approaches.quasimetric_nav.banter import (
    make_banter, TemplateBanter)
from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess as _KC
KittyMATE=_KC.MATE

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_assets")
GAME_LOG = None

PAGE = """<!doctype html><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1">
<title>catspace</title>
<style>__CG_CSS__</style>
<style>
body{margin:0;background:#161512;color:#bababa;font:14px/1.4 'Noto Sans',system-ui,sans-serif}
.top{padding:10px 16px;font-size:15px;color:#dedede}
.top b{color:#fff}
.main{display:grid;grid-template-columns:24px minmax(280px,600px) minmax(260px,420px);gap:10px;
padding:0 16px 20px;max-width:1100px}
@media(max-width:760px){.main{grid-template-columns:20px 1fr;} .right{grid-column:1/3}}
#evalbar{position:relative;border-radius:3px;overflow:hidden;background:#333;align-self:stretch}
#evalbar div{width:100%;transition:height .3s}
#eb-b{background:#403d39}#eb-d{background:#8b8680}#eb-w{background:#f0efeb}
#board{width:100%;aspect-ratio:1}
.nav{display:flex;gap:4px;margin-top:8px}
.nav button{flex:1;font-size:15px;padding:5px 0;background:#262421;border:none;color:#bababa;
border-radius:3px;cursor:pointer}
.nav button:hover{background:#3a3733}
.right{display:flex;flex-direction:column;gap:10px;min-width:0}
.box{background:#262421;border-radius:4px;padding:10px 12px}
.engrow{display:flex;align-items:center;gap:10px;font-size:12.5px;color:#8f8a82;flex-wrap:wrap}
.engrow b{color:#dedede;font-size:14px}
select{background:#161512;color:#bababa;border:1px solid #3a3733;border-radius:3px;font:inherit}
.lines .ln{display:block;padding:4px 2px;border-bottom:1px solid #33312e;cursor:pointer;
font-family:'Noto Sans',sans-serif;font-size:13px}
.lines .lnr{display:flex;gap:8px;align-items:center}
.lines .minibar{display:inline-flex;width:64px;height:9px;border-radius:2px;overflow:hidden;
flex:none;background:#333}
.lines .minibar i{display:block;height:100%}
.lines .sub{font-size:10.5px;color:#8f8a82;margin-top:1px;font-variant-numeric:tabular-nums;
overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lines .ln:hover{background:#302d2a}
.lines .ev{min-width:52px;font-weight:600;color:#dedede;font-size:12px;font-variant-numeric:tabular-nums}
.lines .mv{white-space:normal;word-break:break-word;color:#bababa}
.lines .ln.best .ev{color:#759900}
#moves{font-size:13.5px;line-height:1.9;max-height:170px;overflow-y:auto;overflow-wrap:anywhere}
@media(max-width:760px){#moves{max-height:110px;font-size:12.5px;line-height:1.6}}
#moves .no{color:#6f6b66;margin:0 3px 0 6px}
#moves .m{padding:1px 4px;border-radius:2px;cursor:pointer}
#moves .m:hover{background:#3a3733}
#moves .m.cur{background:#759900;color:#fff}
#wdltxt{margin-top:6px;font-variant-numeric:tabular-nums;font-size:12px;color:#8f8a82}
#wdltxt table{border-collapse:collapse}
#wdltxt th{font-weight:400;color:#6f6b66;text-align:right;padding:1px 10px 1px 0;font-size:11px}
#wdltxt td{text-align:right;padding:1px 10px 1px 0;white-space:nowrap}
#wdltxt b{color:#dedede;font-weight:600}
#ctable{border-collapse:collapse;font-size:11.5px;width:100%;font-variant-numeric:tabular-nums}
#ctable th{font-weight:400;color:#6f6b66;text-align:left;padding:2px 6px;font-size:10.5px}
#ctable td{padding:2px 6px;color:#8f8a82;white-space:nowrap}
#ctable td.on{color:#759900;font-weight:600}
#ctable tr.tokhit td{background:#2e2c26}
.lines .ev{cursor:help}
#lnlegend{font-size:10.5px;color:#6f6b66;margin-top:4px}
.spin{opacity:.6}
label.sw{display:flex;gap:5px;align-items:center;cursor:pointer}
.navbtn{background:#3a3733;border:none;color:#bababa;border-radius:3px;padding:3px 10px;
cursor:pointer;font:inherit;font-size:12.5px}
.navbtn.on{background:#759900;color:#fff}
body.playmode #evalbar,body.playmode #wdltxt,body.playmode #conceptbox{visibility:hidden}
body.playmode .right .box:first-child{display:none}
body:not(.playmode) #surprbox{display:none!important}
#who{float:right;color:#8f8a82;cursor:pointer;font-size:13px}
#who:hover{color:#dedede}
#emote{position:absolute;z-index:9;top:6px;right:6px;display:none;max-width:250px;
background:#262421;border:1px solid #3a3733;border-radius:12px;padding:8px 12px;
font-size:13px;color:#dedede;box-shadow:0 3px 12px #000c;align-items:center;gap:8px}
#emote .e{font-size:28px;line-height:1}
#emote.show{display:flex;animation:pop .25s ease-out}
@keyframes pop{0%{transform:scale(.6);opacity:0}100%{transform:scale(1);opacity:1}}
#surprbox b{color:#dedede}
</style>
<div class="top"><b>catspace</b>
  <span style="margin-left:18px">
    <button class="navbtn" id="nav-an">Analysis board</button>
    <button class="navbtn" id="nav-play">Play catspace</button>
  </span>
  <span id="who" title="the engine keeps a per-player memory: your games, where you surprised it, and prepared lines against you"></span>
</div>
<div class="main">
<div id="evalbar"><div id="eb-b" style="height:33%"></div><div id="eb-d" style="height:34%"></div><div id="eb-w" style="height:33%"></div></div>
<div style="position:relative">
  <div id="emote"></div>
  <div id="exitline" style="display:none;font-size:13px;color:#bababa;padding:2px 0"></div>
  <div id="mat-top" style="min-height:16px;font-size:13px;color:#8f8a82;letter-spacing:1px"></div>
  <div id="board" class="cg-wrap"></div>
  <div id="mat-bot" style="min-height:16px;font-size:13px;color:#8f8a82;letter-spacing:1px"></div>
  <div class="nav">
    <button id="first">&#x23EE;</button><button id="prev">&#x25C0;</button>
    <button id="next">&#x25B6;</button><button id="last">&#x23ED;</button>
    <select id="new" style="flex:2;background:#262421;border:none;color:#bababa;border-radius:3px">
      <option value="">new game…</option>__POSITIONS__
    </select>
  </div>
  <div id="wdltxt"></div>
</div>
<div class="right">
  <div class="box">
    <div class="engrow"><b>catspace</b>
      <span id="dinfo"></span>
      <label title="search depth in plies (half-moves): how far ahead the engine reads every line. Higher = stronger and slower; the streaming analysis deepens one level at a time">depth <select id="depth" title="search depth (plies ahead)"><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option><option selected>6</option></select></label>
      <button id="plantog" title="PLANNER: the SubgoalFormer's live certificate — committed subgoal with revised probability, counterfactual worries, premove-safety — plus the alert diff since the last move and what the pointer policy would do (action + search budget)" style="background:#3a3733;border:none;color:#8f8a82;border-radius:3px;padding:2px 8px;cursor:pointer">plan</button>
      <button id="sqtog" title="SQUARE CONCEPTS: each square's learned concept code and its additive contribution to the evaluation (green helps white, red helps black). The per-square vocabulary is trained jointly; contributions are exact by construction (additive decoder)." style="background:#3a3733;border:none;color:#8f8a82;border-radius:3px;padding:2px 8px;cursor:pointer">sq</button>
      <button id="whytog" title="WHY overlay: positional importance beyond material (counterfactual removal, material-fitted residual). Green = load-bearing (doing MORE than its face value), red = underperforming or misplaced (removal costs little — or even helps its owner). Top 3 each. Kings excluded." style="background:#3a3733;border:none;color:#8f8a82;border-radius:3px;padding:2px 8px;cursor:pointer">why</button>
      <button id="sftog" title="toggle a Stockfish second opinion for the current position (referee only — never feeds our engine)" style="background:#3a3733;border:none;color:#8f8a82;border-radius:3px;padding:2px 8px;cursor:pointer">SF</button>
      <button id="flip" style="background:#3a3733;border:none;color:#bababa;border-radius:3px;padding:2px 8px;cursor:pointer">flip</button>
      <button id="pgnbtn" title="copy the game as PGN — the engine's reactions ride along as annotations ({comments} + !?/?! glyphs)" style="background:#3a3733;border:none;color:#8f8a82;border-radius:3px;padding:2px 8px;cursor:pointer">PGN</button>
      <button id="nexttog" title="FUTURE CONCEPTS: a causal transformer over this game's concept stream predicts which concepts NEWLY activate soon (within 2/6 plies), for each side — the sequence layer, history-aware where the geometry is memoryless" style="background:#3a3733;border:none;color:#8f8a82;border-radius:3px;padding:2px 8px;cursor:pointer">next</button>
    </div>
    <div class="lines" id="lnbox"></div>
    <div id="sfbox" style="display:none;font-size:12px;color:#dbac16;padding:3px 0"></div>
  <div id="sqbox" style="display:none;font-size:11px;color:#8f8a82;padding:3px 0;word-break:break-word"></div>
  <div id="planbox" style="display:none;font-size:12px;color:#bababa;padding:4px 0;line-height:1.6;border-top:1px solid #3a3733"></div>
  <div id="nextbox" style="display:none;font-size:12px;color:#bababa;padding:4px 0;line-height:1.6;border-top:1px solid #3a3733"></div>
  <div id="sqfloorrow" style="display:none;font-size:11px;color:#6f6b66;padding:1px 0">noise floor
    <input id="sqfloor" type="range" min="0" max="2" step="0.05" value="0.4" style="width:120px;vertical-align:middle">
    <span id="sqfloorval">0.40% E</span></div>
    <div id="lnlegend">E = expected points, white (probability head) &nbsp;·&nbsp; ⚡ only move = one move far above the rest &nbsp;·&nbsp; ↘ = committal descent &nbsp;·&nbsp; ranked by E (calibrated committor)</div>
  </div>
  <div class="box"><div id="moves"></div></div>
  <div class="box" id="surprbox" style="display:none" title="SURPRISE: the engine ponders on your clock and holds a probability over your legal moves; your move scores -log2 P(move) in bits. High = you left its expectation. It REMEMBERS what surprised it — and preps."></div>
  <div class="box" id="cdbox" style="display:none">
    <div style="font-size:12px;color:#8f8a82;margin-bottom:4px">position concepts</div>
    <select id="cdsel" style="width:100%;background:#262421;color:#bababa;border:1px solid #3a3733;border-radius:3px;padding:3px;font-size:12px"></select>
    <div id="cddeets" style="font-size:12px;margin-top:6px;line-height:1.55"></div>
  </div>
  <div class="box" id="conceptbox" style="display:none">
    <div style="font-size:12px;color:#8f8a82;margin-bottom:4px">concepts &nbsp;<span id="postoks" style="color:#6f6b66"></span></div>
    <table id="ctable"></table>
  </div>
</div>
</div>
<script>__CG_JS__</script>
<script>
"use strict";
let seq=0, orient="white", mode="analysis", playCfg={engineWhite:false,depth:2};
let user=localStorage.kcUser||"", surprHist=[], lastState=null;
function buildPGN(d){
  // the engine's reactions as PGN annotations (Kaveh 2026-08-13 ':D')
  let out='[Event "catspace"]\\n[Site "kittychess"]\\n';
  if(user) out+=`[${mode==='play'&&playCfg.engineWhite?'Black':'White'} "${user}"]\\n`;
  out+=`[${mode==='play'&&playCfg.engineWhite?'White':'Black'} "catspace"]\\n`;
  if(d.start_fen) out+=`[SetUp "1"]\\n[FEN "${d.start_fen}"]\\n`;
  out+='\\n';
  (d.san||[]).forEach((sn,i)=>{
    if(i%2===0) out+=((i/2+1)+'. ');
    const nt=(d.notes||{})[String(i)];
    out+=sn+(nt?(nt.glyph||''):'')+' ';
    if(nt) out+=`{ ${nt.emoji} ${nt.text.replace(/[{}]/g,'')} } `;
  });
  out+=(d.over?(d.over.startsWith('white')?'1-0':d.over.startsWith('black')?'0-1':'1/2-1/2'):'*');
  return out;
}
document.addEventListener('click',e=>{
  if(e.target&&e.target.id==='pgnbtn'&&lastState){
    navigator.clipboard.writeText(buildPGN(lastState)).then(()=>{
      e.target.textContent='copied!';setTimeout(()=>e.target.textContent='PGN',1200);});}});
function whoRender(){document.getElementById('who').textContent=user?('☺ '+user):'set player name';}
function setUser(){const u=prompt('Your name — the engine builds a per-player profile (games, surprises, prep):',user);
  if(u&&u.trim()){user=u.trim();localStorage.kcUser=user;}whoRender();}
document.getElementById('who').onclick=setUser;
whoRender();
let sfOn=false, sfLastFen=null, whyOn=false, whyLastFen=null, sqOn=false, sqLastFen=null, sqLast=null, planOn=false, planLastFen=null;
function planRefresh(fen){
  const box=document.getElementById('planbox');
  if(!planOn){box.style.display='none';return;}
  if(fen===planLastFen) return;
  planLastFen=fen; box.style.display=''; box.textContent='planning…';
  fetch('/api',{method:'POST',body:JSON.stringify({action:'plan'})}).then(r=>r.json()).then(e=>{
    if(e.err){box.textContent='plan: '+e.err;return;}
    let h=`<b>committed:</b> ${e.committed} · p̂ <b>${e.p_hat}</b>`+
      (e.premove_safe?' · <span style="color:#7fbf5f">PREMOVE-SAFE</span>':'');
    if(e.worries&&e.worries.length)
      h+='<br><b>worries:</b> '+e.worries.map(w=>`${w.name} (Δp̂ ${w.dp}, attn ${w.attn})`).join(' · ');
    if(e.alerts&&e.alerts.length)
      h+='<br><b>alerts since last move:</b> '+e.alerts.map(a=>`${a.kind==='opportunity'?'⬆':'⚠'} ${a.name} (${a.dp>0?'+':''}${a.dp})`).join(' · ');
    if(e.policy)
      h+=`<br><b>policy:</b> ${e.policy.action} · verify with ${e.policy.budget===0?'NO search (premove)':e.policy.budget+'s search'}`;
    box.innerHTML=h;
  }).catch(()=>{box.textContent='plan: unavailable';});
}
let nextOn=false, nextLastFen=null;
function nextRefresh(fen){
  const box=document.getElementById('nextbox');
  if(!nextOn){box.style.display='none';return;}
  if(fen===nextLastFen) return;
  nextLastFen=fen; box.style.display=''; box.textContent='reading the stream…';
  fetch('/api',{method:'POST',body:JSON.stringify({action:'futures'})}).then(r=>r.json()).then(e=>{
    if(e.err){box.textContent='next: '+e.err;return;}
    const opp=e.mover==='white'?'black':'white';
    const grp=s=>e.rows.filter(r=>r.side===s).slice(0,5)
      .map(r=>`<span title="P(newly activates) within 6 plies ${(100*r.p).toFixed(0)}% · within 2 plies ${(100*r.p2).toFixed(0)}% · leverage ${r.lev}">${r.name} <b>${(100*r.p).toFixed(0)}%</b></span>`).join(' · ');
    let h=`<b>coming for ${opp}</b> (your opponent${e.mover==='white'?'':''}): `+(grp(opp)||'nothing above 10%');
    h+=`<br><b>coming for ${e.mover}</b>: `+(grp(e.mover)||'nothing above 10%');
    const w=e.wdl_seq;
    h+=`<br><span style="color:#6f6b66;font-size:11px" title="outcome read from the concept SEQUENCE alone (no board eval)">sequence says w/d/b ${w.map(x=>(100*x).toFixed(0)+'%').join('/')}</span>`;
    box.innerHTML=h;
  }).catch(()=>{box.textContent='next: unavailable';});
}
function sqRefresh(fen){
  const box=document.getElementById('sqbox');
  if(!sqOn){if(!whyOn)cg.setShapes([]);box.style.display='none';return;}
  if(fen===sqLastFen) return;
  sqLastFen=fen;
  fetch('/api',{method:'POST',body:JSON.stringify({action:'sqconcepts'})}).then(r=>r.json()).then(e=>{
    if(!e.sal){box.style.display='';box.textContent='sq: '+(e.err||'n/a');return;}
    sqLast=e.sal;
    renderSq();
  }).catch(()=>{});
}
function renderSq(){
    if(!sqLast) return;
    const box=document.getElementById('sqbox');
    const FLOOR=(+document.getElementById('sqfloor').value)/100;
    document.getElementById('sqfloorval').textContent=(FLOOR*100).toFixed(2)+'% E';
    const sig=sqLast.filter(x=>Math.abs(x.imp)>=FLOOR).slice(0,8);
    const mx=Math.max(...sig.map(x=>Math.abs(x.imp)))||1;
    cg.setShapes(sig.map(x=>({orig:x.sq,
      brush:(x.imp>0?(Math.abs(x.imp)>=0.6*mx?'green':'paleGreen')
                    :(Math.abs(x.imp)>=0.6*mx?'red':'paleRed'))})));
    box.style.display='';
    box.innerHTML=sig.length?('sq concepts: '+sig.map(x=>
      `<b>${x.sq}</b>:s${x.code}(${x.imp>0?'+':''}${(x.imp*100).toFixed(1)})`).join(' · '))
      :'sq concepts: no square stands out at this floor';
}
function whyRefresh(fen){
  if(!whyOn){cg.setShapes([]);return;}
  if(fen===whyLastFen) return;
  whyLastFen=fen;
  fetch('/api',{method:'POST',body:JSON.stringify({action:'saliency'})}).then(r=>r.json()).then(e=>{
    if(!e.sal) return;
    const byX=e.sal.slice().sort((a,b)=>Math.abs(b.excess)-Math.abs(a.excess));
    const shapes=[];
    let pos=0,neg=0;
    for(const x of byX){
      if(x.excess>0.02&&pos<3){shapes.push({orig:x.sq,brush:x.excess>0.08?'green':'paleGreen'});pos++;}
      else if(x.excess<-0.02&&neg<3){shapes.push({orig:x.sq,brush:x.excess<-0.08?'red':'paleRed'});neg++;}
    }
    cg.setShapes(shapes);
  }).catch(()=>{});
}
function sfRefresh(fen){
  if(!sfOn){document.getElementById('sfbox').style.display='none';return;}
  const box=document.getElementById('sfbox');
  box.style.display=''; 
  if(fen===sfLastFen) return;
  sfLastFen=fen; box.textContent='SF: …';
  fetch('/api',{method:'POST',body:JSON.stringify({action:'sfeval'})}).then(r=>r.json()).then(e=>{
    if(e.err){box.textContent='SF: '+e.err;return;}
    const sc=e.mate!=null?('#'+e.mate):((e.cp>=0?'+':'')+(e.cp/100).toFixed(2));
    box.innerHTML=`SF: <b>${sc}</b>`+(e.wdl?` · w/d/l ${e.wdl.join('/')}‰`:'')+(e.pv?` · ${e.pv}`:'');
  }).catch(()=>{box.textContent='SF: unavailable';});
}
function setMode(m){
  mode=m;
  document.body.classList.toggle('playmode', m==='play');
  document.getElementById('nav-an').classList.toggle('on', m==='analysis');
  document.getElementById('nav-play').classList.toggle('on', m==='play');
}
const cg=window.Chessground(document.getElementById('board'),{coordinates:true});
const PIECEV={p:1,n:3,b:3,r:5,q:9};
const PIECEU={w:{p:'♙',n:'♘',b:'♗',r:'♖',q:'♕'},b:{p:'♟',n:'♞',b:'♝',r:'♜',q:'♛'}};
function matDiff(fen){
  // lichess-style: NET captured pieces per side + point advantage
  const start={p:8,n:2,b:2,r:2,q:1};
  const cnt={w:{p:0,n:0,b:0,r:0,q:0},b:{p:0,n:0,b:0,r:0,q:0}};
  for(const ch of fen.split(' ')[0]){
    const lo=ch.toLowerCase();
    if(cnt.w[lo]!==undefined) cnt[ch===lo?'b':'w'][lo]++;
  }
  let wPts=0,bPts=0,wCap='',bCap='';   // wCap = black pieces white has WON (shown by white)
  for(const t of ['p','n','b','r','q']){
    const lostByB=start[t]-cnt.b[t], lostByW=start[t]-cnt.w[t];
    const net=lostByB-lostByW;         // >0: white is up in this piece type
    if(net>0) wCap+=PIECEU.b[t].repeat(net);
    if(net<0) bCap+=PIECEU.w[t].repeat(-net);
    wPts+=cnt.w[t]*PIECEV[t]; bPts+=cnt.b[t]*PIECEV[t];
  }
  const d=wPts-bPts;
  return {w:wCap+(d>0?' +'+d:''), b:bCap+(d<0?' +'+(-d):'')};
}
function renderMat(fen){
  const m=matDiff(fen);
  const whiteBottom=(orient==='white');
  document.getElementById('mat-bot').textContent=whiteBottom?m.w:m.b;
  document.getElementById('mat-top').textContent=whiteBottom?m.b:m.w;
}
const knobs=()=>({depth:+document.getElementById('depth').value,
  lines:3,
  pvlen:8});
// Two-phase like lichess: the position updates INSTANTLY, the engine lines stream in after.
// A stale analysis (older seq) is discarded, so mashing the nav buttons stays responsive.
async function api(body){
  const my=++seq;
  const extra = mode==='play' ? {play:true, engineWhite:playCfg.engineWhite,
                                 playDepth:playCfg.depth} : {};
  if(mode==='play'){document.getElementById('dinfo').textContent="thinking…";
    document.getElementById('dinfo').className="spin";}
  const r=await fetch('/state',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({...body,...knobs(),...extra,user})});
  const d=await r.json();
  if(my!==seq)return;
  render(d);
  if(mode==='play'){document.getElementById('dinfo').textContent=d.over||"";
    document.getElementById('dinfo').className="";return;}
  if(d.over){document.getElementById('dinfo').textContent=d.over;return;}
  document.getElementById('dinfo').textContent="thinking…";
  document.getElementById('dinfo').className="spin";
  document.getElementById('lnbox').style.opacity=.4;
  await fetch('/state',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({action:"analyze",...knobs()})});
  // lichess-style streaming: poll partial lines while the search deepens
  while(my===seq){
    await new Promise(r=>setTimeout(r,250));
    if(my!==seq)return;
    const rp=await fetch('/state',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({action:"lines"})});
    const dp=await rp.json();
    if(my!==seq)return;
    if(dp.think.length){renderLines(dp);document.getElementById('lnbox').style.opacity=1;}
    const only=(dp.eff_moves!=null&&dp.eff_moves<=1.6);
    document.getElementById('dinfo').innerHTML=(only?`<span style="color:#dbac16" title="ONLY MOVE: one move is much better than the rest (effective moves ≈ ${dp.eff_moves})">⚡ only move</span> · `:"")+"depth "+dp.depth+(dp.done?"":"…")
      +(dp.resid!=null?"  ·  Bellman residual "+dp.resid.toFixed(1):"");
    document.getElementById('dinfo').className=dp.done?"":"spin";
    if(dp.done)break;
  }
}
function render(d){
  // ENGINE EMOTES (Kaveh 2026-08-13): the engine expresses itself; a generic {who,kind,
  // emoji,text} channel the user side will post into later
  if(d.emote){const em=document.getElementById('emote');
    em.innerHTML=`<span class="e">${d.emote.emoji}</span><span>${d.emote.text}</span>`;
    em.classList.add('show');clearTimeout(window._emT);
    window._emT=setTimeout(()=>em.classList.remove('show'),6000);
    if(d.emote.pending){                       // the LLM is still typing: swap the line in
      const sq=d.emote.seq;let tries=0;
      const poll=setInterval(async()=>{
        if(++tries>16){clearInterval(poll);return;}
        const r=await fetch('/state',{method:'POST',headers:{'content-type':'application/json'},
          body:JSON.stringify({action:'emote',seq:sq})});
        const e=await r.json();
        if(e.text){clearInterval(poll);
          em.innerHTML=`<span class="e">${e.emoji||d.emote.emoji}</span><span>${e.text}</span>`;
          em.classList.add('show');clearTimeout(window._emT);
          window._emT=setTimeout(()=>em.classList.remove('show'),6000);}},600);}}
  if(mode==='play'&&d.surprise){
    surprHist.push(d.surprise.xbits);
    const mean=(surprHist.reduce((a,b)=>a+b,0)/surprHist.length).toFixed(1);
    const sb=document.getElementById('surprbox');sb.style.display='';
    const x=d.surprise.xbits;
    sb.innerHTML=`<div style="font-size:12px;color:#8f8a82;margin-bottom:3px">surprise`+
      `${d.surprise.pondered?'':' <span title="you moved before the ponder finished — quick estimate">⏱</span>'}</div>`+
      `your move: <b>${x>0?'+':''}${x} bits</b> vs expected${x>=1.7?' 😮':''}`+
      ` · engine expected <b>${d.surprise.expected}</b> (p ${(100*d.surprise.p).toFixed(0)}%)`+
      `<div style="font-size:11px;color:#6f6b66;margin-top:2px" title="raw surprisal ${d.surprise.bits} bits · position entropy ${d.surprise.H} bits · expectation temperature τ=${d.surprise.tau} (high for new players, settles with data)">game mean ${mean>0?'+':''}${mean} bits`+
      `${d.prepared?' · 📖 replied from prep':''}</div>`;}
  const mvcolor = mode==='play' ? (playCfg.engineWhite?'black':'white') : d.turn;
  cg.set({fen:d.fen, turnColor:d.turn, check:d.check, lastMove:d.lastMove||undefined,
    orientation:orient,
    movable:{free:false, color:mvcolor, dests:new Map(Object.entries(d.dests)),
      events:{after:(o,t)=>api({action:"move",uci:o+t})}}});
  // tricolor committor bar: white at the bottom (lichess convention), draw grey between
  document.getElementById('eb-w').style.height=(d.wdl[0]*100)+"%";
  document.getElementById('eb-d').style.height=(d.wdl[1]*100)+"%";
  document.getElementById('eb-b').style.height=(d.wdl[2]*100)+"%";
  let lbl = d.turb==null ? "static" :
    (d.turb<0.08 ? "static · quiet" : "static · sharp τ"+d.turb.toFixed(2));
  if(d.trap&&d.trap.stalemate) lbl += " · ⚠"+d.trap.stalemate+" stalemate-in-1";
  if(d.trap&&d.trap.mate) lbl += " · #1 available";
  drawPosTable(d.wdl, d.dists, lbl);
  const xb=document.getElementById('exitline');
  if(d.exit){xb.style.display='';
    const xc={white:'#e8e6e1',draw:'#9a9a9a',black:'#403d39'}[d.exit.exit];
    xb.innerHTML=`E <b>${d.exit.E.toFixed(2)}</b> · exit: <b style="color:${xc};text-shadow:0 0 2px #000">${d.exit.exit.toUpperCase()}</b>`+
      (d.exit.plies!=null?` · ~<b>${Math.round(d.exit.plies)}</b> plies`:'');
  } else xb.style.display='none';
  const mv=document.getElementById('moves'); mv.innerHTML="";
  lastState=d;
  (d.san||[]).forEach((sn,i)=>{
    if(i%2===0){const no=document.createElement('span');no.className="no";no.textContent=(i/2+1)+".";mv.appendChild(no);}
    const sp=document.createElement('span');sp.className="m"+(i===d.ptr-1?" cur":"");
    const nt=(d.notes||{})[String(i)];
    sp.textContent=sn+(nt?(nt.glyph||'')+' '+nt.emoji:'');   // PGN annotation, inline
    if(nt) sp.title=nt.text;
    sp.onclick=()=>api({action:"goto",ptr:i+1}); mv.appendChild(sp);});
  const cur=mv.querySelector('.cur');
  if(cur){ // scroll ONLY the move list, never the page (scrollIntoView nudged the page)
    const t=cur.offsetTop-mv.offsetTop;
    if(t<mv.scrollTop||t>mv.scrollTop+mv.clientHeight-24) mv.scrollTop=t-mv.clientHeight/2;
  }
  if(d.concepts&&d.concepts.length){
    document.getElementById('conceptbox').style.display="";
    document.getElementById('postoks').textContent="position tokens: "+
      (d.tokens||[]).map((c,h)=>"h"+h+":c"+c).join(" ");
    const t=document.getElementById('ctable');
    t.innerHTML="<tr><th>known concept</th><th>here?</th><th>best token</th><th>P(concept|token)</th></tr>"+
      d.concepts.map(c=>
        `<tr class="${c.tok_here?'tokhit':''}"><td>${c.name}</td>`+
        `<td class="${c.active?'on':''}">${c.active?'✓':'–'}</td>`+
        `<td>${c.anti?'anti ':''}h${c.head}/c${c.code}${c.tok_here?' ●':''}</td>`+
        `<td>${Math.round(c.p*100)}% (base ${Math.round(c.base*100)}%)</td></tr>`).join("");
  }
  if(d.cdeets&&d.cdeets.length){
    document.getElementById('cdbox').style.display="";
    const sel=document.getElementById('cdsel');
    const prev=sel.value;
    const opt=(c,i)=>`<option value="${i}">${c.name}${c.p_act!=null?'  P(act) '+Math.round(c.p_act*100)+'%':''}${Math.abs(c.lev)>0.01?'  ('+(c.lev>0?'+':'')+c.lev.toFixed(2)+'E)':''}</option>`;
    const wtm=(d.turn==='white');
    const grp={act:[],mine:[],theirs:[],fn:[]};
    d.cdeets.forEach((c,i)=>{
      const servesMover=(c.lev>0.01)===wtm&&Math.abs(c.lev)>0.01;
      if(c.active) grp.act.push([c,i]);
      else if(Math.abs(c.lev)<=0.01) grp.fn.push([c,i]);
      else if(servesMover) grp.mine.push([c,i]);
      else grp.theirs.push([c,i]);});
    const mv=wtm?'WHITE':'BLACK', op=wtm?'BLACK':'WHITE';
    sel.innerHTML=
      `<optgroup label="● active in this position">${grp.act.map(([c,i])=>opt(c,i)).join('')}</optgroup>`+
      `<optgroup label="future — serves ${mv} (to move: pursue)">${grp.mine.map(([c,i])=>opt(c,i)).join('')}</optgroup>`+
      `<optgroup label="future — serves ${op} (threats: deny)">${grp.theirs.map(([c,i])=>opt(c,i)).join('')}</optgroup>`+
      (grp.fn.length?`<optgroup label="future — neutral">${grp.fn.map(([c,i])=>opt(c,i)).join('')}</optgroup>`:'');
    if(prev && prev<d.cdeets.length) sel.value=prev;
    const rend=()=>{const c=d.cdeets[+sel.value]; if(!c) return;
      document.getElementById('cddeets').innerHTML=
        `<b>${c.name}</b> ${c.active?'<span style="color:#7fbf5f">active here</span>':'<span style="color:#8f8a82">not active</span>'}<br>`+
        `serves: ${c.lev>0.01?'white':c.lev<-0.01?'black':'neutral'} (${(c.lev>=0?'+':'')+c.lev.toFixed(3)} E/activation)`+
        (c.br!=null?`<br>base rate: ${(c.br*100).toFixed(1)}%/move`:'')+
        (c.p_act!=null?`<br>from HERE: P(activate) ${Math.round(c.p_act*100)}%${c.dA!=null?' · dA '+c.dA:''}${c.dA_opp!=null?' · with THEM to move: dA '+c.dA_opp+(c.dA!=null?(c.dA_opp<c.dA?'  ⚠ they get there first':'  (we are closer)'):''):''}`:'')+
        (c.gates&&c.gates.length?`<br>leads into: ${c.gates.join(', ')}`:'');};
    sel.onchange=rend; rend();
  }
  if(d.over)document.getElementById('dinfo').textContent=d.over;
  sfRefresh(d.fen);
  whyRefresh(d.fen);
  sqRefresh(d.fen);
  planRefresh(d.fen);
  nextRefresh(d.fen);
  renderMat(d.fen);
}
function drawPosTable(wdl, dists, label){
  const W=document.getElementById('wdltxt');
  W.innerHTML=`<table><tr><th>${label}</th><th>white</th><th>draw</th><th>black</th></tr>`+
    `<tr><th>dist</th><td><b>${dists?dists[0].toFixed(1):"—"}</b></td>`+
    `<td><b>${dists?dists[1].toFixed(1):"—"}</b></td>`+
    `<td><b>${dists?dists[2].toFixed(1):"—"}</b></td></tr>`+
    `<tr><th>prob</th><td><b>${(wdl[0]*100).toFixed(1)}%</b></td>`+
    `<td><b>${(wdl[1]*100).toFixed(1)}%</b></td>`+
    `<td><b>${(wdl[2]*100).toFixed(1)}%</b></td></tr></table>`;
}
function renderLines(d){
  // turn-aware bar (Kaveh 2026-08-11: the static field is measurably TURN-BLIND,
  // sensitivity 0.000 on a mid-exchange probe): once analysis exists, the main bar
  // shows the committor at the END of the best line -- search resolves the exchange
  if(d.think&&d.think[0]&&d.think[0].wdl){
    const w=d.think[0].wdl;
    document.getElementById('eb-w').style.height=(w[0]*100)+"%";
    document.getElementById('eb-d').style.height=(w[1]*100)+"%";
    document.getElementById('eb-b').style.height=(w[2]*100)+"%";
    drawPosTable(w, d.think[0].dists, "best line");   // bar and table: ONE source
  }
  const lb=document.getElementById('lnbox'); lb.innerHTML="";
  (d.think||[]).forEach((row,i)=>{
    const div=document.createElement('div'); div.className="ln"+(i===0?" best":"");
    let ev="";
    if(row.tb){ev=`<span class="ev" title="tablebase / forced mate">${row.margin}</span>`;}
    else if(row.wdl){
      const E=(row.wdl[0]+0.5*row.wdl[1]);
      ev=`<span class="ev" title="expected points for white after this move (probability head)">E ${(E*100).toFixed(0)}%</span>`+
         "";
    } else {ev=`<span class="ev" title="searched margin (white POV)">${row.margin}</span>`;}
    let top=`<div class="lnr">`+ev;
    if(row.wdl){
      top+=`<span class="minibar"><i style="width:${row.wdl[0]*100}%;background:#f0efeb"></i>`+
           `<i style="width:${row.wdl[1]*100}%;background:#8b8680"></i>`+
           `<i style="width:${row.wdl[2]*100}%;background:#403d39"></i></span>`+
           `<span style="font-size:11px;color:#8f8a82">${Math.round(row.wdl[0]*100)}/${Math.round(row.wdl[1]*100)}/${Math.round(row.wdl[2]*100)}</span>`;
    }

    if(row.force&&row.force.drop>=0.7&&row.force.mono>=0.7)
      top+=`<span style="color:#8f8a82;font-size:11px" title="committal: distance to ${row.force.fav==='w'?'white':'black'}-win drops ${row.force.drop}/ply, monotone ${Math.round(row.force.mono*100)}%">↘${row.force.drop}</span>`;
    else if(row.force&&row.force.drop>=0.35)
      top+=`<span style="color:#8f8a82;font-size:11px" title="coherent progress: ${row.force.drop}/ply toward ${row.force.fav==='w'?'white':'black'}-win">→${row.force.drop}</span>`;
    top+=`<span class="mv">${row.line||row.uci}</span></div>`;
    let sub="";
    if(row.dists) sub+=`dW <b>${row.dists[0].toFixed(1)}</b> · dD <b>${row.dists[1].toFixed(1)}</b> · dB <b>${row.dists[2].toFixed(1)}</b>`;
    if(row.cnotes&&row.cnotes.length) sub+=(sub?" &nbsp;·&nbsp; ":"")+`<span style="color:#759900">→ ${row.cnotes.join(", ")}</span>`;
    if(sub) top+=`<div class="sub">${sub}</div>`;
    div.innerHTML=top;
    div.onclick=()=>api({action:"move",uci:row.uci});
    lb.appendChild(div);});
}
document.getElementById('new').onchange=e=>{
  if(e.target.value==='startpos'){api({action:'new'});e.target.value='';return;}
  const v=e.target.value; e.target.value="";
  api(v?{action:"load",fen:v}:{action:"new"});};
document.getElementById('first').onclick=()=>api({action:"goto",ptr:0});
document.getElementById('prev').onclick=()=>api({action:"rel",d:-1});
document.getElementById('next').onclick=()=>api({action:"rel",d:1});
document.getElementById('last').onclick=()=>api({action:"goto",ptr:99999});
document.getElementById('depth').onchange=()=>api({action:"noop"});
document.getElementById('plantog').onclick=()=>{planOn=!planOn;planLastFen=null;
  document.getElementById('plantog').style.color=planOn?'#7fbf5f':'#8f8a82';
  if(!planOn)document.getElementById('planbox').style.display='none';else api({action:"noop"});};
document.getElementById('nexttog').onclick=()=>{nextOn=!nextOn;nextLastFen=null;
  document.getElementById('nexttog').style.color=nextOn?'#7fbf5f':'#8f8a82';
  if(!nextOn)document.getElementById('nextbox').style.display='none';else api({action:"noop"});};
document.getElementById('sqtog').onclick=()=>{sqOn=!sqOn;sqLastFen=null;
  document.getElementById('sqtog').style.color=sqOn?'#7fbf5f':'#8f8a82';
  document.getElementById('sqfloorrow').style.display=sqOn?'':'none';
  if(!sqOn){document.getElementById('sqbox').style.display='none';cg.setShapes([]);}else api({action:"noop"});};
document.getElementById('sqfloor').oninput=renderSq;
document.getElementById('whytog').onclick=()=>{whyOn=!whyOn;whyLastFen=null;
  document.getElementById('whytog').style.color=whyOn?'#7fbf5f':'#8f8a82';
  if(!whyOn)cg.setShapes([]);else api({action:"noop"});};
document.getElementById('sftog').onclick=()=>{sfOn=!sfOn;sfLastFen=null;
  document.getElementById('sftog').style.color=sfOn?'#dbac16':'#8f8a82';api({action:"noop"});};
document.getElementById('flip').onclick=()=>{orient=orient==="white"?"black":"white";api({action:"noop"});};
document.getElementById('nav-an').onclick=()=>{
  setMode('analysis'); api({action:"noop"});};
document.getElementById('nav-play').onclick=()=>{
  if(!user)setUser();                    // per-player memory needs an identity
  surprHist=[];document.getElementById('surprbox').style.display='none';
  // one click, lichess-quick-pairing style: RANDOM side, fixed strength 6 (1.5s cap)
  playCfg={engineWhite:Math.random()<0.5, depth:6};
  orient=playCfg.engineWhite?'black':'white';
  setMode('play');
  cg.set({movable:{color:undefined}});           // freeze immediately: no moves during load
  document.getElementById('lnbox').innerHTML="";
  document.getElementById('dinfo').textContent="starting…";
  api({action:"new"});};
setMode('analysis');
addEventListener('keydown',e=>{
  if(e.key==="ArrowLeft"){e.preventDefault();api({action:"rel",d:-1});}
  if(e.key==="ArrowRight"){e.preventDefault();api({action:"rel",d:1});}});
api({action:"new"});   // fresh page = opening position (server state is shared/persistent)
</script>"""


class H(BaseHTTPRequestHandler):
    """Analysis-board state: ONE shared game line (moves) + a pointer (ptr).

    PURE analysis board (Kaveh 2026-08-08: "I don't want the engine to play") -- both sides
    free-movable at any point in the line; moving from mid-history truncates the future
    (lichess-simplified: no variation tree yet). Two-phase protocol: every action returns the
    position instantly (wdl included -- one forward); the 'analyze' action runs the search and
    returns the lines, so navigation never blocks on the engine."""
    eng = None
    lock = None                                          # engine/MPS is not thread-safe
    gen = 0                                              # bumped by every action: aborts searches
    an = {"g": -1, "rows": [], "depth": 0, "done": False}
    an_thread = None
    start_fen = None
    moves: list = []
    ptr = 0
    depth = 3
    lines = 3
    # per-player memory (Kaveh 2026-08-13): identity -> store; ponder -> surprise ruler
    user = None
    store = None                                         # PlayerStore of the active player
    ponder = None                                        # {"key","dist","budget"} human-to-move
    game_id = 0
    over_logged = -1
    last_req_t = 0.0
    banter = TemplateBanter()                            # replaced by --banter in main()
    banter_fast = TemplateBanter()                       # instant wording while the LLM types
    notes: dict = {}                                     # ply-index -> PGN annotation (emotes)
    emote_seq = 0
    emote_late = None                                    # {seq, text}: the LLM's line, async

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE_BYTES, "text/html; charset=utf-8")
        else:
            self._send(404, b"{}")

    def _board(self):
        b = chess.Board(H.start_fen) if H.start_fen else chess.Board()
        for m in H.moves[:H.ptr]:
            b.push(m)
        return b

    def _human_dist(self, bb, temp=1.0):
        """P(move) over bb's legal moves from the human layer (caller holds NO lock).
        temp>1 flattens (the stranger anneal: temp = tau/TAU_E)."""
        import numpy as _npH, torch as _tH2
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
            tokenize as _tkH)
        from catspace.research.components.encoder.approaches.reach_probability.experiments.train_human_moves import (
            mid_of, elo_bucket)
        legal = list(bb.legal_moves)
        if not legal:
            return {}
        prof = (H.store.profile.get("current") or {}) if H.store else {}
        eb = int(prof.get("elo_bucket") or elo_bucket(1500))
        tk, gl = _tkH(bb)
        with H.lock, _tH2.no_grad():
            phi = H.eng.net.backbone(
                _tH2.from_numpy(_npH.asarray([tk], dtype="int64")).to(H.eng.device),
                _tH2.from_numpy(_npH.asarray([gl], dtype="float32")).to(H.eng.device))
            mids = _tH2.tensor([mid_of(m) for m in legal], dtype=_tH2.long,
                               device=H.eng.device)
            row_of = _tH2.zeros(len(legal), dtype=_tH2.long, device=H.eng.device)
            eb_t = _tH2.tensor([eb], dtype=_tH2.long, device=H.eng.device)
            lg = H.human.move_logits(phi.float(), eb_t, mids, row_of)
            p = _tH2.softmax(lg / max(temp, 1e-3), 0).cpu().numpy()
        return {m.uci(): float(pv) for m, pv in zip(legal, p)}

    def _recent_san(self, n=6):
        """last n plies as a SAN string for banter context ('' if not reconstructible)."""
        try:
            sb = chess.Board(H.start_fen) if H.start_fen else chess.Board()
            sans = []
            for m in H.moves[:H.ptr]:
                sans.append(sb.san(m))
                sb.push(m)
            return " ".join(sans[-n:])
        except Exception:
            return ""

    def _start_ponder(self, bb):
        """PONDER (Kaveh 2026-08-13): while waiting for the human, keep searching THEIR
        position and hold a live expectation distribution over their legal moves -- the
        surprise ruler sharpens while they think. Aborts via gen; the lock is taken per
        rung so a real request preempts within one rung."""
        import threading
        g = H.gen
        key = fen_key(bb)
        legal = [m.uci() for m in bb.legal_moves]
        tau = H.store.tau() if H.store else TAU_E

        def w():
            for bud in (0.4, 1.0, 2.5):
                if H.gen != g:
                    return
                with H.lock:
                    if H.gen != g:
                        return
                    rows = H.eng.search_coherent(bb, budget=bud)
                    rows = H.eng.rank_by_child_E(bb, rows, top=24)   # exact E for the dist
                if rows:
                    rE = ranked_E(rows)
                    H.ponder = {"key": key, "budget": bud,
                                "dist": expectation_dist(rows, legal, tau=tau),
                                "e_top": sorted(rE.items(), key=lambda kv: -kv[1])[:8]}

        threading.Thread(target=w, daemon=True).start()

    def _log_game(self, b, san, over, play):
        try:
            json.dump({"san": san, "fen": b.fen(), "start_fen": H.start_fen,
                       "ptr": H.ptr, "mode": "play" if play else "analysis",
                       "over": over, "notes": H.notes,
                       "moves_uci": [m.uci() for m in H.moves]},
                      open(GAME_LOG, "w"))
        except Exception:
            pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        cls = H
        H.last_req_t = time.time()
        u = str(req.get("user") or "").strip()
        if u and H.user != u:                            # player identity -> per-player store
            try:
                H.store, H.user = PlayerStore(u), u
                print(f"[players] active player: {H.store.name}", flush=True)
            except Exception:
                H.store = None
        cls.depth = max(1, min(6, int(req.get("depth", cls.depth))))
        cls.lines = max(1, min(5, int(req.get("lines", cls.lines))))
        cls.pvlen = 8                       # pv maxed out (Kaveh 2026-08-12), selector gone
        act = req.get("action", "noop")
        human_ctx = None                    # (board-before, move) when the human just moved
        if act not in ("analyze", "lines"):
            H.gen += 1                                   # cancel any in-flight search NOW
        if act == "new":
            cls.moves, cls.ptr, cls.start_fen, cls.notes = [], 0, None, {}
            cls.game_id += 1
        elif act == "load" and req.get("fen"):
            try:
                chess.Board(req["fen"])
                cls.moves, cls.ptr, cls.start_fen, cls.notes = [], 0, req["fen"], {}
            except Exception:
                pass
        elif act == "goto":
            cls.ptr = max(0, min(len(cls.moves), int(req.get("ptr", 0))))
        elif act == "rel":
            cls.ptr = max(0, min(len(cls.moves), cls.ptr + int(req.get("d", 0))))
        elif act == "move" and req.get("uci"):
            b = self._board()
            try:
                mv = chess.Move.from_uci(req["uci"])
                if mv not in b.legal_moves:            # chessground sends o+t; default-promote
                    mv = chess.Move.from_uci(req["uci"] + "q")
                if mv in b.legal_moves:
                    human_ctx = (b, mv)                  # pre-move board: the surprise ruler
                    cls.moves = cls.moves[:cls.ptr] + [mv]
                    cls.notes = {k: v for k, v in cls.notes.items()
                                 if int(k) < cls.ptr}    # truncated future loses its notes
                    cls.ptr += 1
            except Exception:
                pass
        b = self._board()
        over = self._over(b)
        # PLAY MODE (Kaveh 2026-08-11, lichess-style menu): the engine answers synchronously
        # when it is to move; analysis assistance is hidden client-side during play
        surprise = emote = None
        prepared = False
        if req.get("play") and act in ("move", "new") and not over \
                and cls.ptr == len(cls.moves):
            eng_white = bool(req.get("engineWhite"))
            if (b.turn == chess.WHITE) == eng_white:
                # ---- SURPRISE PHASE (Kaveh 2026-08-13): the human just moved; score the
                # move against the pondered expectation P(move) = softmax(E_mover/tau).
                if human_ctx is not None:
                    pb, hmv = human_ctx
                    key = fen_key(pb)
                    legal = [m.uci() for m in pb.legal_moves]
                    tau = H.store.tau() if H.store else TAU_E
                    pond = e_top = None
                    if H.human is not None:
                        # expectation = HUMANS, not the engine's own taste; the stranger
                        # anneal flattens via temperature tau/TAU_E (1.0 when settled)
                        dist = self._human_dist(pb, temp=tau / TAU_E)
                        pond = {"src": "human"}
                    else:
                        pond = H.ponder if (H.ponder and H.ponder.get("key") == key) \
                            else None
                        dist = pond["dist"] if pond else None
                        e_top = pond["e_top"] if pond else None
                        if dist is None and H.store is not None:  # idle-time study counts
                            pe = H.store.get_prep(key)
                            if pe and pe.get("dist"):
                                dist = dict(pe["dist"])
                                e_top = pe.get("e_top")
                        if dist is None:                          # moved too fast: quick look
                            with H.lock:
                                rws = H.eng.search_coherent(pb, budget=0.5)
                                rws = H.eng.rank_by_child_E(pb, rws)
                            rE = ranked_E(rws)
                            dist = expectation_dist(rws, legal, tau=tau)
                            e_top = sorted(rE.items(), key=lambda kv: -kv[1])[:8]
                    bits = surprisal_bits(dist, hmv.uci())
                    xbits = bits - entropy_bits(dist)   # EXCESS: vs the position's own
                    san_h = pb.san(hmv)                 # expected surprisal (near-flat E)
                    with H.lock:                                  # one-forward dE, mover POV
                        e0w, _ = self._wdl(pb, None)
                        e1w, _ = self._wdl(b, over)
                    e0 = e0w[0] + 0.5 * e0w[1]
                    e1 = e1w[0] + 0.5 * e1w[1]
                    dE = (e1 - e0) if pb.turn == chess.WHITE else (e0 - e1)
                    # EMOTES (Kaveh 2026-08-13, engine-side only for now): gotcha checks
                    # BEFORE this surprise is recorded, so it always refers to a PAST visit
                    gr = H.store.check_gotcha(key, hmv.uci(), xbits) if H.store else None
                    ev = None
                    if gr:
                        ev = {"kind": "gotcha", "san": san_h, "bits": xbits}
                    elif xbits >= SURPRISE_BITS:
                        ev = {"kind": "surprised", "san": san_h, "bits": xbits}
                        if H.store:
                            H.store.note_surprise(key, hmv.uci(), xbits, san_h)
                    if ev is not None:
                        # ASYNC TRASH TALK (Kaveh 2026-08-13: 'make the move first, then
                        # give the trash talk later'): the DECISION and a template line ship
                        # with the move instantly; the LLM's wording lands via the 'emote'
                        # poll and upgrades the bubble + the PGN note in place.
                        fast = H.banter_fast.speak({**ev, "player": H.user})
                        H.emote_seq += 1
                        sid = H.emote_seq
                        llm = H.banter is not H.banter_fast and hasattr(H.banter, "warm")
                        emote = {"who": "engine", **ev, "bits": round(xbits, 2),
                                 **fast, "seq": sid, "pending": bool(llm)}
                        # PGN ANNOTATION (Kaveh 2026-08-13 'reactions should show up as pgn
                        # annotations :D'): a surprising GOOD move earns !?, a surprising
                        # E-loser ?!; gotcha keeps the glyphless smirk
                        glyph = "" if ev["kind"] == "gotcha" else \
                            ("!?" if dE >= -0.02 else "?!")
                        nidx = str(cls.ptr - 1)
                        cls.notes[nidx] = {"glyph": glyph, "emoji": fast["emoji"],
                                           "text": fast["text"], "kind": ev["kind"]}
                        if llm:
                            import threading as _th2
                            ev2 = {**ev, "player": H.user, "moves": self._recent_san(6)}

                            def _talk(sid=sid, nidx=nidx, ev2=ev2):
                                sp = H.banter.speak(ev2)
                                H.emote_late = {"seq": sid, "text": sp["text"],
                                                "emoji": sp["emoji"]}
                                if nidx in H.notes:      # trash talk lands in the PGN too
                                    H.notes[nidx] = {**H.notes[nidx], "text": sp["text"],
                                                     "emoji": sp["emoji"]}

                            _th2.Thread(target=_talk, daemon=True).start()
                    exp_uci = max(dist, key=dist.get)
                    try:
                        exp_san = pb.san(chess.Move.from_uci(exp_uci))
                    except Exception:
                        exp_san = exp_uci
                    surprise = {"bits": round(bits, 2), "xbits": round(xbits, 2),
                                "H": round(bits - xbits, 2), "tau": round(tau, 3),
                                "expected": exp_san,
                                "p": round(dist.get(hmv.uci(), 0.0), 3),
                                "pondered": bool(pond)}
                    if H.store:
                        H.store.log_ply({
                            "type": "ply", "game": cls.game_id, "mover": "human",
                            "fen_before": pb.fen(), "uci": hmv.uci(), "san": san_h,
                            "bits": round(bits, 2), "xbits": round(xbits, 2),
                            "dE_mover": round(dE, 4),
                            "pondered": bool(pond), "n_legal": len(legal),
                            "e_top": [(u, round(e, 4)) for u, e in (e_top or [])],
                            "dist_top": sorted(dist.items(), key=lambda kv: -kv[1])[:5]})
                # ---- REPLY PHASE: a prepped line answers instantly and deeper than live
                eb_before = b
                reply_mv = None
                pe = H.store.get_prep(fen_key(b)) if H.store else None
                if pe and pe.get("uci"):
                    try:
                        cand = chess.Move.from_uci(pe["uci"])
                        if cand in b.legal_moves:
                            # PREP SANITY GATE (Kaveh 2026-08-13: 'we replied from prep the
                            # same exact moves that led to a bad position'): the book is
                            # MEMORY, not judgment -- one-forward committor read of the
                            # child before trusting it. Distrust if the position has
                            # deteriorated vs what the study recorded, or is losing.
                            b.push(cand)
                            with H.lock:
                                prw, _ = self._wdl(b, self._over(b))
                            b.pop()
                            e_child = prw[0] + 0.5 * prw[1]
                            e_child = e_child if b.turn == chess.WHITE else 1.0 - e_child
                            e_book = pe.get("E")
                            if e_child >= 0.42 and (e_book is None
                                                    or e_child >= float(e_book) - 0.12):
                                reply_mv, prepared = cand, True
                            else:
                                self.__class__ and H.store.put_prep(
                                    fen_key(b), {**pe, "uci": None,
                                                 "distrusted": round(e_child, 3)})
                    except Exception:
                        pass
                if reply_mv is None:
                    # time-capped iterative deepening (Kaveh: think <= ~1.5s)
                    with H.lock:                 # BEST VERSION ONLY (Kaveh 2026-08-11):
                        best = H.eng.search_coherent(b, budget=1.5)   # coherence-bounded
                        best = H.eng.rank_by_child_E(b, best)         # ranked by E
                    if best:
                        reply_mv = best[0]["mv"]
                if reply_mv is not None:
                    cls.moves.append(reply_mv); cls.ptr += 1
                    b = self._board(); over = self._over(b)
                    if H.store:
                        H.store.log_ply({
                            "type": "ply", "game": cls.game_id, "mover": "engine",
                            "fen_before": eb_before.fen(), "uci": reply_mv.uci(),
                            "san": eb_before.san(reply_mv), "prepared": prepared})
                    if not over:
                        self._start_ponder(b)    # think on the HUMAN's clock
        if req.get("play") and over and H.store is not None \
                and H.over_logged != cls.game_id:
            H.over_logged = cls.game_id
            H.store.log_ply({"type": "game_end", "game": cls.game_id, "over": over,
                             "n_plies": len(cls.moves)})
            try:
                H.store.aggregate()
            except Exception:
                pass
        # slow phase, requested separately so navigation never waits on the search --
        # the analysis engine IS this engine (Kaveh 2026-08-08), depth + lines are the knobs
        # STREAMING analysis, lichess-style (Kaveh 2026-08-08 "the analysis streams in as the
        # search progresses"): 'analyze' spawns an iterative-deepening worker that publishes
        # partial lines after every root move and every depth; 'lines' polls the latest.
        if act == "analyze":
            if not over and not (H.an_thread and H.an_thread.is_alive()
                                 and H.an["g"] == H.gen):
                import threading
                g, depth, nlines = H.gen, cls.depth, cls.lines
                bb = b.copy()

                def worker():
                    merged = {}       # uci -> best-known display row (Kaveh 2026-08-12:
                                      # 'the list shouldn't lose anything it's already found'
                                      # -- deepening ACCUMULATES; new depths overwrite the
                                      # same move, never evict siblings found earlier)
                    with H.lock:      # UNLOCKED MODEL CALL = the MPS wedge (2026-08-12):
                        cmoves, corder, _ = H.eng.cascade_rank(bb)   # two threads in MPS
                                      # deadlock the GPU runtime at 0%% CPU; every engine
                                      # touch in this server goes through H.lock, no exceptions
                    crank = {cmoves[j].uci(): pos for pos, j in enumerate(corder)} if cmoves else {}

                    def _concept_notes(rows_disp):
                        if H.dyn is None or H.vq is None or not rows_disp:
                            return
                        import numpy as _np, torch as _torch
                        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
                            tokenize as _tok, move_ids as _mid)
                        tk, gl = _tok(bb)
                        with _torch.no_grad():
                            phi = H.eng.net.backbone(
                                _torch.from_numpy(_np.asarray([tk], dtype="int64")).to(H.eng.device),
                                _torch.from_numpy(_np.asarray([gl], dtype="float32")).to(H.eng.device))
                            _, pids, _ = H.vq(phi)
                            mids = _torch.from_numpy(_np.array(
                                [_mid(r["mv"]) for r in rows_disp], dtype="int64")).to(H.eng.device)
                            out = H.dyn(phi.expand(len(mids), -1), mids)
                            logits = out[0] if isinstance(out, tuple) else out
                            pred = logits.argmax(-1).cpu().numpy()
                        par = pids[0].cpu().numpy()
                        for i, r in enumerate(rows_disp):
                            named, raw = [], []
                            for h in range(len(par)):
                                if pred[i][h] != par[h]:
                                    nms = H.code_names.get((h, int(pred[i][h])))
                                    (named.append(nms[0]) if nms
                                     else raw.append(f"h{h}:c{int(pred[i][h])}"))
                            r["cnotes"] = (named + raw)[:3]

                    def publish(rows, d):
                        if H.gen == g:
                            # CASCADE RANKING (Kaveh 2026-08-11): display order = the cascade's
                            # order; the searched value rides along per line
                            # SEARCH RANKS, cascade breaks ties (2026-08-11 second burial:
                            # after 28.Ke1 the cascade top-ranked a rook blunder the search
                            # scored -11 vs +3, and the bar followed it to 'white winning'
                            # with mate-in-3 on the board). Values bucketed to 2.0 so the
                            # cascade still decides among search-indifferent siblings.
                            disp = self._rows_to_display(bb, rows, wdl_top=len(rows))
                            _wtm = bb.turn
                            disp.sort(key=lambda r: (
                                0 if r.get("tb") else 1,
                                -(r.get("E", 0) if _wtm else 1 - r.get("E", 1))))
                            disp = disp[:nlines]
                            try:
                                _concept_notes(disp)
                            except Exception:
                                pass
                            for r in disp:
                                merged[r["uci"]] = {**{k: r.get(k) for k in
                                    ("uci", "margin", "tb", "line", "wdl", "dists",
                                     "force", "eff_replies", "cnotes", "E")}, "_d": d}
                            shown = sorted(merged.values(), key=lambda r: (
                                0 if r.get("tb") else 1,
                                -((r.get("E") or 0) if _wtm else 1 - (r.get("E") or 1)),
                                -r["_d"]))[:nlines]
                            import numpy as _np8
                            _vs = _np8.array([r0["value"] for r0 in rows], float)
                            _sc = (_vs - _vs.max()) / 30.0        # 0.03 E per unit
                            _pr = _np8.exp(_sc); _pr /= _pr.sum()
                            _eff = float(_np8.exp(-(_pr * _np8.log(_pr + 1e-12)).sum()))
                            H.an = {**H.an, "g": g, "depth": d,
                                    "eff_moves": round(_eff, 1),
                                    "rows": [{k: v for k, v in r.items()
                                              if not k.startswith("_")} for r in shown],
                                    "done": False}
                    try:
                        with H.lock:
                            # Bellman-residual gate (Kaveh 2026-08-08): a self-contradictory
                            # field here earns one extra ply; the residual is shown in the UI.
                            resid = H.eng.bellman_residual(bb)
                            top = depth + (1 if resid is not None and abs(resid) >= 2.0 else 0)
                            H.an = {**H.an, "resid": resid, "top": top}
                        # COHERENCE-BOUNDED analysis (Kaveh 2026-08-12 'can you have
                        # coherence based search?'): the depth knob maps to a BUDGET ladder;
                        # each rung re-searches with more time, expansion allocated by
                        # P(reach) -- forcing lines run deep, frayed lines stay shallow, and
                        # the per-line depth SHOWS the coherence. Lock per rung.
                        _lad = {1: [0.4], 2: [0.4, 1.0], 3: [0.4, 1.5],
                                4: [0.5, 1.5, 3.0], 5: [0.5, 2.0, 5.0],
                                6: [0.5, 2.0, 5.0, 10.0]}[max(1, min(6, depth))]
                        for bud in _lad:
                            if H.gen != g:
                                return
                            with H.lock:
                                rows = H.eng.search_coherent(bb, budget=bud)
                                rows = H.eng.rank_by_child_E(bb, rows)
                            if rows:
                                dmax = max(r.get("depth_used", 1) for r in rows)
                                publish(rows, dmax)
                    finally:
                        if H.gen == g:
                            H.an = {**H.an, "done": True}

                H.an = {"g": g, "depth": 0, "rows": [], "done": False}
                H.an_thread = threading.Thread(target=worker, daemon=True)
                H.an_thread.start()
            self._send(200, json.dumps({"started": True, "over": over}).encode())
            return
        if act == "futures":
            # SEQUENCE-LAYER futures (Kaveh 2026-08-13: "prediction of future likely
            # concepts for my opponent"): causal transformer over THIS game's concept
            # stream -> P(newly activates within K plies), both sides, names attached.
            if H.seq is None:
                self._send(200, json.dumps(
                    {"err": "no sequence model yet (trains on jqt4 after it gates)"}).encode())
                return
            try:
                import numpy as _np9, torch as _t9
                from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
                    tokenize as _tk9)
                boards, sb9 = [], (chess.Board(cls.start_fen) if cls.start_fen
                                   else chess.Board())
                boards.append(sb9.copy())
                for m in cls.moves[:cls.ptr]:
                    sb9.push(m)
                    boards.append(sb9.copy())
                boards = boards[-H.seq.max_len:]
                tks, gls, stm9 = [], [], []
                for bb9 in boards:
                    tk9, gl9 = _tk9(bb9)
                    tks.append(tk9); gls.append(gl9)
                    stm9.append(0 if bb9.turn else 1)
                with H.lock, _t9.no_grad():
                    phi9 = H.eng.net.backbone(
                        _t9.from_numpy(_np9.asarray(tks, dtype="int64")).to(H.eng.device),
                        _t9.from_numpy(_np9.asarray(gls, dtype="float32")).to(H.eng.device))
                    _, ids9 = H.gq.jqt.target_codes(phi9)
                    ids9 = ids9.cpu()
                    act9, wdl9 = H.seq(ids9[None].long(),
                                       _t9.tensor(stm9, dtype=_t9.long)[None])
                    p9 = _t9.sigmoid(act9[0, -1])              # (H,C,K) at the current ply
                    pw9 = _t9.softmax(wdl9[0, -1], -1).tolist()
                cur9 = ids9[-1].numpy()
                Ksel = 1                                       # K=6 plies: "this phase"
                rows9 = []
                for h in range(p9.shape[0]):
                    for c in range(p9.shape[1]):
                        if int(cur9[h]) == c:
                            continue                           # already held: persistence
                        pv = float(p9[h, c, Ksel])
                        if pv < 0.10:
                            continue
                        lev9 = (float(H.lev[h, c]) if H.lev is not None
                                and h < H.lev.shape[0] and c < H.lev.shape[1] else 0.0)
                        nm = ((H.code_names or {}).get((h, c)) or [f"h{h}/c{c}"])[0]
                        rows9.append({"name": nm, "h": h, "c": c, "p": round(pv, 3),
                                      "p2": round(float(p9[h, c, 0]), 3),
                                      "side": "white" if lev9 > 0 else
                                              "black" if lev9 < 0 else "either",
                                      "lev": round(lev9, 3)})
                rows9.sort(key=lambda r: -r["p"])
                b9 = self._board()
                self._send(200, json.dumps(
                    {"mover": "white" if b9.turn else "black",
                     "wdl_seq": [round(x, 3) for x in pw9],
                     "rows": rows9[:14]}).encode())
            except Exception as e:
                self._send(200, json.dumps({"err": str(e)[:120]}).encode())
            return
        if act == "emote":                               # async trash-talk poll (play mode)
            late = H.emote_late
            if late and late.get("seq") == int(req.get("seq", -1)):
                self._send(200, json.dumps(late).encode())
            else:
                self._send(200, b"{}")
            return
        if act == "lines":
            cur = H.an if H.an.get("g") == H.gen else {"rows": [], "depth": 0, "done": False}
            self._send(200, json.dumps({"think": cur.get("rows", []),
                                        "depth": cur.get("depth", 0),
                                        "resid": cur.get("resid"),
                                        "done": bool(cur.get("done"))}).encode())
            return
        # fast phase: position + tricolor committor bar [P(white), P(draw), P(black)]
        with H.lock:
            wdl, dists = self._wdl(b, over)
            try:
                xr = H.eng.exit_readout(b) if not over else None
            except Exception:
                xr = None
        turb = None
        trap = None
        if not over:
            # ONE-PLY TERMINAL TRUTH (Kaveh 2026-08-11 stalemate trap: dD read 31.3 with
            # stalemate-in-1 twelve ways): terminal children are EXACT, like the tablebase --
            # clamp displayed distances and flag the trap. Winners never stalemate in the
            # training corpus, so the field cannot know these edges.
            n_stale = n_mate = 0
            for mv in b.legal_moves:
                b.push(mv)
                if b.is_game_over(claim_draw=True):
                    o = b.outcome(claim_draw=True)
                    if o.winner is None:
                        n_stale += 1
                    else:
                        n_mate += 1
                b.pop()
            if dists is not None and n_stale:
                dists = [dists[0], min(dists[1], 1.0), dists[2]]
            if dists is not None and n_mate:
                k = 0 if b.turn == chess.WHITE else 2
                dists = [min(dists[0], 1.0) if k == 0 else dists[0], dists[1],
                         min(dists[2], 1.0) if k == 2 else dists[2]]
            if n_stale or n_mate:
                trap = {"stalemate": n_stale, "mate": n_mate}
        if not over and not req.get("play"):
            try:
                with H.lock:
                    turb = H.eng.turbulence(b)
            except Exception:
                pass
        concepts, tokens = [], []
        if H.vq is not None and not over and not req.get("play"):
            import torch as _torch
            from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
                tokenize as _tok)
            from catspace.research.components.encoder.approaches.reach_probability.experiments.concept_vq import (
                predicates_from_tok)
            tk, gl = _tok(b)
            with H.lock, _torch.no_grad():
                phi = H.eng.net.backbone(
                    _torch.from_numpy(__import__("numpy").asarray([tk], dtype="int64")).to(H.eng.device),
                    _torch.from_numpy(__import__("numpy").asarray([gl], dtype="float32")).to(H.eng.device))
                _, ids, _ = H.vq(phi)
            tokens = [int(x) for x in ids[0].cpu()]
            preds = predicates_from_tok([tk])
            for name, info in H.cmap.items():
                active = bool(preds[name][0]) if name in preds else None
                tok_here = tokens[info["head"]] == info["code"]
                concepts.append({"name": name, "active": active,
                                 "head": info["head"], "code": info["code"],
                                 "p": info["p_given_code"], "base": info["base"],
                                 "anti": bool(info.get("anti")),
                                 "tok_here": bool(tok_here)})
        if act == "saliency":
            try:
                import numpy as _np6, torch as _t6
                from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
                    tokenize as _tk6)
                tk, gl = _tk6(b)
                tk = _np6.asarray(tk); gl = _np6.asarray(gl)
                occ = _np6.flatnonzero(tk > 0)
                toks = [tk] + [_np6.where(_np6.arange(64) == q, 0, tk) for q in occ]
                globs = [gl] * len(toks)
                with H.lock, _t6.no_grad():
                    wds = []
                    for a0 in range(0, len(toks), 128):
                        z = H.eng._embed(toks[a0:a0+128], globs[a0:a0+128]).float()
                        P3s = H.eng.poles[[H.eng.pi[k] for k in ("WIN","DRAW","LOSS")]].to(H.eng.device)
                        DB = _t6.stack([H.eng.net.dB(z, P3s[[k]].expand(len(z), -1))
                                        for k in range(3)], 1)
                        pr = _t6.softmax(-DB / 5.0, 1).cpu().numpy()
                        wds.append(pr[:, 0] + 0.5 * pr[:, 1])
                    E = _np6.concatenate(wds)
                base_E = float(E[0])
                VALS = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 4, 7: 1, 8: 3, 9: 3, 10: 5,
                        11: 9, 12: 4}
                raw = []
                for i, q in enumerate(occ):
                    t = int(tk[int(q)])
                    own = base_E - E[1 + i]              # >0: piece helps WHITE
                    pov = own if t <= 6 else -own        # >0: piece helps ITS OWNER
                    raw.append((int(q), t, pov))
                # MATERIAL RESIDUAL (Kaveh 2026-08-12 'highlights all pieces, but why':
                # removal dE ~ piece value, so raw saliency just re-discovers material.
                # Fit dE ~ s*value per color; the RESIDUAL is positional importance:
                # load-bearing (doing more than its face value) vs bystander vs misplaced
                # (removal HELPS its owner).)
                import numpy as _np7
                out = []
                for color_white in (True, False):
                    grp = [(q, t, pov) for q, t, pov in raw
                           if (t <= 6) == color_white and (t % 6) != 0]   # kings excluded
                    if not grp:
                        continue
                    vals = _np7.array([VALS[t] for _q, t, _p in grp], float)
                    povs = _np7.array([pv for _q, _t, pv in grp], float)
                    sc = float(_np7.median(povs / vals)) if len(grp) else 0.0
                    for (q, t, pov), v in zip(grp, vals):
                        out.append({"sq": chess.square_name(q),
                                    "excess": round(float(pov - sc * v), 4),
                                    "dE": round(float(pov), 4)})
                self._send(200, json.dumps({"base": round(base_E, 3), "sal": out}).encode())
            except Exception as e:
                self._send(200, json.dumps({"err": str(e)[:80]}).encode())
            return
        if act == "plan":
            try:
                import numpy as _np11, torch as _t11
                from catspace.research.components.planner.approaches.quasimetric_nav.subgoal_former import (
                    alert_set)
                if H.former is None or H.gq is None:
                    self._send(200, json.dumps({"err": "no trained planner substrate"}).encode())
                    return
                hc = H.gq.candidates(k_lev=12)
                sides_t = _t11.zeros(len(hc), dtype=_t11.long)
                hct = _t11.as_tensor(hc)
                with H.lock:
                    G_, F_ = H.gq.geometry(b, hc)
                # committed = highest p-hat among concepts serving the MOVER
                with _t11.no_grad():
                    p_all, _ = H.former(hct, sides_t, F_, G_)
                mover_white = b.turn
                lev_row = _np11.array([H.gq.lev[h2, c2] if H.gq.lev is not None else 0.0
                                       for h2, c2 in hc])
                serve = (lev_row > 0.005) if mover_white else (lev_row < -0.005)
                pa = p_all.numpy()
                cand_idx = _np11.flatnonzero(serve)
                ci = int(cand_idx[_np11.argmax(pa[cand_idx])]) if len(cand_idx) else int(pa.argmax())
                cert = H.former.certificate(hct, sides_t, F_, G_, committed_idx=ci)
                prev = H.plan_prev.get("cert")
                alerts = alert_set(prev, cert, F_, k=8, lev=H.gq.lev) if prev is not None else []
                H.plan_prev["cert"] = cert
                nm = lambda h2, c2: ((H.code_names or {}).get((h2, c2)) or [f"h{h2}/c{c2}"])[0]
                worries = [{"name": nm(*cert.hc[i]), "dp": round(float(cert.worry[i]), 3),
                            "attn": round(float(cert.attn[i]), 2)}
                           for i in _np11.argsort(-cert.worry)[:4] if cert.worry[i] > 0.003]
                act_out = None
                if H.pointer is not None:
                    a, bud, _lp = H.pointer.act(cert, alerts, greedy=True)
                    act_out = {"action": ("HOLD plan" if a >= len(alerts) else
                                          f"{'pursue' if alerts[a].kind=='opportunity' else 'deny'} "
                                          f"{nm(*alerts[a].hc)}"),
                               "budget": bud}
                out = {"committed": nm(*cert.hc[cert.committed]),
                       "p_hat": round(cert.p_hat, 3),
                       "premove_safe": bool(cert.premove_safe()),
                       "worries": worries,
                       "alerts": [{"kind": a2.kind, "name": nm(*a2.hc),
                                   "dp": round(a2.d_p, 3)} for a2 in alerts[:5]],
                       "policy": act_out}
                self._send(200, json.dumps(out).encode())
            except Exception as e:
                self._send(200, json.dumps({"err": str(e)[:100]}).encode())
            return
        if act == "sqconcepts":
            try:
                import numpy as _np10, torch as _t10
                jm = H.gq.jqt if H.gq is not None else None
                if jm is None or not getattr(jm, "square_codes", 0):
                    self._send(200, json.dumps({"err": "no square stream in this champion"}).encode())
                    return
                from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
                    tokenize as _tk10)
                tk, gl = _tk10(b)
                with H.lock, _t10.no_grad():
                    tok_t = _t10.from_numpy(_np10.asarray([tk], dtype="int64")).to(H.eng.device)
                    gl_t = _t10.from_numpy(_np10.asarray([gl], dtype="float32")).to(H.eng.device)
                    _phi, sqt = H.eng.net.enc.forward_tokens(tok_t, gl_t)
                    hh = jm.sq_proj(sqt)
                    qq, ids, _vl = jm.sq_vq(hh)
                    contrib = jm.sq_dec(qq)[0]          # (64, 6) normalized-space additive
                # white-E contribution per square: P(W) minus P(B) parts, de-normalized
                cE = (contrib[:, 3] * jm.y_sd[3] - contrib[:, 5] * jm.y_sd[5]).cpu().numpy()
                ids = ids[0].cpu().numpy()
                order = _np10.argsort(-_np10.abs(cE))[:12]
                out = [{"sq": chess.square_name(int(q)), "code": int(ids[q]),
                        "imp": round(float(cE[q]), 4)} for q in order]
                self._send(200, json.dumps({"sal": out}).encode())
            except Exception as e:
                self._send(200, json.dumps({"err": str(e)[:80]}).encode())
            return
        if act == "sfeval":
            try:
                import chess.engine as _che
                if getattr(H, "sf", None) is None:
                    H.sf = _che.SimpleEngine.popen_uci("stockfish")
                    from catspace.io import paths as _paths
                    H.sf.configure({"Threads": 1, "Hash": 128, "UCI_ShowWDL": True,
                                    "SyzygyPath": str(_paths.syzygy_dir())})
                    import threading as _th
                    H.sf_lock = _th.Lock()
                with H.sf_lock:
                    info = H.sf.analyse(b, _che.Limit(nodes=200_000))
                sc = info["score"].white()
                out = {"cp": (sc.score() if not sc.is_mate() else None),
                       "mate": (sc.mate() if sc.is_mate() else None)}
                if "wdl" in info:
                    out["wdl"] = list(info["wdl"].white())
                pvm = info.get("pv") or []
                bb2 = b.copy(); sans2 = []
                for mv2 in pvm[:6]:
                    sans2.append(bb2.san(mv2)); bb2.push(mv2)
                out["pv"] = " ".join(sans2)
                self._send(200, json.dumps(out).encode())
            except Exception as e:
                self._send(200, json.dumps({"err": str(e)[:80]}).encode())
            return
        cdeets = []
        if H.vq is not None and not over and not req.get("play") and tokens:
            try:
                import numpy as _np3, torch as _torch3
                gates_by_node = {}
                node_of = {}
                if H.graph:
                    for i, nd in enumerate(H.graph["nodes"]):
                        node_of[(nd["h"], nd["c"])] = i
                    for e in H.graph["edges"]:
                        if e["kind"] != "gateway":
                            continue
                        a, bb = (e["a"], e["b"]) if e["dir"] > 0 else (e["b"], e["a"])
                        gates_by_node.setdefault(a, []).append(bb)
                active_hc = [(h, int(c)) for h, c in enumerate(tokens)]
                # POSITION-CONDITIONED candidates (Kaveh 2026-08-12 'why only a limited
                # set'): rank the WHOLE vocabulary by live P(activate) from this position,
                # then add the top-leverage staples. One batched pass over all 512 anchors.
                extra_hc = []
                if H.gq is not None:
                    import torch as _t5
                    with H.lock, _t5.no_grad():
                        allhc = _t5.tensor([(h, c) for h in range(H.gq.H)
                                            for c in range(H.gq.C)],
                                           dtype=_t5.long, device=H.gq.device)
                        A_all = H.gq.jqt.anchors_for(allhc).float()
                        z_us, _zo = H.gq.state_embed(b)
                        dB_all = H.eng.net.dB(z_us.expand(len(A_all), -1), A_all)
                        p_all = _t5.sigmoid(H.gq.jqt.activation_logit(dB_all))
                    topr = p_all.argsort(descending=True)[:14].cpu().numpy()
                    extra_hc = [(int(i // H.gq.C), int(i % H.gq.C)) for i in topr]
                if H.lev is not None:
                    fl = _np3.argsort(-_np3.abs(H.lev).ravel())[:6]
                    Cw = H.lev.shape[1]
                    extra_hc += [(int(i // Cw), int(i % Cw)) for i in fl]
                seen_hc, cand = set(active_hc), list(active_hc)
                for hc_ in extra_hc:
                    if hc_ not in seen_hc:
                        seen_hc.add(hc_); cand.append(hc_)
                pa = da = dao = None
                if H.gq is not None:
                    with H.lock:                    # NEVER touch the model unlocked: an
                        G_, F_ = H.gq.geometry(     # unlocked MPS call races the streaming
                            b, _np3.array(cand, _np3.int64))   # analysis thread (the hang)
                    da = F_[:, 0].tolist(); pa = F_[:, 2].tolist()
                    dao = F_[:, 1].tolist()         # opponent-to-move distance (the race)
                for k, (hh, cc) in enumerate(cand):
                    names = (H.code_names or {}).get((hh, cc), [])
                    gouts = []
                    ni = node_of.get((hh, cc))
                    if ni is not None:
                        for gj in (gates_by_node.get(ni) or [])[:3]:
                            gn = H.graph["nodes"][gj]
                            gnm = ((H.code_names or {}).get((gn["h"], gn["c"]))
                                   or [f"h{gn['h']}/c{gn['c']}"])[0]
                            gouts.append(gnm)
                    cdeets.append({
                        "h": hh, "c": cc, "name": (names or [f"h{hh}/c{cc}"])[0],
                        "active": (hh, cc) in active_hc and cand.index((hh, cc)) < len(tokens),
                        "lev": (float(H.lev[hh, cc]) if H.lev is not None
                                and hh < H.lev.shape[0] and cc < H.lev.shape[1] else 0.0),
                        "br": (float(H.br[hh, cc]) if H.br is not None
                               and hh < H.br.shape[0] and cc < H.br.shape[1] else None),
                        "p_act": (round(float(pa[k]), 3) if pa else None),
                        "dA": (round(float(da[k]), 2) if da else None),
                        "dA_opp": (round(float(dao[k]), 2) if dao else None),
                        "gates": gouts})
                # SORTED for the dropdown (Kaveh 2026-08-12): active concepts first
                # (by |leverage| -- highest-stakes on top), then candidates by live
                # P(activate) from this position, descending.
                cdeets.sort(key=lambda c: (0 if c["active"] else 1,
                                           -abs(c["lev"]) if c["active"]
                                           else -(c["p_act"] or 0.0)))
            except Exception:
                cdeets = []
        dests = {}
        if not over:                                    # free analysis: mover always movable
            for m in b.legal_moves:
                dests.setdefault(chess.square_name(m.from_square), []).append(
                    chess.square_name(m.to_square))
        last = None
        if cls.ptr:
            m = cls.moves[cls.ptr - 1]
            last = [chess.square_name(m.from_square), chess.square_name(m.to_square)]
        sb, san = chess.Board(), []
        for m in cls.moves:
            san.append(sb.san(m)); sb.push(m)
        self._log_game(b, san, over, bool(req.get("play")))
        out = {"fen": b.fen(), "turn": "white" if b.turn else "black",
               "depth": cls.depth, "san": san, "ptr": cls.ptr, "wdl": wdl,
               "dists": dists, "check": b.is_check(), "dests": dests, "exit": xr,
               "concepts": concepts, "tokens": tokens, "cdeets": cdeets,
               "turb": (round(turb[0], 3) if turb else None), "trap": trap,
               "lastMove": last, "over": over,
               "surprise": surprise, "emote": emote, "prepared": prepared,
               "player": H.user, "notes": cls.notes,
               "start_fen": cls.start_fen}
        self._send(200, json.dumps(out).encode())

    def _wdl(self, b, over):
        if over:
            o = b.outcome(claim_draw=True)
            return ([1.0, 0.0, 0.0] if o.winner is True
                    else [0.0, 0.0, 1.0] if o.winner is False else [0.0, 1.0, 0.0]), None
        return H.eng.wdl(b)                             # already white-POV

    def _over(self, b):
        if b.is_game_over(claim_draw=True):
            o = b.outcome(claim_draw=True)
            if o.winner is None:
                return f"draw ({o.termination.name.lower()})"
            return ("white" if o.winner else "black") + " wins"
        return None

    def _rows_to_display(self, b, rows, wdl_top=0):
        """WHITE-POV values, lichess-style (Kaveh 2026-08-08): the number shown is always
        white's margin (negated when black is to move), so it doesn't flip sign every ply.
        Search order (best-for-mover first) is kept: for black that IS lowest-white-value
        first, which is the requested sort."""
        out = []
        for r in rows[:18]:
            bb = b.copy()
            sans = []
            for mv in r["pv"][:2 * getattr(H, "pvlen", 6)]:
                sans.append(bb.san(mv)); bb.push(mv)
            v = r["value"] if b.turn == chess.WHITE else -r["value"]
            disp = (("#" if v > 0 else "-#") if abs(v) >= KittyMATE / 4
                    else round(v, 3) if abs(v) < 10 else round(v, 1))
            row = {"mv": r["mv"], "uci": r["mv"].uci(), "margin": disp,
                   "tb": abs(v) >= KittyMATE / 4, "deep": True,
                   "line": " ".join(sans)}
            # per-line committor (Kaveh 2026-08-11: wdl distances + probabilities per line):
            # the position AFTER the line's first move, one forward each, top rows only
            if len(out) < wdl_top:
                # evaluate at the END of the shown variation, not after the first move --
                # a capture's immediate child is mid-exchange (Kaveh 2026-08-11: Nxe5 read
                # as a won knight before the recapture existed)
                bc = b.copy()
                for mv in r["pv"][:2 * getattr(H, "pvlen", 6)]:
                    bc.push(mv)
                try:
                    row["wdl"], row["dists"] = H.eng.wdl(bc)
                    row["E"] = round(row["wdl"][0] + 0.5 * row["wdl"][1], 4)
                except Exception:
                    pass
                try:
                    coh = H.eng.line_coherence(b, r["pv"], max_plies=2 * getattr(H, "pvlen", 6))
                    if coh is not None:
                        row["force"] = {"drop": round(coh[0], 2), "mono": round(coh[1], 2),
                                        "fav": coh[2]}
                except Exception:
                    pass
                try:
                    # TRUE forcingness (Kaveh 2026-08-12, the 2.Bg5 complaint: descending is
                    # not FORCING): entropy of the opponent's plausible-reply distribution
                    # after the move. eff_replies = exp(H); <= ~2.5 = genuinely forcing.
                    import math as _m
                    b1 = b.copy(); b1.push(r["pv"][0])
                    reps = list(b1.legal_moves)
                    if reps:
                        kids = []
                        for mv2 in reps:
                            b1.push(mv2); kids.append(b1.copy(stack=False)); b1.pop()
                        mg = H.eng.margins(kids)         # child-mover POV; reply prefers max
                        import numpy as _np4
                        sc = _np4.array(mg) / 0.35
                        sc -= sc.max()
                        pr = _np4.exp(sc); pr /= pr.sum()
                        Hent = float(-(pr * _np4.log(pr + 1e-12)).sum())
                        row["eff_replies"] = round(float(_m.exp(Hent)), 1)
                except Exception:
                    pass
            out.append(row)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--cond-elo", type=float, default=None)
    ap.add_argument("--device", default="mps",
                    help="mps | cuda | cpu | auto (cuda -> mps -> cpu)")
    ap.add_argument("--banter", default="template",
                    help="emote voice: template | ollama | ollama:<model> "
                         "(free local LLM phrases the emotes; decision stays mechanical)")
    args = ap.parse_args()
    H.banter = make_banter(args.banter)
    if hasattr(H.banter, "warm"):
        import threading as _th
        _th.Thread(target=H.banter.warm, daemon=True).start()   # pay the model load NOW
    print(f"[kitty-server] banter: {args.banter}", flush=True)
    if args.device == "auto":
        import torch as _ta
        args.device = ("cuda" if _ta.cuda.is_available()
                       else "mps" if _ta.backends.mps.is_available() else "cpu")
        print(f"[kitty-server] device auto -> {args.device}", flush=True)

    import threading
    from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess
    H.eng = KittyChess(args.ckpt, args.device, args.cond_elo)
    global GAME_LOG
    GAME_LOG = str(paths.experiment("kitty_current_game.json"))
    try:                                             # restore the game across restarts
        st = json.load(open(GAME_LOG))
        H.start_fen = st.get("start_fen")
        H.moves = [chess.Move.from_uci(u) for u in st.get("moves_uci", [])]
        H.ptr = min(int(st.get("ptr", len(H.moves))), len(H.moves))
        H.notes = st.get("notes") or {}
        print(f"[kitty-server] restored game: {len(H.moves)} moves", flush=True)
    except Exception:
        pass
    # concept quantizer + known-concept map (Kaveh 2026-08-11): optional sidecars
    H.vq = H.cmap = None
    try:
        base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
        import json as _json, torch as _torch
        from catspace.research.components.encoder.approaches.reach_probability.experiments.concept_vq import (
            ConceptVQ)
        pv = _torch.load(base + "_vq.pt", map_location=args.device, weights_only=False)
        H.vq = ConceptVQ(d_in=pv["d_in"], heads=pv["heads"], codes=pv["codes"]).to(args.device)
        H.vq.load_state_dict(pv["state_dict"]); H.vq.eval()
        H.cmap = _json.load(open(base + "_conceptmap.json"))
        H.code_names = {}
        for nm, info in H.cmap.items():
            H.code_names.setdefault((info["head"], info["code"]), []).append(nm)
        print(f"[kitty-server] concept quantizer loaded ({pv['heads']}x{pv['codes']})", flush=True)
        try:
            from catspace.research.components.encoder.approaches.reach_probability.experiments.concept_dynamics import (
                ConceptDynamics)
            pdyn = _torch.load(base + "_dyn.pt", map_location=args.device, weights_only=False)
            H.dyn = ConceptDynamics(d_in=pdyn["d_in"], heads=pdyn["heads"],
                                    codes=pdyn["codes"],
                                    reply=pdyn.get("reply", False)).to(args.device)
            H.dyn.load_state_dict(pdyn["state_dict"]); H.dyn.eval()
            print("[kitty-server] concept dynamics loaded", flush=True)
        except Exception as e:
            H.dyn = None
            print(f"[kitty-server] no dynamics head ({e})", flush=True)
    except Exception as e:
        print(f"[kitty-server] no concept sidecars ({e})", flush=True)
    try:
        import re as _re0, os as _os0, json as _json0
        _stem0 = _re0.sub(r"_(latest|step\d+)$", "", base)
        for _c0 in (base, _stem0):
            if _os0.path.exists(_c0 + "_conceptmap.json") and not H.cmap:
                H.cmap = _json0.load(open(_c0 + "_conceptmap.json"))
                H.code_names = {}
                for nm0, info0 in H.cmap.items():
                    if not info0.get("anti"):
                        H.code_names.setdefault((info0["head"], info0["code"]), []).append(nm0)
                print(f"[kitty-server] concept names loaded: {len(H.cmap)}", flush=True)
    except Exception:
        pass
    # 2026-08-12: jqt2's decode trained under freeze-at-50 stats (dA dims ~untrained);
    # concept DISPLAY stays (code ids are valid), concept-mediated PLAY VALUES demoted to
    # field-direct until a properly-calibrated quantizer (jqt3) gates.
    H.eng.concept_eval = False
    if H.vq is None and getattr(H.eng, "cvq", None) is not None:
        # JQT-native quantizer (2026-08-12): the engine already consumes the _jqt sidecar;
        # the server reuses it -- same forward(phi) -> (y, ids, .) surface.
        H.vq = H.eng.cvq
        H.cmap = H.cmap or {}
        H.code_names = getattr(H, "code_names", {}) or {}
        print("[kitty-server] concept quantizer: jqt-native (engine sidecar)", flush=True)
    # per-position concept DETAILS substrate (Kaveh 2026-08-12: "concepts as it applies to
    # the position under analysis"): leverage, base rates, gateways, live reachability
    H.lev = H.br = H.graph = H.gq = None
    try:
        import re as _re, os as _os3, numpy as _np2, json as _json2
        stem = _re.sub(r"_(latest|step\d+)$", "", base)
        for cand in (base, stem):
            if _os3.path.exists(cand + "_concept_leverage.npz"):
                _lz = _np2.load(cand + "_concept_leverage.npz")
                dims = (int(_lz["head"].max()) + 1, int(_lz["code"].max()) + 1)
                H.lev = _np2.zeros((max(8, dims[0]), max(64, dims[1])), _np2.float32)
                for sw, hh, cc in zip(_lz["swing"], _lz["head"], _lz["code"]):
                    H.lev[int(hh), int(cc)] = float(sw)
            if _os3.path.exists(cand + "_code_baserates.npy"):
                H.br = _np2.load(cand + "_code_baserates.npy")
            if _os3.path.exists(cand + "_concept_graph.json"):
                H.graph = _json2.load(open(cand + "_concept_graph.json"))
        jqp = next((c + "_jqt.pt" for c in (base, stem)
                    if _os3.path.exists(c + "_jqt.pt")), None)
        if jqp:
            from catspace.research.components.planner.approaches.quasimetric_nav.subgoal_former import (
                GeoQuery)
            lvp = next((c + "_concept_leverage.npz" for c in (base, stem)
                        if _os3.path.exists(c + "_concept_leverage.npz")), None)
            H.gq = GeoQuery(H.eng, jqp, lvp, args.device)
        print(f"[kitty-server] concept details: lev={H.lev is not None} "
              f"br={H.br is not None} graph={H.graph is not None} gq={H.gq is not None}",
              flush=True)
    except Exception as e:
        print(f"[kitty-server] no concept-details substrate ({e})", flush=True)
    # HUMAN PREDICTION LAYER (Kaveh 2026-08-13): P(move|position,Elo) replaces the engine's
    # own E-softmax as the expectation over the human's moves wherever a sidecar exists
    H.human = None
    try:
        import os as _osH, re as _reH, torch as _tH
        from catspace.research.components.encoder.approaches.reach_probability.experiments.train_human_moves import (
            HumanMoves)
        _stemH = _reH.sub(r"_(latest|step\d+)$", "", base)
        for cH in (base, _stemH):
            if _osH.path.exists(cH + "_human.pt") and H.human is None:
                pH = _tH.load(cH + "_human.pt", map_location=args.device, weights_only=False)
                H.human = HumanMoves(d_in=pH["d_in"]).to(args.device)
                H.human.load_state_dict(pH["state_dict"])
                H.human.eval()
        print(f"[kitty-server] human layer: {H.human is not None}", flush=True)
    except Exception as e:
        print(f"[kitty-server] no human layer ({e})", flush=True)
    H.seq = None
    try:
        import os as _os9, re as _re9, torch as _t9b
        from catspace.research.components.encoder.approaches.reach_probability.experiments.concept_sequence import (
            ConceptSequence)
        _stem9 = _re9.sub(r"_(latest|step\d+)$", "", base)
        for c9 in (base, _stem9):
            if _os9.path.exists(c9 + "_seq.pt") and H.seq is None:
                pq = _t9b.load(c9 + "_seq.pt", map_location="cpu", weights_only=False)
                H.seq = ConceptSequence(heads=pq["heads"], codes=pq["codes"],
                                        d=pq.get("d", 128), layers=pq.get("layers", 3))
                H.seq.load_state_dict(pq["state_dict"])
                H.seq.eval()
        print(f"[kitty-server] sequence layer: {H.seq is not None}", flush=True)
    except Exception as e:
        print(f"[kitty-server] no sequence layer ({e})", flush=True)
    H.former = H.pointer = None
    H.plan_prev = {}
    try:
        import re as _re2, os as _os2, torch as _t2
        from catspace.research.components.planner.approaches.quasimetric_nav.subgoal_former import (
            SubgoalFormer)
        from catspace.research.components.planner.approaches.quasimetric_nav.pointer_policy import (
            PointerPolicy)
        _stem2 = _re2.sub(r"_(latest|step\d+)$", "", base)
        for c2 in (base, _stem2):
            fp = c2 + "_former.pt"
            if _os2.path.exists(fp) and H.former is None and H.gq is not None:
                H.former = SubgoalFormer(n_head=H.gq.H, n_code=H.gq.C)
                H.former.load_state_dict(_t2.load(fp, map_location="cpu"))
                H.former.eval()
            pp = c2 + "_pointer.pt"
            if _os2.path.exists(pp) and H.pointer is None:
                H.pointer = PointerPolicy()
                H.pointer.load_state_dict(_t2.load(pp, map_location="cpu"))
                H.pointer.eval()
        print(f"[kitty-server] planner: former={H.former is not None} "
              f"pointer={H.pointer is not None}", flush=True)
    except Exception as e:
        print(f"[kitty-server] no planner ({e})", flush=True)
    H.lock = threading.Lock()
    H.depth = args.depth
    global PAGE_BYTES
    cg_js = open(os.path.join(ASSETS, "bundle.js")).read()
    cg_css = open(os.path.join(ASSETS, "bundle.css")).read()
    # POSITION LIBRARY (Kaveh 2026-08-11): curated interesting starts -- forced-mate nets,
    # conversion tests, and live picks from the behavioral sanity suite
    pos = [("Standard start", "startpos"),
           ("KRK: convert the rook", "8/8/8/4k3/8/8/8/R3K3 w - - 0 1"),
           ("KQK: convert the queen", "8/8/8/4k3/8/8/3Q4/4K3 w - - 0 1"),
           ("KPK: promote", "8/8/8/8/4k3/8/4P3/4K3 w - - 0 1"),
           ("back-rank net (W wins)", "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"),
           ("7pc: Q+R vs R (convert)", "3rk3/8/8/8/8/2Q5/2R5/4K3 w - - 0 1")]
    try:
        import json as _json
        cats = {}
        for ln in open(paths.experiment("sanity_suite.jsonl")):
            r = _json.loads(ln)
            cats.setdefault(r["cat"], []).append(r["fen"])
        for cat in ("resist", "save-draw", "convert"):
            for i, fen in enumerate(cats.get(cat, [])[:3]):
                pos.append((f"suite {cat} #{i+1}", fen))
    except Exception:
        pass
    try:                                             # mined subgoal exemplars (invariant only)
        import json as _json
        n_sub = 0
        for ln in open(paths.experiment("subgoal_candidates.jsonl")):
            r = _json.loads(ln)
            if r.get("invariant") and r.get("fens") and n_sub < 6:
                n_sub += 1
                pos.append((f"subgoal c{r['cluster']} (E {r['E_mean']:.2f} d {r['dOwn_mean']:.0f})",
                            r["fens"][0]))
    except Exception:
        pass
    opts = "".join(f'<option value="{f}">{n}</option>' for n, f in pos)
    PAGE_BYTES = (PAGE.replace("__CG_CSS__", cg_css).replace("__CG_JS__", cg_js)
                  .replace("__POSITIONS__", opts).encode())
    # IDLE PREP WORKER (Kaveh 2026-08-13: "in this five, ten minutes, expected my return and
    # studied the lines we went down"): when nobody has touched the server for 90s and a
    # player is known, replay their logged positions -- systematic weaknesses first -- with
    # 4x the live budget and persist the results in their prep cache. Live play answers
    # from the cache instantly (and 'gotcha' becomes earnable: the reply is pre-studied).
    H.last_req_t = time.time()

    def prep_worker():
        while True:
            time.sleep(20)
            st = H.store
            if st is None or time.time() - H.last_req_t < 90:
                continue
            try:
                pend = st.pending_prep(limit=1)
                if not pend:
                    continue
                fen, mover = pend[0]
                bb = chess.Board(fen)
                with H.lock:
                    if time.time() - H.last_req_t < 90:   # someone came back: yield
                        continue
                    rows = H.eng.search_coherent(bb, budget=6.0)
                    rows = H.eng.rank_by_child_E(bb, rows)
                if not rows:
                    st.put_prep(fen_key(bb), {"uci": None, "budget": 6.0})
                    continue
                legal = [m.uci() for m in bb.legal_moves]
                rE = ranked_E(rows)
                dist = expectation_dist(rows, legal, tau=st.tau())
                st.put_prep(fen_key(bb), {
                    "uci": rows[0]["mv"].uci(), "E": rE.get(rows[0]["mv"].uci()),
                    "budget": 6.0, "mover": mover,
                    "e_top": sorted(rE.items(), key=lambda kv: -kv[1])[:8],
                    "dist": sorted(dist.items(), key=lambda kv: -kv[1])[:8]})
                prof = st.aggregate()
                # PER-PLAYER ELO FIT (Kaveh 2026-08-13, human layer): pick the Elo bucket
                # whose conditioned move distribution best explains this player's logged
                # moves (min NLL) -- refit every 40 new plies
                if (H.human is not None and prof.get("plies", 0) >= 40
                        and prof.get("plies", 0) - int(prof.get("elo_fit_plies") or 0) >= 40):
                    try:
                        import torch as _tF
                        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
                            tokenize as _tkF)
                        from catspace.research.components.encoder.approaches.reach_probability.experiments.train_human_moves import (
                            HumanMoves as _HM, mid_of as _mo, N_BUCKET)
                        obs = [(r["fen_before"], r["uci"]) for r in st.games.scan()
                               if r.get("type") == "ply" and r.get("mover") == "human"][-200:]
                        nll = [0.0] * N_BUCKET
                        for fenF, uciF in obs:
                            bF = chess.Board(fenF)
                            legalF = list(bF.legal_moves)
                            try:
                                ji = [m.uci() for m in legalF].index(uciF)
                            except ValueError:
                                continue
                            tkf, glf = _tkF(bF)
                            with H.lock, _tF.no_grad():
                                phiF = H.eng.net.backbone(
                                    _tF.tensor([tkf], dtype=_tF.long, device=H.eng.device),
                                    _tF.tensor([glf], dtype=_tF.float32,
                                               device=H.eng.device)).float()
                                midsF = _tF.tensor([_mo(m) for m in legalF],
                                                   dtype=_tF.long, device=H.eng.device)
                                rowF = _tF.zeros(len(legalF), dtype=_tF.long,
                                                 device=H.eng.device)
                                for bkt in range(N_BUCKET):
                                    lgF = H.human.move_logits(
                                        phiF, _tF.tensor([bkt], device=H.eng.device),
                                        midsF, rowF)
                                    nll[bkt] -= float(_tF.log_softmax(lgF, 0)[ji])
                        best_b = int(min(range(N_BUCKET), key=lambda i: nll[i]))
                        st.profile.put("current", {**prof, "elo_bucket": best_b,
                                                   "elo_fit_plies": prof.get("plies", 0)})
                        print(f"[prep] {st.name}: elo bucket fitted -> {best_b} "
                              f"(~{800 + max(0, best_b - 1) * 100} Elo, {len(obs)} moves)",
                              flush=True)
                    except Exception as e2:
                        print(f"[prep] elo fit skipped ({e2})", flush=True)
                print(f"[prep] {st.name}: studied {fen_key(bb)} "
                      f"-> {rows[0]['mv'].uci()}", flush=True)
            except Exception as e:
                print(f"[prep] skipped ({e})", flush=True)

    threading.Thread(target=prep_worker, daemon=True).start()
    import faulthandler, signal as _sig
    faulthandler.register(_sig.SIGUSR1)      # kill -USR1 <pid> -> all thread stacks to stderr
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), H)
    print(f"[kitty-server] serving {args.ckpt} on 0.0.0.0:{args.port}", flush=True)
    srv.serve_forever()


PAGE_BYTES = b""

if __name__ == "__main__":
    main()
