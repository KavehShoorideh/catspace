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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess

from catspace.io import paths
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
.lines .mv{color:#bababa;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lines .ln.best .ev{color:#759900}
#moves{font-size:13.5px;line-height:1.9;max-height:300px;overflow-y:auto}
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
</style>
<div class="top"><b>catspace</b>
  <span style="margin-left:18px">
    <button class="navbtn" id="nav-an">Analysis board</button>
    <button class="navbtn" id="nav-play">Play catspace</button>
  </span>

</div>
<div class="main">
<div id="evalbar"><div id="eb-b" style="height:33%"></div><div id="eb-d" style="height:34%"></div><div id="eb-w" style="height:33%"></div></div>
<div>
  <div id="board" class="cg-wrap"></div>
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
      <label>depth <select id="depth"><option>1</option><option selected>2</option><option>3</option><option>4</option><option>5</option><option>6</option></select></label>
      <label>pv <select id="pvlen"><option>4</option><option selected>6</option><option>8</option></select></label>
      <label>lines <select id="lines"><option>1</option><option>2</option><option selected>3</option><option>4</option><option>5</option></select></label>
      <button id="flip" style="background:#3a3733;border:none;color:#bababa;border-radius:3px;padding:2px 8px;cursor:pointer">flip</button>
    </div>
    <div class="lines" id="lnbox"></div>
    <div id="lnlegend">E = expected points, white (probability head) &nbsp;·&nbsp; Δd = exit gap dB−dW (length head) &nbsp;·&nbsp; ⚡ = line descends ≥0.7 plies/ply toward one ending &nbsp;·&nbsp; ranked by CASCADE</div>
  </div>
  <div class="box"><div id="moves"></div></div>
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
function setMode(m){
  mode=m;
  document.body.classList.toggle('playmode', m==='play');
  document.getElementById('nav-an').classList.toggle('on', m==='analysis');
  document.getElementById('nav-play').classList.toggle('on', m==='play');
}
const cg=window.Chessground(document.getElementById('board'),{coordinates:true});
const knobs=()=>({depth:+document.getElementById('depth').value,
  lines:+document.getElementById('lines').value,
  pvlen:+document.getElementById('pvlen').value});
// Two-phase like lichess: the position updates INSTANTLY, the engine lines stream in after.
// A stale analysis (older seq) is discarded, so mashing the nav buttons stays responsive.
async function api(body){
  const my=++seq;
  const extra = mode==='play' ? {play:true, engineWhite:playCfg.engineWhite,
                                 playDepth:playCfg.depth} : {};
  if(mode==='play'){document.getElementById('dinfo').textContent="thinking…";
    document.getElementById('dinfo').className="spin";}
  const r=await fetch('/state',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({...body,...knobs(),...extra})});
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
    document.getElementById('dinfo').textContent="depth "+dp.depth+(dp.done?"":"…")
      +(dp.resid!=null?"  ·  Bellman residual "+dp.resid.toFixed(1):"");
    document.getElementById('dinfo').className=dp.done?"":"spin";
    if(dp.done)break;
  }
}
function render(d){
  const mvcolor = mode==='play' ? (playCfg.engineWhite?'black':'white') : d.turn;
  cg.set({fen:d.fen, turnColor:d.turn, check:d.check, lastMove:d.lastMove||undefined,
    orientation:orient,
    movable:{free:false, color:mvcolor, dests:new Map(Object.entries(d.dests)),
      events:{after:(o,t)=>api({action:"move",uci:o+t})}}});
  // tricolor committor bar: white at the bottom (lichess convention), draw grey between
  document.getElementById('eb-w').style.height=(d.wdl[0]*100)+"%";
  document.getElementById('eb-d').style.height=(d.wdl[1]*100)+"%";
  document.getElementById('eb-b').style.height=(d.wdl[2]*100)+"%";
  drawPosTable(d.wdl, d.dists, "static");
  const mv=document.getElementById('moves'); mv.innerHTML="";
  (d.san||[]).forEach((sn,i)=>{
    if(i%2===0){const no=document.createElement('span');no.className="no";no.textContent=(i/2+1)+".";mv.appendChild(no);}
    const sp=document.createElement('span');sp.className="m"+(i===d.ptr-1?" cur":"");sp.textContent=sn;
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
  if(d.over)document.getElementById('dinfo').textContent=d.over;
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
         (row.dists?`<span class="ev" style="min-width:58px" title="decisive-exit gap dB−dW (length head): positive = white's ending closer">Δd ${(row.dists[2]-row.dists[0]).toFixed(1)}</span>`:"");
    } else {ev=`<span class="ev" title="searched margin (white POV)">${row.margin}</span>`;}
    let top=`<div class="lnr">`+ev;
    if(row.wdl){
      top+=`<span class="minibar"><i style="width:${row.wdl[0]*100}%;background:#f0efeb"></i>`+
           `<i style="width:${row.wdl[1]*100}%;background:#8b8680"></i>`+
           `<i style="width:${row.wdl[2]*100}%;background:#403d39"></i></span>`+
           `<span style="font-size:11px;color:#8f8a82">${Math.round(row.wdl[0]*100)}/${Math.round(row.wdl[1]*100)}/${Math.round(row.wdl[2]*100)}</span>`;
    }
    if(row.force&&row.force.drop>=0.7&&row.force.mono>=0.7)
      top+=`<span style="color:#dbac16;font-size:11px" title="forcing: distance to ${row.force.fav==='w'?'white':'black'}-win drops ${row.force.drop}/ply, monotone ${Math.round(row.force.mono*100)}%">⚡forced</span>`;
    else if(row.force&&row.force.drop>=0.35)
      top+=`<span style="color:#8f8a82;font-size:11px" title="coherent progress: ${row.force.drop}/ply toward ${row.force.fav==='w'?'white':'black'}-win">→${row.force.drop}</span>`;
    top+=`<span class="mv">${row.line||row.uci}</span></div>`;
    if(row.dists) top+=`<div class="sub">dW <b>${row.dists[0].toFixed(1)}</b> · dD <b>${row.dists[1].toFixed(1)}</b> · dB <b>${row.dists[2].toFixed(1)}</b></div>`;
    div.innerHTML=top;
    div.onclick=()=>api({action:"move",uci:row.uci});
    lb.appendChild(div);});
}
document.getElementById('new').onchange=e=>{
  const v=e.target.value; e.target.value="";
  api(v?{action:"load",fen:v}:{action:"new"});};
document.getElementById('first').onclick=()=>api({action:"goto",ptr:0});
document.getElementById('prev').onclick=()=>api({action:"rel",d:-1});
document.getElementById('next').onclick=()=>api({action:"rel",d:1});
document.getElementById('last').onclick=()=>api({action:"goto",ptr:99999});
document.getElementById('depth').onchange=()=>api({action:"noop"});
document.getElementById('lines').onchange=()=>api({action:"noop"});
document.getElementById('pvlen').onchange=()=>api({action:"noop"});
document.getElementById('flip').onclick=()=>{orient=orient==="white"?"black":"white";api({action:"noop"});};
document.getElementById('nav-an').onclick=()=>{
  setMode('analysis'); api({action:"noop"});};
document.getElementById('nav-play').onclick=()=>{
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

    def _log_game(self, b, san, over, play):
        try:
            json.dump({"san": san, "fen": b.fen(), "start_fen": H.start_fen,
                       "ptr": H.ptr, "mode": "play" if play else "analysis",
                       "over": over,
                       "moves_uci": [m.uci() for m in H.moves]},
                      open(GAME_LOG, "w"))
        except Exception:
            pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        cls = H
        cls.depth = max(1, min(6, int(req.get("depth", cls.depth))))
        cls.lines = max(1, min(5, int(req.get("lines", cls.lines))))
        cls.pvlen = max(4, min(8, int(req.get("pvlen", getattr(cls, "pvlen", 6)))))
        act = req.get("action", "noop")
        if act not in ("analyze", "lines"):
            H.gen += 1                                   # cancel any in-flight search NOW
        if act == "new":
            cls.moves, cls.ptr, cls.start_fen = [], 0, None
        elif act == "load" and req.get("fen"):
            try:
                chess.Board(req["fen"])
                cls.moves, cls.ptr, cls.start_fen = [], 0, req["fen"]
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
                    cls.moves = cls.moves[:cls.ptr] + [mv]
                    cls.ptr += 1
            except Exception:
                pass
        b = self._board()
        over = self._over(b)
        # PLAY MODE (Kaveh 2026-08-11, lichess-style menu): the engine answers synchronously
        # when it is to move; analysis assistance is hidden client-side during play
        if req.get("play") and act in ("move", "new") and not over \
                and cls.ptr == len(cls.moves):
            eng_white = bool(req.get("engineWhite"))
            if (b.turn == chess.WHITE) == eng_white:
                pd = max(1, min(6, int(req.get("playDepth", 2))))
                # time-capped iterative deepening (Kaveh: think <= ~1.5s): keep the deepest
                # COMPLETED iteration; a timed-out partial iteration is discarded
                import time as _time
                deadline = _time.time() + 1.5
                best = None
                with H.lock:
                    for d in range(1, pd + 1):
                        rows = H.eng.search(b, depth=d,
                                            stop=lambda: _time.time() > deadline)
                        if _time.time() > deadline:
                            if best is None:
                                best = rows
                            break
                        best = rows
                if best:
                    cls.moves.append(best[0]["mv"]); cls.ptr += 1
                    b = self._board(); over = self._over(b)
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
                    cmoves, corder, _ = H.eng.cascade_rank(bb)
                    crank = {cmoves[j].uci(): pos for pos, j in enumerate(corder)} if cmoves else {}

                    def publish(rows, d):
                        if H.gen == g:
                            # CASCADE RANKING (Kaveh 2026-08-11): display order = the cascade's
                            # order; the searched value rides along per line
                            # PROVEN results outrank the learned ordering: a searched mate/TB
                            # value is a certificate; cascade orders only the unproven rest
                            # (2026-08-11: cascade buried Rd8# under quiet pawn pushes)
                            rs = sorted(rows, key=lambda r: (
                                -r["value"] if abs(r["value"]) >= KittyMATE / 4 else 0,
                                crank.get(r["mv"].uci(), 999)))
                            H.an = {**H.an, "g": g, "depth": d,
                                    "rows": [{k: r.get(k) for k in
                                              ("uci", "margin", "tb", "line", "wdl", "dists",
                                               "force")}
                                             for r in self._rows_to_display(
                                                 bb, rs, wdl_top=nlines)[:nlines]],
                                    "done": False}
                    try:
                        with H.lock:
                            # Bellman-residual gate (Kaveh 2026-08-08): a self-contradictory
                            # field here earns one extra ply; the residual is shown in the UI.
                            resid = H.eng.bellman_residual(bb)
                            top = depth + (1 if resid is not None and abs(resid) >= 2.0 else 0)
                            H.an = {**H.an, "resid": resid, "top": top}
                            for d in range(1, top + 1):
                                if H.gen != g:
                                    return
                                rows = H.eng.search(bb, depth=d, stop=lambda: H.gen != g,
                                                    progress=lambda rs, _d=d: publish(rs, _d))
                                publish(rows, d)
                    finally:
                        if H.gen == g:
                            H.an = {**H.an, "done": True}

                H.an = {"g": g, "depth": 0, "rows": [], "done": False}
                H.an_thread = threading.Thread(target=worker, daemon=True)
                H.an_thread.start()
            self._send(200, json.dumps({"started": True, "over": over}).encode())
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
               "dists": dists, "check": b.is_check(), "dests": dests,
               "concepts": concepts, "tokens": tokens,
               "lastMove": last, "over": over}
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
            for mv in r["pv"][:getattr(H, "pvlen", 6)]:
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
                for mv in r["pv"][:getattr(H, "pvlen", 6)]:
                    bc.push(mv)
                try:
                    row["wdl"], row["dists"] = H.eng.wdl(bc)
                except Exception:
                    pass
                try:
                    coh = H.eng.line_coherence(b, r["pv"], max_plies=getattr(H, "pvlen", 6))
                    if coh is not None:
                        row["force"] = {"drop": round(coh[0], 2), "mono": round(coh[1], 2),
                                        "fav": coh[2]}
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
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

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
        print(f"[kitty-server] concept quantizer loaded ({pv['heads']}x{pv['codes']})", flush=True)
    except Exception as e:
        print(f"[kitty-server] no concept sidecars ({e})", flush=True)
    H.lock = threading.Lock()
    H.depth = args.depth
    global PAGE_BYTES
    cg_js = open(os.path.join(ASSETS, "bundle.js")).read()
    cg_css = open(os.path.join(ASSETS, "bundle.css")).read()
    # POSITION LIBRARY (Kaveh 2026-08-11): curated interesting starts -- forced-mate nets,
    # conversion tests, and live picks from the behavioral sanity suite
    pos = [("KRK: convert the rook", "8/8/8/4k3/8/8/8/R3K3 w - - 0 1"),
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
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), H)
    print(f"[kitty-server] serving {args.ckpt} on 0.0.0.0:{args.port}", flush=True)
    srv.serve_forever()


PAGE_BYTES = b""

if __name__ == "__main__":
    main()
