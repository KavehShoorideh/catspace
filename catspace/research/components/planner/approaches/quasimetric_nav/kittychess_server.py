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
from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess as _KC
KittyMATE=_KC.MATE

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_assets")

PAGE = """<!doctype html><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1">
<title>KittyChess analysis</title>
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
.lines .ln{display:flex;gap:8px;padding:4px 2px;border-bottom:1px solid #33312e;cursor:pointer;
font-family:'Noto Sans',sans-serif;font-size:13px;align-items:baseline}
.lines .ln:hover{background:#302d2a}
.lines .ev{min-width:52px;font-weight:600;color:#dedede;font-size:12px;font-variant-numeric:tabular-nums}
.lines .mv{color:#bababa;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lines .ln.best .ev{color:#759900}
#moves{font-size:13.5px;line-height:1.9;max-height:300px;overflow-y:auto}
#moves .no{color:#6f6b66;margin:0 3px 0 6px}
#moves .m{padding:1px 4px;border-radius:2px;cursor:pointer}
#moves .m:hover{background:#3a3733}
#moves .m.cur{background:#759900;color:#fff}
#wdltxt{font-size:11.5px;color:#8f8a82;margin-top:6px;font-variant-numeric:tabular-nums}
.spin{opacity:.6}
label.sw{display:flex;gap:5px;align-items:center;cursor:pointer}
</style>
<div class="top"><b>KittyChess</b> analysis board &mdash; three distances, softmaxed into the bar</div>
<div class="main">
<div id="evalbar"><div id="eb-b" style="height:33%"></div><div id="eb-d" style="height:34%"></div><div id="eb-w" style="height:33%"></div></div>
<div>
  <div id="board" class="cg-wrap"></div>
  <div class="nav">
    <button id="first">&#x23EE;</button><button id="prev">&#x25C0;</button>
    <button id="next">&#x25B6;</button><button id="last">&#x23ED;</button>
    <button id="new" style="flex:2">new</button>
  </div>
  <div id="wdltxt"></div>
</div>
<div class="right">
  <div class="box">
    <div class="engrow"><b>KittyChess</b>
      <span id="dinfo"></span>
      <label>depth <select id="depth"><option>1</option><option selected>2</option><option>3</option><option>4</option></select></label>
      <label>lines <select id="lines"><option>1</option><option>2</option><option selected>3</option><option>4</option><option>5</option></select></label>
      <button id="flip" style="background:#3a3733;border:none;color:#bababa;border-radius:3px;padding:2px 8px;cursor:pointer">flip</button>
    </div>
    <div class="lines" id="lnbox"></div>
  </div>
  <div class="box"><div id="moves"></div></div>
</div>
</div>
<script>__CG_JS__</script>
<script>
"use strict";
let seq=0, orient="white";
const cg=window.Chessground(document.getElementById('board'),{coordinates:true});
const knobs=()=>({depth:+document.getElementById('depth').value,
  lines:+document.getElementById('lines').value});
// Two-phase like lichess: the position updates INSTANTLY, the engine lines stream in after.
// A stale analysis (older seq) is discarded, so mashing the nav buttons stays responsive.
async function api(body){
  const my=++seq;
  const r=await fetch('/state',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({...body,...knobs()})});
  const d=await r.json();
  if(my!==seq)return;
  render(d);
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
  cg.set({fen:d.fen, turnColor:d.turn, check:d.check, lastMove:d.lastMove||undefined,
    orientation:orient,
    movable:{free:false, color:d.turn, dests:new Map(Object.entries(d.dests)),
      events:{after:(o,t)=>api({action:"move",uci:o+t})}}});
  // tricolor committor bar: white at the bottom (lichess convention), draw grey between
  document.getElementById('eb-w').style.height=(d.wdl[0]*100)+"%";
  document.getElementById('eb-d').style.height=(d.wdl[1]*100)+"%";
  document.getElementById('eb-b').style.height=(d.wdl[2]*100)+"%";
  document.getElementById('wdltxt').textContent=
    `white ${(d.wdl[0]*100).toFixed(1)}%  ·  draw ${(d.wdl[1]*100).toFixed(1)}%  ·  black ${(d.wdl[2]*100).toFixed(1)}%`
    +(d.dists?`   |   d→white-win ${d.dists[0].toFixed(2)}  d→draw ${d.dists[1].toFixed(2)}  d→black-win ${d.dists[2].toFixed(2)}`:"   |   exact (terminal/tablebase)");
  const mv=document.getElementById('moves'); mv.innerHTML="";
  (d.san||[]).forEach((sn,i)=>{
    if(i%2===0){const no=document.createElement('span');no.className="no";no.textContent=(i/2+1)+".";mv.appendChild(no);}
    const sp=document.createElement('span');sp.className="m"+(i===d.ptr-1?" cur":"");sp.textContent=sn;
    sp.onclick=()=>api({action:"goto",ptr:i+1}); mv.appendChild(sp);});
  const cur=mv.querySelector('.cur'); if(cur)cur.scrollIntoView({block:"nearest"});
  if(d.over)document.getElementById('dinfo').textContent=d.over;
}
function renderLines(d){
  const lb=document.getElementById('lnbox'); lb.innerHTML="";
  (d.think||[]).forEach((row,i)=>{
    const div=document.createElement('div'); div.className="ln"+(i===0?" best":"");
    div.innerHTML=`<span class="ev">${row.margin}${row.tb?" tb":""}</span><span class="mv">${row.line||row.uci}</span>`;
    div.onclick=()=>api({action:"move",uci:row.uci});
    lb.appendChild(div);});
}
document.getElementById('new').onclick=()=>api({action:"new"});
document.getElementById('first').onclick=()=>api({action:"goto",ptr:0});
document.getElementById('prev').onclick=()=>api({action:"rel",d:-1});
document.getElementById('next').onclick=()=>api({action:"rel",d:1});
document.getElementById('last').onclick=()=>api({action:"goto",ptr:99999});
document.getElementById('depth').onchange=()=>api({action:"noop"});
document.getElementById('lines').onchange=()=>api({action:"noop"});
document.getElementById('flip').onclick=()=>{orient=orient==="white"?"black":"white";api({action:"noop"});};
addEventListener('keydown',e=>{
  if(e.key==="ArrowLeft"){e.preventDefault();api({action:"rel",d:-1});}
  if(e.key==="ArrowRight"){e.preventDefault();api({action:"rel",d:1});}});
api({action:"noop"});
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
        b = chess.Board()
        for m in H.moves[:H.ptr]:
            b.push(m)
        return b

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        cls = H
        cls.depth = max(1, min(4, int(req.get("depth", cls.depth))))
        cls.lines = max(1, min(5, int(req.get("lines", cls.lines))))
        act = req.get("action", "noop")
        if act not in ("analyze", "lines"):
            H.gen += 1                                   # cancel any in-flight search NOW
        if act == "new":
            cls.moves, cls.ptr = [], 0
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
                    def publish(rows, d):
                        if H.gen == g:
                            H.an = {**H.an, "g": g, "depth": d,
                                    "rows": [{k: r.get(k) for k in ("uci", "margin", "tb", "line")}
                                             for r in self._rows_to_display(bb, rows)[:nlines]],
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
        out = {"fen": b.fen(), "turn": "white" if b.turn else "black",
               "depth": cls.depth, "san": san, "ptr": cls.ptr, "wdl": wdl,
               "dists": dists, "check": b.is_check(), "dests": dests,
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

    def _rows_to_display(self, b, rows):
        """WHITE-POV values, lichess-style (Kaveh 2026-08-08): the number shown is always
        white's margin (negated when black is to move), so it doesn't flip sign every ply.
        Search order (best-for-mover first) is kept: for black that IS lowest-white-value
        first, which is the requested sort."""
        out = []
        for r in rows[:18]:
            bb = b.copy()
            sans = []
            for mv in r["pv"][:6]:
                sans.append(bb.san(mv)); bb.push(mv)
            v = r["value"] if b.turn == chess.WHITE else -r["value"]
            disp = (("#" if v > 0 else "-#") if abs(v) >= KittyMATE / 4
                    else round(v, 3) if abs(v) < 10 else round(v, 1))
            out.append({"mv": r["mv"], "uci": r["mv"].uci(), "margin": disp,
                        "dwin": "", "ddraw": "", "dloss": "",
                        "tb": abs(v) >= KittyMATE / 4, "deep": True,
                        "line": " ".join(sans)})
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
    H.lock = threading.Lock()
    H.depth = args.depth
    global PAGE_BYTES
    cg_js = open(os.path.join(ASSETS, "bundle.js")).read()
    cg_css = open(os.path.join(ASSETS, "bundle.css")).read()
    PAGE_BYTES = PAGE.replace("__CG_CSS__", cg_css).replace("__CG_JS__", cg_js).encode()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), H)
    print(f"[kitty-server] serving {args.ckpt} on 0.0.0.0:{args.port}", flush=True)
    srv.serve_forever()


PAGE_BYTES = b""

if __name__ == "__main__":
    main()
