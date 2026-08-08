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
<title>KittyChess (full engine)</title>
<style>
body{margin:0;background:#f5f4f1;color:#1d1c1a;font:14px/1.5 system-ui,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:24px 18px}
h1{font-size:20px;margin:0 0 2px}.sub{color:#75716a;font-size:12.5px;margin-bottom:16px}
.cols{display:grid;grid-template-columns:minmax(300px,480px) 1fr;gap:22px}
@media(max-width:760px){.cols{grid-template-columns:1fr}}
#board{width:100%;max-width:480px;aspect-ratio:1}
.bar{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap;align-items:center}
button{font:inherit;font-size:13px;padding:6px 12px;border:1px solid #ddd9d2;background:#fff;border-radius:5px;cursor:pointer}
#status{font-family:ui-monospace,monospace;font-size:12.5px;color:#75716a}
.panel{background:#fff;border:1px solid #ddd9d2;border-radius:6px;padding:12px 14px}
.panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#75716a;margin:0 0 8px}
table{border-collapse:collapse;width:100%;font-family:ui-monospace,monospace;font-size:11.5px}
th,td{text-align:right;padding:2.5px 7px;border-bottom:1px solid #eee}
th:first-child,td:first-child{text-align:left}
tr.pick td{color:#b3502e;font-weight:700}
__CG_CSS__
</style>
<div class="wrap"><h1>KittyChess <span style="font-size:12px;color:#75716a">full engine · tablebase endgames</span></h1>
<div class="sub">threat-first navigation on the learned quasimetric field — served live from the training machine.</div>
<div class="cols"><div>
<div id="board" class="cg-wrap"></div>
<div class="bar"><button id="new">new game</button><button id="flip">play black</button>
<label style="font-size:12px;color:#75716a">depth <select id="depth"><option>1</option><option>2</option><option selected>3</option><option>4</option></select></label>
<label style="font-size:12px;color:#75716a">lines <select id="lines"><option>1</option><option>2</option><option selected>3</option><option>4</option><option>5</option></select></label>
<span id="status"></span></div>
</div>
<div class="panel"><h2>analysis — your engine, current position <span style="text-transform:none">(alpha-beta on the margin)</span></h2>
<table id="linesT"><tbody></tbody></table>
</div></div></div>
<script>__CG_JS__</script>
<script>
"use strict";
let humanWhite=true, busy=false, fen="start";
const cg=window.Chessground(document.getElementById('board'),{coordinates:true});
const S=t=>document.getElementById('status').textContent=t;
function knobs(){return {depth:+document.getElementById('depth').value,
  lines:+document.getElementById('lines').value};}
async function refresh(move,analyzeOnly){
  const r=await fetch('/state',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({move:move||null, newGame:move===undefined&&!analyzeOnly, humanWhite,
      analyzeOnly:!!analyzeOnly, ...knobs()})});
  const d=await r.json();
  fen=d.fen;
  cg.set({fen:d.fen, turnColor:d.turn, check:d.check, lastMove:d.lastMove||undefined,
    orientation:humanWhite?"white":"black",
    movable:{free:false, color:d.over?undefined:(humanWhite?"white":"black"),
      dests:new Map(Object.entries(d.dests)), events:{after:(o,t)=>play(o+t)}}});
  const lt=document.querySelector('#linesT tbody'); lt.innerHTML="";
  (d.think||[]).slice(0,d.lines_n||3).forEach((row,i)=>{
    const tr=document.createElement('tr'); if(i===0)tr.className="pick";
    tr.innerHTML=`<td style="width:4em">${row.margin}${row.tb?" tb":""}</td>`+
      `<td style="text-align:left">${row.line||row.uci}</td>`;
    lt.appendChild(tr);});
  S(d.over||d.thinkingNote||"your move");
}
async function play(uci){ if(busy)return; busy=true; S("thinking…");
  try{ await refresh(uci); } finally{ busy=false; } }
document.getElementById('new').onclick=()=>refresh();
document.getElementById('depth').onchange=()=>refresh(null,true);
document.getElementById('lines').onchange=()=>refresh(null,true);
document.getElementById('flip').onclick=e=>{humanWhite=!humanWhite;
  e.target.textContent=humanWhite?"play black":"play white"; refresh();};
refresh();
</script>"""


class H(BaseHTTPRequestHandler):
    eng = None
    board = chess.Board()
    human_white = True
    depth = 3

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

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        cls = H
        cls.depth = max(1, min(4, int(req.get("depth", cls.depth))))
        cls.lines = max(1, min(5, int(req.get("lines", getattr(cls, "lines", 3)))))
        if req.get("newGame"):
            cls.board = chess.Board()
            cls.human_white = bool(req.get("humanWhite", True))
        elif req.get("move"):
            try:
                cls.board.push_uci(req["move"])
            except Exception:
                pass
        # engine replies whenever it is to move
        over = self._over()
        if not over and not req.get("analyzeOnly")                 and (cls.board.turn == chess.WHITE) != cls.human_white:
            rows = self._deliberate()
            if rows:
                cls.board.push(rows[0]["mv"])
            over = self._over()
        # ANALYSIS of the *current* position -- the analysis engine IS this engine
        # (Kaveh 2026-08-08), lichess-style: depth + number of lines are user knobs.
        think = []
        if not over:
            think = [{k: r.get(k) for k in ("uci", "margin", "tb", "line")}
                     for r in self._deliberate()[:max(cls.lines, 8)]]
        dests = {}
        if not over and (cls.board.turn == chess.WHITE) == cls.human_white:
            for m in cls.board.legal_moves:
                dests.setdefault(chess.square_name(m.from_square), []).append(
                    chess.square_name(m.to_square))
        last = None
        if cls.board.move_stack:
            m = cls.board.move_stack[-1]
            last = [chess.square_name(m.from_square), chess.square_name(m.to_square)]
        out = {"fen": cls.board.fen(), "turn": "white" if cls.board.turn else "black",
               "lines_n": cls.lines, "depth": cls.depth,
               "check": cls.board.is_check(), "dests": dests, "think": think,
               "lastMove": last, "over": over}
        self._send(200, json.dumps(out).encode())

    def _over(self):
        b = H.board
        if b.is_game_over(claim_draw=True):
            o = b.outcome(claim_draw=True)
            if o.winner is None:
                return f"draw ({o.termination.name.lower()})"
            return ("white" if o.winner else "black") + " wins"
        return None

    def _deliberate(self):
        b = H.board
        rows = H.eng.search(b, depth=H.depth)
        out = []
        for r in rows[:18]:
            bb = b.copy()
            sans = []
            for mv in r["pv"][:6]:
                sans.append(bb.san(mv)); bb.push(mv)
            v = r["value"]
            disp = ("#" if abs(v) >= KittyMATE / 4 else round(v, 1))
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

    from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess
    H.eng = KittyChess(args.ckpt, args.device, args.cond_elo)
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
