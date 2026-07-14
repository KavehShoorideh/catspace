#!/usr/bin/env python
"""
experiments/viz/neighbor_explorer.py — interactive embedding-neighbour explorer.

Kaveh, 2026-07-13: "I wanna play my own game (both sides) to create a new
position, then see the nearest neighbours for my custom position (start with
KRRvKBP, I'll edit). Use lichess or some other open-source board editor."

A tiny local web server (stdlib http.server -- no new dependency) that:
  - serves a Lichess-style board (chessboard.js, MIT) with PLAY (both sides,
    legal moves via chess.js) and EDIT (free placement + spare pieces) modes,
    starting from a KRRvKBP position;
  - on "Analyse", takes the board's FEN, embeds F(pos) with the trained model,
    and returns the k nearest positions from a same-region bank (cosine on the
    L2-normalised F), each rendered as a board with its reach-to-MATE_W and its
    DISTANCE-TO-MATE (Kaveh: compare against DTM, not DTZ -- the quasimetric has
    no handle on distance-to-zeroing; DTM is a position property the embedding
    could represent). DTM is computed from the Syzygy tablebase by playing the
    optimal line to mate (winner fastest, loser resisting), since Syzygy stores
    DTZ/WDL but not DTM at 6 pieces.

Run:  python experiments/viz/neighbor_explorer.py --ckpt <ckpt> [--port 8765]
then open http://localhost:8765
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import chess.svg
import chess.syzygy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from catspace.nn.features import feature_planes, omega_ids
from experiments.viz.build_embedding_neighbors import (
    Bank, load_bank_positions, material_sig, svg_board)


def dtm_line(board, tb, cap=250):
    """Plies to mate along a tablebase-optimal line: the winning side plays the
    fastest DTZ-progress move (preferring a zeroing move and avoiding repeats,
    so it can't cycle into a fivefold draw), the losing side resists (max |DTZ|).
    Returns ply count to mate, or None if the position is drawn/unresolved. A
    position-property distance-to-mate proxy -- Syzygy stores DTZ/WDL, not true
    DTM, at 6 pieces, and greedy DTZ play isn't game-theoretically exact, so this
    is a faithful upper-boundish distance, not the exact DTM."""
    b = board.copy(stack=False)
    try:
        wdl0 = tb.probe_wdl(b)
    except (KeyError, chess.syzygy.MissingTableError, ValueError, IndexError):
        return None
    if wdl0 == 0:
        return None
    winner = b.turn if wdl0 > 0 else (not b.turn)
    seen = set()
    for n in range(cap):
        if b.is_checkmate():
            return n
        if b.is_stalemate() or b.is_insufficient_material():
            return None
        cand = []                                            # (move, child, mover_wdl, dtz)
        for m in b.legal_moves:
            c = b.copy(stack=False); c.push(m)
            if c.is_checkmate():
                if b.turn == winner:
                    return n + 1
                continue
            try:
                wdl = tb.probe_wdl(c); dtz = tb.probe_dtz(c)
            except (KeyError, chess.syzygy.MissingTableError, ValueError, IndexError):
                continue
            cand.append((m, c, -wdl, dtz))
        if not cand:
            return None
        if b.turn == winner:
            wins = [x for x in cand if x[2] > 0] or cand
            def wkey(x):
                m, c, mw, dtz = x
                zeroing = 0 if (b.is_capture(m) or
                                b.piece_type_at(m.from_square) == chess.PAWN) else 1
                repeat = 1 if c.board_fen() in seen else 0
                return (repeat, abs(dtz), zeroing)           # avoid repeats, fastest, then zero
            m, c, _, _ = min(wins, key=wkey)
        else:
            m, c, _, _ = max(cand, key=lambda x: abs(x[3]))  # resist: max |dtz|
        seen.add(b.board_fen())
        b = c
    return None


class Explorer:
    def __init__(self, ckpt, bank_shards, bank_size, syzygy_dir, device, seed=0):
        import torch
        from catspace.nn.fb import load_ckpt, pick_device
        self.torch = torch
        self.device = pick_device(device)
        self.fb, payload = load_ckpt(Path(ckpt), self.device)
        self.z = torch.as_tensor(payload["zgoals"]["MATE_W"], dtype=torch.float32,
                                 device=self.device)
        self.omega = omega_ids(np.array([1800]), np.array([1800]), np.array([float("nan")]))[0]
        self.tb = chess.syzygy.open_tablebase(syzygy_dir)
        rng = np.random.default_rng(seed)
        bpacked, bmeta = load_bank_positions(bank_shards, bank_size, rng)
        self.bank = Bank(self.fb, self.omega, self.z, bpacked, bmeta, self.device)
        self.lock = threading.Lock()
        print(f"explorer ready: bank {len(self.bank.F)} positions, ckpt {ckpt}", flush=True)

    def embed_reach(self, board):
        from catspace.data.encode import encode_packed, encode_meta
        pl = self.torch.from_numpy(feature_planes(encode_packed(board)[None],
                                                  encode_meta(board)[None])).to(self.device)
        om = self.torch.from_numpy(self.omega[None]).to(self.device)
        with self.torch.no_grad():
            f = self.fb.embed_F(pl, om)
            reach = float(self.fb.score(f, self.z).cpu().numpy()[0])
        return f.cpu().numpy()[0], reach

    def analyse(self, fen, k):
        from catspace.data.encode import encode_packed, encode_meta
        board = chess.Board(fen)                              # may raise
        qmat = material_sig(board)
        key = encode_packed(board).tobytes() + encode_meta(board).tobytes()
        with self.lock:
            f_q, reach_q = self.embed_reach(board)
            qdtm = dtm_line(board, self.tb)
            qwdl = _safe_wdl(board, self.tb)
            nbrs = []
            for i, cos in self.bank.neighbours(f_q, k, exclude_key=key):
                nb = self.bank.board(i)
                nbrs.append(dict(
                    svg=svg_board(nb, size=170), cos=round(cos, 4),
                    reach=round(float(self.bank.reach[i]), 3),
                    dtm=dtm_line(nb, self.tb), wdl=_safe_wdl(nb, self.tb),
                    stm="W" if nb.turn else "B",
                    same_mat=material_sig(nb) == qmat, fen=nb.fen()))
        return dict(query=dict(fen=fen, svg=svg_board(board, size=260),
                               reach=round(reach_q, 3), dtm=qdtm, wdl=qwdl,
                               legal=board.is_valid()),
                    neighbours=nbrs)


def _safe_wdl(board, tb):
    try:
        return tb.probe_wdl(board)
    except (KeyError, chess.syzygy.MissingTableError, ValueError, IndexError):
        return None


KRRKBP_FEN = "7R/6K1/4p3/8/2k5/R7/8/5b2 w - - 0 1"

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>embedding neighbour explorer</title>
<link rel=stylesheet href="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.css">
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
<style>
 body{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 header{padding:12px 20px;border-bottom:1px solid #2a2e37;background:#161922}
 header b{color:#fff} .muted{color:#8b93a3;font-size:13px}
 #wrap{display:flex;gap:26px;padding:20px;align-items:flex-start;flex-wrap:wrap}
 #left{flex:0 0 400px} #board{width:400px}
 .row{margin:10px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 button,select{background:#2a3140;color:#fff;border:1px solid #3a4150;border-radius:6px;padding:7px 13px;font-size:13px;cursor:pointer}
 button:hover{background:#374050} button.on{background:#2f5fa3;border-color:#3f7fd0}
 #fen{flex:1;min-width:280px;background:#0c0e12;color:#cdd3df;border:1px solid #2a2e37;border-radius:6px;padding:7px;font:12px monospace}
 #qinfo{font-size:14px;margin:6px 0} #qinfo b{color:#fff}
 #grid{flex:1 1 520px;display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
 .cell{background:#161922;border:1px solid #2a2e37;border-radius:8px;padding:6px}
 .cell.mat{border-color:#2f6f43} .cell .cap{font-size:11px;color:#b9c0cd;margin-top:4px;line-height:1.35}
 .cell img{border-radius:4px} h3{font-size:13px;color:#8b93a3;text-transform:uppercase;letter-spacing:.04em;margin:0 0 8px}
 #status{color:#9ec7ff;font-size:13px;margin-left:8px}
</style></head><body>
<header><b>embedding neighbour explorer</b> &nbsp;<span class=muted>PLAY both sides or EDIT freely to build a position, then Analyse to see its nearest neighbours in F-embedding space (same-region bank). Metric: distance-to-mate.</span></header>
<div id=wrap>
 <div id=left>
  <div id=board></div>
  <div class=row>
   <button id=mplay class=on onclick="setMode('play')">Play (both sides)</button>
   <button id=medit onclick="setMode('edit')">Edit</button>
   <button onclick="stmFlip()">side: <span id=stm>w</span></button>
  </div>
  <div class=row>
   <button onclick="loadKRR()">KRRvKBP start</button>
   <button onclick="board.clear();syncFen()">Clear</button>
   <button onclick="board.flip()">Flip</button>
   <button onclick="undo()">Undo ply</button>
  </div>
  <div class=row><input id=fen value=""><button onclick="fromFen()">Load FEN</button></div>
  <div class=row>
   <button onclick="analyse()" style="background:#2f7d4f;border-color:#3f9d63">Analyse ▶</button>
   k=<select id=k><option>6</option><option selected>8</option><option>12</option><option>16</option></select>
   <span id=status></span>
  </div>
  <div id=qinfo></div>
 </div>
 <div style="flex:1 1 520px"><h3>nearest neighbours in embedding space</h3><div id=grid></div></div>
</div>
<script>
const START="__KRR__";
let mode='play', stm='w';
const game=new Chess();
function pieceTheme(p){return 'https://chessboardjs.com/img/chesspieces/wikipedia/'+p+'.png';}
function boardPart(){return board.fen();}                 // placement only
function fullFen(){return boardPart()+' '+stm+' - - 0 1';}
function syncFen(){document.getElementById('fen').value=fullFen();
 document.getElementById('stm').textContent=stm;}
function onDrop(src,tgt){
 if(mode==='edit') {setTimeout(syncFen,10); return;}
 const m=game.move({from:src,to:tgt,promotion:'q'});
 if(m===null) return 'snapback';
 stm=game.turn(); setTimeout(syncFen,10);
}
let cfg={draggable:true,position:START.split(' ')[0],pieceTheme,onDrop,
  onSnapEnd:()=>{if(mode==='play')board.position(game.fen());}};
let board=Chessboard('board',cfg);
function setMode(m){mode=m;
 document.getElementById('mplay').classList.toggle('on',m==='play');
 document.getElementById('medit').classList.toggle('on',m==='edit');
 const pos=board.fen();
 board=Chessboard('board',Object.assign({},cfg,{position:pos,
   sparePieces:m==='edit',dropOffBoard:m==='edit'?'trash':'snapback'}));
 if(m==='play') game.load(fullFen());
 syncFen();
}
function stmFlip(){stm=(stm==='w'?'b':'w'); if(mode==='play')game.load(fullFen()); syncFen();}
function loadKRR(){stm='w'; board.position(START.split(' ')[0]); game.load(START); syncFen();}
function fromFen(){const f=document.getElementById('fen').value.trim();
 board.position(f.split(' ')[0]); stm=(f.split(' ')[1]||'w');
 if(mode==='play'){try{game.load(f);}catch(e){}} syncFen();}
function undo(){if(mode==='play'){game.undo();board.position(game.fen());stm=game.turn();syncFen();}}
function fmtDtm(d,w){ if(w===0)return 'draw'; if(d===null)return (w===null?'off-TB':'dtm —');
 return 'mate in '+Math.ceil(d/2)+(w<0?' (win)':' (loss)'); }
async function analyse(){
 const k=document.getElementById('k').value;
 const fen=fullFen();
 document.getElementById('status').textContent='embedding…';
 try{
  const r=await fetch('/api/neighbors?'+new URLSearchParams({fen,k}));
  if(!r.ok){document.getElementById('status').textContent='error: '+(await r.text());return;}
  const d=await r.json();
  const q=d.query;
  document.getElementById('qinfo').innerHTML=
   `<b>query</b> reach ${q.reach.toFixed(3)} · ${fmtDtm(q.dtm,q.wdl)} · ${q.legal?'legal':'⚠ illegal position'}`;
  document.getElementById('grid').innerHTML=d.neighbours.map(n=>
   `<div class="cell${n.same_mat?' mat':''}">${n.svg}<div class=cap>cos ${n.cos.toFixed(3)} · reach ${n.reach.toFixed(2)}<br>${fmtDtm(n.dtm,n.wdl)} · ${n.stm} · ${n.same_mat?'✓mat':'✗mat'}</div></div>`).join('');
  document.getElementById('status').textContent=d.neighbours.length+' neighbours';
 }catch(e){document.getElementById('status').textContent='error: '+e;}
}
syncFen();
</script></body></html>""".replace("__KRR__", KRRKBP_FEN)


def make_handler(explorer):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif u.path == "/api/neighbors":
                q = urllib.parse.parse_qs(u.query)
                fen = q.get("fen", [""])[0]
                k = int(q.get("k", ["8"])[0])
                try:
                    self._send(200, json.dumps(explorer.analyse(fen, k)))
                except Exception as e:
                    self._send(400, str(e), "text/plain")
            else:
                self._send(404, "not found", "text/plain")
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="data/derived/lichess_fb_4gb_qm_plygap_only.pt")
    ap.add_argument("--bank-shards", nargs="+", default=["data/selfplay/krrkbp_loop"])
    ap.add_argument("--bank-size", type=int, default=15000)
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    explorer = Explorer(args.ckpt, args.bank_shards, args.bank_size,
                        args.syzygy_dir, args.device)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(explorer))
    print(f"\n  serving at  http://localhost:{args.port}\n  (Ctrl-C to stop)\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
