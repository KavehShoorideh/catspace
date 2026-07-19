#!/usr/bin/env python
"""
experiments/viz/play_server.py — COMPONENT B of the play-atlas app.

A tiny stdlib HTTP server that loads the INCUMBENT model on CPU (the training
GPU stays free) and exposes the play + projection API the frontend calls. Run
the precompute (build_play_atlas.py) first so artifacts/generated/play_atlas/
exists, then:

  .venv/bin/python experiments/viz/play_server.py --port 8000
  # open http://localhost:8000

Endpoints (see the frozen contract): GET / /atlas /toy /region_sample
/legal_moves ; POST /apply_move /engine_move /project /set_board. The model is
single-threaded behind a lock; searches use a committor-MCTS at --nodes.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catspace.nn.features import feature_planes, omega_ids  # noqa: E402
from catspace.data.encode import encode_meta, encode_packed  # noqa: E402

TEMPLATE = ROOT / "catspace/viz/templates/play_atlas.html"
ATLAS_DIR = ROOT / "artifacts/generated/play_atlas"
ASSETS_DIR = ROOT / "catspace/viz/assets"      # vendored chessground (JS+CSS)
# A CLEAN, tablebase-won KRRvKBP toy from the real test set (Ka7 Rc7 Rb3 vs
# Kd4 Ba3 pf3) — no rook hanging, kings apart. fens[0] of krrkbp_test_n200
# is a genuine win too but reads as a rook blunder, so it made a bad demo.
TOY_FALLBACK = "8/K1R5/8/8/3k4/bR3p2/8/8 w - - 0 1"
_MIME = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml",
         ".json": "application/json", ".html": "text/html; charset=utf-8"}


class Engine:
    """Model + committor-MCTS + fitted t-SNE projection + atlas, all on CPU."""

    def __init__(self, ckpt: str, phead: str, nodes: int):
        import torch
        from catspace.nn.eval_head import EvalHead
        from catspace.nn.fb import load_ckpt
        from catspace.nn.policy_fb import make_search_policy
        self.torch = torch
        self.lock = threading.Lock()
        self.dev = "cpu"
        fb, pay = load_ckpt(Path(ckpt), self.dev)
        fb.eval()
        hp = torch.load(phead, map_location=self.dev, weights_only=False)
        ph = EvalHead(d_in=hp["d_in"]).to(self.dev)
        ph.load_state_dict(hp["state"])
        ph.eval()
        self.fb, self.phead = fb, ph
        self.omega_row = omega_ids(np.array([1800]), np.array([1800]),
                                   np.array([300.0]))[0]

        class Committor(torch.nn.Module):
            def forward(self, f):
                p = torch.softmax(ph(f), dim=1)
                return -torch.log(p[:, 0].clamp_min(1e-6)).unsqueeze(-1)

        self.pol = make_search_policy("mcts", fb, pay["zgoals"]["MATE_W"],
                                      max_nodes=nodes, device=self.dev,
                                      committor_head=Committor(), mate_stop=True)
        self.rng = np.random.default_rng(0)
        self._atlas = None
        self._proj = None
        # stateful Analyze: keep the last-analyzed tree so repeated Analyze on
        # the SAME position EXTENDS the search (+budget) instead of restarting
        self._an_root = None
        self._an_fen = None
        self._an_total = 0

    # -- lazy atlas + projection ------------------------------------------
    @property
    def atlas(self):
        if self._atlas is None:
            p = ATLAS_DIR / "atlas.json"
            if not p.exists():
                raise FileNotFoundError(
                    "atlas.json missing — run build_play_atlas.py first")
            self._atlas = json.loads(p.read_text())
        return self._atlas

    @property
    def proj(self):
        if self._proj is None:
            from catspace.viz.projection import Normalizer, TSNEProjection
            from catspace.viz.realboard import _FittedProjection
            d = ATLAS_DIR / "tsne_map"
            nz = np.load(d / "normalizer.npz")
            emb = pickle.load(open(d / "embedding.pkl", "rb"))
            tp = TSNEProjection()
            tp._embedding = emb
            self._proj = _FittedProjection(Normalizer(mu=nz["mu"], sd=nz["sd"]), tp)
        return self._proj

    # -- embedding helpers -------------------------------------------------
    def _embed_F(self, board: chess.Board) -> np.ndarray:
        packed = encode_packed(board)[None]
        meta = encode_meta(board)[None]
        planes = self.torch.from_numpy(feature_planes(packed, meta)).to(self.dev)
        om = self.torch.from_numpy(np.tile(self.omega_row, (1, 1))).to(self.dev)
        with self.torch.no_grad():
            return self.fb.embed_F(planes, om).cpu().numpy()[0]

    def winp(self, board: chess.Board) -> float:
        f = self.torch.from_numpy(self._embed_F(board)[None]).to(self.dev)
        with self.torch.no_grad():
            return float(self.torch.softmax(self.phead(f), dim=1)[0, 0])

    def _xy(self, board: chess.Board) -> tuple:
        xy = self.proj.transform(self._embed_F(board)[None])[0]
        return round(float(xy[0]), 3), round(float(xy[1]), 3)

    def _hops(self, board: chess.Board, node, maxlen: int = 6) -> list:
        """The principal variation under `node` as a sequence of HOPS — one per
        ply, alternating White/Black — each carrying its SAN, resulting FEN,
        and PROJECTED (x,y). Descends max-visit children (the line MCTS already
        built). The frontend draws these as a path through the embedding map
        and previews each position on hover. node.move is the first move
        relative to `board`."""
        b = board.copy(stack=False)
        hops, cur = [], node
        while cur is not None and cur.move is not None and len(hops) < maxlen:
            try:
                san = b.san(cur.move)
            except (ValueError, AssertionError):
                break
            white = b.turn == chess.WHITE            # side making THIS move
            b.push(cur.move)
            x, y = self._xy(b)
            hops.append(dict(san=san, fen=b.fen(), x=x, y=y, white=white))
            cur = max(cur.children, key=lambda c: c.N, default=None) if cur.children else None
        return hops

    def project(self, board: chess.Board) -> dict:
        f = self._embed_F(board)
        xy = self.proj.transform(f[None])[0]
        zw = self.pol.z.cpu().numpy() if hasattr(self.pol, "z") else None
        reach = float(f @ (zw / (np.linalg.norm(zw) + 1e-12))) if zw is not None else 0.0
        cl = self.atlas["clusters"]
        near = min(cl, key=lambda c: (c["cx"] - xy[0]) ** 2 + (c["cy"] - xy[1]) ** 2)
        return dict(x=round(float(xy[0]), 3), y=round(float(xy[1]), 3),
                    reach=round(reach, 4), winp=round(self.winp(board), 4),
                    cluster=int(near["id"]))

    def analyze(self, board: chess.Board, topk: int = 3, nodes: int | None = None,
                extend: bool = False) -> dict:
        """Top-k candidate moves for the side to move, each with the PROJECTED
        (x,y) of the resulting position AND its principal-variation LINE (SAN).
        `nodes` sets this call's budget. `extend`=True CONTINUES the previous
        Analyze tree for the SAME position (adds `nodes` more evals on top —
        the tree, visit counts and lines accumulate) instead of restarting;
        a different position (or extend=False) starts fresh. Returns `nodes` =
        the CUMULATIVE budget spent on this position."""
        over, res = self._outcome(board)
        if over:
            self._an_root = self._an_fen = None
            self._an_total = 0
            px, py = self._xy(board)
            return dict(game_over=True, result=res, winp=round(self.winp(board), 4),
                        x=px, y=py, candidates=[], pv=[], nodes=0)
        fen = board.fen()
        with self.lock:
            reuse = (self._an_root if extend and self._an_fen == fen
                     and self._an_root is not None else None)
            old = self.pol.mcts.max_nodes
            self.pol.mcts.max_nodes = int(nodes or old)
            try:
                root = self.pol.mcts.run(board, reuse_root=reuse)
            finally:
                self.pol.mcts.max_nodes = old
            self._an_root, self._an_fen = root, fen
            self._an_total = (self._an_total if reuse is not None else 0) + int(nodes or old)
            total = self._an_total
        cand = sorted(root.children, key=lambda c: c.N, reverse=True)[:topk]
        px, py = self._xy(board)
        out = []
        for c in cand:
            child = board.copy(stack=False)
            child.push(c.move)
            cx, cy = self._xy(child)
            out.append(dict(uci=c.move.uci(), san=board.san(c.move), visits=int(c.N),
                            value=round(float(c.terminal_v if c.terminal_v is not None else c.Q), 3),
                            winp=round(self.winp(child), 3), x=cx, y=cy,
                            hops=self._hops(board, c)))
        pv = [h["san"] for h in out[0]["hops"]] if out else []
        return dict(game_over=False, result=None, winp=round(self.winp(board), 4),
                    x=px, y=py, candidates=out, pv=pv, nodes=total)

    # -- play --------------------------------------------------------------
    @staticmethod
    def _outcome(board: chess.Board):
        out = board.outcome(claim_draw=True)
        if out is None:
            return False, None
        res = ("white" if out.winner is True else
               "black" if out.winner is False else "draw")
        return True, res

    def engine_move(self, board: chess.Board) -> dict:
        from catspace.nn.mcts import game_truth
        over, res = self._outcome(board)
        if over:
            return dict(move=None, san=None, fen=board.fen(), game_over=True,
                        result=res, winp=round(self.winp(board), 4),
                        pv=[], candidates=[])
        with self.lock:
            root = self.pol.mcts.run(board)
        white = board.turn == chess.WHITE
        kids = list(root.children)
        best = None
        for c in kids:
            if game_truth(c) and (c.terminal_v > 0.5 if white else c.terminal_v < -0.5):
                best = c
                break
        if best is None:
            best = max(kids, key=lambda c: (c.N, (c.terminal_v if c.terminal_v
                       is not None else c.Q) * (1 if white else -1)))
        # candidates: top 6 by visits, each with the resulting state's (x,y)
        cand = sorted(kids, key=lambda c: c.N, reverse=True)[:6]
        px, py = self._xy(board)
        candidates = []
        for c in cand:
            child = board.copy(stack=False)
            child.push(c.move)
            cx, cy = self._xy(child)
            candidates.append(dict(
                uci=c.move.uci(), san=board.san(c.move), visits=int(c.N),
                value=round(float(c.terminal_v if c.terminal_v is not None else c.Q), 3),
                winp=round(self.winp(child), 3), x=cx, y=cy,
                hops=self._hops(board, c)))
        # PV: the best move's hop-line, as SAN (the line MCTS already built)
        pv = [h["san"] for h in self._hops(board, best)]
        after = board.copy(stack=False)
        san = board.san(best.move)
        after.push(best.move)
        over2, res2 = self._outcome(after)
        return dict(move=best.move.uci(), san=san, fen=after.fen(),
                    game_over=over2, result=res2, x=px, y=py,
                    winp=round(self.winp(after), 4), pv=pv, candidates=candidates)


ENGINE: Engine | None = None


def toy_fen() -> str:
    return TOY_FALLBACK


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj))

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    @staticmethod
    def _board(fen: str) -> chess.Board:
        return chess.Board(fen)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self._send(200, TEMPLATE.read_text(), "text/html; charset=utf-8")
            elif u.path.startswith("/assets/"):
                # serve vendored chessground JS/CSS (same-origin static files);
                # resolve() + prefix guard prevents path traversal
                rel = u.path[len("/assets/"):]
                fp = (ASSETS_DIR / rel).resolve()
                if not str(fp).startswith(str(ASSETS_DIR.resolve())) or not fp.is_file():
                    self._json({"error": "not found"}, 404)
                    return
                self._send(200, fp.read_bytes(), _MIME.get(fp.suffix, "application/octet-stream"))
            elif u.path == "/atlas":
                self._send(200, (ATLAS_DIR / "atlas.json").read_bytes())
            elif u.path == "/toy":
                self._json({"fen": toy_fen()})
            elif u.path == "/legal_moves":
                b = self._board(q["fen"][0])
                self._json({"moves": [m.uci() for m in b.legal_moves]})
            elif u.path == "/region_sample":
                cid = int(q["cluster"][0])
                i = int(q.get("i", ["0"])[0])
                cl = next(c for c in ENGINE.atlas["clusters"] if c["id"] == cid)
                pool = cl["fens"] or [toy_fen()]
                self._json({"fen": pool[i % len(pool)], "i": i % len(pool),
                            "count": len(pool)})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e)}, 400)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._body()
            if u.path == "/apply_move":
                b = self._board(body["fen"])
                mv = chess.Move.from_uci(body["uci"])
                if mv not in b.legal_moves:
                    self._json({"ok": False, "fen": b.fen(), "san": None,
                                "game_over": False, "result": None})
                    return
                san = b.san(mv)
                b.push(mv)
                over, res = Engine._outcome(b)
                self._json({"ok": True, "fen": b.fen(), "san": san,
                            "game_over": over, "result": res})
            elif u.path == "/engine_move":
                self._json(ENGINE.engine_move(self._board(body["fen"])))
            elif u.path == "/analyze":
                self._json({"ok": True, **ENGINE.analyze(self._board(body["fen"]),
                            int(body.get("topk", 3)), body.get("nodes"),
                            bool(body.get("extend", False)))})
            elif u.path == "/project":
                self._json({"ok": True, **ENGINE.project(self._board(body["fen"]))})
            elif u.path == "/set_board":
                fen = body["fen"]
                try:
                    b = self._board(fen)
                    valid = b.is_valid()
                except Exception:
                    self._json({"ok": True, "valid": False})
                    return
                over, res = Engine._outcome(b)
                self._json({"ok": True, "valid": bool(valid),
                            **ENGINE.project(b),
                            "legal_moves": [m.uci() for m in b.legal_moves],
                            "game_over": over, "result": res})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e)}, 400)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--phead", default="data/derived/sep/cert_base_full_phead.pt")
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    global ENGINE
    print(f"loading incumbent on CPU ({args.ckpt}) ...", flush=True)
    ENGINE = Engine(args.ckpt, args.phead, args.nodes)
    if not (ATLAS_DIR / "atlas.json").exists():
        print("WARNING: atlas.json missing — run experiments/viz/build_play_atlas.py "
              "(the map will 500 until then)", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"play-atlas server on http://{args.host}:{args.port}  "
          f"(mcts nodes={args.nodes}, model on cpu)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
