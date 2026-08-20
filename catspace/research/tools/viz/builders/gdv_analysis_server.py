#!/usr/bin/env python
"""gdv_analysis_server.py -- play + analysis UI for the gdv graph policy with basic search.

Kaveh 2026-08-20: "put on top of it some basic search, allow me to control the search
depth, and put it into the UI so that I can do a game analysis with it and play with it."

A tiny stdlib HTTP server, model on CPU behind a lock (the MPS trainer stays untouched).
The frontend (chessground + chess.js, both already vendored) keeps ALL game state; the
server is stateless: every request carries a FEN.

  .venv/bin/python catspace/research/tools/viz/builders/gdv_analysis_server.py --port 8011
  # open http://localhost:8011

NOTE (background-QoS scar): if this session's harness spawns the server it runs at low
priority and can starve when idle -- for real use launch it from YOUR terminal (the
command above, or `! <command>` in the chat).

Endpoints: GET / (page), GET /assets/*, GET /health, POST /analyze {fen, depth, beam, topk}.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ASSETS_DIR = HERE.parent / "viz/assets"
TEMPLATE = HERE.parent / "viz/templates/gdv_analysis.html"
CTYPES = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml",
          ".png": "image/png", ".woff2": "font/woff2", ".map": "application/json"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/gdv1_latest.pt")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import chess
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_search import GdvSearch

    search = GdvSearch(args.ckpt, device=args.device)
    lock = threading.Lock()
    print(f"[gdv-ui] model {search.tag} on {args.device} | http://localhost:{args.port}",
          flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body if isinstance(body, bytes) else body.encode())

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(200, TEMPLATE.read_bytes(), "text/html")
            elif u.path == "/health":
                self._send(200, json.dumps({"ok": True, "model": search.tag}))
            elif u.path.startswith("/assets/"):
                fp = (ASSETS_DIR / u.path[len("/assets/"):]).resolve()
                if not str(fp).startswith(str(ASSETS_DIR.resolve())) or not fp.is_file():
                    self._send(404, json.dumps({"error": "not found"}))
                    return
                self._send(200, fp.read_bytes(), CTYPES.get(fp.suffix, "application/octet-stream"))
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            u = urlparse(self.path)
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            except Exception:
                self._send(400, json.dumps({"error": "bad json"}))
                return
            if u.path == "/analyze":
                try:
                    b = chess.Board(body["fen"])
                except Exception as e:
                    self._send(400, json.dumps({"error": f"bad fen: {e}"}))
                    return
                depth = max(1, min(int(body.get("depth", 2)), 5))
                beam = max(2, min(int(body.get("beam", 6)), 16))
                topk = max(1, min(int(body.get("topk", 10)), 48))
                t0 = time.time()
                with lock:
                    r = search.analyze(b, depth=depth, beam=beam, topk=topk)
                r["ms"] = int(1000 * (time.time() - t0))
                r["fen"] = body["fen"]
                self._send(200, json.dumps(r))
            else:
                self._send(404, json.dumps({"error": "not found"}))

    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
