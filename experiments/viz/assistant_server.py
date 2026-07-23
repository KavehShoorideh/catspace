#!/usr/bin/env python
"""experiments/viz/assistant_server.py -- THE PLANNER AS CO-ANALYST (Kaveh 2026-07-25):
play against a weak maia (or engine of choice) in the browser while OUR planner assists:
it prompts 'let's calculate here' (probe-triggered), searches while you think, then shows
the top moves and -- when a plan is active -- the most likely LEAVES you'll end up in.
Every suggested idea carries a pencil-editable tag; edited names persist to
artifacts/experiments/concept_tags.jsonl as HUMAN LABELS for field regions/plans
(concept-extraction meets the planner).

Run:  .venv/bin/python experiments/viz/assistant_server.py --port 8777
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import chess.engine
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catspace.engine.fields import FieldModel
from catspace.engine.introspection import ProbeKit
from catspace.nn.mcts import MCTS
from experiments.bootstrap_mate_engine import (OnlineMateBank, harvest, make_batched_energy_prior,
                                               make_boot_value, make_planner, mat_sig)

ASSETS = ROOT / "catspace/viz/assets"
TAGS = ROOT / "artifacts/experiments/concept_tags.jsonl"


class Session:
    def __init__(self, args):
        self.args = args
        self.fm = FieldModel(args.field, device=args.device)
        pfx = args.banks_prefix
        self.bank = OnlineMateBank(self.fm, Path(pfx + "_bank.fens"))
        self.loss = OnlineMateBank(self.fm, Path(pfx + "_lossbank.fens"))
        self.draw = OnlineMateBank(self.fm, Path(pfx + "_drawbank.fens"))
        for bk in (self.bank, self.loss, self.draw):
            bk.sync()
        self.ctx: dict = {"plan": "direct", "hist": {}}
        self.times: dict = {}
        self.vfn = make_boot_value(self.fm, self.bank, self.times, self.loss,
                                   dtm_ckpt=args.last_mile or None,
                                   draw_bank=self.draw, game_ctx=self.ctx)
        self.pfn, self.pfnb = make_batched_energy_prior(args.energy, game_ctx=self.ctx)
        self.planner = make_planner(self.fm, self.bank)
        self.probes = ProbeKit(self.fm, self.bank, self.loss, self.draw,
                               game_ctx=self.ctx, prior_fn=self.pfn)
        self.opp = None
        self._lm = args.last_mile
        self.version = self._version_of(self._lm)
        self.new_game(args.opponent)
        import threading
        threading.Thread(target=self._reloader, daemon=True).start()

    _CUM = {0: 300, 1: 1000, 2: 3000, 3: 6000, 4: 12000}    # per-class cumulative quotas

    def _version_of(self, ckpt: str) -> dict:
        name = Path(ckpt).stem if ckpt else "none"
        pct = None
        if name.startswith("dtm_tok_r"):
            try:
                k = int(name.rsplit("r", 1)[1])
                pct = round(100 * self._CUM.get(k, 12000) / 12000, 1)
            except ValueError:
                pass
        return {"model": name, "data_pct": pct}

    def _reloader(self):
        """CHECKPOINT AUTO-SWAP (Kaveh: 'every checkpoint we wanna restart the backend
        model; local server shouldn't need changing'): watch for new nucleus rounds,
        rebuild the value in place; the page and the game keep running."""
        import glob as _g
        import re as _re
        while True:
            time.sleep(45)
            try:
                cands = _g.glob(str(ROOT / "data/derived/sep/dtm_tok_r*.pt"))
                if not cands:
                    continue
                best = max(cands, key=lambda p: int(_re.search(r"r(\d+)\.pt", p).group(1)))
                if best != self._lm:
                    self.vfn = make_boot_value(self.fm, self.bank, self.times, self.loss,
                                               dtm_ckpt=best, draw_bank=self.draw,
                                               game_ctx=self.ctx)
                    self._lm = best
                    self.version = self._version_of(best)
                    print(f"[assistant] MODEL SWAPPED -> {self.version}", flush=True)
            except Exception as e:                          # noqa: BLE001
                print(f"[assistant] reload check failed: {e}", flush=True)

    def new_game(self, weights):
        if self.opp is not None:
            try:
                self.opp.quit()
            except Exception:
                pass
        self.opp = None
        try:
            self.opp = chess.engine.SimpleEngine.popen_uci(["lc0", f"--weights={weights}"])
        except Exception:
            try:                                    # container fallback: weak stockfish
                self.opp = chess.engine.SimpleEngine.popen_uci(["stockfish"])
                self.opp.configure({"Skill Level": 1})
                print("[assistant] lc0/maia unavailable -> stockfish skill 1", flush=True)
            except Exception:
                print("[assistant] no opponent engine -> analysis-only mode", flush=True)
        self.board = chess.Board()
        from collections import Counter
        self.ctx["hist"] = Counter({self.board.epd(): 1})
        self.ctx["plan"] = "direct"; self.ctx["target_pt"] = None
        self.idea_seq = 0

    def _prompt(self):
        """probe-triggered 'let's calculate here' with the reason."""
        s = self.probes.summary(self.board)
        reasons = []
        if s.get("child_dwin_margin", 1) < 0.02 and s.get("n_win", 0) > 0:
            reasons.append("the field is flat here -- intuition alone won't rank these moves")
        if s.get("prior_entropy", 0) > 2.6:
            reasons.append("many plausible candidates (high prior entropy)")
        if s.get("d_loss", float("inf")) < 15:
            reasons.append("we are NEAR remembered losses -- danger in the air")
        if s.get("seen_across_games", 0) == 0:
            reasons.append("never seen this position before")
        if s.get("clock_headroom", 100) < 30:
            reasons.append("the fifty-move clock is squeezing us")
        return {"suggest": bool(reasons), "reasons": reasons[:3], "probe": s}

    def calculate(self, nodes):
        b = self.board
        ps = self.planner(b, len(b.move_stack))
        self.ctx["plan"] = ps["plan"]; self.ctx["target_pt"] = ps.get("target_pt")
        snap = dict(self.times); t_run = time.time()
        m = MCTS(lambda bs: np.zeros(len(bs)), max_nodes=nodes, mate_stop=True,
                 pw_c=1.5, root_min_visits=10, value_fn=self.vfn, policy_fn=self.pfn,
                 policy_batch_fn=self.pfnb, batch_leaves=32)
        root = m.run(b.copy(stack=True))
        try:
            from catspace.metrics import observe
            tot = time.time() - t_run
            acc = 0.0
            for st, key in (("prior", "prior_s"), ("embF", "embedF_s"),
                            ("dbank", "dbank_s"), ("dtm", "dtm_s")):
                v = self.times.get(key, 0) - snap.get(key, 0)
                observe(st, v); acc += v
            observe("tree", max(tot - acc, 0)); observe("move_total", tot)
        except Exception:
            pass
        wins, losses, stales = harvest(root)
        self.bank.add(wins); self.loss.add(losses); self.draw.add(stales)
        kids = sorted(root.children, key=lambda c: -c.N)[:5]
        top = [{"uci": c.move.uci(), "san": b.san(c.move), "visits": int(c.N),
                "q": round(float(c.W / max(c.N, 1)), 3)} for c in kids]
        # most likely LEAVES under the plan: follow max-N children from each top move
        leaves = []
        for c in kids[:3]:
            node, line = c, [b.san(c.move)]
            depth = 0
            bb = b.copy(stack=False); bb.push(c.move)
            while node.children and depth < 8:
                node = max(node.children, key=lambda x: x.N)
                line.append(bb.san(node.move)); bb.push(node.move)
                depth += 1
            leaves.append({"line": " ".join(line), "fen": bb.fen(),
                           "visits": int(node.N),
                           "v": round(float(node.W / max(node.N, 1)), 3)})
        ideas = []
        def idea(kind, tag, detail):
            self.idea_seq += 1
            ideas.append({"id": f"i{int(time.time())}_{self.idea_seq}", "kind": kind,
                          "tag": tag, "detail": detail})
        if ps["plan"] != "direct":
            idea("plan", ps["plan"] + (f" -> {ps['goal']}" if ps.get("goal") else ""),
                 "planner's active plan (edit the name to what YOU would call it)")
        if top:
            idea("candidate", f"main idea: {top[0]['san']}",
                 f"most-visited move ({top[0]['visits']} visits, v={top[0]['q']})")
        pr = self._prompt()
        for r in pr["reasons"]:
            idea("sense", r, "why the assistant wanted to calculate here")
        return {"top": top, "leaves": leaves, "plan": ps["plan"],
                "goal": ps.get("goal"), "ideas": ideas, "probe": pr["probe"],
                "evals": int(m.evals_used)}


SES: Session | None = None
PAGE = (ROOT / "catspace/viz/templates/assistant.html")


USAGE = ROOT / "artifacts/experiments/usage.jsonl"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _usage(self, path, code, ms):
        try:
            with open(USAGE, "a") as f:
                f.write(json.dumps({"ts": time.time(), "path": path, "code": code,
                                    "ms": round(ms, 1)}) + "\n")
            from catspace.metrics import count, observe
            count(path); observe("http", ms / 1000.0)
        except Exception:
            pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        t = time.time()
        try:
            self._get()
        finally:
            self._usage(self.path, 200, (time.time() - t) * 1000)

    def _get(self):
        if self.path == "/metrics":
            from catspace.metrics import latest
            self._send(200, latest(), "text/plain; version=0.0.4")
        elif self.path == "/health":
            self._send(200, {"ok": True, "banks": {"win": len(SES.bank),
                                                   "loss": len(SES.loss),
                                                   "draw": len(SES.draw)}})
        elif self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE.read_bytes(), "text/html")
        elif self.path.startswith("/assets/"):
            f = ASSETS / self.path[len("/assets/"):]
            if f.exists():
                ct = "text/css" if f.suffix == ".css" else "application/javascript"
                self._send(200, f.read_bytes(), ct)
            else:
                self._send(404, {"err": "no asset"})
        elif self.path == "/state":
            b = SES.board
            dests = {}
            for mv in b.legal_moves:
                dests.setdefault(chess.square_name(mv.from_square), []).append(
                    chess.square_name(mv.to_square))
            self._send(200, {"fen": b.fen(), "turn": "w" if b.turn else "b",
                             "dests": dests, "over": b.is_game_over(claim_draw=True),
                             "result": b.result(claim_draw=True) if b.is_game_over(claim_draw=True) else None,
                             "version": SES.version})
        else:
            self._send(404, {"err": "?"})

    def do_POST(self):
        t = time.time()
        try:
            self._post()
        finally:
            self._usage(self.path, 200, (time.time() - t) * 1000)

    def _post(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/new":
            w = req.get("opponent", SES.args.opponent)
            SES.new_game(w)
            self._send(200, {"ok": True, "fen": SES.board.fen()})
        elif self.path == "/human_move":
            try:
                mv = chess.Move.from_uci(req["uci"])
                if mv not in SES.board.legal_moves:
                    raise ValueError("illegal")
                SES.board.push(mv)
                SES.ctx["hist"][SES.board.epd()] += 1
                reply = None
                if not SES.board.is_game_over(claim_draw=True) and SES.opp is not None:
                    r = SES.opp.play(SES.board, chess.engine.Limit(nodes=1))
                    reply = SES.board.san(r.move)
                    SES.board.push(r.move)
                    SES.ctx["hist"][SES.board.epd()] += 1
                self._send(200, {"ok": True, "fen": SES.board.fen(), "reply": reply,
                                 "assistant": SES._prompt()})
            except Exception as e:                          # noqa: BLE001
                self._send(400, {"err": str(e)})
        elif self.path == "/calculate":
            self._send(200, SES.calculate(int(req.get("nodes", 1500))))
        elif self.path == "/tag":
            with open(TAGS, "a") as f:
                f.write(json.dumps({"id": req.get("id"), "tag": req.get("tag"),
                                    "kind": req.get("kind"), "fen": SES.board.fen(),
                                    "ts": time.time()}) + "\n")
            self._send(200, {"ok": True})
        else:
            self._send(404, {"err": "?"})


def main():
    global SES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--field", default="data/derived/sep/lichess_mc2.pt")
    ap.add_argument("--energy", default="data/derived/sep/opponent_energy_v1.pt")
    ap.add_argument("--last-mile", default="data/derived/sep/dtm_cnn_v2.pt")
    ap.add_argument("--banks-prefix", default="artifacts/experiments/assistant")
    ap.add_argument("--opponent", default="data/engines/maia/maia-1200.pb.gz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    SES = Session(args)
    print(f"assistant on http://localhost:{args.port}  opponent={args.opponent}", flush=True)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
