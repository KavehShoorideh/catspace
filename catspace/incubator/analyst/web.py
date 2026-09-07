"""web.py -- the analyst as a local app (Kaveh 2026-09-05): one server on your machine, one page.
Paste a Lichess game link (or id) and step through the game with the analysis of each move; paste a user
name and get the review of their recent games, each error opening the game at that move. Every view has a
"something is not right" button: it appends what you were looking at to feedback/flags.jsonl (a queue
the coder reads; each line is one flag, its id = its line number) so the explanation can be fixed later.
    python -m catspace.incubator.analyst.web --port 8765      then open http://127.0.0.1:8765
No service: the engine, Maia and the cache are the ones on this machine; Lichess is only asked for PGNs."""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import chess.pgn

from .engine import Engine
from .ideas import analyse_position
from .maia import Maia
from .review import annotate, ERR
from .structure import top_changes
from .reach import progress_lines
from .memo import Memo
from .plans import plans_for, plan_text
from .subgames import holdings_lines
from .regime import regime_lines
from .goal_search import pursue, result_text
from .verbalize import cp_str, render, best_fact, motif_phrase, delta_str
from .horizon import winp

ROOT = Path(__file__).resolve().parents[3]
DERIVED = ROOT / "data" / "derived" / "web"
LICHESS_DIR = ROOT / "data" / "lichess"
FEEDBACK = ROOT / "feedback"
FLAGS = FEEDBACK / "flags.jsonl"
MEMO_PATH = ROOT / "data" / "derived" / "analyst_memo.sqlite"
PAGE = Path(__file__).with_name("web_page.html")
ASSETS = ROOT / "catspace" / "research" / "tools" / "viz" / "viz" / "assets"      # the vendored chessground + chess.js of the play-atlas app
MIME = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".json": "application/json", ".html": "text/html; charset=utf-8"}
UA = "catspace-analyst/0.1 (local desktop tool)"
RATINGS = (1100, 1300, 1500, 1700, 1900)
GAME_ID = re.compile(r"^[A-Za-z0-9]{8}$")
FACT_KEYS = ("takes", "hanging", "forks", "pins", "skewers", "checkmate", "discovering", "attacks", "attacked", "with check",
             "overloaded", "two jobs", "lured", "deflect", "guard", "cut", "won on the exchange")


# ---------------------------------------------------------------- input parsing and Lichess fetches
def parse_input(text: str) -> tuple[str, str] | None:
    """('game', id) | ('user', name) | None. A Lichess game id is 8 case-sensitive characters; a bare token with
    both upper and lower case letters is read as a game id, anything else as a user name."""
    t = text.strip()
    m = re.search(r"lichess\.org/@/([A-Za-z0-9_-]+)", t)
    if m:
        return ("user", m.group(1))
    m = re.search(r"lichess\.org/([A-Za-z0-9]{8})", t)
    if m:
        return ("game", m.group(1))
    if GAME_ID.match(t) and any(c.isupper() for c in t) and any(c.islower() for c in t):
        return ("game", t)
    if re.match(r"^[A-Za-z0-9_-]{2,30}$", t):
        return ("user", t)
    return None


def fetch(url: str, accept: str = "application/x-chess-pgn") -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def game_pgn(gid: str) -> str:
    path = LICHESS_DIR / f"{gid}.pgn"
    if path.exists():
        return path.read_text()
    text = fetch(f"https://lichess.org/game/export/{gid}?evals=1&clocks=0")
    LICHESS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def user_pgn(name: str, max_games: int) -> Path:
    path = LICHESS_DIR / f"{name}_web.pgn"
    text = fetch(f"https://lichess.org/api/games/user/{name}?max={max_games}&perfType=blitz,rapid,classical&evals=true&opening=true")
    LICHESS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def split_games(path: Path) -> list[tuple[str, str]]:
    """[(game id, pgn text)] in file order."""
    out = []
    with open(path) as f:
        while (g := chess.pgn.read_game(f)):
            gid = g.headers.get("GameId") or g.headers.get("Site", "").rstrip("/").split("/")[-1]
            out.append((gid[:8], str(g)))
    return out


# ---------------------------------------------------------------- the game analysis (in-process)
class Analyst:
    """Owns the engine and the Maia ladder for the worker thread."""

    def __init__(self, depth: int, threads: int):
        self.depth = depth
        self.eng = Engine(depth=depth, threads=threads)
        self.maias = {r: Maia(r) for r in RATINGS}
        self.memo = Memo.load(str(MEMO_PATH))                    # the concept store: shared across games and runs

    def close(self):
        self.eng.close()
        for m in self.maias.values():
            m.quit()

    def game(self, gid: str, pgn_text: str, progress) -> dict:
        game = chess.pgn.read_game(io.StringIO(pgn_text)); h = game.headers
        moves = list(game.mainline_moves())
        board = game.board(); boards, sans = [board.copy()], []
        plies = []
        for i, mv in enumerate(moves):
            us = board.turn
            lad = {r: self.maias[r].policy(board) for r in RATINGS}
            prev = boards[-3] if len(boards) >= 3 else None
            rec = analyse_position(board, self.eng, prev=prev, killed_by=sans[-1] if sans else None, played=mv, multipv=3)
            row, _ = annotate(board, mv, rec, self.eng, us, lad, list(RATINGS), level=0, depth=self.depth, headers=h)
            pl = rec.played
            e0 = pl.effects[0] if pl and pl.effects else None
            fact = best_fact(e0, strict=True) if e0 else None
            fact = fact if fact and any(k in fact for k in FACT_KEYS) else None
            mp = motif_phrase(board, pl.pv) if pl and pl.pv else ""
            if not (mp or fact) and e0 is not None:
                sc = top_changes(e0.structure_created, e0.structure_ended, k=1)
                fact = sc[0] if sc else None
            ply = {
                "ply": i, "move_no": i // 2 + 1, "side": "w" if us else "b", "san": board.san(mv), "uci": mv.uci(), "fen": board.fen(),
                "eval_before": cp_str(rec.eval_cp, us, rec.idea.mate if rec.idea else None),
                "eval_after": cp_str(pl.cp, us, pl.mate) if pl else "?",
                "verdict": rec.verdict, "cost": delta_str(rec) if pl else "",
                "cost_wp": round(abs(winp(rec.eval_cp) - winp(pl.cp))) if pl else 0,
                "headline": row.get("headline") if rec.verdict in ERR else None,
                "fact": mp or fact, "best": rec.idea.san if rec.idea else None, "best_uci": rec.idea.uci if rec.idea else None,
                "best_line": rec.idea.line_san if rec.idea else None,
                "text": render(rec, alternatives=False), "text_long": render(rec, long=True, alternatives=True),
                "structure": top_changes(e0.structure_created, e0.structure_ended, k=3) if e0 else [],
                "threats": [t.san_line(board, 4) for t in rec.threats if t.real][:3],
                "progress": progress_lines(board, us, memo=self.memo),
                "plans": [plan_text(board, pl) for pl in plans_for(board, us, memo=self.memo, top=3)],
                "holds": holdings_lines(board, us, memo=self.memo),
                "regime": regime_lines(board, memo=self.memo),
                "p_played": {str(k): round(v * 100) for k, v in row["p_played"].items()},
                "p_best": {str(k): round(v * 100) for k, v in row["p_best"].items()},
                "cause": (row.get("cause") or {}).get("cause"), "horizon": row.get("horizon_text", ""),
                "refutation": row.get("refutation"), "computer_move": row.get("computer_move", False),
            }
            plies.append(ply)
            sans.append(board.san(mv)); board.push(mv); boards.append(board.copy())
            progress(i + 1, len(moves), plies)
        self.memo.save()
        return {"id": gid, "headers": {k: h.get(k, "") for k in ("White", "Black", "WhiteElo", "BlackElo", "Result", "Date", "TimeControl", "Opening", "ECO", "Event")},
                "memo": self.memo.report(),
                "depth": self.depth, "plies": plies, "summary": {s: side_summary(plies, s) for s in ("w", "b")},
                "final_fen": board.fen(), "analysed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def side_summary(plies: list[dict], side: str) -> dict:
    mine = [p for p in plies if p["side"] == side]
    vc = Counter(p["verdict"] for p in mine)
    errs = [p for p in mine if p["verdict"] in ERR]
    ll = {r: sum(math.log(max(p["p_played"].get(str(r), 0) / 100, 1e-3)) for p in mine) / max(len(mine), 1) for r in RATINGS}
    eff = max(ll, key=ll.get) if mine else None
    causes = Counter(p["cause"] for p in errs if p["cause"])
    return {"moves": len(mine), "verdicts": dict(vc), "errors": len(errs), "effective_rating": eff,
            "causes": dict(causes), "worst": sorted(errs, key=lambda p: -p["cost_wp"])[:5]}


# ---------------------------------------------------------------- the user review (the parallel driver, parsed)
def user_review(name: str, max_games: int, workers: int, progress) -> dict:
    progress(0, 1, None, "fetching games from Lichess")
    path = user_pgn(name, max_games)
    games = split_games(path)
    progress(0, len(games), None, f"{len(games)} games; running the review")
    out = ROOT / "tmp" / f"review_{name}_web.txt"
    cmd = [sys.executable, "-m", "catspace.incubator.analyst.parallel", str(path), "--user", name, "--workers", str(workers), "--out", str(out)]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-2000:])
    text = out.read_text()
    summary = text.split("=" * 80)[0].strip().splitlines()
    rows = [json.loads(l) for l in open(out.with_suffix(".jsonl")) if l.strip()]
    by_game = {}
    for i, (gid, pgn_text) in enumerate(games, 1):
        g = chess.pgn.read_game(io.StringIO(pgn_text)); h = g.headers
        by_game[i] = {"no": i, "id": gid, "white": h.get("White", ""), "black": h.get("Black", ""), "result": h.get("Result", ""),
                      "date": h.get("Date", ""), "tc": h.get("TimeControl", ""), "opening": h.get("Opening", ""),
                      "you": "w" if h.get("White", "").lower() == name.lower() else "b", "errors": [], "moves": 0}
    for r in rows:
        g = by_game.get(r["game"])
        if not g: continue
        g["moves"] += 1
        if r["verdict"] in ERR and r.get("headline"):
            g["errors"].append({"ply": r["ply"], "san": r["san"], "verdict": r["verdict"], "headline": r["headline"],
                                "cost_wp": round(abs(winp(r["eval_cp"]) - winp(r["eval_cp"] - r["drop"]))), "fen": r["fen"],
                                "cause": (r.get("cause") or {}).get("cause"), "computer_move": r.get("computer_move", False)})
    return {"user": name, "summary": summary, "games": [by_game[i] for i in sorted(by_game)], "n_moves": len(rows),
            "analysed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


# ---------------------------------------------------------------- jobs
class Jobs:
    def __init__(self, depth: int, threads: int, workers: int, max_games: int):
        self.q: queue.Queue = queue.Queue()
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.depth, self.threads, self.workers, self.max_games = depth, threads, workers, max_games
        self.pgn_cache: dict[str, str] = {}
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def submit(self, kind: str, key: str) -> dict:
        jid = f"{kind}:{key}"
        with self.lock:
            j = self.jobs.get(jid)
            if j and j["status"] in ("queued", "running"):
                return j
            j = {"id": jid, "kind": kind, "key": key, "status": "queued", "progress": {"done": 0, "total": 0, "message": "queued"},
                 "partial": None, "error": None, "started": time.time()}
            self.jobs[jid] = j
        self.q.put(jid)
        return j

    def _run(self):
        analyst = None
        while True:
            jid = self.q.get()
            j = self.jobs[jid]
            j["status"] = "running"
            try:
                if j["kind"] == "goals":
                    if analyst is None:
                        analyst = Analyst(self.depth, self.threads)
                    fen, side = j["fen"], j["side"]
                    b = chess.Board(fen)
                    rs = pursue(b, side == "w", memo=analyst.memo, engine=analyst.eng, budget=300, horizon=6, max_goals=4, engine_depth=8)
                    j["result"] = [result_text(r) for r in rs]
                    j["status"] = "done"; continue
                if j["kind"] == "game":
                    if analyst is None:
                        analyst = Analyst(self.depth, self.threads)
                    gid = j["key"]
                    pgn_text = self.pgn_cache.get(gid) or game_pgn(gid)
                    def prog(done, total, plies):
                        j["progress"] = {"done": done, "total": total, "message": f"move {done} of {total}"}
                        j["partial"] = plies
                    result = analyst.game(gid, pgn_text, prog)
                    (DERIVED / "games").mkdir(parents=True, exist_ok=True)
                    (DERIVED / "games" / f"{gid}.json").write_text(json.dumps(result))
                else:
                    name = j["key"]
                    def prog2(done, total, _unused, message=""):
                        j["progress"] = {"done": done, "total": total, "message": message}
                    result = user_review(name, self.max_games, self.workers, prog2)
                    for gid, pgn_text in split_games(LICHESS_DIR / f"{name}_web.pgn"):
                        self.pgn_cache[gid] = pgn_text
                    (DERIVED / "users").mkdir(parents=True, exist_ok=True)
                    (DERIVED / "users" / f"{name.lower()}.json").write_text(json.dumps(result))
                j["status"] = "done"; j["partial"] = None
            except Exception as e:                                             # a failed job must not kill the worker
                j["status"] = "error"; j["error"] = f"{e}\n{traceback.format_exc()[-1500:]}"


# ---------------------------------------------------------------- flags: the queue the coder reads
def append_flag(payload: dict) -> dict:
    FEEDBACK.mkdir(parents=True, exist_ok=True)
    n = sum(1 for _ in open(FLAGS)) if FLAGS.exists() else 0
    rec = {"id": n + 1, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "open", **payload}
    with open(FLAGS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    jobs: Jobs = None   # set by serve()

    def log_message(self, fmt, *args):            # quiet
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        from urllib.parse import unquote
        path = unquote(self.path.split("?")[0])
        if path in ("/", "/index.html"):
            data = PAGE.read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data); return
        if path.startswith("/assets/"):
            fp = (ASSETS / path[len("/assets/"):]).resolve()
            if not str(fp).startswith(str(ASSETS.resolve())) or not fp.is_file():
                return self._json({"error": "not found"}, 404)
            data = fp.read_bytes()
            self.send_response(200); self.send_header("Content-Type", MIME.get(fp.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "max-age=3600"); self.end_headers(); self.wfile.write(data); return
        if path.startswith("/api/job/"):
            j = self.jobs.jobs.get(path[len("/api/job/"):])
            return self._json(j or {"error": "no such job"}, 200 if j else 404)
        if path.startswith("/api/game/"):
            gid = path[len("/api/game/"):]
            f = DERIVED / "games" / f"{gid}.json"
            if f.exists():
                return self._json({"status": "done", "result": json.loads(f.read_text())})
            j = self.jobs.submit("game", gid)
            return self._json({"status": j["status"], "job": j["id"]})
        if path.startswith("/api/user/"):
            name = path[len("/api/user/"):]
            f = DERIVED / "users" / f"{name.lower()}.json"
            if f.exists() and "refresh" not in self.path:
                return self._json({"status": "done", "result": json.loads(f.read_text())})
            j = self.jobs.submit("user", name)
            return self._json({"status": j["status"], "job": j["id"]})
        if path == "/api/flags":
            rows = [json.loads(l) for l in open(FLAGS)] if FLAGS.exists() else []
            return self._json(rows[-50:])
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/resolve":
            body = self._body()
            kind = parse_input(body.get("input", ""))
            return self._json({"kind": kind[0], "key": kind[1]} if kind else {"error": "paste a Lichess game link, a game id or a user name"}, 200 if kind else 400)
        if path == "/api/goals":
            body = self._body()
            fen, side = body.get("fen", ""), body.get("side", "w")
            try:
                chess.Board(fen)
            except Exception:
                return self._json({"error": "bad fen"}, 400)
            import hashlib
            key = hashlib.sha1(f"{fen}|{side}".encode()).hexdigest()[:12]
            j = self.jobs.submit("goals", key)
            j["fen"], j["side"] = fen, side
            return self._json({"status": j["status"], "job": j["id"]})
        if path == "/api/flag":
            body = self._body()
            note = (body.get("note") or "")[:2000]
            rec = append_flag({"note": note, "context": body.get("context") or {}})
            return self._json({"id": rec["id"], "file": str(FLAGS.relative_to(ROOT)), "line": rec["id"]})
        self._json({"error": "not found"}, 404)


def serve(port: int, depth: int, threads: int, workers: int, max_games: int, host: str = "127.0.0.1"):
    Handler.jobs = Jobs(depth, threads, workers, max_games)
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"analyst web: http://{host}:{port}  (depth {depth}, {threads} engine threads; flags -> {FLAGS.relative_to(ROOT)})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="bind address; 0.0.0.0 to reach it from other devices (e.g. over Tailscale)")
    ap.add_argument("--depth", type=int, default=18, help="engine depth for single games")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4, help="review workers for a user's games")
    ap.add_argument("--max-games", type=int, default=30, help="how many recent games to fetch for a user")
    a = ap.parse_args(argv)
    serve(a.port, a.depth, a.threads, a.workers, a.max_games, a.host)


if __name__ == "__main__":
    main()
