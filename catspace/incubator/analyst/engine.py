"""Stockfish adapter -- a SERVICE to L4/L5, not a layer. python-chess UCI, multi-PV, restricted
root moves, the NULL-MOVE probe, and a sqlite cache keyed by (variant, fen, depth, multipv,
root_moves) so re-analysing a game costs nothing. journal_mode=DELETE on purpose (the WAL
disk-fill scar). Fairy-Stockfish is used for variants when it is installed."""
from __future__ import annotations

import json
import shutil
import sqlite3
from .nodes import tick
from dataclasses import dataclass, asdict

import chess
import chess.engine

from catspace.io import paths
from . import rules as R

MATE_CP = 10000


@dataclass(frozen=True)
class Line:
    move: str                 # uci
    pv: tuple[str, ...]       # uci
    cp: int                   # side-to-move POV, mates folded to +-MATE_CP
    mate: int | None          # MOVES to mate (python-chess Mate(n) = mate in n moves; signed, side-to-move POV) when known

    def san_line(self, board: chess.Board, n: int = 8) -> str:
        t, out = board.copy(stack=False), []
        for u in self.pv[:n]:
            m = chess.Move.from_uci(u)
            out.append(t.san(m)); t.push(m)
        return " ".join(out)


class Engine:
    def __init__(self, path: str | None = None, *, threads: int = 2, hash_mb: int = 256,
                 depth: int = 18, cache_path: str | None = None, variant: str = "chess"):
        if path is None:
            path = shutil.which("fairy-stockfish") if variant != "chess" else None
            path = path or shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"
        if variant != "chess" and "fairy" not in path:
            raise RuntimeError(f"variant {variant!r} needs fairy-stockfish (not found)")
        self.depth, self.variant, self.path = depth, variant, path
        self._eng = chess.engine.SimpleEngine.popen_uci(path)
        self._eng.configure({"Threads": threads, "Hash": hash_mb})
        self._db = sqlite3.connect(cache_path or paths.derived("analyst_cache.sqlite"))
        self._db.execute("PRAGMA journal_mode=DELETE")
        self._db.execute("CREATE TABLE IF NOT EXISTS lines (k TEXT PRIMARY KEY, v TEXT)")
        self.calls = self.hits = 0

    # -------------------------------------------------------------------------------------
    def analyse(self, board: chess.Board, *, multipv: int = 1, root_moves=None, depth: int | None = None) -> list[Line]:
        depth = depth or self.depth
        rm = tuple(sorted(m.uci() for m in root_moves)) if root_moves else ()
        k = json.dumps([self.variant, board.fen(), depth, multipv, rm])
        row = self._db.execute("SELECT v FROM lines WHERE k=?", (k,)).fetchone()
        if row:
            self.hits += 1
            return [Line(**d) for d in json.loads(row[0])]
        self.calls += 1; tick("engine")
        if board.is_game_over():
            return []
        if not board.is_valid():                               # the side not to move in check, a missing king...: Stockfish segfaults on these
            raise ValueError(f"not a legal position for the engine: {board.fen()} (status {board.status()!r})")
        infos = self._eng.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv,
                                  root_moves=list(root_moves) if root_moves else None)
        out = []
        for i in infos:
            pv = i.get("pv") or []
            if not pv:
                continue
            sc = i["score"].pov(board.turn)
            out.append(Line(pv[0].uci(), tuple(m.uci() for m in pv), sc.score(mate_score=MATE_CP), sc.mate()))
        self._db.execute("INSERT OR REPLACE INTO lines VALUES (?,?)", (k, json.dumps([asdict(l) for l in out])))
        self._db.commit()
        return out

    def best(self, board: chess.Board, depth: int | None = None) -> Line | None:
        ls = self.analyse(board, depth=depth)
        return ls[0] if ls else None

    def eval_move(self, board: chess.Board, move: chess.Move, depth: int | None = None) -> Line | None:
        ls = self.analyse(board, root_moves=[move], depth=depth)
        return ls[0] if ls else None

    def trajectory(self, board: chess.Board, move: chess.Move, depth: int | None = None) -> tuple[Line | None, dict[int, int]]:
        """Root-restricted search for ONE move with a CLEAN hash (fresh game => ucinewgame), keeping the
        score at every depth. Serves two purposes at the price of one search: the move's evaluation
        and its horizon trajectory (looks good shallow, collapses deep). Cached."""
        depth = depth or self.depth
        k = json.dumps([self.variant, board.fen(), depth, "traj", move.uci()])
        row = self._db.execute("SELECT v FROM lines WHERE k=?", (k,)).fetchone()
        if row:
            self.hits += 1
            d = json.loads(row[0])
            return (Line(**d["line"]) if d["line"] else None), {int(a): b for a, b in d["scores"].items()}
        self.calls += 1; tick("engine")
        scores, last = {}, None
        if not board.is_game_over():
            with self._eng.analysis(board, chess.engine.Limit(depth=depth), root_moves=[move], game=object()) as an:
                for info in an:
                    if "depth" in info and "score" in info and info.get("multipv", 1) == 1:
                        sc = info["score"].pov(board.turn)
                        scores[info["depth"]] = sc.score(mate_score=MATE_CP)
                        if info.get("pv"):
                            last = Line(info["pv"][0].uci(), tuple(m.uci() for m in info["pv"]), sc.score(mate_score=MATE_CP), sc.mate())
        self._db.execute("INSERT OR REPLACE INTO lines VALUES (?,?)", (k, json.dumps({"line": asdict(last) if last else None, "scores": scores})))
        self._db.commit()
        return last, scores

    def null_move_probe(self, board: chess.Board, *, multipv: int = 3, depth: int | None = None) -> list[Line] | None:
        """The opponent's best lines if the side to move PASSED. None when passing is illegal (in check)."""
        if board.is_check():
            return None
        nb = board.copy(stack=False)
        nb.push(chess.Move.null())
        return self.analyse(nb, multipv=multipv, depth=depth)

    def close(self):
        try:
            self._eng.quit()
        finally:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
