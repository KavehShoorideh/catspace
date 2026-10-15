"""THE TEACHER as a pluggable component (Kaveh 2026-10-15): windows.py sees only this protocol. Positions are FEN strings and moves UCI strings, as the domain
adapter (observables.ChessDomain) defines them; a tablebase, another engine, a learned policy or a non-chess scorer slots in by implementing it.
The teacher is the TEACHER / CHECKER only, never the measure.
"""
from __future__ import annotations

from typing import Protocol


class Teacher(Protocol):
    def score(self, fen: str, side: str) -> float:
        """the position's value for `side` (centipawn-like; forced mates are +-10000)."""

    def reasonable(self, fen: str, band: str) -> list:
        """[(move, score)]: the moves within the band's margin of the teacher's best, scored for the side to move."""

    def rollout(self, fen: str, move: str, plies: int) -> list:
        """the teacher's best play after `move` from `fen`: a line of moves starting with `move`, at least `plies` long when the position allows
        (it may be longer; the domain settles it to a quiet end)."""

    def close(self) -> None: ...


class StockfishTeacher:
    """Stockfish: knobs depth, multipv, margin (a dict by Elo band with a "default"). Single thread so a run is repeatable."""
    MATE_CP = 10000

    def __init__(self, depth: int = 14, multipv: int = 6, margin: dict | None = None, threads: int = 1, engine=None):
        import chess.engine
        self._ce = chess.engine
        self.depth, self.multipv = depth, multipv
        self.margin = margin or {"default": 60}
        self.eng = engine or chess.engine.SimpleEngine.popen_uci("stockfish")
        if engine is None:
            self.eng.configure({"Threads": threads, "Hash": 256})
        self._lines = {}
        self._multi = {}

    def close(self):
        self.eng.quit()

    def _limit(self):
        return self._ce.Limit(depth=self.depth)

    def score(self, fen, side):
        import chess
        b = chess.Board(fen)
        if b.is_game_over():
            return -self.MATE_CP if b.is_checkmate() and b.turn == (side == "white") else (self.MATE_CP if b.is_checkmate() else 0)
        sc = self.eng.analyse(b, self._limit())["score"].pov(side == "white")
        return sc.score(mate_score=self.MATE_CP)

    def reasonable(self, fen, band):
        import chess
        b = chess.Board(fen)
        if fen not in self._multi:
            rows = []
            for inf in self.eng.analyse(b, self._limit(), multipv=min(self.multipv, b.legal_moves.count())):
                pv = inf.get("pv") or []
                if pv:
                    cp = inf["score"].pov(b.turn).score(mate_score=self.MATE_CP)
                    rows.append((pv[0].uci(), cp))
                    self._lines[(fen, pv[0].uci())] = [m.uci() for m in pv]
            self._multi[fen] = rows
        rows = self._multi[fen]
        best = max(cp for _, cp in rows)
        m = self.margin.get(band, self.margin["default"])
        return [(u, cp) for u, cp in rows if cp >= best - m]

    def rollout(self, fen, move, plies):
        import chess
        if (fen, move) not in self._lines:
            b = chess.Board(fen)
            inf = self.eng.analyse(b, self._limit(), root_moves=[chess.Move.from_uci(move)])
            self._lines[(fen, move)] = [m.uci() for m in inf.get("pv") or [chess.Move.from_uci(move)]]
        return self._lines[(fen, move)]
