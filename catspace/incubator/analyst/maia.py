"""maia.py -- Maia (lc0 + rating-conditioned weights) as a HUMAN-MOVE MODEL. Reads the policy at
nodes=1 from lc0's verbose move stats (P: xx%) and samples from it -- that is the Maia
protocol; lc0's own temperature does nothing at nodes=1 because there are no child visits.
One instance per rating; the ladder 1100..1900 is on disk under assets/engines/maia."""
from __future__ import annotations

import random
import re

import chess
import chess.engine

from catspace.io import paths

_P = re.compile(r"(\S+)\s.*\(P:\s*([\d.]+)%\)")


class Maia:
    def __init__(self, rating: int = 1100, weights: str | None = None):
        self.rating = rating
        self._eng = chess.engine.SimpleEngine.popen_uci(
            ["lc0", f"--weights={weights or paths.engine(f'maia/maia-{rating}.pb.gz')}", "--verbose-move-stats"])

    def policy(self, board: chess.Board) -> dict[str, float]:
        """uci -> probability over LEGAL moves (renormalised)."""
        legal = {m.uci() for m in board.legal_moves}
        pol = {}
        with self._eng.analysis(board, chess.engine.Limit(nodes=1)) as an:
            for info in an:
                s = info.get("string")
                if s:
                    m = _P.match(s)
                    if m and m.group(1) in legal:
                        pol[m.group(1)] = float(m.group(2))
        z = sum(pol.values()) or 1.0
        return {k: v / z for k, v in pol.items()}

    def top(self, board: chess.Board, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.policy(board).items(), key=lambda kv: -kv[1])[:k]

    def sample(self, board: chess.Board, temperature: float = 1.0, rng: random.Random | None = None) -> chess.Move:
        pol = self.policy(board)
        if not pol:
            return next(iter(board.legal_moves))
        if temperature <= 0:
            return chess.Move.from_uci(max(pol, key=pol.get))
        w = {k: v ** (1.0 / temperature) for k, v in pol.items()}
        z = sum(w.values())
        return chess.Move.from_uci((rng or random).choices(list(w), weights=[v / z for v in w.values()])[0])

    def quit(self):
        self._eng.quit()
