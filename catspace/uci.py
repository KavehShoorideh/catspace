"""
uci.py — UCIBoardPolicy: any UCI engine (Stockfish here) as a BoardPolicy.
Strength control: elo -> UCI_LimitStrength/UCI_Elo (Stockfish floor 1320),
or skill -> Skill Level (0..20; 0 is much weaker than Elo 1320), plus the
usual nodes/movetime/depth limits. Context manager owns the engine process.
"""
from __future__ import annotations

import chess
import chess.engine
import numpy as np


class UCIBoardPolicy:
    def __init__(self, cmd: str = "stockfish", movetime: float | None = 0.05,
                 nodes: int | None = None, depth: int | None = None,
                 elo: int | None = None, skill: int | None = None, threads: int = 1):
        self.cmd = cmd
        self.limit = chess.engine.Limit(
            time=movetime if nodes is None and depth is None else None,
            nodes=nodes, depth=depth)
        self.elo = elo
        self.skill = skill
        self.threads = threads
        self.engine: chess.engine.SimpleEngine | None = None

    def __enter__(self) -> "UCIBoardPolicy":
        self.engine = chess.engine.SimpleEngine.popen_uci(self.cmd)
        opts: dict = {"Threads": self.threads}
        if self.elo is not None:
            opts["UCI_LimitStrength"] = True
            opts["UCI_Elo"] = max(1320, self.elo)
        if self.skill is not None:
            opts["Skill Level"] = self.skill
        self.engine.configure(opts)
        return self

    def __exit__(self, *exc) -> None:
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        assert self.engine is not None, "use UCIBoardPolicy as a context manager"
        return self.engine.play(board, self.limit).move
