"""horizon.py -- HORIZON TRAPS: moves that look good up to some depth and turn bad beyond it
(Kaveh 2026-09-04: "these are the ones we need to explain"). For each candidate we record
Stockfish's score at every depth of a root-restricted search; a trap is a move that was near-best
at a shallow depth and is clearly worse at the final depth. Its COLLAPSE DEPTH -- the first depth
from which it stays bad -- is the amount of calculation needed to see the refutation, and the PV
at that depth is the refutation to explain. L5 instrument (uses the engine); no objective beyond
the engine's own sign."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import chess.engine

from .engine import MATE_CP


@dataclass
class Trajectory:
    move: str                          # uci
    san: str
    scores: dict[int, int] = field(default_factory=dict)   # depth -> cp (side-to-move POV)
    pvs: dict[int, list[str]] = field(default_factory=dict)

    def at(self, d: int) -> int | None:
        ds = [k for k in self.scores if k <= d]
        return self.scores[max(ds)] if ds else None


@dataclass
class Trap:
    traj: Trajectory
    peak_depth: int                    # a depth at which it was near-best
    peak_cp: int
    collapse_depth: int                # first depth from which it stays clearly worse than the best move
    final_cp: int
    best_san: str
    best_final_cp: int
    refutation: list[str]              # PV (uci) at the collapse depth


def trajectories(board: chess.Board, engine_path: str, moves: list[chess.Move], depth: int = 16,
                 threads: int = 2, hash_mb: int = 128) -> list[Trajectory]:
    """One root-restricted search per candidate, keeping the score at every depth."""
    eng = chess.engine.SimpleEngine.popen_uci(engine_path)
    eng.configure({"Threads": threads, "Hash": hash_mb})
    out = []
    try:
        for m in moves:
            t = Trajectory(m.uci(), board.san(m))
            with eng.analysis(board, chess.engine.Limit(depth=depth), root_moves=[m]) as an:
                for info in an:
                    if "depth" in info and "score" in info and info.get("multipv", 1) == 1:
                        t.scores[info["depth"]] = info["score"].pov(board.turn).score(mate_score=MATE_CP)
                        if info.get("pv"):
                            t.pvs[info["depth"]] = [x.uci() for x in info["pv"]]
            out.append(t)
    finally:
        eng.quit()
    return out


import math


def winp(cp: int) -> float:
    """Centipawns -> win probability in percent (lichess's logistic), the scale a player feels."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * max(-3000, min(3000, cp)))) - 1)


def find_traps(trajs: list[Trajectory], depth: int, drop_pts: float = 10.0, near_pts: float = 5.0, min_peak: int = 2) -> list[Trap]:
    """A trap = a move whose OWN win-probability falls by >= drop_pts from a shallow peak and that
    ends near_pts or more below the best move; collapse depth = first depth from which it stays
    that far below its peak. Thresholds are in win-% points, not centipawns, so they mean the same
    thing at +0.3 and at +3.0."""
    if not trajs:
        return []
    final = {t.move: t.at(depth) for t in trajs}
    best = max(trajs, key=lambda t: final[t.move] if final[t.move] is not None else -MATE_CP)
    bf = final[best.move]
    traps = []
    for t in trajs:
        f = final[t.move]
        if f is None or t.move == best.move or winp(bf) - winp(f) < near_pts:
            continue
        peak = max((d for d in t.scores if d >= min_peak), key=lambda d: winp(t.scores[d]), default=None)
        if peak is None or winp(t.scores[peak]) - winp(f) < drop_pts:
            continue
        collapse = None
        for d in sorted(t.scores):
            if d <= peak:
                continue
            if all(winp(t.scores[peak]) - winp(t.scores[dd]) >= drop_pts for dd in t.scores if dd >= d):
                collapse = d
                break
        if collapse is None:
            continue
        traps.append(Trap(t, peak, t.scores[peak], collapse, f, best.san, bf, t.pvs.get(collapse, [])))
    return traps
