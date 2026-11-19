"""PLAN 78.94 STEP THREE (SYSTEM_SPEC 7 "among equal moves we optimize for the prediction ... whichever gets us closer to our goal(s)", "how it is built" step (3); 8 the rule for choosing a move): THE CHOICE AMONG EQUAL MOVES BY THE FUTURE JUDGE. A prior: it orders the moves the search holds
equal and never scores a leaf (AGENTS.md rule 7); the wake calls the judge and never the oracle. Game-free: a move is a word, a position the moves played.

INTERFACE (written before the bodies).
  Chooser(judge, setting)                 `judge` a fitted future_judge.FutureJudge, `setting` (what, k): the head asked, "goal" | "win" | "lose" | "either" and the window in plies.
  .scores(items) -> [{move: (ours, theirs)}]   `items` [(moves, equal_moves)]: for each equal move, the judge's chance that a goal or winning point of the MOVER holds within k plies after it (the mover is the other side at the position it leaves) and the same chance for the other side
                                          (the mover there). One batch of inputs for all the positions after the moves (future_judge.Inputs).
  .choose(moves, equal_moves) -> move | None   the equal move with the greatest ours less theirs; None when every equal move has the same score (the caller falls back to its own order). A tie within TIE_DIGITS places is no preference.
  Called by: future_measure.py (stage human), the match player of stage five.
"""
from __future__ import annotations

import numpy as np

from catspace.incubator.analyst.discovery import future_judge as FJ

TIE_DIGITS = 6
CHUNK = 2500


class Chooser:
    """See the module doc."""

    def __init__(self, judge: FJ.FutureJudge, setting: tuple, inputs: FJ.Inputs, threads: int = 1):
        self.judge, self.setting, self.inputs, self.threads = judge, setting, inputs, threads

    def chances(self, wires: list) -> "tuple[np.ndarray, np.ndarray]":
        """For positions [(start, moves)] read at their last ply: (the chance for the side that just moved, the chance for the side to move)."""
        what, k = self.setting
        ours, theirs = [], []
        for a in range(0, len(wires), CHUNK):
            part = wires[a:a + CHUNK]
            inp = self.inputs.inputs_of(part, [[len(m)] for _s, m in part], self.threads)
            ours.append(self.judge.chance(FJ.side_matrix(inp, False), k, what))
            theirs.append(self.judge.chance(FJ.side_matrix(inp, True), k, what))
        return np.concatenate(ours), np.concatenate(theirs)

    def scores(self, items: list) -> list:
        wires = [(None, list(moves) + [m]) for moves, eq in items for m in eq]
        ours, theirs = self.chances(wires)
        out, i = [], 0
        for moves, eq in items:
            out.append({m: (float(ours[i + j]), float(theirs[i + j])) for j, m in enumerate(eq)})
            i += len(eq)
        return out

    def pick_of(self, eq: list, sc: dict) -> "str | None":
        val = {m: round(sc[m][0] - sc[m][1], TIE_DIGITS) for m in eq}
        top = max(val.values())
        best = [m for m in eq if val[m] == top]
        return best[0] if len(best) < len(eq) else None

    def choose(self, moves: list, equal_moves: list) -> "str | None":
        sc = self.scores([(moves, equal_moves)])[0]
        return self.pick_of(equal_moves, sc)
