"""PLAN 78.96 PART TWO AND THREE, THE FITTED PRIOR (SYSTEM_SPEC 7 "the prior" and "among equal moves ... whichever gets us closer to our goal(s)", 9 "the oracle labels positions and frames in sleep, for the prior" and THE WALL: the wake never calls the oracle and reads none of its
numbers). A PICK AMONG THE MOVES THE SEARCH HOLDS EQUAL, fitted in sleep on which equal move KEEPS the oracle's class (prior_equal.py made the labels); at play it reads only the two things the existing picks already read: the change in fact patterns a move makes (78.81's reasons, the
counter of 78.82's reader) and the estimator's inputs after the move (the inputs of 78.94's judge), and whether the move takes a unit. No new fact. This file and what it imports at run time NEVER import the oracle wrapper (a test loads it in a fresh process and shows it);
the oracle's class is the target of the fit and appears nowhere in the inputs. It orders the moves the search holds equal and never scores a leaf (AGENTS.md rule 7). Game-free: a move is a word, a position the moves played.

INTERFACE (written before the bodies).
  Features(reader, inputs, threads=1)           the readers of the existing picks (symbols_play.ReasonsReader, future_judge.Inputs).
    .rows(items) -> [Rows]                       `items` [(moves, tied)]; for each, the candidate matrix of the position: columns = the pattern indicators (the reasons' own order), the estimator's inputs after the move for the side that moved and for the side to move, and 1 when the move takes
                                                 a unit; one row a tied move. Rows = {"x": float32 [n, d], "takes": [n]} (the matrix already holds the takes column).
  centred(x) -> x - mean over the rows            a choice is among the moves of one position, so every column is read against the position's own mean.
  Prior(head, features)                         `head` a fitted head of future_judge (fit_linear or fit_tree), `features` a Features.
    .scores(items) -> [array]                    per position the head's chance, per tied move, that the move keeps the class (a probability of the fitted target; no oracle is read).
    .choose(moves, tied) -> move | None          the move with the greatest chance; None when every tied move has the same chance (the caller falls back to the search's own move). A tie within TIE_DIGITS places is no preference.
  Called by: prior_measure.py (stages fit, confirm), prior_play.py (the match player).
"""
from __future__ import annotations

import numpy as np

from catspace.incubator.analyst.discovery import future_judge as FJ
from catspace.incubator.analyst.discovery import symbols as SY

TIE_DIGITS = 6
CHUNK = 1500


def centred(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


class Features:
    """See the module doc."""

    def __init__(self, reader, inputs: FJ.Inputs, threads: int = 1):
        self.reader, self.inputs, self.threads = reader, inputs, threads

    def pattern_rows(self, moves: list, tied: list) -> np.ndarray:
        table = SY.deltas_of({"moves": list(moves)}, tied, self.reader.counter)
        order = self.reader.order
        out = np.zeros((len(tied), len(order)), np.float32)
        for i, m in enumerate(tied):
            have = table.get(m, ())
            for j, e in enumerate(order):
                if e in have:
                    out[i, j] = 1.0
        return out

    def rows(self, items: list) -> list:
        out = []
        for a in range(0, len(items), max(1, CHUNK // 20)):
            part = items[a:a + max(1, CHUNK // 20)]
            wires = [(None, list(moves) + [m]) for moves, tied in part for m in tied]
            inp = self.inputs.inputs_of(wires, [[len(w[1])] for w in wires], self.threads)
            xs, xt = FJ.side_matrix(inp, False), FJ.side_matrix(inp, True)
            off = 0
            for moves, tied in part:
                n = len(tied)
                takes_d, _k = self.reader.controls_of(list(moves), list(tied))
                takes = np.array([[float(bool(takes_d[m]))] for m in tied], np.float32)
                x = np.hstack([self.pattern_rows(moves, tied), xs[off:off + n], xt[off:off + n], takes])
                out.append({"x": x.astype(np.float32), "takes": takes[:, 0]})
                off += n
        return out


class Prior:
    """See the module doc."""

    def __init__(self, head: dict, features: Features):
        self.head, self.features = head, features

    def scores_of(self, rows: list) -> list:
        out = []
        for r in rows:
            if self.head.get("m") is None:
                out.append(np.zeros(len(r["x"])))
            else:
                out.append(FJ.predict(self.head, centred(r["x"])))
        return out

    def scores(self, items: list) -> list:
        return self.scores_of(self.features.rows(items))

    @staticmethod
    def pick_of(tied: list, sc: np.ndarray) -> "str | None":
        val = np.round(np.asarray(sc, float), TIE_DIGITS)
        best = np.flatnonzero(val == val.max())
        return tied[int(best[0])] if len(best) < len(tied) else None

    def choose(self, moves: list, tied: list) -> "str | None":
        if len(tied) < 2:
            return None
        return self.pick_of(list(tied), self.scores([(list(moves), list(tied))])[0])
