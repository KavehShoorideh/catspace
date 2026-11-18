"""PLAN 78.90 PLAY (SYSTEM_SPEC 7 the hint among the moves the search holds equal, 8 the wake, 9 the wake's match answers one question only): THE GROUPED REASONS WITH THE CELLS OF THE OTHER SIDE'S BEARING, AS FITTED ON BOTH SAMPLES
(variant A of symbols_own_measure), AS THE HINT IN PLAY. A sandbox: nothing is wired into any player of the wake; nothing is written to rules/learned or data/derived. Game-free: the game is named only through the pieces that
exist (symbols, symbols_play, symbols_groups_measure's cell reader, symbols_own_measure's own-cell reader, graph_layer1/2). The hint orders only the moves INSIDE the tie set the search hands back; the search's value, depth and
first move are the control's.

INTERFACE (written before the bodies).
  OwnReader()                        per process (use own_reader()): the reasons of A (symbols_own_measure.reasons_a over 78.81's reasons), their groups and the fit of A (fits.pkl of symbols_78_90/full), the engine's pattern counter.
      .pick(moves, tied) -> (move, detail)
                                     `tied` the moves the search holds equal, the search's first move FIRST. A reason that is a cell of the mover's bearing or of the other side's is carried by a tied move when its cell's count rose or fell
                                     across the move (the two cell readers read with the same four patterns), every other reason when the engine's pattern count says so (symbols.deltas_of); a group is carried when any member is; the
                                     score is the sum of the weights of the groups carried (events only, no controls: the pick the scores of 78.90 read). The move returned is ALWAYS in `tied`; a tie of scores gives tied[0].
                                     detail: {"scores", "carried": {move: [[group, weight]]}, "takes": None, "kinds": None}.
  MoverOwn(symbols_play.MoverReasons) the full reasons player of 78.82 with this pick where the spec says "fit": "own"; a player whose spec has no fit is 78.82's reasons player, unchanged. play90(spec) -> game: one game (module level;
                                     the Pool spawns workers that import it).
Called by: symbols_own_measure `match` and `agreement`.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import json
import pickle
from pathlib import Path

from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import symbols as SY
from catspace.incubator.analyst.discovery import symbols_groups_measure as SGM
from catspace.incubator.analyst.discovery import symbols_own_measure as O
from catspace.incubator.analyst.discovery import symbols_play as SP
from catspace.io import paths

FITS = (*O.OUT, "full", "fits.pkl")
TIE_DIGITS = SP.TIE_DIGITS
_READER = None


class OwnReader:
    """See the module doc."""

    def __init__(self):
        self.base = SP.reader82()
        R = self.base.reasons
        saved = pickle.loads(open(paths.experiment(*FITS), "rb").read())
        v = saved["variants"]["A"]
        self.S, self.groups, self.fit = v["R"], v["groups"], v["fit"]
        keys = sorted({int(r["key"]) for r in R})
        col = {k: j for j, k in enumerate(keys)}
        self.event = [(col[int(r["key"])], r["side"], r["dir"]) for r in self.S]
        pkeys, self.raws = O.bear_patterns(R)
        self.pat = {k: j for j, k in enumerate(pkeys)}
        self.w = dict(zip(self.fit["events"], self.fit["weights"]))

    def carried_groups(self, moves: list, tied: list) -> list:
        table = SY.deltas_of({"moves": list(moves)}, tied, self.base.counter)
        mover = SGM._split_job(([(list(moves), list(tied))], self.raws))
        other = O._own_job(([(list(moves), list(tied))], self.raws))
        out = []
        for i, m in enumerate(tied):
            have_m, have_o = set(mover[i]), set(other[i][0])
            on = set()
            for si, r in enumerate(self.S):
                if r["cell"] is None:
                    hit = self.event[si] in table.get(m, ())
                else:
                    hit = ((self.pat[int(r["key"])] * 4 + r["cell"]) * 2 + (0 if r["dir"] == "up" else 1)) in (have_m if r["side"] == "mover" else have_o)
                if hit:
                    on.add(si)
            out.append({gi for gi, g in enumerate(self.groups) if on.intersection(g)})
        return out

    def pick(self, moves: list, tied: list) -> "tuple[str, dict]":
        if len(tied) < 2:
            return tied[0], {"scores": {}, "carried": {}, "takes": None, "kinds": None}
        carried = self.carried_groups(moves, tied)
        score = {m: sum(self.w.get(g, 0.0) for g in carried[i]) for i, m in enumerate(tied)}
        detail = {"scores": score, "carried": {m: [[g, self.w[g]] for g in sorted(carried[i]) if g in self.w] for i, m in enumerate(tied)}, "takes": None, "kinds": None}
        s = {m: round(v, TIE_DIGITS) for m, v in score.items()}
        top = max(s.values())
        best = [m for m in tied if s[m] == top]
        return (tied[0] if tied[0] in best else best[0]), detail


def own_reader() -> OwnReader:
    global _READER
    if _READER is None:
        _READER = OwnReader()
    return _READER


class MoverOwn(SP.MoverReasons):
    """See the module doc."""

    def __init__(self, spec: dict, seconds: float, seed: str, trace: str):
        super().__init__(spec, seconds, seed, trace)
        self.fit_name = spec.get("fit")

    def pick(self, position: dict, tied: list) -> str:
        if self.fit_name != "own":
            return super().pick(position, tied)
        move, self.detail = own_reader().pick(position["moves"], tied)
        assert move in tied
        return move


def play90(spec: dict) -> dict:
    OM.Mover = MoverOwn
    TM.REG.clear()
    g = OM.play_game(spec)
    g["stats"] = [dict(m.stats) for m in TM.REG]
    return g
