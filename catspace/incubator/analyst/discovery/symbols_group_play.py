"""PLAN 78.89 PLAY (SYSTEM_SPEC 7 the hint among the moves the search holds equal, 8 the wake, 9 the wake's match answers one question only): THE GROUPED REASONS WITH THE SPLIT, AS FITTED ON A SAMPLE, AS THE HINT IN PLAY. A sandbox:
nothing is wired into any player of the wake; nothing is written to rules/learned or data/derived. Game-free: the game is named only through the pieces that exist (symbols, symbols_play, symbols_groups_measure's cell reader,
graph_layer1/2). The hint orders only the moves INSIDE the tie set the search hands back; the search's value, depth and first move are the control's.

INTERFACE (written before the bodies).
  GroupReader()                      per process (use group_reader()): the reasons of 78.81 (symbols_play.reader82), their split (symbols.split_borne) and the groups of 78.88's variant ii (fits.pkl of symbols_78_88/full), and the
                                     fits by name: "old" (78.88 variant ii, one fit), "new" (78.89, one fit on the second sample's FIND), "newband" (78.89, one fit for each band of units).
      .pick(moves, tied, fit_name) -> (move, detail)
                                     `tied` the moves the search holds equal, the search's first move FIRST. A reason of the split is carried by a tied move when its cell's instance count rose or fell across the move (the cell reader of
                                     symbols_groups_measure), every other reason when the engine's pattern count says so (symbols.deltas_of); a group is carried when any member is; the score is the sum of the weights of the groups carried
                                     (events only, no controls: the pick the scores of 78.89 read). "newband" reads the fit of the band of the position's units. The move returned is ALWAYS in `tied`; a tie of scores gives tied[0].
                                     detail: {"scores", "carried": {move: [[group, weight]]}, "takes": None, "kinds": None}.
  MoverGroups(symbols_play.MoverReasons)   the window player of 78.83 with this pick: the spec carries "window" (low, high) and "fit" (a name above). play89(spec) -> game: one game (module level; the Pool spawns workers that import it).
Called by: symbols_sample_measure `match` for the players `fitfew` and `fitlow`.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import pickle

from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import symbols as SY
from catspace.incubator.analyst.discovery import symbols_groups_measure as SGM
from catspace.incubator.analyst.discovery import symbols_play as SP
from catspace.io import paths

FITS_NEW = ("symbols_78_89", "full", "fits.pkl")                  # this plan's fits (written by symbols_sample_measure `fit`)
FITS_OLD = ("symbols_78_88", "full", "fits.pkl")
TIE_DIGITS = SP.TIE_DIGITS
_READER = None


class GroupReader:
    """See the module doc."""

    def __init__(self):
        self.base = SP.reader82()
        R = self.base.reasons
        self.split = SY.split_borne(R, [])
        keys = sorted({int(r["key"]) for r in R})
        col = {k: j for j, k in enumerate(keys)}
        self.event = [(col[int(r["key"])], r["side"], r["dir"]) for r in self.split]
        pkeys, self.raws = SGM.split_patterns(R)
        self.pat = {k: j for j, k in enumerate(pkeys)}
        old = pickle.loads(open(paths.experiment(*FITS_OLD), "rb").read())
        new = pickle.loads(open(paths.experiment(*FITS_NEW), "rb").read())
        self.groups = old["variants"]["ii"]["groups"]
        assert [g for g in new["groups"]] == [g for g in self.groups]
        self.fits = {"old": old["variants"]["ii"]["fits"], "new": {"all": new["fits"]["all"]}, "newband": {b: new["fits"][b] for b, _l, _h in SGM.BANDS}}

    def carried_groups(self, moves: list, tied: list) -> list:
        """For each tied move the set of groups it carries."""
        table = SY.deltas_of({"moves": list(moves)}, tied, self.base.counter)
        cells = SGM._split_job(([(list(moves), list(tied))], self.raws))
        out = []
        for i, m in enumerate(tied):
            have = set(cells[i])
            on = set()
            for si, r in enumerate(self.split):
                if r["cell"] is None:
                    if self.event[si] in table.get(m, ()):
                        on.add(si)
                elif ((self.pat[int(r["key"])] * 4 + r["cell"]) * 2 + (0 if r["dir"] == "up" else 1)) in have:
                    on.add(si)
            out.append({gi for gi, g in enumerate(self.groups) if on.intersection(g)})
        return out

    def pick(self, moves: list, tied: list, fit_name: str) -> "tuple[str, dict]":
        if len(tied) < 2:
            return tied[0], {"scores": {}, "carried": {}, "takes": None, "kinds": None}
        fits = self.fits[fit_name]
        if "all" in fits:
            fit = fits["all"]
        else:
            band = LM.band_of(self.base.units_of(moves))
            fit = fits.get(band)
        w = dict(zip(fit["events"], fit["weights"])) if fit else {}
        carried = self.carried_groups(moves, tied)
        score = {m: sum(w.get(g, 0.0) for g in carried[i]) for i, m in enumerate(tied)}
        detail = {"scores": score, "carried": {m: [[g, w[g]] for g in sorted(carried[i]) if g in w] for i, m in enumerate(tied)}, "takes": None, "kinds": None}
        s = {m: round(v, TIE_DIGITS) for m, v in score.items()}
        top = max(s.values())
        best = [m for m in tied if s[m] == top]
        return (tied[0] if tied[0] in best else best[0]), detail


def group_reader() -> GroupReader:
    global _READER
    if _READER is None:
        _READER = GroupReader()
    return _READER


class MoverGroups(SP.MoverReasons):
    """See the module doc."""

    def __init__(self, spec: dict, seconds: float, seed: str, trace: str):
        super().__init__(spec, seconds, seed, trace)
        self.fit_name = spec.get("fit")

    def pick(self, position: dict, tied: list) -> str:
        move, self.detail = group_reader().pick(position["moves"], tied, self.fit_name)
        assert move in tied
        return move


def play89(spec: dict) -> dict:
    OM.Mover = MoverGroups
    TM.REG.clear()
    g = OM.play_game(spec)
    g["stats"] = [dict(m.stats) for m in TM.REG]
    return g
