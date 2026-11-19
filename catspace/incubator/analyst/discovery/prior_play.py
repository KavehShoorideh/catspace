"""PLAN 78.96 PART THREE, PLAY (SYSTEM_SPEC 7 "among equal moves ... whichever gets us closer to our goal(s)", 8 the wake, 9 THE WALL: the wake never calls the oracle and reads none of its numbers; 12 the hint in play). THE MATCH PLAYER OF THE FITTED PRIOR in the match harness of
78.67, 78.82 and 78.95. A sandbox: nothing is wired into any player of the wake; nothing is written to rules/learned or data/derived; outputs under artifacts/experiments/prior_78_96/<tag>/matches/. Game-free: the game is named only through the pieces that exist.

THE WALL. This module and what it imports at run time never import the oracle wrapper (ladder_oracle) and never read a score of the oracle: the prior it loads (prior.pkl) was fitted in sleep on stored labels, its inputs are the pattern changes and the estimator's inputs, the labels'
own file is not opened here. A test imports this module in a fresh process and shows that the wrapper is not among the loaded modules.

THE HINT IS NEVER A VALUE: the player is the SAME search call as the control (`solve_equals_batch`: the equal moves handed back, the search's own move first); only the choice among the equal moves differs (AGENTS.md rule 7).

INTERFACE (written before the bodies).
  PriorPicker(tag)                  per process (use prior_picker()): the fitted head chosen on FIND (prior.pkl), 78.82's reader, the estimator's inputs. `.choose(moves, tied) -> move` always in `tied`; the search's first move when the prior is silent.
  MoverPrior(future_play.MoverFuture)  the kind pr_prior; every other kind is played as before. `pick(position, tied)` as in 78.82.
  play96(spec) -> game              one game; module level (the Pool spawns workers that import it). Called by tiebreak_measure.run_games(play=play96, stop=settled).
  stages (command line)             match | report, as future_play: the load wait first, the load printed at the start and at the end, the early stop's line, the match lines.

  .venv/bin/python -m catspace.incubator.analyst.discovery.prior_play match --a prior --b control --set A --pairs 100 --tag full [--charge]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import os
import pickle
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import future_judge as FJ
from catspace.incubator.analyst.discovery import future_play as FP
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import prior_pick as PP
from catspace.incubator.analyst.discovery import symbols_play as SP
from catspace.io import paths

OUT = ("prior_78_96",)
PRIOR_KINDS = {"prior": "pr_prior"}
KIND = dict(FP.KIND, **PRIOR_KINDS)
SP.MODE.update({"pr_prior": "future"})
_PICKER = None
TAG = "full"


def out_root(tag: str) -> Path:
    return Path(paths.experiment(*OUT, tag))


class PriorPicker:
    """See the module doc."""

    def __init__(self, tag: str = TAG):
        saved = pickle.load(open(out_root(tag) / "prior.pkl", "rb"))
        inputs = FJ.Inputs(GL.model(HD.GAME, HD.VARIANT), HD.statements_chunked)
        self.prior = PP.Prior(saved["heads"][saved["chosen"]], PP.Features(SP.reader82(), inputs, 1))

    def choose(self, moves: list, tied: list) -> str:
        pick = self.prior.choose(list(moves), list(tied))
        return pick if pick in tied else tied[0]


def prior_picker() -> PriorPicker:
    global _PICKER
    if _PICKER is None:
        _PICKER = PriorPicker()
    return _PICKER


class MoverPrior(FP.MoverFuture):
    """See the module doc."""

    def pick(self, position: dict, tied: list) -> str:
        if self.kind != "pr_prior":
            return super().pick(position, tied)
        move = prior_picker().choose(position["moves"], tied)
        self.detail = {"scores": {}, "carried": {m: [] for m in tied}, "takes": None, "kinds": None}
        assert move in tied
        return move


def play96(spec: dict) -> dict:
    OM.Mover = MoverPrior
    TM.REG.clear()
    g = OM.play_game(spec)
    g["stats"] = [dict(m.stats) for m in TM.REG]
    return g


def games_file(tag: str, a: str, b: str, set_name: str, charge: bool = False) -> Path:
    return out_root(tag) / "matches" / f"{a}_v_{b}_{set_name}{'_charged' if charge else ''}.jsonl"


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("match", "report"))
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--a", default="prior", choices=tuple(KIND))
    ap.add_argument("--b", default="control", choices=tuple(KIND))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--charge", action="store_true", help="take the pick's reading time out of the clock of the search")
    ap.add_argument("--tag", default="full")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    (root / "matches").mkdir(parents=True, exist_ok=True)
    obs = root / "obs.txt"
    say = lambda s: TM.emit([s], obs)
    if a.stage == "match":
        load, busy = TM.wait_quiet(say)
        say(f"OBS prior_play load average at the start of {a.a} v {a.b} [set {a.set}{', charged' if a.charge else ''}]: {os.getloadavg()[0]:.2f}")
        t0 = time.perf_counter()
        gf = games_file(a.tag, a.a, a.b, a.set, a.charge)
        games = TM.run_games(TM.match_specs(a.a, a.b, a.set, a.pairs, a.seconds, root / "scratch", a.charge, KIND), a.workers, gf, say, play=play96, stop=TM.settled)
        wall = time.perf_counter() - t0
        say(f"OBS prior_play load average at the end of {a.a} v {a.b} [set {a.set}{', charged' if a.charge else ''}]: {os.getloadavg()[0]:.2f}; games played {len(games)} of {2 * a.pairs}")
        gf.with_suffix(".meta.json").write_text(json.dumps({"load": load, "busy": busy, "wall": wall, "workers": a.workers, "seconds": a.seconds, "games": len(games)}))
        TM.emit(SP.lines_of(a.a, a.b, a.set, games, load, busy, wall, a.seconds, a.workers, a.charge), obs)
    else:
        found: dict = {}
        for p in sorted((root / "matches").glob("*.jsonl")):
            stem = p.stem
            charge = stem.endswith("_charged")
            x, rest = stem.removesuffix("_charged").split("_v_")
            y, s = rest.rsplit("_", 1)
            found.setdefault((x, y, charge), {})[s] = TM.read_games(p)
        for (x, y, charge), sets in sorted(found.items()):
            for s, games in sorted(sets.items()):
                mf = games_file(a.tag, x, y, s, charge).with_suffix(".meta.json")
                meta = json.loads(mf.read_text()) if mf.exists() else {}
                TM.emit(SP.lines_of(x, y, s, games, meta.get("load", float("nan")), meta.get("busy", float("nan")), meta.get("wall", float("nan")), a.seconds, a.workers, charge), None)
            if len(sets) > 1:
                TM.emit(SP.pooled_lines(x, y, sets, charge), None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
