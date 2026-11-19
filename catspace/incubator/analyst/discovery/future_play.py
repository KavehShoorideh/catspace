"""PLAN 78.95 PART FOUR, PLAY (SYSTEM_SPEC 7 "among equal moves ... whichever gets us closer to our goal(s)" and "the prior learns ... the chances of frame outcomes as played", 8 the wake, 9 THE WALL: the wake never calls the oracle and reads none of its numbers; 12 the hint in play). THE MATCH PLAYER OF THE
FUTURE JUDGE'S TWO CHOOSERS in the match harness of 78.67 and 78.82. A sandbox: nothing is wired into any player of the wake; nothing is written to rules/learned or data/derived; outputs under artifacts/experiments/future_78_95/<tag>/matches/. Game-free: the game is named only through
the pieces that exist (odds_measure, symbols_play, future_choose, future_judge, graph_layer1, human_heads_data).

THE WALL. This module and what it imports at run time never import the oracle wrapper (ladder_oracle) and never read a score of the oracle: the judge it loads was fitted on stored labels (the oracle's class and frames made in sleep); the labels' own file is not opened here. A test imports this module in
a fresh process and shows that the wrapper is not among the loaded modules.

THE HINT IS NEVER A VALUE: every player here is the SAME search call as the control (`solve_equals_batch`: the equal moves handed back, the search's own move first); only the choice among the equal moves differs (AGENTS.md rule 7).

THE PLAYERS (kinds of the spec):
  fu_chooser   the weighted chooser of part three (future_choose.Chooser, setting ("frames", k), the weights of settings.json: the stake of each kind of frame times the chance it pays as played, from FIND); silent (all equal) gives the search's first move.
  fu_joined    the plain joining of the two picks of 78.94: where a tied move takes a unit, the judge of 78.94 (its setting from settings.json, chosen on FIND); otherwise the pattern pick of 78.82 (the reasons of 78.81). The search's first move when either is silent.
The opponents are the control (the search's own move) and 78.82's player (the reasons, kind sy_reasons), by the kinds of symbols_play.

INTERFACE (written before the bodies).
  Picker()                              per process (use picker()): settings.json, the kind heads (judge_kinds.pkl), the judge of 78.94 (future_78_94/full/judge.pkl), the estimator's inputs, 78.82's reader of the reasons.
    .chooser(moves, tied) -> move       the weighted chooser's pick, always in `tied`.
    .joined(moves, tied) -> move        the joined pick, always in `tied`.
  MoverFuture(symbols_play.MoverReasons)  the kinds fu_*; every other kind is played as symbols_play plays it. `pick(position, tied)` as in 78.82.
  play95(spec) -> game                  one game; module level (the Pool spawns workers that import it). Called by tiebreak_measure.run_games(play=play95, stop=settled).
  stages (command line)                 match | report. match: the load wait of 78.67 first, the load printed at the start and at the end, `run_games` with the early stop (the early-stop line is printed), then the match lines; report: every match found.

  .venv/bin/python -m catspace.incubator.analyst.discovery.future_play match --a chooser --b control --set A --pairs 100 --tag full [--charge]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import future_choose as FC
from catspace.incubator.analyst.discovery import future_judge as FJ
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import symbols_play as SP
from catspace.io import paths

OUT = ("future_78_95",)
SRC94 = ("future_78_94", "full")
FUTURE_KINDS = {"chooser": "fu_chooser", "joined": "fu_joined"}
KIND = dict(SP.KIND, **FUTURE_KINDS)
SP.MODE.update({"fu_chooser": "future", "fu_joined": "future"})              # the movers of symbols_play route a kind in MODE through `pick`
_PICKER = None


def out_root(tag: str) -> Path:
    return Path(paths.experiment(*OUT, tag))


class Picker:
    """See the module doc."""

    def __init__(self, tag: str = "full"):
        root = out_root(tag)
        self.settings = json.loads((root / "settings.json").read_text())
        self.kinds = pickle.load(open(root / "judge_kinds.pkl", "rb"))["judge"]
        self.old = pickle.load(open(Path(paths.experiment(*SRC94)) / "judge.pkl", "rb"))["judge"]
        self.inputs = FJ.Inputs(GL.model(HD.GAME, HD.VARIANT), HD.statements_chunked)
        self.chooser_ = FC.Chooser(self.kinds, ("frames", int(self.settings["k"])), self.inputs, 1, {k: float(v) for k, v in self.settings["weights"].items()})
        self.old_ = FC.Chooser(self.old, tuple(self.settings["judge94"]), self.inputs, 1)

    def chooser(self, moves: list, tied: list) -> str:
        pick = self.chooser_.choose(list(moves), list(tied))
        return pick if pick in tied else tied[0]

    def joined(self, moves: list, tied: list) -> str:
        rd = SP.reader82()
        takes, _kinds = rd.controls_of(list(moves), list(tied))
        if any(takes.values()):
            pick = self.old_.choose(list(moves), list(tied))
            return pick if pick in tied else tied[0]
        move, _d = rd.pick(list(moves), list(tied), "reasons")
        return move


def picker() -> Picker:
    global _PICKER
    if _PICKER is None:
        _PICKER = Picker()
    return _PICKER


class MoverFuture(SP.MoverReasons):
    """See the module doc."""

    def pick(self, position: dict, tied: list) -> str:
        if self.kind not in ("fu_chooser", "fu_joined"):
            return super().pick(position, tied)
        move = picker().chooser(position["moves"], tied) if self.kind == "fu_chooser" else picker().joined(position["moves"], tied)
        self.detail = {"scores": {}, "carried": {m: [] for m in tied}, "takes": None, "kinds": None}
        assert move in tied
        return move


def play95(spec: dict) -> dict:
    OM.Mover = MoverFuture
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
    ap.add_argument("--a", default="chooser", choices=tuple(KIND))
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
        say(f"OBS future_play load average at the start of {a.a} v {a.b} [set {a.set}{', charged' if a.charge else ''}]: {os.getloadavg()[0]:.2f}")
        t0 = time.perf_counter()
        gf = games_file(a.tag, a.a, a.b, a.set, a.charge)
        games = TM.run_games(TM.match_specs(a.a, a.b, a.set, a.pairs, a.seconds, root / "scratch", a.charge, KIND), a.workers, gf, say, play=play95, stop=TM.settled)
        wall = time.perf_counter() - t0
        say(f"OBS future_play load average at the end of {a.a} v {a.b} [set {a.set}{', charged' if a.charge else ''}]: {os.getloadavg()[0]:.2f}; games played {len(games)} of {2 * a.pairs}")
        gf.with_suffix(".meta.json").write_text(json.dumps({"load": load, "busy": busy, "wall": wall, "workers": a.workers, "seconds": a.seconds, "games": len(games)}))
        TM.emit(SP.lines_of(a.a, a.b, a.set, games, load, busy, wall, a.seconds, a.workers, a.charge), obs)
        if a.a in FUTURE_KINDS or a.a in SP.MODE_PLAYERS:
            TM.emit(SP.decision_lines(a.a, a.b, a.set, games, GL.model(HD.GAME, HD.VARIANT)), obs)
    else:
        layer1 = GL.model(HD.GAME, HD.VARIANT)
        found: dict = {}
        for p in sorted((root / "matches").glob("*.jsonl")):
            stem = p.stem
            charge = stem.endswith("_charged")
            x, rest = stem.removesuffix("_charged").split("_v_")
            y, s = rest.rsplit("_", 1)
            found.setdefault((x, y, charge), {})[s] = TM.read_games(p)
        for (x, y, charge), sets in sorted(found.items()):
            for s, games in sorted(sets.items()):
                meta = json.loads(games_file(a.tag, x, y, s, charge).with_suffix(".meta.json").read_text()) if games_file(a.tag, x, y, s, charge).with_suffix(".meta.json").exists() else {}
                TM.emit(SP.lines_of(x, y, s, games, meta.get("load", float("nan")), meta.get("busy", float("nan")), meta.get("wall", float("nan")), a.seconds, a.workers, charge), None)
                if x in FUTURE_KINDS or x in SP.MODE_PLAYERS:
                    TM.emit(SP.decision_lines(x, y, s, games, layer1), None)
            if len(sets) > 1:
                TM.emit(SP.pooled_lines(x, y, sets, charge), None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
