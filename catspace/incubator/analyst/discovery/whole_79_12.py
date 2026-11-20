"""PLAN 79.12 PARTS C AND E, THE GAMES: the whole wake with the last layer (SYSTEM_SPEC 8 the rule for choosing a move and the wall, 7 "Calculation and estimate, decided together" points 4 to 6, 9 sleep; piece: the wake's last layer).
A sandbox: nothing is wired into any default of the engine, nothing is written to rules/learned or data/derived, no oracle is called, no player id is written; every output goes under artifacts/experiments/whole_79_12/ (through catspace.io.paths). Game-free: the harness of 79.11 (`whole_play`) plays the games, this module only
chooses the openings, the players and the tags.

PART C, GAMES WITH THE EQUAL MOVES VARIED. Two mixes, each a set of opening pairs (colours alternated; the wake against itself, the wake against the old player at EQUAL MEAN WALL: the old player is given `scale x clock`, the one scale found in 79.11 and kept in its scale.json):
  self   the wake (mode Explore) against the wake (mode Explore): both seats' decisions are logged.
  old    the wake (mode Explore) against the old player.
The openings are NEVER those of the sets A and B (they judge in part E): the sets C and D of 78.93 (100 pairs each) and then a set T made here by the rule of those sets with other seeds, skipping every opening of A, B, C and D (`train_openings`). Each decision with two or more moves held equal logs the rows of the last layer
(the inputs of every candidate) and the propensity of the move played (1 over the candidates).
PART E, THE WHOLE WITH THE CHOOSER. `judge` plays the matches of the plan on the sets A and B at 0.2 s, ply cap 120, `tiebreak_measure.settled` with min_pairs=60, one at a time on a quiet machine: wake(Weights) v wake(Fixed) and each against the old player at equal mean wall.

INTERFACE (written before the bodies).
  train_openings(pairs) -> [(seed, moves)]         the training openings: C, D, then T; never an opening of A or B.
  train_specs(mix, pairs, seconds, scratch, log_dir, scale, base_seed) -> [spec]    the games of one mix.
  chooser_spec(mode, weights, seed) -> dict        the last layer of a player as the harness takes it.
  Called by: the command line below and the tests.

  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_12 sanity | train --mix self|old --pairs N | judge --name NAME --set A|B --weights FILE [--tag T] [--workers W]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import os
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import whole_play as WP
from catspace.io import paths

OUT = ("whole_79_12",)
SCALE_FROM = ("whole_79_12", "scale_from_79_11.json")         # the one scale of equal wall found in 79.11 (a copy of its scale.json, put here by hand: the folder of 79.11 is read only)
T_FIRST_SEED = 210000                                           # the set T: the rule of the sets C and D, a first seed far from theirs (A 4100.., B 41000, C 73000, D 105000)
JUDGE_SETS = ("A", "B")


def out_root(tag: str) -> Path:
    return Path(paths.experiment(*OUT, tag))


def judge_openings() -> set:
    """The openings of the sets that judge (A and B): none of them may be played as a training game."""
    return {tuple(o) for name in JUDGE_SETS for _s, o in TM.openings(name, TM.SET_PAIRS)}


def train_openings(pairs: int) -> list:
    """[(seed, moves)] of the training openings: the sets C and D of 78.93 (100 pairs each; they skip every opening of A and B), then the set T (seeds from T_FIRST_SEED on, an opening skipped when A, B, C, D or T so far hold it).
    No opening of A or B is ever returned (asserted)."""
    seen = judge_openings()
    out: list = []
    for name in ("C", "D"):
        for s, o in TM.openings(name, TM.SET_PAIRS):
            if len(out) < pairs and tuple(o) not in seen:
                out.append((s, list(o)))
                seen.add(tuple(o))
    seed = T_FIRST_SEED
    while len(out) < pairs:
        o = tuple(OM.random_opening(None, seed))
        if o not in seen:
            seen.add(o)
            out.append((seed, list(o)))
        seed += 1
    assert not ({tuple(o) for _s, o in out} & judge_openings()), "a judging opening is in the training set"
    return out[:pairs]


def chooser_spec(mode: str, weights: "dict | None" = None, seed: int = 0) -> dict:
    return {"mode": mode, "weights": dict(weights or {}), "seed": int(seed)}


def scale() -> float:
    return json.loads(Path(paths.experiment(*SCALE_FROM)).read_text())["scale"]


def train_specs(mix: str, pairs: int, seconds: float, scratch: Path, log_dir: Path, scale_: float, base_seed: int = 7912) -> list:
    """The games of one mix of part C. `self`: wake (Explore) v wake (Explore); `old`: wake (Explore) v old at `scale_ x seconds`. The set name of a game is "T" + the opening's seed (the sets differ in the seed)."""
    ex = chooser_spec("explore", None, base_seed)
    ops = train_openings(pairs)
    if mix == "self":
        return WP.match_specs("wake", "wake", "T", pairs, seconds, scratch, log_dir, label="self", chooser_a=ex, chooser_b=ex, openings=ops)
    return WP.match_specs("wake", "old", "T", pairs, seconds, scratch, log_dir, clock_b=scale_ * seconds, label="old", chooser_a=ex, openings=ops)


def outcome_lines(name: str, games: list) -> list:
    n = len(games)
    unf = sum(1 for g in games if g.get("unfinished"))
    white = [g["outcome"] / 100.0 for g in games if g.get("outcome") is not None]
    wall = sum(s["wall"] for g in games for s in g["stats"])
    dec = sum(s["decisions"] for g in games for s in g["stats"])
    plies = sum(g["plies"] for g in games) / max(1, n)
    return [f"OBS whole_79_12 {name}: games {n}, unfinished {unf} (counted as draws), white's score over the finished {sum(white) / max(1, len(white)):.4f}, mean plies {plies:.1f}, decisions {dec}, mean wall a decision {wall / max(1, dec):.4f} s"]


def run(name: str, specs: list, workers: int, tag: str, say, stop=None, quiet: bool = True, sides: "tuple | None" = None) -> list:
    root = out_root(tag)
    (root / "matches").mkdir(parents=True, exist_ok=True)
    load0, busy = TM.wait_quiet(say) if quiet else (os.getloadavg()[0], 0.0)
    t0 = time.perf_counter()
    games = TM.run_games(specs, workers, root / "matches" / f"{name}.jsonl", say, play=WP.play11, stop=stop)
    say(f"OBS whole_79_12 {name}: load average at start {load0:.2f}, cores busy {busy:.2f}, workers {workers}, wall {time.perf_counter() - t0:.0f} s")
    for x in outcome_lines(name, games):
        say(x)
    if sides:                                                                   # a match: the score of the first player, W D L, the band by bootstrap over opening pairs (78.67's lines, as 79.11 printed them)
        for x in TM.match_lines(sides[0], sides[1], sides[2], games, load0, busy, time.perf_counter() - t0, 0.2, workers):
            say(x.replace("OBS tiebreak match", f"OBS whole_79_12 {name}", 1))
    return games


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("sanity", "train", "judge"))
    ap.add_argument("--mix", default="self", choices=("self", "old"))
    ap.add_argument("--pairs", type=int, default=0, help="train: opening pairs of the mix (default 500); judge: opening pairs of the set (default 100)")
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--name", default="")
    ap.add_argument("--set", default="A", choices=JUDGE_SETS)
    ap.add_argument("--weights", default="")
    ap.add_argument("--vs", default="fixed", choices=("fixed", "old"), help="judge: the wake with the weights against the wake in mode Fixed, or a wake against the old player at equal mean wall")
    ap.add_argument("--player", default="weights", choices=("weights", "fixed"), help="judge with --vs old: which wake plays the old player")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)

    def say(line: str) -> None:
        TM.emit([line], root / "obs.txt")
    WP.CTX.clear()
    if a.stage in ("sanity", "train"):
        sc = scale()
        mixes = ("self", "old") if a.stage == "sanity" else (a.mix,)
        for mix in mixes:
            pairs = 5 if a.stage == "sanity" else (a.pairs or 500)
            name = f"{a.stage}_{mix}"
            specs = train_specs(mix, pairs, a.seconds, root / "scratch", root / "logs" / name, sc)
            say(f"OBS whole_79_12 {name}: {len(specs)} games ({pairs} opening pairs of the training openings, none of A or B), the wake in mode Explore on every decision; old player clock {sc * a.seconds:.4f} s (the scale {sc:.4f} of 79.11)" if mix == "old"
                else f"OBS whole_79_12 {name}: {len(specs)} games ({pairs} opening pairs of the training openings, none of A or B), the wake against itself, both in mode Explore")
            run(name, specs, a.workers, a.tag, say)
    else:
        weights = json.loads(Path(a.weights).read_text())["weights"]
        sc = scale()
        mine = chooser_spec("weights", weights, 0)
        a.pairs = a.pairs or 100
        name = a.name or f"judge_{a.vs}_{a.set}"
        if a.vs == "fixed":
            specs = WP.match_specs("wake", "wake", a.set, a.pairs, a.seconds, root / "scratch", root / "logs" / name, label=name, chooser_a=mine, chooser_b=chooser_spec("fixed"))
        else:
            specs = WP.match_specs("wake", "old", a.set, a.pairs, a.seconds, root / "scratch", root / "logs" / name, clock_b=sc * a.seconds, label=name, chooser_a=mine if a.player == "weights" else chooser_spec("fixed"))
        sides = ("wake_weights", "wake_fixed", a.set) if a.vs == "fixed" else (f"wake_{a.player}", "old", a.set)
        run(name, specs, a.workers, a.tag, say, stop=lambda g: TM.settled(g, min_pairs=WP.MIN_PAIRS), sides=sides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
