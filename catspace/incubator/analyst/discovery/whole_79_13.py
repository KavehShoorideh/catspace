"""PLAN 79.13 PARTS C, D AND E, THE GAMES (SYSTEM_SPEC 8 the rule for choosing a move and the wall, 7 "Calculation and estimate, decided together" point 2 the estimate per goal and "among equal moves we optimize for the prediction ... whichever gets us closer to our goals", 9 sleep; piece: the estimate per goal).
A sandbox: nothing is wired into any default of the engine, nothing is written to rules/learned or data/derived, no oracle is called, no player id is written; every output goes under artifacts/experiments/whole_79_13/ (through catspace.io.paths). Game-free: the harness of 79.11 (`whole_play`) plays the games, this module only chooses the players, the openings and the tags.
The table of the chance of a goal (goal_table.py, part B) is read from artifacts/experiments/whole_79_13/<table tag>/table.json and given to the wake through the binding (`goal_table`); with it, each goal in the log carries the new chance and the chooser's rows carry the gain inputs (our_gain, their_gain, our_goal_gain, their_goal_gain).

PART D (1), THE FIT: games with the equal moves varied (mode Explore, the table given) from the training openings of 79.12 (sets C, D, T; never an opening of A or B), fitted by chooser_fit.py with the shuffled control beside it. PART D (2), HIS RULE READ PLAINLY, NO FIT: mode Gain. Each is played against mode Fixed and against the old player at equal mean wall
(the scale of 79.11), sets A and B, 0.2 s, ply cap 120, `tiebreak_measure.settled` with min_pairs=60, one match at a time on a quiet machine. PART E: 200 Explore games at ply cap 200 beside ply cap 120, printed only.

INTERFACE (written before the bodies).
  table_for(tag) -> tuple                              the table of the tag in the form the binding takes.
  train_specs(mix, pairs, seconds, scratch, log_dir, scale, table, base_seed) -> [spec]   the Explore games of one mix (`self`, `old`), the wake given the table.
  judge_specs(mode, weights, vs, set_name, pairs, seconds, scratch, log_dir, scale, table, name) -> [spec]   a match of the wake (mode Gain or Weights) against Fixed or the old player.
  cap_specs(cap, pairs, seconds, scratch, log_dir, table) -> [spec]    PART E: the wake against itself in mode Explore at a ply cap.
  Called by: the command line and tests/test_plan79_13.py.

  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_13 sanity | train --mix self|old --pairs N | judge --name NAME --mode gain|weights --set A|B --vs fixed|old [--weights FILE] | cap --cap 120|200
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import os
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import goal_table as GT
from catspace.incubator.analyst.discovery import whole_79_12 as W12
from catspace.incubator.analyst.discovery import whole_play as WP
from catspace.io import paths

OUT = ("whole_79_13",)
TABLE_TAG = "full"                                       # where part B wrote table.json
JUDGE_SETS = W12.JUDGE_SETS


def out_root(tag: str) -> Path:
    return Path(paths.experiment(*OUT, tag))


def table_for(tag: str = TABLE_TAG) -> tuple:
    return GT.binding_form(GT.load_table(tag))


def train_specs(mix: str, pairs: int, seconds: float, scratch: Path, log_dir: Path, scale_: float, table: tuple, base_seed: int = 7913) -> list:
    """The Explore games of one mix, the openings of 79.12's training sets (none of A or B). `self`: the wake against itself; `old`: the wake against the old player at `scale_ x seconds`."""
    ex = W12.chooser_spec("explore", None, base_seed)
    ops = W12.train_openings(pairs)
    if mix == "self":
        return WP.match_specs("wake", "wake", "T", pairs, seconds, scratch, log_dir, label="self", chooser_a=ex, chooser_b=ex, openings=ops, table_a=table, table_b=table)
    return WP.match_specs("wake", "old", "T", pairs, seconds, scratch, log_dir, clock_b=scale_ * seconds, label="old", chooser_a=ex, openings=ops, table_a=table)


def mine_spec(mode: str, weights: "dict | None") -> dict:
    return W12.chooser_spec(mode, weights, 0)


def judge_specs(mode: str, weights: "dict | None", vs: str, set_name: str, pairs: int, seconds: float, scratch: Path, log_dir: Path, scale_: float, table: tuple, name: str) -> list:
    """The wake in mode `mode` (the table given) against the wake in mode Fixed (no table: the wake of main) or against the old player at equal mean wall."""
    mine = mine_spec(mode, weights)
    if vs == "fixed":
        return WP.match_specs("wake", "wake", set_name, pairs, seconds, scratch, log_dir, label=name, chooser_a=mine, chooser_b=W12.chooser_spec("fixed"), table_a=table)
    return WP.match_specs("wake", "old", set_name, pairs, seconds, scratch, log_dir, clock_b=scale_ * seconds, label=name, chooser_a=mine, table_a=table)


def cap_specs(cap: int, pairs: int, seconds: float, scratch: Path, log_dir: Path, table: tuple) -> list:
    ex = W12.chooser_spec("explore", None, 7913)
    return WP.match_specs("wake", "wake", "T", pairs, seconds, scratch, log_dir, label=f"cap{cap}", chooser_a=ex, chooser_b=ex, openings=W12.train_openings(pairs), table_a=table, table_b=table, max_plies=cap)


def run(name: str, specs: list, workers: int, tag: str, say, stop=None, quiet: bool = True, sides: "tuple | None" = None) -> list:
    root = out_root(tag)
    (root / "matches").mkdir(parents=True, exist_ok=True)
    load0, busy = TM.wait_quiet(say) if quiet else (os.getloadavg()[0], 0.0)
    t0 = time.perf_counter()
    games = TM.run_games(specs, workers, root / "matches" / f"{name}.jsonl", say, play=WP.play11, stop=stop)
    say(f"OBS whole_79_13 {name}: load average at start {load0:.2f}, cores busy {busy:.2f}, workers {workers}, wall {time.perf_counter() - t0:.0f} s")
    for x in W12.outcome_lines(name, games):
        say(x.replace("OBS whole_79_12", "OBS whole_79_13", 1))
    if sides:
        for x in TM.match_lines(sides[0], sides[1], sides[2], games, load0, busy, time.perf_counter() - t0, 0.2, workers):
            say(x.replace("OBS tiebreak match", f"OBS whole_79_13 {name}", 1))
    return games


def cap_lines(name: str, cap: int, games: list) -> list:
    """PART E, printed only: the share of games that ended on a result, the mean plies and the wall."""
    n = len(games)
    ended = sum(1 for g in games if not g.get("unfinished"))
    wall = sum(s["wall"] for g in games for s in g["stats"])
    dec = sum(s["decisions"] for g in games for s in g["stats"])
    plies = sum(g["plies"] for g in games) / max(1, n)
    return [f"OBS whole_79_13 {name}: ply cap {cap}: games {n}, ended on a result {ended} ({100 * ended / max(1, n):.1f} pct), reached the cap {n - ended}, mean plies {plies:.1f}, decisions {dec}, mean wall a decision {wall / max(1, dec):.4f} s, wall of the games in all {wall:.0f} s"]


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("sanity", "train", "judge", "cap"))
    ap.add_argument("--mix", default="self", choices=("self", "old"))
    ap.add_argument("--pairs", type=int, default=0, help="train: opening pairs of the mix (default 500); judge: opening pairs of the set (default 100); cap: default 100")
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--table-tag", default=TABLE_TAG)
    ap.add_argument("--name", default="")
    ap.add_argument("--set", default="A", choices=JUDGE_SETS)
    ap.add_argument("--mode", default="gain", choices=("gain", "weights"))
    ap.add_argument("--weights", default="")
    ap.add_argument("--vs", default="fixed", choices=("fixed", "old"))
    ap.add_argument("--cap", type=int, default=120)
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)

    def say(line: str) -> None:
        TM.emit([line], root / "obs.txt")
    WP.CTX.clear()
    table = table_for(a.table_tag)
    sc = W12.scale()
    if a.stage in ("sanity", "train"):
        mixes = ("self", "old") if a.stage == "sanity" else (a.mix,)
        for mix in mixes:
            pairs = 5 if a.stage == "sanity" else (a.pairs or 500)
            name = f"{a.stage}_{mix}"
            specs = train_specs(mix, pairs, a.seconds, root / "scratch", root / "logs" / name, sc, table)
            say(f"OBS whole_79_13 {name}: {len(specs)} games ({pairs} opening pairs of the training openings, none of A or B), the wake in mode Explore on every decision with the table of the chance of a goal (K {table[0]}, {sum(len(c) for _f, c in table[2])} cells), "
                + (f"against the old player at {sc * a.seconds:.4f} s (the scale {sc:.4f} of 79.11)" if mix == "old" else "against itself"))
            run(name, specs, a.workers, a.tag, say)
    elif a.stage == "judge":
        weights = json.loads(Path(a.weights).read_text())["weights"] if a.weights else None
        a.pairs = a.pairs or 100
        name = a.name or f"judge_{a.mode}_{a.vs}_{a.set}"
        specs = judge_specs(a.mode, weights, a.vs, a.set, a.pairs, a.seconds, root / "scratch", root / "logs" / name, sc, table, name)
        sides = (f"wake_{a.mode}", "wake_fixed" if a.vs == "fixed" else "old", a.set)
        run(name, specs, a.workers, a.tag, say, stop=lambda g: TM.settled(g, min_pairs=WP.MIN_PAIRS), sides=sides)
    else:
        pairs = a.pairs or 100
        name = f"cap{a.cap}"
        specs = cap_specs(a.cap, pairs, a.seconds, root / "scratch", root / "logs" / name, table)
        say(f"OBS whole_79_13 {name}: {len(specs)} games ({pairs} opening pairs of the training openings), the wake against itself in mode Explore, ply cap {a.cap}")
        games = run(name, specs, a.workers, a.tag, say)
        for x in cap_lines(name, a.cap, games):
            say(x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
