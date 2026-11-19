"""PLAN 78.99 PART TWO, THE JUDGE'S RECORD OF WORTHS IN PLAY (SYSTEM_SPEC 4 the equal reading, 5 the value and its measured records; piece: tactical reasoning, the match). A sandbox: nothing is wired into any player of the wake, the DEFAULT record of the engine is
not changed anywhere (the record is a parameter of the player's spec, as 78.76 made it a parameter of the search), nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/record_78_99/<tag>/. Game-free: the game is named only through
the pieces that exist (odds_measure, tiebreak_measure, symbols_play). THIS MODULE AND WHAT IT IMPORTS NEVER LOAD THE ORACLE WRAPPER (the wall; tests/test_plan78_99_record.py).

THE PLAYERS. All search with the same call as every player of 78.67 (`solve_equals_batch`: the ledger deepens within a share of the clock and the equal reading takes the rest); what differs between two players is the record of worths that judges the lines, nothing else:
  odds       the odds games' record (gen2 of 78.61), the judge of every match since 78.61;
  selfplay   the self-play results' record (selfplay_78_58);
  human      the human results' record (human_78_59).
Without a pick the player plays the search's own move (78.67's control). With the pick (`+pick`) it is 78.82's player (`sy_reasons`: the reasons' pick inside the tie set the search hands back), the same on both sides. The records are read as stored (threat_concrete.Record.load);
no worth is edited, blended or put in by hand.

INTERFACE (written before the bodies).
  WORTH[name] -> {kind word: integer worth}      the stored record by player name (a function of the name only; `worth_of(name)`).
  player_spec(name, pick) -> dict                 {"kind": "tb_control" | "sy_reasons", "worth": ..., "charge": False}.
  match_specs(a, b, pick, set_name, pairs, seconds, scratch) -> [spec]    the games of one match: sets A and B of 78.67, colours alternated; `a` and `b` are record names; `pick` puts 78.82's pick on both sides.
  play99(spec) -> game                            one game; module level so the spawned workers import it (plays with 78.67's movers, or 78.82's when the spec's players carry the pick).
  stop_of(min_pairs) -> f(games) -> word|None     `tiebreak_measure.settled` with min_pairs given (the early stop is not asked before that many whole opening pairs).
  Called by: the command line (`sanity`, `match`, `report`). The harness is tiebreak_measure (the openings, the clock, the load wait, the match loop and its lines), reused as it is.

  .venv/bin/python -m catspace.incubator.analyst.discovery.record_play sanity --tag T | match --a human --b odds --set A [--pick] --tag T | report --tag T
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import os
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import symbols_play as SP
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.io import paths

OUT = ("record_78_99",)
RECORD_OF = {"odds": "gen2", "selfplay": "selfplay_78_58", "human": "human_78_59"}
MIN_PAIRS = 60                                          # the early stop is not asked before this many whole opening pairs (a stop after thirty reads high)
KINDS = {False: TM.KIND["control"], True: SP.KIND["reasons"]}


def out_root(tag: str) -> Path:
    return Path(paths.experiment(*OUT, tag))


def worth_of(name: str) -> dict:
    return dict(TC.Record.load(RECORD_OF[name]).worth)


def player_spec(name: str, pick: bool) -> dict:
    return {"kind": KINDS[bool(pick)], "worth": worth_of(name), "charge": False}


def stop_of(min_pairs: int = MIN_PAIRS):
    def stop(games: list) -> "str | None":
        return TM.settled(games, min_pairs=min_pairs)
    return stop


def match_specs(a: str, b: str, pick: bool, set_name: str, pairs: int, seconds: float, scratch: Path) -> list:
    pa, pb = player_spec(a, pick), player_spec(b, pick)
    tag = ("+pick" if pick else "")
    specs = []
    for seed, opening in TM.openings(set_name, pairs):
        for colour in (0, 1):
            specs.append({"gid": f"{a}{tag}_v_{b}{tag}:{set_name}{seed}:{colour}", "start": f"{set_name}{seed}", "colour": colour, "text": None, "opening": opening,
                          "players": [pa, pb] if colour == 0 else [pb, pa], "seconds": seconds, "max_plies": TM.PLY_CAP, "scratch": str(scratch)})
    return specs


def play99(spec: dict) -> dict:
    """One game; the pick's movers when the players carry the pick, 78.67's otherwise."""
    if any(p["kind"] == SP.KIND["reasons"] for p in spec["players"]):
        return SP.play82(spec)
    return TM.play67(spec)


def games_file(tag: str, a: str, b: str, pick: bool, set_name: str) -> Path:
    return out_root(tag) / "matches" / f"{a}{'+pick' if pick else ''}_v_{b}{'+pick' if pick else ''}_{set_name}.jsonl"


def lines_of(a: str, b: str, pick: bool, set_name: str, games: list, load0: float, load1: float, busy: float, wall: float, seconds: float, workers: int) -> list:
    nm = "+pick" if pick else ""
    out = TM.match_lines(a + nm, b + nm, set_name, SP._plain(games), load0, busy, wall, seconds, workers)
    out = [x.replace("OBS tiebreak match", "OBS record match", 1) for x in out]
    out.append(f"OBS record match {a + nm} v {b + nm} [set {set_name}]: load average (1 min) at the start {load0:.2f}, at the end {load1:.2f}")
    return out


def cmd_match(a: str, b: str, pick: bool, set_name: str, pairs: int, seconds: float, workers: int, tag: str, say, stop=True) -> list:
    root = out_root(tag)
    (root / "matches").mkdir(parents=True, exist_ok=True)
    say(f"OBS record match {a} v {b} [set {set_name}]: the early stop is asked with min_pairs={MIN_PAIRS} (tiebreak_measure.settled wrapped with that argument; 99 pct band on whole opening pairs, clear of a half or within 0.03 of it); pick {'on both sides (78.82 reasons)' if pick else 'none'}; "
        f"records: {a} {RECORD_OF[a]}, {b} {RECORD_OF[b]}")
    load0, busy = TM.wait_quiet(say)
    t0 = time.perf_counter()
    games = TM.run_games(match_specs(a, b, pick, set_name, pairs, seconds, root / "scratch"), workers, games_file(tag, a, b, pick, set_name), say, play=play99, stop=stop_of() if stop else None)
    load1 = os.getloadavg()[0]
    for x in lines_of(a, b, pick, set_name, games, load0, load1, busy, time.perf_counter() - t0, seconds, workers):
        say(x)
    return games


def read_games(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("sanity", "match", "report"))
    ap.add_argument("--a", default="human", choices=tuple(RECORD_OF))
    ap.add_argument("--b", default="odds", choices=tuple(RECORD_OF))
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--pick", action="store_true")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)
    obs = root / "obs.txt"

    def say(line: str) -> None:
        TM.emit([line], obs)
    if a.stage == "sanity":
        cmd_match("odds", "odds", False, "A", 20, a.seconds, a.workers, a.tag, say, stop=False)
    elif a.stage == "match":
        cmd_match(a.a, a.b, a.pick, a.set, a.pairs, a.seconds, a.workers, a.tag, say)
    else:
        for p in sorted((root / "matches").glob("*.jsonl")):
            print(p.name, len(read_games(p)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
