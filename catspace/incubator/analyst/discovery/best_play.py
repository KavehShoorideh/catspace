"""PLAN 79.03 PART THREE, THE MEND IN PLAY (SYSTEM_SPEC 4 the tactical reasoner's deepening and its prunings, 5 the measured records, 7 the pick among equal moves; piece: tactical reasoning, the match). A sandbox: nothing is wired into any player of the wake, no default of the engine is changed (the switch `root_full` is a parameter of the call), nothing is written to rules/learned or
data/derived; every output goes under artifacts/experiments/best_79_03/<tag>/. Game-free: the game is named only through the pieces that exist (stack_play, cut_play, tiebreak_measure, symbols_play, record_play). THIS MODULE AND WHAT IT IMPORTS NEVER LOAD THE ORACLE WRAPPER (tests/test_plan79_03_best.py checks the import closure).

THE TWO SIDES. Both are the stack's player of 79.00 (the human-result record, the verified choice on in mode slot, 78.82's pick among the equal moves); what differs is the switch `root_full` of the deepening (no late-move reduction at the root). The same two opening sets A and B of 78.67, colours alternated, the early stop of `tiebreak_measure.settled` with min_pairs=60, the sanity
match first (the stack against itself, 20 pairs, no early stop), the load printed at the start and the end, 0.2 s a move, ledger share 0.4, ply cap 120, 9 workers. Played only by the rule of best_mend (rule.json), written before CONFIRM was read.

INTERFACE (written before the bodies).
  PLAYERS[name] -> {"record", "pick", "verify", "root_full"}                the two sides.
  player_spec(name) -> dict                                                  stack_play.player_spec with the switch.
  match_specs(a, b, set_name, pairs, seconds, scratch) -> [spec]             the games of one match.
  depth_totals(games, who) -> dict                                           the mean depth the deepening finished a decision, the seconds and the nodes.
  Called by: the command line (`sanity`, `match`). The harness is stack_play / tiebreak_measure, reused as it is.

  .venv/bin/python -m catspace.incubator.analyst.discovery.best_play sanity --tag T | match --set A --tag T
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import os
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import stack_play as SPL
from catspace.incubator.analyst.discovery.best_out import out_root

PLAYERS = {"stack": {"record": "human", "pick": True, "verify": True, "root_full": False},
           "root": {"record": "human", "pick": True, "verify": True, "root_full": True}}


def player_spec(name: str) -> dict:
    p = PLAYERS[name]
    spec = SPL.player_spec("human+v+pick")
    spec["root_full"] = bool(p["root_full"])
    return spec


def match_specs(a: str, b: str, set_name: str, pairs: int, seconds: float, scratch: Path) -> list:
    pa, pb = player_spec(a), player_spec(b)
    specs = []
    for seed, opening in TM.openings(set_name, pairs):
        for colour in (0, 1):
            specs.append({"gid": f"{a}_v_{b}:{set_name}{seed}:{colour}", "start": f"{set_name}{seed}", "colour": colour, "text": None, "opening": opening,
                          "players": [pa, pb] if colour == 0 else [pb, pa], "seconds": seconds, "max_plies": TM.PLY_CAP, "scratch": str(scratch)})
    return specs


def depth_totals(games: list, who: int) -> dict:
    dec = depth = secs = nodes = 0
    for g in games:
        st = g["stats"][TM.seat_of(g, who)]
        dec += st["decisions"]
        depth += st.get("depth_sum", 0)
        secs += st.get("search", 0.0)
        nodes += st.get("nodes", 0)
    return {"decisions": dec, "depth_a_decision": depth / max(1, dec), "seconds_a_decision": secs / max(1, dec), "nodes_a_decision": nodes / max(1, dec)}


def cmd_match(name: str, a: str, b: str, set_name: str, pairs: int, seconds: float, workers: int, tag: str, say, stop: bool = True) -> list:
    root = out_root(tag)
    (root / "matches").mkdir(parents=True, exist_ok=True)
    say(f"OBS best {name} {a} v {b} [set {set_name}]: the early stop is asked with min_pairs={SPL.MIN_PAIRS} (99 pct band on whole opening pairs, clear of a half or within 0.03 of it); sides: {a} {PLAYERS[a]}, {b} {PLAYERS[b]}; the judge the human-result record on both sides, the pick of 78.82 on both, the verified choice on (slot) on both")
    load0, busy = TM.wait_quiet(say)
    t0 = time.perf_counter()
    games = TM.run_games(match_specs(a, b, set_name, pairs, seconds, root / "scratch"), workers, root / "matches" / f"{name}_{a}_v_{b}_{set_name}.jsonl", say, play=SPL.play79, stop=SPL.stop_of() if stop else None)
    load1 = os.getloadavg()[0]
    wall = time.perf_counter() - t0
    from catspace.incubator.analyst.discovery import symbols_play as SP
    plain = SP._plain(games)
    for x in TM.match_lines(a, b, set_name, plain, load0, busy, wall, seconds, workers):
        say(x.replace("OBS tiebreak match", f"OBS best {name}", 1))
    for who, nm in ((0, a), (1, b)):
        say(f"OBS best {name} {a} v {b} [set {set_name}]: {nm}: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in depth_totals(games, who).items()))
        r = SPL.replacements(games, who)
        if r["vc_asked"]:
            say(f"OBS best {name} {a} v {b} [set {set_name}]: {nm}: the verification asked {r['vc_asked']} times in {r['decisions']} decisions: held {r['vc_held']}, REPLACED {r['vc_replaced']}, failed {r['vc_failed']}, cut {r['vc_cut']}")
    say(f"OBS best {name} {a} v {b} [set {set_name}]: load average (1 min) at the start {load0:.2f}, at the end {load1:.2f}; clocks {seconds} s a move, {workers} workers")
    return games


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("sanity", "match"))
    ap.add_argument("--a", default="root", choices=tuple(PLAYERS))
    ap.add_argument("--b", default="stack", choices=tuple(PLAYERS))
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args(argv)
    obs = out_root(a.tag) / "obs.txt"

    def say(line: str) -> None:
        TM.emit([line], obs)
    if a.stage == "sanity":
        cmd_match("sanity", "stack", "stack", "A", 20, a.seconds, a.workers, a.tag, say, stop=False)
    else:
        cmd_match("m1", a.a, a.b, a.set, a.pairs, a.seconds, a.workers, a.tag, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
