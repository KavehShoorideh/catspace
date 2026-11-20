"""PLAN 79.17 PARTS A AND E: THE SMALL ENDINGS AS THE TEST OF MEANS AND ENDS (SYSTEM_SPEC 7 "the small endings as the test of means and ends, and a measure of progress", "the go on the small endings"; 8 the wall; 9 the oracle rule; piece: the wake).
A harness: nothing is wired into any default of the engine, nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/whole_79_17/ (through catspace.io.paths). This module names no object of the game: the starts, the table's defender and the endings come
from the oracle of the measure (measure_oracle_79_17, the one module that reads the endgame table); the games are played by whole_play (79.11), the wake through the binding, only its JSON is read.

THE CELL: a class of ending (two long-movers and the leader against a lone leader, "two"; one long-mover and the leader against a lone leader, "one"), an ATTACKER (the wake as on main, abcd of 79.16; the old player; the wake with the held goal, "wake_goal", part E), a DEFENDER (the table's reply that holds longest, "table"; the old player, "old"), a clock a move. The same starts in
every cell (won positions drawn once, the stronger side to move, both colours). The attacker never sees the table.
Printed for each cell: games, wins inside the game's own rule on quiet plies, mean plies, the games by how they ended (won, repetition, the quiet-ply rule, ply cap, stalemate, a unit lost), the table's distance at the start by bins, the mean depth of the attacker.
PART E ALSO: the fall of the held measure along each game (`falls`), read from the wake's own log: drops a game, the longest run of our moves with no drop, goals dropped and why.

INTERFACE (written before the bodies).
  MoverCell(spec, seconds, seed, trace)            78.67's/79.11's movers plus the kind "mm_table" (the table's defender); the wake spec may carry "measure" {"term", "d"} (given to the binding as the held goal).
  play17(spec) -> game                             one game (module level so the spawned workers import it).
  player(name, seconds, measure=None) -> dict      "wake" (abcd), "wake_goal" (abcd and the measure given), "old", "table".
  cell_specs(cls, starts, attacker, defender, seconds, scratch, log_dir, label, cap) -> [spec]
  summarise(games, starts, cls) -> dict                 the cell's numbers (see above).
  falls(log_dir) -> dict                           the fall of the measure along the games of a folder.
  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_17 cells --phase A|E --classes two,one --attackers wake,old --defenders table,old --clocks 0.2,2.0 --n 10 [--tag T] [--workers W]
Called by: the command line, whole_79_17_print.py, tests/test_plan79_17.py.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import gzip
import json
import os
import time
from collections import Counter
from pathlib import Path

from catspace.incubator.analyst.discovery import measure_oracle_79_17 as MO
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import whole_79_14 as W14
from catspace.incubator.analyst.discovery import whole_79_15 as W15
from catspace.incubator.analyst.discovery import whole_79_16 as W16
from catspace.incubator.analyst.discovery import whole_play as WP
from catspace.io import paths

OUT = ("whole_79_17",)
KIND_TABLE = "mm_table"
CAP = 200                                           # the ply cap: above the game's own rule on quiet plies (a hundred plies), so that the rule is what ends a game that makes no progress
SEC_SHORT, SEC_LONG = 0.2, 2.0
MEASURE_D = 3                                       # the depth, in our moves, of the held goal's question (October's d from 1 to 3)
ENDINGS = ("won", "repetition", "quiet_rule", "cap", "stalemate", "unit_lost", "lost", "other")


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


def watchdog_for(seconds: float) -> float:
    """A game is up to a hundred plies a side at the clock: scaled to it."""
    return max(600.0, 700.0 * seconds)


# ------------------------------------------------------------------------------------------------------------------------------ the movers

class MoverCell(WP.MoverWhole):
    """79.11's movers plus the table's defender; the wake's spec may carry the measure to hold."""

    def __init__(self, spec: dict, seconds: float, seed: str, trace: str):
        super().__init__(spec, seconds, seed, trace)
        self.measure = spec.get("measure")
        self.start = spec.get("start_text")

    def chooser_kw(self, ply: int) -> dict:
        kw = super().chooser_kw(ply)
        if self.measure:
            kw["measure"] = (self.measure["term"], int(self.measure["d"]))
        return kw

    def choose(self, text: "str | None", moves: list, legal: list) -> tuple:
        if self.kind != KIND_TABLE:
            return super().choose(text, moves, legal)
        t0 = time.perf_counter()
        mv = MO.defender_reply(text, list(moves), list(legal))
        st = self.stats
        st["decisions"] += 1
        st["wall"] += time.perf_counter() - t0
        return mv, None


def play17(spec: dict) -> dict:
    """One game as 79.11 plays it, with the movers above."""
    WP.MoverWhole = MoverCell
    return WP.play11(spec)


def player(name: str, seconds: float, measure: "dict | None" = None) -> dict:
    if name in ("wake", "wake_goal"):
        p = WP.player_spec("wake", seconds, goal_table=W15.table(), pieces=W16.CONFIGS["abcd"], pre_scale=1.0)
        if name == "wake_goal":
            p["measure"] = dict(measure)
        return p
    if name == "old":
        return WP.player_spec("old", seconds)
    if name == "table":
        return {"kind": KIND_TABLE, "worth": {}, "charge": False}
    raise ValueError(name)


def cell_specs(cls: str, starts: list, attacker: dict, defender: dict, seconds: float, scratch: Path, log_dir: Path, label: str, cap: int = CAP) -> list:
    out = []
    for i, st in enumerate(starts):
        out.append({"gid": f"{label}:{cls}{i}:0", "start": f"{cls}{i}", "colour": 0, "text": st["text"], "opening": [], "players": [attacker, defender], "seconds": seconds, "max_plies": cap, "scratch": str(scratch), "log_dir": str(log_dir), "attacker_white": st["attacker_white"]})
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the numbers of a cell

def won_by_attacker(g: dict, start: dict) -> bool:
    return g["outcome"] is not None and g["outcome"] == (100 if start["attacker_white"] else 0)


def ending_of_game(g: dict, start: dict) -> str:
    return MO.ending_of(start["text"], g["moves"], g["outcome"], g["unfinished"], start["attacker_white"])


def summarise(games: list, starts: list, cls: str) -> dict:
    by_gid = {g["gid"].split(":")[1]: g for g in games}
    rows = []
    for i, st in enumerate(starts):
        g = by_gid.get(f"{cls}{i}")
        if g is None:
            continue
        rows.append((g, st))
    n = len(rows)
    wins = [(g, st) for g, st in rows if won_by_attacker(g, st)]
    ends = Counter(ending_of_game(g, st) for g, st in rows)
    depth = sum(g["stats"][0].get("depth_sum", 0) for g, _ in rows)
    dec = sum(g["stats"][0].get("decisions", 0) for g, _ in rows)
    return {"games": n, "wins": len(wins), "mean_plies": sum(g["plies"] for g, _ in rows) / max(1, n), "mean_plies_wins": sum(g["plies"] for g, _ in wins) / max(1, len(wins)),
            "ended": {k: ends.get(k, 0) for k in ENDINGS if ends.get(k, 0)}, "bins": MO.coverage([st for _, st in rows]), "depth": depth / max(1, dec), "decisions": dec,
            "wall": sum(g["stats"][0].get("wall", 0.0) for g, _ in rows) / max(1, dec)}


def cell_line(name: str, s: dict) -> str:
    return (f"OBS whole_79_17 cell {name}: games {s['games']}, wins {s['wins']}, mean plies {s['mean_plies']:.1f} (of the wins {s['mean_plies_wins']:.1f}); ended by {s['ended']}; the table's distance at the start by bins {s['bins']}; "
            f"the attacker's mean depth {s['depth']:.2f} over {s['decisions']} decisions, mean wall a move {s['wall']:.4f} s")


def decisions_of(log_dir: Path) -> list:
    out = []
    for p in sorted(Path(log_dir).glob("*.jsonl.gz")):
        rows = []
        with gzip.open(p, "rt") as f:
            for x in f:
                try:
                    rows.append(json.loads(x))
                except ValueError:
                    continue
        out.append((p.name, [r for r in rows if r.get("kind") == "decision"], [r for r in rows if r.get("kind") == "result"]))
    return out


def falls(log_dir: Path) -> dict:
    """The fall of the held measure along the games of a folder: the drops a game (events of ledger row 24), the longest run of our moves with no drop, the goals dropped and why, the moves toward the goal played."""
    games = decisions_of(log_dir)
    n = drops = longest = toward = decisions = 0
    why: Counter = Counter()
    per_game = []
    for name, decs, _ in games:
        m = [d["wake"]["log"].get("measure") for d in decs if d["wake"]["log"].get("measure")]
        if not m:
            continue
        n += 1
        run = best = d_n = 0
        for x in m:
            decisions += 1
            if x["reached"]:
                d_n += 1
                run = 0
            else:
                run += 1
                best = max(best, run)
            toward += bool(x["played_toward"])
        first_drop = next((x["dropped"] for x in m if x["dropped"]), None)           # a goal is dropped once: the later decisions repeat the cause
        if first_drop:
            why[first_drop.split(":")[0][:60]] += 1
        drops += d_n
        longest = max(longest, best)
        per_game.append({"game": name, "drops": d_n, "longest_run": best, "decisions": len(m)})
    # a cause is counted once a game: the goal is dropped once and the later decisions repeat it
    return {"games": n, "drops": drops, "drops_a_game": drops / max(1, n), "longest_run": longest, "toward_played": toward, "decisions": decisions, "dropped_why": dict(why), "per_game": per_game}


# ------------------------------------------------------------------------------------------------------------------------------ the stages

def starts_for(cls: str, n: int, tag_seed: int = 17) -> list:
    """The starts of a class; a class named with the ending far is the second sample (the greatest table distance among twenty times as many drawn)."""
    if cls.endswith("far"):
        return MO.far_starts(cls[:-3], n, tag_seed)
    return MO.sample_starts(cls, n, tag_seed)


def cmd_cells(a, say) -> None:
    root = out_root(a.tag)
    (root / "matches").mkdir(exist_ok=True)
    measure_file = out_root(a.measures_from or a.tag) / "measures.json"          # the mined measures (part C), by class
    measures = json.loads(measure_file.read_text()) if measure_file.exists() else {}
    for cls in a.classes.split(","):
        starts = starts_for(cls, a.n)
        (root / f"starts_{cls}.json").write_text(json.dumps(starts))
        say(f"OBS whole_79_17 phase {a.phase} class {cls}: {len(starts)} starts, the stronger side to move, colours alternating; the table's distance at the start by bins {MO.coverage(starts)} (the table is read here by the harness only)")
        for sec in [float(x) for x in a.clocks.split(",")]:
            for att in a.attackers.split(","):
                for dfn in a.defenders.split(","):
                    name = f"{a.phase}_{cls}_{att}_v_{dfn}_{sec}"
                    if att == "wake_goal" and cls not in measures:
                        say(f"OBS whole_79_17 {name}: no measure for the class in {measure_file}: not played")
                        continue
                    os.environ["CATSPACE_WATCHDOG"] = str(watchdog_for(sec))
                    log_dir = root / "logs" / name
                    specs = cell_specs(cls, starts, player(att, sec, measures.get(cls)), player(dfn, sec), sec, root / "scratch", log_dir, name)
                    load0, _ = TM.wait_quiet(say)
                    t0 = time.perf_counter()
                    games = TM.run_games(specs, a.workers, root / "matches" / f"{name}.jsonl", say, play=play17, stop=None)
                    s = summarise(games, starts, cls)
                    (root / f"cell_{name}.json").write_text(json.dumps(s))
                    say(cell_line(name, s) + f"; {time.perf_counter() - t0:.0f} s, load at start {load0:.2f}")
                    if att == "wake_goal":
                        say(f"OBS whole_79_17 falls {name}: " + json.dumps({k: v for k, v in falls(log_dir).items() if k != "per_game"}))


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("cells",))
    ap.add_argument("--phase", default="A")
    ap.add_argument("--classes", default="two,one")
    ap.add_argument("--attackers", default="wake,old")
    ap.add_argument("--defenders", default="table,old")
    ap.add_argument("--clocks", default="0.2,2.0")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--measures-from", default=None)
    a = ap.parse_args(argv)
    root = out_root(a.tag)

    def say(line: str) -> None:
        TM.emit([line], root / "obs.txt")
    cmd_cells(a, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
