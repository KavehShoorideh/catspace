"""PLAN 78.85 FOLLOW-UP: THE SECOND READING OF THE EQUAL READING PLAYED AGAINST THE FIRST, by the engine itself. SYSTEM_SPEC 5. Game-free: the game is named only through the pieces that exist.

Both players are the control of 78.67 (the search of `solve_equals_batch`, its first move among the moves it holds equal); one asks the equal reading with the second step (`verify_ties` True, the engine's default as built),
the other without it (`verify_ties` False: the equal reading exactly as it was before plan 78.85). So the match measures only what the second step costs or gains the engine: it never changes the move, only the
search's clock (the equal reading takes the rest of the decision's clock), and, where the second step reaches the clock, nothing more.

INTERFACE. `VerifyMover(TM.Mover67)`: the control with `spec["verify_ties"]`; its stats add `eq_n` (decisions where the equal reading ran), `eq_s` (seconds in it, as the engine reports), `eq_cut` (cut by the clock), `eq_one` (an equal set of one
move), `eq_size` (the sum of the sizes). `play_verify(spec)` plays one game (module level for the spawned Pool). Stages: match (one opening set; the load wait of 78.67 first) | report. Called by the command line only.

  .venv/bin/python -m catspace.incubator.analyst.discovery.debts_verify match --set A --pairs 100 --tag follow | report --tag follow
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import time
from pathlib import Path

import numpy as np

import cs_engine as CE

from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
from catspace.io import paths

OUT = ("reasoner_debts_78_85",)
SIDES = ("verify", "plain")                                         # the player that asks the second reading, and the one that does not


def out_root(tag: str) -> Path:
    return Path(paths.experiment(*OUT, tag))


class VerifyMover(TM.Mover67):
    """The control with the second reading of the equal reading on or off (`spec["verify_ties"]`)."""

    def __init__(self, spec: dict, seconds: float, seed: str, trace: str):
        super().__init__(spec, seconds, seed, trace)
        self.verify = bool(spec.get("verify_ties", True))
        for k in ("eq_n", "eq_s", "eq_cut", "eq_one", "eq_size"):
            self.stats[k] = 0

    def choose(self, text: "str | None", moves: list, legal: list) -> tuple:
        t0 = time.perf_counter()
        row = PM.equal_set(CE.solve_equals_batch([(OM.VARIANT, text, list(moves))], self.seconds, worth=dict(self.worth), ours_first=False, valuation="odds record", threads=1,
                                                 ledger_share=TM.LEDGER_SHARE, verify_ties=self.verify)[0])
        wall = time.perf_counter() - t0
        st = self.stats
        st["decisions"] += 1
        st["search"] += wall
        st["wall"] += wall
        if row["ties_seconds"] and row["ties_seconds"] > 0:         # the equal reading ran (it is skipped where the line decides the game)
            st["eq_n"] += 1
            st["eq_s"] += row["ties_seconds"]
            st["eq_cut"] += int(bool(row["cut"]))
            st["eq_one"] += int(len(row["equals"]) == 1)
            st["eq_size"] += len(row["equals"])
        if row["move"] is not None and len(row["equals"]) >= 2:
            st["ties"] += 1
        return row["move"], row["depth"]


def play_verify(spec: dict) -> dict:
    OM.Mover = VerifyMover
    TM.REG.clear()
    g = OM.play_game(spec)
    g["stats"] = [dict(m.stats) for m in TM.REG]
    return g


def games_file(tag: str, set_name: str) -> Path:
    return out_root(tag) / "matches" / f"verify_v_plain_{set_name}.jsonl"


def specs_of(set_name: str, pairs: int, seconds: float, scratch: Path) -> list:
    base = OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")
    pa = dict(base, kind=TM.KIND["control"], charge=False, verify_ties=True)
    pb = dict(base, kind=TM.KIND["control"], charge=False, verify_ties=False)
    out = []
    for seed, opening in TM.openings(set_name, pairs):
        for colour in (0, 1):
            out.append({"gid": f"verify_v_plain:{set_name}{seed}:{colour}", "start": f"{set_name}{seed}", "colour": colour, "text": None, "opening": opening, "players": [pa, pb] if colour == 0 else [pb, pa],
                        "seconds": seconds, "max_plies": TM.PLY_CAP, "scratch": str(scratch)})
    return out


def per_side(games: list) -> list:
    """Per side (verify, plain): mean ledger depth, mean seconds in the equal reading, share of equal readings cut by the clock, share of equal sets of one move, mean size of the equal set."""
    out = []
    for who in (0, 1):
        dep, tot = [0, 0], {}
        for g in games:
            seat = TM.seat_of(g, who)
            dep[0] += g["depth"][seat][0]
            dep[1] += g["depth"][seat][1]
            for k, v in g["stats"][seat].items():
                if k != "log":
                    tot[k] = tot.get(k, 0) + v
        n = max(1, tot["eq_n"])
        out.append(f"{SIDES[who]}: mean ledger depth {dep[0] / max(1, dep[1]):.2f}; equal readings {tot['eq_n']} of {tot['decisions']} decisions; mean seconds in the equal reading {tot['eq_s'] / n:.4f}; "
                   f"equal readings cut by the clock {100 * tot['eq_cut'] / n:.1f} pct; equal sets of one move {100 * tot['eq_one'] / n:.1f} pct; mean equal set {tot['eq_size'] / n:.2f}; decisions with two or more equal moves {tot['ties']}")
    return out


def match_lines(set_name: str, games: list, load: float, busy: float, wall: float, seconds: float, workers: int) -> list:
    sc = np.array([OM.white_score(g) if g["colour"] == 0 else 1 - OM.white_score(g) for g in games])
    w, d, l = int((sc == 1).sum()), int((sc == 0.5).sum()), int((sc == 0).sum())
    s, lo, hi = OM.score_band(OM.pair_scores(games))
    unf = sum(g["unfinished"] for g in games)
    lines = [f"OBS debts_verify match verify v plain [set {set_name}]: games {len(games)}: verify W {w} D {d} L {l}; score {s:.4f} [{lo:.4f}, {hi:.4f}] 90 pct band by bootstrap over opening pairs; unfinished {unf} ({100 * unf / len(games):.1f} pct, counted as draws)"]
    lines += [f"OBS debts_verify match verify v plain [set {set_name}]: {x}" for x in per_side(games)]
    lines.append(f"OBS debts_verify match verify v plain [set {set_name}]: load average (1 min) at start {load:.2f}, cores busy at the instant of start {busy:.2f}; wall {wall:.0f} s; clock {seconds} s a move, ledger share {TM.LEDGER_SHARE}, "
                 f"ply cap {TM.PLY_CAP}, workers {workers}")
    return lines


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("match", "report"))
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="follow")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)
    obs = root / "obs.txt"
    if a.stage == "match":
        load, busy = TM.wait_quiet(lambda s: TM.emit([s], obs))
        t0 = time.perf_counter()
        gf = games_file(a.tag, a.set)
        games = TM.run_games(specs_of(a.set, a.pairs, a.seconds, root / "scratch"), a.workers, gf, lambda s: TM.emit([s], obs), play=play_verify)
        wall = time.perf_counter() - t0
        gf.with_suffix(".meta.json").write_text(json.dumps({"load": load, "busy": busy, "wall": wall}))
        TM.emit(match_lines(a.set, games, load, busy, wall, a.seconds, a.workers), obs)
    else:
        found = {}
        for s in ("A", "B"):
            p = games_file(a.tag, s)
            if p.exists():
                found[s] = TM.read_games(p)
                meta = json.loads(p.with_suffix(".meta.json").read_text()) if p.with_suffix(".meta.json").exists() else {}
                TM.emit(match_lines(s, found[s], meta.get("load", float("nan")), meta.get("busy", float("nan")), meta.get("wall", float("nan")), a.seconds, a.workers), None)
        if len(found) > 1:
            allg = [g for gs in found.values() for g in gs]
            sc = np.array([OM.white_score(g) if g["colour"] == 0 else 1 - OM.white_score(g) for g in allg])
            s, lo, hi = OM.score_band(OM.pair_scores(allg))
            TM.emit([f"OBS debts_verify pooled verify v plain [sets AB]: games {len(allg)}: verify W {int((sc == 1).sum())} D {int((sc == 0.5).sum())} L {int((sc == 0).sum())}; score {s:.4f} [{lo:.4f}, {hi:.4f}]"] + [f"OBS debts_verify pooled: {x}" for x in per_side(allg)], None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
