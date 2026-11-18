"""PLAN 78.92 PLAY: WHAT EACH PRUNING BUYS IN PLAY. SYSTEM_SPEC 5 (the tactical reasoner's ledger search: its late-move reduction and its null move). Game-free: the game is named only through the pieces that exist.

Both players are the control of 78.67 (the search of `solve_equals_batch`, its first move among the moves it holds equal, the equal reading as it is by default); they differ in the two prunings of the search alone:
the player "now" has both on (the engine as it is), the other has one or both off. So the match measures only what the pruning buys the engine at the play clock: a move is the search's own and the clock is the same.

INTERFACE. `ChoiceMover(TM.Mover67)`: the control with `spec["null_move"]` and `spec["late_reduce"]` (both default True, the engine as it is); its stats add `nodes` (the nodes the searches spent) and `search`. `play_choice(spec)` plays one game
(module level for the spawned Pool). `VARIANTS`: the three matches, each the name of the player against "now": "late_off" (late-move reduction off), "null_off" (null move off), "both_off" (both off), "verify" (both on and the verification of the choice on: only the switch of plan 78.92 itself). Stages: match (one variant, one opening set;
the load wait of 78.67 first, 9 workers) | report (every match found, both sets and pooled). Output folder `reasoner_choice_78_92/<tag>/matches`. Called by the command line only.

  .venv/bin/python -m catspace.incubator.analyst.discovery.choice_play match --variant late_off --set A --pairs 100 --tag full | report --tag full
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

OUT = ("reasoner_choice_78_92",)
VARIANTS = {"late_off": {"null_move": True, "late_reduce": False}, "null_off": {"null_move": False, "late_reduce": True}, "both_off": {"null_move": False, "late_reduce": False},
            "verify": {"null_move": True, "late_reduce": True, "verify_choice": True}}                     # (the last: the switch of the verification of the choice, both prunings on, built only under the condition of choice_measure.py)
NOW = {"null_move": True, "late_reduce": True, "verify_choice": False}
VC_KEYS = ("vc_asked", "vc_held", "vc_replaced", "vc_failed", "vc_cut", "vc_nodes", "vc_seconds")


def out_root(tag: str) -> Path:
    return Path(paths.experiment(*OUT, tag))


class ChoiceMover(TM.Mover67):
    """The control with the two prunings of the search on or off (`spec["null_move"]`, `spec["late_reduce"]`)."""

    def __init__(self, spec: dict, seconds: float, seed: str, trace: str):
        super().__init__(spec, seconds, seed, trace)
        self.null_move = bool(spec.get("null_move", True))
        self.late_reduce = bool(spec.get("late_reduce", True))
        self.verify_choice = bool(spec.get("verify_choice", False))
        self.stats["nodes"] = 0
        for k in VC_KEYS:
            self.stats[k] = 0

    def choose(self, text: "str | None", moves: list, legal: list) -> tuple:
        t0 = time.perf_counter()
        raw = CE.solve_equals_batch([(OM.VARIANT, text, list(moves))], self.seconds, worth=dict(self.worth), ours_first=False, valuation="odds record", threads=1,
                                    ledger_share=TM.LEDGER_SHARE, null_move=self.null_move, late_reduce=self.late_reduce, verify_choice=self.verify_choice, choice_mode=self.choice_mode)[0]
        row = PM.equal_set(raw)
        state, vc_nodes, vc_seconds = raw[12], raw[13], raw[14]                          # (the last three of the row: what the verification of the choice did)
        wall = time.perf_counter() - t0
        st = self.stats
        st["decisions"] += 1
        st["search"] += wall
        st["wall"] += wall
        st["nodes"] += int(row["nodes"] or 0)
        if state:                                                    # 1 held, 2 replaced, 3 failed and kept, 4 cut
            st["vc_asked"] += 1
            st[("vc_held", "vc_replaced", "vc_failed", "vc_cut")[state - 1]] += 1
            st["vc_nodes"] += int(vc_nodes)
            st["vc_seconds"] += float(vc_seconds)
        if row["move"] is not None and len(row["equals"]) >= 2:
            st["ties"] += 1
        return row["move"], row["depth"]


def play_choice(spec: dict) -> dict:
    OM.Mover = ChoiceMover
    TM.REG.clear()
    g = OM.play_game(spec)
    g["stats"] = [dict(m.stats) for m in TM.REG]
    return g


def games_file(tag: str, variant: str, set_name: str) -> Path:
    return out_root(tag) / "matches" / f"now_v_{variant}_{set_name}.jsonl"


def specs_of(variant: str, set_name: str, pairs: int, seconds: float, scratch: Path) -> list:
    base = OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")
    pa = dict(base, kind=TM.KIND["control"], charge=False, **NOW)
    pb = dict(base, kind=TM.KIND["control"], charge=False, **VARIANTS[variant])
    out = []
    for seed, opening in TM.openings(set_name, pairs):
        for colour in (0, 1):
            out.append({"gid": f"choice92_{variant}:{set_name}{seed}:{colour}", "start": f"{set_name}{seed}", "colour": colour, "text": None, "opening": opening, "players": [pa, pb] if colour == 0 else [pb, pa],
                        "seconds": seconds, "max_plies": TM.PLY_CAP, "scratch": str(scratch)})
    return out


def per_side(games: list, names: tuple) -> list:
    """Per side (now, the variant): mean ledger depth, mean nodes a decision, decisions."""
    out = []
    for who in (0, 1):
        dep, tot = [0, 0], {"decisions": 0, "nodes": 0, **{k: 0 for k in VC_KEYS}}
        for g in games:
            seat = TM.seat_of(g, who)
            dep[0] += g["depth"][seat][0]
            dep[1] += g["depth"][seat][1]
            for k in tot:
                tot[k] += g["stats"][seat].get(k, 0)                  # (the games played before the verification's counters existed have none)
        extra = ""
        if tot["vc_asked"]:
            n = tot["vc_asked"]
            extra = (f"; the choice verified in {n} decisions: held {tot['vc_held']}, replaced {tot['vc_replaced']} ({100 * tot['vc_replaced'] / n:.1f} pct), failed and kept {tot['vc_failed']}, cut {tot['vc_cut']}; "
                     f"nodes of the verification mean {tot['vc_nodes'] / n:.0f}, seconds mean {tot['vc_seconds'] / n:.4f}")
        out.append(f"{names[who]}: mean ledger depth {dep[0] / max(1, dep[1]):.2f}; decisions {tot['decisions']}; nodes a decision mean {tot['nodes'] / max(1, tot['decisions']):.0f}{extra}")
    return out


def match_lines(variant: str, set_name: str, games: list, load: float, busy: float, wall: float, seconds: float, workers: int) -> list:
    sc = np.array([OM.white_score(g) if g["colour"] == 0 else 1 - OM.white_score(g) for g in games])
    w, d, l = int((sc == 1).sum()), int((sc == 0.5).sum()), int((sc == 0).sum())
    s, lo, hi = OM.score_band(OM.pair_scores(games))
    unf = sum(g["unfinished"] for g in games)
    tag = f"OBS choice_play match now v {variant} [set {set_name}]"
    lines = [f"{tag}: games {len(games)}: now W {w} D {d} L {l}; score of now {s:.4f} [{lo:.4f}, {hi:.4f}] 90 pct band by bootstrap over opening pairs; unfinished {unf} ({100 * unf / len(games):.1f} pct, counted as draws)"]
    lines += [f"{tag}: {x}" for x in per_side(games, ("now", variant))]
    lines.append(f"{tag}: load average (1 min) at start {load:.2f}, cores busy at the instant of start {busy:.2f}; wall {wall:.0f} s; clock {seconds} s a move, ledger share {TM.LEDGER_SHARE}, ply cap {TM.PLY_CAP}, workers {workers}")
    return lines


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("match", "report"))
    ap.add_argument("--variant", default="late_off", choices=tuple(VARIANTS))
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)
    obs = root / "obs.txt"
    if a.stage == "match":
        load, busy = TM.wait_quiet(lambda s: TM.emit([s], obs))
        t0 = time.perf_counter()
        gf = games_file(a.tag, a.variant, a.set)
        games = TM.run_games(specs_of(a.variant, a.set, a.pairs, a.seconds, root / "scratch"), a.workers, gf, lambda s: TM.emit([s], obs), play=play_choice)
        wall = time.perf_counter() - t0
        gf.with_suffix(".meta.json").write_text(json.dumps({"load": load, "busy": busy, "wall": wall}))
        TM.emit(match_lines(a.variant, a.set, games, load, busy, wall, a.seconds, a.workers), obs)
    else:
        pooled: dict = {}
        for v in VARIANTS:
            found = {}
            for s in ("A", "B"):
                p = games_file(a.tag, v, s)
                if p.exists():
                    found[s] = TM.read_games(p)
                    meta = json.loads(p.with_suffix(".meta.json").read_text()) if p.with_suffix(".meta.json").exists() else {}
                    TM.emit(match_lines(v, s, found[s], meta.get("load", float("nan")), meta.get("busy", float("nan")), meta.get("wall", float("nan")), a.seconds, a.workers), obs)
            if len(found) > 1:
                allg = [g for gs in found.values() for g in gs]
                sc = np.array([OM.white_score(g) if g["colour"] == 0 else 1 - OM.white_score(g) for g in allg])
                s, lo, hi = OM.score_band(OM.pair_scores(allg))
                pooled[v] = (s, lo, hi)
                TM.emit([f"OBS choice_play pooled now v {v} [sets AB]: games {len(allg)}: now W {int((sc == 1).sum())} D {int((sc == 0.5).sum())} L {int((sc == 0).sum())}; score of now {s:.4f} [{lo:.4f}, {hi:.4f}]"]
                        + [f"OBS choice_play pooled now v {v}: {x}" for x in per_side(allg, ("now", v))], obs)
        if "both_off" in pooled:
            lo = pooled["both_off"][1]
            TM.emit([f"OBS choice_play CONDITION part (2): now against both off, pooled: the lower end of now's band {lo:.4f}; the prunings clearly buy strength (above 0.5): {'YES' if lo > 0.5 else 'NO'}"], obs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
