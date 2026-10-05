"""MEASURE: the fixed protocol. Three held-out sets, frozen in round 0 with their baseline dumps (the plain player):
back-rank puzzles (rows i % 97 == 17), pawn-ending puzzles (the same rows), and the general set (stride 997, offset
300). For a round: a static FIRE scan of every active idea at the solver's positions of each set (instances, proved,
one-away), then the player with the round's wiring (active.json) on each set, paired by position against the
baseline: nodes, solved among the tied top moves and as the single top move, gained and lost puzzles, wall time.
Round 0 measures from the dumps already on disk (mode "dumps"); later rounds run the player (mode "player")."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time

import chess

from catspace.io import paths

from ..contracts import Measurement, to_dict
from ..store import read_json, write_json

SETS = {
    "backrank": dict(args=["--n", "300", "--stride", "97", "--offset", "17", "--theme", "backRankMate"],
                     baseline="frame_puzzles_lichess_zone_candidates_off17_backRankMate_lean1004.json"),
    "pawnendings": dict(args=["--n", "300", "--stride", "97", "--offset", "17", "--theme", "pawnEndgame"],
                        baseline="frame_puzzles_lichess_zone_candidates_off17_pawnEndgame_lean1005.json"),
    "general": dict(args=["--n", "300", "--stride", "997", "--offset", "300"],
                    baseline="frame_puzzles_lichess_zone_candidates_off300_lean1004.json"),
}


def load_dump(name: str) -> list:
    return json.load(open(paths.derived(name)))


def summarize(res: list) -> dict:
    ns = [s.get("nodes", 0) or 0 for r in res for s in r["steps"]]
    return dict(puzzles=len(res), steps=len(ns), mean_nodes=sum(ns) / max(len(ns), 1),
                tied=sum(r["solved"] for r in res), single=sum(r["solved_unique"] for r in res))


def pair(base: list, arm: list) -> dict:
    """The arm against the baseline on the same puzzles: nodes on the steps both searched from the same position,
    puzzles gained and lost (whole solution among the tied top moves)."""
    x = {r["id"]: r for r in base}
    y = {r["id"]: r for r in arm}
    common = [k for k in x if k in y]
    pairs = [(sx["nodes"], sy["nodes"]) for k in common for sx in x[k]["steps"] for sy in y[k]["steps"]
             if sx["fen"] == sy["fen"] and sx.get("nodes") is not None and sy.get("nodes") is not None]
    nx = sum(a for a, _ in pairs)
    ny = sum(c for _, c in pairs)
    return dict(steps=len(pairs), before=nx, after=ny, ratio=ny / max(nx, 1),
                fewer=sum(c < a for a, c in pairs), more=sum(c > a for a, c in pairs), equal=sum(c == a for a, c in pairs),
                gained=sorted(k for k in common if y[k]["solved"] and not x[k]["solved"]),
                lost=sorted(k for k in common if x[k]["solved"] and not y[k]["solved"]),
                gained_single=sum(1 for k in common if y[k]["solved_unique"] and not x[k]["solved_unique"]),
                lost_single=sum(1 for k in common if x[k]["solved_unique"] and not y[k]["solved_unique"]))


def positions_of(res: list) -> list[str]:
    return [s["fen"] for r in res for s in r["steps"]]


def fire_scan(active: dict, li, fens: list[str], orderer=None) -> dict:
    """Per active idea: at how many of these positions it has an instance, how many are proved, how many one-away."""
    out = {}
    if "lean:backRankMate" in active.get("closures", []):
        inst = proved = one = 0
        for f in fens:
            got = li.backrank(chess.Board(f))
            inst += bool(got)
            proved += any(i.status == "MATE" for i in got)
            one += any(i.status == "ONEAWAY" for i in got)
        out["lean:backRankMate"] = dict(positions=len(fens), instances=inst, proved=proved, oneaway=one)
    for idea, cmd in (("lean:unstoppablePasser", "passer"), ("lean:unstoppablePasser2", "passer2")):
        if idea not in active.get("closures", []):
            continue
        inst = proved = one = 0
        for f in fens:
            got = li.passer(chess.Board(f)) if cmd == "passer" else li.passer2(chess.Board(f))
            inst += bool(got)
            proved += any(i.status == "PROVED" for i in got)
            one += any(i.status == "ONEAWAY" for i in got)
        out[idea] = dict(positions=len(fens), instances=inst, proved=proved, oneaway=one)
    if orderer is not None:
        for idea, rec in orderer.fire(fens).items():
            out[idea] = rec
    return out


def run_player(name: str, spec: dict, active_path: str | None, tag: str, ctx) -> tuple[str, float]:
    """The player on one set with the round's wiring. -> (dump name, wall seconds)."""
    cmd = [sys.executable, "-m", "catspace.incubator.analyst.frame_puzzles", *spec["args"], "--show", "0", "--tag", tag]
    if active_path:
        cmd += ["--loop-active", active_path]
    ctx.log(f"player on {name}: {' '.join(cmd[2:])}")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=paths.REPO_ROOT, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    (ctx.dir / f"player_{name}.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"frame_puzzles failed on {name}: {r.stderr[-2000:]}")
    m = re.search(r"^wrote (.+\.json)$", r.stdout, re.M)
    if not m:
        raise RuntimeError(f"frame_puzzles wrote no dump on {name}")
    return m.group(1).rsplit("/", 1)[-1], wall


def run(ctx) -> dict:
    from ...lean_ideas import LeanIdeas
    from ..wiring import MinedOrder
    cfg = ctx.cfg("measure")
    use = ctx.load("use") or {}
    active = use.get("active") or {}
    baselines = (read_json(ctx.dir.parent / "round_000" / "measure.json") or {}).get("output", {}).get("baselines") or {}
    out = dict(measurements=[], fire={}, baselines={})
    orderer = MinedOrder(active.get("order", [])) if active.get("order") else None
    with LeanIdeas() as li:
        for name, spec in SETS.items():
            base_name = spec["baseline"]
            base = load_dump(base_name)
            bwall = (baselines.get(name) or {}).get("wall_s")
            if cfg.get("time_baselines") and ctx.round == 0:
                again, bwall = run_player(name, spec, None, "loop000plain", ctx)
                p = pair(base, load_dump(again))
                same = p["before"] == p["after"] and not p["gained"] and not p["lost"]
                ctx.log(f"OBS baseline {name} re-run: identical to the frozen dump: {same} (nodes {p['before']} -> {p['after']}); wall {bwall:.0f}s")
                if not same:
                    raise RuntimeError(f"the baseline {name} is not reproducible: {p}")
            out["baselines"][name] = dict(dump=base_name, wall_s=bwall, **summarize(base))
            out["measurements"].append(to_dict(Measurement(name, "baseline", base_name, wall_s=bwall, paired=None, gained=[], lost=[],
                                                           provenance=ctx.prov("measure"), **summarize(base))))
            if cfg.get("mode") == "dumps":
                arm_name, wall = cfg["arms"][name], None
            else:
                arm_name, wall = run_player(name, spec, use.get("active_path"), f"loop{ctx.round:03d}", ctx)
            arm = load_dump(arm_name)
            p = pair(base, arm)
            m = Measurement(name, "round", arm_name, wall_s=wall, paired=p, gained=p["gained"], lost=p["lost"],
                            provenance=ctx.prov("measure"), **summarize(arm))
            out["measurements"].append(to_dict(m))
            ctx.log(f"OBS {name}: baseline {summarize(base)['tied']}/{len(base)} tied, {summarize(base)['single']} single, "
                    f"{summarize(base)['mean_nodes']:.0f} nodes/step; round {m.tied}/{m.puzzles} tied, {m.single} single, "
                    f"{m.mean_nodes:.0f} nodes/step; paired {p['before']} -> {p['after']} ({p['ratio']:.3f}; fewer {p['fewer']}, "
                    f"more {p['more']}, equal {p['equal']}); gained {len(p['gained'])}, lost {len(p['lost'])}"
                    + (f"; wall {wall:.0f}s" if wall else ""))
            out["fire"][name] = fire_scan(active, li, positions_of(base), orderer)
            for idea, rec in out["fire"][name].items():
                ctx.log(f"OBS fire {name} {idea}: {rec}")
    return out
