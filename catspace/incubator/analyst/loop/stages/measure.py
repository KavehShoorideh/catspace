"""MEASURE: the fixed protocol. Three held-out sets, frozen in round 0 with their baseline dumps (the plain player):
back-rank puzzles (rows i % 97 == 17), pawn-ending puzzles (the same rows), and the general set (stride 997, offset
300). For a round: a static FIRE scan of every active idea at the solver's positions of each set (instances, proved,
one-away), then the player with the round's wiring (active.json) on each set, paired by position against the
baseline: nodes, solved among the tied top moves and as the single top move, the expected score of a random pick among
the tied top moves (what the controls read), gained and lost puzzles, wall time.
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


def expected_score(r: dict) -> float:
    """The chance that a player picking at random among its tied top moves plays a puzzle's whole solution: the product
    over the solution's steps of 1 / (the number of tied top moves), 0 at a step where the solution is not among them.
    'Solution among the tied top moves' is an upper bound (a perfect tie-break) and 'the single top move' a lower bound
    (no tie-break); a tie-break can only shrink tied sets, so it moves both bounds in fixed directions whatever it picks.
    This score moves either way, and it is what the RESULTS FALL control reads (loop round 1, 2026-10-05).
    MATES: where the solution itself mates, frame_puzzles accepts any mating move (Lichess does), so every mating move
    in the tie counts (finding 3, 2026-10-05: ties of mates had read as gambles); where the solution does not mate,
    only the solution counts (pass 2, 2026-10-05). Dumps before 2026-10-05 kept only best[:6]; a tied step whose stored
    list lacks the solution assumes it beyond the list."""
    p = 1.0
    for s in r["steps"]:
        if not s["tied"]:
            return 0.0
        n = max(s.get("n_top") or 1, 1)
        best = s.get("best") or []
        if _mates(s["fen"], s["sol"]):
            good = sum(1 for u in best if u == s["sol"] or _mates(s["fen"], u)) or 1
        else:
            good = 1
        p *= min(good, n) / n
    return p


def _mates(fen: str, uci: str) -> bool:
    b = chess.Board(fen)
    try:
        m = chess.Move.from_uci(uci)
    except ValueError:
        return False
    if m not in b.legal_moves:
        return False
    b.push(m)
    return b.is_checkmate()


def summarize(res: list) -> dict:
    ns = [s.get("nodes", 0) or 0 for r in res for s in r["steps"]]
    return dict(puzzles=len(res), steps=len(ns), mean_nodes=sum(ns) / max(len(ns), 1),
                tied=sum(r["solved"] for r in res), single=sum(r["solved_unique"] for r in res),
                expected=sum(expected_score(r) for r in res))


def pair(base: list, arm: list) -> dict:
    """The arm against the baseline on the same puzzles: nodes on the steps both searched from the same position,
    puzzles gained and lost (whole solution among the tied top moves: context only, see expected_score), and the
    expected scores with the puzzles whose expected score went up and down."""
    x = {r["id"]: r for r in base}
    y = {r["id"]: r for r in arm}
    common = [k for k in x if k in y]
    ex = {k: expected_score(x[k]) for k in common}
    ey = {k: expected_score(y[k]) for k in common}
    pairs = [(sx["nodes"], sy["nodes"]) for k in common for sx in x[k]["steps"] for sy in y[k]["steps"]
             if sx["fen"] == sy["fen"] and sx.get("nodes") is not None and sy.get("nodes") is not None]
    nx = sum(a for a, _ in pairs)
    ny = sum(c for _, c in pairs)
    return dict(steps=len(pairs), before=nx, after=ny, ratio=ny / max(nx, 1),
                fewer=sum(c < a for a, c in pairs), more=sum(c > a for a, c in pairs), equal=sum(c == a for a, c in pairs),
                gained=sorted(k for k in common if y[k]["solved"] and not x[k]["solved"]),
                lost=sorted(k for k in common if x[k]["solved"] and not y[k]["solved"]),
                gained_single=sum(1 for k in common if y[k]["solved_unique"] and not x[k]["solved_unique"]),
                lost_single=sum(1 for k in common if x[k]["solved_unique"] and not y[k]["solved_unique"]),
                expected_before=sum(ex.values()), expected_after=sum(ey.values()),
                up=sorted(k for k in common if ey[k] > ex[k] + 1e-12), down=sorted(k for k in common if ey[k] < ex[k] - 1e-12))


def enrich(out: dict, previous: dict | None) -> dict:
    """The measure output with every expected score, and each round arm paired against the previous kept round's arm
    (the player before this round's ideas: the attribution of the round), all recomputed from the dumps the
    measurements name. Idempotent; a missing dump leaves its fields unset. -> a new dict and its OBS lines."""
    import copy
    out = copy.deepcopy(out or {})
    obs = []

    def dump(name):
        try:
            return load_dump(name)
        except FileNotFoundError:
            return None
    prev_arms = {m["set"]: m["dump"] for m in (previous or {}).get("measurements", []) if m["arm"] == "round"}
    rows = out.get("measurements", [])
    for m in rows:
        res = dump(m["dump"])
        if res is not None:
            m["expected"] = summarize(res)["expected"]
    for m in rows:
        if m["arm"] != "round" or m.get("expected") is None:
            continue
        res = dump(m["dump"])
        base = next((b for b in rows if b["set"] == m["set"] and b["arm"] == "baseline"), None)
        bres = dump(base["dump"]) if base else None
        if bres is not None:
            p = pair(bres, res)
            m["paired"] = {**(m.get("paired") or {}), **{k: p[k] for k in ("expected_before", "expected_after", "up", "down")}}
        pd = prev_arms.get(m["set"])
        pres = dump(pd) if pd and pd != m["dump"] else None
        if pres is not None:
            q, s = pair(pres, res), summarize(pres)
            m["vs_previous"] = dict(dump=pd, expected_before=q["expected_before"], expected_after=q["expected_after"],
                                    up=q["up"], down=q["down"], tied_before=s["tied"], single_before=s["single"],
                                    nodes_ratio=q["ratio"])
        v = m.get("vs_previous") or {}
        obs.append(f"OBS {m['set']}: expected score (random pick among the tied top moves): plain "
                   f"{(m.get('paired') or {}).get('expected_before', float('nan')):.1f}"
                   + (f", previous round {v['expected_before']:.1f}" if v else "")
                   + f", this round {m['expected']:.1f}"
                   + (f"; against the previous round {len(v['up'])} puzzles up, {len(v['down'])} down" if v else ""))
    return out, obs


def identical(a: list, b: list) -> bool:
    """Two dumps of the same puzzles agree step by step: position, tie size, top moves, nodes."""
    key = lambda res: {r["id"]: [(s["fen"], s.get("n_top"), tuple((s.get("best") or ())[:6]), s.get("nodes")) for s in r["steps"]] for r in res}
    # [:6]: the frozen baselines kept best[:6]; the whole tie is stored since 2026-10-05
    return key(a) == key(b)


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
        if idea not in active.get("closures", []) or "lean:unstoppablePasser2" not in active.get("closures", []):
            continue                                                        # the player switches on the passer only with passer2
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
            if cfg.get("time_baselines") and ctx.round == 0 and not bwall:     # timed once; a re-run keeps the timing
                again, bwall = run_player(name, spec, None, "loop000plain", ctx)
                p = pair(base, load_dump(again))
                same = identical(base, load_dump(again))                       # step by step, not node totals
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
            ctx.log(f"OBS {name}: expected score (random pick among the tied top moves): baseline {summarize(base)['expected']:.1f}, "
                    f"round {m.expected:.1f}; puzzles up {len(p['up'])}, down {len(p['down'])}")
            out["fire"][name] = fire_scan(active, li, positions_of(base), orderer)
            for idea, rec in out["fire"][name].items():
                ctx.log(f"OBS fire {name} {idea}: {rec}")
    return out
