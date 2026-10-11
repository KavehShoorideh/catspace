"""ONE GAME, MOVE BY MOVE, TIMED (Kaveh 2026-10-11: "take a random game, step through the moves and analyze each move,
and profile the time, and see what's taking long"). A census game is chosen by seed and walked with the game's PlanState
(the memory kept across moves); for every move: the planner's time, its stages (plan.rs stage_times: the clock marks
the end of each stage of plan_inner, plus everything after it -- the audit, the defences by net), proofs and nodes, the
decision. Then the totals by stage, the slowest moves, and (one process) macOS `sample` over the whole walk for the
Rust functions on top of the stack. Flags as the planner's defaults unless set here.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

from catspace.io import paths


def run(tag: str, seed: int, game: str | None, sample_s: int, flags: dict) -> None:
    os.environ["RAYON_NUM_THREADS"] = "1"
    import analyst_core
    import chess
    from .waiting_frames import _games
    games = _games(tag, 300)
    if game is not None:
        pick = next(g for g in games if str(g[0]) == game)
    else:
        pick = random.Random(seed).choice(games)
    gid, fens = pick
    if flags.pop("policy", None):
        analyst_core.set_forcing_policy(open(paths.derived("forcing_policy.json")).read())
    for k, v in flags.items():
        getattr(analyst_core, f"set_{k}")(v)
    out = paths.experiment(f"game_profile_{gid}.json")
    sample_file = paths.experiment(f"game_profile_{gid}_sample.txt")
    proc = None
    if sample_s:
        proc = subprocess.Popen(["sample", str(os.getpid()), str(sample_s), "1", "-file", str(sample_file)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    st = analyst_core.PlanState()
    rows = []
    totals = defaultdict(float)
    prims = defaultdict(lambda: [0, 0.0])
    t_all = time.time()
    analyst_core.stage_times(); analyst_core.forcing_stats(None); analyst_core.forcing_unknowns()
    unknowns = []
    for ply, fen in enumerate(fens):
        b = chess.Board(fen)
        t0 = time.time()
        r = st.observe(fen)
        wall = (time.time() - t0) * 1000
        stages = dict(analyst_core.stage_times())
        unknowns.extend(analyst_core.forcing_unknowns())
        for k, c, v in analyst_core.prim_times():
            prims[k][0] += c; prims[k][1] += v
        for k, v in stages.items():
            totals[k] += v
        dec = r[1]
        san = ""
        if dec:
            try:
                san = b.san(chess.Move.from_uci(dec[1]))
            except Exception:
                san = dec[1]
        top = sorted(stages.items(), key=lambda kv: -kv[1])[:3]
        rows.append(dict(ply=ply, fen=fen, ms=r[4], wall=wall, proofs=r[2], nodes=r[3], decision=dec[3] if dec else None,
                         san=san, stages=stages))
        print(f"OBS ply {ply:3d} {('W' if b.turn else 'B')} {r[4]:7.1f} ms  proofs {r[2]:3d} nodes {r[3]:8,d}  "
              f"{' | '.join(f'{k} {v:.0f}' for k, v in top):60s}  {dec[3] if dec else '-'}")
    if proc:
        proc.wait()
    n = len(rows)
    tot = sum(r["ms"] for r in rows)
    print(f"OBS game {gid}: {n} plies, {tot / 1000:.1f} s of planner time ({tot / n:.0f} ms a move), wall {time.time() - t_all:.0f} s")
    print("OBS   by stage: " + ", ".join(f"{k} {v / 1000:.1f} s ({100 * v / tot:.0f}%)" for k, v in sorted(totals.items(), key=lambda kv: -kv[1])))
    print("OBS   primitives (inclusive): " + "; ".join(f"{k} {v[1] / 1000:.1f} s ({100 * v[1] / tot:.0f}%, {v[0]:,} calls, {v[1] / max(v[0], 1):.2f} ms each)" for k, v in sorted(prims.items(), key=lambda kv: -kv[1][1])))
    fs = analyst_core.forcing_stats(None)
    print(f"OBS   forcing searches: {fs[0]:,} -- proven {fs[1]:,}, disproven {fs[2]:,}, UNKNOWN (safety net hit) {fs[3]:,} ({100 * fs[3] / max(fs[0], 1):.0f}%), expansions {fs[4]:,}")
    # the safety-net positions re-asked with a budget 10x and 50x: do they resolve, and how?
    import random as _r
    pick = _r.Random(1).sample(unknowns, min(40, len(unknowns))) if unknowns else []
    res = {10: [0, 0, 0], 50: [0, 0, 0]}
    for f in pick:
        for mult in (10, 50):
            v, _ = analyst_core.forcing_value_of(f, 200 * mult)
            r = analyst_core.forcing_gain_of(f, 200 * mult) if hasattr(analyst_core, 'forcing_gain_of') else None
            res[mult][0 if v is not None else 2] += 1
    print(f"OBS   safety-net positions sampled {len(pick)} of {len(unknowns)}: with budget x10 a forcing value found in {res[10][0]}, none in {res[10][2]}; x50 found {res[50][0]}, none {res[50][2]}")
    with open(paths.experiment(f"game_profile_{gid}_unknowns.json"), "w") as fh:
        json.dump(pick, fh, indent=1)
    slow = sorted(rows, key=lambda r: -r["ms"])[:5]
    print("OBS   slowest moves: " + "; ".join(f"ply {r['ply']} {r['ms']:.0f} ms ({max(r['stages'].items(), key=lambda kv: kv[1])[0] if r['stages'] else '-'})" for r in slow))
    with open(out, "w") as fh:
        json.dump(dict(game=gid, rows=rows, totals=totals), fh, indent=1)
    print(f"OBS   -> {out}")
    if sample_s and sample_file.exists():
        txt = sample_file.read_text()
        i = txt.find("Sort by top of stack")
        lines = [l for l in txt[i:].splitlines()[1:40] if l.strip() and "__wait4" not in l and "libsystem" not in l]
        print("OBS   top of stack (Rust): " + "; ".join(l.split("(in")[0].strip()[-60:] + " " + l.split()[-1] for l in lines[:12]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--game", default=None)
    ap.add_argument("--sample", type=int, default=0, help="seconds of macOS sample over the walk (0 off)")
    ap.add_argument("--cert", action="store_true")
    ap.add_argument("--frame", action="store_true")
    ap.add_argument("--bound", action="store_true")
    ap.add_argument("--policy", action="store_true", help="load data/derived/forcing_policy.json")
    a = ap.parse_args(argv)
    flags = {}
    if a.cert:
        flags["cert_reuse"] = True
    if a.frame:
        flags["frame_proofs"] = True
    if a.bound:
        flags["net_bound"] = True
    if a.policy:
        flags["policy"] = True
    run(a.tag, a.seed, a.game, a.sample, flags)


if __name__ == "__main__":
    main()
