"""THE TARGET-FIRST PLANNER, measured (rust/analyst_core/src/plan.rs; Kaveh 2026-10-09: "explain it in a reasonable
amount of calculations").

For every census position: the planner's proofs, nodes and time; what it found (an immediate win, an armed threat, or
nothing); compared with
  the frames     the waiting-frames pool replayed over the same games (a frame ACTIVE for the side to move with a goal
                 of 3 or more is an immediate tactic the planner must also find)
  Stockfish      the only-move tactic labels (tactic_reach.py)
The misses are listed: each is a surprise to explain with a lemma, not a number to average away.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict

import chess

from catspace.io import paths


def _one(args):
    game, ply, fen = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    trace, best, proofs, nodes, ms, imm = analyst_core.plan_trace(fen)
    return game, ply, fen, best, proofs, nodes, ms, imm


def run(tag: str, frames_file: str, workers: int, limit: int, save: str = "") -> None:
    from multiprocessing import Pool
    from .tactic_census_measure import load
    from .tactic_reach import is_tactic
    rows = load(tag)[:limit]
    t0 = time.time()
    with Pool(workers) as p:
        res = list(p.imap_unordered(_one, [(r["game"], r["ply"], r["fen"]) for r in rows], chunksize=32))
    dt = time.time() - t0
    n = len(res)
    proofs = sorted(r[4] for r in res); ms = sorted(r[6] for r in res)
    kinds = Counter("immediate" if r[3] and not r[3][0] else ("defend" if r[3] and r[3][0] == "defend" else ("armed" if r[3] else "nothing")) for r in res)
    print(f"OBS planner on {n} positions ({dt:.0f} s on {workers} processes): proofs a position median {proofs[n // 2]}, "
          f"90th percentile {proofs[int(0.9 * n)]}, max {proofs[-1]}, mean {sum(proofs) / n:.1f}; nodes a position "
          f"{sum(r[5] for r in res) / n:,.0f}; ms a position median {ms[n // 2]:.1f}, max {ms[-1]:.0f}; found {dict(kinds)}")
    plan = {(r[0], r[1]): r for r in res}
    if save:                                                      # per position, for diffing two builds
        with open(paths.experiment(f"plan_check_{tag}_{save}.jsonl"), "w") as fh:
            for r in res:
                fh.write(json.dumps(dict(game=r[0], ply=r[1], fen=r[2], imm=r[7], best=r[3], proofs=r[4], ms=r[6])) + "\n")
    # the frames: immediate tactics (active, goal >= 3) for the side to move
    plies = defaultdict(list)
    for r in rows:
        plies[r["game"]].append(r["ply"])
    frame_tac = {}
    for line in open(paths.experiment(frames_file)):
        g = json.loads(line)
        if not g.get("proofs", True):
            continue
        ps = sorted(plies.get(g["game"], []))
        if len(ps) != len(g["fens"]):
            continue
        for i, fen in enumerate(g["fens"]):
            stm = "w" if chess.Board(fen).turn else "b"
            frame_tac[(g["game"], ps[i])] = [k for k, v in g["hist"][i].items() if k[0] == stm and v[0] == "active" and v[4] >= 3]
    both = [k for k in plan if k in frame_tac]
    ft = [k for k in both if frame_tac[k]]
    found = [k for k in ft if plan[k][7]]
    decided = [k for k in ft if plan[k][3] and not plan[k][3][0]]
    print(f"OBS   frames' immediate tactics (goal >= 3) in {len(ft)} positions: the planner has an immediate win in {len(found)} "
          f"({100 * len(found) / max(len(ft), 1):.0f}%); it decides to play one in {len(decided)}")
    labels = {(r["game"], r["ply"]): r for r in map(json.loads, open(paths.experiment(f"tactic_labels_{tag}_d12_pv.jsonl")))}
    prev = {}
    for r in rows:
        prev[(r["game"], r["ply"])] = r["fen"]
    sf = [k for k in plan if k in labels and (k[0], k[1] - 1) in prev and is_tactic(labels[k], prev[(k[0], k[1] - 1)], plan[k][2])]
    sf_imm = [k for k in sf if plan[k][7]]
    sf_move = [k for k in sf_imm if plan[k][7][1] == labels[k]["move"]]
    print(f"OBS   Stockfish only-move tactics {len(sf)}: the planner has an immediate win in {len(sf_imm)} ({100 * len(sf_imm) / max(len(sf), 1):.0f}%), "
          f"with Stockfish's move {len(sf_move)}")
    for k in [k for k in ft if k not in found][:6]:
        print(f"OBS   frames found, planner did not: {frame_tac[k][:3]}  {plan[k][2]}")
    for k in [k for k in sf if k not in sf_imm][:6]:
        print(f"OBS   Stockfish tactic, planner none: best {labels[k]['move']} ({labels[k]['best']:+d} vs {labels[k]['second']:+d})  {plan[k][2]}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--frames", default="waiting_frames_v2_c300_300.jsonl")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--save", default="", help="write each position's result to plan_check_<tag>_<save>.jsonl")
    a = ap.parse_args(argv)
    run(a.tag, a.frames, a.workers, a.limit, a.save)


if __name__ == "__main__":
    main()
