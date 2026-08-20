#!/usr/bin/env python
"""gen_counterfactual_branches.py -- one random move, SF plays the continuation; contrast outcomes.

"play stockfish versus stockfish games, then at some random point play a RANDOM move as opposed
to a stockfish move, and contrast the outcomes. One move becomes random, only one move -- the
rest stockfish plays the continuation." (Kaveh 2026-08-19)

WHAT THIS BUYS, in the project's own terms. The no-ranking ruling established that played-only
data carries ~5 bits per position (the argmax identity) and NOTHING about unplayed siblings; the
two legal sources of sibling information are TD recombination and GENERATION. This is
generation as a PAIRED TRIAL: same position, treatment = one random legal move, control = the
move SF played, identical continuation policy on both branches. The outcome difference is a
grounded value-delta for an UNPLAYED move -- the exact gradient every scorer has measurably
lacked -- with no evaluator anywhere: the engine only ever PLAYS. Robotics-clean: intervention
rollouts under your own policy.

DESIGN CALLS, stated:
    control  = the corpus game's actual result from that row (free, already on disk). Caveat
               carried: corpus-SF and local-SF configs differ slightly, a small asymmetric bias
               accepted in v1 rather than doubling the cost by re-playing controls.
    treatment= random legal move != played, then ONE local SF instance plays both sides at
               --movetime ms until the game ends or --cap plies; at the cap the branch is
               adjudicated to the side up >= 3 material points, else draw (arena convention).
    outcomes are from the PARENT MOVER's point of view, +-1/0.

    .venv/bin/python -m ...gen_counterfactual_branches --n 200 --workers 2      # pilot
"""
from __future__ import annotations

import argparse
import time

import numpy as np

VAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}


def _run_shard(job):
    """worker: own engine instance, plays out its share of branches."""
    (rows, toks, globs, played_next, engine_path, movetime, cap, seed) = job
    import chess
    import chess.engine
    from catspace.research.tools.training_infra.concepts_reference import board_from_row
    rng = np.random.default_rng(seed)
    eng = chess.engine.SimpleEngine.popen_uci(engine_path)
    out = []
    for i in range(len(rows)):
        b = board_from_row(toks[i], globs[i])
        mv = list(b.legal_moves)
        if len(mv) < 2 or b.is_game_over():
            continue
        # the played move, identified by child-board match (same convention as every cache)
        played = None
        for m in mv:
            b.push(m)
            t = np.zeros(64, np.uint8)
            for q, p in b.piece_map().items():
                from catspace.research.tools.training_infra.concepts_reference import TOK_PIECES
                t[q] = TOK_PIECES.index((p.piece_type, p.color)) + 1
            if np.array_equal(t, played_next[i]):
                played = m
            b.pop()
            if played is not None:
                break
        alts = [m for m in mv if m != played]
        if played is None or not alts:
            continue
        rnd = alts[rng.integers(0, len(alts))]
        mover_white = b.turn
        b.push(rnd)
        plies = 1
        res = None
        while plies < cap:
            oc = b.outcome(claim_draw=True)
            if oc is not None:
                res = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[oc.result()]
                break
            b.push(eng.play(b, chess.engine.Limit(time=movetime / 1000)).move)
            plies += 1
        if res is None:
            oc = b.outcome(claim_draw=True)
            if oc is not None:
                res = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[oc.result()]
            else:
                d = sum((1 if p.color else -1) * VAL[p.piece_type]
                        for p in b.piece_map().values())
                res = 1 if d >= 3 else (-1 if d <= -3 else 0)
        res_mover = res if mover_white else -res
        out.append((int(rows[i]), rnd.from_square, rnd.to_square,
                    int(res_mover), int(plies)))
    eng.quit()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=50000, help="parents to branch")
    ap.add_argument("--movetime", type=float, default=10.0, help="ms per continuation move")
    ap.add_argument("--cap", type=int, default=240)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    from multiprocessing import Pool
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    n = len(tr.tok)
    gor = tr.game_of_row()
    # sample parents that are not terminal and have a next row in the same game
    rng = np.random.default_rng(args.seed)
    cand = np.flatnonzero((np.arange(n) + 1 < n)
                          & (gor[np.clip(np.arange(n) + 1, 0, n - 1)] == gor))
    rows = np.sort(cand[rng.choice(len(cand), min(args.n, len(cand)), replace=False)])

    shards = np.array_split(np.arange(len(rows)), args.workers)
    jobs = [(rows[s], tr.tok[rows[s]], tr.glob[rows[s]], tr.tok[rows[s] + 1],
             args.engine, args.movetime, args.cap, args.seed * 1000 + wi)
            for wi, s in enumerate(shards)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        parts = pool.map(_run_shard, jobs)
    res = [r for p in parts for r in p]
    el = time.time() - t0
    print(f"[cf] {len(res):,} counterfactual branches in {el/60:.1f}m "
          f"({len(res)/max(el,1e-9):.1f} branches/s, {args.workers} workers)")

    A = np.array(res, dtype=np.int64)
    r_i = A[:, 0]
    ctrl = np.where(tr.glob[r_i, 0].astype(bool), tr.result[gor[r_i]], -tr.result[gor[r_i]])
    treat = A[:, 3]
    delta = ctrl - treat
    print(f"[cf] control (played) outcome, mover POV: W {np.mean(ctrl==1):.2f} "
          f"D {np.mean(ctrl==0):.2f} L {np.mean(ctrl==-1):.2f}")
    print(f"[cf] treatment (random) outcome:          W {np.mean(treat==1):.2f} "
          f"D {np.mean(treat==0):.2f} L {np.mean(treat==-1):.2f}")
    print(f"[cf] played strictly better: {np.mean(delta>0):.2f}  equal: {np.mean(delta==0):.2f} "
          f" worse: {np.mean(delta<0):.2f}   mean delta {delta.mean():+.3f}")
    if args.save:
        p = paths.derived(f"counterfactual_{len(res)}_{n}.npz")
        np.savez_compressed(p, row=A[:, 0], rnd_from=A[:, 1], rnd_to=A[:, 2],
                            treat=A[:, 3], branch_plies=A[:, 4], ctrl=ctrl)
        print(f"[cf] wrote {p}")


if __name__ == "__main__":
    main()
