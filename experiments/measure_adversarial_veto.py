#!/usr/bin/env python
"""experiments/measure_adversarial_veto.py -- Kaveh 2026-07-22: cooperative vs adversarial
reachability ("what are all the positions that are reachable [cooperative] vs what human
games reach; positions never reached are either bad, or good positions the adversary
doesn't let us into"). The gap between the two reachabilities IS the opponent's veto, and
veto-lapses (blunders) are the sparse entries into denied winning regions. Formally the
HJ-reachability best-case/worst-case-disturbance distinction, with tablebase ground truth.

Measurement (toy, exact): from won anchors (White to move, DTM in a band):
  1. COOPERATIVE reach set at depth j: dedup'd ends of N random walks (both sides random).
  2. tb-label each target g: White-POV WDL.
  3. FORCEABILITY of g (exact, strict): does White have a strategy reaching EXACTLY g
     within j plies against tb-OPTIMAL Black? (Black deterministic -> DFS over White's
     choices only.)
  4. Report the veto table: of cooperative targets, %won; of won targets, %forceable vs
     %DENIED (= won but only reachable if Black cooperates/blunders); of non-won targets,
     %forceable (regions WE veto by never steering there).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed
from experiments.value_fixed_point import TB, tb_best_move


def wdl_white(b: chess.Board, tb) -> int | None:
    w, _ = tb.wdl_dtz(b)
    if w is None:
        return None
    return w if b.turn == chess.WHITE else -w


def forceable(board: chess.Board, hit, tb, plies_left: int) -> bool:
    """White-to-move-or-not DFS: White picks any legal move, Black replies tb-optimally
    (deterministic). True iff some White strategy reaches a position with hit(b) True
    within plies_left."""
    if hit(board):
        return True
    if plies_left == 0 or board.is_game_over(claim_draw=True):
        return False
    if board.turn == chess.WHITE:
        for m in board.legal_moves:
            c = board.copy(stack=False); c.push(m)
            if forceable(c, hit, tb, plies_left - 1):
                return True
        return False
    m = tb_best_move(board, tb)
    if m is None:
        return False
    c = board.copy(stack=False); c.push(m)
    return forceable(c, hit, tb, plies_left - 1)


def neighborhood_of(g: chess.Board):
    """Kaveh's region goal: 'maybe we'll reach a neighboring state with similar dtm' --
    same material signature + black king within 1 king-step + white king within 2."""
    mat = "".join(sorted(p.symbol() for p in g.piece_map().values()))
    gbk, gwk = g.king(chess.BLACK), g.king(chess.WHITE)

    def hit(b: chess.Board) -> bool:
        if "".join(sorted(p.symbol() for p in b.piece_map().values())) != mat:
            return False
        bk, wk = b.king(chess.BLACK), b.king(chess.WHITE)
        return chess.square_distance(bk, gbk) <= 1 and chess.square_distance(wk, gwk) <= 2
    return hit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--n-anchors", type=int, default=20)
    ap.add_argument("--j", type=int, default=4, help="reach horizon (plies)")
    ap.add_argument("--walks", type=int, default=300)
    ap.add_argument("--targets-per-anchor", type=int, default=60)
    ap.add_argument("--dtm-lo", type=int, default=10)
    ap.add_argument("--dtm-hi", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed); tb = TB("data/syzygy")

    dz = np.load(args.dtm_npz)
    P, M, dtm = np.asarray(dz["packed"]), np.asarray(dz["meta"]), np.asarray(dz["dtm"])
    cand = np.flatnonzero((dtm >= args.dtm_lo) & (dtm <= args.dtm_hi))
    rng.shuffle(cand)

    tot = dict(targets=0, won=0, drawn=0, lost=0, won_forceable=0, won_denied=0,
               won_nbhd_forceable=0, nonwon_forceable=0, nonwon=0)
    n_anchor = 0
    for ci in cand:
        if n_anchor >= args.n_anchors:
            break
        anchor = board_from_packed(P[ci], M[ci])
        if anchor.turn != chess.WHITE or anchor.is_game_over():
            continue
        # 1. cooperative reach set (random walks)
        seen = {}
        for _ in range(args.walks):
            b = anchor.copy(stack=False)
            deadend = False
            for _t in range(args.j):
                moves = list(b.legal_moves)
                if not moves:
                    deadend = True; break
                b.push(moves[int(rng.integers(len(moves)))])
            if deadend or b.is_game_over(claim_draw=True):
                continue
            seen.setdefault(b._transposition_key(), b.copy(stack=False))
        targets = list(seen.values())
        rng.shuffle(targets)
        targets = targets[: args.targets_per_anchor]
        if len(targets) < 10:
            continue
        n_anchor += 1
        # 2-3. label + forceability
        for g in targets:
            w = wdl_white(g, tb)
            if w is None:
                continue
            tot["targets"] += 1
            gk = g._transposition_key()
            forc = forceable(anchor, lambda b: b._transposition_key() == gk, tb, args.j)
            if w == 2:
                tot["won"] += 1
                tot["won_forceable" if forc else "won_denied"] += 1
                if forc or forceable(anchor, neighborhood_of(g), tb, args.j):
                    tot["won_nbhd_forceable"] += 1
            else:
                tot["drawn" if w in (-1, 0, 1) else "lost"] += 1
                tot["nonwon"] += 1
                if forc:
                    tot["nonwon_forceable"] += 1
        print(f"  anchor {n_anchor}/{args.n_anchors} (dtm {dtm[ci]}): {len(targets)} targets  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    tb.close()
    T = max(tot["targets"], 1); W = max(tot["won"], 1); NW = max(tot["nonwon"], 1)
    print(f"VERDICT ADVERSARIAL_VETO j={args.j} anchors={n_anchor} targets={tot['targets']}  "
          f"coop-reachable: won {tot['won']/T:.0%} drawn {tot['drawn']/T:.0%} lost {tot['lost']/T:.0%}  |  "
          f"of WON targets: EXACT-forceable {tot['won_forceable']/W:.0%} DENIED {tot['won_denied']/W:.0%} "
          f"NEIGHBORHOOD-forceable {tot['won_nbhd_forceable']/W:.0%}  |  "
          f"of NON-won: forceable {tot['nonwon_forceable']/NW:.0%}  [{time.time()-t0:.0f}s]", flush=True)
    print("  reading: DENIED = winning positions reachable only if the adversary cooperates/blunders "
          "-- the region plans cannot target directly (needs forcing or a veto lapse).", flush=True)


if __name__ == "__main__":
    main()
