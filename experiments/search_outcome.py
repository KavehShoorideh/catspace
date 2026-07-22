#!/usr/bin/env python
"""experiments/search_outcome.py -- OUTCOME VIA SEARCH (Kaveh 2026-07-20, M2 pivot). The
acceptance test showed a static L2 head on the frozen L1 is starved: outcome is a sparse,
search-defined (attractor-rank) quantity, not a smooth field. So GLOBAL outcome comes from
recursive minimax that bottoms out in the tablebase at the solved frontier; the learned head is
only a near-terminal leaf shortcut.

This VALIDATES that architecture on ground truth by artificially LOWERING the frontier: pretend
only <= `frontier` pieces are solved, run the minimax from 5-6p positions, and check it recovers
the TRUE (full <=6 tablebase) outcome -- as a function of search depth. Where the search reaches
the (lowered) frontier it is EXACT; where the depth cap is hit first it falls back to a cheap
material leaf (a stand-in for the near-terminal L2 head). The accuracy-vs-depth curve is the
"outcome via search recovers ground truth" result, and the frontier-reach fraction is how much we
must rely on a leaf estimate.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed
from experiments.value_fixed_point import TB, white_pov_value

PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def pcount(b):
    return len(b.piece_map())


def material_leaf(board):
    """Cheap depth-cap leaf (stand-in for the near-terminal L2 head): side-to-move-POV sign of the
    material balance, in {-1,0,+1}."""
    w = sum(PIECE_VAL.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.WHITE)
    b = sum(PIECE_VAL.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.BLACK)
    adv = (w - b) if board.turn == chess.WHITE else (b - w)
    return float(np.sign(adv))


def negamax_frontier(board, tb, frontier, depth, alpha, beta, stats):
    """Side-to-move-POV outcome in {-1,0,+1}; tablebase leaf at <= frontier pieces, material leaf
    at the depth cap. stats=[tb_leaves, heuristic_leaves]."""
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    if pcount(board) <= frontier:
        v = white_pov_value(board, tb)
        if v is not None:
            stats[0] += 1
            stm = v if board.turn == chess.WHITE else (1.0 - v)
            return 2.0 * stm - 1.0
    if depth == 0:
        return quiescence(board, tb, frontier, alpha, beta, stats)             # capture-extension to the frontier
    best = -2.0
    moves = sorted(board.legal_moves, key=lambda m: not board.is_capture(m))   # captures first -> reach frontier + prune sooner
    for m in moves:
        c = board.copy(stack=False); c.push(m)
        v = -negamax_frontier(c, tb, frontier, depth - 1, -beta, -alpha, stats)
        if v > best:
            best = v
        alpha = max(alpha, v)
        if alpha >= beta:
            break
    return best if best > -2.0 else 0.0


def quiescence(board, tb, frontier, alpha, beta, stats):
    """Capture-only extension at the depth cap: follow captures until the frontier (tb-exact) or a
    QUIET position (material leaf). Bounds cost (captures are few) while reaching forcing
    conversions. `stats`=[tb_leaves, quiet_leaves]."""
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    if pcount(board) <= frontier:
        v = white_pov_value(board, tb)
        if v is not None:
            stats[0] += 1
            stm = v if board.turn == chess.WHITE else (1.0 - v)
            return 2.0 * stm - 1.0
    caps = [m for m in board.legal_moves if board.is_capture(m)]
    if not caps:
        stats[1] += 1
        return material_leaf(board)                                            # quiet -> leaf estimate
    best = material_leaf(board)                                                # stand-pat (option not to capture)
    for m in caps:
        c = board.copy(stack=False); c.push(m)
        v = -quiescence(c, tb, frontier, -beta, -alpha, stats)
        if v > best:
            best = v
        alpha = max(alpha, v)
        if alpha >= beta:
            break
    return best


def _wpov_wdl(stm_val, turn):
    """stm-POV {-1,0,1} -> White-POV outcome class {+1 win, 0 draw, -1 loss}."""
    wp = stm_val if turn == chess.WHITE else -stm_val
    return 1 if wp > 0.5 else (-1 if wp < -0.5 else 0)


def eval_chunk(task):
    packed, meta, true_wdl, frontier, depths, syzygy_dir = task
    tb = TB(syzygy_dir)
    # per-depth: correct count, and frontier-reach fraction (tb leaves / all leaves)
    correct = {d: 0 for d in depths}
    reach = {d: [0, 0] for d in depths}
    n = 0
    for i in range(len(packed)):
        b = board_from_packed(packed[i], meta[i])
        if b.is_game_over():
            continue
        n += 1
        for d in depths:
            stats = [0, 0]
            v = negamax_frontier(b, tb, frontier, d, -1e9, 1e9, stats)
            if _wpov_wdl(v, b.turn) == int(true_wdl[i]):
                correct[d] += 1
            reach[d][0] += stats[0]; reach[d][1] += stats[1]
    tb.close()
    return n, correct, reach


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/derived/stratified_perfect.npz")
    ap.add_argument("--syzygy", default="data/syzygy")
    ap.add_argument("--frontier", type=int, default=4, help="pretend only <= this many pieces are solved")
    ap.add_argument("--depths", default="1,2,3,4,6,8")
    ap.add_argument("--n", type=int, default=400, help="positions per source stratum (5p, 6p)")
    ap.add_argument("--out", default="artifacts/experiments/search_outcome.png")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); Wk = max(1, args.workers)
    depths = [int(x) for x in args.depths.split(",")]

    nz = np.load(args.data, allow_pickle=True)
    P, M, WDL, PCNT = (np.asarray(nz["packed"]), np.asarray(nz["meta"]),
                       np.asarray(nz["wdl"]), np.asarray(nz["pcount"]).astype(int))
    rng = np.random.default_rng(args.seed)
    # source positions strictly ABOVE the lowered frontier (must search down to it)
    src = np.flatnonzero((PCNT > args.frontier) & (PCNT <= 6))
    sel = []
    for pc in sorted(set(PCNT[src].tolist())):
        idx = src[PCNT[src] == pc]
        sel.append(idx[rng.permutation(len(idx))[: args.n]])
    sel = np.concatenate(sel)
    print(f"[stage] outcome-via-search: frontier<= {args.frontier}p, {len(sel)} positions "
          f"(strata {sorted(set(PCNT[sel].tolist()))}), depths {depths}, {Wk} workers", flush=True)

    bnd = np.linspace(0, len(sel), Wk + 1, dtype=int)
    tasks = [(P[sel[bnd[i]:bnd[i+1]]], M[sel[bnd[i]:bnd[i+1]]], WDL[sel[bnd[i]:bnd[i+1]]],
              args.frontier, depths, args.syzygy) for i in range(Wk) if bnd[i+1] > bnd[i]]
    N = 0; corr = {d: 0 for d in depths}; rch = {d: [0, 0] for d in depths}
    with ProcessPoolExecutor(max_workers=Wk) as ex:
        for n, c, r in ex.map(eval_chunk, tasks):
            N += n
            for d in depths:
                corr[d] += c[d]; rch[d][0] += r[d][0]; rch[d][1] += r[d][1]
    acc = {d: corr[d] / N for d in depths}
    reachfrac = {d: rch[d][0] / max(rch[d][0] + rch[d][1], 1) for d in depths}
    # baseline: material leaf alone (depth 0, no search)
    base = acc[min(depths)] if min(depths) == 0 else None
    for d in depths:
        print(f"  depth {d}: WDL recovery {acc[d]:.3f}  frontier-reach {reachfrac[d]:.3f} "
              f"(rest = material leaf)", flush=True)
    _plot(args, depths, acc, reachfrac, N)
    best_d = max(depths)
    print(f"VERDICT SEARCH_OUTCOME frontier={args.frontier} n={N} "
          f"acc@d{min(depths)}={acc[min(depths)]:.3f} acc@d{best_d}={acc[best_d]:.3f} "
          f"reach@d{best_d}={reachfrac[best_d]:.3f} ({time.time()-t0:.0f}s)")


def _plot(args, depths, acc, reachfrac, N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0f1115")
    for a in (a1, a2):
        a.set_facecolor("#0f1115"); a.tick_params(colors="#9aa4b2")
        for s in a.spines.values():
            s.set_color("#2a2e37")
        a.title.set_color("#e6e6e6"); a.xaxis.label.set_color("#9aa4b2"); a.yaxis.label.set_color("#9aa4b2")
    a1.plot(depths, [acc[d] for d in depths], "-o", color="#33cc77", ms=6)
    a1.axhline(1.0, color="#4fa3ff", ls=":", alpha=0.5)
    a1.set_ylim(0, 1.03); a1.set_xlabel("search depth (plies)"); a1.set_ylabel("WDL recovery vs true tablebase")
    a1.set_title(f"Outcome via search recovers ground truth\n(frontier lowered to <= {args.frontier}p, n={N})")
    a2.plot(depths, [reachfrac[d] for d in depths], "-o", color="#4fa3ff", ms=6)
    a2.set_ylim(0, 1.03); a2.set_xlabel("search depth (plies)"); a2.set_ylabel("fraction of leaves that reach the frontier")
    a2.set_title("How much search reaches the solved frontier\n(rest falls back to the leaf estimate)")
    fig.suptitle("OUTCOME VIA SEARCH: recursive minimax to the solved frontier recovers exact outcome",
                 color="#e6e6e6", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=115, facecolor="#0f1115"); plt.close(fig)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
