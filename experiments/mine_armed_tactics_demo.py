"""M7 detection demo run (MILESTONES.md sec.M7, 2026-08-03 sequencing override):
apply catspace.armed.detect.find_armed_tactic_candidates to real mid-game
positions and print whatever it actually finds. Not a benchmark, not a store
-- just "does this surface real armed tactics on real data" per Kaveh's ask
("the goal is to find some tactics").

Positions: random mid-game plies (ply 16-50) from SF-vs-SF games in
data/derived/opening_pool_sfsf_moves.tsv, replayed with python-chess.
"""
from __future__ import annotations

import argparse
import random
import shutil
import time

import chess
import chess.engine

from catspace.armed.detect import find_armed_tactic_candidates


def sample_positions(moves_tsv: str, n: int, seed: int, ply_lo: int = 16, ply_hi: int = 50):
    rng = random.Random(seed)
    lines = []
    with open(moves_tsv) as f:
        for line in f:
            lines.append(line.rstrip("\n"))
    rng.shuffle(lines)
    out = []
    for line in lines:
        if len(out) >= n:
            break
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        gid, result, moves_str = parts[0], parts[1], parts[2]
        ucis = moves_str.split()
        if len(ucis) <= ply_lo:
            continue
        ply = rng.randint(ply_lo, min(ply_hi, len(ucis) - 1))
        board = chess.Board()
        for u in ucis[:ply]:
            board.push(chess.Move.from_uci(u))
        if board.is_game_over():
            continue
        out.append((gid, ply, board))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves-tsv", default="data/derived/opening_pool_sfsf_moves.tsv")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--k-candidates", type=int, default=3)
    ap.add_argument("--k-moves", type=int, default=4)
    ap.add_argument("--min-gain", type=float, default=0.15)
    ap.add_argument("--decay-tol", type=float, default=0.05)
    args = ap.parse_args()

    positions = sample_positions(args.moves_tsv, args.n, args.seed)
    print(f"sampled {len(positions)} mid-game positions")

    eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish"))
    try:
        eng.configure({"UCI_ShowWDL": True})
    except Exception:
        pass

    found = []
    t0 = time.time()
    try:
        for i, (gid, ply, board) in enumerate(positions):
            cands = find_armed_tactic_candidates(
                eng, board, k_candidates=args.k_candidates, k_moves=args.k_moves,
                depth=args.depth, min_gain=args.min_gain, decay_tol=args.decay_tol)
            for c in cands:
                found.append((gid, ply, c))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(positions)} positions, {len(found)} armed tactics so far, "
                      f"{time.time()-t0:.0f}s elapsed")
    finally:
        eng.quit()

    print(f"\nDONE: {len(found)} armed-tactic candidates found across {len(positions)} positions "
          f"({time.time()-t0:.0f}s)")
    print("=" * 80)
    for gid, ply, c in found:
        print(f"\ngame {gid} ply {ply}  fen={c.fen}")
        print(f"  candidate move: {c.move}  immediate_gain={c.immediate_gain:+.3f}")
        print(f"  committor trend (mover POV): {[round(x, 3) for x in c.trend]}")
        print(f"  payoff if unblocked: {c.payoff_if_unblocked:+.3f}  blocked by: {c.blocking_move}"
              f" (ply offset {c.blocking_ply})")
        if c.blocking is not None:
            print(f"  blocking condition: {c.blocking.describe()}")


if __name__ == "__main__":
    main()
