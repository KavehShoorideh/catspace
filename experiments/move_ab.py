#!/usr/bin/env python
"""
experiments/move_ab.py — paired MOVE-LEVEL A/B between two checkpoints.

Game-conversion is too high-variance to rank variants (CI +-0.38 at n=200 games,
because most KRRvKBP positions draw-or-win for both sides so the per-game diff is
mostly 0). This measures play at the MOVE level instead: over the test positions
AND positions sampled along their tablebase-optimal lines (thousands of
White-to-move winning decisions), what fraction of each model's hop-search top
move PRESERVES the win (resulting position still winning per Syzygy)? Paired on
the SAME positions, with a bootstrap CI on the difference -> thousands of samples,
tight interval, real power to distinguish variants. This is top1_win made paired.

VERDICT line: preserve-win A vs B, diff, bootstrap CI, McNemar decisive split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import chess.syzygy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from experiments.reach_curvature import _reach_children, _tb_truth, optimal_line


def top_move_progress(pol, board, tb):
    """Is the model's #1-reach move a PROGRESS move -- winning AND DTZ-optimal
    (minimises |DTZ| among winning children, i.e. maintains fastest conversion)?
    'Preserve the win' is too easy (most moves do); THIS captures the shuffle/no-
    progress failure. Returns 1.0 progress, 0.0 stall/blunder, or None if the
    position has no winning move / no tablebase (skip)."""
    truth = _tb_truth(board, tb)                    # {move: (wdl, dtz)}
    win = [(m, abs(truth[m][1])) for m in truth if truth[m][0] < 0]
    if not win:
        return None
    best_dtz = min(d for _, d in win)
    top = _reach_children(pol, board)[0][0]
    return 1.0 if (truth[top][0] < 0 and abs(truth[top][1]) <= best_dtz) else 0.0


def collect_positions(fens, tb, along_line, cap):
    """White-to-move winning positions from the test fens + their optimal lines."""
    out = []
    for fen in fens:
        b0 = chess.Board(fen)
        boards = [b0] if along_line == 0 else optimal_line(b0, tb, along_line)
        for b in boards:
            if b.turn == chess.WHITE and not b.is_game_over():
                try:
                    if tb.probe_wdl(b) > 0:
                        out.append(b.fen())
                except (KeyError, chess.syzygy.MissingTableError, ValueError, IndexError):
                    pass
        if len(out) >= cap:
            break
    return out[:cap]


def preserve_vector(ckpt, positions, tb, nodes, beam, device):
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import FBSearchPolicy
    dev = pick_device(device)
    fb, pay = load_ckpt(Path(ckpt), dev)
    pol = FBSearchPolicy(fb, pay["zgoals"]["MATE_W"], max_nodes=nodes, beam=beam, device=dev)
    out, moves = [], []
    for f in positions:
        b = chess.Board(f)
        v = top_move_progress(pol, b, tb)
        out.append(0.0 if v is None else v)         # None depends on board only (same for A & B)
        moves.append(_reach_children(pol, b)[0][0].uci())
    return np.array(out), moves


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_test_n200.json")
    ap.add_argument("--along-line", type=int, default=8)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    args = ap.parse_args()

    import torch  # noqa: F401
    tb = chess.syzygy.open_tablebase(args.syzygy_dir)
    fens = json.loads(Path(args.fixed_set).read_text())["fens"]
    positions = collect_positions(fens, tb, args.along_line, args.cap)
    a, moves_a = preserve_vector(args.ckpt_a, positions, tb, args.nodes, args.beam, args.device)
    b, moves_b = preserve_vector(args.ckpt_b, positions, tb, args.nodes, args.beam, args.device)
    tb.close()

    n = len(positions)
    agree = float(np.mean([ma == mb for ma, mb in zip(moves_a, moves_b)]))
    print(f"MOVE_AGREEMENT A vs B pick the SAME top move on {agree*100:.1f}% of {n} positions "
          f"(100% => the fine-tune didn't change play)")
    diff = float(b.mean() - a.mean())
    # paired bootstrap CI over positions (fixed rng: Math.random-free determinism)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(args.boot, n))
    boot = (b[idx].mean(1) - a[idx].mean(1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    b_better = int(((b == 1) & (a == 0)).sum()); a_better = int(((a == 1) & (b == 0)).sum())
    sig = (lo > 0 or hi < 0)
    print(f"MOVE_AB preserve-win A={a.mean():.3f} vs B={b.mean():.3f}  "
          f"diff={diff:+.3f} CI=[{lo:+.3f},{hi:+.3f}]  "
          f"(n={n} moves; B>A on {b_better}, A>B on {a_better}) "
          f"[{'SIGNIFICANT' if sig else 'ns'}]")


if __name__ == "__main__":
    main()
