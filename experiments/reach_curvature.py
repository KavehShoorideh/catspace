#!/usr/bin/env python
"""
experiments/reach_curvature.py — how much CURVATURE (usable gradient) has the
reachability field developed in the KRRvKBP region we care about?

The drill-down (JOURNAL 2026-07-13) found the field is FLAT there: across the
legal moves of a won KRRvKBP position the model's reach spans ~0.009 and its
move-ordering is ~uncorrelated with the tablebase truth, so search has nothing
to descend. As self-play carves that region, we want to WATCH the curvature
appear. This probe turns "curvature/sensitivity" into scalars, measured on a
FIXED set of positions so successive checkpoints are directly comparable:

  For each start position (and, optionally, positions sampled along the
  tablebase-optimal line from it) with White to move in a won position:
    - enumerate legal moves, score each child with the model's reach to MATE_W
    - get the tablebase truth per child: WDL and, for winning children, |DTZ|
      (smaller = closer to mate)

  metrics (averaged over positions):
    move_spread   std of reach across the legal children. Flat field -> ~0;
                  curvature -> grows. The raw "sensitivity" of the field.
    dtz_rho       Spearman rho between model reach(child) and -|DTZ|(child),
                  over WIN-preserving children only. This is curvature WHERE WE
                  WANT IT: does higher reach actually mean closer to mate?
                  ~0 = field is curved but wrong/random; +1 = perfectly aligned.
    best_rank     normalized rank (0=top) of the tablebase-fastest-mate move in
                  the model's reach order. 0 = model's #1 IS the best move.
    top1_win      fraction where the model's #1-reach move preserves the win
                  (wdl < 0 or mate). The bottom line the search actually uses.

Run on the incumbent for a baseline, then after each self-play round; the JSON
records stack into a sensitivity-vs-round trajectory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import chess.syzygy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catspace.io.paths import derived_dir


def _spearman(a, b):
    """Spearman rho without scipy (rank-transform then Pearson). NaN if <3 pts
    or a constant input (no ordering information)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _reach_children(pol, board):
    """[(move, reach)] over all legal children, model's reach to its goal."""
    moves = list(board.legal_moves)
    succ = [board.copy(stack=False) for _ in moves]
    for b, m in zip(succ, moves):
        b.push(m)
    reach = pol._reach_batch(succ)
    return list(zip(moves, (float(x) for x in reach)))


def _tb_truth(board, tb):
    """{move: (wdl, dtz)} from the mover's-opponent POV after each move."""
    out = {}
    for m in board.legal_moves:
        b2 = board.copy(stack=False); b2.push(m)
        if b2.is_checkmate():
            out[m] = (-2, 0); continue
        try:
            out[m] = (tb.probe_wdl(b2), tb.probe_dtz(b2))
        except (KeyError, chess.syzygy.MissingTableError):
            out[m] = (0, 0)
    return out


def curvature_at(pol, board, tb):
    """Per-position curvature metrics, or None if the position is not a clean
    White-to-move win with >=3 legal moves and tablebase coverage."""
    if board.turn != chess.WHITE or board.is_game_over():
        return None
    children = _reach_children(pol, board)
    if len(children) < 3:
        return None
    truth = _tb_truth(board, tb)
    reach = np.array([r for _, r in children])
    # curvature WHERE WE WANT IT: over win-preserving children, does reach track
    # -|DTZ| (fastest mate)? opponent losing after our move <=> wdl < 0.
    win_moves = [(m, r) for m, r in children if truth[m][0] < 0]
    if len(win_moves) < 3:
        dtz_rho = float("nan")
    else:
        wr = [r for _, r in win_moves]
        neg_dtz = [-abs(truth[m][1]) if not (truth[m][0] == -2 and truth[m][1] == 0)
                   else 0 for m, _ in win_moves]  # mate = 0 distance = best
        dtz_rho = _spearman(wr, neg_dtz)
    # best_rank: where the fastest-mate winning move sits in the reach order
    order = sorted(children, key=lambda t: -t[1])
    if win_moves:
        best_move = min(win_moves, key=lambda t: (
            0 if truth[t[0]][0] == -2 else abs(truth[t[0]][1])))[0]
        best_rank = next(i for i, (m, _) in enumerate(order) if m == best_move)
        best_rank_norm = best_rank / (len(order) - 1)
    else:
        best_rank_norm = float("nan")
    top1_win = 1.0 if truth[order[0][0]][0] < 0 else 0.0
    return dict(move_spread=float(reach.std()), dtz_rho=dtz_rho,
                best_rank=best_rank_norm, top1_win=top1_win, n_win=len(win_moves))


def optimal_line(board, tb, max_plies):
    """Positions along a tablebase-optimal line from `board` (both sides play a
    DTZ-optimal move), giving deeper KRRvKBP positions than just the starts."""
    out = [board.copy(stack=False)]
    b = board.copy(stack=False)
    for _ in range(max_plies):
        if b.is_game_over():
            break
        best = None; best_key = None
        for m in b.legal_moves:
            b2 = b.copy(stack=False); b2.push(m)
            if b2.is_checkmate():
                best = m; break
            try:
                wdl, dtz = tb.probe_wdl(b2), tb.probe_dtz(b2)
            except (KeyError, chess.syzygy.MissingTableError):
                continue
            key = (wdl, abs(dtz))  # prefer opp most-losing, fastest
            if best_key is None or key < best_key:
                best_key = key; best = m
        if best is None:
            break
        b.push(best)
        out.append(b.copy(stack=False))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="data/derived/lichess_fb_4gb_qm_plygap_only.pt")
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_fixed_set_n60.json")
    ap.add_argument("--max-nodes", type=int, default=200)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--along-line", type=int, default=6,
                    help="also sample this many plies along the tablebase-optimal line "
                         "from each start (deeper KRRvKBP positions); 0 = starts only")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--round", default=None, help="label stamped into the JSON record")
    ap.add_argument("--out", default="artifacts/experiments/reach_curvature.jsonl",
                    help="append one JSON record per run here (the trajectory)")
    args = ap.parse_args()

    import torch  # noqa: F401
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import FBSearchPolicy

    device = pick_device(args.device)
    fb, payload = load_ckpt(Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt", device)
    pol = FBSearchPolicy(fb, payload["zgoals"]["MATE_W"], max_nodes=args.max_nodes,
                         beam=args.beam, device=device)
    tb = chess.syzygy.open_tablebase(args.syzygy_dir)
    fens = json.loads(Path(args.fixed_set).read_text())["fens"]

    per_pos = []
    for fen in fens:
        start = chess.Board(fen)
        boards = [start] if args.along_line == 0 else optimal_line(start, tb, args.along_line)
        for b in boards:
            m = curvature_at(pol, b, tb)
            if m is not None:
                per_pos.append(m)
    tb.close()

    def agg(key):
        vals = [p[key] for p in per_pos if not np.isnan(p[key])]
        return float(np.mean(vals)) if vals else float("nan")

    rec = dict(round=args.round, ckpt=str(args.ckpt), n_positions=len(per_pos),
               move_spread=agg("move_spread"), dtz_rho=agg("dtz_rho"),
               best_rank=agg("best_rank"), top1_win=agg("top1_win"))
    print(f"=== reach curvature ({args.round or args.ckpt}) over {len(per_pos)} positions ===")
    print(f"  move_spread (field sensitivity)      : {rec['move_spread']:.4f}   (flat ~0.009)")
    print(f"  dtz_rho    (curvature where we want) : {rec['dtz_rho']:+.3f}   (0=random, +1=aligned)")
    print(f"  best_rank  (0=model's #1 is best)    : {rec['best_rank']:.3f}")
    print(f"  top1_win   (frac #1 preserves win)   : {rec['top1_win']:.3f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  -> appended to {out}")


if __name__ == "__main__":
    main()
