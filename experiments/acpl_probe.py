#!/usr/bin/env python
"""
experiments/acpl_probe.py — Average Centipawn Loss (ACPL) probe: a cheap,
general-purpose measure of how often a policy blunders material/tactics,
independent of any specific endgame scenario or opponent's own strength.
Samples held-out positions from real Lichess games, asks the policy for its
move, and scores the move against a STRONG, fixed-depth (deterministic,
no skill/elo limiting) Stockfish -- exactly the standard chess-analysis
ACPL/blunder-rate metric, applied to a policy instead of a human player.

2026-07-11 motivation (JOURNAL.md): the KRRvKBP diagnostic found FBSearchPolicy
hangs a whole rook for free in an out-of-distribution endgame. This probe
answers the broader question -- how often does that happen across NORMAL,
in-distribution positions too -- fast enough to iterate on training changes
without waiting on full games.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chess
import chess.engine
import numpy as np

from catspace.data.encode import board_from_packed
from catspace.data.shards import sample_shard_rows
from catspace.io.paths import derived_dir, newest_shard_dir

MATE_CP = 1000  # standard ACPL convention: cap mate scores near the top of the
                # normal cp range, NOT at an arbitrarily huge sentinel -- a rare
                # forced-mate-in-N detection would otherwise dominate the MEAN
                # with a near-lottery-sized outlier (2026-07-11 finding: n=100
                # runs looked fine at ACPL~300, n=400 runs jumped to ACPL~1000-
                # 1600 purely from a handful of ~100000-loss positions once
                # sample size made hitting one more likely -- not a real signal)


def sample_positions(shard_dir, n: int, seed: int) -> list:
    """Held-out (never-trained) boards, game-over/near-forced positions dropped."""
    picks = sample_shard_rows(shard_dir, n * 2, seed, holdout_only=True)
    boards = []
    by_shard: dict = {}
    for name, row in picks:
        by_shard.setdefault(name, []).append(row)
    for name, rows in by_shard.items():
        npz = np.load(Path(shard_dir) / name)
        packed, meta = npz["packed"], npz["meta"]
        for row in rows:
            board = board_from_packed(packed[row], meta[row])
            if not board.is_game_over() and len(list(board.legal_moves)) >= 2:
                boards.append(board)
            if len(boards) >= n:
                break
        if len(boards) >= n:
            break
    return boards[:n]


def cp_loss_for_move(engine: chess.engine.SimpleEngine, board: chess.Board, move: chess.Move,
                     limit: chess.engine.Limit) -> tuple[int, int, float]:
    """(cp_before, cp_after, cp_loss), all from the MOVER's own perspective.
    cp_loss is clipped at 0 -- engine-eval noise can make a move look
    slightly "better" than the pre-move analysis; that's not a real gain
    worth reporting as negative loss."""
    mover = board.turn
    info_before = engine.analyse(board, limit)
    cp_before = info_before["score"].pov(mover).score(mate_score=MATE_CP)
    after = board.copy(stack=False)
    after.push(move)
    info_after = engine.analyse(after, limit)
    cp_after = info_after["score"].pov(mover).score(mate_score=MATE_CP)
    return cp_before, cp_after, max(0.0, cp_before - cp_after)


def run_probe(policy, boards: list, engine: chess.engine.SimpleEngine, limit: chess.engine.Limit,
             rng: np.random.Generator, verbose: bool = True) -> dict:
    losses, rows = [], []
    for i, board in enumerate(boards):
        move = policy.move(board.copy(stack=False), rng)
        cp_before, cp_after, loss = cp_loss_for_move(engine, board, move, limit)
        losses.append(loss)
        rows.append(dict(fen=board.fen(), move=move.uci(), cp_before=cp_before,
                         cp_after=cp_after, cp_loss=loss))
        if verbose:
            print(f"  pos {i:03d}  {move.uci():6}  cp_before={cp_before:+6d}  "
                 f"cp_after={cp_after:+6d}  loss={loss:6.0f}", flush=True)
    losses_arr = np.array(losses)
    return dict(n=len(losses_arr), acpl=float(losses_arr.mean()),
               blunder_rate=float((losses_arr >= 300).mean()),
               mistake_rate=float((losses_arr >= 100).mean()),
               rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--max-nodes", type=int, default=200)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch  # noqa: F401  (fail early with a clear message if .[nn] absent)
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import FBSearchPolicy

    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    boards = sample_positions(shard_dir, args.n, args.seed)
    print(f"{len(boards)} held-out positions sampled from {shard_dir.name}")

    device = pick_device(args.device)
    fb, payload = load_ckpt(Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt", device)
    if "MATE_W" not in payload.get("zgoals", {}):
        raise SystemExit("checkpoint has no zgoals -- finish a train_lichess_fb.py run first")
    zw, zb = payload["zgoals"]["MATE_W"], payload["zgoals"]["MATE_B"]

    class ColorAwarePolicy:
        """FBSearchPolicy needs the z matching whichever side is to move;
        wrap one instance per color so the probe's single `policy.move()`
        call handles a mixed-color position sample transparently."""
        def __init__(self):
            self._w = FBSearchPolicy(fb, zw, max_nodes=args.max_nodes, beam=args.beam, device=device)
            self._b = FBSearchPolicy(fb, zb, max_nodes=args.max_nodes, beam=args.beam, device=device)

        def move(self, board, rng):
            return (self._w if board.turn == chess.WHITE else self._b).move(board, rng)

    policy = ColorAwarePolicy()
    rng = np.random.default_rng(args.seed)
    limit = chess.engine.Limit(depth=args.sf_depth)

    print(f"FBSearchPolicy(max_nodes={args.max_nodes}, beam={args.beam}) vs stockfish depth={args.sf_depth} "
         f"analysis, n={len(boards)}, ckpt={args.ckpt or 'default'}, device={device}")
    with chess.engine.SimpleEngine.popen_uci("stockfish") as engine:
        engine.configure({"Threads": 1})
        result = run_probe(policy, boards, engine, limit, rng, verbose=True)

    print(f"\nVERDICT n={result['n']}  ACPL={result['acpl']:.1f}  "
         f"blunder_rate(>=300cp)={result['blunder_rate']:.3f}  "
         f"mistake_rate(>=100cp)={result['mistake_rate']:.3f}")


if __name__ == "__main__":
    main()
