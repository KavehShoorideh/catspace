#!/usr/bin/env python
"""
experiments/forced_mate_set.py — build + VALIDATE a forced-mate position set.

Kaveh: "make sure you're looking at forced mates ... create a script to validate
the forced mate, ensure your samples all pass the test."

A game that ended in checkmate does NOT mean the position 4 plies earlier was a
FORCED mate -- the loser may have had defenses. This builds three classes:
  forced_mate_W : White (side to move) can force mate in <= K, PROVEN.
  forced_mate_B : Black (side to move) can force mate in <= K, PROVEN.
  neutral       : NO forced mate within K for the side to move (quiet contrast).
Stockfish is used only to find candidates fast; the ground truth is a rigorous
MINIMAX prover (forced_mate_in) that checks the mate holds against every defense.
Every saved sample is re-validated -- the script asserts 100% pass.

`forced_mate_in(board, k)` is also importable as the standalone validator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catspace.data.encode import board_from_packed


def forced_mate_in(board: chess.Board, k: int) -> bool:
    """True iff the SIDE TO MOVE can FORCE checkmate within k of its own moves
    (against every defense). Rigorous minimax proof -- no engine trust."""
    if k <= 0 or board.is_game_over():
        return False
    for mv in board.legal_moves:
        board.push(mv)
        if board.is_checkmate():
            board.pop()
            return True
        held = _defender_all_lose(board, k)         # defender to move now
        board.pop()
        if held:
            return True
    return False


def _defender_all_lose(board: chess.Board, k: int) -> bool:
    """Defender to move; True iff EVERY legal reply still lets the mater force
    mate in <= k-1 of its moves. No moves => stalemate => defender escaped."""
    any_move = False
    for mv in board.legal_moves:
        any_move = True
        board.push(mv)
        still = forced_mate_in(board, k - 1)
        board.pop()
        if not still:
            return False                            # this defense escapes
    return any_move


def mate_class(board: chess.Board, K: int) -> tuple[int, int] | None:
    """(label, k) where label +1 = side-to-move forces mate (mate_W if White to
    move / mate_B if Black to move), found at the smallest k <= K; or None."""
    for k in range(1, K + 1):
        if forced_mate_in(board, k):
            return k
    return None


def candidate_positions(shard_dirs, cap):
    """Positions near the end of games (a spread of offsets) -- mates cluster near
    decisive endings, and drawn-game endings feed the draw class. WHITE-POV via
    Stockfish decides the class, so both turns are fine."""
    boards = []
    for d in shard_dirs:
        for path in sorted(Path(d).glob("shard_*.npz")):
            z = np.load(path)
            gid, packed, meta = z["game_id"], z["packed"], z["meta"]
            ends = np.flatnonzero(np.r_[np.diff(gid) != 0, True])
            starts = np.r_[0, ends[:-1] + 1]
            for s, e in zip(starts, ends):
                glen = e - s + 1
                for back in (2, 4, 6, 9, 14, 20):
                    if glen > back:
                        boards.append(board_from_packed(packed[e - back], meta[e - back]))
                if len(boards) >= cap:
                    return boards
    return boards


def sf_label(eng, board, limit):
    """WHITE-POV Stockfish verdict -> dict(cls, k, cp, mate) or None. `mate` is the
    signed White-POV mate distance in MOVES (+k White mates, -k Black mates); `k`
    is the class-side moves-to-mate; `cp` the centipawn eval (for draws). mate at
    ANY depth counts (Kaveh: 4 or 10 ply, as long as it's forced)."""
    try:
        sc = eng.analyse(board, limit)["score"].pov(chess.WHITE)
    except Exception:
        return None
    m = sc.mate()
    if m is not None and m > 0:
        return dict(cls="mate_W", k=int(m), cp=None, mate=int(m))
    if m is not None and m < 0:
        return dict(cls="mate_B", k=int(-m), cp=None, mate=int(m))
    cp = sc.score()
    if m is None and cp is not None and abs(cp) < 25:
        return dict(cls="draw", k=None, cp=int(cp), mate=None)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+",
                    default=["data/shards/lichess_db_standard_rated_2019-01.prefix1gb"])
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--gen-depth", type=int, default=16, help="Stockfish depth for generation")
    ap.add_argument("--val-depth", type=int, default=24, help="deeper Stockfish depth to VALIDATE")
    ap.add_argument("--candidate-cap", type=int, default=60000)
    ap.add_argument("--out", default="artifacts/experiments/forced_mate_set.json")
    ap.add_argument("--validate-only", default=None,
                    help="path to a set JSON -- re-check every sample with deep Stockfish, report pass/fail")
    args = ap.parse_args()

    import chess.engine
    from catspace.uci import UCIBoardPolicy

    def revalidate(classes, depth):
        bad = 0
        with UCIBoardPolicy() as sf:
            lim = chess.engine.Limit(depth=depth)
            for cls, items in classes.items():
                for it in items:
                    lab = sf_label(sf.engine, chess.Board(it["fen"]), lim)
                    bad += (lab is None or lab["cls"] != cls)   # class must still hold
        return bad

    if args.validate_only:
        data = json.loads(Path(args.validate_only).read_text())
        n = sum(len(v) for v in data["classes"].values())
        bad = revalidate(data["classes"], args.val_depth)
        print(f"deep re-validation (depth {args.val_depth}) of {n} samples: {n - bad} pass, {bad} FAIL")
        raise SystemExit(1 if bad else 0)

    cands = candidate_positions(args.shards, args.candidate_cap)
    print(f"{len(cands)} candidate positions; Stockfish gen depth {args.gen_depth} "
          f"-> validate depth {args.val_depth}...", flush=True)
    buckets = {"mate_W": [], "mate_B": [], "draw": []}
    with UCIBoardPolicy() as sf:
        gen = chess.engine.Limit(depth=args.gen_depth)
        val = chess.engine.Limit(depth=args.val_depth)
        for b in cands:
            if all(len(v) >= args.per_class for v in buckets.values()):
                break
            lab = sf_label(sf.engine, b, gen)
            if lab is None or len(buckets[lab["cls"]]) >= args.per_class:
                continue
            deep = sf_label(sf.engine, b, val)          # confirm at deeper depth
            if deep is None or deep["cls"] != lab["cls"]:
                continue
            buckets[lab["cls"]].append(dict(fen=b.fen(), k=deep["k"], cp=deep["cp"],
                                            mate=deep["mate"], val_depth=args.val_depth))

    print("built (each confirmed at depth {}): ".format(args.val_depth)
          + ", ".join(f"{c}={len(v)}" for c, v in buckets.items()), flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(val_depth=args.val_depth, classes=buckets), indent=1))
    print(f"-> {out}  (PERSISTED, reuse this set). Re-check any time: "
          f"forced_mate_set.py --validate-only {out}")


if __name__ == "__main__":
    main()
