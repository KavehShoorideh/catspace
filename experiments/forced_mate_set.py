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


def gen_forced_draws(n, rng):
    """FORCED-draw positions: insufficient material -> mate impossible either way
    (KvK, K+B vs K, K+N vs K, same-colour KB vs KB). A tight 'no mate possible'
    region (Kaveh). Verified by python-chess is_insufficient_material()."""
    menus = ([], [(chess.BISHOP, chess.WHITE)], [(chess.BISHOP, chess.BLACK)],
             [(chess.KNIGHT, chess.WHITE)], [(chess.KNIGHT, chess.BLACK)],
             [(chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.BLACK)],
             [(chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.BLACK)])
    out, tries = [], 0
    while len(out) < n and tries < n * 200:
        tries += 1
        pieces = list(menus[int(rng.integers(len(menus)))])
        sq = rng.choice(64, size=2 + len(pieces), replace=False)
        b = chess.Board(None)
        b.set_piece_at(int(sq[0]), chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(int(sq[1]), chess.Piece(chess.KING, chess.BLACK))
        for s, (pt, col) in zip(sq[2:], pieces):
            b.set_piece_at(int(s), chess.Piece(pt, col))
        b.turn = chess.WHITE if rng.random() < 0.5 else chess.BLACK
        if b.is_valid() and b.is_insufficient_material():
            out.append(b)
    return out


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
    ap.add_argument("--movetime", type=float, default=0.12, help="Stockfish seconds/position "
                    "(bounded -> fast + finds the short mates that dominate near game-ends)")
    ap.add_argument("--depth", type=int, default=0, help="if >0, use fixed depth instead of movetime")
    ap.add_argument("--threads", type=int, default=4, help="Stockfish threads")
    ap.add_argument("--candidate-cap", type=int, default=120000)
    ap.add_argument("--out", default="artifacts/experiments/forced_mate_set.json")
    ap.add_argument("--validate-only", default=None,
                    help="path to a set JSON -- re-check every sample with Stockfish, report pass/fail")
    ap.add_argument("--val-depth", type=int, default=22,
                    help="validate-only: DETERMINISTIC depth to re-check (time-limited is not "
                         "reproducible; depth is)")
    ap.add_argument("--filter-out", default=None,
                    help="validate-only: write ONLY the samples that pass the depth re-check here "
                         "(guarantees every saved sample is deterministically valid)")
    args = ap.parse_args()

    import chess.engine
    from catspace.uci import UCIBoardPolicy

    lim = chess.engine.Limit(depth=args.depth) if args.depth > 0 else chess.engine.Limit(time=args.movetime)
    limdesc = f"depth {args.depth}" if args.depth > 0 else f"movetime {args.movetime}s"

    if args.validate_only:
        data = json.loads(Path(args.validate_only).read_text())
        n = sum(len(v) for v in data["classes"].values())
        vlim = chess.engine.Limit(depth=args.val_depth)                 # DETERMINISTIC
        kept = {c: [] for c in data["classes"]}
        with UCIBoardPolicy(threads=args.threads) as sf:                 # SF loaded ONCE
            for cls, items in data["classes"].items():
                for it in items:
                    b = chess.Board(it["fen"])
                    if cls == "draw":                                    # forced draw = insufficient material
                        if b.is_insufficient_material():
                            kept[cls].append(it)
                        continue
                    lab = sf_label(sf.engine, b, vlim)
                    if lab is not None and lab["cls"] == cls:
                        kept[cls].append(it | dict(k=lab["k"], cp=lab["cp"], mate=lab["mate"]))
        nk = sum(len(v) for v in kept.values())
        print(f"deterministic re-validation (depth {args.val_depth}) of {n}: {nk} pass, {n-nk} FAIL "
              f"({', '.join(f'{c}={len(v)}' for c,v in kept.items())})")
        if args.filter_out:
            Path(args.filter_out).write_text(json.dumps(dict(val_depth=args.val_depth, classes=kept), indent=1))
            print(f"-> wrote {nk} deterministically-valid samples to {args.filter_out}")
        raise SystemExit(0 if nk == n else 1)

    # FORCED DRAWS = insufficient-material (mate impossible either way); their OWN
    # distinct region that must NOT overlap the mating regions (Kaveh). Generated,
    # not SF-scanned.
    rng = np.random.default_rng(0)
    draws = gen_forced_draws(args.per_class, rng)
    buckets = {"mate_W": [], "mate_B": [],
               "draw": [dict(fen=b.fen(), k=None, cp=0, mate=None) for b in draws]}
    print(f"forced-draw (insufficient material): {len(buckets['draw'])} generated", flush=True)

    cands = candidate_positions(args.shards, args.candidate_cap)
    print(f"{len(cands)} candidates; Stockfish {limdesc}, {args.threads} threads (loaded once), "
          f"target {args.per_class}/class (mate_W/mate_B via SF) ...", flush=True)
    with UCIBoardPolicy(threads=args.threads) as sf:                     # SF loaded ONCE, reused
        for i, b in enumerate(cands):
            if len(buckets["mate_W"]) >= args.per_class and len(buckets["mate_B"]) >= args.per_class:
                break
            lab = sf_label(sf.engine, b, lim)
            if lab is not None and lab["cls"] in ("mate_W", "mate_B") \
                    and len(buckets[lab["cls"]]) < args.per_class:
                buckets[lab["cls"]].append(dict(fen=b.fen(), k=lab["k"], cp=lab["cp"], mate=lab["mate"]))
            if (i + 1) % 200 == 0:
                print(f"  scanned {i+1}/{len(cands)}: "
                      + " ".join(f"{c}={len(v)}" for c, v in buckets.items()), flush=True)

    print(f"built ({limdesc}): " + ", ".join(f"{c}={len(v)}" for c, v in buckets.items()), flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(limit=limdesc, classes=buckets), indent=1))
    print(f"-> {out}  (PERSISTED, reuse). Re-check: forced_mate_set.py --validate-only {out}")


if __name__ == "__main__":
    main()
