#!/usr/bin/env python
"""gen_piecedown_sfsf.py -- SF-vs-SF from PIECE-DOWN starts, played to the true end.

Kaveh 2026-08-07: 'create data by simulating sf vs sf from piece down positions on one side,
e.g. missing a rook or a bishop.' Purpose: the Option-B skeleton is walled by SF games only, and
ordinary SF-vs-SF dies to adjudication in balanced middlegames -- the SF x endgame region has
almost no walls, and arrived-WIN terminals barely exist. A material handicap makes near-optimal
play DECISIVE: SF converts the extra piece to mate, producing witnessed optimal paths all the way
into the TB region (and with SyzygyPath set, the <=5-piece play is tb-OPTIMAL -- the ceilings
there are exact minimax lengths).

Starts: human-game positions at ply 8-20, one random non-king non-pawn piece (Q/R/B/N) removed
from one side (side and piece uniform), position validity checked. Games play to the BOARD end
(mate / stalemate / insufficient / 75-move / fivefold), no adjudication, ply cap 300.

Output TSV: id \t result(1 white win/0 draw/-1 black win) \t start_fen \t uci moves...
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import chess
import chess.engine
import numpy as np

from catspace.io import paths


def _one_game(task):
    seed, fen, nodes, syzygy, max_plies = task
    import chess, chess.engine
    board = chess.Board(fen)
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    try:
        eng.configure({"SyzygyPath": syzygy, "Threads": 1, "Hash": 64})
        moves = []
        while not board.is_game_over(claim_draw=True) and len(moves) < max_plies:
            r = eng.play(board, chess.engine.Limit(nodes=nodes))
            if r.move is None:
                break
            moves.append(r.move.uci())
            board.push(r.move)
        out = board.outcome(claim_draw=True)
        res = 0
        if out is not None and out.winner is not None:
            res = 1 if out.winner == chess.WHITE else -1
        return (fen, res, moves, out is not None)
    finally:
        eng.quit()


def make_starts(n, seed, min_ply=8, max_ply=20):
    """Piece-down start FENs from human-game prefixes."""
    from catspace.research.components.encoder.approaches.reach_probability.src.trajectories import (
        load_human_games)
    rng = np.random.default_rng(seed)
    games = load_human_games(n * 3, seed, None, max_ply + 1)
    starts = []
    for gid, res, ucis, flagged, elos in games:
        if len(starts) >= n or len(ucis) < min_ply:
            continue
        b = chess.Board()
        stop = int(rng.integers(min_ply, min(max_ply, len(ucis)) + 1))
        okpush = True
        for u in ucis[:stop]:
            try:
                b.push_uci(u)
            except Exception:
                okpush = False; break
        if not okpush or b.is_check():
            continue
        side = bool(rng.integers(0, 2))
        cands = [sq for sq, pc in b.piece_map().items()
                 if pc.color == side and pc.piece_type in
                 (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)]
        if not cands:
            continue
        b.remove_piece_at(int(rng.choice(cands)))
        if not b.is_valid():
            continue
        starts.append(b.fen())
    return starts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--nodes", type=int, default=20000, help="SF nodes/move (near-optimal, fast)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=paths.derived("piecedown_sfsf_moves.tsv"))
    args = ap.parse_args()

    t0 = time.time()
    starts = make_starts(args.games, args.seed)
    print(f"[gen] {len(starts)} piece-down starts [{time.time()-t0:.0f}s]", flush=True)
    syz = str(paths.syzygy_dir())
    tasks = [(i, f, args.nodes, syz, args.max_plies) for i, f in enumerate(starts)]
    n_done = n_dec = 0
    with mp.get_context("spawn").Pool(args.workers) as pool, open(args.out, "w") as fh:
        for fen, res, moves, ended in pool.imap_unordered(_one_game, tasks, chunksize=4):
            fh.write(f"{n_done}\t{res}\t{fen}\t{' '.join(moves)}\n")
            n_done += 1; n_dec += res != 0
            if n_done % 100 == 0:
                fh.flush()
                print(f"[gen] {n_done}/{len(starts)} | decisive {n_dec/n_done:.0%} | "
                      f"{(time.time()-t0)/n_done:.1f}s/game", flush=True)
    print(f"[gen] DONE {n_done} games, {n_dec/max(n_done,1):.0%} decisive -> {args.out} "
          f"[{(time.time()-t0)/60:.0f}m]")


if __name__ == "__main__":
    main()
