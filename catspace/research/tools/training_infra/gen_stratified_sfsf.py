#!/usr/bin/env python
"""gen_stratified_sfsf.py -- STRATIFIED SF-vs-SF corpus generation (Kaveh 2026-08-12: "devote
all resources to a good data corpus ... balanced set of differing advantage states ... all
sorts of different tempos ... any variable we want our geometry to learn properly").

The grid, every cell filled to the same quota of STARTS:

  imbalance (removal recipe, side uniform):
    even        nothing removed                      (~0)
    pawn        one pawn                             (~1)
    two_pawns   two pawns, distinct files, one side  (~2)
    exchange    a rook vs a minor (cross removal)    (~2)
    exch_pawn   rook + pawn vs a minor               (~3)
    minor       one minor                            (~3)
    minor_pawn  minor + pawn, same side              (~4)
    rook        one rook                             (~5)
    queen       the queen                            (~9)
  phase of the base position:
    opening     human-game prefix, stop ply 8-20
    middle      human-game prefix, stop ply 20-60
    endgame     SF-corpus game replayed to its first <=12-piece position
  tempo (THE TURN-FORK INJECTION -- the minimal pairs the corpus lacks):
    every start is emitted TWICE: natural side to move AND the null-move twin (turn flipped,
    ep cleared). A start is kept only if BOTH variants are legal, so tempo is balanced by
    construction and every fork pair is complete. Both games are played out fully by SF
    (Syzygy in the engine: TB-optimal in the <=5-piece region), so both members carry REAL
    outcomes and REAL plies -- game-grounded, never evaluation-labeled.

Output: 4-col TSV (id, result, start_fen, ucis) -- the piecedown loader format -- plus a meta
TSV (id, imbalance, phase, tempo, pair_id) for audits and stratified sampling.

    .venv/bin/python -m ...gen_stratified_sfsf --starts-per-cell 350 --workers 9
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import chess
import numpy as np

from catspace.io import paths
from catspace.research.tools.training_infra.gen_piecedown_sfsf import _one_game

IMBALANCES = ("even", "pawn", "two_pawns", "exchange", "exch_pawn",
              "minor", "minor_pawn", "rook", "queen")
PHASES = ("opening", "middle", "endgame")


def _remove(b, recipe, side, rng):
    """apply a removal recipe in place; False if the base cannot support it."""
    def picks(color, types):
        return [sq for sq, pc in b.piece_map().items()
                if pc.color == color and pc.piece_type in types]
    P, N, B_, R, Q = (chess.PAWN,), (chess.KNIGHT,), (chess.BISHOP,), (chess.ROOK,), (chess.QUEEN,)
    minors = (chess.KNIGHT, chess.BISHOP)
    if recipe == "even":
        return True
    if recipe == "pawn":
        c = picks(side, P)
        if not c: return False
        b.remove_piece_at(int(rng.choice(c))); return True
    if recipe == "two_pawns":
        c = picks(side, P)
        files = {}
        for sq in c: files.setdefault(chess.square_file(sq), []).append(sq)
        if len(files) < 2: return False
        f1, f2 = rng.choice(sorted(files), 2, replace=False)
        b.remove_piece_at(int(rng.choice(files[f1])))
        b.remove_piece_at(int(rng.choice(files[f2]))); return True
    if recipe == "exchange":
        r, m = picks(side, R), picks(not side, minors)
        if not r or not m: return False
        b.remove_piece_at(int(rng.choice(r))); b.remove_piece_at(int(rng.choice(m))); return True
    if recipe == "exch_pawn":
        r, p, m = picks(side, R), picks(side, P), picks(not side, minors)
        if not r or not p or not m: return False
        b.remove_piece_at(int(rng.choice(r))); b.remove_piece_at(int(rng.choice(p)))
        b.remove_piece_at(int(rng.choice(m))); return True
    if recipe == "minor":
        c = picks(side, minors)
        if not c: return False
        b.remove_piece_at(int(rng.choice(c))); return True
    if recipe == "minor_pawn":
        m, p = picks(side, minors), picks(side, P)
        if not m or not p: return False
        b.remove_piece_at(int(rng.choice(m))); b.remove_piece_at(int(rng.choice(p))); return True
    if recipe == "rook":
        c = picks(side, R)
        if not c: return False
        b.remove_piece_at(int(rng.choice(c))); return True
    if recipe == "queen":
        c = picks(side, Q)
        if not c: return False
        b.remove_piece_at(int(rng.choice(c))); return True
    raise ValueError(recipe)


def _bases(phase, n, seed):
    """base boards for a phase, replayed from real games (never synthesized placements)."""
    rng = np.random.default_rng(seed)
    out = []
    if phase in ("opening", "middle"):
        from catspace.research.components.encoder.approaches.reach_probability.src.trajectories import (
            load_human_games)
        lo, hi = (8, 20) if phase == "opening" else (20, 60)
        games = load_human_games(n * 4, seed, None, hi + 1)
        for _gid, _res, ucis, _fl, _el in games:
            if len(out) >= n or len(ucis) <= lo:
                continue
            b = chess.Board()
            stop = int(rng.integers(lo, min(hi, len(ucis)) + 1))
            try:
                for u in ucis[:stop]:
                    b.push_uci(u)
            except Exception:
                continue
            if not b.is_check() and not b.is_game_over(claim_draw=True):
                out.append(b)
    else:
        from catspace.research.components.encoder.approaches.reach_probability.src.trajectories import (
            load_sf_games)
        games = load_sf_games(n * 6, seed)
        for _gid, _res, ucis, _fl, _el in games:
            if len(out) >= n:
                break
            b = chess.Board()
            hit = None
            try:
                for u in ucis:
                    b.push_uci(u)
                    if len(b.piece_map()) <= 12 and not b.is_check() \
                            and not b.is_game_over(claim_draw=True):
                        hit = b.copy(); break
            except Exception:
                continue
            if hit is not None:
                out.append(hit)
    rng.shuffle(out)
    return out


def fork_pair(b):
    """-> (fen_natural, fen_flipped) or None if the null-move twin is illegal."""
    if not b.is_valid() or b.is_game_over(claim_draw=True):
        return None
    b2 = b.copy()
    b2.turn = not b2.turn
    b2.ep_square = None
    if not b2.is_valid() or b2.is_game_over(claim_draw=True):
        return None
    return b.fen(), b2.fen()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--starts-per-cell", type=int, default=350,
                    help="fork pairs per (imbalance x phase) cell; games = 2x this")
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=paths.derived("stratified_sfsf_moves.tsv"))
    ap.add_argument("--meta", default=paths.derived("stratified_sfsf_meta.tsv"))
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    plan = []                                   # (fen, imbalance, phase, tempo, pair_id)
    pair_id = 0
    for phase in PHASES:
        bases = _bases(phase, args.starts_per_cell * len(IMBALANCES) * 3, args.seed + hash(phase) % 1000)
        bi = 0
        for imb in IMBALANCES:
            got = 0
            tries = 0
            while got < args.starts_per_cell and bi < len(bases) and tries < len(bases) * 2:
                tries += 1
                b = bases[bi % len(bases)].copy(); bi += 1
                side = bool(rng.integers(0, 2))
                if not _remove(b, imb, side, rng) or not b.is_valid():
                    continue
                fp = fork_pair(b)
                if fp is None:
                    continue
                plan.append((fp[0], imb, phase, "nat", pair_id))
                plan.append((fp[1], imb, phase, "flip", pair_id))
                pair_id += 1; got += 1
            print(f"[strat] {phase:8s} {imb:10s}: {got} pairs", flush=True)
    print(f"[strat] PLAN: {len(plan):,} games ({pair_id:,} fork pairs) "
          f"across {len(IMBALANCES)}x{len(PHASES)} cells [{time.time()-t0:.0f}s]", flush=True)

    syz = str(paths.syzygy_dir())
    order = rng.permutation(len(plan))          # interleave cells so partial output is balanced
    tasks = [(i, plan[i][0], args.nodes, syz, args.max_plies) for i in order]
    meta = {plan[i][0]: plan[i][1:] for i in range(len(plan))}
    n_done = n_dec = 0
    with mp.get_context("spawn").Pool(args.workers) as pool, \
            open(args.out, "w") as fh, open(args.meta, "w") as mh:
        mh.write("id\timbalance\tphase\ttempo\tpair\n")
        for fen, res, moves, _ended in pool.imap_unordered(_one_game, tasks, chunksize=2):
            imb, phase, tempo, pid = meta[fen]
            fh.write(f"{n_done}\t{res}\t{fen}\t{' '.join(moves)}\n")
            mh.write(f"{n_done}\t{imb}\t{phase}\t{tempo}\t{pid}\n")
            n_done += 1; n_dec += res != 0
            if n_done % 200 == 0:
                fh.flush(); mh.flush()
                r = (time.time() - t0) / n_done
                print(f"[strat] {n_done}/{len(plan)} | decisive {n_dec/n_done:.0%} | "
                      f"{r:.1f}s/game | ETA {(len(plan)-n_done)*r/3600:.1f}h", flush=True)
    print(f"[strat] DONE {n_done:,} games ({n_dec/max(n_done,1):.0%} decisive) -> {args.out} "
          f"[{(time.time()-t0)/3600:.1f}h]")


if __name__ == "__main__":
    main()
