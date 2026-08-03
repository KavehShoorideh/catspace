#!/usr/bin/env python
"""experiments/gen_stockfish_continuations.py -- BEST-PLAY continuations from sampled human positions
(Kaveh 2026-07-21). Human lichess games are AVERAGE play; to let the L2 (IQE+QRL) field see OPTIMAL
play in the regions humans actually reach, sample real human positions and roll a strong engine
(Stockfish by default; any UCI engine -- Leela -- via --engine) forward for a few plies of best play.

Output: engine continuations written in the SAME shard schema as the lichess shards (each continuation
is one "game": shared game_id, ply-ordered rows), so they drop straight into training via
`train_lichess_fb.py --selfplay-shards <this dir> --selfplay-frac F`. The QRL pairing then consumes
best-play 1-ply transitions (optimal local constraints) + best-play goal pairs -- optimal successor
geometry layered onto the human occupancy.

Engine strength is CPU-bound; parallelised one engine process per worker. Run AFTER the human field
finishes (don't stack it on GPU training). Checkpointed per shard so it resumes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from experiments.value_fixed_point import TB, tb_best_move

ENGINE_ELO = 3500                 # mark continuations as strong play (feeds the omega/elo channel)
DRAW_BAND_CP = 40                 # |eval| < this at the end -> drawn line


def _score_cp(info, board):
    """white-POV centipawns from an engine analysis dict (mate scored as a large cp)."""
    sc = info["score"].white()
    return float(sc.score(mate_score=100000)) if sc is not None else 0.0


def gen_chunk(task):
    name, rows, plies, movetime, depth, nodes, engine_path, seed, frontier, syzygy, cap = task
    rng = np.random.default_rng(seed)
    try:
        eng = chess.engine.SimpleEngine.popen_uci(engine_path)
        eng.configure({"Threads": 1})
    except Exception as ex:
        return dict(name=name, err=f"engine launch failed: {ex}", got=0)
    tb = TB(syzygy) if syzygy else None                         # exact conversion once <=frontier pieces
    limit = chess.engine.Limit(time=movetime) if movetime > 0 else \
        (chess.engine.Limit(nodes=nodes) if nodes > 0 else chess.engine.Limit(depth=depth))
    out = {k: [] for k in ["packed", "meta", "ply", "clock", "eval_cp", "result",
                           "white_elo", "black_elo", "game_id"]}
    gid = seed * 1_000_000                                       # globally-unique game ids per worker
    got = 0
    for (pk, mt) in rows:
        board = board_from_packed(pk, mt)
        if board.is_game_over(claim_draw=True):
            continue
        traj, evals = [board.copy()], []
        b = board.copy()
        try:
            evals.append(_score_cp(eng.analyse(b, limit), b))
            for _ in range(plies):
                if b.is_game_over(claim_draw=True):
                    break
                if tb is not None and len(b.piece_map()) <= frontier:
                    break                                       # hand off to exact tablebase completion
                info = eng.analyse(b, limit)
                mv = info.get("pv", [None])[0]
                if mv is None:
                    break
                b.push(mv); traj.append(b.copy()); evals.append(_score_cp(info, b))
            if tb is not None:                                  # COMPLETE the trajectory to mate (both sides tb-optimal)
                for _ in range(cap):
                    if b.is_game_over(claim_draw=True) or len(b.piece_map()) > frontier:
                        break
                    mv = tb_best_move(b, tb)
                    if mv is None:
                        break
                    b.push(mv); traj.append(b.copy()); evals.append(evals[-1] if evals else 0.0)
        except Exception:
            continue
        if len(traj) < 2:
            continue
        final = evals[-1]
        res = 1 if final > DRAW_BAND_CP else (-1 if final < -DRAW_BAND_CP else 0)
        for j, (bb, ev) in enumerate(zip(traj, evals)):
            out["packed"].append(encode_packed(bb)); out["meta"].append(encode_meta(bb))
            out["ply"].append(j); out["clock"].append(0.0); out["eval_cp"].append(float(ev))
            out["result"].append(res); out["white_elo"].append(ENGINE_ELO)
            out["black_elo"].append(ENGINE_ELO); out["game_id"].append(gid)
        gid += 1; got += 1
    eng.quit()
    if tb is not None:
        tb.close()
    return dict(name=name, got=got,
                packed=np.array(out["packed"], np.uint64) if out["packed"] else np.zeros((0, 12), np.uint64),
                meta=np.array(out["meta"], np.uint8) if out["meta"] else np.zeros((0, 8), np.uint8),
                ply=np.array(out["ply"], np.int32), clock=np.array(out["clock"], np.float32),
                eval_cp=np.array(out["eval_cp"], np.float32), result=np.array(out["result"], np.int8),
                white_elo=np.array(out["white_elo"], np.uint16), black_elo=np.array(out["black_elo"], np.uint16),
                game_id=np.array(out["game_id"], np.uint32))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-shard", required=True, help="lichess shard_*.npz to sample human positions from")
    ap.add_argument("--n", type=int, default=4000, help="human positions to seed continuations from")
    ap.add_argument("--plies", type=int, default=8, help="best-play plies per continuation")
    ap.add_argument("--movetime", type=float, default=0.05, help="engine seconds/move (0 -> use --nodes/--depth)")
    ap.add_argument("--nodes", type=int, default=0)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--min-ply", type=int, default=0, help="only seed from source positions with ply>=this")
    ap.add_argument("--max-ply", type=int, default=1000)
    ap.add_argument("--max-pieces", type=int, default=32,
                    help="only seed from positions with <=this many pieces (targets endgame-entry so best-play "
                         "continuations reach the tablebase frontier and complete to mate)")
    ap.add_argument("--engine", default="stockfish", help="UCI engine path (stockfish or e.g. lc0)")
    ap.add_argument("--syzygy", default="data/syzygy", help="tablebase dir; once <=frontier pieces, COMPLETE the line to mate exactly")
    ap.add_argument("--frontier", type=int, default=6, help="piece count at/below which the tablebase takes over")
    ap.add_argument("--complete-cap", type=int, default=120, help="max tablebase-completion plies")
    ap.add_argument("--out-dir", default="data/shards/stockfish_continuations")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    nz = np.load(args.src_shard)
    P, M, PLY = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["ply"]).astype(int)
    pool = np.flatnonzero((PLY >= args.min_ply) & (PLY <= args.max_ply))
    rng = np.random.default_rng(args.seed)
    pool = pool[rng.permutation(len(pool))]
    if args.max_pieces < 32:                                     # scan for low-piece endgame-entry seeds
        sel = []
        for i in pool:
            if len(board_from_packed(P[i], M[i]).piece_map()) <= args.max_pieces:
                sel.append(i)
            if len(sel) >= args.n:
                break
        idx = np.array(sel, dtype=int)
        print(f"[seed] found {len(idx)} seeds with <= {args.max_pieces} pieces "
              f"(from a {len(pool)}-position ply pool)", flush=True)
    else:
        idx = pool[:args.n]
    Wk = max(1, args.workers)
    chunks = np.array_split(idx, Wk)
    tasks = [(f"w{w}", [(P[i], M[i]) for i in chunks[w]], args.plies, args.movetime, args.depth,
              args.nodes, args.engine, args.seed + 1 + w, args.frontier, args.syzygy, args.complete_cap)
             for w in range(Wk) if len(chunks[w])]
    print(f"[stage] {len(idx)} seed positions, {len(tasks)} workers, {args.plies} plies/cont, "
          f"engine={args.engine} ({'t=%.2fs' % args.movetime if args.movetime>0 else ('nodes=%d'%args.nodes if args.nodes else 'depth=%d'%args.depth)})",
          flush=True)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["packed", "meta", "ply", "clock", "eval_cp", "result", "white_elo", "black_elo", "game_id"]
    done_conts = 0
    with ProcessPoolExecutor(max_workers=Wk) as ex:
        futs = {ex.submit(gen_chunk, t): t for t in tasks}
        for si, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r.get("err"):
                print(f"  [WARN] {r['name']}: {r['err']}", flush=True); continue
            if r["got"] == 0:
                continue
            shard = out_dir / f"shard_{si:05d}.npz"
            np.savez_compressed(shard, **{k: r[k] for k in keys})   # per-shard checkpoint (resumable)
            done_conts += r["got"]
            print(f"  {r['name']}: {r['got']} continuations ({len(r['packed'])} rows) -> {shard.name} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"VERDICT STOCKFISH_CONT continuations={done_conts} plies={args.plies} -> {out_dir} "
          f"({time.time()-t0:.0f}s). Mix with: train_lichess_fb.py --selfplay-shards {out_dir} --selfplay-frac 0.3")


if __name__ == "__main__":
    main()
