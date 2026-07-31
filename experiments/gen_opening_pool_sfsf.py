#!/usr/bin/env python
"""experiments/gen_opening_pool_sfsf.py -- full-strength, DETERMINISTIC Stockfish-vs-Stockfish
continuations from the human opening pool (Kaveh 2026-07-31: "take only the first few opening
plies from the human database as starting positions, then run SF-vs-SF continuations on top of
those -- full strength, deterministic; the randomness comes from the starting position, not the
engine"). One game per pool position (SF is deterministic at fixed depth, so replaying the same
start would just repeat the same line -- diversity lives entirely in the 100k distinct openings).

Same STANDARD schema as gen_field_data_fullgame.py (planes/move/result/ending/dtz/game/ply/
stm_id/stm_elo/opp_elo) so this drops straight into the existing IQE/field training pipeline
(train_iqe_head.py) unchanged. Real move-history planes: each worker replays the pool's actual
opening UCI prefix from the start position (not just the bare FEN) so lc0's 8-position-history
input is correct, then continues with SF-vs-SF to game end (or <=7p tablebase handoff).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DRAW_REP = 4
ENGINE_ELO = 3500
STM_ID = np.uint64(0xE1E1E1E1E1E1E1E1)   # constant marker: "the engine", not a real player


def worker(task):
    lines, depth, cap, stride, per_game, tail, syzygy, seed = task
    import shutil
    import chess
    import chess.engine
    import torch
    from lczerolens import LczeroBoard
    from catspace.tb import TB
    from experiments.value_fixed_point import white_pov_value

    sf = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")
    sf.configure({"Threads": 1})
    tb = TB(str(syzygy), cache_db=None); syz = tb.tb
    limit = chess.engine.Limit(depth=depth)

    t0 = time.time()
    planes, moves, dtzs, ends, res_out, gouts, plies, sids, selos, oelos = ([] for _ in range(10))
    for gid, (fen, prefix) in enumerate(lines):
        if gid and gid % 200 == 0:
            print(f"    [w seed={seed}] {gid}/{len(lines)} games, {len(planes)} rows "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        board = LczeroBoard()
        ok = True
        for u in prefix.split(" "):
            try:
                board.push_uci(u)
            except Exception:
                ok = False; break
        if not ok:
            continue
        ucis: list[str] = []
        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < cap:
            if chess.popcount(board.occupied) <= 7:
                break                                            # hand off to exact tablebase completion
            try:
                r = sf.play(board, limit)
            except Exception:
                break
            if r.move is None:
                break
            board.push(r.move); ucis.append(r.move.uci())
        if chess.popcount(board.occupied) <= 7:
            from experiments.value_fixed_point import tb_best_move
            for _ in range(cap - len(board.move_stack)):
                if board.is_game_over(claim_draw=True) or chess.popcount(board.occupied) > 7:
                    break
                mv = tb_best_move(board, tb)
                if mv is None:
                    break
                board.push(mv); ucis.append(mv.uci())
        result = {"1-0": 1, "0-1": -1}.get(board.result(claim_draw=True), 0)
        base_end = {1: 0, -1: 5}.get(result, None)
        n_open = len(prefix.split(" "))
        n = n_open + len(ucis); tail_start = n - tail

        rb = LczeroBoard(); taken = 0
        for u in prefix.split(" "):
            rb.push_uci(u)
        for ply, u in enumerate(ucis, start=n_open):
            rb.push_uci(u)
            on_stride = (ply - n_open) % stride == 0 and taken < per_game
            is_tail = ply >= tail_start
            if not (on_stride or is_tail):
                continue
            ending = base_end if base_end is not None else DRAW_REP
            dtz = -1
            if chess.popcount(rb.occupied) <= 7 and not rb.is_game_over():
                try:
                    v = white_pov_value(rb, tb)
                    if v == 1.0:
                        ending = 0
                        dd = abs(syz.probe_dtz(rb)); dtz = dd if dd <= (100 - rb.halfmove_clock) else -1
                    elif v == 0.0:
                        ending = 5
                    else:
                        ending = DRAW_REP
                except Exception:
                    pass
            stm_white = (ply % 2 == 1)
            planes.append(rb.to_input_tensor().to(dtype=torch.uint8).numpy())
            moves.append(ucis[ply - n_open + 1] if ply - n_open + 1 < len(ucis) else "")
            dtzs.append(dtz); ends.append(ending); res_out.append(result)
            gouts.append(seed * 1_000_000 + gid); plies.append(ply)
            sids.append(STM_ID); selos.append(ENGINE_ELO if stm_white else ENGINE_ELO)
            oelos.append(ENGINE_ELO); taken += 1
    sf.quit(); tb.close()
    if not planes:
        z = np.zeros
        return (z((0, 112, 8, 8), np.uint8), np.array([], "U6"), z(0, np.int32), z(0, np.int8),
                z(0, np.int8), z(0, np.int64), z(0, np.int32), z(0, np.uint64), z(0, np.int16), z(0, np.int16))
    return (np.stack(planes), np.asarray(moves, "U6"), np.asarray(dtzs, np.int32), np.asarray(ends, np.int8),
            np.asarray(res_out, np.int8), np.asarray(gouts, np.int64), np.asarray(plies, np.int32),
            np.asarray(sids, np.uint64), np.asarray(selos, np.int16), np.asarray(oelos, np.int16))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="data/derived/opening_pool_ply8.txt")
    ap.add_argument("--limit", type=int, default=0, help="0 = all pool positions")
    ap.add_argument("--depth", type=int, default=12, help="SF search depth (matches gen_engine_games.py's M0 setting)")
    ap.add_argument("--cap", type=int, default=300, help="max total plies per game")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--per-game", type=int, default=8)
    ap.add_argument("--tail", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="data/derived/opening_pool_sfsf.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from catspace.tb import DEFAULT_SYZYGY
    t0 = time.time()

    lines = []
    with open(args.pool) as fh:
        for ln in fh:
            fen, cnt, prefix = ln.rstrip("\n").split("\t")
            lines.append((fen, prefix))
    if args.limit:
        lines = lines[:args.limit]

    W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    chunks = [lines[i::W] for i in range(W)]
    tasks = [(chunks[i], args.depth, args.cap, args.stride, args.per_game, args.tail,
              str(DEFAULT_SYZYGY), args.seed + i) for i in range(W)]
    print(f"[gen-opening-pool-sfsf] {len(lines):,} positions | SF depth {args.depth} deterministic "
          f"| {W} workers | STANDARD format", flush=True)
    cols = [[] for _ in range(10)]
    with ProcessPoolExecutor(max_workers=W) as ex:
        for i, r in enumerate(ex.map(worker, tasks)):
            for k in range(10):
                cols[k].append(r[k])
            print(f"  worker {i+1}/{W}: {len(r[2]):,} positions [{time.time()-t0:.0f}s]", flush=True)
    planes, moves, dtz, end, res, gid, ply, sid, selo, oelo = [np.concatenate(c) for c in cols]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, planes=planes, move=moves, result=res, ending=end, dtz=dtz,
                        game=gid, ply=ply, stm_id=sid, stm_elo=selo, opp_elo=oelo)
    w = int((end == 0).sum()); l = int((end == 5).sum()); dr = len(end) - w - l
    print(f"\n=== {args.out}: {len(dtz):,} positions ({planes.nbytes/1e6:.0f}MB) games "
          f"{len(np.unique(gid)):,} | ending W {w/len(end):.0%} D {dr/len(end):.0%} L {l/len(end):.0%} "
          f"[{time.time()-t0:.0f}s] ===")
    print("DONE gen_opening_pool_sfsf", flush=True)


if __name__ == "__main__":
    main()
