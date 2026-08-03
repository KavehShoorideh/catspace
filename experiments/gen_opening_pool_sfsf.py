#!/usr/bin/env python
"""experiments/gen_opening_pool_sfsf.py -- full-strength, DETERMINISTIC Stockfish-vs-Stockfish
continuations from the human opening pool (Kaveh 2026-07-31: "take only the first few opening
plies from the human database as starting positions, then run SF-vs-SF continuations on top of
those -- full strength, deterministic; the randomness comes from the starting position, not the
engine"). One game per pool position (SF is deterministic at fixed depth, so replaying the same
start would just repeat the same line -- diversity lives entirely in the 100k distinct openings).

Same STANDARD schema as gen_field_data_fullgame.py (planes/move/result/ending/dtz/game/ply/
stm_id/stm_elo/opp_elo) so this drops straight into the existing IQE/field training pipeline
(train_iqe_head.py) unchanged. Real move-history planes: each shard replays the pool's actual
opening UCI prefix from the start position (not just the bare FEN) so lc0's 8-position-history
input is correct, then continues with SF-vs-SF to game end (or <=7p tablebase handoff).

CHECKPOINTED (Kaveh 2026-07-31): the pool is split into small SHARDS; each shard is written
atomically (temp-then-rename, the m2b_cache.py pattern) to <out>_shards/; a shard whose output
already exists is SKIPPED on restart -- kill/crash loses at most one shard's worth of work, not
the whole run. `--collect` merges finished shards into the final combined npz + moves.tsv without
recomputing anything.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DRAW_REP = 4
ENGINE_ELO = 3500
STM_ID = np.uint64(0xE1E1E1E1E1E1E1E1)   # constant marker: "the engine", not a real player


def play_shard(lines, depth, cap, stride, per_game, tail, syzygy, gid0):
    """The actual generation logic for one shard (list of (fen, prefix)). Unchanged from the
    original single-process-per-worker version, just factored out of the old `worker`."""
    import shutil
    import chess
    import chess.engine
    import torch
    from lczerolens import LczeroBoard
    from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB
    from experiments.value_fixed_point import white_pov_value, tb_best_move

    sf = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")
    sf.configure({"Threads": 1})
    tb = TB(str(syzygy), cache_db=None); syz = tb.tb
    limit = chess.engine.Limit(depth=depth)

    planes, moves, dtzs, ends, res_out, gouts, plies, sids, selos, oelos = ([] for _ in range(10))
    full_gids, full_moves, full_res = [], [], []
    for i, (fen, prefix) in enumerate(lines):
        gid = gid0 + i
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
        full_gids.append(gid); full_moves.append(prefix + " " + " ".join(ucis)); full_res.append(result)

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
            gouts.append(gid); plies.append(ply)
            sids.append(STM_ID); selos.append(ENGINE_ELO); oelos.append(ENGINE_ELO)
            taken += 1
    sf.quit(); tb.close()
    z = np.zeros
    arrs = (dict(planes=np.stack(planes), move=np.asarray(moves, "U6"), dtz=np.asarray(dtzs, np.int32),
                ending=np.asarray(ends, np.int8), result=np.asarray(res_out, np.int8),
                game=np.asarray(gouts, np.int64), ply=np.asarray(plies, np.int32),
                stm_id=np.asarray(sids, np.uint64), stm_elo=np.asarray(selos, np.int16),
                opp_elo=np.asarray(oelos, np.int16))
            if planes else
            dict(planes=z((0, 112, 8, 8), np.uint8), move=np.array([], "U6"), dtz=z(0, np.int32),
                ending=z(0, np.int8), result=z(0, np.int8), game=z(0, np.int64), ply=z(0, np.int32),
                stm_id=z(0, np.uint64), stm_elo=z(0, np.int16), opp_elo=z(0, np.int16)))
    arrs["moves_gid"] = np.asarray(full_gids, np.int64)
    arrs["moves_uci"] = np.asarray(full_moves, dtype=object)
    arrs["moves_result"] = np.asarray(full_res, np.int8)
    return arrs


def run_shard(task):
    shard_idx, out_path, lines, depth, cap, stride, per_game, tail, syzygy, gid0 = task
    if out_path.exists():
        return shard_idx, "skipped", 0
    arrs = play_shard(lines, depth, cap, stride, per_game, tail, syzygy, gid0)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrs, allow_pickle=True)
    os.replace(tmp, out_path)
    return shard_idx, "done", len(arrs["ply"])


def bar(frac: float, width: int = 30) -> str:
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + f"] {frac:5.1%}"


def fmt_eta(seconds: float) -> str:
    if seconds != seconds or seconds < 0:      # nan guard
        return "?"
    m, s = divmod(int(seconds), 60); h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def collect(shard_dir: Path, n_shards: int, out: str):
    cols = {k: [] for k in ("planes", "move", "dtz", "ending", "result", "game", "ply",
                            "stm_id", "stm_elo", "opp_elo")}
    full_gids, full_moves, full_res = [], [], []
    missing = []
    for i in range(n_shards):
        p = shard_dir / f"shard_{i:05d}.npz"
        if not p.exists():
            missing.append(i); continue
        d = np.load(p, allow_pickle=True)
        for k in cols:
            cols[k].append(d[k])
        full_gids.append(d["moves_gid"]); full_moves.append(d["moves_uci"]); full_res.append(d["moves_result"])
    if missing:
        print(f"[collect] WARNING: {len(missing)}/{n_shards} shards missing (not yet run) -- "
              f"collecting the {n_shards - len(missing)} that exist", flush=True)
    merged = {k: np.concatenate(v) for k, v in cols.items()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **merged)
    moves_path = str(Path(out).with_suffix("")) + "_moves.tsv"
    with open(moves_path, "w") as fh:
        for gids, ucis, ress in zip(full_gids, full_moves, full_res):
            for g, u, r in zip(gids, ucis, ress):
                fh.write(f"{g}\t{r}\t{u}\n")
    end = merged["ending"]
    w = int((end == 0).sum()); l = int((end == 5).sum()); dr = len(end) - w - l
    print(f"\n=== {out}: {len(end):,} positions ({merged['planes'].nbytes/1e6:.0f}MB) games "
          f"{len(np.unique(merged['game'])):,} | ending W {w/len(end):.0%} D {dr/len(end):.0%} "
          f"L {l/len(end):.0%} === | full move lists -> {moves_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="data/derived/opening_pool_ply8.txt")
    ap.add_argument("--limit", type=int, default=0, help="0 = all pool positions")
    ap.add_argument("--depth", type=int, default=12, help="SF search depth (matches gen_engine_games.py's M0 setting)")
    ap.add_argument("--cap", type=int, default=300, help="max total plies per game")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--per-game", type=int, default=8)
    ap.add_argument("--tail", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6, help="default 6 = perf-core count on this Mac; "
                    "os.cpu_count()-1 (10) oversubscribed the 5 efficiency cores and was ~3x SLOWER")
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--out", default="data/derived/opening_pool_sfsf.npz")
    ap.add_argument("--collect-only", action="store_true", help="skip generation, just merge existing shards")
    args = ap.parse_args()
    from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import DEFAULT_SYZYGY

    lines = []
    with open(args.pool) as fh:
        for ln in fh:
            fen, cnt, prefix = ln.rstrip("\n").split("\t")
            lines.append((fen, prefix))
    if args.limit:
        lines = lines[:args.limit]

    shard_dir = Path(args.out).with_suffix("") .parent / (Path(args.out).stem + "_shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = [lines[i:i + args.shard_size] for i in range(0, len(lines), args.shard_size)]
    n_shards = len(shards)

    if not args.collect_only:
        tasks = [(i, shard_dir / f"shard_{i:05d}.npz", shards[i], args.depth, args.cap, args.stride,
                  args.per_game, args.tail, str(DEFAULT_SYZYGY), i * args.shard_size)
                 for i in range(n_shards)]
        already = sum(1 for t in tasks if t[1].exists())
        print(f"[gen-opening-pool-sfsf] {len(lines):,} positions | {n_shards} shards x {args.shard_size} "
              f"| SF depth {args.depth} deterministic | {args.workers} workers | "
              f"{already} shard(s) already done (resume)", flush=True)
        t0 = time.time(); done = already
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_shard, t) for t in tasks if not t[1].exists()]
            for fut in as_completed(futs):
                idx, status, n = fut.result()
                done += 1
                elapsed = time.time() - t0
                rate = (done - already) / elapsed if elapsed > 0 else 0
                eta = (n_shards - done) / rate if rate > 0 else float("nan")
                print(f"  {bar(done / n_shards)} shard {idx:05d} {status} ({n} rows) | "
                      f"{done}/{n_shards} shards | elapsed {fmt_eta(elapsed)} | ETA {fmt_eta(eta)}",
                      flush=True)
        print(f"\n[gen-opening-pool-sfsf] all {n_shards} shards on disk [{fmt_eta(time.time()-t0)}]",
              flush=True)

    collect(shard_dir, n_shards, args.out)
    print("DONE gen_opening_pool_sfsf", flush=True)


if __name__ == "__main__":
    main()
