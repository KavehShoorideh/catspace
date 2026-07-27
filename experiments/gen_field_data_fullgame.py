#!/usr/bin/env python
"""experiments/gen_field_data_fullgame.py -- STAGE C: balanced identity-preserving GAME RECORDS
(parquet, build/balance_game_records.py) -> FULL-BOARD field training data (lc0 112-plane npz), the
substrate for the single-space committor/quasimetric field on REAL games (not just endgames).

Per sampled position it emits the SAME npz schema the field trainer consumes
(planes, dtz, ending, game, ply), so train_field_fullgame.py / train_lc0_field.py load it directly:
  * planes  : lc0 112-plane REAL-history tensor (uint8), rebuilt by replaying the game's UCI moves.
  * ending  : the position's OUTCOME class, WHITE-POV = the game result (Monte-Carlo sample under the
              HUMAN play measure -> the metastability committor). win->0 (WIN_MATE), draw->1..4
              (subtype from the final board), loss->5 (LOSS_MATE). GROUNDED: at <=7 pieces the label
              is OVERRIDDEN by the exact tablebase WDL (the committor boundary condition, ARCH 8).
  * dtz     : tablebase DTZ (plies) when <=7 pieces AND white-won (drives the mate readout + WDL
              hinge on GROUND TRUTH); -1 ("inf") otherwise. So endgame-specific losses train only
              where they're exact; multi-goal + committor train on the whole board.
  * game/ply: for the multi-goal quasimetric (same-line pairs d(phi(s_i),phi(s_j)) -> ply gap).

Parallelized with a ProcessPoolExecutor over record shards (Kaveh: parallelize to the frameworks'
ability). Sampling: every --stride plies from --skip-open onward, capped at --per-game positions.
"""
from __future__ import annotations

import argparse, glob, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DRAW_REP = 4          # generic draw class when the subtype isn't a terminal on the board


def _draw_subtype(board) -> int:
    import chess
    if board.is_stalemate():
        return 2
    if board.is_insufficient_material():
        return 3
    if board.can_claim_fifty_moves():
        return 1
    return DRAW_REP


def worker(task):
    shard_path, sample_rows, stride, skip_open, per_game, tail, syzygy = task
    import chess
    import pyarrow.parquet as pq
    import torch
    from lczerolens import LczeroBoard
    from catspace.tb import TB
    from experiments.value_fixed_point import white_pov_value

    tb = TB(str(syzygy), cache_db=None); syz = tb.tb
    tbl = pq.read_table(shard_path, columns=["game_id", "result", "moves"])
    gids = tbl.column("game_id").to_numpy(); results = tbl.column("result").to_numpy()
    moves_col = tbl.column("moves").to_pylist()
    sample_rows = set(sample_rows.tolist()) if sample_rows is not None else None

    planes, dtzs, ends, gouts, plies = [], [], [], [], []
    for r in range(len(gids)):
        if sample_rows is not None and r not in sample_rows:
            continue
        ucis = moves_col[r].split()
        if len(ucis) < skip_open + 2:
            continue
        result = int(results[r]); gid = int(gids[r])
        base_end = {1: 0, -1: 5}.get(result, None)
        n = len(ucis); tail_start = n - tail
        board = LczeroBoard()
        taken = 0
        for ply, u in enumerate(ucis):
            try:
                mv = chess.Move.from_uci(u); board.push(mv)
            except Exception:
                break
            on_stride = ply >= skip_open and (ply - skip_open) % stride == 0 and taken < per_game
            is_tail = ply >= tail_start                     # always capture the endgame tail (<=7-piece grounding)
            if not (on_stride or is_tail):
                continue
            npieces = chess.popcount(board.occupied)
            ending = base_end
            dtz = -1
            if base_end is None:                       # draw game -> subtype from final position later; use generic now
                ending = DRAW_REP
            if npieces <= 7 and not board.is_game_over():
                try:
                    v = white_pov_value(board, tb)      # exact WDL, white-POV (grounding)
                    if v == 1.0:
                        ending = 0
                        d = abs(syz.probe_dtz(board)); dtz = d if d <= (100 - board.halfmove_clock) else -1
                    elif v == 0.0:
                        ending = 5
                    else:
                        ending = DRAW_REP
                except Exception:
                    pass
            planes.append(board.to_input_tensor().to(dtype=torch.uint8).numpy())
            dtzs.append(dtz); ends.append(ending); gouts.append(gid); plies.append(ply)
            taken += 1
    tb.close()
    if not planes:
        z = np.zeros
        return (z((0, 112, 8, 8), np.uint8), z(0, np.int32), z(0, np.int8), z(0, np.int64), z(0, np.int32))
    return (np.stack(planes), np.asarray(dtzs, np.int32), np.asarray(ends, np.int8),
            np.asarray(gouts, np.int64), np.asarray(plies, np.int32))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="data/records/smoke_lichess_balanced")
    ap.add_argument("--out", default="data/derived/field_fullgame.npz")
    ap.add_argument("--games", type=int, default=0, help="0 = all games in the records")
    ap.add_argument("--stride", type=int, default=6, help="sample every Nth ply")
    ap.add_argument("--skip-open", type=int, default=10, help="drop the first N opening plies")
    ap.add_argument("--per-game", type=int, default=8, help="max positions per game")
    ap.add_argument("--tail", type=int, default=4, help="always capture the last N plies (endgame grounding)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from catspace.tb import DEFAULT_SYZYGY
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    files = sorted(glob.glob(str(Path(args.records) / "*.parquet")))
    if not files:
        sys.exit(f"no parquet under {args.records}")

    # optional per-shard subsampling of games
    tasks = []
    import pyarrow.parquet as pq
    if args.games:
        # distribute the game budget across shards
        per_shard = max(1, args.games // len(files))
        for f in files:
            n = pq.read_metadata(f).num_rows
            sel = rng.choice(n, size=min(per_shard, n), replace=False)
            tasks.append((f, sel, args.stride, args.skip_open, args.per_game, args.tail, str(DEFAULT_SYZYGY)))
    else:
        tasks = [(f, None, args.stride, args.skip_open, args.per_game, args.tail, str(DEFAULT_SYZYGY)) for f in files]

    print(f"[gen-field-fullgame] {len(files)} shard(s) x {W} workers | stride {args.stride} "
          f"per_game {args.per_game} | out {args.out}", flush=True)
    P, D, E, G, PL = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for i, r in enumerate(ex.map(worker, tasks)):
            P.append(r[0]); D.append(r[1]); E.append(r[2]); G.append(r[3]); PL.append(r[4])
            print(f"  shard {i+1}/{len(tasks)}: {len(r[1])} positions [{time.time()-t0:.0f}s]", flush=True)
    planes = np.concatenate(P); dtz = np.concatenate(D); end = np.concatenate(E)
    gid = np.concatenate(G); ply = np.concatenate(PL)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, planes=planes, dtz=dtz, ending=end, game=gid, ply=ply)
    # outcome distribution (committor target sanity -- inspect BEFORE training, per standards)
    w = int((end == 0).sum()); l = int((end == 5).sum()); d = len(end) - w - l
    print(f"\n=== {args.out}: {len(dtz)} positions ({planes.nbytes/1e6:.0f}MB) games {len(np.unique(gid))} "
          f"[{time.time()-t0:.0f}s] ===")
    print(f"  ENDING dist: win {w/len(end):.1%} draw {d/len(end):.1%} loss {l/len(end):.1%} | "
          f"tb-grounded dtz>=0: {int((dtz>=0).sum())}")
    print("DONE gen_field_data_fullgame", flush=True)


if __name__ == "__main__":
    main()
