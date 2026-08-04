#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/reachability_field/experiments/gen_field_data_fullgame.py -- STAGE C: balanced identity-preserving GAME RECORDS
(parquet, build/balance_game_records.py) -> a STANDARD, BROADLY-USABLE position dataset (lc0
112-plane npz). ONE dataset that serves EVERY downstream model (Kaveh: spend the energy once, make
the data broadly usable, all phases, standard like others generate):

  * planes   : lc0 112-plane REAL-history tensor (uint8), rebuilt by replaying the game's UCI moves.
  * move      : the PLAYED move (UCI) from this position -- the POLICY target (AZ/lc0-style).
  * result    : game result WHITE-POV (+1/0/-1) -- the VALUE target.
  * ending    : outcome class WHITE-POV (win 0 / draw 1..4 / loss 5); at <=7 pieces OVERRIDDEN by
                exact tablebase WDL (the committor boundary condition). Committor target.
  * dtz       : tablebase DTZ (plies) when <=7p & white-won, else -1 -- grounds the mate readout.
  * game/ply  : the quasimetric multi-goal key (same-game pairs d(phi(s_i),phi(s_j)) -> ply gap).
  * stm_id    : STABLE HASH of the side-to-move's username (name-MASKED player id for the z-encoder,
                group-by without leaking the name). stm_elo / opp_elo: ratings for conditioning.

ALL PHASES (default --skip-open 0: openings included -- a value field must evaluate every phase;
skipping the opening blinds it, the v3->v4 lesson). Parallelized (ProcessPoolExecutor).
"""
from __future__ import annotations

import argparse, glob, hashlib, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from catspace.io import paths


DRAW_REP = 4


def _pid(user: str) -> np.uint64:
    """Stable name-masked player id (group-by key for the z-encoder; the name never enters a model)."""
    return np.uint64(int.from_bytes(hashlib.blake2b(user.encode("utf-8", "replace"), digest_size=8).digest(), "big"))


def worker(task):
    shard_path, sample_rows, stride, skip_open, per_game, tail, syzygy = task
    import chess
    import pyarrow.parquet as pq
    import torch
    from lczerolens import LczeroBoard
    from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB
    from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import white_pov_value

    tb = TB(str(syzygy), cache_db=None); syz = tb.tb
    tbl = pq.read_table(shard_path, columns=["game_id", "result", "moves", "white_id", "black_id",
                                             "white_elo", "black_elo"])
    d = tbl.to_pydict()
    gids = d["game_id"]; results = d["result"]; moves_col = d["moves"]
    wid = d["white_id"]; bid = d["black_id"]; welo = d["white_elo"]; belo = d["black_elo"]
    sample_rows = set(sample_rows.tolist()) if sample_rows is not None else None

    planes, moves, dtzs, ends, res_out, gouts, plies, sids, selos, oelos = ([] for _ in range(10))
    for r in range(len(gids)):
        if sample_rows is not None and r not in sample_rows:
            continue
        ucis = moves_col[r].split()
        if len(ucis) < skip_open + 2:
            continue
        result = int(results[r]); gid = int(gids[r])
        base_end = {1: 0, -1: 5}.get(result, None)
        n = len(ucis); tail_start = n - tail
        pid_w, pid_b = _pid(wid[r]), _pid(bid[r]); ew, eb = int(welo[r]), int(belo[r])
        board = LczeroBoard(); taken = 0
        for ply, u in enumerate(ucis):
            try:
                board.push(chess.Move.from_uci(u))
            except Exception:
                break
            on_stride = ply >= skip_open and (ply - skip_open) % stride == 0 and taken < per_game
            is_tail = ply >= tail_start
            if not (on_stride or is_tail):
                continue
            ending = base_end if base_end is not None else DRAW_REP
            dtz = -1
            if chess.popcount(board.occupied) <= 7 and not board.is_game_over():
                try:
                    v = white_pov_value(board, tb)
                    if v == 1.0:
                        ending = 0
                        dd = abs(syz.probe_dtz(board)); dtz = dd if dd <= (100 - board.halfmove_clock) else -1
                    elif v == 0.0:
                        ending = 5
                    else:
                        ending = DRAW_REP
                except Exception:
                    pass
            stm_white = (ply % 2 == 1)                       # side to move at `board` (after `ply` half-moves)
            planes.append(board.to_input_tensor().to(dtype=torch.uint8).numpy())
            moves.append(ucis[ply + 1] if ply + 1 < n else "")   # the POLICY target (move played next)
            dtzs.append(dtz); ends.append(ending); res_out.append(result)
            gouts.append(gid); plies.append(ply)
            sids.append(pid_w if stm_white else pid_b)
            selos.append(ew if stm_white else eb); oelos.append(eb if stm_white else ew)
            taken += 1
    tb.close()
    if not planes:
        z = np.zeros
        return (z((0, 112, 8, 8), np.uint8), np.array([], "U6"), z(0, np.int32), z(0, np.int8),
                z(0, np.int8), z(0, np.int64), z(0, np.int32), z(0, np.uint64), z(0, np.int16), z(0, np.int16))
    return (np.stack(planes), np.asarray(moves, "U6"), np.asarray(dtzs, np.int32), np.asarray(ends, np.int8),
            np.asarray(res_out, np.int8), np.asarray(gouts, np.int64), np.asarray(plies, np.int32),
            np.asarray(sids, np.uint64), np.asarray(selos, np.int16), np.asarray(oelos, np.int16))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=paths.records("smoke_lichess_balanced"))
    ap.add_argument("--out", default=paths.derived("field_fullgame.npz"))
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--skip-open", type=int, default=0, help="0 = include openings (all phases; a value field needs them)")
    ap.add_argument("--per-game", type=int, default=8)
    ap.add_argument("--tail", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import DEFAULT_SYZYGY
    import pyarrow.parquet as pq
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    files = sorted(glob.glob(str(Path(args.records) / "*.parquet")))
    if not files:
        sys.exit(f"no parquet under {args.records}")
    if args.games:
        per_shard = max(1, args.games // len(files))
        tasks = []
        for f in files:
            n = pq.read_metadata(f).num_rows
            sel = rng.choice(n, size=min(per_shard, n), replace=False)
            tasks.append((f, sel, args.stride, args.skip_open, args.per_game, args.tail, str(DEFAULT_SYZYGY)))
    else:
        tasks = [(f, None, args.stride, args.skip_open, args.per_game, args.tail, str(DEFAULT_SYZYGY)) for f in files]

    print(f"[gen-field-fullgame] {len(files)} shard(s) x {W} workers | stride {args.stride} skip_open "
          f"{args.skip_open} (all-phase) | STANDARD format (planes/move/result/ending/dtz/game/ply/stm_id/elo)", flush=True)
    cols = [[] for _ in range(10)]
    with ProcessPoolExecutor(max_workers=W) as ex:
        for i, r in enumerate(ex.map(worker, tasks)):
            for k in range(10):
                cols[k].append(r[k])
            print(f"  shard {i+1}/{len(tasks)}: {len(r[2])} positions [{time.time()-t0:.0f}s]", flush=True)
    planes, moves, dtz, end, res, gid, ply, sid, selo, oelo = [np.concatenate(c) for c in cols]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, planes=planes, move=moves, result=res, ending=end, dtz=dtz,
                        game=gid, ply=ply, stm_id=sid, stm_elo=selo, opp_elo=oelo)
    w = int((end == 0).sum()); l = int((end == 5).sum()); dr = len(end) - w - l
    print(f"\n=== {args.out}: {len(dtz)} positions ({planes.nbytes/1e6:.0f}MB) games {len(np.unique(gid))} "
          f"| unique players {len(np.unique(sid))} [{time.time()-t0:.0f}s] ===")
    print(f"  ENDING {w/len(end):.1%}W {dr/len(end):.1%}D {l/len(end):.1%}L | tb-grounded {int((dtz>=0).sum())} "
          f"| with-policy-move {int((moves!='').sum())} | ply span {int(ply.min())}-{int(ply.max())}")
    print("DONE gen_field_data_fullgame", flush=True)


if __name__ == "__main__":
    main()
