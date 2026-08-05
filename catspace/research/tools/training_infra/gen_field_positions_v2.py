#!/usr/bin/env python
"""catspace/research/tools/training_infra/gen_field_positions_v2.py -- v2 position sampler for the
field datasets, shared by BOTH populations (lichess human, SF-vs-SF), replacing the two independent
samplers in gen_field_data_fullgame.py and gen_opening_pool_sfsf.py.

WHY v2 exists. Both v1 samplers select rows with

    on_stride = (ply - open) % stride == 0 and taken < per_game        # stride 6, per_game 8

which has two defects, both measured and both real (JOURNAL 2026-08-04):

 1. FIXED PHASE. Every game is sampled on the same residue class mod 6, so the ply axis is a COMB
    and no bin width avoids aliasing. Density-vs-ply plots had to be rewritten as conditionals
    P(x|ply) to dodge it.
 2. PLY-42 CEILING. `taken < per_game` stops stride sampling after 8 rows, i.e. at ply 8*6 = 42.
    Past ply ~54 the dataset is 100% TAIL rows (median 1-2 plies from the end), so a ply axis
    beyond that silently swaps population from mid-game to game-endings. A "humans fan out over
    time" result measured on that axis had to be retracted.

v2 draws the mid-game plies UNIFORMLY AT RANDOM from the WHOLE game instead. The phase is random
per game (so pooled coverage is smooth, not a comb) and there is no ceiling (so ply coverage
extends as far as games actually go). The per-game RNG is seeded from (seed, game_id), so the
sample is reproducible and independent of shard order or worker count.

WHY ONE SCRIPT FOR BOTH SOURCES. The hazard analysis this feeds compares a human-trained field
against an SF-trained field on the same states. Any difference in how the two datasets sample ply
would show up as a difference between the fields -- indistinguishable from the dynamics difference
being measured. Sharing the sampler makes that confound impossible by construction rather than by
inspection. The only asymmetry left is the input format, which is a reader concern:

  * human  : parquet game records (data/records/lichess_2019-01), columns game_id/result/moves/...
  * SF-vs-SF: the moves TSV written by gen_opening_pool_sfsf.py (game_id \\t result \\t move_list).
              Its move list is `prefix + SF continuation`, i.e. the FULL game from the initial
              position, so replay is identical to the human path.

ONE POPULATION NOTE, recorded rather than hidden: the SF games start from the top-100k most
FREQUENT human ply-8 prefixes, so below ply 8 the two populations differ in opening COMPOSITION
(head of the human distribution vs all of it), not in dynamics. v1 dodged this by not sampling the
prefix at all; v2 samples it, because a position at ply 3 whose continuation is SF play is a
legitimate sample of SF dynamics and the ply axes must otherwise be aligned. Downstream comparisons
that want like-for-like openings should restrict to ply >= 8; the column is there to do it with.

Schema is the STANDARD one (planes/move/result/ending/dtz/game/ply/stm_id/stm_elo/opp_elo), so this
drops into build_combined_field_data.py -> train_iqe_head.py unchanged.
"""
from __future__ import annotations

import argparse, glob, hashlib, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from catspace.io import paths

DRAW_REP = 4
ENGINE_ELO = -1                      # SF has no meaningful rating; -1 is build_combined's own
STM_ID_ENGINE = np.uint64(0xE1E1E1E1E1E1E1E1)   # marker used by gen_opening_pool_sfsf.py
NCOL = 10


def _pid(user: str) -> np.uint64:
    """Stable name-masked player id (group-by key for the z-encoder; the name never enters a model).
    Byte-identical to gen_field_data_fullgame.py's, so v1 and v2 player ids are comparable."""
    return np.uint64(int.from_bytes(hashlib.blake2b(user.encode("utf-8", "replace"),
                                                    digest_size=8).digest(), "big"))


def sample_plies(n: int, per_game: int, tail: int, rng) -> np.ndarray:
    """Which plies of an n-ply game to record: the last `tail` plies, plus `per_game` drawn
    UNIFORMLY WITHOUT REPLACEMENT from everything before them.

    The tail block is deterministic on purpose. It carries the terminal row (the only row with
    move == "", one per game, which build_combined_field_data.py asserts on) and the pre-terminal
    rows that actually anchor the win pole -- these are structural requirements of the radial
    anchor, not a sampling preference. Randomising them would leave games with no anchor at all.
    The mid-game block is what v1 got wrong and is what becomes uniform here.
    """
    n_tail = min(tail, n)
    tail_idx = np.arange(n - n_tail, n, dtype=np.int64)
    pool_hi = n - n_tail
    k = min(per_game, pool_hi)
    if k <= 0:
        return tail_idx
    mid = rng.choice(pool_hi, size=k, replace=False).astype(np.int64)
    return np.union1d(mid, tail_idx)                      # sorted + unique


def extract_game(board_cls, chess, torch, tb, syz, white_pov_value, gid, result, ucis,
                 per_game, tail, seed, ids, elos):
    """Replay one game and return the sampled rows in the STANDARD schema.

    `ids`/`elos` are (white, black) pairs already resolved by the caller, so this function is
    identical for both populations.
    """
    n = len(ucis)
    if n < 2:
        return None
    rng = np.random.default_rng([seed, int(gid) & 0x7FFFFFFF])
    want = set(sample_plies(n, per_game, tail, rng).tolist())
    base_end = {1: 0, -1: 5}.get(int(result), None)
    pid_w, pid_b = ids
    ew, eb = elos
    board = board_cls()
    rows = [[] for _ in range(NCOL)]
    for ply, u in enumerate(ucis):
        try:
            board.push(chess.Move.from_uci(u))
        except Exception:
            break                                          # malformed move list: keep what we have
        if ply not in want:
            continue
        ending = base_end if base_end is not None else DRAW_REP
        dtz = -1
        if chess.popcount(board.occupied) <= 7 and not board.is_game_over():
            try:
                v = white_pov_value(board, tb)
                if v == 1.0:
                    ending = 0
                    dd = abs(syz.probe_dtz(board))
                    dtz = dd if dd <= (100 - board.halfmove_clock) else -1
                elif v == 0.0:
                    ending = 5
                else:
                    ending = DRAW_REP
            except Exception:
                pass
        stm_white = (ply % 2 == 1)                          # White moves on ODD ply -- see
        rows[0].append(board.to_input_tensor().to(dtype=torch.uint8).numpy())  # basin_labels()
        rows[1].append(ucis[ply + 1] if ply + 1 < n else "")   # POLICY target; "" marks terminal
        rows[2].append(dtz); rows[3].append(ending); rows[4].append(int(result))
        rows[5].append(int(gid)); rows[6].append(ply)
        rows[7].append(pid_w if stm_white else pid_b)
        rows[8].append(ew if stm_white else eb)
        rows[9].append(eb if stm_white else ew)
    return rows if rows[2] else None


def _empty():
    z = np.zeros
    return (z((0, 112, 8, 8), np.uint8), np.array([], "U6"), z(0, np.int32), z(0, np.int8),
            z(0, np.int8), z(0, np.int64), z(0, np.int32), z(0, np.uint64), z(0, np.int16),
            z(0, np.int16))


def _pack(cols):
    return (np.stack(cols[0]), np.asarray(cols[1], "U6"), np.asarray(cols[2], np.int32),
            np.asarray(cols[3], np.int8), np.asarray(cols[4], np.int8),
            np.asarray(cols[5], np.int64), np.asarray(cols[6], np.int32),
            np.asarray(cols[7], np.uint64), np.asarray(cols[8], np.int16),
            np.asarray(cols[9], np.int16))


def worker(task):
    kind, payload, per_game, tail, seed, syzygy = task
    import chess
    import torch
    from lczerolens import LczeroBoard
    from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB
    from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import white_pov_value

    tb = TB(str(syzygy), cache_db=None); syz = tb.tb
    cols = [[] for _ in range(NCOL)]

    def add(gid, result, ucis, ids, elos):
        r = extract_game(LczeroBoard, chess, torch, tb, syz, white_pov_value, gid, result, ucis,
                         per_game, tail, seed, ids, elos)
        if r is not None:
            for k in range(NCOL):
                cols[k] += r[k]

    if kind == "parquet":
        import pyarrow.parquet as pq
        shard_path, sample_rows = payload
        d = pq.read_table(shard_path, columns=["game_id", "result", "moves", "white_id", "black_id",
                                               "white_elo", "black_elo"]).to_pydict()
        take = range(len(d["game_id"])) if sample_rows is None else sample_rows.tolist()
        for r in take:
            add(int(d["game_id"][r]), int(d["result"][r]), d["moves"][r].split(),
                (_pid(d["white_id"][r]), _pid(d["black_id"][r])),
                (int(d["white_elo"][r]), int(d["black_elo"][r])))
    else:                                                  # kind == "tsv": pre-read lines
        for line in payload:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            add(int(p[0]), int(p[1]), p[2].split(),
                (STM_ID_ENGINE, STM_ID_ENGINE), (ENGINE_ELO, ENGINE_ELO))
    tb.close()
    return _empty() if not cols[2] else _pack(cols)


def parquet_tasks(records, games, per_game, tail, seed, syzygy, rng):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(Path(records) / "*.parquet")))
    if not files:
        sys.exit(f"no parquet under {records}")
    tasks = []
    for f in files:
        sel = None
        if games:
            n = pq.read_metadata(f).num_rows
            per_shard = max(1, games // len(files))
            sel = np.sort(rng.choice(n, size=min(per_shard, n), replace=False))
        tasks.append(("parquet", (f, sel), per_game, tail, seed, syzygy))
    return tasks


def tsv_tasks(tsv, games, per_game, tail, seed, syzygy, rng, chunk=400):
    lines = [l for l in open(tsv) if l.strip()]
    if games and games < len(lines):
        keep = np.sort(rng.choice(len(lines), size=games, replace=False))
        lines = [lines[i] for i in keep]
    return [("tsv", lines[i:i + chunk], per_game, tail, seed, syzygy)
            for i in range(0, len(lines), chunk)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["human", "sf"], required=True)
    ap.add_argument("--records", default="data/records/lichess_2019-01",
                    help="human: directory of parquet game records")
    ap.add_argument("--tsv", default="data/derived/opening_pool_sfsf_moves.tsv",
                    help="sf: game_id/result/move-list TSV from gen_opening_pool_sfsf.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=0, help="0 = every game available")
    ap.add_argument("--per-game", type=int, default=3, help="mid-game plies drawn UNIFORMLY")
    ap.add_argument("--tail", type=int, default=2, help="final plies always recorded (anchors)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--syzygy", default="", help="tablebase dir (default: repo assets)")
    args = ap.parse_args()

    from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import DEFAULT_SYZYGY
    syzygy = args.syzygy or str(DEFAULT_SYZYGY)
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    tasks = (parquet_tasks(args.records, args.games, args.per_game, args.tail, args.seed, syzygy, rng)
             if args.source == "human" else
             tsv_tasks(args.tsv, args.games, args.per_game, args.tail, args.seed, syzygy, rng))

    print(f"[gen-v2:{args.source}] {len(tasks)} task(s) x {W} workers | UNIFORM per-game ply "
          f"sampling: {args.per_game} mid + {args.tail} tail | syzygy {syzygy}", flush=True)
    cols = [[] for _ in range(NCOL)]
    with ProcessPoolExecutor(max_workers=W) as ex:
        for i, r in enumerate(ex.map(worker, tasks)):
            for k in range(NCOL):
                cols[k].append(r[k])
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                nrow = sum(len(c) for c in cols[2])
                print(f"  task {i+1}/{len(tasks)}: {nrow:,} rows so far [{time.time()-t0:.0f}s]",
                      flush=True)
    planes, moves, dtz, end, res, gid, ply, sid, selo, oelo = [np.concatenate(c) for c in cols]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, planes=planes, move=moves, result=res, ending=end, dtz=dtz,
                        game=gid, ply=ply, stm_id=sid, stm_elo=selo, opp_elo=oelo)

    ngames = len(np.unique(gid))
    w = int((end == 0).sum()); l = int((end == 5).sum()); dr = len(end) - w - l
    print(f"\n=== {args.out}: {len(dtz):,} positions ({planes.nbytes/1e6:.0f}MB planes) "
          f"games {ngames:,} ({len(dtz)/max(ngames,1):.2f} rows/game) [{time.time()-t0:.0f}s] ===")
    print(f"  ENDING {w/len(end):.1%}W {dr/len(end):.1%}D {l/len(end):.1%}L | tb-grounded "
          f"{int((dtz>=0).sum()):,} | terminals {int((moves=='').sum()):,} "
          f"| ply span {int(ply.min())}-{int(ply.max())}")
    # The v1 defects are exactly what these two lines test, so they are printed, not assumed.
    ph = np.bincount(ply % 6, minlength=6) / len(ply)
    print(f"  PHASE ply%6 shares {' '.join(f'{p:.3f}' for p in ph)}  "
          f"(v1 comb was 1/0/0/0/0/0; uniform is ~0.167 each)")
    for lo, hi in [(0, 20), (20, 42), (42, 60), (60, 90), (90, 999)]:
        m = (ply >= lo) & (ply < hi)
        print(f"  ply {lo:>3d}-{hi if hi<999 else 'inf':>3}: {int(m.sum()):>8,} rows "
              f"({100*m.mean():>5.1f}%)")
    print("DONE gen_field_positions_v2", flush=True)


if __name__ == "__main__":
    main()
