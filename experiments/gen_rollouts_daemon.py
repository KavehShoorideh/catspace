#!/usr/bin/env python
"""experiments/gen_rollouts_daemon.py -- CONTINUOUS shared-anchor rollout generation across a
CAPABILITY LADDER (Kaveh 2026-07-23: "different stockfish strengths, leela, maia, all of them,
so we can show spread in capabilities" + "keep sampling, keep saving, don't let the CPU idle").

Every anchor (sampled from the human lichess shards) is rolled out under EVERY regime in the
ladder -- shared anchors make all capabilities directly comparable, and the shards double as
engine-cohort move-selection data for the opponent model.

REGIME LADDER (ids are the multichannel vocabulary; 0=human stream, 1=random walks, elsewhere):
  2  sf_full     Stockfish, full strength, self-play
  3  rand_vs_sf  anchor's mover drifts randomly, Stockfish resists (the drift channel)
  4  sf_1400     Stockfish UCI_LimitStrength 1400, self-play
  5  sf_1700     ... 1700
  6  sf_2000     ... 2000
  7  lc0         Leela with a real network (pending install/net; auto-skipped if absent)
  8  maia_1100   lc0 + maia-1100 weights, nodes=1 (the Maia human-move protocol)
  9  maia_1500   ...
  10 maia_1900   ...
Unavailable engines are skipped per-worker with a one-line note -- the daemon runs with
whatever exists. Controls: touch <out-dir>/STOP; guards: --max-gb / --min-free-gb.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REGIMES = {
    2:  dict(name="sf_full",   kind="sf",   opts={}),
    3:  dict(name="rand_vs_sf", kind="rand_vs_sf"),
    4:  dict(name="sf_1400",   kind="sf",   opts={"UCI_LimitStrength": True, "UCI_Elo": 1400}),
    5:  dict(name="sf_1700",   kind="sf",   opts={"UCI_LimitStrength": True, "UCI_Elo": 1700}),
    6:  dict(name="sf_2000",   kind="sf",   opts={"UCI_LimitStrength": True, "UCI_Elo": 2000}),
    7:  dict(name="lc0",       kind="lc0",  weights=None),
    8:  dict(name="maia_1100", kind="lc0",  weights="data/engines/maia/maia-1100.pb.gz", nodes=1),
    9:  dict(name="maia_1500", kind="lc0",  weights="data/engines/maia/maia-1500.pb.gz", nodes=1),
    10: dict(name="maia_1900", kind="lc0",  weights="data/engines/maia/maia-1900.pb.gz", nodes=1),
}


def gen_chunk(task):
    """One worker: n anchors from one source shard, each rolled out under every available
    regime. Own engines (opened once per regime), own RNG."""
    src_path, n_anchors, j, sf_depth, regime_ids, seed = task
    import chess
    import chess.engine
    from catspace.data.encode import board_from_packed, encode_meta, encode_packed
    rng = np.random.default_rng(seed)
    z = np.load(src_path)
    P, M = z["packed"], z["meta"]
    WE, BE, CK = z["white_elo"], z["black_elo"], z["clock"]
    rows = rng.choice(len(P), size=n_anchors * 2, replace=False)

    engines, skipped = {}, []
    def get_engine(rid):
        spec = REGIMES[rid]
        key = (spec["kind"], spec.get("weights"), json.dumps(spec.get("opts", {}), sort_keys=True))
        if key in engines:
            return engines[key]
        try:
            if spec["kind"] in ("sf", "rand_vs_sf"):
                e = chess.engine.SimpleEngine.popen_uci("stockfish")
                e.configure({"Threads": 1, **spec.get("opts", {})})
            else:                                        # lc0 / maia
                w = spec.get("weights")
                if w and not Path(w).exists():
                    raise FileNotFoundError(w)
                cmd = ["lc0"] + ([f"--weights={w}"] if w else [])
                e = chess.engine.SimpleEngine.popen_uci(cmd)
        except Exception as ex:
            engines[key] = None
            skipped.append(f"{spec['name']}: {type(ex).__name__}")
            return None
        engines[key] = e
        return e

    def engine_move(rid, b):
        spec = REGIMES[rid]
        e = get_engine(rid)
        if e is None:
            return None
        lim = (chess.engine.Limit(nodes=spec["nodes"]) if spec.get("nodes")
               else chess.engine.Limit(depth=sf_depth))
        try:
            return e.play(b, lim).move
        except Exception:
            return None

    cols = {k: [] for k in ("pk", "mt", "ply", "clk", "res", "we", "be", "gid", "reg", "aidx")}
    anchors_meta = []
    gid = made = 0
    for r in rows:
        if made >= n_anchors:
            break
        b0 = board_from_packed(P[r], M[r])
        if b0.is_game_over(claim_draw=True) or len(b0.piece_map()) < 5:
            continue
        drift_side = b0.turn
        walks = {}
        for rid in regime_ids:
            spec = REGIMES[rid]
            b = b0.copy(stack=False); walk = [b.copy(stack=False)]
            dead = False
            for _t in range(j):
                if b.is_game_over(claim_draw=True):
                    break
                if spec["kind"] == "rand_vs_sf" and b.turn == drift_side:
                    lm = list(b.legal_moves)
                    mv = lm[int(rng.integers(len(lm)))]
                else:
                    mv = engine_move(rid, b)
                    if mv is None:
                        dead = True; break
                b.push(mv); walk.append(b.copy(stack=False))
            if not dead and len(walk) >= 4:
                walks[rid] = walk
        if not walks or 2 not in walks:                  # require at least the sf_full reference
            continue
        aidx = len(anchors_meta)
        anchors_meta.append(dict(source_file=Path(src_path).name, source_row=int(r),
                                 fen=b0.fen(), white_elo=int(WE[r]), black_elo=int(BE[r]),
                                 clock=float(CK[r]),
                                 anchor_turn="w" if drift_side == chess.WHITE else "b",
                                 regimes=sorted(walks)))
        for rid, walk in walks.items():
            for t, w in enumerate(walk):
                cols["pk"].append(encode_packed(w)); cols["mt"].append(encode_meta(w))
                cols["ply"].append(t); cols["clk"].append(float(CK[r])); cols["res"].append(0)
                cols["we"].append(int(WE[r])); cols["be"].append(int(BE[r]))
                cols["gid"].append(gid); cols["reg"].append(rid); cols["aidx"].append(aidx)
            gid += 1
        made += 1
    for e in engines.values():
        if e is not None:
            try: e.quit()
            except Exception: pass
    return cols, anchors_meta, made, sorted(set(skipped))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default="data/shards/lichess_db_standard_rated_2019-01.prefix4gb")
    ap.add_argument("--out-dir", default="data/shards/regime_rollouts_v1")
    ap.add_argument("--chunk-anchors", type=int, default=200)
    ap.add_argument("--j", type=int, default=12)
    ap.add_argument("--sf-depth", type=int, default=8)
    ap.add_argument("--regimes", default="2,3,4,5,6,8,9,10",
                    help="comma-separated regime ids (see REGIMES; 7=lc0-full pending a net)")
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count - 4")
    ap.add_argument("--max-gb", type=float, default=10.0)
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    stop_file = out / "STOP"
    regime_ids = [int(x) for x in args.regimes.split(",")]
    workers = args.workers or max(2, (os.cpu_count() or 8) - 4)
    src_files = sorted(glob.glob(str(Path(args.shards) / "shard_*.npz")))
    existing = sorted(out.glob("shard_*.npz"))
    shard_no = 1 + max([int(p.stem.split("_")[1]) for p in existing], default=-1)
    print(f"[daemon] workers={workers} chunk={args.chunk_anchors} regimes={regime_ids} "
          f"sources={len(src_files)} starting at shard_{shard_no:03d}", flush=True)

    total_anchors = total_rows = 0
    warned = set()
    seed = args.seed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        while True:
            if stop_file.exists():
                print("[daemon] STOP sentinel found -- clean shutdown", flush=True)
                break
            size_gb = sum(p.stat().st_size for p in out.glob("*.npz")) / 1e9
            free_gb = shutil.disk_usage(out).free / 1e9
            if size_gb > args.max_gb or free_gb < args.min_free_gb:
                print(f"[daemon] guard stop: dataset {size_gb:.2f}GB free {free_gb:.0f}GB", flush=True)
                break
            tasks = [(src_files[(shard_no + w) % len(src_files)], args.chunk_anchors,
                      args.j, args.sf_depth, regime_ids, seed + w) for w in range(workers)]
            seed += workers
            for cols, ameta, made, skipped in ex.map(gen_chunk, tasks):
                for s in skipped:
                    if s not in warned:
                        warned.add(s); print(f"[daemon] regime unavailable: {s}", flush=True)
                if not made:
                    continue
                sp = out / f"shard_{shard_no:03d}.npz"
                np.savez_compressed(sp,
                    packed=np.stack(cols["pk"]), meta=np.stack(cols["mt"]),
                    ply=np.array(cols["ply"], np.int32), clock=np.array(cols["clk"], np.float32),
                    result=np.array(cols["res"], np.int8),
                    white_elo=np.array(cols["we"], np.uint16), black_elo=np.array(cols["be"], np.uint16),
                    game_id=np.array(cols["gid"], np.uint32), regime=np.array(cols["reg"], np.int8),
                    anchor_idx=np.array(cols["aidx"], np.int32))
                (out / f"anchors_{shard_no:03d}.json").write_text(json.dumps(ameta))
                total_anchors += made; total_rows += len(cols["pk"]); shard_no += 1
            man = dict(total_anchors=total_anchors, total_rows=total_rows, shards=shard_no,
                       regimes=regime_ids, elapsed_s=int(time.time() - t0), workers=workers,
                       rate_anchors_per_hr=int(total_anchors / max(time.time() - t0, 1) * 3600))
            (out / "manifest.json").write_text(json.dumps(man))
            print(f"[daemon] shards {shard_no}  anchors {total_anchors:,}  rows {total_rows:,}  "
                  f"({man['rate_anchors_per_hr']:,}/hr)  [{time.time()-t0:.0f}s]", flush=True)
    print(f"VERDICT ROLLOUT_DAEMON anchors={total_anchors:,} rows={total_rows:,} shards={shard_no} "
          f"regimes={regime_ids} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
