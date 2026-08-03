#!/usr/bin/env python
"""experiments/m2b_cache.py -- M2b step 2: cache the two FROZEN feature sources for every sampled
position (both are the build bottleneck; compute once, DVC-track):
  MAIA-2 base : batched inference_batch -> per-position candidate set (top-K move-probs u {played}),
                base log-probs (indexed in the MOVER frame), played slot, white-POV win_prob.
  FIELD phi   : frozen ReachabilityField phi(s) in R^64.

RESUMABLE + CRASH-SAFE (Kaveh 2026-07-27: "don't lose progress"). The precompute is the long single-
shot job; a crash (disk-full / sleep / OOM) must not throw it away. So we write SHARDS, not one file:
  <out>/meta.npz              global metadata (pidx, elos, split, ... ; N, K) -- written once
  <out>/shard_0000.npz ...    contiguous position ranges [c*S : (c+1)*S), features only
Each shard is written to a .tmp then os.replace'd (atomic -> no half-written shard can exist). On
restart, chunks whose shard already exists are SKIPPED -> resume from where it died. A disk PRE-FLIGHT
aborts before starting if there isn't room. Load with catspace.research.components.planner.approaches.opponent_model.src.style_dataio.load_cache (auto-detects
this dir layout or a legacy single .npz).
"""
from __future__ import annotations

import argparse, math, os, shutil, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VOCAB = 1858
K = 17                                                     # top-16 candidates u {played}
BYTES_PER_POS = K * 4 + K * 4 + 2 + 4 + 64 * 2 + 1         # cand_idx+logp+slot+win+phi(f16)+valid


def atomic_savez(path: Path, **arrays):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:                             # file handle -> np.savez won't append ".npz"
        np.savez(f, **arrays)
    os.replace(tmp, path)                                  # atomic on POSIX: no partial shard survives


def cat_rep_elo(elo, elo_dict, map_to_category):
    reps = {0: 1050, 10: 2050}
    for k in range(1, 10):
        reps[k] = 1100 + (k - 1) * 100 + 50
    return reps[map_to_category(int(elo), elo_dict)]


def build_chunk(idx, pos, prepared, maia, inference, map_to_category, mirror_move, field, args, dev):
    """Compute Maia-base + phi features for the positions at `idx` (a contiguous range). Returns a
    dict of feature arrays + running diagnostics (played_in, base P(played) sum, valid count)."""
    all_moves_dict, elo_dict, _ = prepared
    fen = [pos["fen"][i] for i in idx]; played = [pos["played"][i] for i in idx]
    es = [pos["elo_self"][i] for i in idx]; eo = [pos["elo_oppo"][i] for i in idx]
    white = [pos["white"][i] for i in idx]
    n = len(idx)

    # --- Maia base, deduped by (fen, self_cat, oppo_cat) WITHIN the chunk (self-contained shard) ---
    uniq = {}; u_fen = []; u_es = []; u_eo = []; key_of = np.empty(n, np.int64)
    for j in range(n):
        cs = map_to_category(int(es[j]), elo_dict); co = map_to_category(int(eo[j]), elo_dict)
        k = (fen[j], cs, co); q = uniq.get(k)
        if q is None:
            q = len(u_fen); uniq[k] = q
            u_fen.append(fen[j]); u_es.append(cat_rep_elo(es[j], elo_dict, map_to_category))
            u_eo.append(cat_rep_elo(eo[j], elo_dict, map_to_category))
        key_of[j] = q
    df = pd.DataFrame({"fen": u_fen, "move": ["0000"] * len(u_fen), "elo_self": u_es, "elo_oppo": u_eo})
    df, _ = inference.inference_batch(df, maia, verbose=False, batch_size=args.batch_size,
                                      num_workers=args.num_workers)
    u_probs = df["move_probs"].tolist(); u_win = np.asarray(df["win_probs"].tolist(), np.float32)

    cand_idx = np.full((n, K), VOCAB, np.int32); cand_logp = np.full((n, K), -30.0, np.float32)
    played_slot = np.zeros(n, np.int16); win_prob = np.zeros(n, np.float32)
    played_in = 0; basep_sum = 0.0
    for j in range(n):
        d = u_probs[key_of[j]]; win_prob[j] = u_win[key_of[j]]; pl = played[j]
        top = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:K - 1]
        cand = [m for m, _ in top]
        if pl in cand:
            played_in += 1
        else:
            cand.append(pl)
        cand = cand[:K]
        if pl not in cand:
            cand[-1] = pl
        basep_sum += d.get(pl, 1e-6)
        wj = white[j]
        for c, m in enumerate(cand):
            key = m if wj else mirror_move(m)              # mover-frame index (matches phi)
            cand_idx[j, c] = all_moves_dict.get(key, VOCAB)
            cand_logp[j, c] = math.log(max(d.get(m, 1e-6), 1e-6))
        played_slot[j] = cand.index(pl)
    valid = cand_idx[np.arange(n), played_slot.astype(np.int64)] != VOCAB

    # --- phi, deduped by fen WITHIN the chunk ---
    from lczerolens import LczeroBoard
    ufen = {}; fen_key = np.empty(n, np.int64); uf = []
    for j in range(n):
        q = ufen.get(fen[j])
        if q is None:
            q = len(uf); ufen[fen[j]] = q; uf.append(fen[j])
        fen_key[j] = q
    phi_u = np.empty((len(uf), 64), np.float16)
    for s in range(0, len(uf), args.phi_batch):
        boards = [LczeroBoard(f) for f in uf[s:s + args.phi_batch]]
        phi_u[s:s + len(boards)] = field.phi(boards).to(torch.float16).cpu().numpy()
    phi = phi_u[fen_key]

    feats = dict(cand_idx=cand_idx, cand_logp=cand_logp, played_slot=played_slot,
                 win_prob=win_prob, phi=phi, valid=valid)
    return feats, played_in, basep_sum, int(valid.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", default="data/derived/m2b/positions.parquet")
    ap.add_argument("--out", default="data/derived/m2b/cache", help="OUTPUT DIRECTORY (shard layout)")
    ap.add_argument("--maia-type", default="rapid")
    ap.add_argument("--shard-size", type=int, default=100000, help="positions per resumable shard")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--phi-batch", type=int, default=4096)
    ap.add_argument("--min-free-gb", type=float, default=2.0, help="abort pre-flight if less free")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    t0 = time.time()
    from catspace.research.tools.training_infra.train.scaffold import resolve_device
    dev = resolve_device(args.device)

    pos = pq.read_table(args.positions).to_pydict()
    N = len(pos["fen"])
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    n_chunks = math.ceil(N / args.shard_size)

    # --- disk pre-flight: refuse to start if there isn't room for what's still missing ---
    done_chunks = {int(p.stem.split("_")[1]) for p in outdir.glob("shard_*.npz")}
    remaining = [c for c in range(n_chunks) if c not in done_chunks]
    need = len(remaining) * args.shard_size * BYTES_PER_POS * 1.3
    free = shutil.disk_usage(outdir).free
    print(f"[m2b-cache] {N:,} positions -> {n_chunks} shards of {args.shard_size:,} | "
          f"{len(done_chunks)} already done, {len(remaining)} to do | need ~{need/1e9:.2f}G, "
          f"free {free/1e9:.1f}G [{time.time()-t0:.0f}s]", flush=True)
    if free < max(need, args.min_free_gb * 1e9):
        sys.exit(f"[m2b-cache] ABORT pre-flight: free {free/1e9:.1f}G < needed "
                 f"{max(need, args.min_free_gb*1e9)/1e9:.1f}G. Free disk before caching.")

    # --- metadata (written once, atomically) ---
    meta_path = outdir / "meta.npz"
    if not meta_path.exists():
        atomic_savez(meta_path, N=np.int64(N), K=np.int64(K),
                     pidx=np.asarray(pos["pidx"], np.int32), prov=np.asarray(pos["prov"], np.bool_),
                     elo_self=np.asarray(pos["elo_self"], np.int16), elo_oppo=np.asarray(pos["elo_oppo"], np.int16),
                     white=np.asarray(pos["white"], np.bool_), player_id=np.asarray(pos["player_id"], np.uint64),
                     game_id=np.asarray(pos["game_id"], np.int32), split=np.asarray(pos["split"], "U8"),
                     ply=np.asarray(pos["ply"], np.int16))

    if not remaining:
        print(f"[m2b-cache] all shards present -> nothing to do [{time.time()-t0:.0f}s]"); print("DONE m2b_cache", flush=True); return

    # --- load frozen models ONCE, then fill missing shards (resume-safe) ---
    from maia2 import model as maia_model, inference
    from maia2.inference import map_to_category, mirror_move
    prepared = inference.prepare()
    maia = maia_model.from_pretrained(type=args.maia_type, device=str(dev))
    from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField
    field = ReachabilityField(device=args.device)

    tot_in = tot_valid = 0; tot_basep = 0.0; tot_done = 0
    for c in remaining:
        shard_path = outdir / f"shard_{c:04d}.npz"
        lo, hi = c * args.shard_size, min((c + 1) * args.shard_size, N)
        idx = list(range(lo, hi))
        feats, pin, bsum, nvalid = build_chunk(idx, pos, prepared, maia, inference, map_to_category,
                                               mirror_move, field, args, dev)
        atomic_savez(shard_path, **feats)
        tot_in += pin; tot_valid += nvalid; tot_basep += bsum; tot_done += len(idx)
        free = shutil.disk_usage(outdir).free
        print(f"[m2b-cache] shard {c:04d} [{lo:,}:{hi:,}] written | valid {100*nvalid/len(idx):.2f}% | "
              f"free {free/1e9:.1f}G [{time.time()-t0:.0f}s]", flush=True)

    if tot_done:
        print(f"[m2b-cache] this run: {tot_done:,} positions | played-in-top16 {100*tot_in/tot_done:.1f}% | "
              f"mean base P(played) {tot_basep/tot_done:.3f} | valid {100*tot_valid/tot_done:.3f}%")
    print(f"\n=== {outdir}: {N:,} positions cached in {n_chunks} shards [{time.time()-t0:.0f}s] ===")
    print("DONE m2b_cache", flush=True)


if __name__ == "__main__":
    main()
