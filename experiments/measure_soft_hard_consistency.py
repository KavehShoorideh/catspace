#!/usr/bin/env python
"""experiments/measure_soft_hard_consistency.py -- do the SOFT (rho head, -log-odds) and HARD
(IQE d) sides of the energy algebra agree at the ORDERING level? (Unit-level softmin<=hardmin
needs ply calibration -- define-identifications rule -- and is NOT claimed here.)

Per regime, on same-walk (x, future g) pairs from the banked rollouts:
  spearman( -log-odds_rho(x,g), d_IQE(F_c(x), B(g)) )   -- rank agreement soft vs hard
  cross-regime soft gap |soft_c(x,g) - soft_c'(x,g)|    -- do channels separate on the SOFT side?
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="data/derived/sep/lichess_mc2.pt")
    ap.add_argument("--rho", default="data/derived/sep/rho_head_v1.pt")
    ap.add_argument("--shards", default="data/shards/regime_rollouts_v1")
    ap.add_argument("--n-states", type=int, default=60_000)
    ap.add_argument("--n-pairs", type=int, default=4000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1)   # different seed than training run
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    from experiments.train_rho_head import RhoHead
    rp = torch.load(args.rho, map_location=dev, weights_only=False)
    head = RhoHead(d=rp["d"]).to(dev); head.load_state_dict(rp["state"]); head.eval()

    shard_files = sorted(glob.glob(str(Path(args.shards) / "shard_*.npz")))
    rng.shuffle(shard_files)
    P_, M_, WID, PLY, REG = [], [], [], [], []
    wid_next = 0
    for f in shard_files:
        z = np.load(f)
        gid, reg, ply = z["game_id"], z["regime"], z["ply"]
        change = np.flatnonzero((np.diff(gid.astype(np.int64)) != 0) | (np.diff(ply) <= 0)) + 1
        starts = np.concatenate([[0], change]); ends = np.concatenate([change, [len(gid)]])
        for s, e in zip(starts, ends):
            P_.append(z["packed"][s:e]); M_.append(z["meta"][s:e])
            WID.append(np.full(e - s, wid_next)); PLY.append(ply[s:e])
            REG.append(np.full(e - s, reg[s])); wid_next += 1
        if sum(len(p) for p in P_) >= args.n_states:
            break
    P = np.concatenate(P_)[:args.n_states]; M = np.concatenate(M_)[:args.n_states]
    WID = np.concatenate(WID)[:args.n_states]; PLY = np.concatenate(PLY)[:args.n_states]
    REG = np.concatenate(REG)[:args.n_states]
    n = len(P)
    walks = {}
    for i in range(n):
        walks.setdefault(WID[i], []).append(i)
    print(f"[data] {n} states, {len(walks)} walks, regimes {sorted(set(REG.tolist()))} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    om_base = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    regs_present = sorted(set(REG.tolist()))

    def embed(idx, regime_override=None):
        Fs = torch.zeros(len(idx), rp["d"], device=dev); Bs = torch.zeros(len(idx), rp["d"], device=dev)
        for s in range(0, len(idx), 1024):
            sl = idx[s:s + 1024]
            pl = torch.from_numpy(feature_planes(P[sl], M[sl])).to(dev)
            r = (REG[sl] if regime_override is None
                 else np.full(len(sl), regime_override)).astype(np.int64)
            om = np.concatenate([np.tile(om_base, (len(sl), 1)).astype(np.int64), r[:, None]], axis=1)
            with torch.no_grad():
                Fs[s:s + len(sl)] = fb.embed_F(pl, torch.from_numpy(om).to(dev))
                Bs[s:s + len(sl)] = fb.embed_B(pl)
        return Fs, Bs

    # ---- sample same-walk future pairs (any gap 1..20, uniform: spread of horizons) --
    xi = np.empty(args.n_pairs, np.int64); gi = np.empty(args.n_pairs, np.int64)
    k = 0
    while k < args.n_pairs:
        i = rng.integers(0, n)
        w = walks[WID[i]]
        pos = int(PLY[i])
        if len(w) - 1 <= pos:
            continue
        gap = int(rng.integers(1, min(21, len(w) - pos)))
        xi[k] = i; gi[k] = w[pos + gap]; k += 1

    from scipy.stats import spearmanr
    Fx, _ = embed(xi); _, Bg = embed(gi)
    with torch.no_grad():
        soft = (-head(Fx, Bg)).cpu().numpy()                       # -log-odds
    # IQE d row-aligned: small chunks, full matrix -> diagonal (the n^2 OOM lesson)
    hard = np.empty(args.n_pairs, np.float32)
    with torch.no_grad():
        for s in range(0, args.n_pairs, 512):
            e = min(s + 512, args.n_pairs)
            hard[s:e] = torch.diagonal(fb.distance_matrix(Fx[s:e], Bg[s:e])).cpu().numpy()

    print("VERDICT SOFT_HARD rank agreement, same-walk future pairs (gap 1-20):", flush=True)
    for r in regs_present:
        m = REG[xi] == r
        if m.sum() > 100:
            rho_s = spearmanr(soft[m], hard[m]).correlation
            print(f"    regime {r:2d}: spearman(soft, hard) {rho_s:+.3f}  n={int(m.sum())}", flush=True)
    rho_all = spearmanr(soft, hard).correlation
    print(f"    ALL: spearman {rho_all:+.3f}  n={args.n_pairs}", flush=True)

    # ---- cross-regime SOFT gaps on identical pairs (channel separation, soft side) ---
    sub = np.arange(min(1500, args.n_pairs))
    softs = {}
    for r in regs_present:
        Fr, _ = embed(xi[sub], regime_override=r)
        with torch.no_grad():
            softs[r] = (-head(Fr, Bg[sub])).cpu().numpy()
    base = softs[regs_present[0]]
    print("VERDICT SOFT_CHANNEL_GAPS vs regime %d (same pairs):" % regs_present[0], flush=True)
    for r in regs_present[1:]:
        d = softs[r] - base
        print(f"    regime {r:2d}: mean {d.mean():+.3f}  sd {d.std():.3f}  "
              f"mean|gap| {np.abs(d).mean():.3f}", flush=True)
    print(f"done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
