#!/usr/bin/env python
"""experiments/build_zopp_causal_v2.py -- causal opponent-style estimates for the v2 reach data.

v2 cache holds BOTH sides' decisions, so per game we run TWO OpponentEstimators (one per color);
each row's z_opp(t) is the OTHER side's estimator state from rows with ply < t (strictly causal,
train == play conditioning). elo_known per estimator = that side's rating as seen in the records.
Same vocab-overflow guard, doubling recompute schedule, and audits as v1
(experiments/build_zopp_causal.py, kept for provenance).

Output: data/derived/reach/zopp_causal_v2.npz -- z_opp_t (N,16) + n_obs (N,) in cache_v2 row
order (game_id/ply carried for the trainer's alignment assert).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.research.components.planner.approaches.opponent_model.src.style_dataio import load_cache                     # noqa: E402
from catspace.research.components.planner.approaches.opponent_model.src.style_estimator import OpponentEstimator           # noqa: E402
from catspace.research.components.planner.approaches.opponent_model.src.style_model import VOCAB, StyleResidual            # noqa: E402

RECOMPUTE_AT = (1, 2, 4, 8, 16, 32, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/derived/m2b/cache_v2")
    ap.add_argument("--model", default="artifacts/experiments/m2b_style_3k.pt")
    ap.add_argument("--train-cache", default="data/derived/m2b/cache_3k.npz")
    ap.add_argument("--out", default="data/derived/reach/zopp_causal_v2.npz")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--elo-band", type=int, default=100)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    t0 = time.time()

    ck = torch.load(args.model, map_location=args.device, weights_only=False)
    model = StyleResidual(n_individual=ck["n_individual"], d_z=ck["d_z"], lam_prior=ck["lam"],
                          learn_mu=ck.get("learn_mu", False)).to(args.device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    z_train = model.delta.weight.detach()
    tc = dict(np.load(args.train_cache, allow_pickle=True))
    tp, te = tc["pidx"], tc["elo_self"]
    train_elo = np.zeros(ck["n_individual"], np.float32)
    for p in range(ck["n_individual"]):
        train_elo[p] = te[tp == p].mean()

    c = load_cache(args.cache)
    K = c["cand_idx"].shape[1]
    ci_raw = c["cand_idx"].astype(np.int64)
    overflow = ci_raw >= VOCAB
    F = {"phi": torch.from_numpy(c["phi"].astype(np.float32)),
         "cand_idx": torch.from_numpy(np.where(overflow, VOCAB, ci_raw)),
         "cand_logp": torch.from_numpy(c["cand_logp"].astype(np.float32)),
         "played_slot": torch.from_numpy(c["played_slot"].astype(np.int64)),
         "elo": torch.from_numpy(c["elo_self"].astype(np.float32))}
    F["cand_mask"] = torch.from_numpy(~overflow) & (F["cand_idx"] != VOCAB)
    F["rank"] = (torch.arange(K).float() / (K - 1)).unsqueeze(0).expand(len(F["phi"]), -1).contiguous()
    played_ok = ~overflow[np.arange(len(ci_raw)), c["played_slot"].astype(np.int64)]
    print(f"vocab-overflow: {overflow.any(1).sum():,} rows w/ masked candidates; "
          f"{(~played_ok).sum():,} observations dropped")

    game = np.asarray(c["game_id"]); ply = np.asarray(c["ply"])
    white = np.asarray(c["white"]).astype(bool)
    elo_s = np.asarray(c["elo_self"], np.float32); elo_o = np.asarray(c["elo_oppo"], np.float32)
    N = len(game); d_z = ck["d_z"]
    z_out = np.zeros((N, d_z), np.float32); n_out = np.zeros(N, np.int16)
    FEAT_KEYS = ("phi", "cand_idx", "cand_logp", "cand_mask", "rank", "played_slot", "elo")

    order = np.lexsort((ply, game))
    starts = np.flatnonzero(np.r_[True, game[order][1:] != game[order][:-1]])
    bounds = np.r_[starts, len(order)]
    for gi, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        rows = order[s:e]                                       # one game, ply-ascending
        est, cur_z, cur_n, next_rc = {}, {}, {}, {}
        for side in (True, False):                              # estimator FOR side's moves
            side_rows = rows[white[rows] == side]
            eo = float(elo_s[side_rows[0]]) if len(side_rows) else 1500.0
            est[side] = OpponentEstimator(model, z_train, train_elo, k=args.k,
                                          elo_band=args.elo_band, lam=args.lam,
                                          elo_known=eo, device=args.device)
            cur_z[side] = np.zeros(d_z, np.float32); cur_n[side] = 0; next_rc[side] = 0
        for row in rows:
            mover = white[row]; opp = not mover
            # this row's z_opp = estimate of the OPPONENT from THEIR moves so far (ply < row's)
            z_out[row] = cur_z[opp]; n_out[row] = cur_n[opp]
            # then the mover's own move becomes an observation for the MOVER's estimator
            if played_ok[row]:
                idx = torch.tensor([int(row)])
                est[mover].observe({k: F[k][idx] for k in FEAT_KEYS})
                n = est[mover].n_observed
                if n > cur_n[mover]:
                    cur_n[mover] = n
                    if next_rc[mover] < len(RECOMPUTE_AT) and n >= RECOMPUTE_AT[next_rc[mover]]:
                        with torch.no_grad():
                            cur_z[mover] = est[mover].z().cpu().numpy().astype(np.float32)
                        while (next_rc[mover] < len(RECOMPUTE_AT)
                               and n >= RECOMPUTE_AT[next_rc[mover]]):
                            next_rc[mover] += 1
        if (gi + 1) % 2000 == 0:
            print(f"  {gi+1:,} games, {time.time()-t0:.0f}s", flush=True)

    print(f"AUDIT n_obs: med {np.median(n_out):.0f} | >=1 {(n_out>=1).mean():.3f} | "
          f">=10 {(n_out>=10).mean():.3f} | >=40 {(n_out>=40).mean():.3f}")
    nz = np.linalg.norm(z_out, axis=1)
    print(f"AUDIT |z_opp|: zero-frac {(nz==0).mean():.3f} | med(nonzero) {np.median(nz[nz>0]):.3f} "
          f"| corr(|z|,n_obs) {np.corrcoef(nz, n_out)[0,1]:.3f}")
    assert (nz[n_out == 0] == 0).all()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, z_opp_t=z_out, n_obs=n_out, game_id=game.astype(np.int64),
                        ply=ply.astype(np.int32), meta_model=args.model,
                        meta_recompute=np.array(RECOMPUTE_AT))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
