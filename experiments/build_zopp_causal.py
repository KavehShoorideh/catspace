#!/usr/bin/env python
"""experiments/build_zopp_causal.py -- the CAUSAL in-game opponent-style estimate for the reach
field's z_opp slot (THESIS §3/§4; memory z_conditioned_field_two_evaluators amendment).

For every game: feed the OPPONENT's observed decisions (cache_dense_opp) to the M2c
OpponentEstimator in ply order (elo_known = their rating, as at play time) and record, for each
of OUR 90k decision rows, the estimate available STRICTLY BEFORE that ply:
    z_opp_t (N,16)  -- infer-then-condition blend from opponent moves with ply < ours
                       (zeros = population prior when nothing observed yet)
    n_obs   (N,)    -- how many opponent moves that estimate has seen
Causality: train-time conditioning == play-time conditioning; no future leakage by construction.
Cost control (optimize-before-long-runs): the z is recomputed on a DOUBLING n_obs schedule
(1,2,4,8,16,32,64) and held between recomputes -- the estimate a live filter would cheaply keep.

Output: data/derived/reach/zopp_causal_v1.npz, row-aligned to reach_v1.npz (asserted on
game_id/ply). DVC-track after build.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.style.dataio import load_cache                     # noqa: E402
from catspace.style.estimator import OpponentEstimator           # noqa: E402
from catspace.style.model import VOCAB, StyleResidual            # noqa: E402

RECOMPUTE_AT = (1, 2, 4, 8, 16, 32, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp-cache", default="data/derived/m2b/cache_dense_opp")
    ap.add_argument("--reach", default="data/derived/reach/reach_v1.npz")
    ap.add_argument("--model", default="artifacts/experiments/m2b_style_3k.pt")
    ap.add_argument("--train-cache", default="data/derived/m2b/cache_3k.npz",
                    help="source of per-player train Elos (z-table retrieval bands)")
    ap.add_argument("--out", default="data/derived/reach/zopp_causal_v1.npz")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--elo-band", type=int, default=100)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--device", default="cpu", help="cpu by default -- MPS stays free for training")
    args = ap.parse_args()
    t0 = time.time()

    # ---- frozen style model + clean training styles + their Elos ----
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

    # ---- opponent decisions (features) and our rows (targets) ----
    oc = load_cache(args.opp_cache)
    K = oc["cand_idx"].shape[1]
    ci_raw = oc["cand_idx"].astype(np.int64)
    # LATENT-TRAP GUARD: maia2's all_moves_dict emits indices up to ~1879, but the M2b model's
    # move_emb is sized VOCAB(1858)+pad -- the 3k training cache never contained the overflow,
    # the dense caches do. Moves the model cannot score are PADDING (strict mask); observations
    # whose PLAYED move is unscorable are skipped entirely.
    overflow = ci_raw >= VOCAB
    O = {"phi": torch.from_numpy(oc["phi"].astype(np.float32)),
         "cand_idx": torch.from_numpy(np.where(overflow, VOCAB, ci_raw)),
         "cand_logp": torch.from_numpy(oc["cand_logp"].astype(np.float32)),
         "played_slot": torch.from_numpy(oc["played_slot"].astype(np.int64)),
         "elo": torch.from_numpy(oc["elo_self"].astype(np.float32))}
    O["cand_mask"] = torch.from_numpy(~overflow) & (O["cand_idx"] != VOCAB)
    played_ok = ~overflow[np.arange(len(ci_raw)), oc["played_slot"].astype(np.int64)]
    print(f"vocab-overflow: {overflow.any(1).sum()} rows with masked candidates; "
          f"{(~played_ok).sum()} observations dropped (played move unscorable)")
    O["rank"] = (torch.arange(K).float() / (K - 1)).unsqueeze(0).expand(len(O["phi"]), -1).contiguous()
    o_game, o_ply = np.asarray(oc["game_id"]), np.asarray(oc["ply"])

    r = dict(np.load(args.reach, allow_pickle=True))
    r_game, r_ply = r["game_id"], r["ply"]
    N = len(r_game)
    d_z = ck["d_z"]
    z_out = np.zeros((N, d_z), np.float32)
    n_out = np.zeros(N, np.int16)

    o_order = np.lexsort((o_ply, o_game))
    r_order = np.lexsort((r_ply, r_game))
    o_ptr = 0
    done_games = 0
    # walk our rows game by game; advance the opponent stream in lockstep
    starts = np.flatnonzero(np.r_[True, r_game[r_order][1:] != r_game[r_order][:-1]])
    bounds = np.r_[starts, len(r_order)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        rows = r_order[s:e]
        gid = r_game[rows[0]]
        # opponent observations for this game, ply-ascending
        oi = []
        while o_ptr < len(o_order) and o_game[o_order[o_ptr]] < gid:
            o_ptr += 1
        p = o_ptr
        while p < len(o_order) and o_game[o_order[p]] == gid:
            oi.append(o_order[p]); p += 1
        est = OpponentEstimator(model, z_train, train_elo, k=args.k, elo_band=args.elo_band,
                                lam=args.lam, elo_known=float(r["elo_oppo"][rows[0]]),
                                device=args.device)
        cur_z = np.zeros(d_z, np.float32); cur_n = 0; next_rc = 0
        j = 0
        for row in rows:                                   # rows ply-ascending
            while j < len(oi) and o_ply[oi[j]] < r_ply[row]:
                if played_ok[oi[j]]:
                    idx = torch.tensor([oi[j]])
                    est.observe({k: O[k][idx] for k in
                                 ("phi", "cand_idx", "cand_logp", "cand_mask", "rank",
                                  "played_slot", "elo")})
                j += 1
            if est.n_observed > cur_n:
                cur_n = est.n_observed
                if next_rc < len(RECOMPUTE_AT) and cur_n >= RECOMPUTE_AT[next_rc]:
                    with torch.no_grad():
                        cur_z = est.z().cpu().numpy().astype(np.float32)
                    while next_rc < len(RECOMPUTE_AT) and cur_n >= RECOMPUTE_AT[next_rc]:
                        next_rc += 1
            z_out[row] = cur_z; n_out[row] = cur_n
        done_games += 1
        if done_games % 500 == 0:
            print(f"  {done_games} games, {time.time()-t0:.0f}s", flush=True)

    # ---- audits ----
    print(f"AUDIT n_obs at our rows: med {np.median(n_out):.0f} | frac>=1 {(n_out>=1).mean():.3f} "
          f"| >=10 {(n_out>=10).mean():.3f} | >=40 {(n_out>=40).mean():.3f}")
    nz = np.linalg.norm(z_out, axis=1)
    print(f"AUDIT |z_opp|: zero-frac {(nz==0).mean():.3f} | med(nonzero) {np.median(nz[nz>0]):.3f} "
          f"| corr(|z|, n_obs) {np.corrcoef(nz, n_out)[0,1]:.3f}")
    assert (n_out >= 10).mean() > 0.5, "most rows should have >=10 observed opponent moves by mid-game"
    assert (nz[n_out == 0] == 0).all(), "cold rows must be exactly the population prior (z=0)"

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, z_opp_t=z_out, n_obs=n_out,
                        game_id=r_game, ply=r_ply, meta_model=args.model,
                        meta_recompute=np.array(RECOMPUTE_AT))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
