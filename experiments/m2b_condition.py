#!/usr/bin/env python
"""experiments/m2b_condition.py -- test Kaveh's decomposition (2026-07-27): INFER identity, then
CONDITION on a CLEAN style to predict, instead of applying the overfit per-player point-estimate z.

The direct residual z_P (recovered from ~150 support moves) DISCRIMINATES the player (A2>A3) but
OVERFITS support, so applied additively it net-hurts move prediction (A2<A0). Fix: use the recovered
z_P only to RETRIEVE -- k-NN to the TRAINING players' z (each fit on that player's FULL data, so it
GENERALIZES) -- and predict the held-out player's query moves with that averaged clean style z_cond.

Arms on HELD-OUT players (player-clustered CIs): A0 base (raw Maia) | A2 direct z_P | A_knn conditioned
z_cond. Gate question: does A_knn beat A0 where A2 could not?
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.research.components.planner.approaches.opponent_model.src.style_model import StyleResidual, VOCAB
from catspace.research.components.planner.approaches.opponent_model.src.style_recover import recover_delta, score_nll, base_nll
from catspace.research.components.planner.approaches.opponent_model.src.style_dataio import load_cache as load_cache_arrays
from catspace.research.tools.stats_eval.stats import paired_nll_ci
from catspace.research.tools.training_infra.train.scaffold import resolve_device


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/derived/m2b/cache_3k.npz")
    ap.add_argument("--model", default="artifacts/experiments/m2b_style_3k.pt")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 10, 50], help="k nearest training styles to blend")
    ap.add_argument("--elo-band", type=int, default=0, help="restrict retrieval to training players within +/- this Elo (0 = all)")
    ap.add_argument("--min-support", type=int, default=40); ap.add_argument("--max-support", type=int, default=150)
    ap.add_argument("--min-query", type=int, default=15); ap.add_argument("--support-frac", type=float, default=0.6)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); rng = np.random.default_rng(args.seed)

    z = load_cache_arrays(args.cache); K = z["cand_idx"].shape[1]
    A = {"phi": torch.from_numpy(z["phi"].astype(np.float32)),
         "cand_idx": torch.from_numpy(z["cand_idx"].astype(np.int64)),
         "cand_logp": torch.from_numpy(z["cand_logp"].astype(np.float32)),
         "played_slot": torch.from_numpy(z["played_slot"].astype(np.int64)),
         "elo": torch.from_numpy(z["elo_self"].astype(np.float32))}
    A["cand_mask"] = A["cand_idx"] != VOCAB
    A["rank"] = (torch.arange(K).float() / (K - 1)).unsqueeze(0).expand(A["phi"].shape[0], -1).contiguous()
    split = z["split"]; pid = z["player_id"]; game = z["game_id"]

    ck = torch.load(args.model, map_location=dev, weights_only=False)
    model = StyleResidual(n_individual=ck["n_individual"], d_z=ck["d_z"], lam_prior=ck["lam"],
                          learn_mu=ck.get("learn_mu", False)).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    z_train = model.delta.weight.detach()                 # (n_train, d_z) -- clean, full-data styles
    # training-player Elos (for Elo-conditioned retrieval -- "we also know the elo", Kaveh 2026-07-27)
    tr = split == "train"; n_tr = int(z_train.shape[0])
    s = np.bincount(z["pidx"][tr], weights=z["elo_self"][tr].astype(float), minlength=n_tr)
    c = np.bincount(z["pidx"][tr], minlength=n_tr)
    train_elo = np.full(n_tr, 1500.0); train_elo[c > 0] = s[c > 0] / c[c > 0]
    train_elo_t = torch.tensor(train_elo, dtype=torch.float32, device=dev)

    def feats(idx):
        return {k: A[k][idx].to(dev) for k in ("phi", "cand_idx", "cand_logp", "cand_mask", "rank",
                                               "played_slot", "elo")}

    def retrieve(delta, elo, k):                           # k-NN clean training styles, optionally Elo-banded
        d2 = ((z_train - delta.unsqueeze(0)) ** 2).sum(-1)
        if args.elo_band > 0:
            d2 = d2 + ((train_elo_t - elo).abs() > args.elo_band).float() * 1e9
        return z_train[torch.argsort(d2)[:k]].mean(0)

    held = np.flatnonzero(split == "heldout"); players = np.unique(pid[held])
    per = []
    for p in players:
        ridx = held[pid[held] == p]
        games = np.unique(game[ridx]); rng.shuffle(games)
        sup_g = set(games[:max(1, int(len(games) * args.support_frac))].tolist())
        sup = ridx[np.isin(game[ridx], list(sup_g))]; qry = ridx[~np.isin(game[ridx], list(sup_g))]
        if len(sup) < args.min_support or len(qry) < args.min_query:
            continue
        if len(sup) > args.max_support:
            sup = rng.choice(sup, args.max_support, replace=False)
        delta, _ = recover_delta(model, feats(sup), lam=args.lam, steps=60, device=dev)
        rec = {"qry": qry, "delta": delta, "elo": float(np.median(z["elo_self"][ridx])),
               "a0": base_nll(model, feats(qry), device=dev).numpy(),
               "a2": score_nll(model, feats(qry), delta, device=dev).numpy()}
        for k in args.k:
            rec[f"aknn{k}"] = score_nll(model, feats(qry), retrieve(delta, rec["elo"], k), device=dev).numpy()
        per.append(rec)
    if len(per) < 4:
        print(f"[condition] only {len(per)} eligible players -- aborting"); return

    # IDENTITY CONTROL: condition on a RATING-MATCHED OTHER player's retrieval (wrong-z). If the
    # correct player's conditioning beats this, the win is player-SPECIFIC, not a generic style shift.
    elos = np.array([q["elo"] for q in per])
    for i, q in enumerate(per):
        dd = np.abs(elos - q["elo"]); dd[i] = 1e9; j = int(np.argmin(dd))
        for k in args.k:
            q[f"aknnw{k}"] = score_nll(model, feats(q["qry"]), retrieve(per[j]["delta"], per[j]["elo"], k), device=dev).numpy()

    clusters = np.concatenate([np.full(len(q["qry"]), i) for i, q in enumerate(per)])
    cat = lambda key: np.concatenate([q[key] for q in per])
    a0, a2 = cat("a0"), cat("a2")
    print(f"\n===== INFER-then-CONDITION | held-out players {len(per)} | query {len(a0):,} =====")
    print(f"  mean NLL: base {a0.mean():.4f} | direct z_P {a2.mean():.4f}")

    def report(name, better, baseline, tag="vs base"):
        lift, lo, hi, pb = paired_nll_ci(better, baseline, clusters=clusters, n_boot=2000, seed=1)
        print(f"  {name:<30} lift {lift:+.4f} [{lo:+.4f},{hi:+.4f}] {tag}  P={pb:.3f}  {'PASS' if lo>0 else 'fail'}")

    report("direct z_P (A2)", a2, a0)
    for k in args.k:
        ak = cat(f"aknn{k}"); akw = cat(f"aknnw{k}")
        print(f"  --- k={k}: cond {ak.mean():.4f} | wrong-cond {akw.mean():.4f} ---")
        report(f"conditioned (k={k})", ak, a0)
        report(f"  identity: cond vs wrong", ak, akw, tag="vs wrong-cond")
    print(f"VERDICT condition-test done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
