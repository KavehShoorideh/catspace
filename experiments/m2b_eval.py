#!/usr/bin/env python
"""experiments/m2b_eval.py -- M2b gate: does the recovered style residual z improve held-out move
prediction, and is it THIS player (not generic capacity)? Per Kaveh (style_z_allows_strength) we keep
only VALIDITY controls, not purity firewalls:

  A0 BASE      : raw Maia-2 (z=0), NLL over the candidate set
  A_prior      : z = mu(Elo) only (Delta=0) -- the rating-conditioned prior
  A2 STYLE     : z recovered post-hoc from the player's SUPPORT games (game-disjoint from query)
  A3 WRONG-Z   : score the player's query with a RATING-MATCHED OTHER player's z (the decisive placebo)

Evaluated on HELD-OUT PLAYERS (never trained). Significance = player-clustered bootstrap
(catspace/stats.paired_nll_ci; resample players, never positions). GATE: A2>A0 and A2>A3, CI floor>0.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.style.model import StyleResidual, VOCAB
from catspace.style.recover import recover_delta, score_nll, base_nll
from catspace.style.dataio import load_cache as load_cache_arrays
from catspace.stats import paired_nll_ci
from catspace.train.scaffold import resolve_device


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/derived/m2b/cache.npz")
    ap.add_argument("--model", default="artifacts/experiments/m2b_style.pt")
    ap.add_argument("--min-support", type=int, default=40); ap.add_argument("--max-support", type=int, default=150)
    ap.add_argument("--min-query", type=int, default=15)
    ap.add_argument("--support-frac", type=float, default=0.6)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--band", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); rng = np.random.default_rng(args.seed)

    z = load_cache_arrays(args.cache)                     # dict; shard-dir or legacy .npz
    K = z["cand_idx"].shape[1]
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

    def feats(idx):
        return {k: A[k][idx].to(dev) for k in ("phi", "cand_idx", "cand_logp", "cand_mask", "rank",
                                               "played_slot", "elo")}

    valid = z["valid"] if "valid" in z else np.ones(len(split), bool)
    held = np.flatnonzero((split == "heldout") & valid)
    players = np.unique(pid[held])
    print(f"[m2b-eval] heldout positions {len(held):,} across {len(players):,} players [{time.time()-t0:.0f}s]", flush=True)

    per = []                                               # per eligible player: dict of arrays/deltas
    for p in players:
        ridx = held[pid[held] == p]
        games = np.unique(game[ridx]); rng.shuffle(games)
        n_sup_g = max(1, int(len(games) * args.support_frac))
        sup_g = set(games[:n_sup_g].tolist())
        sup = ridx[np.isin(game[ridx], list(sup_g))]
        qry = ridx[~np.isin(game[ridx], list(sup_g))]
        if len(sup) < args.min_support or len(qry) < args.min_query:
            continue
        if len(sup) > args.max_support:
            sup = rng.choice(sup, args.max_support, replace=False)
        delta, _ = recover_delta(model, feats(sup), lam=args.lam, steps=60, device=dev)
        fq = feats(qry)
        per.append({"pid": int(p), "qry": qry, "delta": delta,
                    "elo": float(np.median(z["elo_self"][ridx])),
                    "a0": base_nll(model, fq, device=dev).numpy(),
                    "aprior": score_nll(model, fq, torch.zeros(model.d_z, device=dev), device=dev).numpy(),
                    "a2": score_nll(model, fq, delta, device=dev).numpy()})
    if len(per) < 4:
        print(f"[m2b-eval] only {len(per)} eligible players -- underpowered; increase data. Aborting gate.")
        return
    # A3 wrong-z: rating-matched partner (nearest median-Elo other player)
    elos = np.array([q["elo"] for q in per])
    for i, q in enumerate(per):
        d = np.abs(elos - q["elo"]); d[i] = 1e9
        j = int(np.argmin(d)); q["partner_elo_gap"] = float(d[j])
        q["a3"] = score_nll(model, feats(q["qry"]), per[j]["delta"], device=dev).numpy()

    def cat(key):
        return np.concatenate([q[key] for q in per])
    clusters = np.concatenate([np.full(len(q["qry"]), q["pid"]) for q in per])
    a0, aprior, a2, a3 = cat("a0"), cat("aprior"), cat("a2"), cat("a3")
    n_pos = len(a0); n_pl = len(per); med_gap = float(np.median([q["partner_elo_gap"] for q in per]))

    print(f"\n===== M2b STYLE-z gate | held-out players {n_pl} | query positions {n_pos:,} | "
          f"median wrong-z Elo gap {med_gap:.0f} =====")
    print(f"  mean NLL: base {a0.mean():.4f} | prior(mu) {aprior.mean():.4f} | "
          f"style(z) {a2.mean():.4f} | wrong-z {a3.mean():.4f}")

    def report(name, better, base_):
        lift, lo, hi, p = paired_nll_ci(better, base_, clusters=clusters, n_boot=2000, seed=1)
        floor = lo > 0
        print(f"  {name:<22} lift {lift:+.4f} [{lo:+.4f},{hi:+.4f}] nats  P(better)={p:.3f}  {'PASS' if floor else 'fail'}")
        return floor

    g_base = report("A2 vs A0 (z helps)", a2, a0)
    g_prior = report("A2 vs prior (indiv.)", a2, aprior)
    g_wrong = report("A2 vs A3 (this player)", a2, a3)
    passed = g_base and g_wrong
    print(f"\nVERDICT M2b-z: {'PASS' if passed else 'FAIL'} "
          f"(gate = A2>A0 and A2>A3, player-clustered CI floor>0; A2>prior informational) "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
