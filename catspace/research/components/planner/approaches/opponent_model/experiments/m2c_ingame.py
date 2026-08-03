#!/usr/bin/env python
"""catspace/research/components/planner/approaches/opponent_model/experiments/m2c_ingame.py -- M2c: in-game z tightening. As a player's moves ARRIVE (Kaveh: "z from
history OR live moves as they come in"), how few observed moves until the conditioned opponent model
beats the prior? We sweep the number of OBSERVED moves N: recover z from N of the player's moves,
RETRIEVE-and-CONDITION (memory infer_then_condition_z: k-NN to clean training styles), and score the
player's held-out (game-disjoint) query moves. Cold start N=0 = raw Maia (the prior).

Break-even curve: NLL-lift vs base, and the player-specific gap vs a rating-matched wrong player's
N-move z, as functions of N. Reuses the M2b cache + trained style library (no re-precompute).
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from catspace.research.components.planner.approaches.opponent_model.src.style_model import StyleResidual, VOCAB
from catspace.research.components.planner.approaches.opponent_model.src.style_recover import recover_delta, _precompute_U
from catspace.research.components.planner.approaches.opponent_model.src.style_dataio import load_cache as load_cache_arrays
from catspace.research.tools.stats_eval.stats import paired_nll_ci
from catspace.research.tools.training_infra.train.scaffold import resolve_device
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=paths.derived("m2b/cache_3k.npz"))
    ap.add_argument("--model", default=paths.experiment("m2b_style_3k.pt"))
    ap.add_argument("--observed", type=int, nargs="+", default=[5, 10, 20, 40, 80, 160])
    ap.add_argument("--k", type=int, default=50, help="nearest clean training styles to blend")
    ap.add_argument("--min-query", type=int, default=15); ap.add_argument("--query-frac", type=float, default=0.4)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); rng = np.random.default_rng(args.seed)
    Ns = sorted(args.observed)

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
    z_train = model.delta.weight.detach()

    def feats(idx):
        return {k: A[k][idx].to(dev) for k in ("phi", "cand_idx", "cand_logp", "cand_mask", "rank",
                                               "played_slot", "elo")}

    def cond_z(delta):                                     # infer-then-condition: retrieve clean styles
        order = torch.argsort(((z_train - delta.unsqueeze(0)) ** 2).sum(-1))
        return z_train[order[:args.k]].mean(0)

    held = np.flatnonzero(split == "heldout"); players = np.unique(pid[held])
    # --- pass 1: per player, recover z from N observed moves (game-disjoint from query) ---
    per = []
    for p in players:
        ridx = held[pid[held] == p]
        games = np.unique(game[ridx]); rng.shuffle(games)
        nq = max(1, int(len(games) * args.query_frac))
        qgames = set(games[:nq].tolist())
        qry = ridx[np.isin(game[ridx], list(qgames))]
        obs_pool = ridx[~np.isin(game[ridx], list(qgames))]
        if len(qry) < args.min_query or len(obs_pool) < Ns[0]:
            continue
        deltas = {}
        for N in Ns:
            if len(obs_pool) < N:
                continue
            sup = rng.choice(obs_pool, N, replace=False)
            d, _ = recover_delta(model, feats(sup), lam=args.lam, steps=60, device=dev)
            deltas[N] = d
        per.append({"pid": int(p), "qry": qry, "elo": float(np.median(z["elo_self"][ridx])), "deltas": deltas})
    if len(per) < 8:
        print(f"[m2c] only {len(per)} eligible players -- aborting"); return

    # --- pass 2: score each player's query at each N (base once; correct + rating-matched wrong) ---
    elos = np.array([q["elo"] for q in per])
    for i, q in enumerate(per):
        Uq, muq, logpq, maskq, playedq = _precompute_U(model, feats(q["qry"]), dev)

        def qnll(zc):                                      # cheap: U precomputed, mu=0
            logit = (logpq + (Uq * zc.unsqueeze(0)).sum(-1)).masked_fill(~maskq, -1e9)
            return (-F.log_softmax(logit, -1).gather(1, playedq.view(-1, 1)).squeeze(1)).cpu().numpy()

        dd = np.abs(elos - q["elo"]); dd[i] = 1e9; j = int(np.argmin(dd))
        q["base"] = qnll(torch.zeros(model.d_z, device=dev))
        q["cor"] = {}; q["wrong"] = {}
        for N in Ns:
            if N in q["deltas"]:
                q["cor"][N] = qnll(cond_z(q["deltas"][N]))
            if N in per[j]["deltas"]:
                q["wrong"][N] = qnll(cond_z(per[j]["deltas"][N]))

    print(f"\n===== M2c in-game tightening | held-out players {len(per)} | k={args.k} =====")
    print(f"  cold start (N=0) = raw Maia (prior). break-even curve:")
    print(f"  {'N obs':>6} {'players':>8} {'cond vs base (nats)':>28} {'vs wrong-player':>26}")
    for N in Ns:
        idxs = [i for i, q in enumerate(per) if N in q["cor"] and N in q["wrong"]]
        if len(idxs) < 8:
            continue
        clv = np.concatenate([np.full(len(per[i]["qry"]), i) for i in idxs])
        base = np.concatenate([per[i]["base"] for i in idxs])
        cor = np.concatenate([per[i]["cor"][N] for i in idxs])
        wro = np.concatenate([per[i]["wrong"][N] for i in idxs])
        lb, lo_b, hi_b, pb = paired_nll_ci(cor, base, clusters=clv, n_boot=1500, seed=1)
        lw, lo_w, hi_w, pw = paired_nll_ci(cor, wro, clusters=clv, n_boot=1500, seed=2)
        fb = "PASS" if lo_b > 0 else "fail"; fw = "PASS" if lo_w > 0 else "fail"
        print(f"  {N:>6} {len(idxs):>8}   {lb:+.4f} [{lo_b:+.4f},{hi_b:+.4f}] {fb:>4}   {lw:+.4f} [{lo_w:+.4f},{hi_w:+.4f}] {fw}")
    print(f"VERDICT M2c done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
