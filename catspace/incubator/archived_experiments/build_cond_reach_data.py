#!/usr/bin/env python
"""catspace/incubator/archived_experiments/build_cond_reach_data.py -- assemble the z-CONDITIONED reachability training set for the
FiLM adapter (Kaveh 2026-07-27): player-label the positions, run the z-embedding over each player, attach
z per position, then MASK the player IDs -- only z (+ context) is carried forward; RAW IDs never reach
the adapter (decision 4: condition on STYLE, not identity; generalizes to unseen players via their z).

Reuses the M2b cache (player-grouped positions: phi, pidx, elo, game, ply) + the trained style model
(the per-player z = the estimator's embedding). Output npz carries phi, elo, ply, game (for same-game
reachability pairs), z (16), z_unc (1) -- and NO player_id / pidx.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import torch

from catspace.research.components.planner.approaches.opponent_model.src.style_model import StyleResidual
from catspace.research.components.planner.approaches.opponent_model.src.style_dataio import load_cache
from catspace.io import paths


def main():
    cache = paths.derived("m2b/cache_3k.npz"); model_path = paths.experiment("m2b_style_3k.pt")
    out = paths.derived("cond_reach_data.npz")
    t0 = time.time()
    z = load_cache(cache)
    split = z["split"]; pidx = z["pidx"].astype(np.int64)
    m = (split == "train") & (pidx >= 0)                          # train players carry a trained z
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    model = StyleResidual(n_individual=ck["n_individual"], d_z=ck["d_z"], lam_prior=ck["lam"],
                          learn_mu=ck.get("learn_mu", False))
    model.load_state_dict(ck["state_dict"])
    z_train = model.delta.weight.detach().numpy()                # (n_train, d_z) = per-player z embedding

    pi = pidx[m]
    cnt = np.bincount(pi, minlength=len(z_train)).astype(np.float32)
    z_unc_player = 1.0 / np.sqrt(np.maximum(cnt, 1.0))           # more games -> lower uncertainty
    z_unc_player = z_unc_player / (z_unc_player.max() + 1e-9)     # normalize to [0,1]

    zpos = z_train[pi].astype(np.float32)                        # per-position z (via masked-away pidx)
    zunc = z_unc_player[pi].astype(np.float32)
    phi = z["phi"][m].astype(np.float32); elo = z["elo_self"][m].astype(np.int16)
    game = z["game_id"][m].astype(np.int32); ply = z["ply"][m].astype(np.int16)
    n_players = len(np.unique(pi))
    # (player_id / pidx deliberately NOT saved -- masked. only z reaches the adapter.)
    np.savez(out, phi=phi, elo=elo, ply=ply, game=game, z=zpos, z_unc=zunc)
    print(f"=== {out}: {m.sum():,} positions | {n_players:,} players | z {zpos.shape[1]}d + z_unc | "
          f"IDs MASKED (only z carried) [{time.time()-t0:.0f}s] ===")
    print("DONE build_cond_reach_data", flush=True)


if __name__ == "__main__":
    main()
