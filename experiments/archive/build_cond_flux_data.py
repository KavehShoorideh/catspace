#!/usr/bin/env python
"""experiments/build_cond_flux_data.py -- combined set to test z-CONDITIONED REACHABILITY-TO-OUTCOMES
(= flux; Kaveh: rigidity=reachability=flux). Has BOTH the masked style z (per player) AND outcome-basin
labels (3-way WDL), which no single existing set had. From positions_3k (fen, player, game, ply, elo) +
cache_3k (phi, pidx->z from the trained model) + the trunk's 3-way WDL (fast iterate; SF referee for the
real run). Player IDs are used only to attach z, then dropped. Output: phi, z, z_unc, elo, ply, game, wdl.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.style.model import StyleResidual
from catspace.style.dataio import load_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", default="data/derived/m2b/positions_3k.parquet")
    ap.add_argument("--cache", default="data/derived/m2b/cache_3k.npz")
    ap.add_argument("--model", default="artifacts/experiments/m2b_style_3k.pt")
    ap.add_argument("--n-players", type=int, default=400); ap.add_argument("--out", default="data/derived/cond_flux_data.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)

    pos = pq.read_table(args.positions).to_pydict()
    cache = load_cache(args.cache)
    fen = np.array(pos["fen"], object); pidx = cache["pidx"].astype(np.int64)
    split = cache["split"]; phi_all = cache["phi"].astype(np.float32)
    game = cache["game_id"].astype(np.int32); ply = cache["ply"].astype(np.int16); elo = cache["elo_self"].astype(np.int16)

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = StyleResidual(n_individual=ck["n_individual"], d_z=ck["d_z"], lam_prior=ck["lam"], learn_mu=ck.get("learn_mu", False))
    model.load_state_dict(ck["state_dict"]); z_train = model.delta.weight.detach().numpy()

    tr_players = np.unique(pidx[(split == "train") & (pidx >= 0)])
    keep_players = rng.choice(tr_players, min(args.n_players, len(tr_players)), replace=False)
    m = np.isin(pidx, keep_players) & (split == "train")
    idx = np.flatnonzero(m)
    pi = pidx[idx]
    cnt = np.bincount(pi, minlength=len(z_train)).astype(np.float32)
    zunc_player = 1.0 / np.sqrt(np.maximum(cnt, 1.0)); zunc_player /= (zunc_player.max() + 1e-9)
    z = z_train[pi].astype(np.float32); zunc = zunc_player[pi].astype(np.float32)
    print(f"[flux-data] {len(idx):,} positions | {len(keep_players)} players [{time.time()-t0:.0f}s]", flush=True)

    from lczerolens import LczeroBoard
    from catspace.field import ReachabilityField
    field = ReachabilityField()
    fens = fen[idx]; wdl = np.empty((len(idx), 3), np.float32); B = 2048
    for s in range(0, len(idx), B):
        chunk = fens[s:s + B]
        x = torch.stack([LczeroBoard(f).to_input_tensor() for f in chunk]).float().to(field.dev)
        with torch.no_grad():
            wdl[s:s + len(chunk)] = field.trunk(x)["wdl"].cpu().numpy()
        if s % (B * 10) == 0:
            print(f"  wdl {s+len(chunk):,}/{len(idx):,} [{time.time()-t0:.0f}s]", flush=True)
    wdl = wdl / wdl.sum(1, keepdims=True).clip(1e-6)

    # player IDs masked: only z carried
    np.savez(args.out, phi=phi_all[idx], z=z, z_unc=zunc, elo=elo[idx], ply=ply[idx], game=game[idx], wdl=wdl)
    bc = np.bincount(wdl.argmax(1), minlength=3)
    print(f"=== {args.out}: {len(idx):,} pos | z16+unc | basin W/D/L {bc[0]}/{bc[1]}/{bc[2]} | IDs MASKED "
          f"[{time.time()-t0:.0f}s] ===")
    print("DONE build_cond_flux_data", flush=True)


if __name__ == "__main__":
    main()
