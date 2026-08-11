#!/usr/bin/env python
"""concept_interactions.py -- the EMPIRICAL concept-interaction matrix (Kaveh 2026-08-12):
does one concept's presence make another's ACTIVATION more or less likely on the next move?

For every pair of codes (a, b):  lift(a -> b) = P(b activates | a active) / P(b activates)
over the cached corpus transitions. lift >> 1 = synergy edge, lift << 1 = blocking edge.
Pure counting, zero training -- the premise check before the concept-transformer pays its
bill: if this matrix is white noise, the codes are too churny for relational structure.

    .venv/bin/python -m ...concept_interactions --ckpt <field.pt> [--n 800000]
writes <ckpt>_interactions.npz {lift, n_act, n_a, base}; prints top synergy/blocking edges
named via the concept map where a name exists.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.experiments.concept_vq import (
    ConceptVQ)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=800_000)
    ap.add_argument("--min-support", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    base_path = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    pv = torch.load(base_path + "_vq.pt", map_location=args.device, weights_only=False)
    vq = ConceptVQ(d_in=pv["d_in"], heads=pv["heads"], codes=pv["codes"]).to(args.device)
    vq.load_state_dict(pv["state_dict"]); vq.eval()
    net, pay = load_net(args.ckpt, args.device)
    c = pay["cfg"]
    tr = T.build(n_human=0, n_sf=c["games"], seed=c["traj_seed"], max_plies=c["max_plies"],
                 n_piecedown=c.get("n_piecedown", 0), verbose=False)
    H, C = pv["heads"], pv["codes"]
    K = H * C
    cache = paths.derived(f"game_transitions_{len(tr.tok)}.npz")
    z = np.load(cache)
    par = z["par"]
    rng = np.random.default_rng(0)
    par = par[rng.choice(len(par), min(args.n, len(par)), replace=False)]
    print(f"[ix] {len(par):,} transitions; concept space {H}x{C} = {K}")

    def codes_of(rows):
        out = []
        for a in range(0, len(rows), 4096):
            rr = rows[a:a + 4096]
            with torch.no_grad():
                phi = net.backbone(
                    torch.from_numpy(tr.tok[rr].astype(np.int64)).to(args.device),
                    torch.from_numpy(tr.glob[rr].astype(np.float32)).to(args.device))
                _, ids, _ = vq(phi)
            out.append(ids.cpu().numpy())
        return np.concatenate(out)

    print("[ix] coding parents/children...", flush=True)
    CP = codes_of(par)
    CC = codes_of(par + 1)

    # one-hot active-at-parent (N,K) and activates-on-this-move (N,K)
    N = len(CP)
    flat_p = CP + np.arange(H)[None, :] * C
    A = np.zeros((N, K), np.float32)
    np.put_along_axis(A, flat_p, 1.0, axis=1)
    act = (CC != CP)                                     # head changed
    flat_c = CC + np.arange(H)[None, :] * C
    B = np.zeros((N, K), np.float32)
    np.put_along_axis(B, flat_c, act.astype(np.float32), axis=1)

    n_a = A.sum(0)                                       # times each code was active
    n_b = B.sum(0)                                       # times each code activated
    count = A.T @ B                                      # (K,K): b-activations with a active
    base = n_b / N
    with np.errstate(divide="ignore", invalid="ignore"):
        p_given = count / n_a[:, None]
        lift = p_given / base[None, :]
    np.savez(base_path + "_interactions.npz", lift=lift, count=count, n_a=n_a, base=base)

    # name codes where the concept map knows them
    names = {}
    try:
        cm = json.load(open(base_path + "_conceptmap.json"))
        for k, v in cm.items():
            names[v["head"] * C + v["code"]] = k
    except Exception:
        pass
    def nm(i):
        return names.get(i, f"h{i//C}/c{i%C}")

    ok = (n_a[:, None] >= args.min_support) & (count + 0 >= 0) & (base[None, :] >= 0.002)
    ok &= ~np.eye(K, dtype=bool)
    L = np.where(ok, lift, 1.0)
    # significance guard: expected count under independence must be >= 20
    exp = n_a[:, None] * base[None, :]
    L = np.where(exp >= 20, L, 1.0)
    flat = L.ravel()
    top = np.argsort(-flat)[:12]
    bot = np.argsort(flat)[:12]
    print("\n[ix] SYNERGY edges (a active -> b activates far MORE than base):")
    for t in top:
        a, b = t // K, t % K
        print(f"  {nm(a):24s} -> {nm(b):24s} lift {L[a, b]:5.1f}x  "
              f"(P {count[a, b]/n_a[a]:.3f} vs base {base[b]:.3f}, n_a {int(n_a[a])})")
    print("\n[ix] BLOCKING edges (a active -> b activates far LESS than base):")
    for t in bot:
        a, b = t // K, t % K
        print(f"  {nm(a):24s} -| {nm(b):24s} lift {L[a, b]:5.2f}x  "
              f"(P {count[a, b]/n_a[a]:.4f} vs base {base[b]:.3f}, n_a {int(n_a[a])})")
    sig = float(((L > 2.0) | (L < 0.5)).mean())
    print(f"\n[ix] fraction of pairs with strong interaction (lift >2 or <0.5): {sig:.1%}")
    print(f"[ix] saved {base_path}_interactions.npz")


if __name__ == "__main__":
    main()
