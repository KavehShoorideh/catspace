#!/usr/bin/env python
"""eval_split_axioms.py -- is each block STILL a quasimetric, and is the split real?

Kaveh 2026-08-08: 'make sure the coordinates make sense as a quasimetric.' Per block (A=length,
B=outcome), on held-out positions:
  identity        d(x,x) = 0 exactly (IQE structural, verified anyway)
  non-negativity  min d >= 0
  triangle        sampled u,v,w: d(u->w) <= d(u->v) + d(v->w), violation rate (must be 0)
  asymmetry       median |d(u->v) - d(v->u)| / mean d -- a quasimetric should HAVE some
  A/B honesty     corr(d_A, d_B) over pairs -- near 1 means the split is cosmetic
  composition     along held-out game paths: d(a->c) vs d(a->b)+d(b->c) tightness
  block ranks     effective rank of each 16-dim half separately
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.experiments.plot_strata_figures import (
    embed)
from catspace.research.components.encoder.approaches.reachability_field.experiments.arch_bakeoff import (
    eff_rank)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    net, pay = load_net(args.ckpt, args.device)
    assert getattr(net, "split_head", False), "not a split-head checkpoint"
    c = pay["cfg"]
    tr = T.build(n_human=0 if c.get("sf_only") else c["games"] // 2,
                 n_sf=c["games"] if c.get("sf_only") else c["games"] // 2,
                 seed=c["traj_seed"], max_plies=c["max_plies"],
                 n_piecedown=c.get("n_piecedown", 0), verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    test = np.flatnonzero(split == 2)
    game, ply = tr.game_of_row(), tr.ply_of_row()
    rows = np.flatnonzero(np.isin(game, test))
    rng = np.random.default_rng(0)
    sel = rng.choice(rows, args.n, replace=False)
    Z = embed(net, tr, sel, args.device).to(args.device)
    h = Z.shape[1] // 2

    for name, dfn, blk in (("A(length)", net.dA, Z[:, :h]), ("B(outcome)", net.dB, Z[:, h:])):
        u, v, w = (rng.integers(0, len(sel), 4000) for _ in range(3))
        with torch.no_grad():
            duv = dfn(Z[u], Z[v]); dvw = dfn(Z[v], Z[w]); duw = dfn(Z[u], Z[w])
            dvu = dfn(Z[v], Z[u])
            dself = dfn(Z[:200], Z[:200])
        tri_viol = float((duw > duv + dvw + 1e-4).float().mean())
        asym = float((duv - dvu).abs().median() / duv.mean().clamp(min=1e-9))
        print(f"[{name}] identity max d(x,x) = {float(dself.max()):.2e} | min d = "
              f"{float(duv.min()):.4f} | triangle violations = {tri_viol:.4%} | "
              f"asymmetry |d-dT|/mean = {asym:.3f} | block eff-rank = "
              f"{eff_rank(blk.detach().float().cpu().numpy()):.1f}/{h}")

    # A/B honesty: are the two rulers measuring different things?
    u, v = rng.integers(0, len(sel), 6000), rng.integers(0, len(sel), 6000)
    with torch.no_grad():
        da = net.dA(Z[u], Z[v]).float().cpu().numpy()
        db = net.dB(Z[u], Z[v]).float().cpu().numpy()
    print(f"[split] corr(d_A, d_B) over random pairs = {np.corrcoef(da, db)[0,1]:+.3f}  "
          f"(near +1 = cosmetic split; low/moderate = genuinely two rulers)")

    # composition along real held-out paths (a < b < c in one game)
    g0 = rng.choice(test, 800)
    ok = tr.length[g0] > 12
    g0 = g0[ok][:500]
    a = tr.start[g0] + 2
    b = a + 4
    cc = b + 4
    Za, Zb, Zc = (embed(net, tr, x, args.device).to(args.device) for x in (a, b, cc))
    for name, dfn in (("A", net.dA), ("B", net.dB)):
        with torch.no_grad():
            lhs = dfn(Za, Zc); rhs = dfn(Za, Zb) + dfn(Zb, Zc)
        r = (lhs / rhs.clamp(min=1e-9)).float().cpu().numpy()
        print(f"[{name}] path composition d(a->c)/[d(a->b)+d(b->c)]: median {np.median(r):.3f} "
              f"(<=1 required; ~1 = tight additive chains)")


if __name__ == "__main__":
    main()
