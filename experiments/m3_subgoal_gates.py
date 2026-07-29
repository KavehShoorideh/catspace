#!/usr/bin/env python
"""experiments/m3_subgoal_gates.py -- the M3 DoD gates (MILESTONES M3), printed as VERDICTs:
  (1) LATENCY: per-query cost of the subgoal API at play budgets (measured, recorded);
  (2) OUT-OF-SAMPLE ENRICHMENT: on HELD-OUT games (odd game hash; the table used even only),
      positions inside the table's top-decile net-flux regions must show >= 2x the base
      SF-refereed crossing rate, for >= 2 rating bands;
  (3) BANDS DIFFER: the two bands' region flux maps are measurably different (Spearman + top-
      decile Jaccard), i.e. rating-conditioning is real, not one shared map.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.subgoals import SubgoalRanker                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="data/derived/transition_data_labeled.npz")
    ap.add_argument("--reach", default="data/derived/reach/reach_v2.npz")
    ap.add_argument("--table", default="data/derived/reach/region_table_v1.npz")
    ap.add_argument("--field", default="artifacts/experiments/reach_v2_full_latest.pt")
    ap.add_argument("--thr", type=float, default=0.2)
    ap.add_argument("--n-lat", type=int, default=200)
    ap.add_argument("--flux-t", default="", help="m3_flux_T checkpoint -> T-scored region flux")
    args = ap.parse_args()

    rk = SubgoalRanker(args.field, args.reach, args.table, flux_t=args.flux_t)
    d = dict(np.load(args.labeled, allow_pickle=True))
    bank = rk.bank.cpu().numpy().astype(np.float32)

    # ---- (1) latency ----
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d["phi"]), args.n_lat, replace=False)
    t0 = time.perf_counter()
    for i in idx:
        rk.rank(d["phi"][i], float(d["elo_mover"][i]), float(d["elo_opp"][i]),
                z_opp=np.zeros(16), n_obs=12)
    ms = (time.perf_counter() - t0) / args.n_lat * 1e3
    print(f"VERDICT M3 latency: {ms:.2f} ms/query (single-position, CPU, bank=256, "
          f"n={args.n_lat}) -- per-move-usable at play budgets")

    # ---- (2) out-of-sample enrichment on held-out games ----
    held = d["game"].astype(np.int64) % 2 == 1
    phi = d["phi"][held].astype(np.float32)
    d2 = (phi * phi).sum(1)[:, None] + (bank * bank).sum(1)[None, :] - 2.0 * phi @ bank.T
    region = d2.argmin(1)
    crossing = (d["mover_loss"][held] >= args.thr).astype(float)
    elo = d["elo_mover"][held]
    passes = []
    for b, name in ((0, "<1500"), (1, ">=1500")):
        m = (np.searchsorted(rk.band_edges, elo, side="right") == b)
        # top-decile regions by the API's flux path for this band (T-scored when --flux-t)
        band_rep = 1300 if b == 0 else 1800
        fl = rk.t_flux(band_rep, band_rep) if rk.T is not None else rk.flux[:, b]
        top_regions = set(np.argsort(-fl)[: len(bank) // 10].tolist())
        in_top = np.array([r in top_regions for r in region])
        base = crossing[m].mean()
        top = crossing[m & in_top].mean() if (m & in_top).sum() > 100 else float("nan")
        lift = top / base if base > 0 else float("nan")
        ok = lift >= 2.0
        passes.append(ok)
        print(f"VERDICT M3 enrichment band {name}: base {base:.1%} | top-decile-flux regions "
              f"{top:.1%} (n={(m & in_top).sum():,}) | lift {lift:.2f}x  "
              f"{'PASS' if ok else 'FAIL'} (gate >=2x)")

    # ---- (3) bands differ ----
    from scipy.stats import spearmanr
    f0 = rk.t_flux(1300, 1300) if rk.T is not None else rk.flux[:, 0]
    f1 = rk.t_flux(1800, 1800) if rk.T is not None else rk.flux[:, 1]
    rho = spearmanr(f0, f1).correlation
    t0s = set(np.argsort(-f0)[: len(bank) // 10].tolist())
    t1s = set(np.argsort(-f1)[: len(bank) // 10].tolist())
    jac = len(t0s & t1s) / len(t0s | t1s)
    print(f"VERDICT M3 bands-differ: Spearman(map<1500, map>=1500) {rho:+.3f} | top-decile "
          f"Jaccard {jac:.2f}  (maps {'DIFFER' if rho < 0.95 else 'IDENTICAL -- fail'})")
    print(f"VERDICT M3 GATES: {'ALL PASS' if all(passes) and rho < 0.95 else 'NOT ALL PASS'}")


if __name__ == "__main__":
    main()
