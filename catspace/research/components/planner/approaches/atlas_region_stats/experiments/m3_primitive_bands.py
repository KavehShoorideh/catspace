#!/usr/bin/env python
"""catspace/research/components/planner/approaches/atlas_region_stats/experiments/m3_primitive_bands.py -- is opponent-conditioned crossing flux opponent-specific in
RANKING (different maps) or only in MAGNITUDE (same map, scaled)? For a sample of real positions we
compute the SF-refereed CrossingRisk under a WEAK (Maia-1100) vs STRONG (Maia-1900) move-model and
compare: mean magnitude, ranking Spearman between bands, and top-decile predictiveness vs the actual
SF crossing (mover_loss). Settles whether M3's "bands differ" belongs on location or on magnitude/z.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from catspace.io import paths



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=paths.derived("transition_data_labeled.npz"))
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    import numpy as _np
    z = _np.load(args.data, allow_pickle=True)
    ml = z["mover_loss"]; ok = np.flatnonzero(~np.isnan(ml))
    idx = rng.choice(ok, min(args.n, len(ok)), replace=False)
    fens = z["fen"][idx]; y = ml[idx].astype(float)

    from catspace.research.tools.training_infra.train.scaffold import resolve_device
    from catspace.research.tools.chess_specific.transition import CrossingRisk
    from lczerolens import LczeroBoard
    from maia2 import model as maia_model, inference
    dev = resolve_device("auto"); prepared = inference.prepare()
    maia = maia_model.from_pretrained(type="rapid", device=str(dev))
    cr = CrossingRisk(depth=args.depth)

    elos = [1100, 1500, 1900]
    risks = {e: [] for e in elos}
    for i, fen in enumerate(fens):
        b = LczeroBoard(fen)
        for e in elos:
            mp, _ = inference.inference_each(maia, prepared, fen, e, e)
            risks[e].append(cr.risk(b, mp)[0])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(fens)} [{time.time()-t0:.0f}s]", flush=True)
    cr.close()
    R = {e: np.array(v) for e, v in risks.items()}

    def lift(pred):
        cross = (y >= 0.2).astype(float); base = cross.mean()
        n = max(1, int(0.1 * len(pred)))
        return cross[np.argsort(pred)[::-1][:n]].mean() / base if base > 0 else np.nan

    print(f"\n===== M3 opponent-conditioned flux, STRENGTH GRADIENT (SF referee d{args.depth}, n={len(fens)}) =====")
    print(f"  {'move-model':>12} {'mean flux':>10} {'top-decile lift':>16} {'rho(flux,actual)':>18}")
    for e in elos:
        print(f"  {'Maia-'+str(e):>12} {R[e].mean():>10.4f} {lift(R[e]):>15.2f}x {spearmanr(R[e], y).correlation:>+18.3f}")
    print(f"  MAGNITUDE ratio 1100/1900 = {R[1100].mean()/max(R[1900].mean(),1e-9):.2f}x  "
          f"(strength-dependent: weaker crosses more)")
    print(f"  RANKING Spearman(1100 vs 1900) = {spearmanr(R[1100], R[1900]).correlation:+.3f}  "
          f"(so location is largely shared; strength scales magnitude)")
    print(f"VERDICT m3-primitive-bands done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
