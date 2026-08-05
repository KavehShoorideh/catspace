#!/usr/bin/env python
"""calibrate_conformal.py -- turn the JEPA's score into an eps-level guarantee, using POSITIVES ONLY.

Split conformal. Score a set of held-out pairs that are KNOWN reachable (the calibration split), and
rank a query against them:

    p(a,b) = (1 + #{calibration scores <= s(a,b)}) / (n + 1)

Under exchangeability with the calibration positives, p is a valid p-value for the null "this pair
is reachable". Declaring IMPOSSIBLE when p <= eps therefore has a false-"impossible" rate <= eps --
distribution-free, finite-sample, and requiring NO negative class, which is the only reason this is
available at all given the approach trains on positives only.

WHAT IS CHECKED HERE, in order of how much it can embarrass us:

  1. VALIDITY. On the TEST split -- positives, disjoint by game from calibration -- the realised
     rate of IMPOSSIBLE verdicts must come in at or under eps. This is the guarantee actually
     holding rather than being asserted. It is checked at several eps, not one.

  2. MONDRIAN VALIDITY, under a taxonomy that introduces NO chess. Pooled coverage routinely hides a
     badly miscalibrated bucket, and a search never queries the pooled distribution -- it queries
     whatever bucket its position sits in. But the taxonomy cannot be material or ply (Kaveh
     2026-08-05: "I don't want to bucket on ply or material count because I'm worried that will
     effectively create strata") -- calibrating per material band would install the very
     stratification this approach claims to discover. The taxonomy is therefore the MODEL'S OWN
     predicted region volume; material/ply survive only as a diagnostic readout.
     Worst-bucket is compared against a NULL: it is a max over bins and so exceeds eps on sampling
     noise alone even when every bin is perfectly calibrated.

  3. POWER. The rate at which cross-game pairs are flagged IMPOSSIBLE. Validity alone is trivially
     achievable by never flagging anything; power is what makes the filter worth having. Both
     numbers must be read together.

Writes the calibration scores so ReachPredicate can be constructed without recomputing them.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net, piece_counts, score_pairs)
from catspace.research.components.encoder.approaches.reach_probability.src.probability_less_than import (
    ReachPredicate)

EPSS = (0.001, 0.005, 0.01, 0.05, 0.10)


def diagnostic_buckets(pc_a, ply_a):
    """Material x ply bands -- DIAGNOSTIC READOUT ONLY, never a calibration taxonomy.

    Kaveh 2026-08-05: "I don't want to bucket on ply or material count because I'm worried that will
    effectively create strata." Correct, and it would have been circular: calibrating per material
    band installs the material stratification INTO the method, and this approach then claims to have
    discovered that stratification from data. So material and ply obey the same rule piece count
    obeys in interpret_reach.py -- they may LABEL an analysis, they may never be an INPUT to the
    method. These bands are printed to show where coverage varies; no threshold is ever derived
    from them."""
    return np.clip((pc_a.astype(int) - 2) // 6, 0, 4) * 5 + np.clip(ply_a.astype(int) // 30, 0, 4)


def learned_buckets(vol_cal, vol_q, n_bins=5):
    """Mondrian taxonomy built from the MODEL'S OWN predicted region volume (sum of log sigma at the
    source position) -- how uncertain the model says it is, not anything about chess.

    Legitimate where material bands are not: it introduces no external structure, it is available at
    query time without a board, and it targets the failure that actually matters -- coverage drifting
    between positions the model is confident about and positions it is not. Edges come from the
    CALIBRATION distribution so the query mapping is fixed before any query is seen."""
    edges = np.quantile(vol_cal, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, vol_cal), np.searchsorted(edges, vol_q)


@torch.no_grad()
def region_volume(net, feats, idx, device, batch=8192):
    """(n,) sum of log sigma over dimensions -- the log-volume of the region the model predicts from
    each SOURCE position. The model's own statement of how uncertain it is about the future."""
    out = []
    for s in range(0, len(idx), batch):
        f = torch.from_numpy(feats[idx[s:s + batch]]).to(device, torch.float32)
        _, ls = net.predict(net.encode(f))
        out.append(ls.sum(-1).float().cpu().numpy())
    return np.concatenate(out)


def mondrian_p(s_q, b_q, s_cal, b_cal, min_n=200):
    """Conformal p-values computed WITHIN each bucket.

    Pooled calibration guarantees only MARGINAL coverage: it can be exactly valid overall while
    being 3-4x over in a particular material band, which is what v1 measured. A search does not
    query the pooled distribution, it queries whatever bucket the position sits in, so the threshold
    has to be per bucket. Buckets too thin to support a quantile fall back to pooled -- an honest
    fallback, reported rather than hidden.
    """
    p = np.empty(len(s_q))
    pooled = np.sort(s_cal)
    fell_back = 0
    for b in np.unique(b_q):
        m = b_q == b
        ref = np.sort(s_cal[b_cal == b])
        if len(ref) < min_n:
            ref = pooled
            fell_back += int(m.sum())
        p[m] = (1.0 + np.searchsorted(ref, s_q[m], side="right")) / (len(ref) + 1.0)
    return p, fell_back


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=paths.experiment("reach_jepa_v1_latest.pt"))
    ap.add_argument("--pairs", default=paths.derived("reach_pairs_v1.npz"))
    ap.add_argument("--data", default=paths.derived("field_std_v2.npz"))
    ap.add_argument("--feats", default=paths.derived("trunk_feats/t1-256x10__field_std_v2.npy"))
    ap.add_argument("--pc-cache", default=paths.derived("field_std_v2_piececount.npz"))
    ap.add_argument("--rows", type=int, default=131072)
    ap.add_argument("--n-cross", type=int, default=100_000, help="cross-game pairs, for POWER")
    ap.add_argument("--n-bins", type=int, default=5, help="Mondrian bins over predicted region volume")
    ap.add_argument("--gap-band", choices=["all", "out_hist"], default="out_hist",
                    help="out_hist = ply gap > 8 only, where b's lc0 history cannot contain a")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=paths.experiment("reach_conformal_v1.npz"))
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    net, payload = load_net(args.ckpt, args.device)
    feats = np.ascontiguousarray(np.load(args.feats, mmap_mode="r")[:args.rows])
    zd = np.load(args.data, allow_pickle=True)
    game_all, ply_all = zd["game"], zd["ply"]
    pc = piece_counts(args.data, args.pc_cache)

    zp = np.load(args.pairs, allow_pickle=True)
    base = (zp["i"] < args.rows) & (zp["j"] < args.rows)
    if args.gap_band == "out_hist":
        base &= zp["gap"] > 8
    cal = base & (zp["split"] == 1)
    tst = base & (zp["split"] == 2)

    s_cal = score_pairs(net, feats, zp["i"][cal], zp["j"][cal], args.device)
    s_tst = score_pairs(net, feats, zp["i"][tst], zp["j"][tst], args.device)
    print(f"[conformal] ckpt step {payload.get('step')} | band {args.gap_band} | "
          f"cal {len(s_cal):,} | test {len(s_tst):,} [{time.time()-t0:.0f}s]", flush=True)

    pred = ReachPredicate(net, s_cal, device=args.device)
    p_tst = pred.p_value(s_tst)

    # cross-game pairs for POWER (ply-matched so the flag cannot be driven by time alone)
    n = args.rows
    order = np.argsort(ply_all[:n], kind="stable")
    ps = ply_all[:n][order]
    ia = rng.integers(0, n, args.n_cross)
    lo = np.searchsorted(ps, ply_all[ia] - 5, "left")
    hi = np.searchsorted(ps, ply_all[ia] + 5, "right")
    ib = order[np.clip((lo + rng.random(args.n_cross) * np.maximum(hi - lo, 1)).astype(np.int64),
                       0, n - 1)]
    ok = game_all[ia] != game_all[ib]
    ia, ib = ia[ok], ib[ok]
    p_cross = pred.p_value(score_pairs(net, feats, ia, ib, args.device))

    # Mondrian taxonomy from the model's OWN predicted region volume. No chess quantity enters.
    vol_cal = region_volume(net, feats, zp["i"][cal], args.device)
    vol_tst = region_volume(net, feats, zp["i"][tst], args.device)
    vol_crs = region_volume(net, feats, ia, args.device)
    b_cal, b_tst = learned_buckets(vol_cal, vol_tst, args.n_bins)
    _, b_crs = learned_buckets(vol_cal, vol_crs, args.n_bins)
    s_crs = score_pairs(net, feats, ia, ib, args.device)
    p_tst_m, fb = mondrian_p(s_tst, b_tst, s_cal, b_cal)
    p_crs_m, _ = mondrian_p(s_crs, b_crs, s_cal, b_cal)
    print(f"[conformal] mondrian taxonomy = predicted region volume, {len(np.unique(b_cal))} bins | "
          f"{fb:,}/{len(s_tst):,} queries fell back to pooled (thin bin)", flush=True)

    # DIAGNOSTIC ONLY -- material/ply bands are read, never calibrated on (see diagnostic_buckets).
    d_tst = diagnostic_buckets(pc[zp["i"][tst]], ply_all[zp["i"][tst]])

    def worst(p, b, eps):
        return max((float((p[b == k] <= eps).mean()) for k in np.unique(b)
                    if (b == k).sum() > 200), default=float("nan"))

    def worst_null(b, eps, n_sim=2000, rs=None):
        """The worst-bucket rate EXPECTED under exact validity (p95).

        worst-bucket is a MAX over ~18 buckets, so it exceeds eps from sampling noise alone even
        when every bucket is perfectly calibrated. Comparing a raw max against eps would manufacture
        a miscalibration finding out of multiple comparisons; this is the null that decides whether
        the excess is real."""
        rs = rs or np.random.default_rng(0)
        ns = np.array([(b == k).sum() for k in np.unique(b) if (b == k).sum() > 200])
        if not len(ns):
            return float("nan")
        draws = rs.binomial(ns[None, :], eps, size=(n_sim, len(ns))) / ns[None, :]
        return float(np.percentile(draws.max(1), 95))

    print(f"\n  taxonomy = predicted region volume (learned). Material/ply columns are DIAGNOSTIC.")
    print(f"  {'':>7} {'---- POOLED ----':^17} {'--- MONDRIAN(learned) ---':^26} {'diagnostic':>11}")
    print(f"  {'eps':>7} {'validity':>8} {'power':>7} {'validity':>9} {'worst':>7} "
          f"{'null p95':>9} {'power':>7} {'worst mat/ply':>14}")
    rows_out = []
    for eps in EPSS:
        v, pw = float((p_tst <= eps).mean()), float((p_cross <= eps).mean())
        vm, wm, pwm = (float((p_tst_m <= eps).mean()), worst(p_tst_m, b_tst, eps),
                       float((p_crs_m <= eps).mean()))
        wn = worst_null(b_tst, eps)
        wd = worst(p_tst_m, d_tst, eps)                 # diagnostic: material/ply bands
        rows_out.append((eps, v, pw, vm, wm, wn, pwm, wd))
        print(f"  {eps:>7.3f} {v:>8.4f} {pw:>7.4f} {vm:>9.4f} {wm:>7.4f} {wn:>9.4f} {pwm:>7.4f} "
              f"{wd:>14.4f}" + ("" if wm <= wn else "  <-- over null"))

    np.savez_compressed(args.out, cal_scores=s_cal, test_scores=s_tst, cal_buckets=b_cal,
                        test_p=p_tst, cross_p=p_cross, test_p_mondrian=p_tst_m,
                        cross_p_mondrian=p_crs_m,
                        _meta=np.array([repr(dict(ckpt=args.ckpt, step=payload.get("step"),
                                                  band=args.gap_band, n_cal=len(s_cal)))]))
    v01, p01, vm01, wm01, wn01, pm01, wd01 = {r[0]: r[1:] for r in rows_out}[0.01]
    print(f"\nVERDICT REACH-CONFORMAL band={args.gap_band} n_cal={len(s_cal)} "
          f"pooled_validity@0.01={v01:.4f} pooled_power@0.01={p01:.4f} "
          f"mondrian_validity@0.01={vm01:.4f} mondrian_worst@0.01={wm01:.4f} "
          f"worst_null_p95@0.01={wn01:.4f} mondrian_power@0.01={pm01:.4f} "
          f"diag_worst_matply@0.01={wd01:.4f} step={payload.get('step')} [{time.time()-t0:.0f}s]")
    print("  validity must be <= eps. Worst bucket is a MAX over bins and exceeds eps on noise alone")
    print("  even when perfectly calibrated, so only an excess over the NULL p95 is real.")
    print("  The taxonomy is the model's own predicted region volume -- no chess quantity is used to")
    print("  calibrate anything; material/ply appear as a diagnostic readout only.")


if __name__ == "__main__":
    main()
