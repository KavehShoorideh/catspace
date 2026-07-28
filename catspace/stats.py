"""catspace/stats.py -- statistical utilities for eval VERDICTs (MILESTONES locked decision 7:
every comparison carries uncertainty; no bare point estimates).

CLUSTER bootstrap: our eval units cluster (same-game pairs share a game; tb rows share games), so
rows are NOT iid -- resample CLUSTERS (games) with replacement, not rows. PAIRED delta: two models
evaluated on the SAME rows share noise; the honest comparison is the bootstrap distribution of
DELTA-rho on shared resamples (tighter than comparing marginal CIs), reported with P(delta>0).

Tested (run `python -m catspace.stats`); no new stat enters a VERDICT without a passing test here
(same discipline as experiments/losses.py).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _cluster_resample_idx(clusters, rng):
    """indices for one cluster-bootstrap resample: sample unique clusters with replacement, take
    all rows of each sampled cluster (with multiplicity)."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    picks = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([rows_of[p] for p in picks])


def spearman_ci(pred, true, clusters=None, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Spearman rho with a percentile bootstrap CI. clusters: per-row cluster id (e.g. game id) ->
    CLUSTER bootstrap; None -> iid row bootstrap. Returns (rho, lo, hi)."""
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    rho = float(spearmanr(pred, true).correlation)
    rng = np.random.default_rng(seed)
    n = len(pred); clusters = np.asarray(clusters) if clusters is not None else None
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = _cluster_resample_idx(clusters, rng) if clusters is not None else rng.integers(0, n, n)
        boots[b] = spearmanr(pred[idx], true[idx]).correlation
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return rho, float(lo), float(hi)


def paired_delta_ci(pred_a, pred_b, true, clusters=None, n_boot: int = 2000, seed: int = 0,
                    alpha: float = 0.05):
    """PAIRED comparison of two models on the SAME rows: bootstrap distribution of
    delta = rho_A - rho_B on shared resamples. Returns (delta, lo, hi, p_a_better) where
    p_a_better = fraction of resamples with delta > 0."""
    pred_a = np.asarray(pred_a, float); pred_b = np.asarray(pred_b, float); true = np.asarray(true, float)
    delta = float(spearmanr(pred_a, true).correlation - spearmanr(pred_b, true).correlation)
    rng = np.random.default_rng(seed)
    n = len(true); clusters = np.asarray(clusters) if clusters is not None else None
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = _cluster_resample_idx(clusters, rng) if clusters is not None else rng.integers(0, n, n)
        boots[b] = (spearmanr(pred_a[idx], true[idx]).correlation
                    - spearmanr(pred_b[idx], true[idx]).correlation)
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return delta, float(lo), float(hi), float(np.mean(boots > 0))


def paired_nll_ci(nll_better, nll_base, clusters=None, n_boot: int = 2000, seed: int = 0,
                  alpha: float = 0.05):
    """PAIRED per-position NLL comparison (M2b z-lift). nll_better / nll_base are per-position
    negative log-likelihoods of the played move under the two models on the SAME positions.
    LIFT = mean(nll_base - nll_better) in nats/move (>0 means `better` model assigns the played
    move higher probability). clusters = per-position player_id -> resample PLAYERS with replacement
    (never positions), so the CI reflects player-level n (the M2a lesson). Returns
    (lift, lo, hi, p_better) where p_better = fraction of resamples with lift > 0."""
    a = np.asarray(nll_better, float); b = np.asarray(nll_base, float)
    diff = b - a                                             # per-position lift
    lift = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    n = len(diff); clusters = np.asarray(clusters) if clusters is not None else None
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = _cluster_resample_idx(clusters, rng) if clusters is not None else rng.integers(0, n, n)
        boots[i] = np.mean(diff[idx])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lift, float(lo), float(hi), float(np.mean(boots > 0))


def fmt_ci(rho, lo, hi):
    return f"{rho:+.3f} [{lo:+.3f},{hi:+.3f}]"


def _tests():
    ok = True

    def check(name, cond):
        nonlocal ok; ok &= bool(cond); print(f"  {'OK ' if cond else 'FAIL'} {name}")

    rng = np.random.default_rng(0)
    n_g, per = 400, 8
    g = np.repeat(np.arange(n_g), per)
    true = rng.normal(size=n_g * per)
    # model A: strong signal; model B: weaker signal; shared game-level noise (clustering)
    game_noise = np.repeat(rng.normal(0, 0.8, n_g), per)
    A = true + 0.5 * rng.normal(size=len(true)) + 0.3 * game_noise
    B = true + 1.2 * rng.normal(size=len(true)) + 0.3 * game_noise

    rho_a, lo_a, hi_a = spearman_ci(A, true, clusters=g, n_boot=400, seed=1)
    check("CI brackets the point estimate", lo_a < rho_a < hi_a)
    check("CI has sane width for n=3200 clustered", 0.005 < (hi_a - lo_a) < 0.2)

    # paired delta detects the real difference
    d, lo, hi, p = paired_delta_ci(A, B, true, clusters=g, n_boot=400, seed=2)
    check("paired delta positive and CI excludes 0", d > 0 and lo > 0)
    check("P(A better) ~ 1", p > 0.99)

    # null case: identical-quality models -> delta CI covers 0
    B2 = true + 0.5 * rng.normal(size=len(true)) + 0.3 * game_noise
    d0, lo0, hi0, p0 = paired_delta_ci(A, B2, true, clusters=g, n_boot=400, seed=3)
    check("null delta CI covers 0", lo0 < 0 < hi0)

    # cluster CI should be WIDER than iid CI when strong game-level noise correlates rows
    Ac = true + 0.2 * rng.normal(size=len(true)) + 1.5 * game_noise
    _, lo_c, hi_c = spearman_ci(Ac, true, clusters=g, n_boot=400, seed=4)
    _, lo_i, hi_i = spearman_ci(Ac, true, clusters=None, n_boot=400, seed=4)
    check("cluster bootstrap wider than iid under game-level noise", (hi_c - lo_c) > (hi_i - lo_i))

    # paired NLL lift: model with lower NLL on the played move -> positive lift, CI excludes 0
    nll_base = -np.log(rng.uniform(0.05, 0.3, size=len(true)))          # base assigns played move modest prob
    nll_z = nll_base - rng.uniform(0.0, 0.1, size=len(true))            # z shaves NLL (lower = better)
    lift, lo_n, hi_n, p_n = paired_nll_ci(nll_z, nll_base, clusters=g, n_boot=400, seed=5)
    check("NLL lift positive and CI excludes 0", lift > 0 and lo_n > 0)
    check("P(z better) ~ 1", p_n > 0.99)
    # null: same NLL distribution -> lift CI covers 0
    nll_z0 = nll_base + rng.uniform(-0.05, 0.05, size=len(true))
    l0v, l0lo, l0hi, _ = paired_nll_ci(nll_z0, nll_base, clusters=g, n_boot=400, seed=6)
    check("null NLL lift CI covers 0", l0lo < 0 < l0hi)

    print("ALL STATS TESTS PASSED" if ok else "STATS TESTS FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _tests() else 1)
