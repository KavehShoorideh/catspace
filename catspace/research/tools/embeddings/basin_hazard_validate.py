#!/usr/bin/env python
"""catspace/research/tools/embeddings/basin_hazard_validate.py -- does h = q_SF - q_human measure a
DYNAMICS difference, or two independently-trained models disagreeing?

Everything here is a real-vs-NULL comparison. The null pair is two fields trained on DISJOINT
halves of the SAME human data (train_iqe_head --game-mod 2:0 / 2:1, different seeds), scored on the
same games. It differs from the real pair in training data and initialisation but NOT in dynamics,
so any quantity that looks the same for both is training noise, whatever it looks like alone.

RETRACTED (2026-08-05). An earlier version of CHECK 1 stratified on |q_SF| and asked whether the
high-h quartile had worse realized results. It passed 8/8 strata with large effects, and it was
CIRCULAR: at fixed q_SF, h is an affine function of -q_human, and the measured within-stratum
correlation between h and -q_human is +0.89 to +1.00. So it was testing whether the human-trained
field predicts human game outcomes -- which is what that field is fit to do -- and not whether h
locates hazards. It is replaced by CHECK 1 below, which asks the incremental question against the
null, and no conclusion should be drawn from the retracted numbers.

CHECK 1 -- INCREMENTAL VALUE. Does knowing the SF field's evaluation improve prediction of the
REALIZED human result beyond the human field's own evaluation? Held-out R^2, split by game, for
predicting the white-POV result from q_human alone vs from both. The null pair gets the identical
treatment: if adding ANY second field's opinion buys the same amount, the gain is not about
dynamics.

CHECK 2 -- MAGNITUDE. |h| against |h_null|. If the ratio is ~1 the raw magnitude of h is noise.

CHECK 3 -- STRUCTURE. The magnitude can be noise-dominated while the SYSTEMATIC part is real, so
this is the one that matters: how much of h is a predictable function of interpretable position
features (held-out R^2, by game), for the real pair vs the null? Two fields differing only by data
half also disagree more where data is thin -- which is itself structured by material and ply -- so
this comparison, not the real pair's R^2 alone, is the evidence.

CHECK 4 -- CALIBRATION PARITY. A systematic confidence gap would masquerade as hazard everywhere.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

PIECE_VAL = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}


def load(path, min_ply, human_only=True):
    z = np.load(path)
    m = z["ply"] >= min_ply
    if human_only:
        m &= z["source"] == 0
    return {k: z[k][m] for k in ("q_human", "q_sf", "ply", "result", "gid", "fen", "source")}


def features(d):
    """Interpretable per-position features + material difference."""
    w, b, npc, pawns = [], [], [], []
    for f in d["fen"]:
        board = str(f).split(" ")[0]
        w.append(sum(PIECE_VAL.get(c.lower(), 0) for c in board if c.isupper()))
        b.append(sum(PIECE_VAL.get(c.lower(), 0) for c in board if c.islower()))
        npc.append(sum(c.isalpha() for c in board))
        pawns.append(board.count("p") + board.count("P"))
    w, b = np.array(w, float), np.array(b, float)
    return np.column_stack([w, b, w - b, np.array(npc, float), np.array(pawns, float),
                            d["ply"].astype(float)]), (w - b)


def split_by_game(gid, seed, test_frac=0.3):
    rng = np.random.default_rng(seed)
    games = np.unique(gid)
    test = set(rng.choice(games, max(1, int(len(games) * test_frac)), replace=False).tolist())
    te = np.array([int(g) in test for g in gid])
    return ~te, te


def held_out_r2(X, y, tr, te, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=200, random_state=seed).fit(X[tr], y[tr])
    return 1.0 - ((y[te] - m.predict(X[te])) ** 2).sum() / ((y[te] - y[tr].mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="artifacts/experiments/basin_hazard_data.npz")
    ap.add_argument("--null-data", default="artifacts/experiments/basin_hazard_null_data.npz")
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    real = load(args.data, args.min_ply)
    null = load(args.null_data, args.min_ply) if args.null_data else None
    Xr, matr = features(real)
    hr = real["q_sf"] - real["q_human"]
    tr, te = split_by_game(real["gid"], args.seed)
    print(f"[validate] real {len(hr):,} human positions | "
          f"null {len(null['q_sf']) if null else 0:,} | ply >= {args.min_ply} "
          f"| {int(tr.sum()):,} train / {int(te.sum()):,} test rows, split BY GAME")
    if null is not None:
        Xn, matn = features(null)
        hn = null["q_sf"] - null["q_human"]
        trn, ten = split_by_game(null["gid"], args.seed)

    # ---- CHECK 1 -------------------------------------------------------------------------------
    y = real["result"].astype(float)
    base = held_out_r2(real["q_human"][:, None], y, tr, te, args.seed)
    both = held_out_r2(np.column_stack([real["q_human"], real["q_sf"]]), y, tr, te, args.seed)
    print(f"\nCHECK 1 -- INCREMENTAL VALUE for predicting the REALIZED human result (held-out R^2)")
    print(f"  {'pair':>6s} {'first field alone':>18s} {'+ second field':>15s} {'gain':>8s}")
    print(f"  {'real':>6s} {base:>18.4f} {both:>15.4f} {both-base:>+8.4f}")
    if null is not None:
        yn = null["result"].astype(float)
        bn = held_out_r2(null["q_human"][:, None], yn, trn, ten, args.seed)
        on = held_out_r2(np.column_stack([null["q_human"], null["q_sf"]]), yn, trn, ten, args.seed)
        print(f"  {'null':>6s} {bn:>18.4f} {on:>15.4f} {on-bn:>+8.4f}")
        print("  The real gain counts only insofar as it EXCEEDS the null gain: adding a second "
              "field\n  trained on different data helps a little regardless of its dynamics.")

    # ---- CHECK 2 -------------------------------------------------------------------------------
    print("\nCHECK 2 -- MAGNITUDE")
    print(f"  real |h|: median {np.median(np.abs(hr)):.4f}  mean {np.abs(hr).mean():.4f}  "
          f"p95 {np.percentile(np.abs(hr), 95):.4f}")
    if null is not None:
        print(f"  null |h|: median {np.median(np.abs(hn)):.4f}  mean {np.abs(hn).mean():.4f}  "
              f"p95 {np.percentile(np.abs(hn), 95):.4f}")
        print(f"  ratio of medians {np.median(np.abs(hr))/max(np.median(np.abs(hn)),1e-9):.2f}x "
              f"-- at ~1 the raw magnitude of h is training noise, not dynamics.")

    # ---- CHECK 3 (the one that matters) --------------------------------------------------------
    print("\nCHECK 3 -- STRUCTURE: how much of h is a predictable function of the position?")
    r2r = held_out_r2(Xr, hr, tr, te, args.seed)
    print(f"  {'pair':>6s} {'R^2 of h from material+ply':>28s}")
    print(f"  {'real':>6s} {r2r:>28.4f}")
    if null is not None:
        r2n = held_out_r2(Xn, hn, trn, ten, args.seed)
        print(f"  {'null':>6s} {r2n:>28.4f}")
        print(f"  A real R^2 well above the null's means the two dynamics disagree SYSTEMATICALLY, "
              f"in a\n  way two same-dynamics fields do not -- even where the per-position "
              f"magnitude is noisy.")

    # material antisymmetry: the specific structure claimed
    print(f"\n  mean h by material difference (the claimed structure), real vs null:")
    print(f"  {'mat diff':>10s} {'real':>9s} {'null':>9s} {'n':>9s}")
    for lo, hi in [(-99, -5), (-5, -2), (-2, 2), (2, 5), (5, 99)]:
        mr = (matr >= lo) & (matr < hi)
        row = f"  {lo if lo>-99 else '-inf':>5}..{hi if hi<99 else 'inf':<4} {hr[mr].mean():>+9.4f}"
        if null is not None:
            mn = (matn >= lo) & (matn < hi)
            row += f" {hn[mn].mean():>+9.4f}"
        print(row + f" {int(mr.sum()):>9,}")

    # ---- CHECK 4 -------------------------------------------------------------------------------
    print("\nCHECK 4 -- CALIBRATION PARITY")
    print(f"  real  mean|q| first {np.abs(real['q_human']).mean():.4f} second "
          f"{np.abs(real['q_sf']).mean():.4f} | corr {np.corrcoef(real['q_human'], real['q_sf'])[0,1]:+.4f}"
          f" | slope {np.polyfit(real['q_human'], real['q_sf'], 1)[0]:+.4f}")
    if null is not None:
        print(f"  null  mean|q| first {np.abs(null['q_human']).mean():.4f} second "
              f"{np.abs(null['q_sf']).mean():.4f} | corr "
              f"{np.corrcoef(null['q_human'], null['q_sf'])[0,1]:+.4f} | slope "
              f"{np.polyfit(null['q_human'], null['q_sf'], 1)[0]:+.4f}")
    print(f"\ndone [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
