#!/usr/bin/env python
"""diagnose_lobes.py -- the UMAP shows THREE lobes. What are they?

Kaveh, looking at the ply explorer: "i see three lobes, but they're not win loss draw. all of them
appear in all. There is some ply-level movement from one side to the other two, but why would there
be three?"

Right question, and it is answerable rather than arguable. Cluster the shared UMAP into k lobes,
then ask which categorical property of the position best PREDICTS lobe membership, scored by
adjusted mutual information (chance-corrected, so a label with many values cannot win by having
many values). Candidates, all read off the tokens and globals -- the same six bytes the model gets:

  side to move        binary -- would give 2 lobes, not 3, but it is the first thing to rule out
  castling rights     THE leading hypothesis. They are IRREVERSIBLE, exactly like material, and
                      the natural coarse states are "both sides can still castle" / "one side has
                      lost the right" / "neither can" -- which is three, and which migrates one way
                      with ply, matching the observed drift from one lobe into the other two.
  en passant          binary and rare
  piece-count band    game phase; also monotone in ply
  ply band            time itself
  population          human vs engine
  eventual outcome    the thing Kaveh already ruled out by eye; included so the ruling-out is
                      quantitative rather than visual

Prints the ranked table plus a lobe x best-label contingency, so the answer is legible rather than
a single number.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=paths.experiment("reach_vit_v1_ply_umap.json"))
    ap.add_argument("--games", type=int, default=200_000)
    ap.add_argument("--per-ply", type=int, default=400)
    ap.add_argument("--max-ply", type=int, default=120)
    ap.add_argument("--n-term", type=int, default=9000)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    D = json.load(open(args.json))
    XY = np.stack([D["x"], D["y"]], 1)

    # Reproduce the EXACT sample the export used (same seed, same order) so the per-point globals
    # line up with the exported coordinates. Any mismatch here would silently scramble the labels,
    # so it is asserted against the exported ply/pc rather than trusted.
    tr = T.build(n_human=args.games // 2, n_sf=args.games // 2, seed=0, max_plies=400, verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), 0)
    keep_games = np.flatnonzero(split == 2)
    game, ply, pc = tr.game_of_row(), tr.ply_of_row(), tr.piece_count()
    in_test = np.isin(game, keep_games)
    rng = np.random.default_rng(0)
    rows = []
    for p in range(0, args.max_ply + 1):
        cand = np.flatnonzero(in_test & (ply == p))
        if len(cand) == 0:
            continue
        rows.append(cand if len(cand) <= args.per_ply
                    else rng.choice(cand, args.per_ply, replace=False))
    rows = np.concatenate(rows)
    t_rows, t_term = tr.terminal_rows()
    t_keep = np.isin(game[t_rows], keep_games) & (ply[t_rows] <= args.max_ply)
    t_rows = t_rows[t_keep]
    if len(t_rows) > args.n_term:
        t_rows = t_rows[rng.choice(len(t_rows), args.n_term, replace=False)]
    rows = np.unique(np.concatenate([rows, t_rows]))
    assert len(rows) == len(XY), f"sample mismatch {len(rows)} vs {len(XY)}"
    assert (ply[rows] == np.array(D["ply"])).all(), "row order does not match the export"
    print(f"[lobes] reproduced the exact {len(rows):,}-position sample")

    g = tr.glob[rows]
    turn = g[:, 0].astype(int)
    w_castle = (g[:, 1] > 0) | (g[:, 2] > 0)
    b_castle = (g[:, 3] > 0) | (g[:, 4] > 0)
    castle3 = np.where(w_castle & b_castle, 0, np.where(w_castle | b_castle, 1, 2))
    labels = {
        "castling rights (both/one/neither)": castle3,
        "castling rights (full 4-bit)": (g[:, 1] > 0) * 8 + (g[:, 2] > 0) * 4
                                        + (g[:, 3] > 0) * 2 + (g[:, 4] > 0),
        "side to move": turn,
        "en passant available": (g[:, 5] > 0).astype(int),
        "piece-count band": np.clip((pc[rows] - 2) // 6, 0, 4),
        "ply band": np.clip(np.array(D["ply"]) // 15, 0, 7),
        "population (human/SF)": np.array(D["src"]),
        "eventual outcome": np.array(D["out"]),
        "arrived terminal": np.array(D["arr"]),
    }

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score as ami
    lobe = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit_predict(XY)
    sizes = np.bincount(lobe, minlength=args.k)
    print(f"[lobes] k={args.k} sizes: {sizes.tolist()}\n")

    print(f"  {'candidate':<38} {'AMI vs lobe':>12}")
    print(f"  {'-'*38} {'-'*12}")
    ranked = sorted(((ami(v, lobe), k) for k, v in labels.items()), reverse=True)
    for score, k in ranked:
        print(f"  {k:<38} {score:>12.4f}")

    best = ranked[0][1]
    v = labels[best]
    uv = np.unique(v)
    print(f"\n  contingency -- lobe x {best} (row %):")
    print("    " + "".join(f"{str(u):>12}" for u in uv))
    for L in range(args.k):
        m = lobe == L
        row = [100 * ((v[m] == u).mean()) for u in uv]
        print(f"  L{L}" + "".join(f"{r:>11.1f}%" for r in row))

    # ply is the thing Kaveh saw moving; show where each lobe sits in time
    P = np.array(D["ply"])
    print(f"\n  lobe ply profile (median, IQR):")
    for L in range(args.k):
        pl = P[lobe == L]
        print(f"  L{L}  median {np.median(pl):>5.0f}   "
              f"IQR {np.percentile(pl,25):.0f}-{np.percentile(pl,75):.0f}   n {len(pl):,}")


if __name__ == "__main__":
    main()
