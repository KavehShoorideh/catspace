#!/usr/bin/env python
"""build_reach_pairs.py -- POSITIVE-ONLY reachable pairs for the reach_probability JEPA.

A pair (a, b) is a POSITIVE iff b was actually observed later in the same game as a. There is no
negative class anywhere in this approach (Kaveh 2026-08-05: "I don't want negatives"), so this
script emits reachable pairs and nothing else; the "b is not reachable" side is supplied at query
time by a conformal threshold calibrated on held-out positives, never by a trained-against label.

POINTERS, NOT DATA. A pair is two row indices into the source npz, 8 bytes. Trunk features are
gathered from the existing fp16 memmap at train time. This is build_combined_field_data.py's
(source, local_row) trick one level up -- materialising the planes for both sides of ~600k pairs
would cost ~86GB to say what two int32s already say.

PLY GAP IS REPORTED, NOT FILTERED (Kaveh 2026-08-05: "don't worry about the 8 plus lc0 trunk, I'm
ok with knowing history in an actual game"). lc0's 112-plane input carries EIGHT PLIES OF HISTORY,
so for a pair whose plies are within 8 of each other, position a is literally present inside b's own
input tensor. That is not cheating -- in a real game you do know your own history, and a is in fact
a genuine ancestor of b -- so those pairs are kept and trained on.

What it does mean is that for gap <= 8 the answer is READABLE OFF THE INPUT, and a model can be
right there while having learned nothing about irreversibility. Since the headline question is
whether strata appear without anything chess-specific being programmed, every downstream metric is
stratified by gap band, and the strata verdict is read on the gap > 8 band where the history planes
cannot have supplied the answer. Hence `gap` is stored per pair and --min-gap defaults to 1.

SPLICING IS DEFERRED (Kaveh 2026-08-05: "let's skip the splicing for now and only do it if we need
more data"). When it returns, the join must check threefold repetition and the 50-move clock across
the seam: g1[:p] + g2[p:] can revisit a position g1's prefix already saw, or carry a different
halfmove clock, so the spliced continuation may be terminated before it ever reaches b -- in which
case the pair is NOT reachable by that path and must not be emitted. Position identity for a splice
point must come from chess.polyglot.zobrist_hash, NOT from the planes: en passant is verifiably not
encoded anywhere in the lc0 112-plane input, so a plane-derived key would happily splice two
positions that differ in whether a capture is available.

THREE-WAY SPLIT, BY GAME. Conformal calibration needs a set disjoint from training, and checking
that the resulting guarantee actually holds needs a third set disjoint from calibration. Games never
straddle a split, so no pair shares a game with a pair in another split.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from catspace.io import paths


def ordered_pairs(game: np.ndarray, ply: np.ndarray, min_gap: int):
    """All (i, j) with same game and ply[j] - ply[i] >= min_gap. Returns (i, j, gap).

    Rows are sorted by (game, ply) first so each game is a contiguous block and the within-game
    pairing is a small dense triangle -- 5 rows per game in field_std_v2, so this stays trivial.
    """
    order = np.lexsort((ply, game))
    g, p = game[order], ply[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    ii, jj = [], []
    for s, e in zip(starts, ends):
        n = e - s
        if n < 2:
            continue
        a, b = np.triu_indices(n, 1)                      # a < b, and ply is sorted so ply[b] >= ply[a]
        keep = (p[s + b] - p[s + a]) >= min_gap
        if not keep.any():
            continue
        ii.append(order[s + a[keep]])
        jj.append(order[s + b[keep]])
    i = np.concatenate(ii).astype(np.int32)
    j = np.concatenate(jj).astype(np.int32)
    return i, j, (ply[j] - ply[i]).astype(np.int32)


def split_by_game(games_of_pair: np.ndarray, fracs, seed: int) -> np.ndarray:
    """0 = train, 1 = calibration, 2 = test. Assigned per GAME, so no game straddles a split."""
    uniq = np.unique(games_of_pair)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_tr = int(len(uniq) * fracs[0])
    n_ca = int(len(uniq) * fracs[1])
    lab = np.empty(len(uniq), np.int8)
    lab[:n_tr] = 0
    lab[n_tr:n_tr + n_ca] = 1
    lab[n_tr + n_ca:] = 2
    # uniq was shuffled, so sort it back before searchsorted and carry the labels along.
    order = np.argsort(uniq)
    return lab[order[np.searchsorted(uniq[order], games_of_pair)]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=paths.derived("field_std_v2.npz"),
                    help="STANDARD-schema position npz (needs game/ply); planes are never read")
    ap.add_argument("--min-gap", type=int, default=1,
                    help="minimum ply gap. Default 1 keeps everything; the gap column is stored so "
                         "reporting can stratify at the 8-ply lc0 history boundary instead")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--cal-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=paths.derived("reach_pairs_v1.npz"))
    args = ap.parse_args()

    t0 = time.time()
    z = np.load(args.data, allow_pickle=True)             # npz is lazy: planes never touched
    game, ply = z["game"], z["ply"]
    print(f"[pairs] {len(game):,} positions | {len(np.unique(game)):,} games "
          f"[{time.time()-t0:.0f}s]", flush=True)

    i, j, gap = ordered_pairs(game, ply, args.min_gap)
    gp = game[i]
    assert (game[j] == gp).all(), "pair spans two games"
    assert (gap >= args.min_gap).all(), "min-gap violated"
    split = split_by_game(gp, (args.train_frac, args.cal_frac), args.seed)

    np.savez_compressed(args.out, i=i, j=j, gap=gap, game=gp, split=split,
                        _meta=np.array([repr(dict(data=args.data, min_gap=args.min_gap,
                                                  seed=args.seed, n=len(i)))]))

    n_tr, n_ca, n_te = [(split == k).sum() for k in (0, 1, 2)]
    in_hist = int((gap <= 8).sum())                       # readable off b's lc0 history planes
    out_hist = int((gap > 8).sum())                       # the band the strata verdict is read on
    print(f"\n  pairs                 {len(i):,}")
    print(f"  ply gap  p10/50/90    {np.percentile(gap,[10,50,90]).astype(int).tolist()}")
    print(f"  gap <= 8 (in history) {in_hist:,}  ({in_hist/len(i):.1%})")
    print(f"  gap >  8 (out)        {out_hist:,}  ({out_hist/len(i):.1%})  <- strata verdict band")
    print(f"  train / cal / test    {n_tr:,} / {n_ca:,} / {n_te:,}")
    print(f"  games   train/cal/test {len(np.unique(gp[split==0])):,} / "
          f"{len(np.unique(gp[split==1])):,} / {len(np.unique(gp[split==2])):,}")
    print(f"\nVERDICT REACH-PAIRS n={len(i)} min_gap={args.min_gap} "
          f"in_hist={in_hist} out_hist={out_hist} "
          f"train={n_tr} cal={n_ca} test={n_te} out={args.out} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
