#!/usr/bin/env python
"""experiments/opening_alphabet.py -- the opening ALPHABET from real game DYNAMICS (Kaposi 2026-07-21,
option A). The IQE+QRL field is a VALUE/reachability field and (correctly) does NOT cluster positions
by opening; the concept structure lives in the DYNAMICS. So build the metastable basins from the
transition matrix of actual games, not from any field:

  microstates = recurring opening positions (keyed by placement + side-to-move + castling -> byte-fast,
                transposition-robust, no per-position board reconstruction);
  transitions = consecutive positions WITHIN real games (game_id + ply);
  macrostates = sklearn SpectralClustering on the symmetrized transition matrix (metastable sets);
  the coarse DIRECTED transition graph over macrostates = literally the alphabet the openings are
  written in ("you linger in a family, transitions between families are rare + directional").

Validation gate: label each microstate by ECO/opening-family (maintained lichess chess-openings DB,
EPD-matched) and measure per-basin purity. Field-free -- this is the dynamics, exactly as described.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from experiments.metastable_macrostates import (build_eco_book, coarse_transition, pos_key,
                                                purity, spectral_macrostates, stationary)


def _resolve_shards(spec):
    p = Path(spec)
    if p.is_dir():
        return sorted(p.glob("shard_*.npz"))
    return sorted(Path().glob(spec)) if any(c in spec for c in "*?[") else [p]


def build_dynamics(shards, max_ply, min_visits, top_k):
    """(P, visits, reps) over the top-K most-visited opening microstates, aggregated across shards.
    P = row-stochastic real-game transition matrix; reps[m] = a (packed, meta) representative for
    board reconstruction / ECO label. game_ids are offset per shard so no false cross-shard edges."""
    paths = _resolve_shards(shards)
    Ps, Ms, plys, gids = [], [], [], []
    for si, path in enumerate(paths):
        nz = np.load(path)
        ply = np.asarray(nz["ply"]).astype(int)
        keep = ply <= max_ply
        Ps.append(np.asarray(nz["packed"])[keep]); Ms.append(np.asarray(nz["meta"])[keep])
        plys.append(ply[keep])
        gids.append(np.asarray(nz["game_id"]).astype(np.int64)[keep] + si * 10_000_000)  # per-shard offset
    P_, M_ = np.concatenate(Ps), np.concatenate(Ms)
    G_ = np.concatenate(gids)
    print(f"[stage] {len(paths)} shard(s), {len(P_)} opening rows (ply<={max_ply})", flush=True)
    # microstate key = 12 piece bitboards (96 bytes) + [stm, K,Q,k,q] castling (5 bytes)
    key = np.concatenate([P_.view(np.uint8), M_[:, :5]], axis=1)
    uniq, first_idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    visits = np.bincount(inv, minlength=len(uniq))
    # transitions: consecutive rows of the SAME game (ply is contiguous within the kept prefix)
    same = G_[:-1] == G_[1:]
    src, dst = inv[:-1][same], inv[1:][same]
    # keep the top-K most-visited microstates (the recurring opening positions)
    order = np.argsort(-visits)
    keep_ms = order[:top_k]
    keep_ms = keep_ms[visits[keep_ms] >= min_visits]
    remap = -np.ones(len(uniq), dtype=int); remap[keep_ms] = np.arange(len(keep_ms))
    ms, md = remap[src], remap[dst]
    ok = (ms >= 0) & (md >= 0)
    T = csr_matrix((np.ones(int(ok.sum())), (ms[ok], md[ok])), shape=(len(keep_ms), len(keep_ms)))
    rs = np.asarray(T.sum(1)).ravel(); rs[rs == 0] = 1.0
    Pm = T.multiply(1.0 / rs[:, None]).tocsr()
    reps = [(P_[first_idx[m]], M_[first_idx[m]]) for m in keep_ms]
    return Pm, visits[keep_ms], reps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, help="lichess shard dir, glob, or a single shard_*.npz")
    ap.add_argument("--eco-dir", default="data/eco")
    ap.add_argument("--max-ply", type=int, default=40, help="opening horizon")
    ap.add_argument("--min-visits", type=int, default=20, help="a microstate must recur >= this many times")
    ap.add_argument("--top-k", type=int, default=3000, help="cap on microstates (most-visited)")
    ap.add_argument("--macrostates", type=int, default=24)
    ap.add_argument("--out", default="artifacts/experiments/opening_alphabet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); m = args.macrostates

    book = build_eco_book(args.eco_dir)
    P, visits, reps = build_dynamics(args.shards, args.max_ply, args.min_visits, args.top_k)
    n = P.shape[0]
    print(f"[stage] {n} recurring opening microstates (ply<={args.max_ply}, >={args.min_visits} visits), "
          f"{int(P.nnz)} transitions ({time.time()-t0:.0f}s)", flush=True)
    if n < m * 2:
        raise SystemExit(f"too few microstates ({n}) for {m} macrostates -- lower --min-visits or raise --max-ply")

    eco, fam = [], []
    for pk, mt in reps:
        hit = book.get(pos_key(board_from_packed(pk, mt)))
        eco.append(hit[0] if hit else None); fam.append(hit[1] if hit else None)
    inbook = sum(x is not None for x in fam)

    pi = stationary(P)
    lab = spectral_macrostates(P, m, args.seed)
    T = coarse_transition(P, lab, pi, m)
    meta_diag = np.diag(T)
    sizes = [int((lab == c).sum()) for c in range(m)]

    print(f"VERDICT OPENING_ALPHABET shards={Path(args.shards).name} microstates={n} m={m} "
          f"in-book={inbook}/{n} ({100*inbook/n:.0f}%)")
    print(f"  metastability (self-transition T_ii): mean={meta_diag.mean():.3f}  "
          f"min={meta_diag.min():.3f} max={meta_diag.max():.3f}")
    off = T - np.diag(meta_diag); asym = np.abs(off - off.T).sum() / (np.abs(off).sum() + 1e-9)
    print(f"  coarse-graph directional asymmetry: {asym:.3f} (0=reversible, 1=fully one-way)")
    print(f"  BASIN PURITY vs opening-family={purity(lab, fam, m):.3f}  vs eco-code={purity(lab, eco, m):.3f}")
    gate = "PASS" if purity(lab, fam, m) >= 0.80 else "below-0.80"
    print(f"  GATE [~90% one opening family]: {gate}")
    print("  --- alphabet (macrostate = most-visited member = prototype opening) ---")
    for c in range(m):
        idx = np.flatnonzero(lab == c)
        if len(idx) == 0:
            continue
        hub = idx[np.argmax(visits[idx])]                       # prototype = most-visited position
        members = [fam[i] for i in idx if fam[i] is not None]
        top = Counter(members).most_common(1)[0] if members else ("out-of-book", 0)
        frac = top[1] / max(len(members), 1)
        # top outgoing edges of this macrostate (the "alphabet transitions")
        succ = [f"M{j}" for j in np.argsort(-T[c])[:3] if j != c and T[c, j] > 0.05]
        print(f"    M{c:2d} (n={sizes[c]:4d}, meta {meta_diag[c]:.2f}): prototype={fam[hub] or 'out-of-book'} "
              f"| plurality '{top[0]}' {100*frac:.0f}% | -> {','.join(succ) if succ else '(absorbing)'}")

    _viz(P, lab, T, meta_diag, fam, visits, args)
    print(f"[done] {time.time()-t0:.0f}s")


def _viz(P, labels, T, meta, fam, visits, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import spectral_embedding
    m = args.macrostates
    try:
        emb = spectral_embedding((P + P.T) * 0.5, n_components=2, random_state=args.seed)
    except Exception:
        emb = np.random.default_rng(args.seed).standard_normal((P.shape[0], 2))
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    for c in range(m):
        s = labels == c
        ax[0].scatter(emb[s, 0], emb[s, 1], s=8, alpha=0.6)
    ax[0].set_title("opening microstates (spectral layout), colored by macrostate")
    cent = np.stack([emb[labels == c].mean(0) if (labels == c).any() else np.zeros(2) for c in range(m)])
    ax[1].scatter(cent[:, 0], cent[:, 1], s=[400 * meta[c] + 80 for c in range(m)], c=range(m), cmap="tab20", zorder=3)
    for a in range(m):
        for b in range(m):
            if a != b and T[a, b] > 0.05:
                ax[1].annotate("", xy=cent[b], xytext=cent[a], zorder=2,
                               arrowprops=dict(arrowstyle="-|>", lw=1 + 6 * T[a, b], alpha=0.5, color="0.3"))
    for c in range(m):
        lab_fam = Counter([fam[i] for i in np.flatnonzero(labels == c) if fam[i]]).most_common(1)
        name = (lab_fam[0][0][:10] if lab_fam else f"M{c}")
        ax[1].text(cent[c, 0], cent[c, 1], name, fontsize=8, fontweight="bold", ha="center", zorder=4)
    ax[1].set_title("coarse DIRECTED transition graph over openings (the alphabet)")
    fig.tight_layout()
    out = Path(args.out).with_suffix(".png"); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  figure -> {out}")


if __name__ == "__main__":
    main()
