#!/usr/bin/env python
"""catspace/research/tools/embeddings/basin_hazard_areas.py -- turn the hazard SCALAR into named,
lookable-at AREAS: cluster positions in the shared phi coordinate, rank the clusters by how much
hazard they actually carry, and print/draw what each one is.

basin_hazard_field.py answers "how much does human play leak, and where on the tent". It does not
answer "leak doing WHAT", because a point on a (ply, q) chart is not a chess situation. This script
closes that: microstates in phi, ranked, then shown as boards.

THREE CHOICES WORTH STATING.

1. **The clustering coordinate is the SHARED field's phi, not either dynamics-conditioned field's.**
   The human and SF fields are separately trained and their phi spaces are unalignable -- that is
   the whole reason the hazard is defined on probabilities rather than on geometry. Clustering in
   either one would define regions in a coordinate that only one population's field can read. The
   incumbent field trained on BOTH populations is the one chart both are entitled to.

2. **Ranking is hazard x OCCUPANCY, not hazard alone.** A cluster with h = +0.9 that humans reach
   twice a year is a curiosity; one with h = +0.25 that they reach constantly is where the rating
   points are. The headline number is therefore expected leaked score per game,
   sum over cluster of P(human visits) * mean h, and the table reports both factors so a reader
   can see which one is doing the work.

3. **Every cluster gets a bootstrap CI and clusters whose CI straddles zero are marked.** With ~120
   clusters, the most extreme mean h is an order statistic and will look impressive by chance
   alone; the null pair of fields (--null-*) exists to give that noise a measured scale rather than
   an argued one.

Reuses MiniBatchKMeans microstates over phi exactly as engine_vs_human_basins.py does, so "region"
means the same thing in both analyses.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from catspace.research.tools.embeddings.basin_simplex_chart import INK, MUTED

CMAP_HAZARD = "coolwarm"
COLOR_HUMAN, COLOR_SF = "#2a78d6", "#e34948"
PIECE_VAL = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}


def material_signature(fens):
    """Median (white material, black material, total pieces) over a cluster -- the cheapest honest
    description of 'what kind of position is this', and the one used by committor_by_material.py."""
    w, b, npc = [], [], []
    for f in fens:
        board = f.split(" ")[0]
        w.append(sum(PIECE_VAL.get(c.lower(), 0) for c in board if c.isupper()))
        b.append(sum(PIECE_VAL.get(c.lower(), 0) for c in board if c.islower()))
        npc.append(sum(c.isalpha() for c in board))
    return float(np.median(w)), float(np.median(b)), float(np.median(npc))


def boot_ci(x, n=2000, seed=0, lo=2.5, hi=97.5):
    if len(x) < 8:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, lo)), float(np.percentile(m, hi))


def phase_of(ply):
    return "opening" if ply < 24 else ("middlegame" if ply < 60 else "endgame")


def base_rate_residual(h, q, nbins=25):
    """h with the GLOBAL h-vs-q trend removed, so a region is credited only for hazard beyond
    what its evaluation level already implies.

    Why this is not optional. The two populations have very different outcome base rates -- SF-vs-SF
    is 73% draws against 6.5% for lichess -- so the SF-trained committor is pulled toward 0
    everywhere, and h therefore has a large component that is a function of q alone. That component
    is a real and interesting finding about the two dynamics, but it is GLOBAL: it says nothing
    about which regions are hazardous, and left in it would rank regions mostly by how sharp their
    evaluations are. Binning on q and subtracting the bin mean removes exactly that, with no
    functional-form assumption beyond smoothness in q.
    """
    edges = np.quantile(q, np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    b = np.clip(np.digitize(q, edges[1:-1]), 0, nbins - 1)
    mean = np.bincount(b, h, nbins) / np.maximum(np.bincount(b, minlength=nbins), 1)
    return h - mean[b]


def simple_features(fen, ply, qh):
    """Interpretable per-position features: the ones a player could actually be told about."""
    w, b, npc, pawns, queens = [], [], [], [], []
    for f in fen:
        board = str(f).split(" ")[0]
        w.append(sum(PIECE_VAL.get(c.lower(), 0) for c in board if c.isupper()))
        b.append(sum(PIECE_VAL.get(c.lower(), 0) for c in board if c.islower()))
        npc.append(sum(c.isalpha() for c in board))
        pawns.append(board.count("p") + board.count("P"))
        queens.append(board.count("q") + board.count("Q"))
    w, b = np.array(w, float), np.array(b, float)
    return np.column_stack([w, b, w - b, np.array(npc, float), np.array(pawns, float),
                            np.array(queens, float), ply.astype(float), qh])


FEATURE_NAMES = ["mat_W", "mat_B", "mat_diff", "n_pieces", "n_pawns", "n_queens", "ply", "q_human"]


def hazard_rules(simple, h, gid, seed, max_depth=4, min_leaf=500, test_frac=0.3, cols=None):
    """Hazard AREAS as readable rules, fitted on train games and scored on held-out ones.

    A shallow regression tree over the interpretable features is the right shape for this once
    localisability has shown that (a) h is largely a function of those features and (b) it is NOT a
    function of the shared phi. A leaf is a region stated in terms a player can check -- material,
    pawn count, ply, how good the position looks to a human -- and its held-out mean h is how much
    score that region actually leaks. Every number reported is on games the tree never saw, so leaf
    means are not the fitted values.
    """
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(seed)
    games = np.unique(gid)
    test = set(rng.choice(games, max(1, int(len(games) * test_frac)), replace=False).tolist())
    te = np.array([int(g) in test for g in gid])
    tr = ~te
    cols = list(range(simple.shape[1])) if cols is None else list(cols)
    X = simple[:, cols]
    names = [FEATURE_NAMES[c] for c in cols]
    t = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=seed)
    t.fit(X[tr], h[tr])
    leaf_te = t.apply(X[te])
    h_te = h[te]

    tree, rules = t.tree_, {}

    def walk(node, conds):
        if tree.children_left[node] == -1:
            rules[node] = list(conds)
            return
        f, thr = names[tree.feature[node]], tree.threshold[node]
        walk(tree.children_left[node], conds + [f"{f} <= {thr:.2f}"])
        walk(tree.children_right[node], conds + [f"{f} > {thr:.2f}"])

    walk(0, [])
    out = []
    for node, conds in rules.items():
        m = leaf_te == node
        if m.sum() < 30:
            continue
        lo, hi = boot_ci(h_te[m], seed=seed + node)
        out.append(dict(n=int(m.sum()), occ=m.mean(), mean_h=float(h_te[m].mean()),
                        ci_lo=lo, ci_hi=hi, rule=" AND ".join(conds)))
    out.sort(key=lambda r: -r["mean_h"])
    return out, int(tr.sum()), int(te.sum())


def localisability(phi, simple, h, gid, seed, test_frac=0.3):
    """Is h a FUNCTION OF THE POSITION at all, and in which description?

    The ranked table below can only mean something if hazard actually varies BETWEEN regions rather
    than within them. This measures that directly, as held-out R^2 (split BY GAME, since positions
    within a game are near-duplicates) for predicting h from:
      * simple interpretable features -- material, piece counts, ply, the human evaluation;
      * the shared field's 64-d phi;
      * both.
    An R^2 near zero in every description means h is real but NOT localisable in these coordinates:
    the hazard would then not be a property of where the position IS, and no clustering of this phi
    can produce honest 'areas'. Reported before the table so the table is read correctly.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(seed)
    games = np.unique(gid)
    test = set(rng.choice(games, max(1, int(len(games) * test_frac)), replace=False).tolist())
    te = np.array([int(g) in test for g in gid])
    tr = ~te
    base = ((h[te] - h[tr].mean()) ** 2).sum()
    out = {}
    for name, X in [("material+ply+q_human", simple), ("shared phi (64-d)", phi),
                    ("both", np.hstack([simple, phi]))]:
        m = HistGradientBoostingRegressor(max_iter=200, random_state=seed).fit(X[tr], h[tr])
        out[name] = 1.0 - ((h[te] - m.predict(X[te])) ** 2).sum() / base
    return out, int(tr.sum()), int(te.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="artifacts/experiments/basin_hazard_data.npz")
    ap.add_argument("--k", type=int, default=120, help="microstates (engine_vs_human_basins uses 120)")
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--min-count", type=int, default=40, help="minimum HUMAN positions in a cluster")
    ap.add_argument("--top", type=int, default=12, help="clusters to describe in full")
    ap.add_argument("--boards", type=int, default=3, help="example boards per described cluster")
    ap.add_argument("--null-h", default="", help="optional npz from a NULL field pair; its |h| "
                                                 "distribution is the noise floor for this one")
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_hazard_areas")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import MiniBatchKMeans

    z = np.load(args.data)
    m = z["ply"] >= args.min_ply
    phi = z["phi"][m].astype(np.float32)
    h = (z["q_sf"] - z["q_human"])[m]
    qh, qs = z["q_human"][m], z["q_sf"][m]
    ply, src, fen, res = z["ply"][m], z["source"][m], z["fen"][m], z["result"][m]
    hum = src == 0
    hres = np.zeros_like(h)
    hres[hum] = base_rate_residual(h[hum], qh[hum])
    print(f"[areas] {len(phi):,} positions ({int(hum.sum()):,} human) | k={args.k} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"  global base-rate component: mean h {h[hum].mean():+.4f}, and h residualised on "
          f"q_human has mean {hres[hum].mean():+.4f} with sd {hres[hum].std():.4f} "
          f"(vs raw sd {h[hum].std():.4f}) -- the ratio is how much of h is REGIONAL")

    gid = z["gid"][m]
    simple = simple_features(fen[hum], ply[hum], qh[hum])
    r2, ntr, nte = localisability(phi[hum], simple, h[hum], gid[hum], args.seed)
    print(f"\nLOCALISABILITY of h -- held-out R^2, split BY GAME ({ntr:,} train / {nte:,} test rows)")
    for k_, v in r2.items():
        print(f"  {k_:>22s}  R^2 {v:>+7.4f}")
    print("  Near-zero everywhere = h is real but NOT a function of where the position is in these\n"
          "  coordinates, and no clustering of this phi can yield honest 'areas'.")

    rules, n_tr, n_te = hazard_rules(simple, h[hum], gid[hum], args.seed)
    print(f"\nHAZARD AREAS AS RULES -- depth-4 tree on the interpretable features, every number "
          f"scored on HELD-OUT games ({n_tr:,} train / {n_te:,} test rows)")
    print(f"  {'occ%':>6s} {'n':>7s} {'mean h':>8s} {'95% CI':>19s}   rule")
    for r in rules[:8]:
        print(f"  {100*r['occ']:>5.2f}% {r['n']:>7,} {r['mean_h']:>+8.3f} "
              f"[{r['ci_lo']:>+7.3f},{r['ci_hi']:>+7.3f}]   {r['rule']}")
    print("  ... swindle end (most negative):")
    for r in rules[-4:]:
        print(f"  {100*r['occ']:>5.2f}% {r['n']:>7,} {r['mean_h']:>+8.3f} "
              f"[{r['ci_lo']:>+7.3f},{r['ci_hi']:>+7.3f}]   {r['rule']}")

    # The tree above splits mainly on q_human, which is largely the SCALE difference between the
    # two fields (measured slope 0.63) rather than position-specific hazard. Re-fit on the
    # base-rate RESIDUAL with q_human excluded from the features: this asks the sharper question --
    # GIVEN how good the position looks to a human, which KINDS of position leak extra?
    rres, _, _ = hazard_rules(simple, hres[hum], gid[hum], args.seed,
                              cols=[i for i, n in enumerate(FEATURE_NAMES) if n != "q_human"])
    print(f"\nEXTRA HAZARD BEYOND THE EVALUATION LEVEL -- same tree on the q_human-residualised h, "
          f"with q_human removed from the features (held-out)")
    print(f"  {'occ%':>6s} {'n':>7s} {'resid h':>8s} {'95% CI':>19s}   rule")
    for r in rres[:6]:
        print(f"  {100*r['occ']:>5.2f}% {r['n']:>7,} {r['mean_h']:>+8.3f} "
              f"[{r['ci_lo']:>+7.3f},{r['ci_hi']:>+7.3f}]   {r['rule']}")
    print("  ... swindle end:")
    for r in rres[-3:]:
        print(f"  {100*r['occ']:>5.2f}% {r['n']:>7,} {r['mean_h']:>+8.3f} "
              f"[{r['ci_lo']:>+7.3f},{r['ci_hi']:>+7.3f}]   {r['rule']}")

    km = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, n_init=10, batch_size=4096)
    lab = km.fit_predict(phi)
    bet = np.array([h[hum][lab[hum] == c].mean() for c in range(args.k)
                    if (lab[hum] == c).sum() > 0])
    wts = np.array([(lab[hum] == c).sum() for c in range(args.k) if (lab[hum] == c).sum() > 0])
    bvar = float(np.average((bet - np.average(bet, weights=wts)) ** 2, weights=wts))
    print(f"\n  k-means variance decomposition: BETWEEN-cluster variance of h {bvar:.5f} of total "
          f"{h[hum].var():.5f} = {100*bvar/h[hum].var():.2f}% "
          f"(the rest is WITHIN clusters, i.e. not captured by region identity)")

    n_human_tot = int(hum.sum())
    rows = []
    for c in range(args.k):
        cm = lab == c
        ch = cm & hum
        nh = int(ch.sum())
        if nh < args.min_count:
            continue
        occ = nh / n_human_tot                              # P(a human position lands here)
        mh = float(h[ch].mean())
        lo, hi = boot_ci(h[ch], seed=args.seed + c)
        w, b, npc = material_signature(fen[ch])
        rlo, rhi = boot_ci(hres[ch], seed=args.seed + 7919 + c)
        rows.append(dict(c=c, n_human=nh, n_sf=int((cm & ~hum).sum()), occ=occ, mean_h=mh,
                         ci_lo=lo, ci_hi=hi, leak=occ * mh,
                         resid=float(hres[ch].mean()), r_lo=rlo, r_hi=rhi,
                         q_human=float(qh[ch].mean()), q_sf=float(qs[ch].mean()),
                         med_ply=float(np.median(ply[ch])), mat_w=w, mat_b=b, n_pieces=npc,
                         phase=Counter(phase_of(p) for p in ply[ch]).most_common(1)[0][0],
                         dec=float((res[ch] != 0).mean())))
    rows.sort(key=lambda r: -r["leak"])
    kept = len(rows)
    tot_leak = sum(r["leak"] for r in rows)
    print(f"  {kept} of {args.k} clusters have >= {args.min_count} human positions; they hold "
          f"{100*sum(r['occ'] for r in rows):.1f}% of human occupancy")
    print(f"  EXPECTED LEAKED SCORE per human position (sum occ x mean h over kept clusters): "
          f"{tot_leak:+.4f}")

    if args.null_h:
        zn = np.load(args.null_h)
        hn = (zn["q_sf"] - zn["q_human"])[zn["ply"] >= args.min_ply]
        print(f"  NULL floor (|h| from a same-data field pair): median {np.median(np.abs(hn)):.4f} "
              f"| p95 {np.percentile(np.abs(hn), 95):.4f}   vs THIS pair median "
              f"{np.median(np.abs(h[hum])):.4f} | p95 {np.percentile(np.abs(h[hum]), 95):.4f}")

    hdr = (f"\n  {'#':>3s} {'n_hum':>6s} {'occ%':>6s} {'mean h':>8s} {'95% CI':>17s} "
           f"{'resid':>8s} {'leak':>8s} {'q_H':>6s} {'q_SF':>6s} {'ply':>5s} {'mat W/B':>9s} "
           f"{'pcs':>4s} {'phase':>11s} {'dec%':>5s}")
    print("\nHAZARD AREAS ranked by leaked score (occupancy x mean h); + = humans give it away")
    print(hdr)
    for i, r in enumerate(rows[:args.top]):
        star = " " if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "~"     # ~ = CI straddles zero
        rstar = " " if (r['r_lo'] > 0 or r['r_hi'] < 0) else "~"
        print(f"  {i:>3d} {r['n_human']:>6,} {100*r['occ']:>5.2f}% {r['mean_h']:>+8.3f}{star}"
              f"[{r['ci_lo']:>+6.3f},{r['ci_hi']:>+6.3f}] {r['resid']:>+7.3f}{rstar} "
              f"{r['leak']:>+8.4f} "
              f"{r['q_human']:>+6.2f} {r['q_sf']:>+6.2f} {r['med_ply']:>5.0f} "
              f"{r['mat_w']:>4.0f}/{r['mat_b']:<4.0f} {r['n_pieces']:>4.0f} {r['phase']:>11s} "
              f"{100*r['dec']:>4.0f}%")
    print("\n  (~ marks clusters whose bootstrap CI straddles zero -- with k clusters the extreme "
          "means are order statistics and some will look large by chance)")
    print("\nSWINDLE AREAS -- most NEGATIVE mean h (humans do better here than perfect play concedes)")
    print(hdr)
    for i, r in enumerate(sorted(rows, key=lambda r: r["leak"])[:5]):
        star = " " if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "~"
        rstar = " " if (r['r_lo'] > 0 or r['r_hi'] < 0) else "~"
        print(f"  {i:>3d} {r['n_human']:>6,} {100*r['occ']:>5.2f}% {r['mean_h']:>+8.3f}{star}"
              f"[{r['ci_lo']:>+6.3f},{r['ci_hi']:>+6.3f}] {r['resid']:>+7.3f}{rstar} "
              f"{r['leak']:>+8.4f} "
              f"{r['q_human']:>+6.2f} {r['q_sf']:>+6.2f} {r['med_ply']:>5.0f} "
              f"{r['mat_w']:>4.0f}/{r['mat_b']:<4.0f} {r['n_pieces']:>4.0f} {r['phase']:>11s} "
              f"{100*r['dec']:>4.0f}%")

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    # ---- Figure: the ranked areas -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axes[0]
    occ = np.array([r["occ"] for r in rows]); mh = np.array([r["mean_h"] for r in rows])
    sz = 18 + 900 * np.array([abs(r["leak"]) for r in rows]) / max(abs(tot_leak), 1e-9)
    sc = ax.scatter(100 * occ, mh, s=sz, c=mh, cmap=CMAP_HAZARD,
                    vmin=-np.abs(mh).max(), vmax=np.abs(mh).max(), edgecolor=INK, linewidth=0.5)
    ax.axhline(0, color=MUTED, lw=1, ls=":")
    for i, r in enumerate(rows[:6]):                        # selective direct labels, not all
        ax.annotate(f"{i}", (100 * r["occ"], r["mean_h"]), fontsize=9, color=INK,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("human occupancy (% of human positions)")
    ax.set_ylabel("mean h = q_SF - q_human")
    ax.set_title(f"{kept} supported regions -- area = leaked score", color=INK)
    fig.colorbar(sc, ax=ax, label="mean h")

    ax2 = axes[1]
    top = rows[:args.top]
    y = np.arange(len(top))[::-1]
    ax2.barh(y, [r["mean_h"] for r in top], height=0.62,
             color=[COLOR_SF if r["mean_h"] > 0 else COLOR_HUMAN for r in top])
    ax2.errorbar([r["mean_h"] for r in top], y,
                 xerr=[[r["mean_h"] - r["ci_lo"] for r in top],
                       [r["ci_hi"] - r["mean_h"] for r in top]],
                 fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{i}  {r['phase']}, {r['n_pieces']:.0f}p, ply~{r['med_ply']:.0f}"
                         for i, r in enumerate(top)], fontsize=8)
    ax2.axvline(0, color=MUTED, lw=1, ls=":")
    ax2.set_xlabel("mean h (bars: red = hazard, blue = swindle) with bootstrap 95% CI")
    ax2.set_title("top regions by leaked score", color=INK)
    fig.suptitle("Hazard AREAS -- where human dynamics lose ground the engine keeps")
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}.png", dpi=140)

    # ---- Example boards for the top clusters --------------------------------------------------
    import chess
    import chess.svg
    svg_parts = []
    for i, r in enumerate(rows[:args.top]):
        ch = (lab == r["c"]) & hum
        idx = np.flatnonzero(ch)
        pick = idx[np.argsort(-h[idx])[:args.boards]]       # the worst examples in the cluster
        svg_parts.append(f"<h3>region {i} &mdash; mean h {r['mean_h']:+.3f}, occupancy "
                         f"{100*r['occ']:.2f}%, {r['phase']}, median ply {r['med_ply']:.0f}</h3><div>")
        for j in pick:
            b = chess.Board(str(fen[j]))
            svg_parts.append(f"<figure style='display:inline-block;margin:6px'>"
                             f"{chess.svg.board(b, size=260)}"
                             f"<figcaption style='font:12px sans-serif;color:#555'>"
                             f"h {h[j]:+.2f} &nbsp; q_H {qh[j]:+.2f} &rarr; q_SF {qs[j]:+.2f}"
                             f" &nbsp; ply {ply[j]}</figcaption></figure>")
        svg_parts.append("</div>")
    Path(f"{args.out_prefix}_boards.html").write_text(
        "<meta charset='utf-8'><title>hazard areas</title>"
        "<body style='font:14px sans-serif;max-width:1100px;margin:24px auto'>"
        "<h1>Hazard areas for humans</h1><p>Each region is a microstate of the shared field's "
        "phi. Boards are the highest-h human positions in that region: h = q_SF - q_human, so a "
        "large positive h means the engine keeps what the human is about to lose.</p>"
        + "".join(svg_parts) + "</body>")

    np.savez(f"{args.out_prefix}_data.npz", label=lab, h=h, ply=ply, source=src,
             rank=np.array([r["c"] for r in rows]),
             mean_h=np.array([r["mean_h"] for r in rows]),
             occ=np.array([r["occ"] for r in rows]))
    print(f"\nwrote {args.out_prefix}.png + _boards.html + _data.npz [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
