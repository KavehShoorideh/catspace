#!/usr/bin/env python
"""catspace/research/components/planner/approaches/atlas_region_stats/experiments/metastable_macrostates.py -- the ALPHABET of the field (Kaveh 2026-07-20/21):
openings / standard structures are METASTABLE BASINS in IQE space -- you linger in them, and the
transitions between them are rare and DIRECTIONAL. Pipeline:

  1. embed positions with the (quasimetric) field -> a directed distance matrix D (d(F_i -> B_j),
     asymmetric by construction);
  2. k-NN reachability graph via sklearn.neighbors.kneighbors_graph (metric='precomputed' on D),
     weight exp(-d/sigma) -> row-stochastic transition matrix P;
  3. macrostates via sklearn.cluster.SpectralClustering on the symmetrized affinity (the
     'spectral clustering' path -- Kaveh chose sklearn end-to-end over hand-rolled PCCA+, since
     deeptime/msmtools don't build on Python 3.14);
  4. PROTOTYPE per macrostate = its MEDOID (the actual member minimizing summed intra-cluster
     distance -- centroids are meaningless in a non-metric space). The medoid set is the BASIN
     CODEBOOK;
  5. VALIDATION GATE (lichess): label each member by the ECO code / opening family of its position
     (maintained lichess chess-openings DB, matched by EPD so it is robust to transpositions) and
     measure per-basin purity. ~90% one opening family => the geometry recovered known structure,
     proceed. Mush => stop and reconsider before building anything downstream.
  6. coarse DIRECTED transition graph over macrostates = the alphabet the plans are written in.

Everything numeric is imported (sklearn / scipy); only the coarse-graining and medoid selection are
plain matrix arithmetic (definitional, not an algorithm).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.io import paths

BOARD_ONLY = (18, 19)


# ----------------------------------------------------------------------------- ECO (maintained DB)
def build_eco_book(eco_dir):
    """{position-key -> (eco, family)} for every position along every named opening line in the
    lichess chess-openings TSVs (a-e). Keyed by piece-placement + side-to-move so it matches by
    POSITION (transposition-robust), and covers prefixes so mid-opening positions match, not just
    line-ends. family = name before ':' (e.g. 'Sicilian Defense').

    Collision rule: each line's FINAL (named) position is canonical and wins over prefixes -- so a
    transposition-shared position like '1.e4 e6' gets its named label (French C00), not whichever
    file was processed first."""
    import chess
    named, prefix = {}, {}
    for tsv in sorted(Path(eco_dir).glob("[a-e].tsv")):
        for row in tsv.read_text().splitlines()[1:]:
            parts = row.split("\t")
            if len(parts) < 3:
                continue
            eco, name, pgn = parts[0], parts[1], parts[2]
            val = (eco, name.split(":")[0].strip())
            board, keys = chess.Board(), []
            for san in re.sub(r"\d+\.+", " ", pgn).split():
                try:
                    board.push_san(san)
                except Exception:
                    break
                keys.append(pos_key(board))
            if keys:
                named[keys[-1]] = val                             # canonical named endpoint
                for kk in keys[:-1]:
                    prefix.setdefault(kk, val)                    # prefixes only fill gaps
    return {**prefix, **named}                                    # named endpoints win collisions


def pos_key(board):
    # placement + side-to-move + CASTLING rights (all stored in meta and seen by the 20-plane field,
    # so the ECO label must distinguish them too). En-passant is omitted (transient, flaky in FEN, and
    # rarely defines an opening). Position-based -> still transposition-robust.
    return f"{board.board_fen()} {'w' if board.turn else 'b'} {board.castling_xfen() or '-'}"


# ----------------------------------------------------------------------------- graph / MSM
def directed_knn_P(D, k, sigma):
    """row-stochastic P from a precomputed DIRECTED distance matrix D via sklearn kneighbors_graph.
    sigma<=0 -> self-tune to the median kept-edge distance. Returns (P_sparse, sigma)."""
    finite = D[np.isfinite(D)]
    big = (float(finite.max()) * 10 + 1.0) if finite.size else 1e6
    D = np.nan_to_num(D, nan=big, posinf=big, neginf=big)         # sklearn rejects non-finite
    np.fill_diagonal(D, big)                                       # self = farthest -> never a neighbor
    G = kneighbors_graph(D, n_neighbors=k, metric="precomputed", mode="distance", include_self=False)
    G = G.tocoo()
    if sigma <= 0:
        sigma = float(np.median(G.data)) or 1.0
    W = csr_matrix((np.exp(-G.data / sigma), (G.row, G.col)), shape=D.shape)
    rs = np.asarray(W.sum(1)).ravel(); rs[rs == 0] = 1.0
    return W.multiply(1.0 / rs[:, None]).tocsr(), sigma


def stationary(P, iters=2000, tol=1e-10):
    n = P.shape[0]; pi = np.ones(n) / n; Pt = P.T.tocsr()
    for _ in range(iters):
        nx = Pt @ pi; s = nx.sum(); nx = nx / (s if s else 1.0)
        if np.abs(nx - pi).sum() < tol:
            return nx
        pi = nx
    return pi


def spectral_macrostates(P, m, seed):
    """sklearn SpectralClustering on the symmetrized affinity (undirected reachability)."""
    A = (P + P.T).tocsr(); A.setdiag(0.0); A.eliminate_zeros()
    lab = SpectralClustering(n_clusters=m, affinity="precomputed", assign_labels="discretize",
                             random_state=seed).fit_predict(A)
    return lab


def medoids(D, labels, m):
    """prototype of each macrostate = member minimizing summed SYMMETRIZED distance to its cluster
    (medoid; centroids are meaningless under a quasimetric). Returns global indices."""
    Dsym = D + D.T
    out = []
    for c in range(m):
        idx = np.flatnonzero(labels == c)
        if len(idx) == 0:
            out.append(-1); continue
        sub = Dsym[np.ix_(idx, idx)]
        out.append(int(idx[np.argmin(sub.sum(1))]))
    return out


def coarse_transition(P, labels, pi, m):
    """coarse DIRECTED transition matrix: T_AB = sum_{i in A,j in B} pi_i P_ij / sum_{i in A} pi_i."""
    Pc = P.tocoo(); T = np.zeros((m, m))
    np.add.at(T, (labels[Pc.row], labels[Pc.col]), pi[Pc.row] * Pc.data)
    rs = T.sum(1); rs[rs == 0] = 1.0
    return T / rs[:, None]


def purity(labels, ref, m):
    """mass-weighted mean plurality fraction over macrostates, ignoring members with ref=None."""
    num = den = 0
    for c in range(m):
        vals = [r for r, l in zip(ref, labels) if l == c and r is not None]
        if vals:
            num += Counter(vals).most_common(1)[0][1]; den += len(vals)
    return (num / den) if den else float("nan")


# ----------------------------------------------------------------------------- embed / main
def embed(fb, dev, P, M, om, board_only, block=1024):
    """board_only=True zeros planes 18,19 (the toy-field 'board-only geometry' convention). The
    lichess field is trained on FULL feature_planes, so it must embed with board_only=False."""
    Fs, Bs = [], []
    with torch.no_grad():
        for s in range(0, len(P), block):
            e = min(len(P), s + block)
            pl = feature_planes(P[s:e], M[s:e])
            if board_only:
                pl[:, BOARD_ONLY] = 0.0
            t = torch.from_numpy(pl).to(dev)
            Fs.append(fb.embed_F(t, torch.from_numpy(np.tile(om, (e - s, 1))).to(dev)))
            Bs.append(fb.embed_B(t))
    return torch.cat(Fs), torch.cat(Bs)


def full_distance(fb, F, B, block=1024):
    N = F.shape[0]; D = np.empty((N, N), np.float32)
    with torch.no_grad():
        for s in range(0, N, block):
            e = min(N, s + block)
            D[s:e] = fb.distance_matrix(F[s:e], B).cpu().numpy()
    return D


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default=paths.sep("iqe_nucleus_gn.pt"))
    ap.add_argument("--data", default=paths.derived("stratified_perfect.npz"))
    ap.add_argument("--lichess-shard", default=None, help="a lichess shard_*.npz -> ECO-labelled opening basins")
    ap.add_argument("--eco-dir", default=str(paths.eco_dir()))
    ap.add_argument("--max-ply", type=int, default=30, help="lichess: keep positions with ply<=this (opening)")
    ap.add_argument("--n", type=int, default=3500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=-1.0, help="<=0 = self-tune to median-NN")
    ap.add_argument("--macrostates", type=int, default=10)
    ap.add_argument("--cluster-metric", choices=["quasi", "cosine-F", "cosine-B"], default="quasi",
                    help="graph distance for clustering: 'quasi'=field quasimetric REACHABILITY kNN "
                         "(good for value/planning); 'cosine-F/B'=cosine SIMILARITY of F/B embeddings "
                         "(isolates whether the field encodes opening-similarity vs reachability)")
    ap.add_argument("--out", default=paths.experiment("macrostates"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    m = args.macrostates

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]

    # ---- load positions + reference labels ----
    if args.lichess_shard:
        book = build_eco_book(args.eco_dir)
        print(f"[stage] ECO book: {len(book)} opening positions from lichess chess-openings", flush=True)
        nz = np.load(args.lichess_shard)
        P_all, M_all, PLY = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["ply"]).astype(int)
        pool = np.flatnonzero(PLY <= args.max_ply)
        order = rng.permutation(len(pool)); seen = 0
        Pl, Ml, eco, fam = [], [], [], []
        for j in order:                                          # keep IN-BOOK opening positions only
            i = pool[j]; seen += 1
            hit = book.get(pos_key(board_from_packed(P_all[i], M_all[i])))
            if hit is None:
                continue
            Pl.append(P_all[i]); Ml.append(M_all[i]); eco.append(hit[0]); fam.append(hit[1])
            if len(Pl) >= args.n:
                break
        Pk, Mk = np.stack(Pl), np.stack(Ml)
        ref1, r1name = fam, "opening-family"
        ref2, r2name = eco, "eco-code"
        print(f"[stage] {len(Pk)} in-book opening positions (ply<={args.max_ply}; "
              f"{100*len(Pk)/max(seen,1):.0f}% of scanned were in-book), "
              f"{len(set(fam))} families / {len(set(eco))} eco-codes", flush=True)
    else:
        nz = np.load(args.data, allow_pickle=True)
        P_all, M_all = np.asarray(nz["packed"]), np.asarray(nz["meta"])
        idx = rng.permutation(len(P_all))[:args.n]
        Pk, Mk = P_all[idx], M_all[idx]
        ref1 = np.asarray(nz["pcount"]).astype(int)[idx].tolist(); r1name = "piece-count"
        ref2 = np.asarray(nz["matid"]).astype(int)[idx].tolist(); r2name = "material"
    print(f"[stage] field={Path(args.field).stem} n={len(Pk)} k={args.k} m={m}", flush=True)

    # ---- embed -> distances -> graph -> macrostates ----
    board_only = args.lichess_shard is None           # toy fields zero planes 18,19; lichess uses full planes
    F, B = embed(fb, dev, Pk, Mk, om, board_only)
    if args.cluster_metric == "quasi":
        D = full_distance(fb, F, B)                   # field quasimetric reachability (directed)
    else:                                             # cosine SIMILARITY of embeddings (symmetric)
        E = F if args.cluster_metric == "cosine-F" else B
        En = torch.nn.functional.normalize(E, dim=1).cpu().numpy()
        D = np.clip(1.0 - En @ En.T, 0.0, None).astype(np.float32)   # clip float noise (cos slightly >1)
        np.fill_diagonal(D, 0.0)
    P, sigma = directed_knn_P(D, args.k, args.sigma)
    pi = stationary(P)
    lab = spectral_macrostates(P, m, args.seed)
    med = medoids(D, lab, m)
    T = coarse_transition(P, lab, pi, m)
    meta = np.diag(T)
    print(f"[stage] transition matrix + macrostates (sigma={sigma:.2f}) ({time.time()-t0:.0f}s)", flush=True)

    # ---- verdicts + the gate ----
    sizes = [int((lab == c).sum()) for c in range(m)]
    print(f"VERDICT MACROSTATES field={Path(args.field).stem} n={len(Pk)} m={m} sizes={sizes}")
    print(f"  metastability (self-transition T_ii): {np.round(meta,3).tolist()} mean={meta.mean():.3f}")
    off = T - np.diag(meta); asym = np.abs(off - off.T).sum() / (np.abs(off).sum() + 1e-9)
    print(f"  coarse-graph directional asymmetry: {asym:.3f} (0=reversible, 1=fully one-way)")
    p1, p2 = purity(lab, ref1, m), purity(lab, ref2, m)
    print(f"  BASIN PURITY vs {r1name}={p1:.3f}   vs {r2name}={p2:.3f}")
    if args.lichess_shard:
        gate = "PASS (>=0.80 -> geometry recovered opening structure)" if p1 >= 0.80 else \
               "FAIL (<0.80 -> mush; reconsider before downstream)"
        print(f"  GATE [~90% one opening family]: {gate}")
        print("  --- basin codebook (medoids) ---")
        for c in range(m):
            if med[c] < 0:
                continue
            b = board_from_packed(Pk[med[c]], Mk[med[c]]); hit = book.get(pos_key(b))
            members = [ref1[i] for i in np.flatnonzero(lab == c) if ref1[i] is not None]
            top = Counter(members).most_common(1)[0] if members else ("-", 0)
            frac = top[1] / max(len(members), 1)
            print(f"    M{c} (n={sizes[c]:4d}, meta {meta[c]:.2f}): medoid={hit[1] if hit else 'out-of-book'} "
                  f"| plurality '{top[0]}' {100*frac:.0f}%")

    _visualize(F, lab, T, meta, ref1, r1name, args)
    print(f"[done] {time.time()-t0:.0f}s")


def _visualize(F, labels, T, meta, ref, refname, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    m = args.macrostates
    Fn = torch.nn.functional.normalize(F, dim=1).cpu().numpy()
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=args.seed).fit_transform(Fn)
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    for c in range(m):
        s = labels == c
        ax[0].scatter(emb[s, 0], emb[s, 1], s=6, alpha=0.5, label=f"M{c}")
    ax[0].set_title("microstates in field t-SNE, colored by macrostate"); ax[0].legend(fontsize=7, markerscale=2)
    cent = np.stack([emb[labels == c].mean(0) if (labels == c).any() else np.zeros(2) for c in range(m)])
    ax[1].scatter(cent[:, 0], cent[:, 1], s=[400 * meta[c] + 80 for c in range(m)], c=range(m), cmap="tab10", zorder=3)
    for a in range(m):
        for b in range(m):
            if a != b and T[a, b] > 0.06:
                ax[1].annotate("", xy=cent[b], xytext=cent[a], zorder=2,
                               arrowprops=dict(arrowstyle="-|>", lw=1 + 6 * T[a, b], alpha=0.6, color="0.3"))
    for c in range(m):
        ax[1].text(cent[c, 0], cent[c, 1], f"M{c}", fontsize=11, fontweight="bold", ha="center", zorder=4)
    ax[1].set_title("coarse DIRECTED transition graph over macrostates (the alphabet)")
    cats = [x for x in dict.fromkeys([r for r in ref if r is not None])][:12]
    comp = np.array([[sum(1 for i in np.flatnonzero(labels == c) if ref[i] == p) for p in cats]
                     for c in range(m)], float)
    comp = comp / np.clip(comp.sum(1, keepdims=True), 1e-9, None)
    bottom = np.zeros(m)
    for j, p in enumerate(cats):
        ax[2].bar(range(m), comp[:, j], bottom=bottom, label=str(p)[:18]); bottom += comp[:, j]
    ax[2].set_xticks(range(m)); ax[2].set_xticklabels([f"M{c}" for c in range(m)])
    ax[2].set_title(f"macrostate composition by {refname}"); ax[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    out = Path(args.out).with_suffix(".png"); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  figure -> {out}")


if __name__ == "__main__":
    main()
