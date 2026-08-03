#!/usr/bin/env python
"""catspace/research/tools/viz/viz_b_mate_clusters.py -- Kaveh 2026-07-22: "I want to see clusters in B
(and approach directions). I wanna see if king rook mates cluster together."

Harvest REAL mate positions across the KRRvKBP tree's materials (dtm==1 positions from
dtm_endgame.npz, mating move pushed), label each by (material, checker piece, mated-king
location), plus tablebase-optimal APPROACH trajectories (last ~8 plies into mate). Embed
all with the field's B tower (BOARD_ONLY zeroed), t-SNE to 2D, draw approach arrows, and
mark the learned MATE_W goal vector.

The transfer question, quantified in raw B-space (before t-SNE): do rook-delivered
edge-mates from DIFFERENT materials sit closer to each other than random mate pairs do?
cohesion < 1 = the cornering PATTERN clusters across material (a field-native concept
home); cohesion ~ 1 with high material silhouette = the field files mates by MATERIAL,
not by pattern.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import matplotlib
import numpy as np
import torch
from catspace.io import paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.ladder_mate import random_krrvk
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import TB, tb_best_move

BOARD_ONLY = (18, 19)


def mate_labels(b: chess.Board):
    """(material, pattern, king_loc) for a checkmate board (black mated). Pattern is the
    mate GEOMETRY -- the meaningful contrast in this tree, where ~95% of mates are rook
    edge-mates: 'ladder' = a second rook holds the inner line (the two-rook lawnmower);
    'ksupport' = the white king stands in near-opposition guarding the escape squares
    (the KRvK pattern); 'other' = neither."""
    mat = "".join(sorted(p.symbol() for p in b.piece_map().values()))
    bk = b.king(chess.BLACK); wk = b.king(chess.WHITE)
    f, r = chess.square_file(bk), chess.square_rank(bk)
    corner = (f in (0, 7)) and (r in (0, 7))
    loc = "corner" if corner else ("edge" if (f in (0, 7) or r in (0, 7)) else "center")
    checkers = list(b.checkers())
    chk_rooks = [s for s in checkers if b.piece_type_at(s) == chess.ROOK]
    pattern = "other"
    if chk_rooks and loc != "center":
        # the edge line the king sits on + its inner neighbour line
        if r in (0, 7):
            inner = 1 if r == 0 else 6
            others = [s for s in b.pieces(chess.ROOK, chess.WHITE) if s not in chk_rooks]
            if any(chess.square_rank(s) == inner for s in others):
                pattern = "ladder"
        if f in (0, 7) and pattern == "other":
            inner = 1 if f == 0 else 6
            others = [s for s in b.pieces(chess.ROOK, chess.WHITE) if s not in chk_rooks]
            if any(chess.square_file(s) == inner for s in others):
                pattern = "ladder"
        if pattern == "other" and chess.square_distance(wk, bk) <= 2:
            pattern = "ksupport"
    return mat, pattern, loc


def harvest_mates(dtm_npz, cap, rng):
    dz = np.load(dtm_npz)
    P, M, dtm = np.asarray(dz["packed"]), np.asarray(dz["meta"]), np.asarray(dz["dtm"])
    idx = np.flatnonzero(dtm == 1)
    rng.shuffle(idx)
    boards, labels = [], []
    for i in idx:
        b = board_from_packed(P[i], M[i])
        for m in b.legal_moves:
            c = b.copy(stack=False); c.push(m)
            if c.is_checkmate():
                boards.append(c); labels.append(mate_labels(c))
                break
        if len(boards) >= cap:
            break
    return boards, labels


def harvest_approaches(tb, rng, n_games, tail=8):
    """tb-optimal playouts to mate; return list of trajectories (each = last `tail`
    boards ending in the mate board)."""
    trajs = []
    # half KRRvK ladders, half KRRvKBP full-tree conversions
    import json
    fens = json.loads(Path(paths.experiment("krrkbp_test_n200.json")).read_text())["fens"]
    starts = [random_krrvk(rng, central=True) for _ in range(n_games // 2)]
    starts += [chess.Board(f) for f in rng.choice(fens, n_games - len(starts), replace=False)]
    for s in starts:
        if s is None:
            continue
        b = s.copy(stack=False); seen = set(); hist = []
        for _ in range(200):
            if b.is_game_over(claim_draw=True):
                break
            m = tb_best_move(b, tb, seen)
            if m is None:
                break
            seen.add(b.board_fen()); b.push(m); hist.append(b.copy(stack=False))
        if b.is_checkmate():
            trajs.append(hist[-tail:])
    return trajs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", nargs="+", default=[paths.sep("iqe_geom_field.pt"),
                                                    paths.sep("nucleus_distilled.pt")])
    ap.add_argument("--n-mates", type=int, default=1200)
    ap.add_argument("--n-games", type=int, default=40)
    ap.add_argument("--perplexity", type=float, default=40.0)
    ap.add_argument("--out", default=paths.experiment("b_mate_clusters.png"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)

    mates, labels = harvest_mates(paths.derived("dtm_endgame.npz"), args.n_mates, rng)
    tb = TB(str(paths.syzygy_dir()))
    trajs = harvest_approaches(tb, rng, args.n_games)
    tb.close()
    mats = np.array([l[0] for l in labels]); pats = np.array([l[1] for l in labels])
    print(f"[harvest] {len(mates)} mates across {len(set(mats))} materials; patterns: "
          f"{ {p: int((pats == p).sum()) for p in np.unique(pats)} }; "
          f"{len(trajs)} approach trajectories  [{time.time()-t0:.0f}s]", flush=True)

    fig, axes = plt.subplots(1, len(args.fields), figsize=(11 * len(args.fields), 10))
    axes = np.atleast_1d(axes)
    for ax, fpath in zip(axes, args.fields):
        fb, pay = load_ckpt(Path(fpath), dev); fb.eval()

        def embB(boards):
            pk = np.stack([encode_packed(b) for b in boards]); mt = np.stack([encode_meta(b) for b in boards])
            out = []
            for s in range(0, len(pk), 1024):
                pl = feature_planes(pk[s:s + 1024], mt[s:s + 1024]); pl[:, BOARD_ONLY] = 0.0
                with torch.no_grad():
                    out.append(fb.embed_B(torch.from_numpy(pl).to(dev)).cpu().numpy())
            return np.concatenate(out)

        Bm = embB(mates)
        traj_B = [embB(t) for t in trajs if len(t) >= 2]
        _zw = pay["zgoals"]["MATE_W"]
        zW = (_zw.detach().float().numpy() if torch.is_tensor(_zw) else np.asarray(_zw, np.float32))

        # ---- cohesion in RAW B-space: same-PATTERN pairs across DIFFERENT materials,
        # normalized by all cross-mate pairs. <1 = the mate GEOMETRY clusters across
        # material (a field-native concept home); ~1 = pattern invisible to B.
        def med_dist(idx_a, idx_b, need_diff_mat, n=4000):
            ds = []
            for _ in range(n * 3):
                i, j = idx_a[rng.integers(len(idx_a))], idx_b[rng.integers(len(idx_b))]
                if i == j or (need_diff_mat and mats[i] == mats[j]):
                    continue
                ds.append(np.linalg.norm(Bm[i] - Bm[j]))
                if len(ds) >= n:
                    break
            return float(np.median(ds)) if ds else float("nan")

        alli = np.arange(len(Bm))
        base = med_dist(alli, alli, need_diff_mat=False)
        coh = {}
        for p in ("ladder", "ksupport"):
            pi = np.flatnonzero(pats == p)
            if len(pi) >= 20 and len(set(mats[pi])) >= 2:
                coh[p] = med_dist(pi, pi, need_diff_mat=True) / base
        from sklearn.metrics import silhouette_score
        top_mats = [m for m, c in zip(*np.unique(mats, return_counts=True)) if c >= 30]
        sel = np.isin(mats, top_mats)
        sil_mat = float(silhouette_score(Bm[sel], mats[sel])) if len(top_mats) >= 2 else float("nan")
        lk = np.isin(pats, ("ladder", "ksupport"))
        sil_pat = float(silhouette_score(Bm[lk], pats[lk])) if lk.sum() >= 40 else float("nan")
        name = Path(fpath).stem
        print(f"VERDICT B_MATE_CLUSTERS field={name}  cross-material same-pattern cohesion: "
              + "  ".join(f"{p} {v:.2f}" for p, v in coh.items())
              + f"  (<1 = pattern clusters across material)  silhouette pattern {sil_pat:+.2f} "
              f"vs material {sil_mat:+.2f}", flush=True)

        # ---- what DOES B organize mates by? correlate pairwise B-dist with board factors
        from scipy.stats import spearmanr
        bks = np.array([b.king(chess.BLACK) for b in mates])
        wks = np.array([b.king(chess.WHITE) for b in mates])
        ii = rng.integers(0, len(Bm), 4000); jj = rng.integers(0, len(Bm), 4000)
        ok_ = ii != jj; ii, jj = ii[ok_], jj[ok_]
        bd = np.linalg.norm(Bm[ii] - Bm[jj], axis=1)
        bk_d = np.array([chess.square_distance(a, c) for a, c in zip(bks[ii], bks[jj])])
        wk_d = np.array([chess.square_distance(a, c) for a, c in zip(wks[ii], wks[jj])])
        mat_d = (mats[ii] != mats[jj]).astype(float)
        pat_d = (pats[ii] != pats[jj]).astype(float)
        print(f"  [what-B-encodes] spearman(B-dist, factor): black-king dist {spearmanr(bd, bk_d).correlation:+.2f}  "
              f"white-king dist {spearmanr(bd, wk_d).correlation:+.2f}  "
              f"diff-material {spearmanr(bd, mat_d).correlation:+.2f}  "
              f"diff-pattern {spearmanr(bd, pat_d).correlation:+.2f}", flush=True)

        # ---- t-SNE on mates + trajectory points + zW
        from sklearn.manifold import TSNE
        stack = [Bm] + traj_B + [zW[None, :]]
        X = np.concatenate(stack, 0)
        XY = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
                  early_exaggeration=1.6, random_state=args.seed).fit_transform(X)
        xy_m = XY[:len(Bm)]
        off = len(Bm); xy_t = []
        for tB in traj_B:
            xy_t.append(XY[off:off + len(tB)]); off += len(tB)
        xy_z = XY[-1]

        # color = material (top 8), marker = mate pattern (o ladder / ^ ksupport / s other)
        uniq, cnt = np.unique(mats, return_counts=True)
        top8 = list(uniq[np.argsort(-cnt)][:8])
        cmap = plt.get_cmap("tab10")
        mstyle = {"ladder": "o", "ksupport": "^", "other": "s"}
        for k, mmat in enumerate(top8 + ["_rest"]):
            m_ = ~np.isin(mats, top8) if mmat == "_rest" else (mats == mmat)
            if not m_.any():
                continue
            color = "0.7" if mmat == "_rest" else cmap(k)
            lbl = f"other-mat ({m_.sum()})" if mmat == "_rest" else f"{mmat} ({m_.sum()})"
            first = True
            for p, mk in mstyle.items():
                pm = m_ & (pats == p)
                if pm.any():
                    ax.scatter(xy_m[pm, 0], xy_m[pm, 1], s=22 if p != "other" else 12,
                               marker=mk, color=color, alpha=0.65, linewidths=0,
                               label=lbl if first else None)
                    first = False
        for p, mk in mstyle.items():   # marker legend entries (in black)
            ax.scatter([], [], marker=mk, color="black",
                       label=f"{p} ({int((pats == p).sum())})")
        for xy in xy_t:                                     # approach arrows
            for a, b in zip(xy[:-1], xy[1:]):
                ax.annotate("", xy=b, xytext=a,
                            arrowprops=dict(arrowstyle="->", color="crimson", alpha=0.45, lw=1.0))
        ax.scatter(*xy_z, marker="*", s=420, color="gold", edgecolors="black", zorder=5,
                   label="learned MATE_W goal")
        coh_s = " ".join(f"{p} {v:.2f}" for p, v in coh.items())
        ax.set_title(f"B-space mates -- {name}\ncross-material cohesion: {coh_s} | "
                     f"sil pattern {sil_pat:+.2f} / material {sil_mat:+.2f}")
        ax.legend(loc="best", fontsize=8, markerscale=1.4)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Mate clusters in B (t-SNE) -- color=material, marker=mate geometry "
                 "(o ladder / ^ king-support / s other), red arrows=tb-optimal approach (last 8 plies)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"saved {args.out}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
