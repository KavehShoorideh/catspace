#!/usr/bin/env python
"""
experiments/viz/wdl_regions.py — does the embedding find win / draw / loss regions?

Kaveh, 2026-07-13: "mix in planner-vs-Stockfish so we see how Stockfish kills and
might even win. I want to see if the representation finds three distinct
win/draw/loss regions."

Takes a bank of positions, labels each by the Syzygy tablebase outcome
(win/draw/loss), embeds F, and asks whether the three outcome classes occupy
distinct regions of embedding space -- visually (2D projections) and
quantitatively (how well a simple classifier recovers W/D/L from F alone).

POV of the label matters:
  --pov stm    : outcome for the side to move (tablebase-native). In an
                 asymmetric material like KRRvKBP this is partly STM-driven
                 (White-to-move usually winning, Black-to-move usually losing),
                 so W-vs-L separation can be read straight off the STM plane.
  --pov white  : outcome for WHITE (the MATE_W reachability frame). Needs the
                 net to encode actual value, not just whose turn it is.
To defuse the STM confound we ALSO report separability on the White-to-move-only
subset (there, stm is constant, so any W/D/L separation is genuine value).

Projections: a nonlinear manifold embedding (UMAP if installed, else t-SNE) that
folds ALL dims into 2 preserving neighbourhoods -- unlike PCA, which keeps only
the top-2 linear axes -- plus LDA (supervised: is there ANY linear separation?).
Metrics: 5-fold kNN accuracy, linear (logistic) accuracy, silhouette. Output is
a PNG + a small HTML wrapper in artifacts/generated/.
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import chess
import chess.syzygy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from catspace.data.encode import board_from_packed
from catspace.nn.features import omega_ids
from experiments.viz.build_embedding_neighbors import Bank, load_bank_positions


def wdl_label(board, tb, pov):
    """+1 win / 0 draw / -1 loss, or None if off-tablebase. `pov` in {stm,white}."""
    try:
        w = tb.probe_wdl(board)
    except (KeyError, chess.syzygy.MissingTableError, ValueError, IndexError):
        return None
    if pov == "white" and board.turn == chess.BLACK:
        w = -w
    return 1 if w > 0 else (-1 if w < 0 else 0)


def separability(F, y):
    """BALANCED 5-fold kNN + linear accuracy (draws are a minority, so plain
    accuracy just rewards guessing the majority) + silhouette. Balanced-accuracy
    chance = 1/n_classes, so any lift over that is genuine class structure."""
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score
    nclass = len(np.unique(y))
    if nclass < 2 or len(y) < 50:
        return dict(knn=float("nan"), linear=float("nan"), silhouette=float("nan"),
                    chance=float("nan"), nclass=nclass)
    ba = "balanced_accuracy"
    out = dict(nclass=nclass, chance=1.0 / nclass)
    out["knn"] = float(cross_val_score(KNeighborsClassifier(15), F, y, cv=5, scoring=ba).mean())
    out["linear"] = float(cross_val_score(
        LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"), F, y, cv=5,
        scoring=ba).mean())
    try:
        out["silhouette"] = float(silhouette_score(F, y))
    except Exception:
        out["silhouette"] = float("nan")
    return out


def _manifold_2d(F, seed=0):
    """Nonlinear 2D embedding that folds all dims in (UMAP if available, else
    t-SNE) -- unlike PCA it doesn't just keep the top-2 linear axes."""
    try:
        import umap
        return umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                         random_state=seed).fit_transform(F), "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        perp = min(30, max(5, len(F) // 100))
        return (TSNE(n_components=2, perplexity=perp, init="pca",
                     random_state=seed).fit_transform(F), "t-SNE")


def scatter_png(F, y, wm_mask, meta_full, meta_wm, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    colors = {1: "#33aa55", 0: "#8b93a3", -1: "#d24b4b"}
    names = {1: "win", 0: "draw", -1: "loss"}
    manifold, mname = _manifold_2d(F)
    try:
        lda = LDA(n_components=2).fit(F, y).transform(F)
    except Exception:
        lda = manifold
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="#0f1115")
    for ax, proj, name in ((axes[0], manifold, f"{mname} (unsupervised, nonlinear)"),
                           (axes[1], lda, "LDA (supervised)")):
        ax.set_facecolor("#0f1115")
        for cls in (1, 0, -1):
            m = y == cls
            if m.any():
                ax.scatter(proj[m, 0], proj[m, 1], s=6, c=colors[cls], alpha=0.45,
                           label=f"{names[cls]} (n={int(m.sum())})", edgecolors="none")
        ax.set_title(name, color="#e6e6e6")
        ax.tick_params(colors="#6b7280"); [s.set_color("#2a2e37") for s in ax.spines.values()]
        leg = ax.legend(loc="upper right", framealpha=0.2, fontsize=9)
        for t in leg.get_texts():
            t.set_color("#e6e6e6")
    fig.suptitle(title, color="#e6e6e6", fontsize=12)
    fig.text(0.5, 0.01, meta_full + "\n" + meta_wm, ha="center", color="#9ec7ff", fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    import io
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, facecolor="#0f1115")
    plt.close(fig)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="data/derived/lichess_fb_4gb_qm_plygap_only.pt")
    ap.add_argument("--bank-shards", nargs="+",
                    default=["data/selfplay/krrkbp_sfsf", "data/selfplay/krrkbp_pvsf"])
    ap.add_argument("--bank-size", type=int, default=12000)
    ap.add_argument("--pov", choices=("stm", "white"), default="stm")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--label", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch  # noqa: F401
    from catspace.nn.fb import load_ckpt, pick_device

    device = pick_device(args.device)
    fb, payload = load_ckpt(Path(args.ckpt), device)
    z = torch.as_tensor(payload["zgoals"]["MATE_W"], dtype=torch.float32, device=device)
    omega = omega_ids(np.array([1800]), np.array([1800]), np.array([float("nan")]))[0]
    tb = chess.syzygy.open_tablebase(args.syzygy_dir)

    rng = np.random.default_rng(args.seed)
    packed, meta = load_bank_positions(args.bank_shards, args.bank_size, rng)
    bank = Bank(fb, omega, z, packed, meta, device)

    # label every position + track which are White-to-move (confound control)
    labels, keep, wm = [], [], []
    for i in range(len(packed)):
        b = board_from_packed(packed[i], meta[i])
        lab = wdl_label(b, tb, args.pov)
        if lab is None:
            continue
        keep.append(i); labels.append(lab); wm.append(b.turn == chess.WHITE)
    tb.close()
    keep = np.array(keep); y = np.array(labels); wm = np.array(wm)
    F = bank.F[keep]
    counts = {int(k): int((y == k).sum()) for k in (1, 0, -1)}
    print(f"{len(y)} labelled positions (pov={args.pov}); win/draw/loss = "
          f"{counts[1]}/{counts[0]}/{counts[-1]}")

    full = separability(F, y)
    # White-to-move-only, labelled White-POV (== stm there): no stm confound
    from catspace.data.encode import board_from_packed as _bfp  # noqa
    ywm = y[wm]
    sub = separability(F[wm], ywm) if wm.sum() >= 50 else {}
    meta_full = (f"all positions ({full['nclass']}-class, balanced-acc chance {full['chance']:.2f}): "
                 f"kNN {full['knn']:.2f} · linear {full['linear']:.2f} · "
                 f"silhouette {full['silhouette']:+.2f}")
    meta_wm = (f"White-to-move only (no STM confound; {sub.get('nclass', 0)}-class, chance "
               f"{sub.get('chance', float('nan')):.2f}): kNN {sub.get('knn', float('nan')):.2f} · "
               f"linear {sub.get('linear', float('nan')):.2f} · "
               f"silhouette {sub.get('silhouette', float('nan')):+.2f} "
               f"(n={int(wm.sum())})") if sub else ""
    print(" ", meta_full); print(" ", meta_wm)

    label = args.label or Path(args.ckpt).stem
    title = f"W/D/L embedding regions — {label} (pov={args.pov})"
    png = scatter_png(F, y, wm, meta_full, meta_wm, title)
    out_png = Path("artifacts/generated") / f"wdl_regions_{label}_{args.pov}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_png.write_bytes(png)
    b64 = base64.b64encode(png).decode()
    out_html = out_png.with_suffix(".html")
    out_html.write_text(
        f"<!doctype html><meta charset=utf-8><title>{title}</title>"
        f"<body style='margin:0;background:#0f1115;color:#e6e6e6;font:14px system-ui'>"
        f"<img style='max-width:100%' src='data:image/png;base64,{b64}'></body>")
    print(f"-> {out_png}\n-> {out_html}")


if __name__ == "__main__":
    main()
