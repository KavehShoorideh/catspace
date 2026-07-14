#!/usr/bin/env python
"""
experiments/viz/near_mate_regions.py — do NEAR-MATE positions separate in F-space?

Kaveh: "visualize a few near-mate positions (4-ply away) -- near mate_W, near
mate_B, and draw -- and see where their embeddings fall; I hope the regions are
clearly separated."

These are the EXTREME, clearest cases: the position 4 plies before a checkmate
White delivered (near mate_W = White winning decisively), 4 plies before a
checkmate Black delivered (near mate_B = White losing decisively), and 4 plies
before a drawn game's end (near draw). If F encodes who-is-winning at all, these
three should fall in clearly separated regions; if not, that is the cleanest
possible refutation. Harvested from real game shards (labelled by GAME RESULT, so
material-agnostic), embedded with F, projected (UMAP + LDA), separability scored.
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from catspace.data.encode import board_from_packed
from catspace.nn.features import feature_planes, omega_ids
from experiments.viz.wdl_regions import separability, _manifold_2d


def harvest_near_mate(shard_dirs, per_class, back=4):
    """result -> list of (packed_row, meta_row). mate_W/mate_B require the game to
    END in checkmate; draw uses the drawn game's end. Position taken `back` plies
    before the final."""
    out = {1: [], 0: [], -1: []}
    for d in shard_dirs:
        for path in sorted(Path(d).glob("shard_*.npz")):
            z = np.load(path)
            gid, result, packed, meta = z["game_id"], z["result"], z["packed"], z["meta"]
            ends = np.flatnonzero(np.r_[np.diff(gid) != 0, True])
            starts = np.r_[0, ends[:-1] + 1]
            for s, e in zip(starts, ends):
                r = int(result[e])
                if len(out[r]) >= per_class:
                    continue
                if e - s < back:                       # game too short
                    continue
                if r != 0 and not board_from_packed(packed[e], meta[e]).is_checkmate():
                    continue                            # decisive-by-mate only for the mate classes
                ni = e - back
                out[r].append((packed[ni].copy(), meta[ni].copy()))
            if all(len(v) >= per_class for v in out.values()):
                return out
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/lichess_fb_4gb_qm_plygap_only.pt")
    ap.add_argument("--shards", nargs="+",
                    default=["data/shards/lichess_db_standard_rated_2019-01.prefix1gb"])
    ap.add_argument("--forced-set", default=None,
                    help="use a VALIDATED forced-mate set JSON (mate_W/mate_B/draw) instead of "
                         "harvesting by game-result -- these are proven forced mates (Kaveh)")
    ap.add_argument("--per-class", type=int, default=600)
    ap.add_argument("--back", type=int, default=4, help="plies before the final position")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--label", default="near_mate")
    args = ap.parse_args()

    import torch  # noqa: F401
    from catspace.nn.fb import load_ckpt, pick_device

    dev = pick_device(args.device)
    fb, pay = load_ckpt(Path(args.ckpt), dev)
    def _z(x):
        return x.to(dev).float() if torch.is_tensor(x) else torch.as_tensor(
            np.asarray(x), dtype=torch.float32, device=dev)
    zW, zB = _z(pay["zgoals"]["MATE_W"]), _z(pay["zgoals"]["MATE_B"])
    omega = omega_ids(np.array([1800]), np.array([1800]), np.array([float("nan")]))[0]

    names = {1: "mate_W", 0: "draw", -1: "mate_B"}
    from catspace.data.encode import encode_packed, encode_meta
    import chess as _chess
    packed, meta, y = [], [], []
    if args.forced_set:
        import json
        data = json.loads(Path(args.forced_set).read_text())["classes"]
        cls2y = {"mate_W": 1, "draw": 0, "mate_B": -1}
        for cls, items in data.items():
            for it in items:
                b = _chess.Board(it["fen"])
                packed.append(encode_packed(b)); meta.append(encode_meta(b)); y.append(cls2y[cls])
        print("forced-mate set: " + ", ".join(
            f"{c}={sum(1 for yy in y if yy==cls2y[c])}" for c in ("mate_W", "draw", "mate_B")))
    else:
        buckets = harvest_near_mate(args.shards, args.per_class, args.back)
        print("harvested: " + ", ".join(f"near {names[k]}={len(buckets[k])}" for k in (1, 0, -1)))
        for k in (1, 0, -1):
            for p, m in buckets[k]:
                packed.append(p); meta.append(m); y.append(k)
    packed = np.stack(packed); meta = np.stack(meta); y = np.array(y)
    with torch.no_grad():
        pl = torch.from_numpy(feature_planes(packed, meta)).to(dev)
        om = torch.from_numpy(np.tile(omega, (len(packed), 1))).to(dev)
        fF = fb.embed_F(pl, om)
        F = fF.cpu().numpy()
        reachW = fb.score(fF, zW).cpu().numpy()          # reach to White-mate goal
        reachB = fb.score(fF, zB).cpu().numpy()          # reach to Black-mate goal

    full = separability(F, y)
    reach = np.stack([reachW, reachB], 1)
    rsep = separability(reach, y)                        # separability in 2-D reach-space
    corr = float(np.corrcoef(reachW, reachB)[0, 1])      # shared "finality" if ~1
    diff = (reachW - reachB)[:, None]                    # MATE_DIFF = the intended VALUE axis
    dsep = separability(diff, y)
    print(f"  F-space separability   (chance {full['chance']:.2f}): "
          f"kNN {full['knn']:.2f} · linear {full['linear']:.2f} · silhouette {full['silhouette']:+.2f}")
    print(f"  reach-space separability          : kNN {rsep['knn']:.2f} · linear {rsep['linear']:.2f} "
          f"· silhouette {rsep['silhouette']:+.2f}")
    print(f"  corr(reach->mate_W, reach->mate_B) = {corr:+.3f}  (positive => a shared 'near-a-mate' "
          f"finality component partly dilutes the who-is-winning signal)")
    print(f"  VALUE axis (reachW - reachB = MATE_DIFF) separability: kNN {dsep['knn']:.2f} "
          f"· linear {dsep['linear']:.2f}")

    _plot(args, F, y, names, full, reachW, reachB)


def _plot(args, F, y, names, full, reachW, reachB):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    colors = {1: "#33aa55", 0: "#8b93a3", -1: "#d24b4b"}
    manifold, mname = _manifold_2d(F)
    try:
        lda = LDA(n_components=2).fit(F, y).transform(F)
    except Exception:
        lda = manifold
    reach = np.stack([reachW, reachB], 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0f1115")
    panels = ((axes[0], manifold, f"{mname} (unsupervised F)", None, None),
              (axes[1], lda, "LDA (supervised F)", None, None),
              (axes[2], reach, "REACH-space (how the embedding is USED)",
               "reach -> mate_W", "reach -> mate_B"))
    for ax, proj, title, xl, yl in panels:
        ax.set_facecolor("#0f1115")
        for k in (1, 0, -1):
            m = y == k
            ax.scatter(proj[m, 0], proj[m, 1], s=9, c=colors[k], alpha=0.5,
                       label=f"{names[k]} (n={int(m.sum())})", edgecolors="none")
        ax.set_title(title, color="#e6e6e6")
        if xl:
            ax.set_xlabel(xl, color="#9ec7ff"); ax.set_ylabel(yl, color="#9ec7ff")
        ax.tick_params(colors="#6b7280"); [s.set_color("#2a2e37") for s in ax.spines.values()]
        leg = ax.legend(framealpha=0.2)
        for t in leg.get_texts():
            t.set_color("#e6e6e6")
    fig.suptitle(f"Near-mate positions ({args.back}-ply from end) in F-space — "
                 f"kNN {full['knn']:.2f} · linear {full['linear']:.2f} · "
                 f"silhouette {full['silhouette']:+.2f}", color="#e6e6e6", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path("artifacts/generated") / f"{args.label}_regions.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, facecolor="#0f1115"); plt.close(fig)
    b64 = base64.b64encode(out.read_bytes()).decode()
    out.with_suffix(".html").write_text(
        f"<!doctype html><meta charset=utf-8><body style='margin:0;background:#0f1115'>"
        f"<img style='max-width:100%' src='data:image/png;base64,{b64}'></body>")
    print(f"-> {out}\n-> {out.with_suffix('.html')}")


if __name__ == "__main__":
    main()
