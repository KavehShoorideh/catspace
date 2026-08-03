#!/usr/bin/env python
"""experiments/viz_stratified_umap.py -- does the BOTTOM-UP stratified field's embedding
show the piece-count strata cleanly, and does the 7p above-frontier stratum sit one-way
above the solved 6p set? (Kaveh 2026-07-20: "go a few plies past the tablebase frontier
and see what the umap looks like.")

Embeds every position with the field (board-only F), UMAP -> 2D, and draws 4 panels:
  (1) piece-count STRATA        (2) MATERIAL clusters
  (3) signed DTM (to terminal region)    (4) WDL (win/draw/loss; 7p ungrounded = grey)
On panel (1) it overlays arrows from sample 7p positions to their actual 6p capture-child
(the stratum-crossing capture) -- embedded in the SAME UMAP fit so the arrow is faithful.

Also prints the quantitative structure (no projection needed): per-material spearman(d,DTM),
capture one-way asymmetry, 7p->6p one-way, k-NN piece-count + material purity, and a
piece-count silhouette.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device

BOARD_ONLY = (18, 19)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/derived/sep/iqe_stratified.pt")
    ap.add_argument("--data", default="data/derived/stratified_perfect.npz")
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--arrows", type=int, default=14)
    ap.add_argument("--out", default="artifacts/experiments/stratified_umap.png")
    args = ap.parse_args()
    dev = pick_device("auto")
    fb, _ = load_ckpt(Path(args.ckpt), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.data, allow_pickle=True)
    P, M = np.asarray(nz["packed"]), np.asarray(nz["meta"])
    SDTM, WDL, PCNT, MATID = (np.asarray(nz["sdtm"]), np.asarray(nz["wdl"]),
                              np.asarray(nz["pcount"]).astype(int), np.asarray(nz["matid"]))
    GR = np.asarray(nz["grounded"]).astype(bool)
    EP, EM, EC, ECM = (np.asarray(nz["e_p_packed"]), np.asarray(nz["e_p_meta"]),
                       np.asarray(nz["e_c_packed"]), np.asarray(nz["e_c_meta"]))
    EDROP = np.asarray(nz["e_drop"]).astype(bool)
    names = list(nz["material_names"])
    rng = np.random.default_rng(0)

    def embF(pk, mt, bs=2048):
        out = []
        for i in range(0, len(pk), bs):
            pl = feature_planes(pk[i:i+bs], mt[i:i+bs]); pl[:, BOARD_ONLY] = 0.0
            o = torch.from_numpy(np.tile(om, (len(pl), 1))).to(dev)
            with torch.no_grad():
                out.append(fb.embed_F(torch.from_numpy(pl).to(dev), o).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def embB(pk, mt, bs=2048):
        out = []
        for i in range(0, len(pk), bs):
            pl = feature_planes(pk[i:i+bs], mt[i:i+bs]); pl[:, BOARD_ONLY] = 0.0
            with torch.no_grad():
                out.append(fb.embed_B(torch.from_numpy(pl).to(dev)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    # subsample positions, but KEEP all 7p (rarer, the headline)
    sub7 = np.flatnonzero(PCNT >= 7)
    sub6 = np.flatnonzero(PCNT <= 6)
    keep6 = rng.choice(sub6, min(args.n - len(sub7), len(sub6)), replace=False)
    sel = np.concatenate([keep6, sub7])
    rng.shuffle(sel)
    pk, mt = P[sel], M[sel]
    pcnt, matid, sdtm, wdl, gr = PCNT[sel], MATID[sel], SDTM[sel], WDL[sel], GR[sel]
    mat = np.array([names[i] for i in matid])
    print(f"[stage] embedding {len(sel)} positions ({len(sub7)} are 7p)...", flush=True)
    EF = embF(pk, mt)

    # ---- arrow pairs: sample 7p drop-edges (7->6 captures), embed both ends ----
    s7 = np.flatnonzero(EDROP & (2 + _pc(EP) > 6))
    A_xy = B_xy = None
    if len(s7):
        ai = rng.choice(s7, min(args.arrows, len(s7)), replace=False)
        AF = embF(EP[ai], EM[ai]); CF = embF(EC[ai], ECM[ai])
    else:
        AF = CF = np.zeros((0, EF.shape[1]), np.float32)

    # ---- quantitative structure (raw embedding space) ----
    from scipy.stats import spearmanr
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score

    def cap_oneway(mask7):
        idx = np.flatnonzero(EDROP & ((2 + _pc(EP) > 6) if mask7 else (2 + _pc(EP) <= 6)))
        if len(idx) < 20:
            return float("nan")
        idx = rng.choice(idx, min(400, len(idx)), replace=False)
        f = _dm(fb, embF(EP[idx], EM[idx]), embB(EC[idx], ECM[idx]), dev)
        b = _dm(fb, embF(EC[idx], ECM[idx]), embB(EP[idx], EM[idx]), dev)
        return float(np.median(b / np.maximum(f, 1e-6)))

    cap6, cap7 = cap_oneway(False), cap_oneway(True)
    # per-material spearman(d_to_nearmate, sdtm)
    won = np.flatnonzero((sdtm > 0) & (pcnt <= 6))
    sps = []
    for mid in sorted(set(matid[won].tolist())):
        ii = won[matid[won] == mid]
        if len(ii) < 40:
            continue
        ii = ii[rng.permutation(len(ii))[:200]]
        nm = ii[np.argsort(sdtm[ii])[:8]]
        d = _dm(fb, EF[ii], embB(pk[nm], mt[nm]), dev, diag=False).min(1)
        sps.append(spearmanr(d, sdtm[ii]).correlation)
    permat = float(np.nanmean(sps)) if sps else float("nan")
    # purities + silhouette
    ss = rng.permutation(len(sel))[:6000]
    nn = NearestNeighbors(n_neighbors=11, metric="euclidean").fit(EF[ss])
    _, nb = nn.kneighbors(EF[ss]); nb = nb[:, 1:]
    pc_pur = float(np.mean([(pcnt[ss][nb[i]] == pcnt[ss][i]).mean() for i in range(len(ss))]))
    mat_pur = float(np.mean([(matid[ss][nb[i]] == matid[ss][i]).mean() for i in range(len(ss))]))
    sil = float(silhouette_score(EF[ss], pcnt[ss], metric="euclidean", sample_size=4000, random_state=0))
    print(f"cap_oneway 6p={cap6:.1f}x  7p->6p={cap7:.1f}x | permat_spearman(d,DTM)={permat:+.3f}")
    print(f"kNN purity: piece-count={pc_pur:.3f} material={mat_pur:.3f} | piece-count silhouette={sil:+.3f}")

    # ---- UMAP (fit positions + arrow endpoints together) ----
    print("[stage] UMAP -> 2D...", flush=True)
    import umap
    allE = np.concatenate([EF, AF, CF], axis=0)
    XY = umap.UMAP(n_neighbors=30, min_dist=0.12, metric="euclidean", random_state=0).fit_transform(allE)
    nP = len(EF); nA = len(AF)
    XYp = XY[:nP]; XYa = XY[nP:nP+nA]; XYc = XY[nP+nA:]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    (a1, a2), (a3, a4) = axes

    # (1) piece-count strata + arrows
    strata = sorted(set(pcnt.tolist()))
    cmap = plt.get_cmap("turbo")
    for k, pc in enumerate(strata):
        s = pcnt == pc
        a1.scatter(XYp[s, 0], XYp[s, 1], s=5, alpha=0.55, color=cmap(k / max(1, len(strata)-1)),
                   label=f"{pc}p (n={int(s.sum())})")
    for j in range(nA):
        a1.annotate("", xy=(XYc[j, 0], XYc[j, 1]), xytext=(XYa[j, 0], XYa[j, 1]),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6, alpha=0.9))
    a1.legend(markerscale=2, fontsize=8, loc="best")
    a1.set_title(f"PIECE-COUNT STRATA  (arrows: 7p -> 6p capture; silhouette {sil:+.2f}, kNN purity {pc_pur:.2f})")

    # (2) material
    top = [m for m, _ in sorted(zip(*np.unique(mat, return_counts=True)), key=lambda x: -x[1])][:12]
    for m in top:
        s = mat == m
        a2.scatter(XYp[s, 0], XYp[s, 1], s=5, alpha=0.55, label=m)
    a2.legend(markerscale=2, fontsize=7, ncol=2, loc="best")
    a2.set_title(f"MATERIAL  (kNN purity {mat_pur:.2f}, per-material spearman(d,DTM) {permat:+.2f})")

    # (3) signed DTM (to terminal region), <=6 only
    m6 = pcnt <= 6
    sc = a3.scatter(XYp[m6, 0], XYp[m6, 1], s=5, c=np.clip(sdtm[m6], -40, 40),
                    cmap="coolwarm", alpha=0.7)
    a3.scatter(XYp[~m6, 0], XYp[~m6, 1], s=5, c="0.6", alpha=0.4)
    plt.colorbar(sc, ax=a3, label="signed DTM (+White mate / -Black mate, plies, capped 40)")
    a3.set_title("signed DTM to terminal REGION  (not a pole; grey = 7p, no exact DTM)")

    # (4) WDL
    colw = {1: "#2c7", 0: "#89a", -1: "#d44"}
    for v, lab in [(1, "White win"), (0, "draw"), (-1, "Black win")]:
        s = (wdl == v) & (gr | (pcnt <= 6))
        a4.scatter(XYp[s, 0], XYp[s, 1], s=5, c=colw[v], alpha=0.6, label=lab)
    s = (~gr) & (pcnt >= 7)
    a4.scatter(XYp[s, 0], XYp[s, 1], s=5, c="0.7", alpha=0.4, label="7p ungrounded")
    a4.legend(markerscale=2, fontsize=8, loc="best")
    a4.set_title("OUTCOME  (W/D/L under perfect play; draws+Black-wins = the failing data)")

    for a in (a1, a2, a3, a4):
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Bottom-up stratified field -- {Path(args.ckpt).name} -- {len(sel)} positions, board-only F",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=115)
    print(f"VERDICT STRAT_UMAP silhouette={sil:+.3f} pc_purity={pc_pur:.3f} mat_purity={mat_pur:.3f} "
          f"permat={permat:+.3f} cap6={cap6:.1f}x cap7={cap7:.1f}x -> {args.out}")


def _pc(packed):
    """piece count minus kings, from packed bitboards (popcount of non-king planes)."""
    b = packed.astype(np.uint64).view(np.uint8).reshape(len(packed), 12, 8)
    nk = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    return np.unpackbits(b, axis=2).sum(2)[:, nk].sum(1)


def _dm(fb, F, Bemb, dev, diag=True):
    import torch as _t
    with _t.no_grad():
        d = fb.distance_matrix(_t.from_numpy(F).to(dev), _t.from_numpy(Bemb).to(dev))
        return d.diagonal().cpu().numpy() if diag else d.cpu().numpy()


if __name__ == "__main__":
    main()
