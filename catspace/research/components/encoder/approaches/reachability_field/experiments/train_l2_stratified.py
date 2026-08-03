#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/reachability_field/experiments/train_l2_stratified.py -- the L2 adversarial OUTCOME head on the FROZEN stratified
L1 (Kaveh 2026-07-20, M2). L1 stays cooperative reachability geometry; this head reads its
board-only F-embedding and predicts a DISTRIBUTION over the game-theoretic outcome under PERFECT
play, supervised on EXACT tablebase labels (signed DTM to the terminal REGION -- not a pole).

Classes (signed, symmetric -- we now HAVE losses in the data, unlike the old win+draw-only head):
    [White-win DTM bins] + [DRAW] + [Black-win DTM bins].
From the softmax we read BOTH:
    * expected SIGNED distance-to-terminal-region  = P @ signed_bin_centers   (the remoteness readout)
    * the COMMITTOR P(W/D/L) = summed mass in the White-win / draw / Black-win regions.

This is the acceptance test for whether the frozen cooperative L1 carries enough structure for the
adversarial head (the old frozen-L2 on a reachability-only L1 was poor: near-mate bin acc 0.13).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.tools.viz.builders.live_curves import log_and_render
from scipy.stats import spearmanr
from catspace.io import paths

BOARD_ONLY = (18, 19)
WIN_EDGES = np.array([1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 46, 64], dtype=float)   # DTM bin upper edges
N_WIN = len(WIN_EDGES) + 1                                                      # 13 win-distance bins
DRAW = N_WIN                                                                    # draw class index
N_CLASS = 2 * N_WIN + 1                                                         # white-bins + draw + black-bins
_WCEN = np.concatenate([[1.0], (WIN_EDGES[:-1] + WIN_EDGES[1:]) / 2, [WIN_EDGES[-1] * 1.3]])  # 13 centers
BIN_CENTER = np.concatenate([_WCEN, [0.0], -_WCEN])                             # signed centers (N_CLASS,)


def labels_of(sdtm):
    """signed DTM -> class: >0 White-win bin, ==0 draw, <0 Black-win bin (mirrored)."""
    y = np.full(len(sdtm), DRAW, np.int64)
    w, l = sdtm > 0, sdtm < 0
    y[w] = np.searchsorted(WIN_EDGES, sdtm[w], side="left")
    y[l] = N_WIN + 1 + np.searchsorted(WIN_EDGES, -sdtm[l], side="left")
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1", default=paths.sep("iqe_stratified.pt"))
    ap.add_argument("--data", default=paths.derived("stratified_perfect.npz"))
    ap.add_argument("--out", default=paths.sep("l2_stratified.pt"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--feat", choices=["embed", "trunk"], default="embed",
                    help="embed = final IQE F (d); trunk = encoder output pre-head (enc_out, richer)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = pick_device(args.device)
    live_stem = Path(str(paths.experiments_dir())) / (Path(args.out).stem + "_curves")
    fb, _ = load_ckpt(Path(args.l1), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.data, allow_pickle=True)
    P, M = np.asarray(nz["packed"]), np.asarray(nz["meta"])
    SDTM, WDL, PCNT = np.asarray(nz["sdtm"]), np.asarray(nz["wdl"]), np.asarray(nz["pcount"]).astype(int)
    keep = PCNT <= 6                                                            # exact-labeled nodes only
    P, M, SDTM, WDL = P[keep], M[keep], SDTM[keep], WDL[keep]
    y = labels_of(SDTM)

    def embF(pk, mt, bs=4096):
        out = []
        for i in range(0, len(pk), bs):
            pl = feature_planes(pk[i:i+bs], mt[i:i+bs]); pl[:, BOARD_ONLY] = 0.0
            pl_t = torch.from_numpy(pl).to(dev)
            with torch.no_grad():
                if args.feat == "trunk":
                    h = fb.encF(pl_t)                                          # richer pre-head trunk
                else:
                    o = torch.from_numpy(np.tile(om, (len(pl), 1))).to(dev)
                    h = fb.embed_F(pl_t, o)                                    # final IQE embedding
            out.append(h.cpu())
        return torch.cat(out)

    print(f"[stage] embedding {len(P)} exact-labeled positions (frozen L1 {Path(args.l1).name})...", flush=True)
    E = embF(P, M).to(dev); Y = torch.from_numpy(y).to(dev)
    n = len(E); rng = np.random.default_rng(0)
    perm = rng.permutation(n); tr, te = perm[: int(0.85 * n)], perm[int(0.85 * n):]
    nb = np.bincount(y, minlength=N_CLASS)
    print(f"[stage] {n} nodes, {N_CLASS} classes ({N_WIN} win + draw + {N_WIN} loss); "
          f"W={int((SDTM>0).sum())} D={int((SDTM==0).sum())} L={int((SDTM<0).sum())}", flush=True)

    head = torch.nn.Sequential(torch.nn.Linear(E.shape[1], args.hidden), torch.nn.ReLU(),
                               torch.nn.Linear(args.hidden, N_CLASS)).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    w = torch.from_numpy(nb.sum() / (nb + 1.0)).float().to(dev)                 # class-balance (losses rare)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    center = torch.from_numpy(BIN_CENTER).float().to(dev)
    sdtm_te = SDTM[te]
    nm = np.flatnonzero(np.abs(sdtm_te) <= 10)                                  # near-terminal (|DTM|<=10)

    def evaluate():
        head.eval()
        with torch.no_grad():
            logit = head(E[te]); pred = logit.argmax(1).cpu().numpy()
            P_te = torch.softmax(logit, 1)
            exp_d = (P_te @ center).cpu().numpy()
            # committor P(W/D/L) = summed region mass
            pw = P_te[:, :N_WIN].sum(1); pl = P_te[:, N_WIN+1:].sum(1)
            pred_wdl = torch.where(pw > pl, torch.where(pw > P_te[:, DRAW], torch.ones_like(pw), torch.zeros_like(pw)),
                                   torch.where(pl > P_te[:, DRAW], -torch.ones_like(pw), torch.zeros_like(pw))).cpu().numpy()
        yte = y[te]
        acc = float((pred == yte).mean())
        nm_acc = float((pred[nm] == yte[nm]).mean()) if len(nm) else float("nan")
        nm_sp = spearmanr(exp_d[nm], sdtm_te[nm]).correlation if len(nm) else float("nan")
        wdl_te = WDL[te]
        wdl_acc = float((pred_wdl == wdl_te).mean())
        draw_rec = float((pred_wdl[wdl_te == 0] == 0).mean()) if (wdl_te == 0).any() else float("nan")
        loss_rec = float((pred_wdl[wdl_te == -1] == -1).mean()) if (wdl_te == -1).any() else float("nan")
        head.train()
        return acc, nm_acc, nm_sp, wdl_acc, draw_rec, loss_rec

    for ep in range(args.epochs):
        head.train()
        for i in range(0, len(tr), 8192):
            b = tr[i:i+8192]
            opt.zero_grad(); lossf(head(E[b]), Y[b]).backward(); opt.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            acc, nm_acc, nm_sp, wdl_acc, draw_rec, loss_rec = evaluate()
            print(f"  ep {ep:3d}  bin_acc {acc:.3f}  near-mate bin_acc {nm_acc:.3f} "
                  f"exp-dist spearman {nm_sp:+.3f}  committor WDL_acc {wdl_acc:.3f} "
                  f"(draw-rec {draw_rec:.2f} loss-rec {loss_rec:.2f})", flush=True)
            log_and_render(live_stem, ep,
                           {"bin_acc": acc, "near_mate_bin_acc": nm_acc, "exp_dist_spearman": nm_sp,
                            "committor_WDL_acc": wdl_acc, "draw_recall": draw_rec, "loss_recall": loss_rec},
                           title=f"L2 head on frozen {Path(args.l1).stem}")

    acc, nm_acc, nm_sp, wdl_acc, draw_rec, loss_rec = evaluate()
    torch.save({"state": head.state_dict(), "d_in": int(E.shape[1]), "hidden": args.hidden, "n_class": N_CLASS,
                "win_edges": WIN_EDGES, "bin_center": BIN_CENTER, "l1": str(args.l1)}, args.out)
    print(f"saved {args.out}")
    print(f"VERDICT L2_STRAT near_mate_bin_acc={nm_acc:.3f} exp_dist_spearman={nm_sp:+.3f} "
          f"committor_WDL_acc={wdl_acc:.3f} draw_recall={draw_rec:.3f} loss_recall={loss_rec:.3f}")


if __name__ == "__main__":
    main()
