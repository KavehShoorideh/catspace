#!/usr/bin/env python
"""experiments/train_l2_head.py -- the L2 categorical OUTCOME head on top of the frozen
L1 reachability geometry (Kaveh 2026-07-20). The geometry stays policy-independent; this
head reads its F-embedding and predicts a DISTRIBUTION over board-based outcomes,
supervised on the EXACT tablebase labels:

  classes = win-distance bins (from DTM: log-spaced upper edges) + a DRAW class.

Draws land in the DRAW class (not a fake 0), wins land in their DTM bin. (Losses/black-
mate are absent from this data, so the head is win+draw here; a symmetric distance-to-
black-mate head would add them.) Trained on ALL nucleus positions -- every one has an
EXACT ground-truth tablebase label, which we record: L2 is exact ONLY on these labeled
nodes; off-nucleus it extrapolates (its entropy flags where it's unsure).

Acceptance target (Kaveh): within ~5 moves of mate the head must be sharp enough to
guide exactly to mate -> we report near-mate (DTM<=10) bin accuracy + expected-distance
correlation.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from scipy.stats import spearmanr

BOARD_ONLY = (18, 19)
WIN_EDGES = np.array([1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 46, 64], dtype=float)  # DTM bin upper edges
N_WIN = len(WIN_EDGES) + 1
DRAW = N_WIN
N_CLASS = N_WIN + 1
BIN_CENTER = np.concatenate([[1], (WIN_EDGES[:-1] + WIN_EDGES[1:]) / 2, [WIN_EDGES[-1] * 1.3], [200.0]])  # incl DRAW=200


def labels_of(dtm):
    return np.where(dtm > 0, np.searchsorted(WIN_EDGES, dtm, side="left"), DRAW).astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1", default="data/derived/sep/iqe_geom_min.pt")
    ap.add_argument("--data", default="data/derived/lichess_nearmate.npz")
    ap.add_argument("--out", default="data/derived/sep/l2_head.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = pick_device(args.device)
    fb, _ = load_ckpt(Path(args.l1), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.data)
    allp, allm = np.asarray(nz["packed"]), np.asarray(nz["meta"])
    dtm = nz["dtm"].astype(np.float32)
    y = labels_of(dtm)

    def embF(pk, mt, bs=4096):
        out = []
        for i in range(0, len(pk), bs):
            pl = feature_planes(pk[i:i+bs], mt[i:i+bs]); pl[:, BOARD_ONLY] = 0.0
            o = torch.from_numpy(np.tile(om, (len(pl), 1))).to(dev)
            with torch.no_grad():
                out.append(fb.embed_F(torch.from_numpy(pl).to(dev), o).cpu())
        return torch.cat(out)

    print(f"[stage] embedding {len(allp)} positions (frozen L1 {Path(args.l1).name}, board-only F)...", flush=True)
    E = embF(allp, allm).to(dev)
    Y = torch.from_numpy(y).to(dev)
    n = len(E); rng = np.random.default_rng(0)
    perm = rng.permutation(n); tr, te = perm[: int(0.85 * n)], perm[int(0.85 * n):]
    print(f"[stage] {n} ground-truth-labeled nodes ({int((dtm>0).sum())} win / {int((dtm==0).sum())} draw); "
          f"{N_CLASS} classes ({N_WIN} win-bins + DRAW)", flush=True)

    head = torch.nn.Sequential(torch.nn.Linear(fb.d, args.hidden), torch.nn.ReLU(),
                               torch.nn.Linear(args.hidden, N_CLASS)).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    # class weights (draws dominate) so win bins aren't ignored
    cnt = np.bincount(y, minlength=N_CLASS).astype(float); w = torch.from_numpy((cnt.sum() / (cnt + 1))).float().to(dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    for ep in range(args.epochs):
        head.train()
        for i in range(0, len(tr), 8192):
            b = tr[i:i+8192]
            opt.zero_grad(); lossf(head(E[b]), Y[b]).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        logit = head(E[te]); pred = logit.argmax(1).cpu().numpy()
    yte = y[te]; dtmte = dtm[te]
    acc = float((pred == yte).mean())
    # NEAR-MATE (DTM<=10 won) accuracy + expected-distance correlation -- the guidance test
    nm = np.flatnonzero((dtmte > 0) & (dtmte <= 10))
    nm_acc = float((pred[nm] == yte[nm]).mean()) if len(nm) else float("nan")
    with torch.no_grad():
        P = torch.softmax(head(E[te][nm]), 1).cpu().numpy()
    exp_d = P @ BIN_CENTER
    nm_sp = spearmanr(exp_d, dtmte[nm]).correlation if len(nm) else float("nan")
    # draw vs win separation
    draw_recall = float((pred[yte == DRAW] == DRAW).mean())
    win_recall = float((pred[yte < DRAW] < DRAW).mean())
    torch.save({"state": head.state_dict(), "d_in": fb.d, "hidden": args.hidden,
                "n_class": N_CLASS, "win_edges": WIN_EDGES, "bin_center": BIN_CENTER,
                "l1": str(args.l1)}, args.out)
    print(f"saved {args.out}")
    print(f"  overall bin acc {acc:.3f}   draw-recall {draw_recall:.3f}  win-recall {win_recall:.3f}")
    print(f"  NEAR-MATE (DTM<=10): bin acc {nm_acc:.3f}  expected-distance spearman {nm_sp:+.3f}  (n={len(nm)})")
    print(f"VERDICT L2_HEAD nm_bin_acc={nm_acc:.3f} nm_dist_spearman={nm_sp:+.3f} draw_recall={draw_recall:.3f}")


if __name__ == "__main__":
    main()
