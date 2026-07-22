#!/usr/bin/env python
"""experiments/validate_decoupled.py -- acceptance test for the DECOUPLED field design
(Kaveh 2026-07-22: "Separate DTM head, don't overload d"; DECISIONS.md sec 7).

The one distance `d` was being forced to encode BOTH material-reachability (navigation
geometry) AND mate-distance (DTM); those compete for the field's ~1-D axis, so cramming
DTM in broke it (6p spearman 0.19->0.05) without raising rank. Resolution: `d` carries
REACHABILITY GEOMETRY ONLY (geometry + repulsion + best-play, --w-dtm 0), and DTM/mate-
distance comes from a SEPARATE head. This script proves both hold at once:

  A. d is a healthy quasimetric        d_step << d_rand (ratio), asym irr>rev, eff-rank
  B. d is DECOUPLED from DTM           spearman(d(F,MATE_W), DTM) is LOW by design (confirm,
                                       not a failure -- DTM was never trained into d)
  C. a SEPARATE DTM head recovers it   small MLP on the FROZEN shared trunk encF(planes)
                                       -> DTM; spearman by piece vs the plain-CNN baseline
                                       (0.89/0.61/0.355) and the distilled field (.88/.71/.53)

All numbers are printed VERDICT lines (no hand-quoted figures). Conventions match
train_geometry_l1.py exactly: BOARD_ONLY (18,19) zeroed, omega=(1800,1800,300), MATE_W goal.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed, encode_meta, encode_packed  # noqa: F401
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device

BOARD_ONLY = (18, 19)
NONKING = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]


def count_vectors(packed):
    b = packed.astype(np.uint64).view(np.uint8).reshape(len(packed), 12, 8)
    return np.unpackbits(b, axis=2).sum(2).astype(np.int16)[:, NONKING]


def reach_mask(A, B):
    okp = (B[:, 0] <= A[:, 0]) & (B[:, 5] <= A[:, 5])
    addw = np.maximum(0, B[:, 1:5] - A[:, 1:5]).sum(1)
    addb = np.maximum(0, B[:, 6:10] - A[:, 6:10]).sum(1)
    return okp & (addw <= A[:, 0] - B[:, 0]) & (addb <= A[:, 5] - B[:, 5])


def eff_rank(X: np.ndarray) -> float:
    """entropy-of-singular-values effective rank (the standard collapse gate)."""
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="data/derived/sep/iqe_geom_field.pt")
    ap.add_argument("--data", default="data/derived/geom_pool.npz")
    ap.add_argument("--edges", default="data/derived/geom_pool_edges.npz")
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--n", type=int, default=4000, help="sample size for d_step/d_rand/eff-rank")
    ap.add_argument("--head-steps", type=int, default=3000)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    fb, pay = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    _zw = pay["zgoals"]["MATE_W"]
    zW = (_zw.detach().float() if torch.is_tensor(_zw) else torch.tensor(np.asarray(_zw, np.float32))).to(dev)

    def bp(pk, mt):
        pl = feature_planes(pk, mt); pl[:, BOARD_ONLY] = 0.0
        return torch.from_numpy(pl).to(dev)

    def eF(pk, mt):
        with torch.no_grad():
            return fb.embed_F(bp(pk, mt), torch.from_numpy(np.tile(om, (len(pk), 1))).to(dev))

    def eB(pk, mt):
        with torch.no_grad():
            return fb.embed_B(bp(pk, mt))

    def trunk(pk, mt):                          # frozen SHARED trunk features (pre-head)
        with torch.no_grad():
            return fb.encF(bp(pk, mt))

    def d_pairs(pk1, mt1, pk2, mt2, chunk=512):
        """row-aligned d(F(x_i), B(y_i)) in chunks -- NEVER the full n x n distance_matrix
        (IQE pairwise on 4000 rows allocates tens of GB and gets OOM-killed silently)."""
        out = []
        for s in range(0, len(pk1), chunk):
            with torch.no_grad():
                out.append(fb.distance_matrix(eF(pk1[s:s + chunk], mt1[s:s + chunk]),
                                              eB(pk2[s:s + chunk], mt2[s:s + chunk])).diagonal().cpu().numpy())
        return np.concatenate(out)

    # ---- load pools
    nz = np.load(args.data)
    allp, allm, dtm_pool = np.asarray(nz["packed"]), np.asarray(nz["meta"]), nz["dtm"].astype(np.float32)
    ez = np.load(args.edges)
    EPK, EPM = np.asarray(ez["p_packed"]), np.asarray(ez["p_meta"])
    ECK, ECM = np.asarray(ez["c_packed"]), np.asarray(ez["c_meta"])
    EIR = np.asarray(ez["irrev"]).astype(bool)
    pcv = count_vectors(allp)
    print(f"[load] {len(allp)} pool pos, {len(EPK)} edges ({int(EIR.mean()*100)}% irrev)  [{time.time()-t0:.0f}s]", flush=True)

    # ===== A. quasimetric health: d_step vs d_rand, asym, eff-rank =====
    ei = rng.integers(0, len(EPK), size=args.n)
    d_step = d_pairs(EPK[ei], EPM[ei], ECK[ei], ECM[ei])
    ra = rng.integers(0, len(allp), size=args.n); rb = rng.integers(0, len(allp), size=args.n)
    d_rand = d_pairs(allp[ra], allm[ra], allp[rb], allm[rb])
    unreach = ~reach_mask(pcv[ra], pcv[rb])
    d_step_m, d_rand_m = float(np.median(d_step)), float(np.median(d_rand))
    d_unr_m = float(np.median(d_rand[unreach])) if unreach.any() else float("nan")
    ratio = d_rand_m / max(d_step_m, 1e-9)

    # asym on irreversible vs reversible edges (irr reverse should read farther)
    rf = d_step
    rb_ = d_pairs(ECK[ei], ECM[ei], EPK[ei], EPM[ei])
    asym = rb_ / np.maximum(rf, 1e-6)
    asym_irr = float(np.median(asym[EIR[ei]])) if EIR[ei].any() else float("nan")
    asym_rev = float(np.median(asym[~EIR[ei]])) if (~EIR[ei]).any() else float("nan")

    si = rng.choice(len(allp), min(3000, len(allp)), replace=False)
    Fb = eF(allp[si], allm[si]).cpu().numpy(); Bb = eB(allp[si], allm[si]).cpu().numpy()
    er_F, er_B = eff_rank(Fb), eff_rank(Bb)

    print(f"VERDICT DECOUPLE.A field={Path(args.field).stem} d={fb.d}  "
          f"d_step {d_step_m:.2f}  d_rand {d_rand_m:.2f} (unreach {d_unr_m:.2f})  ratio {ratio:.1f}x  "
          f"asym irr {asym_irr:.2f} / rev {asym_rev:.2f} (sep {asym_irr/max(asym_rev,1e-6):.2f}x)  "
          f"eff-rank F {er_F:.1f} B {er_B:.1f} /{fb.d}", flush=True)

    # ===== B. decoupling: d(F,MATE_W) vs DTM should be LOW (DTM not trained into d) =====
    dz = np.load(args.dtm_npz)
    P, M = np.asarray(dz["packed"]), np.asarray(dz["meta"])
    dtm = np.asarray(dz["dtm"]).astype(np.float32)
    ok = np.flatnonzero(dtm > 0)
    pc = np.array([len(board_from_packed(P[i], M[i]).piece_map()) for i in ok])
    d_mate = []
    for s in range(0, len(ok), 2048):
        sl = ok[s:s + 2048]
        with torch.no_grad():
            d_mate.append(fb.distance_matrix(eF(P[sl], M[sl]), zW[None, :])[:, 0].cpu().numpy())
    d_mate = np.concatenate(d_mate)
    dcpl = {}
    for k in (3, 4, 6):
        sel = pc == k
        if sel.sum() >= 50:
            dcpl[k] = float(spearmanr(d_mate[sel], dtm[ok][sel]).correlation)
    print("VERDICT DECOUPLE.B  spearman(d(F,MATE_W), DTM) by piece -- LOW = decoupled (d has no DTM term): "
          + "  ".join(f"{k}p {v:+.3f}" for k, v in dcpl.items()), flush=True)

    # ===== C. SEPARATE DTM head on the FROZEN shared trunk =====
    planes_ok = feature_planes(P[ok], M[ok]); dtm_ok = dtm[ok]
    tr = np.flatnonzero(rng.random(len(ok)) < 0.85); te = np.setdiff1d(np.arange(len(ok)), tr)
    # precompute frozen trunk features once (chunked)
    H = np.zeros((len(ok), 0), np.float32)
    feats = []
    for s in range(0, len(ok), 2048):
        h = trunk(P[ok][s:s + 2048], M[ok][s:s + 2048]).cpu().numpy()
        feats.append(h)
    H = np.concatenate(feats, 0).astype(np.float32)
    enc_out = H.shape[1]
    er_trunk = eff_rank(H[rng.choice(len(H), min(3000, len(H)), replace=False)])
    print(f"[trunk] frozen shared-trunk features {H.shape}  eff-rank {er_trunk:.1f}/{enc_out}  "
          f"(vs embedding F {er_F:.1f}) -- richer trunk => a head CAN carry DTM  [{time.time()-t0:.0f}s]", flush=True)

    head = nn.Sequential(nn.Linear(enc_out, 256), nn.ReLU(), nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=args.head_lr)
    Ht = torch.from_numpy(H).to(dev); yt = torch.from_numpy(dtm_ok / args.scale).to(dev)
    trt = torch.from_numpy(tr).to(dev)
    for s in range(args.head_steps):
        idx = trt[torch.randint(0, len(trt), (256,), device=dev)]
        pred = head(Ht[idx])[:, 0]
        loss = nn.functional.smooth_l1_loss(pred, yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 1000 == 0:
            print(f"  [head] step {s} loss {float(loss.detach()):.4f}", flush=True)
    head.eval()
    with torch.no_grad():
        predall = head(Ht)[:, 0].cpu().numpy()
    headsp = {}
    for k in (3, 4, 6):
        sel = te[pc[te] == k]
        if len(sel) >= 50:
            headsp[k] = float(spearmanr(predall[sel], dtm_ok[sel]).correlation)
    print("VERDICT DECOUPLE.C  DTM-head-on-trunk spearman(pred,DTM) held-out by piece "
          "(CNN 0.89/0.61/0.355 · distilled .88/.71/.53): "
          + "  ".join(f"{k}p {v:+.3f}" for k, v in headsp.items()), flush=True)

    Path("data/derived/sep").mkdir(parents=True, exist_ok=True)
    torch.save({"state": head.state_dict(), "enc_out": enc_out, "scale": args.scale,
                "field": str(args.field)}, "data/derived/sep/dtm_head_trunk.pt")
    print(f"  saved data/derived/sep/dtm_head_trunk.pt  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
