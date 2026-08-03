#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/reachability_field/experiments/train_stratified_field.py -- the BOTTOM-UP STRATIFIED reachability field
(Kaveh 2026-07-20). Trains the F/B IQE quasimetric on PERFECT-PLAY signals from
stratified_perfect.npz, with the strata boundary = PIECE COUNT (captures), grounded
bottom-up on the tablebase.

  L_pos     d(F(s)->B(s')) ~ 1                     legal 1-ply edges (local metric)
  L_strata  d(F(child)->B(parent)) >= 1+margin     ONLY count-DROP (capture) edges: you
                                                    can't un-capture -> the piece-count
                                                    strata are one-way DOWN. (7p->6p drops
                                                    make the 7p stratum sit one-way above
                                                    the solved frontier = the extrapolation.)
  L_dtm     d(F(a)->B(b)) ~ gap                     optimal-line pairs (EXACT perfect-play
                                                    ply-gap) -> within-material order + a
                                                    terminal region, composing via the triangle
                                                    inequality (1 ply = 1 unit, set by L_pos).
  L_repel   d(F(x)->B(y)) >= floor                  material-UNREACHABLE random pairs (count-
                                                    vector reachability) -> separate same-
                                                    piece-count materials the strata don't.

Bottom-up curriculum: PHASE 1 trains the solved frontier (<= 6 pieces) to convergence;
PHASE 2 adds the 7p edges so the above-frontier stratum anchors onto an already-solid 6p
geometry (the DP-in-stratum-order that makes this non-circular).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device, save_ckpt
from catspace.research.tools.viz.builders.live_curves import log_and_render
from catspace.io import paths

BOARD_ONLY = (18, 19)
NONKING = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]   # packed planes: white PNBRQ, black pnbrq (skip kings)


def count_vectors(packed):
    b = packed.astype(np.uint64).view(np.uint8).reshape(len(packed), 12, 8)
    return np.unpackbits(b, axis=2).sum(2).astype(np.int16)[:, NONKING]   # (N,10)


def reach_mask(A, B):
    """Can material B be reached from A? Can't gain pawns; non-pawn pieces gained <= pawns
    available to promote. cols: [Pw,Nw,Bw,Rw,Qw, Pb,Nb,Bb,Rb,Qb]."""
    okp = (B[:, 0] <= A[:, 0]) & (B[:, 5] <= A[:, 5])
    addw = np.maximum(0, B[:, 1:5] - A[:, 1:5]).sum(1)
    addb = np.maximum(0, B[:, 6:10] - A[:, 6:10]).sum(1)
    return okp & (addw <= A[:, 0] - B[:, 0]) & (addb <= A[:, 5] - B[:, 5])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=paths.sep("iqe_nucleus_gn.pt"))
    ap.add_argument("--data", default=paths.derived("stratified_perfect.npz"))
    ap.add_argument("--out", default=paths.sep("iqe_stratified.pt"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--phase1-frac", type=float, default=0.55,
                    help="fraction of steps on the <=6p frontier before adding 7p")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--strata-margin", type=float, default=12.0)
    ap.add_argument("--repel-floor", type=float, default=18.0)
    ap.add_argument("--dtm-cap", type=float, default=24.0, help="cap pair gap for L_dtm (compose beyond)")
    ap.add_argument("--w-pos", type=float, default=2.0)
    ap.add_argument("--w-strata", type=float, default=1.0)
    ap.add_argument("--w-dtm", type=float, default=0.5)
    ap.add_argument("--w-repel", type=float, default=0.3)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device); torch.manual_seed(args.seed)
    live_stem = Path(str(paths.experiments_dir())) / (Path(args.out).stem + "_curves")
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.train()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    opt = torch.optim.Adam(fb.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    nz = np.load(args.data, allow_pickle=True)
    P, M = np.asarray(nz["packed"]), np.asarray(nz["meta"])
    SDTM, WDL, PCNT, MATID = (np.asarray(nz["sdtm"]), np.asarray(nz["wdl"]),
                              np.asarray(nz["pcount"]).astype(int), np.asarray(nz["matid"]))
    EP, EM, EC, ECM = (np.asarray(nz["e_p_packed"]), np.asarray(nz["e_p_meta"]),
                       np.asarray(nz["e_c_packed"]), np.asarray(nz["e_c_meta"]))
    EDROP = np.asarray(nz["e_drop"]).astype(bool)
    AP, AM, BP, BM, GAP = (np.asarray(nz["a_packed"]), np.asarray(nz["a_meta"]),
                           np.asarray(nz["b_packed"]), np.asarray(nz["b_meta"]),
                           np.asarray(nz["gap"]).astype(np.float32))
    names = list(nz["material_names"])
    pcv = count_vectors(P)
    e_pcnt = 2 + count_vectors(EP).sum(1)                     # parent piece count per edge
    a_pcnt = 2 + count_vectors(AP).sum(1)
    e_le6, a_le6 = e_pcnt <= 6, a_pcnt <= 6
    print(f"[stage] {len(P)} pos, {len(EP)} edges ({int(EDROP.mean()*100)}% drop), {len(AP)} pairs; "
          f"strata {sorted(set(PCNT.tolist()))}; 7p edges {int((~e_le6).sum())}", flush=True)

    def bp(pk, mt):
        pl = feature_planes(pk, mt); pl[:, BOARD_ONLY] = 0.0
        return torch.from_numpy(pl).to(dev)

    def eF(pk, mt):
        return fb.embed_F(bp(pk, mt), torch.from_numpy(np.tile(om, (len(pk), 1))).to(dev))

    def eB(pk, mt):
        return fb.embed_B(bp(pk, mt))

    # ---- probes: capture one-way, per-material order, per-stratum mate distance ----
    from scipy.stats import spearmanr
    won = np.flatnonzero((SDTM > 0) & (PCNT <= 6))

    def probe():
        # capture one-way separation: drop edges, d(parent->child) vs d(child->parent)
        di = rng.choice(np.flatnonzero(EDROP & e_le6), min(300, int((EDROP & e_le6).sum())), replace=False)
        with torch.no_grad():
            fwd = fb.distance_matrix(eF(EP[di], EM[di]), eB(EC[di], ECM[di])).diagonal().cpu().numpy()
            bwd = fb.distance_matrix(eF(EC[di], ECM[di]), eB(EP[di], EM[di])).diagonal().cpu().numpy()
        cap_asym = float(np.median(bwd / np.maximum(fwd, 1e-6)))
        # per-material spearman(d_to_nearmate, sdtm): within each material, does d order DTM?
        sps = []
        for mid in sorted(set(MATID[won].tolist())):
            idx = won[MATID[won] == mid]
            if len(idx) < 40:
                continue
            idx = idx[rng.permutation(len(idx))[:200]]
            nm = idx[np.argsort(SDTM[idx])[:8]]                # 8 nearest-mate as goal anchor
            with torch.no_grad():
                d = fb.distance_matrix(eF(P[idx], M[idx]), eB(P[nm], M[nm])).min(1).values.cpu().numpy()
            sps.append(spearmanr(d, SDTM[idx]).correlation)
        permat = float(np.nanmean(sps)) if sps else float("nan")
        # 7p above frontier: is 7p one-way above 6p? sample 7p drop edges (7->6)
        up = float("nan")
        s7 = np.flatnonzero(EDROP & ~e_le6)
        if len(s7) > 20:
            di7 = rng.choice(s7, min(200, len(s7)), replace=False)
            with torch.no_grad():
                f7 = fb.distance_matrix(eF(EP[di7], EM[di7]), eB(EC[di7], ECM[di7])).diagonal().cpu().numpy()
                b7 = fb.distance_matrix(eF(EC[di7], ECM[di7]), eB(EP[di7], EM[di7])).diagonal().cpu().numpy()
            up = float(np.median(b7 / np.maximum(f7, 1e-6)))
        return cap_asym, permat, up

    # BOTTOM-UP CURRICULUM (Kaveh): introduce strata by piece count; the capture
    # boundaries are the NATURAL CHECKPOINTS -- a solved lower stratum is frozen ground
    # for the next, so each _le{pc} checkpoint is safe to resume from and audit. Weighted
    # toward the frontier (6p) and above (7p): [3,4,5,6,7] at cumulative step fractions.
    SCHED = [(0.12, 3), (0.24, 4), (0.42, 5), (0.78, 6), (1.00, 7)]

    def cur_max(step):
        frac = (step + 1) / args.steps
        for f, pc in SCHED:
            if frac <= f:
                return pc
        return 7

    pos_by_max = {pc: np.flatnonzero(PCNT <= pc) for _, pc in SCHED}
    prev = None
    t0 = time.time()
    for step in range(args.steps):
        cm = cur_max(step)
        if prev is not None and cm != prev:
            fb.eval(); save_ckpt(fb, Path(str(args.out).replace(".pt", f"_le{prev}.pt")),
                                 step=pay.get("step", 0), zgoals=pay.get("zgoals")); fb.train()
            print(f"  [checkpoint] bottom-up frontier <= {prev}p -> saved _le{prev}.pt", flush=True)
        prev = cm
        eidx = np.flatnonzero(e_pcnt <= cm); aidx = np.flatnonzero(a_pcnt <= cm)
        posidx = pos_by_max[cm]

        ei = eidx[rng.integers(0, len(eidx), size=args.batch)]
        d_pos = fb.distance_matrix(eF(EP[ei], EM[ei]), eB(EC[ei], ECM[ei])).diagonal()
        L_pos = ((d_pos - 1.0) ** 2).mean()

        drop = EDROP[ei]
        if drop.any():
            ci = ei[drop]
            d_bwd = fb.distance_matrix(eF(EC[ci], ECM[ci]), eB(EP[ci], EM[ci])).diagonal()
            # RELATIVE margin anchored to the forward capture distance d_pos[drop] (~1): the
            # one-way reverse must sit `strata_margin` beyond the capture itself. Absolute
            # (1+margin) inflated the whole embedding when captures entered (JOURNAL 2026-07-20).
            L_strata = torch.relu(d_pos[drop].detach() + args.strata_margin - d_bwd).pow(2).mean()
        else:
            L_strata = torch.zeros((), device=dev)

        pi = aidx[rng.integers(0, len(aidx), size=args.batch)]
        d_pair = fb.distance_matrix(eF(AP[pi], AM[pi]), eB(BP[pi], BM[pi])).diagonal()
        tgt = torch.from_numpy(np.minimum(GAP[pi], args.dtm_cap)).to(dev).to(d_pair.dtype)
        L_dtm = torch.nn.functional.smooth_l1_loss(d_pair.clamp(max=args.dtm_cap + 8), tgt)

        ra = posidx[rng.integers(0, len(posidx), size=args.batch)]
        rb = posidx[rng.integers(0, len(posidx), size=args.batch)]
        unreach = ~reach_mask(pcv[ra], pcv[rb])
        d_cross = fb.distance_matrix(eF(P[ra], M[ra]), eB(P[rb], M[rb])).diagonal()
        keep = torch.from_numpy(unreach).to(dev)
        L_repel = (torch.relu(args.repel_floor - d_cross)[keep].pow(2).mean()
                   if unreach.any() else torch.zeros((), device=dev))

        loss = (args.w_pos * L_pos + args.w_strata * L_strata
                + args.w_dtm * L_dtm + args.w_repel * L_repel)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 100 == 0 or step == args.steps - 1:
            cap_asym, permat, up = probe()
            ph = f"<={cm}p"
            lp, ls, ld, lr = (float(L_pos.detach()), float(L_strata.detach()),
                              float(L_dtm.detach()), float(L_repel.detach()))
            dpos = float(d_pos.median().detach())
            print(f"  {ph} step {step:4d}  Lpos {lp:.3f} Lstr {ls:.3f} "
                  f"Ldtm {ld:.3f} Lrep {lr:.3f}  d_pos {dpos:.2f}  "
                  f"cap_asym {cap_asym:.1f}x permat {permat:+.3f} 7p_up {up:.1f}x  ({time.time()-t0:.0f}s)",
                  flush=True)
            log_and_render(live_stem, step,
                           {"L_pos": lp, "L_strata": ls, "L_dtm": ld, "L_repel": lr, "d_pos": dpos,
                            "cap_asym_x": cap_asym, "permat_spearman": permat, "frontier_pc": cm},
                           title=f"stratified L1 field ({Path(args.out).stem})")
        if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
            fb.eval(); save_ckpt(fb, Path(args.out), step=pay.get("step", 0), zgoals=pay.get("zgoals")); fb.train()

    fb.eval(); save_ckpt(fb, Path(args.out), step=pay.get("step", 0), zgoals=pay.get("zgoals"))
    cap_asym, permat, up = probe()
    print(f"saved {args.out}")
    print(f"VERDICT STRAT_FIELD cap_asym={cap_asym:.1f}x permat={permat:+.3f} 7p_up={up:.1f}x")


if __name__ == "__main__":
    main()
