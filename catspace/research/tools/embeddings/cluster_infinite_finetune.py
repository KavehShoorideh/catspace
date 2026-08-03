#!/usr/bin/env python
"""catspace/research/tools/embeddings/cluster_infinite_finetune.py — fine-tune the IQE nucleus with
CLUSTERING + PAWN-CAPTURE INFINITE one-way distance (Kaveh 2026-07-20). A pawn
capture is the canonical no-way-back move; push d(child->parent) toward infinity
(IQE outputs large values for unreachable pairs -- 2211.15120), while pinning
forward d(parent->child) ~ 1 so the scale can't inflate (the failure mode of the
earlier scalar strata hinge). Clustering (symmetry + material separation + DTM
ranking) runs on the near-mate nucleus.

Measures, after: pawn-capture asymmetry d(child->parent)/d(parent->child) (>>1 =
one-way learned) alongside the cluster metrics.

Usage:
  .venv/bin/python catspace/research/tools/embeddings/cluster_infinite_finetune.py --steps 3000 \
    --ckpt data/derived/sep/iqe_nucleus.pt --out data/derived/sep/iqe_infinite.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device, save_ckpt
from scipy.stats import spearmanr
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=paths.sep("iqe_nucleus.pt"))
    ap.add_argument("--nearmate", default=paths.derived("lichess_nearmate.npz"))
    ap.add_argument("--pawncap", default=paths.derived("pawndeath_pairs.npz"))
    ap.add_argument("--out", default=paths.sep("iqe_infinite.pt"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--inf-floor", type=float, default=60.0, help="pawn-capture backward >> this (~infinite)")
    ap.add_argument("--w-inf", type=float, default=1.0)
    ap.add_argument("--w-step", type=float, default=1.0)
    ap.add_argument("--w-sym", type=float, default=1.0)
    ap.add_argument("--w-sep", type=float, default=0.3)
    ap.add_argument("--sep-margin", type=float, default=10.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device)
    torch.manual_seed(args.seed)
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.train()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    opt = torch.optim.Adam(fb.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    # near-mate (clustering) -- won positions + material key
    nz = np.load(args.nearmate)
    won = nz["dtm"] > 0
    nmp, nmm, nmd = nz["packed"][won], nz["meta"][won], nz["dtm"][won].astype(np.float32)
    # material key + pawnless flag in one pass. Material class is a STABLE invariant only
    # when the board is pawnless -- with pawns, promotion bridges material classes
    # (KP->KQ is one move away), so L_sep must NOT separate pawn positions (Kaveh 2026-07-20).
    matkey = np.empty(len(nmd), dtype=object)
    haspawn = np.zeros(len(nmd), dtype=bool)
    for i in range(len(nmd)):
        ps = list(board_from_packed(nmp[i], nmm[i]).piece_map().values())
        matkey[i] = " ".join(sorted(p.symbol() for p in ps))
        haspawn[i] = any(p.piece_type == chess.PAWN for p in ps)
    matkey = matkey.astype(str)
    uniq = {m: k for k, m in enumerate(sorted(set(matkey)))}
    mat = np.array([uniq[m] for m in matkey], dtype=np.int64)
    low = np.argsort(nmd)[:512]
    # pawn-death pairs (one-way / infinity)
    pz = np.load(args.pawncap)

    def eF(pk, mt):
        return fb.embed_F(torch.from_numpy(feature_planes(pk, mt)).to(dev),
                          torch.from_numpy(np.tile(om, (len(pk), 1))).to(dev))

    def eB(pk, mt):
        return fb.embed_B(torch.from_numpy(feature_planes(pk, mt)).to(dev))

    def pawncap_asym():
        i = rng.choice(len(pz["p_packed"]), 300, replace=False)
        with torch.no_grad():
            Fp = eF(pz["p_packed"][i], pz["p_meta"][i]); Bc = eB(pz["c_packed"][i], pz["c_meta"][i])
            Fc = eF(pz["c_packed"][i], pz["c_meta"][i]); Bp = eB(pz["p_packed"][i], pz["p_meta"][i])
            fwd = fb.distance_matrix(Fp, Bc).diagonal().cpu().numpy()
            bwd = fb.distance_matrix(Fc, Bp).diagonal().cpu().numpy()
        return float(np.median(bwd / np.maximum(fwd, 1e-6))), float(np.median(fwd)), float(np.median(bwd))

    a0 = pawncap_asym()
    print(f"[before] pawncap asym d_bwd/d_fwd={a0[0]:.2f} (fwd={a0[1]:.2f} bwd={a0[2]:.2f})", flush=True)
    t0 = time.time()
    for step in range(args.steps):
        if step % 500 == 0:
            with torch.no_grad():
                zmate = eB(nmp[low], nmm[low]).mean(0, keepdim=True).detach()
        # -- clustering on near-mate --
        ni = rng.integers(0, len(nmd), size=args.batch)
        f = eF(nmp[ni], nmm[ni])
        dm = fb.distance_matrix(f, zmate)[:, 0]
        dtm_b = torch.from_numpy(nmd[ni]).to(dev); mat_b = torch.from_numpy(mat[ni]).to(dev)
        same = mat_b[:, None] == mat_b[None, :]
        mask = same & (dtm_b[:, None] < dtm_b[None, :])
        diff = dm[None, :] - dm[:, None]
        L_rank = torch.relu(1.0 - diff)[mask].mean() if mask.any() else torch.zeros((), device=dev)
        nsym = ni[:32]                                                # symmetry on a sub-batch (speed)
        mir = [board_from_packed(nmp[i], nmm[i]).transform(chess.flip_horizontal) for i in nsym]
        fm = eF(np.stack([encode_packed(b) for b in mir]), np.stack([encode_meta(b) for b in mir]))
        L_sym = ((f[:32] - fm) ** 2).sum(1).mean()
        pl = torch.from_numpy(~haspawn[ni]).to(dev)                  # pawnless: material is a hard invariant
        sepm = (~same) & pl[:, None] & pl[None, :]                   # separate ONLY pawnless different-material
        L_sep = torch.relu(args.sep_margin - torch.cdist(f, f)[sepm]).pow(2).mean() if sepm.any() else torch.zeros((), device=dev)
        # -- pawn-capture INFINITE one-way (fused encoder passes: 1 eF over [p;c] and
        #    1 eB over [c;p;s] instead of 2 eF + 3 eB -- ~2x fewer GPU forward passes) --
        pi = rng.integers(0, len(pz["p_packed"]), size=args.batch)
        n = len(pi)
        Fpc = eF(np.concatenate([pz["p_packed"][pi], pz["c_packed"][pi]]),
                 np.concatenate([pz["p_meta"][pi], pz["c_meta"][pi]]))
        Fp, Fc = Fpc[:n], Fpc[n:]
        Bcps = eB(np.concatenate([pz["c_packed"][pi], pz["p_packed"][pi], pz["s_packed"][pi]]),
                  np.concatenate([pz["c_meta"][pi], pz["p_meta"][pi], pz["s_meta"][pi]]))
        Bc, Bp, Bs = Bcps[:n], Bcps[n:2 * n], Bcps[2 * n:]
        d_fwd = fb.distance_matrix(Fp, Bc).diagonal()                 # capture forward ~ 1 ply
        d_step = fb.distance_matrix(Fp, Bs).diagonal()                # unit-step child ~ 1 ply
        d_bwd = fb.distance_matrix(Fc, Bp).diagonal()                 # child -> parent = INFINITE
        L_step = ((d_fwd - 1.0) ** 2).mean() + ((d_step - 1.0) ** 2).mean()   # pin the scale
        L_inf = torch.relu(args.inf_floor - d_bwd).pow(2).mean()      # push backward to ~infinity
        loss = (L_rank + args.w_sym * L_sym + args.w_sep * L_sep
                + args.w_step * L_step + args.w_inf * L_inf)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            with torch.no_grad():
                sp = spearmanr(dm.detach().cpu().numpy(), nmd[ni]).correlation
            print(f"  step {step:4d}  L_rank {float(L_rank):.3f} L_sym {float(L_sym):.3f} "
                  f"L_step {float(L_step):.3f} L_inf {float(L_inf):.3f}  d_fwd {float(d_fwd.median()):.2f} "
                  f"d_bwd {float(d_bwd.median()):.2f}  sp(d,DTM) {sp:+.3f}  ({time.time()-t0:.0f}s)", flush=True)

    fb.eval()
    a1 = pawncap_asym()
    save_ckpt(fb, Path(args.out), step=pay.get("step", 0), zgoals=pay.get("zgoals"))
    print(f"saved {args.out}")
    print(f"VERDICT INFINITE pawncap_asym {a0[0]:.2f}->{a1[0]:.2f} "
          f"(fwd {a0[1]:.2f}->{a1[1]:.2f}, bwd {a0[2]:.2f}->{a1[2]:.2f}; >>1 = one-way learned)")


if __name__ == "__main__":
    main()
