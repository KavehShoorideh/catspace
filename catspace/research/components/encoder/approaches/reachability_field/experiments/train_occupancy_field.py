#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/reachability_field/experiments/train_occupancy_field.py -- the L2 adversarial value field, trained on the
OPTIMAL-PLAY OCCUPANCY (Kaveh 2026-07-20: L1 cooperative reachability is obsolete; embed what
STRONG play does). Pins d(F(a)->B(b)) ~ optimal-play ply-gap on trajectory pairs across ALL
outcomes (wins teach 'distance to mate', losing/drawn lines represent the regions to steer AWAY
from). A contrastive push keeps non-successors far. The result: distance-to-a-terminal-region that
IS the DTM gradient the cooperative field lacked -- winning positions sit close to the White-mate
region, losing ones far from it.

Probe: within-material spearman of d(F(won)->B(near-mate)) vs true DTM -- the number to beat is
the cooperative field's 0.13.
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


def color_flip(pk, mt):
    """mirror (swap colors + flip vertically) -> turns a White-win line into a Black-win line, so
    the losing signal is balanced by symmetry. Uses python-chess Board.mirror()."""
    from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
    out_p, out_m = [], []
    for i in range(len(pk)):
        b = board_from_packed(pk[i], mt[i]).mirror()
        out_p.append(encode_packed(b)); out_m.append(encode_meta(b))
    return np.stack(out_p), np.stack(out_m)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=paths.sep("iqe_nucleus_gn.pt"))
    ap.add_argument("--data", default=paths.derived("optimal_occupancy.npz"))
    ap.add_argument("--probe-data", default=paths.derived("stratified_perfect.npz"))
    ap.add_argument("--out", default=paths.sep("iqe_occupancy.pt"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--push-margin", type=float, default=8.0)
    ap.add_argument("--w-pos", type=float, default=1.0)
    ap.add_argument("--w-push", type=float, default=1.0)
    ap.add_argument("--color-flip", action="store_true", help="augment with mirrored (loss-balancing) pairs")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device); torch.manual_seed(args.seed)
    live_stem = Path(str(paths.experiments_dir())) / (Path(args.out).stem + "_curves")
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.train()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    opt = torch.optim.Adam(fb.parameters(), lr=args.lr); rng = np.random.default_rng(args.seed)

    nz = np.load(args.data, allow_pickle=True)
    AP, AM = np.asarray(nz["a_packed"]), np.asarray(nz["a_meta"])
    BP, BM = np.asarray(nz["b_packed"]), np.asarray(nz["b_meta"])
    GAP, OUT = np.asarray(nz["gap"]).astype(np.float32), np.asarray(nz["outcome"])
    if args.color_flip:
        fap, fam = color_flip(AP, AM); fbp, fbm = color_flip(BP, BM)
        AP = np.concatenate([AP, fap]); AM = np.concatenate([AM, fam])
        BP = np.concatenate([BP, fbp]); BM = np.concatenate([BM, fbm])
        GAP = np.concatenate([GAP, GAP]); OUT = np.concatenate([OUT, -OUT])
    print(f"[stage] {len(AP)} occupancy pairs  W/D/L={int((OUT==1).sum())}/{int((OUT==0).sum())}/"
          f"{int((OUT==-1).sum())}  gap[med={np.median(GAP):.0f} max={GAP.max():.0f}]", flush=True)

    def bp(pk, mt):
        pl = feature_planes(pk, mt); pl[:, BOARD_ONLY] = 0.0
        return torch.from_numpy(pl).to(dev)

    def eF(pk, mt):
        return fb.embed_F(bp(pk, mt), torch.from_numpy(np.tile(om, (len(pk), 1))).to(dev))

    def eB(pk, mt):
        return fb.embed_B(bp(pk, mt))

    # ---- probe: DTM gradient on held-out won positions (beat 0.13) ----
    from scipy.stats import spearmanr
    pz = np.load(args.probe_data, allow_pickle=True)
    PP, PM, PS, PC = (np.asarray(pz["packed"]), np.asarray(pz["meta"]),
                      np.asarray(pz["sdtm"]), np.asarray(pz["pcount"]).astype(int))
    won = np.flatnonzero((PS > 0) & (PC <= 6))
    probe_idx = won[rng.permutation(len(won))[:2500]]          # ONE fixed probe set
    mat = np.array(["".join(sorted(p.symbol() for p in board_from_packed(PP[i], PM[i]).piece_map().values()))
                    for i in probe_idx])

    def probe():
        idx = probe_idx
        with torch.no_grad():
            F = eF(PP[idx], PM[idx])
            nm = idx[np.argsort(PS[idx])[:12]]                 # near-mate anchors of the won set
            B = eB(PP[nm], PM[nm])
            d = fb.distance_matrix(F, B).min(1).values.cpu().numpy()
        overall = spearmanr(d, PS[idx]).correlation
        sps = []
        for mm in set(mat.tolist()):
            s = mat == mm
            if s.sum() >= 40:
                sps.append(spearmanr(d[s], PS[idx][s]).correlation)
        return overall, (float(np.nanmean(sps)) if sps else float("nan"))

    t0 = time.time()
    for step in range(args.steps):
        pi = rng.integers(0, len(AP), size=args.batch)
        d_pos = fb.distance_matrix(eF(AP[pi], AM[pi]), eB(BP[pi], BM[pi])).diagonal()
        gap = torch.from_numpy(GAP[pi]).to(dev)
        L_pos = torch.nn.functional.smooth_l1_loss(d_pos, gap)
        ni = rng.integers(0, len(BP), size=args.batch)         # random non-successor negatives
        d_neg = fb.distance_matrix(eF(AP[pi], AM[pi]), eB(BP[ni], BM[ni])).diagonal()
        L_push = torch.relu(gap.detach() + args.push_margin - d_neg).pow(2).mean()
        loss = args.w_pos * L_pos + args.w_push * L_push
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            ov, permat = probe()
            lp, lpu, dp = float(L_pos.detach()), float(L_push.detach()), float(d_pos.median().detach())
            print(f"  step {step:4d}  Lpos {lp:.3f} Lpush {lpu:.3f}  d_pos {dp:.2f}  "
                  f"DTM-grad overall {ov:+.3f} within-material {permat:+.3f} (beat 0.13)  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            log_and_render(live_stem, step, {"L_pos": lp, "L_push": lpu, "d_pos": dp,
                                             "DTM_grad_overall": ov, "DTM_grad_within_material": permat},
                           title=f"occupancy L2 field ({Path(args.out).stem})")
        if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
            fb.eval(); save_ckpt(fb, Path(args.out), step=pay.get("step", 0), zgoals=pay.get("zgoals")); fb.train()

    fb.eval(); save_ckpt(fb, Path(args.out), step=pay.get("step", 0), zgoals=pay.get("zgoals"))
    ov, permat = probe()
    print(f"saved {args.out}")
    print(f"VERDICT OCC_FIELD DTM_grad_overall={ov:+.3f} within_material={permat:+.3f} (baseline 0.13)")


if __name__ == "__main__":
    main()
