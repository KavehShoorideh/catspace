#!/usr/bin/env python
"""probe_value_proxy.py -- how good is the committor as a value estimate? (Kaveh 2026-08-15:
"how do we estimate this value currently? using the probability block of jqt6? how good of a
proxy is that?")

WHAT WE CURRENTLY CALL VALUE. The field has no value head. It infers one from GEOMETRY:

    dB3 = dB(z_B, pole_k)  for k in (WIN, DRAW, LOSS)
    prb = softmax(-dB3 / basin_temp)                  # the "probability block"
    E   = prb[WIN] + 0.5 * prb[DRAW]                  # white-POV expected score

So E is a softmax over distances to three learned poles, not a calibrated probability of
anything. This measures how well it tracks Stockfish's WDL expectation on the SAME positions.

TWO QUESTIONS, AND THEY ARE NOT THE SAME ONE:
  GLOBAL   does E rank all positions correctly? (easy -- "winning positions look winning" is
           mostly material, and a constant-ish material readout scores well here)
  WITHIN-PARENT  among the legal children of ONE position, does E rank the moves correctly?
           This is the one planning needs, and it is where a global correlation can hide a
           proxy that is useless for choosing a move.

POV bookkeeping (the easy bug): our E is WHITE-POV (basin_pov=white). The shards store each
child's wdl from the CHILD MOVER's POV and the code converts to PARENT-mover POV. At a child,
the parent mover is whoever is NOT to move, so parent-mover is white iff child glob[0]==0.

    .venv/bin/python -m ...probe_value_proxy --gen jqt6 --parents 3000
"""
from __future__ import annotations

import argparse
import glob as _glob
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", default="jqt6")
    ap.add_argument("--parents", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.src.reach_vit_jepa \
        import ReachViT

    ck = torch.load(f"artifacts/experiments/reach_{args.gen}_latest.pt", map_location="cpu",
                    weights_only=False)
    A = ck["args"]; A = vars(A) if not isinstance(A, dict) else A
    net = ReachViT(split_head=bool(A["split_head"]), move_head=bool(A["move_head"]),
                   proj_mlp=int(A.get("proj_mlp", 0)), d_model=A["d_model"], layers=A["layers"],
                   heads=A["heads"], d=A["d"], hidden=A["hidden"],
                   components=A["components"], ema_decay=A["ema"], dual=False)
    net.attach_poles(torch.tensor(ck["cfg"]["pole_parent"]), n_sources=1,
                     fixed=not ck["cfg"].get("learned_poles"),
                     height=ck["cfg"].get("pole_height", 3.0))
    miss = net.load_state_dict(ck["state_dict"], strict=False).missing_keys
    assert not miss, miss[:5]
    net.eval()
    btemp = float(A.get("basin_temp", 5.0))
    print(f"[value] {args.gen} step {ck.get('step')}, basin_temp {btemp}", flush=True)

    # ---- merge the WDL shards ---------------------------------------------------------
    # Offsets are PER SHARD, so they cannot simply be concatenated. Rebuild them globally from
    # each shard's per-parent child COUNTS, in the same sorted file order as the payload arrays.
    rows, toks, globs, wdls, cnt = [], [], [], [], []
    for f in sorted(_glob.glob("data/derived/wdl_labels/*.npz")):
        z = np.load(f)
        rows.append(z["row"]); toks.append(z["tok"]); globs.append(z["glob"])
        wdls.append(z["wdl"]); cnt.append(np.diff(z["off"]))
    rows, toks, globs, wdls = (np.concatenate(rows), np.concatenate(toks),
                               np.concatenate(globs), np.concatenate(wdls))
    off = np.concatenate([[0], np.cumsum(np.concatenate(cnt))]).astype(np.int64)
    assert off[-1] == len(toks), f"offset/child mismatch {off[-1]} vs {len(toks)}"
    print(f"[value] {len(rows):,} labelled parents, {len(toks):,} children", flush=True)

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(rows), size=min(args.parents, len(rows)), replace=False)

    ct, cg, ce, cp = [], [], [], []          # child tok/glob, SF E (white POV), parent id
    for j, pi in enumerate(pick):
        a, b = int(off[pi]), int(off[pi + 1])
        if b - a < 2:
            continue
        w = wdls[a:b].astype(np.float32)
        tot = np.clip(w.sum(1), 1.0, None)
        e_parent = (w[:, 2] + 0.5 * w[:, 1]) / tot        # parent-mover POV, as in training
        g = globs[a:b]
        # parent mover is white  <=>  child side-to-move is black  <=>  glob[0] == 0
        white_moved = (g[:, 0] == 0)
        e_white = np.where(white_moved, e_parent, 1.0 - e_parent)
        ct.append(toks[a:b]); cg.append(g); ce.append(e_white)
        cp.append(np.full(b - a, j, np.int64))
    CT = torch.from_numpy(np.concatenate(ct).astype(np.int64))
    CG = torch.from_numpy(np.concatenate(cg).astype(np.float32))
    CE = np.concatenate(ce); CP = np.concatenate(cp)
    print(f"[value] {len(CT):,} children from {len(np.unique(CP)):,} parents", flush=True)

    # ---- our estimate: the probability block -------------------------------------------
    t0 = time.time()
    POL = net.poles.poles
    outs = []
    with torch.no_grad():
        for a in range(0, len(CT), args.batch):
            phi = net.backbone(CT[a:a + args.batch], CG[a:a + args.batch])
            zb = net.proj_b(phi)
            dB3 = torch.stack([net.dB(zb, POL[[k]].expand(len(zb), -1)) for k in range(3)], 1)
            prb = torch.softmax(-dB3 / btemp, 1)
            outs.append((prb[:, 0] + 0.5 * prb[:, 1]).cpu().numpy())
    OURS = np.concatenate(outs)
    print(f"[value] embedded [{time.time()-t0:.0f}s]", flush=True)

    # ---- material baseline: the thing any proxy must beat -------------------------------
    VAL = np.array([0, 1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, 0], np.float32)  # tok id -> white cp
    MAT = VAL[CT.numpy()].sum(1)

    def rank_corr(a, b):
        from scipy.stats import spearmanr, pearsonr
        return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])

    pr_o, sp_o = rank_corr(OURS, CE)
    pr_m, sp_m = rank_corr(MAT, CE)
    print(f"\n  GLOBAL (all {len(CE):,} children, target = Stockfish white-POV E)")
    print(f"    committor E   pearson {pr_o:+.3f}  spearman {sp_o:+.3f}")
    print(f"    material only pearson {pr_m:+.3f}  spearman {sp_m:+.3f}   <- the bar to beat")
    print(f"    committor E   range [{OURS.min():.3f}, {OURS.max():.3f}] "
          f"mean {OURS.mean():.3f} sd {OURS.std():.3f}")
    print(f"    stockfish E   range [{CE.min():.3f}, {CE.max():.3f}] "
          f"mean {CE.mean():.3f} sd {CE.std():.3f}")
    print(f"    MAE {np.abs(OURS - CE).mean():.4f}   "
          f"MAE of predicting the constant mean {np.abs(CE - CE.mean()).mean():.4f}")

    # ---- WITHIN-PARENT: the metric planning actually needs ------------------------------
    from scipy.stats import spearmanr
    sps, tops, ns = [], [], []
    for j in np.unique(CP):
        m = CP == j
        if m.sum() < 3 or np.ptp(CE[m]) < 1e-6:
            continue
        s = spearmanr(OURS[m], CE[m])[0]
        if np.isfinite(s):
            sps.append(s)
        tops.append(float(np.argmax(OURS[m]) == np.argmax(CE[m])))
        ns.append(int(m.sum()))
    print(f"\n  WITHIN-PARENT (ranking the legal moves of one position -- what planning needs)")
    print(f"    mean spearman over {len(sps):,} parents : {np.mean(sps):+.3f} "
          f"(sd {np.std(sps):.3f})")
    print(f"    top-1 agreement with Stockfish's best   : {np.mean(tops):.1%} "
          f"(random ~{np.mean([1/n for n in ns]):.1%})")
    # ---- RESOLUTION vs DIRECTION: does E even MOVE between siblings? --------------------
    # A near-zero within-parent correlation has two very different causes and they need
    # different fixes: the estimate moves the WRONG WAY (direction problem -- retrain), or it
    # barely moves at all between one-ply-apart positions (RESOLUTION problem -- the smooth
    # geometry cannot separate siblings, and no amount of training on this readout fixes it).
    so, sc = [], []
    for j in np.unique(CP):
        m = CP == j
        if m.sum() < 3:
            continue
        so.append(OURS[m].std()); sc.append(CE[m].std())
    print(f"\n  RESOLUTION (spread among the children of ONE parent)")
    print(f"    committor E  mean within-parent sd : {np.mean(so):.4f}")
    print(f"    stockfish E  mean within-parent sd : {np.mean(sc):.4f}")
    print(f"    ratio ours/SF                      : {np.mean(so)/max(np.mean(sc),1e-9):.3f}"
          f"   (<<1 = cannot separate siblings at all)")
    print(f"\n[value] READ: a high GLOBAL number with a ~0 WITHIN-PARENT number means the "
          f"estimate tracks material, not the move choice.")


if __name__ == "__main__":
    main()
