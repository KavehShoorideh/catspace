#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/reachability_field/experiments/cluster_finetune.py — encourage CLUSTER FORMATION in the field
(Kaveh 2026-07-19: the field should embed equivalent positions -- symmetry
variants, and shuffles at the same distance-to-mate -- close, so the subgoal
planner can jump cluster to cluster). The incumbent has NO such structure
(mirror not closer than random; within-DTM = between-DTM). Fine-tune F with:

  L_sym    = || F(pos) - F(horiz-mirror(pos)) ||^2      symmetry-invariance
  L_clust  = pull same-DTM(+material) pairs together, push different-DTM apart
  L_anchor = || F(pos) - F_frozen(pos) ||^2             keep conversion structure

Measures symmetry-invariance + DTM-clustering (within/between ratio) before/after.

Usage:
  .venv/bin/python catspace/research/components/encoder/approaches/reachability_field/experiments/cluster_finetune.py --steps 1500 \
    --out data/derived/sep/cert_base_cluster.pt
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
from catspace.io import paths


def planes_of(boards):
    pk = np.stack([encode_packed(b) for b in boards])
    mt = np.stack([encode_meta(b) for b in boards])
    return feature_planes(pk, mt)


def irrev_child(board):
    """First child via an IRREVERSIBLE move (capture/pawn/promo) -- no way back."""
    for m in board.legal_moves:
        if board.is_irreversible(m):
            c = board.copy(stack=False); c.push(m)
            if not c.is_game_over():
                return c
    return None


def strata_ratio(fb, boards, om, dev):
    """median (irreversible d_bwd/d_fwd) / (reversible d_bwd/d_fwd); >>1 = strata."""
    rev_r, irr_r, fwds = [], [], []
    for b in boards[:200]:
        for m in b.legal_moves:
            c = b.copy(stack=False); c.push(m)
            if c.is_game_over():
                continue
            with torch.no_grad():
                o = torch.from_numpy(np.tile(om, (1, 1))).to(dev)
                Fp = fb.embed_F(torch.from_numpy(planes_of([b])).to(dev), o)
                Bp = fb.embed_B(torch.from_numpy(planes_of([b])).to(dev))
                Fc = fb.embed_F(torch.from_numpy(planes_of([c])).to(dev), o)
                Bc = fb.embed_B(torch.from_numpy(planes_of([c])).to(dev))
                fwd = float(fb.distance_matrix(Fp, Bc)[0, 0])
                bwd = float(fb.distance_matrix(Fc, Bp)[0, 0])
            fwds.append(fwd)
            (irr_r if b.is_irreversible(m) else rev_r).append(bwd / max(fwd, 1e-6))
            break
    r = np.median(rev_r) if rev_r else 1.0
    i = np.median(irr_r) if irr_r else 1.0
    return i / max(r, 1e-6), np.median(fwds)


def cluster_metrics(fb, boards, dtm, om, dev):
    """symmetry-invariance ratio + DTM within/between ratio (higher=more clustered)."""
    with torch.no_grad():
        pl = torch.from_numpy(planes_of(boards)).to(dev)
        o = torch.from_numpy(np.tile(om, (len(boards), 1))).to(dev)
        emb = fb.embed_F(pl, o).cpu().numpy()
        mir = [b.transform(chess.flip_horizontal) for b in boards]
        embm = fb.embed_F(torch.from_numpy(planes_of(mir)).to(dev), o).cpu().numpy()
    perm = np.random.default_rng(0).permutation(len(emb))
    d_sym = np.linalg.norm(emb - embm, axis=1).mean()
    d_rand = np.linalg.norm(emb - emb[perm], axis=1).mean()
    within, between = [], []
    rng = np.random.default_rng(1)
    for k in np.unique(dtm):
        ii = np.flatnonzero(dtm == k)
        if len(ii) < 2:
            continue
        for a in ii[:30]:
            oth = ii[ii != a]
            within.append(np.linalg.norm(emb[a] - emb[rng.choice(oth)]))
            dif = np.flatnonzero(dtm != k)
            between.append(np.linalg.norm(emb[a] - emb[rng.choice(dif)]))
    return d_rand / d_sym, np.mean(between) / np.mean(within)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=paths.sep("cert_base_full.pt"))
    ap.add_argument("--dtm-npz", default=paths.derived("dtm_endgame.npz"))
    ap.add_argument("--out", default=paths.sep("cert_base_cluster.pt"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-sym", type=float, default=1.0)
    ap.add_argument("--w-clust", type=float, default=1.0)
    ap.add_argument("--w-anchor", type=float, default=1.0)
    ap.add_argument("--w-strata", type=float, default=0.0,
                    help="LOCAL strata hinge: for an irreversible move (parent->child), "
                         "push d(child->parent) above --strata-floor (no way back)")
    ap.add_argument("--strata-floor", type=float, default=None,
                    help="floor for the irreversible backward distance (default 8x the "
                         "field's median forward step)")
    ap.add_argument("--margin", type=float, default=0.6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device)
    torch.manual_seed(args.seed)
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.train()
    fb_frozen, _ = load_ckpt(Path(args.ckpt), dev); fb_frozen.eval()
    for p in fb_frozen.parameters():
        p.requires_grad_(False)
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    dz = np.load(args.dtm_npz)
    dtm_all, mat_all = dz["dtm"].astype(np.float32), dz["material"]
    rng = np.random.default_rng(args.seed)

    # eval set (held-out, krvk for the symmetry check clarity)
    ev = np.flatnonzero(mat_all == 2)[:400]
    ev_boards = [board_from_packed(dz["packed"][i], dz["meta"][i]) for i in ev]
    sym0, clu0 = cluster_metrics(fb_frozen, ev_boards, dtm_all[ev], om, dev)
    str0, med_fwd = strata_ratio(fb_frozen, ev_boards, om, dev)
    print(f"[before] symmetry_ratio={sym0:.2f} (mirror vs random; >1 better)  "
          f"dtm_clustering={clu0:.2f} (between/within; >1 better)  "
          f"strata_ratio={str0:.2f} (irrev/rev asym; >>1 better)", flush=True)
    floor = args.strata_floor if args.strata_floor is not None else 8.0 * med_fwd
    if args.w_strata > 0:
        print(f"[strata] floor = {floor:.2f} (irreversible backward >> this)", flush=True)

    opt = torch.optim.Adam(fb.parameters(), lr=args.lr)
    t0 = time.time()
    for step in range(args.steps):
        idx = rng.integers(0, len(dtm_all), size=args.batch)
        boards = [board_from_packed(dz["packed"][i], dz["meta"][i]) for i in idx]
        pl = torch.from_numpy(planes_of(boards)).to(dev)
        o = torch.from_numpy(np.tile(om, (len(boards), 1))).to(dev)
        f = fb.embed_F(pl, o)
        # symmetry
        mir = [b.transform(chess.flip_horizontal) for b in boards]
        fm = fb.embed_F(torch.from_numpy(planes_of(mir)).to(dev), o)
        L_sym = ((f - fm) ** 2).sum(1).mean()
        # anchor
        with torch.no_grad():
            f0 = fb_frozen.embed_F(pl, o)
        L_anchor = ((f - f0) ** 2).sum(1).mean()
        # DTM clustering: pull same-dtm+material close, push diff apart (margin)
        dtm_b = torch.from_numpy(dtm_all[idx]).to(dev)
        mat_b = torch.from_numpy(mat_all[idx].astype(np.int64)).to(dev)
        D = torch.cdist(f, f)                                   # (B,B)
        same = (dtm_b[:, None] == dtm_b[None, :]) & (mat_b[:, None] == mat_b[None, :])
        same.fill_diagonal_(False)
        diff = ~same
        diff.fill_diagonal_(False)
        L_clust = (D[same] ** 2).mean() if same.any() else torch.zeros((), device=dev)
        L_clust = L_clust + torch.relu(args.margin - D[diff]).pow(2).mean()
        # STRATA: irreversible move (parent->child) => d(child->parent) must be HUGE
        L_strata = torch.zeros((), device=dev)
        if args.w_strata > 0:
            children = [irrev_child(b) for b in boards]
            ki = [j for j, c in enumerate(children) if c is not None]
            if ki:
                par = [boards[j] for j in ki]; chi = [children[j] for j in ki]
                op = torch.from_numpy(np.tile(om, (len(par), 1))).to(dev)
                Fc = fb.embed_F(torch.from_numpy(planes_of(chi)).to(dev), op)
                Bp = fb.embed_B(torch.from_numpy(planes_of(par)).to(dev))
                d_bwd = fb.distance_matrix(Fc, Bp).diagonal()      # child -> parent
                L_strata = torch.relu(floor - d_bwd).pow(2).mean()
        loss = (args.w_sym * L_sym + args.w_clust * L_clust + args.w_anchor * L_anchor
                + args.w_strata * L_strata)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  L_sym {float(L_sym):.4f} L_clust {float(L_clust):.4f} "
                  f"L_anchor {float(L_anchor):.4f} L_strata {float(L_strata):.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    fb.eval()
    sym1, clu1 = cluster_metrics(fb, ev_boards, dtm_all[ev], om, dev)
    str1, _ = strata_ratio(fb, ev_boards, om, dev)
    save_ckpt(fb, Path(args.out), step=pay.get("step", 0), zgoals=pay.get("zgoals"),
              provenance=pay.get("provenance"))
    print(f"saved {args.out}")
    print(f"VERDICT CLUSTER symmetry_ratio {sym0:.2f}->{sym1:.2f}  "
          f"dtm_clustering {clu0:.2f}->{clu1:.2f}  strata_ratio {str0:.2f}->{str1:.2f} "
          f"(all >1 and rising => structure formed)")


if __name__ == "__main__":
    main()
