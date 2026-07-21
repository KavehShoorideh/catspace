#!/usr/bin/env python
"""experiments/train_geometry_min.py -- the MINIMAL L1 reachability geometry
(Kaveh 2026-07-20). ONLY two terms; strata / material clusters / multi-step distances
all EMERGE from them + the IQE triangle inequality:

  L_pos   d(F(s) -> B(s')) ~ 1        legal 1-ply successors (one-directional PER EDGE;
                                      a reversible move's reverse is ITS OWN edge -> also
                                      pinned -> symmetric; an irreversible move's reverse
                                      is NOT an edge -> never pinned).
  L_push  d(F(a) -> B(b)) >= floor    INDEPENDENTLY-sampled random a,b (huge), EXCLUDING
                                      pairs where b is a TRUE 1-ply successor of a (those
                                      must stay ~1) and a==b. Exact via precomputed
                                      board-only successor key-sets (Kaveh 2026-07-20).

The critical fix vs the old L_neg: a and b are sampled INDEPENDENTLY, so child-F/parent-B
pairs (the reverse of an irreversible move) actually get pushed -- the old in-batch
off-diagonal only ever touched FORWARD (parent-F/child-B) pairs, which is why strata never
emerged. NO DTM ranking, mate pole, separation, or irreversibility term: all of
DTM/outcomes moves to the L2 categorical head. Board-only geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed, encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device, save_ckpt

BOARD_ONLY = (18, 19)


def _bokey(packed, meta) -> bytes:
    """Board-only identity: pieces + turn/castling/ep (meta[:6]), EXCLUDING clock/rep."""
    return hashlib.blake2b(packed.tobytes() + np.asarray(meta)[:6].tobytes(), digest_size=8).digest()


def _succ_worker(task):
    packed, meta = task
    out = []
    for i in range(len(packed)):
        b = board_from_packed(packed[i], meta[i])
        ks = set()
        for m in b.legal_moves:
            c = b.copy(stack=False); c.push(m)
            ks.add(_bokey(encode_packed(c), encode_meta(c)))
        out.append(frozenset(ks))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/sep/iqe_nucleus_gn.pt")
    ap.add_argument("--data", default="data/derived/lichess_nearmate.npz")
    ap.add_argument("--edges", default="data/derived/successor_edges_all.npz")
    ap.add_argument("--out", default="data/derived/sep/iqe_geom_min.pt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--neg-margin", type=float, default=20.0,
                    help="RELATIVE: d(anchor->random) >= d(anchor->successor)+this (triplet, anchored)")
    ap.add_argument("--w-pos", type=float, default=1.0)
    ap.add_argument("--w-push", type=float, default=1.0)
    ap.add_argument("--w-sym", type=float, default=0.0, help="optional mirror-invariance (0 = pure two-term)")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device)
    torch.manual_seed(args.seed)
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.train()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    opt = torch.optim.Adam(fb.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    nz = np.load(args.data)
    allp, allm = np.asarray(nz["packed"]), np.asarray(nz["meta"])   # ALL positions (materialize once)
    _ez = np.load(args.edges)
    EPK, EPM = np.asarray(_ez["p_packed"]), np.asarray(_ez["p_meta"])   # materialize (npz is lazy -> O(n^2))
    ECK, ECM = np.asarray(_ez["c_packed"]), np.asarray(_ez["c_meta"])
    ez = {"p_packed": EPK, "p_meta": EPM, "c_packed": ECK, "c_meta": ECM}

    # exact successor key-sets for push-masking (parallel precompute)
    t0 = time.time()
    W = max(1, args.workers)
    bnd = np.linspace(0, len(allp), W + 1, dtype=int)
    tasks = [(allp[bnd[i]:bnd[i + 1]], allm[bnd[i]:bnd[i + 1]]) for i in range(W) if bnd[i + 1] > bnd[i]]
    succ_sets = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for r in ex.map(_succ_worker, tasks):
            succ_sets.extend(r)
    allp_hash = [_bokey(allp[i], allm[i]) for i in range(len(allp))]
    hash_to_idx = {h: i for i, h in enumerate(allp_hash)}          # board-only key -> allp index
    ep_idx = np.array([hash_to_idx[_bokey(ez["p_packed"][i], ez["p_meta"][i])]
                       for i in range(len(ez["p_packed"]))])       # each edge-parent's allp index
    print(f"[stage] {len(allp)} positions, {len(ez['p_packed'])} edges; "
          f"successor key-sets precomputed in {time.time()-t0:.0f}s", flush=True)

    def bp(pk, mt):
        pl = feature_planes(pk, mt); pl[:, BOARD_ONLY] = 0.0
        return torch.from_numpy(pl).to(dev)

    def eF(pk, mt):
        return fb.embed_F(bp(pk, mt), torch.from_numpy(np.tile(om, (len(pk), 1))).to(dev))

    def eB(pk, mt):
        return fb.embed_B(bp(pk, mt))

    # -- probes (watch strata + cross-material emerge) --------------------
    won = np.flatnonzero(nz["dtm"] > 0); dtm = nz["dtm"].astype(np.float32)
    mk = np.array(["".join(sorted(p.symbol() for p in board_from_packed(allp[i], allm[i]).piece_map().values()))
                   for i in won])

    def probe():
        rp, rc, ip, ic = [], [], [], []
        for j in rng.choice(won, 4000, replace=False):
            b = board_from_packed(allp[j], allm[j])
            if b.is_game_over():
                continue
            for m in b.legal_moves:
                c = b.copy(stack=False); c.push(m)
                if c.is_game_over():
                    continue
                if b.is_irreversible(m) and len(ip) < 150:
                    ip.append(b.copy(stack=False)); ic.append(c)
                elif not b.is_irreversible(m) and len(rp) < 150:
                    rp.append(b.copy(stack=False)); rc.append(c)
                break
            if len(ip) >= 150 and len(rp) >= 150:
                break

        def asym(P, C):
            pp = np.stack([encode_packed(x) for x in P]); pm = np.stack([encode_meta(x) for x in P])
            cp = np.stack([encode_packed(x) for x in C]); cm = np.stack([encode_meta(x) for x in C])
            with torch.no_grad():
                f = fb.distance_matrix(eF(pp, pm), eB(cp, cm)).diagonal().cpu().numpy()
                b = fb.distance_matrix(eF(cp, cm), eB(pp, pm)).diagonal().cpu().numpy()
            return float(np.median(f)), float(np.median(b))
        rf, rb_ = asym(rp, rc); irf, irb = asym(ip, ic)
        src = won[mk == "KRk"]; tgt = won[mk == "KQk"]
        xm = float("nan")
        if len(src) > 5 and len(tgt) > 5:
            s = src[np.argsort(-dtm[src])[:40]]; t = tgt[np.argsort(dtm[tgt])[:40]]
            with torch.no_grad():
                xm = float(np.median(fb.distance_matrix(eF(allp[s], allm[s]), eB(allp[t], allm[t])).min(1).values.cpu().numpy()))
        return rb_ / max(rf, 1e-6), irb / max(irf, 1e-6), xm

    t0 = time.time()
    for step in range(args.steps):
        # anchor = edge parent; positive = its child (pin d~1); negative = a random
        # non-successor (push d >= d_pos + margin). Same anchor F(parent), RELATIVE.
        ei = rng.integers(0, len(ez["p_packed"]), size=args.batch)
        Fp = eF(ez["p_packed"][ei], ez["p_meta"][ei])
        d_pos = fb.distance_matrix(Fp, eB(ez["c_packed"][ei], ez["c_meta"][ei])).diagonal()
        L_pos = ((d_pos - 1.0) ** 2).mean()
        pidx = ep_idx[ei]                                          # parents' allp indices
        rb = rng.integers(0, len(allp), size=args.batch)          # random negatives
        mask = np.array([rb[i] == pidx[i] or (allp_hash[rb[i]] in succ_sets[pidx[i]]) for i in range(len(rb))])
        keep = torch.from_numpy(~mask).to(dev)
        d_neg = fb.distance_matrix(Fp, eB(allp[rb], allm[rb])).diagonal()
        L_push = torch.relu(d_pos.detach() + args.neg_margin - d_neg)[keep].pow(2).mean()
        loss = args.w_pos * L_pos + args.w_push * L_push
        D, keep_ = d_neg, keep   # for logging
        if args.w_sym > 0:
            ni = rng.integers(0, len(won), size=64); wi = won[ni]
            f = eF(allp[wi], allm[wi])
            mir = [board_from_packed(allp[i], allm[i]).transform(chess.flip_horizontal) for i in wi]
            fm = eF(np.stack([encode_packed(b) for b in mir]), np.stack([encode_meta(b) for b in mir]))
            loss = loss + args.w_sym * ((f - fm) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            rev, irr, xm = probe()
            n_excl = int(mask.sum())
            print(f"  step {step:4d}  L_pos {float(L_pos):.3f} L_push {float(L_push):.3f}  "
                  f"d_pos {float(d_pos.median()):.2f} d_push {float(D[keep].median()):.1f}  "
                  f"asym rev {rev:.2f} irr {irr:.2f} (sep {irr/max(rev,1e-6):.2f}x)  "
                  f"KRk->KQk {xm:.1f}  excl {n_excl}  ({time.time()-t0:.0f}s)", flush=True)
        if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
            fb.eval(); save_ckpt(fb, Path(args.out), step=pay.get("step", 0), zgoals=pay.get("zgoals")); fb.train()

    fb.eval()
    save_ckpt(fb, Path(args.out), step=pay.get("step", 0), zgoals=pay.get("zgoals"))
    rev, irr, xm = probe()
    print(f"saved {args.out}")
    print(f"VERDICT GEOM_MIN d_pos~1 asym_sep={irr/max(rev,1e-6):.2f}x (irr {irr:.2f}/rev {rev:.2f}) "
          f"KRk->KQk={xm:.1f} (should be HUGE = unreachable)")


if __name__ == "__main__":
    main()
