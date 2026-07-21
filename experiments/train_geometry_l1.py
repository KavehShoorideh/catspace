#!/usr/bin/env python
"""experiments/train_geometry_l1.py -- the L1 reachability geometry with TARGETED
negatives (Kaveh 2026-07-20). The pure-random-push objective was refuted: TURN PARITY
means a move's reverse is never a 1-ply edge, so a huge random push can't tell a
reversible reverse (reachable in a few plies) from an irreversible one (unreachable) and
inflates both. Strata + cross-material need TARGETED negatives:

  L_pos    d(F(s)->B(s')) ~ 1                    legal 1-ply successors (forward hinge)
  L_hard   d(F(child)->B(parent)) >= 1+margin    ONLY for IRREVERSIBLE edges (is_irreversible)
                                                 -> the one-way strata (reversible reverses,
                                                 not pushed, stay low via composition)
  L_repel  d(F(a)->B(b)) >= floor                ONLY for MATERIAL-UNREACHABLE random pairs
                                                 (count-vector reachability) -> cross-material

No huge general push (it destroys reversible reverses). Board-only geometry, pool coverage
(nucleus + children). NO DTM -- that moves to the L2 categorical head.
"""
from __future__ import annotations
import argparse, hashlib, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import chess, numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.data.encode import board_from_packed, encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device, save_ckpt

BOARD_ONLY = (18, 19)
NONKING = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]   # packed planes: white PNBRQ, black pnbrq (skip kings 5,11)


def count_vectors(packed):
    b = packed.astype(np.uint64).view(np.uint8).reshape(len(packed), 12, 8)
    return np.unpackbits(b, axis=2).sum(2).astype(np.int16)[:, NONKING]   # (N,10)


def reach_mask(A, B):
    """Can material B be reached from material A? Can't gain pawns; non-pawn pieces gained
    must be <= pawns available to promote. A,B: (n,10) [Pw,Nw,Bw,Rw,Qw, Pb,Nb,Bb,Rb,Qb]."""
    okp = (B[:, 0] <= A[:, 0]) & (B[:, 5] <= A[:, 5])
    addw = np.maximum(0, B[:, 1:5] - A[:, 1:5]).sum(1)
    addb = np.maximum(0, B[:, 6:10] - A[:, 6:10]).sum(1)
    return okp & (addw <= A[:, 0] - B[:, 0]) & (addb <= A[:, 5] - B[:, 5])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/sep/iqe_nucleus_gn.pt")
    ap.add_argument("--data", default="data/derived/geom_pool.npz")
    ap.add_argument("--edges", default="data/derived/geom_pool_edges.npz")
    ap.add_argument("--out", default="data/derived/sep/iqe_geom_l1.pt")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hard-margin", type=float, default=15.0)
    ap.add_argument("--repel-floor", type=float, default=30.0)
    ap.add_argument("--w-pos", type=float, default=2.0)
    ap.add_argument("--w-hard", type=float, default=1.0)
    ap.add_argument("--w-repel", type=float, default=1.0)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device); torch.manual_seed(args.seed)
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.train()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    opt = torch.optim.Adam(fb.parameters(), lr=args.lr); rng = np.random.default_rng(args.seed)

    nz = np.load(args.data)
    allp, allm, dtm = np.asarray(nz["packed"]), np.asarray(nz["meta"]), nz["dtm"].astype(np.float32)
    _ez = np.load(args.edges)
    EPK, EPM = np.asarray(_ez["p_packed"]), np.asarray(_ez["p_meta"])
    ECK, ECM = np.asarray(_ez["c_packed"]), np.asarray(_ez["c_meta"])
    EIR = np.asarray(_ez["irrev"]).astype(bool)
    pcv = count_vectors(allp)
    print(f"[stage] {len(allp)} pool positions, {len(EPK)} edges ({int(EIR.mean()*100)}% irrev)", flush=True)

    def bp(pk, mt):
        pl = feature_planes(pk, mt); pl[:, BOARD_ONLY] = 0.0
        return torch.from_numpy(pl).to(dev)

    def eF(pk, mt):
        return fb.embed_F(bp(pk, mt), torch.from_numpy(np.tile(om, (len(pk), 1))).to(dev))

    def eB(pk, mt):
        return fb.embed_B(bp(pk, mt))

    won = np.flatnonzero(dtm > 0)
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
                (ip if b.is_irreversible(m) else rp).append(b.copy(stack=False))
                (ic if b.is_irreversible(m) else rc).append(c)
                break
            if len(ip) >= 150 and len(rp) >= 150:
                break

        def asym(P, C):
            pp = np.stack([encode_packed(x) for x in P]); pm = np.stack([encode_meta(x) for x in P])
            cp = np.stack([encode_packed(x) for x in C]); cm = np.stack([encode_meta(x) for x in C])
            with torch.no_grad():
                f = fb.distance_matrix(eF(pp, pm), eB(cp, cm)).diagonal().cpu().numpy()
                b = fb.distance_matrix(eF(cp, cm), eB(pp, pm)).diagonal().cpu().numpy()
            return float(np.median(b / np.maximum(f, 1e-6)))
        rev = asym(rp[:150], rc[:150]); irr = asym(ip[:150], ic[:150])
        src = won[mk == "KRk"]; tgt = won[mk == "KQk"]; xm = float("nan")
        if len(src) > 5 and len(tgt) > 5:
            with torch.no_grad():
                xm = float(np.median(fb.distance_matrix(eF(allp[src[:40]], allm[src[:40]]),
                                                        eB(allp[tgt[:40]], allm[tgt[:40]])).min(1).values.cpu().numpy()))
        return rev, irr, xm

    t0 = time.time()
    for step in range(args.steps):
        ei = rng.integers(0, len(EPK), size=args.batch)
        d_pos = fb.distance_matrix(eF(EPK[ei], EPM[ei]), eB(ECK[ei], ECM[ei])).diagonal()
        L_pos = ((d_pos - 1.0) ** 2).mean()
        irr = EIR[ei]
        if irr.any():
            ci = ei[irr]
            d_bwd = fb.distance_matrix(eF(ECK[ci], ECM[ci]), eB(EPK[ci], EPM[ci])).diagonal()
            L_hard = torch.relu(1.0 + args.hard_margin - d_bwd).pow(2).mean()
        else:
            L_hard = torch.zeros((), device=dev)
        ra = rng.integers(0, len(allp), size=args.batch); rb = rng.integers(0, len(allp), size=args.batch)
        unreach = ~reach_mask(pcv[ra], pcv[rb])
        d_cross = fb.distance_matrix(eF(allp[ra], allm[ra]), eB(allp[rb], allm[rb])).diagonal()
        keep = torch.from_numpy(unreach).to(dev)
        L_repel = torch.relu(args.repel_floor - d_cross)[keep].pow(2).mean() if unreach.any() else torch.zeros((), device=dev)
        loss = args.w_pos * L_pos + args.w_hard * L_hard + args.w_repel * L_repel
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            rev, irr_a, xm = probe()
            print(f"  step {step:4d}  L_pos {float(L_pos):.3f} L_hard {float(L_hard):.3f} L_repel {float(L_repel):.3f}  "
                  f"d_pos {float(d_pos.median()):.2f}  asym rev {rev:.2f} irr {irr_a:.2f} (sep {irr_a/max(rev,1e-6):.2f}x)  "
                  f"KRk->KQk {xm:.1f}  ({time.time()-t0:.0f}s)", flush=True)
        if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
            fb.eval(); save_ckpt(fb, Path(args.out), step=pay.get("step", 0), zgoals=pay.get("zgoals")); fb.train()

    fb.eval(); save_ckpt(fb, Path(args.out), step=pay.get("step", 0), zgoals=pay.get("zgoals"))
    rev, irr_a, xm = probe()
    print(f"saved {args.out}")
    print(f"VERDICT GEOM_L1 asym_sep={irr_a/max(rev,1e-6):.2f}x (irr {irr_a:.2f}/rev {rev:.2f}) KRk->KQk={xm:.1f}")


if __name__ == "__main__":
    main()
