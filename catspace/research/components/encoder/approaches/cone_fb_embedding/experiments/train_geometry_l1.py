#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/cone_fb_embedding/experiments/train_geometry_l1.py -- the L1 reachability geometry with TARGETED
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
from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device, save_ckpt
from catspace.research.tools.stats_eval.tracking import track_run
from catspace.io import paths

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
    ap.add_argument("--ckpt", default=paths.sep("iqe_nucleus_gn.pt"))
    ap.add_argument("--data", default=paths.derived("geom_pool.npz"))
    ap.add_argument("--edges", default=paths.derived("geom_pool_edges.npz"))
    ap.add_argument("--out", default=paths.sep("iqe_geom_l1.pt"))
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hard-margin", type=float, default=15.0)
    ap.add_argument("--repel-floor", type=float, default=30.0)
    ap.add_argument("--repel-floor-all", type=float, default=12.0,
                    help="ANTI-COLLAPSE (Kaveh 2026-07-21): floor pushing EVERY random pair apart, not just "
                         "material-unreachable ones. Without this the within-material bulk collapses "
                         "(rank 6/512, d_step~=d_rand). 0 disables the general tier.")
    ap.add_argument("--w-pos", type=float, default=2.0)
    ap.add_argument("--w-hard", type=float, default=1.0)
    ap.add_argument("--w-repel", type=float, default=4.0)   # was 1.0 -- too weak to beat L_pos's pull-to-1
    ap.add_argument("--w-dtm", type=float, default=1.0,
                    help="DTM ALIGNMENT (Kaveh 2026-07-21): regress d(F(s),MATE_W)->dtm/scale on won positions "
                         "(the mate 'direction' half -- geometry gives sharpness, this gives direction). 0 disables.")
    ap.add_argument("--dtm-scale", type=float, default=20.0)
    ap.add_argument("--bp-shards", default=None,
                    help="best-play continuation shards dir (e.g. data/shards/sf_cont_endgame_v1): consecutive "
                         "plies are added as OPTIMAL-successor L_pos edges (best-play geometry, not just legal).")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--contrast-npz", default=None,
                    help="matched-anchor contrast tuples (gen_contrast_mate_tuples.py): directed-vs-random "
                         "branches from the SAME anchor -> hinge d(F(pos_t),B(mate)) + m*t < d(F(neg_t),B(mate)). "
                         "Material is matched by construction, so only STRUCTURE-of-progress can satisfy it "
                         "(Kaveh 2026-07-22: separate distance-to-mate from piece count).")
    ap.add_argument("--w-contrast", type=float, default=1.0)
    ap.add_argument("--contrast-margin", type=float, default=1.0,
                    help="margin per ply of depth (true DTM gap grows ~2t; 1.0*t is conservative)")
    ap.add_argument("--w-rev", type=float, default=0.0,
                    help="ASYM FIX (JOURNAL 2026-07-22 inversion): pull REVERSIBLE-edge reverses "
                         "into the cheap band d <= rev-cap (turn parity makes them ~2-3 plies via "
                         "triangulation, but the all-pairs repel floor inflates them past the "
                         "irreversible ones). 0 = off.")
    ap.add_argument("--rev-cap", type=float, default=4.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from contextlib import ExitStack
    _stack = ExitStack()
    trk = _stack.enter_context(track_run("geometry_l1", args, run_name=Path(args.out).stem))
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
    if args.bp_shards:                                      # best-play optimal-successor edges (consecutive plies)
        import glob
        bp_pk, bp_pm, bp_ck, bp_cm = [], [], [], []
        for f in sorted(glob.glob(str(Path(args.bp_shards) / "*.npz"))):
            z = np.load(f); gid = np.asarray(z["game_id"]); ply = np.asarray(z["ply"]).astype(int)
            zp, zm = np.asarray(z["packed"]), np.asarray(z["meta"]); o = np.lexsort((ply, gid))
            for k in range(len(o) - 1):
                i, j = o[k], o[k + 1]
                if gid[i] == gid[j] and ply[j] == ply[i] + 1:
                    bp_pk.append(zp[i]); bp_pm.append(zm[i]); bp_ck.append(zp[j]); bp_cm.append(zm[j])
        if bp_pk:
            EPK = np.concatenate([EPK, np.stack(bp_pk)]); EPM = np.concatenate([EPM, np.stack(bp_pm)])
            ECK = np.concatenate([ECK, np.stack(bp_ck)]); ECM = np.concatenate([ECM, np.stack(bp_cm)])
            EIR = np.concatenate([EIR, np.zeros(len(bp_pk), bool)])   # best-play edges feed L_pos only
            print(f"[stage] +{len(bp_pk)} best-play edges -> {len(EPK)} total", flush=True)
    _zw = pay["zgoals"]["MATE_W"]
    zW = (_zw.detach().float() if torch.is_tensor(_zw) else torch.tensor(np.asarray(_zw, np.float32))).to(dev)

    # matched-anchor contrast pairs: (pos_state, neg_state, mate_exemplar, depth) flattened
    CPI = CNI = CMI = CDT = None
    if args.contrast_npz:
        cz = np.load(args.contrast_npz)
        CPK, CMT = np.asarray(cz["packed"]), np.asarray(cz["meta"])
        ctid, crole, cdep = cz["tuple_id"], cz["role"], cz["depth"]
        pi_, ni_, mi_, dt_ = [], [], [], []
        for t in range(int(ctid.max()) + 1):
            sel = ctid == t
            mates_i = np.flatnonzero(sel & (crole == 2))
            if not len(mates_i):
                continue
            mi = mates_i[0]
            pos_i = np.flatnonzero(sel & (crole == 1)); pos_d = cdep[pos_i]
            neg_i = np.flatnonzero(sel & (crole == -1)); neg_d = cdep[neg_i]
            for d_ in np.intersect1d(pos_d, neg_d):        # depth-matched pairs only
                pi_.append(pos_i[pos_d == d_][0]); ni_.append(neg_i[neg_d == d_][0])
                mi_.append(mi); dt_.append(int(d_))
        CPI, CNI, CMI = np.array(pi_), np.array(ni_), np.array(mi_)
        CDT = np.array(dt_, np.float32)
        print(f"[stage] contrast: {len(CPI)} depth-matched pos/neg pairs from {int(ctid.max())+1} tuples", flush=True)

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
        if args.w_rev > 0 and (~irr).any():                 # ASYM FIX: reversible reverses stay CHEAP
            ri_ = ei[~irr]
            d_bwd_rev = fb.distance_matrix(eF(ECK[ri_], ECM[ri_]), eB(EPK[ri_], EPM[ri_])).diagonal()
            L_rev = torch.relu(d_bwd_rev - args.rev_cap).pow(2).mean()
        else:
            L_rev = torch.zeros((), device=dev)
        ra = rng.integers(0, len(allp), size=args.batch); rb = rng.integers(0, len(allp), size=args.batch)
        unreach = ~reach_mask(pcv[ra], pcv[rb])
        d_cross = fb.distance_matrix(eF(allp[ra], allm[ra]), eB(allp[rb], allm[rb])).diagonal()
        keep = torch.from_numpy(unreach).to(dev)
        # ANTI-COLLAPSE two tiers: (a) EVERY random pair pushed to a moderate floor (random positions are
        # almost always far in transition-space; without this the within-material bulk collapses), plus
        # (b) material-UNREACHABLE pairs get the higher floor on top.
        L_repel = torch.relu(args.repel_floor_all - d_cross).pow(2).mean() if args.repel_floor_all > 0 else torch.zeros((), device=dev)
        if unreach.any():
            L_repel = L_repel + torch.relu(args.repel_floor - d_cross)[keep].pow(2).mean()
        if args.w_dtm > 0:                                  # DTM ALIGNMENT: d(F(s),MATE_W) ~ dtm/scale on won pos
            wi = won[rng.integers(0, len(won), size=args.batch)]
            d_mate = fb.distance_matrix(eF(allp[wi], allm[wi]), zW[None, :])[:, 0]
            L_dtm = ((d_mate - torch.from_numpy(dtm[wi] / args.dtm_scale).to(dev)) ** 2).mean()
        else:
            L_dtm = torch.zeros((), device=dev)
        if CPI is not None:                                 # MATCHED-ANCHOR CONTRAST (hinge, per-depth margin)
            ci_ = rng.integers(0, len(CPI), size=min(args.batch, len(CPI)))
            eM = eB(CPK[CMI[ci_]], CMT[CMI[ci_]])
            d_pm = fb.distance_matrix(eF(CPK[CPI[ci_]], CMT[CPI[ci_]]), eM).diagonal()
            d_nm = fb.distance_matrix(eF(CPK[CNI[ci_]], CMT[CNI[ci_]]), eM).diagonal()
            marg = torch.from_numpy(args.contrast_margin * CDT[ci_]).to(dev)
            L_con = torch.relu(marg + d_pm - d_nm).pow(2).mean()
        else:
            L_con = torch.zeros((), device=dev)
        loss = (args.w_pos * L_pos + args.w_hard * L_hard + args.w_repel * L_repel
                + args.w_dtm * L_dtm + args.w_contrast * L_con + args.w_rev * L_rev)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            rev, irr_a, xm = probe()
            print(f"  step {step:4d}  L_pos {float(L_pos):.3f} L_hard {float(L_hard):.3f} L_repel {float(L_repel):.3f} L_dtm {float(L_dtm):.3f} L_con {float(L_con):.3f}  "
                  f"d_pos {float(d_pos.median()):.2f}  asym rev {rev:.2f} irr {irr_a:.2f} (sep {irr_a/max(rev,1e-6):.2f}x)  "
                  f"KRk->KQk {xm:.1f}  ({time.time()-t0:.0f}s)", flush=True)
            trk.metrics(dict(L_pos=float(L_pos.detach()), L_hard=float(L_hard.detach()),
                             L_repel=float(L_repel.detach()), L_dtm=float(L_dtm.detach()),
                             L_con=float(L_con.detach()), d_pos=float(d_pos.detach().median()),
                             asym_rev=rev, asym_irr=irr_a), step=step)
        if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
            # step-suffixed ladder (KEPT, like train_lichess_fb) + rolling latest at --out
            out = Path(args.out)
            fb.eval()
            prov = dict(args=vars(args), resumed_from=str(args.ckpt))   # run metadata travels with the ckpt
            save_ckpt(fb, out.with_name(f"{out.stem}_step{step}{out.suffix}"),
                      step=step, zgoals=pay.get("zgoals"), provenance=prov)
            save_ckpt(fb, out, step=step, zgoals=pay.get("zgoals"), provenance=prov)
            fb.train()

    fb.eval()
    save_ckpt(fb, Path(args.out), step=args.steps, zgoals=pay.get("zgoals"),
              provenance=dict(args=vars(args), resumed_from=str(args.ckpt)))
    rev, irr_a, xm = probe()
    print(f"saved {args.out}")
    print(f"VERDICT GEOM_L1 asym_sep={irr_a/max(rev,1e-6):.2f}x (irr {irr_a:.2f}/rev {rev:.2f}) KRk->KQk={xm:.1f}")
    trk.tag("verdict", f"asym_sep={irr_a/max(rev,1e-6):.2f}x KRk->KQk={xm:.1f}")
    _stack.close()


if __name__ == "__main__":
    main()
