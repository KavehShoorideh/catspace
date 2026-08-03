#!/usr/bin/env python
"""catspace/research/components/planner/approaches/endgame_groundtruth/experiments/bootstrap_dtm.py -- ENDGAME-OUTWARD distance bootstrap (Kaveh 2026-07-26).

The bake-off (JOURNAL 2026-07-25) proved the middlegame distance-to-mate failure is a
LABELS problem, not architecture: no backbone extrapolates DTM past its training range,
but every backbone FITS long distance when given labels. Long labels don't exist
(tablebase stops at 6 pieces; human games resign before mate). So MANUFACTURE them:
value-iteration on the quasimetric, growing the trusted endgame distance outward via the
Bellman/minimax backup for distance-to-mate --

    winner to move s:  d(s) = 1 + min_m  V(s.m)
    loser  to move t:  V(t) = 0 if t is checkmate (winner just mated),
                              1 + max_m' g(t.m')  otherwise
  => non-terminal move: candidate = 1 + (1 + max_m' g(grandchild))
     mating move:       candidate = 1  (V=0)
     d(s) = 1 + min over move candidates' V.

This is a FALSIFIABLE test of the mechanism: we run it on tablebase positions where the
TRUE long DTM is known but HIDDEN from the anchor (train g on DTM<=anchor only), bootstrap
using g's own 2-ply lookahead, and measure whether the held-out FAR slice (DTM>anchor)
ordering recovers from the bake-off's -0.44 toward positive. The 2-ply lookahead STRUCTURE
is precomputed once (move-gen is the only python-chess cost); each value-iteration sweep is
pure tensor ops -- net forward on the flat grandchild set + segment min/max reductions.

No policy target anywhere: g outputs distance-to-mate; the planner stays one layer above.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


import chess

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
from catspace.research.components.encoder.approaches.reachability_field.experiments.arch_bakeoff import CNNBackbone, TransformerBackbone, eff_rank, tokens
from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.dtm_arch_bakeoff import DTMNet, spearman
from catspace.io import paths


def build_lookahead(packed, meta, parents):
    """Precompute the 2-ply minimax structure for `parents` (winner-to-move, won).

    Returns flat grandchild packed/meta plus segment maps:
      gc_packed (M,12), gc_meta (M,8): every grandchild (winner to move again)
      gc_branch (M,):   branch id each grandchild belongs to (max over these)
      br_parent (B,):   parent-local id each branch belongs to (min over these)
      br_term  (B,) bool: branch is a mating move (V=0, no grandchildren)
      keep (P',):       parent indices that had >=1 winning branch (others dropped)
    """
    gc_pk, gc_mt, gc_branch = [], [], []
    br_parent, br_term = [], []
    keep = []
    b_id = 0
    for pj, i in enumerate(parents):
        b = board_from_packed(packed[i], meta[i])
        if b.turn != chess.WHITE:
            continue
        has_branch = False
        for m in b.legal_moves:
            b.push(m)
            if b.is_checkmate():                      # winner mates -> V=0
                br_parent.append(pj); br_term.append(True); b_id += 1
                has_branch = True
            elif b.is_game_over():                    # stalemate/draw -> threw the win
                pass
            else:                                     # opponent (loser) to move
                replies = list(b.legal_moves)
                for m2 in replies:
                    b.push(m2)
                    gc_pk.append(encode_packed(b)); gc_mt.append(encode_meta(b))
                    gc_branch.append(b_id)
                    b.pop()
                br_parent.append(pj); br_term.append(False); b_id += 1
                has_branch = True
            b.pop()
        if has_branch:
            keep.append(pj)
    return (np.stack(gc_pk), np.stack(gc_mt),
            np.asarray(gc_branch, np.int64), np.asarray(br_parent, np.int64),
            np.asarray(br_term, bool), np.asarray(keep, np.int64))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", choices=["cnn", "xf"], default="cnn")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--data", default=paths.derived("dtm_endgame_v2.npz"))
    ap.add_argument("--anchor", type=int, default=25, help="train true labels on DTM<=anchor")
    ap.add_argument("--n-boot", type=int, default=3000, help="# DTM>anchor bootstrap parents")
    ap.add_argument("--base-steps", type=int, default=5000)
    ap.add_argument("--sweeps", type=int, default=10)
    ap.add_argument("--sweep-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cap", type=float, default=300.0, help="clamp bootstrap targets (plies)")
    # --- iteration-2 stability levers (defaults reproduce iteration 1) ---
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help=">0 uses a target/EMA net to compute bootstrap targets (TD stability)")
    ap.add_argument("--anchor-ratio", type=float, default=0.5,
                    help="fraction of each mixed batch drawn from the true-label anchor")
    ap.add_argument("--rank-weight", type=float, default=0.0,
                    help="weight of pairwise margin-rank loss on the bootstrap slice")
    ap.add_argument("--phased", action="store_true",
                    help="alternate anchor/boot PHASES (dedicated boot windows) using anchor-ratio "
                         "as the split; propagates where mixed batches stall")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    z = np.load(args.data)
    packed, meta, dtm = z["packed"], z["meta"], z["dtm"].astype(np.float32)
    ok = dtm > 0
    packed, meta, dtm = packed[ok], meta[ok], dtm[ok]
    ids_all, stm_all = tokens(packed, meta)
    n = len(dtm); A = args.anchor
    heldout = rng.random(n) < 0.1
    anchor_tr = np.flatnonzero((dtm <= A) & ~heldout)
    te_near = np.flatnonzero((dtm <= A) & heldout)
    far_pool = np.flatnonzero((dtm > A) & ~heldout)
    te_far = np.flatnonzero((dtm > A) & heldout)
    boot = rng.choice(far_pool, size=min(args.n_boot, len(far_pool)), replace=False)
    tag = f"boot-{args.backbone}-d{args.d}-L{args.layers}-A{A}"
    print(f"[{tag}] anchor(DTM<={A}) {len(anchor_tr)} | boot(DTM>{A}) {len(boot)} "
          f"| te_near {len(te_near)} | te_far {len(te_far)}", flush=True)

    # --- precompute 2-ply lookahead structure once (only python-chess cost) ---
    tb0 = time.time()
    gc_pk, gc_mt, gc_branch, br_parent, br_term, keep = build_lookahead(packed, meta, boot)
    boot = boot[keep]                                     # drop parents w/ no winning branch
    # remap br_parent (parent-local ids) onto the kept-parent index space
    remap = -np.ones(len(keep) if len(keep) else 1, np.int64)
    kmap = {int(k): j for j, k in enumerate(keep)}
    br_parent = np.array([kmap[int(p)] for p in br_parent], np.int64)
    gc_ids, gc_stm = tokens(gc_pk, gc_mt)
    gc_ids = torch.from_numpy(gc_ids.astype(np.int64))    # (M,64)
    gc_stm = torch.from_numpy(gc_stm.astype(np.int64))
    n_branch = len(br_parent); n_boot = len(boot); M = len(gc_ids)
    print(f"  lookahead: {M} grandchildren, {n_branch} branches, {n_boot} parents kept "
          f"[{time.time()-tb0:.0f}s]", flush=True)

    bb = (CNNBackbone(args.d, args.layers) if args.backbone == "cnn"
          else TransformerBackbone(args.d, args.layers))
    net = DTMNet(bb, args.d).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # target/EMA net: bootstrap targets come from a lagged copy so the field can't chase
    # its own moving estimate (the overshoot/near-erosion fix; standard Double-DQN trick).
    import copy
    tnet = copy.deepcopy(net) if args.ema_decay > 0 else net

    def ema_update():
        if args.ema_decay <= 0:
            return
        with torch.no_grad():
            for pt, p in zip(tnet.parameters(), net.parameters()):
                pt.mul_(args.ema_decay).add_(p, alpha=1 - args.ema_decay)
            for bt, b in zip(tnet.buffers(), net.buffers()):
                bt.copy_(b)

    ids_t = torch.from_numpy(ids_all.astype(np.int64))
    stm_t = torch.from_numpy(stm_all.astype(np.int64))

    def feed(idx):
        return ids_t[idx].to(dev), stm_t[idx].to(dev)

    def train_on(idx, target_log, steps):                    # iteration-1 path (anchor-ratio=0.5, no mix)
        net.train()
        for s in range(steps):
            bi = rng.integers(0, len(idx), args.batch)
            di, ds = ids_t[idx[bi]].to(dev), stm_t[idx[bi]].to(dev)
            pred, _ = net(di, ds)
            loss = F.huber_loss(pred, target_log[bi], delta=1.0)
            opt.zero_grad(); loss.backward(); opt.step(); ema_update()
        return float(loss)

    def train_mixed(a_idx, a_tlog, b_idx, b_tlog, steps):
        """One field-update stream: every batch mixes true-anchor + bootstrap examples
        (anchor_ratio split) so anchor calibration never erodes, plus an optional pairwise
        rank loss on the bootstrap slice to sharpen far ORDERING (not just magnitude)."""
        net.train()
        na = max(1, int(round(args.batch * args.anchor_ratio)))
        nb = max(2, args.batch - na)
        for s in range(steps):
            ai = a_idx[rng.integers(0, len(a_idx), na)]
            bj = rng.integers(0, len(b_idx), nb)
            bi = b_idx[bj]
            cat = np.concatenate([ai, bi])
            di, ds = ids_t[cat].to(dev), stm_t[cat].to(dev)
            pred, _ = net(di, ds)
            at = a_tlog_full[ai]; bt = b_tlog[bj]             # true-anchor + bootstrap targets
            loss = F.huber_loss(pred, torch.cat([at, bt]), delta=1.0)
            if args.rank_weight > 0:                          # sharpen far ORDERING, not just magnitude
                pb = pred[na:]
                perm = torch.from_numpy(rng.permutation(nb)).to(dev)
                y = torch.sign(bt - bt[perm])
                loss = loss + args.rank_weight * F.margin_ranking_loss(
                    pb, pb[perm], y, margin=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); ema_update()
        return float(loss)

    @torch.no_grad()
    def predict(ids_x, stm_x, use=None):
        m = use if use is not None else net
        m.eval(); out = []
        for s in range(0, len(ids_x), 4096):
            di, ds = ids_x[s:s + 4096].to(dev), stm_x[s:s + 4096].to(dev)
            out.append(m(di, ds)[0].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def far_report(label):
        pf = predict(ids_t[te_far], stm_t[te_far])
        pn = predict(ids_t[te_near], stm_t[te_near])
        sp_f = spearman(pf, dtm[te_far]); sp_n = spearman(pn, dtm[te_near])
        mae_f = float(np.abs(np.expm1(pf) - dtm[te_far]).mean())
        with torch.no_grad():
            di, ds = feed(te_far[:1500]); _, pooled = net.bb(di, ds)
        er = eff_rank(pooled.cpu().numpy())
        print(f"  [{label}] near sp {sp_n:+.3f} | FAR sp {sp_f:+.3f} MAE {mae_f:.1f} "
              f"| far_eff_rank {er:.1f} [{time.time()-t0:.0f}s]", flush=True)
        return sp_f

    # --- Phase A: anchor-only baseline (reproduces bake-off far ~ -0.44) ---
    ltr = torch.from_numpy(np.log1p(dtm[anchor_tr])).to(dev)  # aligned target for anchor idx
    a_tlog_full = torch.from_numpy(np.log1p(dtm)).to(dev)     # full-length true log-DTM (anchors only used)
    net.train()
    for s in range(args.base_steps):
        bi = rng.integers(0, len(anchor_tr), args.batch)
        di, ds = feed(anchor_tr[bi])
        pred, _ = net(di, ds)
        loss = F.huber_loss(pred, ltr[bi], delta=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"VERDICT BOOT {tag} phase-A(anchor-only):", flush=True); sp0 = far_report("A0")

    # --- Phase B: value-iteration bootstrap sweeps ---
    br_parent_t = br_parent; gc_branch_t = gc_branch
    mixed = (not args.phased) and (args.ema_decay > 0 or args.anchor_ratio != 0.5 or args.rank_weight > 0)
    for sw in range(args.sweeps):
        # bootstrap targets from the TARGET net (tnet==net when ema off) -- 2-ply minimax
        gval = np.expm1(predict(gc_ids, gc_stm, use=tnet)).clip(0, args.cap)
        branch_max = np.full(n_branch, -1.0, np.float32)
        np.maximum.at(branch_max, gc_branch_t, gval)
        branch_V = np.where(br_term, 0.0, 1.0 + np.maximum(branch_max, 0.0))
        parent_min = np.full(n_boot, np.inf, np.float32)
        np.minimum.at(parent_min, br_parent_t, branch_V)
        target_d = np.clip(1.0 + parent_min, 1.0, args.cap)
        good = np.isfinite(target_d)
        boot_idx = boot[good]
        boot_tlog = torch.from_numpy(np.log1p(target_d[good])).to(dev)
        if args.phased:                                      # dedicated boot window; ratio sets the split
            a_steps = int(round(args.sweep_steps * args.anchor_ratio))
            train_on(anchor_tr, ltr, a_steps)
            train_on(boot_idx, boot_tlog, args.sweep_steps - a_steps)
        elif mixed:
            train_mixed(anchor_tr, ltr, boot_idx, boot_tlog, args.sweep_steps)
        else:                                                # iteration-1 path: 50/50 phased
            train_on(anchor_tr, ltr, args.sweep_steps // 2)
            train_on(boot_idx, boot_tlog, args.sweep_steps // 2)
        med = float(np.median(target_d[good]))
        print(f"  sweep {sw}: boot_target median {med:.0f} plies (true median "
              f"{np.median(dtm[boot[good]]):.0f})", flush=True)
        far_report(f"B{sw}")

    print(f"VERDICT BOOT {tag}: far_spearman {sp0:+.3f} (anchor-only) -> "
          f"see B{args.sweeps-1} above | [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
