#!/usr/bin/env python
"""interpret_reach.py -- did the model infer the STRATA without being told any chess?

This is the headline measurement of the reach_probability approach (Kaveh 2026-08-05: "key point is
whether we can get strata without programming anything chess specific"). Nothing in the training
pipeline is told about piece count, captures, or legality: the model sees frozen trunk features and
pairs of positions that really followed one another. Piece count enters HERE, at analysis time, as a
LABEL only -- never as an input. If the structure comes back, it was inferred.

THE TEST, and why it is shaped this way.

Total piece count never rises: promotion preserves it, only captures reduce it. So for any pair,

    pieces(b) > pieces(a)   =>   b is UNREACHABLE from a, always.

That gives exact ground truth for one side without any learned or hand-written reachability rule.
And the question is posed forward, which is the only direction that matters in a game (Kaveh:
"the interesting questions in game are always about the future"): does the predicted reachable
region from a EXCLUDE positions carrying more material than a?

THE CONFOUND, and the control that kills it. A pair differing by +8 pieces also differs in every
other way, so a model scoring it low may simply be detecting gross dissimilarity rather than
irreversibility. The control is a MATCHED-MAGNITUDE comparison: for each k, score pairs with
delta = +k against pairs with delta = -k. Both carry the same material gap; only +k violates the
ratchet. Equal scores => the model learned dissimilarity. Lower scores at +k => it learned that the
ratchet has a DIRECTION, which is the strata claim.

Ply is matched by construction too (--ply-tol): cross-game partners are drawn from a similar ply, so
"later positions look different" cannot masquerade as the effect.

READ THE VERDICT ON `out_hist` PAIRS. For pairs within 8 plies, position a sits inside b's own lc0
history planes, so that band can be right for reasons that are not reachability at all.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.src.reach_jepa import ReachJEPA


def piece_counts(data_npz, cache):
    """(N,) total piece count per position. ANALYSIS-TIME LABEL ONLY -- never a model input.

    planes 0..11 are the twelve piece bitboards (verified: row sums match len(board.piece_map())).
    Cached because materialising `planes` costs ~2.9 GB and the answer is 400 KB.
    """
    try:
        return np.load(cache)["pc"]
    except Exception:
        pass
    z = np.load(data_npz, allow_pickle=True)
    pc = z["planes"][:, :12].sum((1, 2, 3)).astype(np.int16)
    np.savez_compressed(cache, pc=pc)
    return pc


def load_net(ckpt, device):
    p = torch.load(ckpt, map_location=device, weights_only=False)
    c = p["cfg"]
    net = ReachJEPA(in_ch=c["in_ch"], d=c["d"], adapter_ch=c["adapter_ch"], hidden=c["hidden"])
    net.load_state_dict(p["state_dict"])
    return net.to(device).eval(), p


@torch.no_grad()
def score_pairs(net, feats, ia, ib, device, batch=4096):
    out = []
    for s in range(0, len(ia), batch):
        fa = torch.from_numpy(feats[ia[s:s + batch]]).to(device, torch.float32)
        fb = torch.from_numpy(feats[ib[s:s + batch]]).to(device, torch.float32)
        out.append(net.score(net.encode(fa), net.encode_target(fb)).float().cpu().numpy())
    return np.concatenate(out)


def paired_ratchet(net, feats, pc, ply_all, game_all, k, n, rng, device, ply_tol=8):
    """THE ratchet test: hold the TARGET fixed, vary only the source.

    For one target b, compare
        a_dn : pieces(a) = pieces(b) - k   =>  delta = +k, b has MORE material: IMPOSSIBLE
        a_up : pieces(a) = pieces(b) + k   =>  delta = -k, b has LESS material: plausible
    and ask whether the model scores a_up above a_dn. Both comparisons share the SAME b, so anything
    that depends on b alone cancels exactly.

    Why this replaced the matched-|delta| version (a measured failure, not a theoretical worry): in
    that design the +k group's targets are piece-RICH and the -k group's are piece-POOR, so a model
    whose score depends only on the target reproduces the whole effect. Verified -- a degenerate run
    whose predictor input layer was entirely zeroed (support 0/64, mu a constant, so it provably
    cannot read the source at all) still scored 0.575 there, against 0.595 for the real model. Under
    the paired design that same model must score exactly 0.5, because its two scores are identical.
    """
    n_all = len(pc)
    key = pc.astype(np.int64) * 1000 + np.clip(ply_all // 10, 0, 40)
    order = np.argsort(key, kind="stable")
    ks, ke = np.unique(key[order], return_index=True)

    def pool(pcv, plyv):
        want = int(pcv) * 1000 + int(np.clip(plyv // 10, 0, 40))
        p = np.searchsorted(ks, want)
        if p >= len(ks) or ks[p] != want:
            return None
        s = ke[p]
        e = ke[p + 1] if p + 1 < len(ke) else len(order)
        return order[s:e]

    b_idx, up_idx, dn_idx = [], [], []
    for b in rng.integers(0, n_all, n):
        pb, yb = int(pc[b]), int(ply_all[b])
        pu, pd = pool(pb + k, yb), pool(pb - k, yb)
        if pu is None or pd is None or not len(pu) or not len(pd):
            continue
        a_up, a_dn = pu[rng.integers(len(pu))], pd[rng.integers(len(pd))]
        if game_all[a_up] == game_all[b] or game_all[a_dn] == game_all[b]:
            continue
        if abs(int(ply_all[a_up]) - yb) > ply_tol or abs(int(ply_all[a_dn]) - yb) > ply_tol:
            continue
        b_idx.append(b); up_idx.append(a_up); dn_idx.append(a_dn)
    if len(b_idx) < 100:
        return float("nan"), 0
    b_idx = np.array(b_idx); up_idx = np.array(up_idx); dn_idx = np.array(dn_idx)
    s_up = score_pairs(net, feats, up_idx, b_idx, device)
    s_dn = score_pairs(net, feats, dn_idx, b_idx, device)
    # Paired: fraction of targets for which the plausible source outscores the impossible one.
    # Ties count as 0.5 -- a source-blind model produces EXACTLY equal scores, and scoring those 0
    # would report 0.000 for a model whose true value is chance. (Measured: the zeroed-predictor run
    # gave 0.000 under strict `>`; it is 0.500 here, which is the null this design is built to give.)
    return float((s_up > s_dn).mean() + 0.5 * (s_up == s_dn).mean()), len(b_idx)


def auc(pos, neg):
    """P(a random `pos` scores above a random `neg`), via rank statistic."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(x, y):
    if len(x) < 8:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=paths.experiment("reach_jepa_v1_latest.pt"))
    ap.add_argument("--pairs", default=paths.derived("reach_pairs_v1.npz"))
    ap.add_argument("--data", default=paths.derived("field_std_v2.npz"))
    ap.add_argument("--feats", default=paths.derived("trunk_feats/t1-256x10__field_std_v2.npy"))
    ap.add_argument("--pc-cache", default=paths.derived("field_std_v2_piececount.npz"))
    ap.add_argument("--rows", type=int, default=131072)
    ap.add_argument("--n-cross", type=int, default=200_000, help="cross-game probe pairs")
    ap.add_argument("--ply-tol", type=int, default=5, help="match cross-game partners on ply")
    ap.add_argument("--n-paired", type=int, default=60_000, help="targets sampled for the paired test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    net, payload = load_net(args.ckpt, args.device)
    feats = np.ascontiguousarray(np.load(args.feats, mmap_mode="r")[:args.rows])
    zd = np.load(args.data, allow_pickle=True)
    game_all, ply_all = zd["game"], zd["ply"]
    pc = piece_counts(args.data, args.pc_cache)
    print(f"[interpret] ckpt step {payload.get('step')} | {args.rows:,} rows | "
          f"piece count {pc.min()}..{pc.max()} [{time.time()-t0:.0f}s]", flush=True)

    zp = np.load(args.pairs, allow_pickle=True)
    m = (zp["i"] < args.rows) & (zp["j"] < args.rows) & (zp["split"] == 2)   # TEST games only
    ti, tj, tgap = zp["i"][m], zp["j"][m], zp["gap"][m]

    # ---- 1. real forward pairs (reachable, by observation) -------------------------------------
    s_real = score_pairs(net, feats, ti, tj, args.device)
    real_out = s_real[tgap > 8]

    # ---- 2. cross-game probe pairs, matched on ply ---------------------------------------------
    n = args.rows
    order = np.argsort(ply_all[:n], kind="stable")
    ply_sorted = ply_all[:n][order]
    ia = rng.integers(0, n, args.n_cross)
    lo = np.searchsorted(ply_sorted, ply_all[ia] - args.ply_tol, "left")
    hi = np.searchsorted(ply_sorted, ply_all[ia] + args.ply_tol, "right")
    pick = (lo + (rng.random(args.n_cross) * np.maximum(hi - lo, 1))).astype(np.int64)
    ib = order[np.clip(pick, 0, n - 1)]
    ok = game_all[ia] != game_all[ib]                    # genuinely cross-game
    ia, ib = ia[ok], ib[ok]
    s_cross = score_pairs(net, feats, ia, ib, args.device)
    dpc = (pc[ib].astype(int) - pc[ia].astype(int))
    dply = np.abs(ply_all[ia].astype(int) - ply_all[ib].astype(int))
    print(f"[interpret] cross-game pairs {len(ia):,} | ply match |dply| median {np.median(dply):.0f} "
          f"| delta-pieces p5/50/95 {np.percentile(dpc,[5,50,95]).astype(int).tolist()}", flush=True)

    # ---- 3a. THE RATCHET TEST (paired: same target, sources differ) ------------------------------
    print(f"\n  PAIRED ratchet test -- same target b, source varies. A model that ignores the")
    print(f"  source scores exactly 0.500 here by construction.")
    print(f"  {'k':>3} {'n pairs':>9} {'P(plausible > impossible)':>26}")
    paired = []
    for k in (1, 2, 3, 4, 6, 8):
        r, nn = paired_ratchet(net, feats, pc[:args.rows], ply_all[:args.rows],
                               game_all[:args.rows], k, args.n_paired, rng, args.device)
        if nn:
            paired.append(r)
            print(f"  {k:>3} {nn:>9,} {r:>26.4f}")
    paired_auc = float(np.mean(paired)) if paired else float("nan")

    # ---- 3b. the OLD matched-magnitude test, retained and labelled CONFOUNDED --------------------
    print(f"\n  [CONFOUNDED -- retained for comparison only] matched-|delta| test. Its +k group has")
    print(f"  piece-RICH targets and its -k group piece-POOR ones, so a target-only model passes it.")
    print(f"  {'|dpc|':>5} {'n(+k)':>8} {'n(-k)':>8} {'mean s(+k)':>11} {'mean s(-k)':>11} "
          f"{'gap':>8} {'AUC(-k>+k)':>11}")
    rows_tbl, aucs = [], []
    for k in range(1, 9):
        p_up, p_dn = s_cross[dpc == k], s_cross[dpc == -k]
        if len(p_up) < 50 or len(p_dn) < 50:
            continue
        a = auc(p_dn, p_up)                              # want > 0.5 if the ratchet is learned
        rows_tbl.append((k, len(p_up), len(p_dn), p_up.mean(), p_dn.mean(), p_dn.mean()-p_up.mean(), a))
        aucs.append(a)
        print(f"  {k:>5} {len(p_up):>8,} {len(p_dn):>8,} {p_up.mean():>11.3f} {p_dn.mean():>11.3f} "
              f"{p_dn.mean()-p_up.mean():>8.3f} {a:>11.3f}")
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")

    # ---- 4. the confound it must beat -----------------------------------------------------------
    rho_signed = spearman(s_cross, dpc.astype(float))     # direction-sensitive (the ratchet)
    rho_abs = spearman(s_cross, -np.abs(dpc).astype(float))  # magnitude only (mere dissimilarity)

    # ---- 5. does the model separate observed-reachable from cross-game at all? ------------------
    sep = auc(real_out, s_cross)

    # ---- 6. what the L1 kept, and what it correlates with ---------------------------------------
    w = net.head_in.weight.detach().abs().mean(0).cpu().numpy()   # per-input-coordinate importance
    keep = int((net.head_in.weight.detach().abs() > 0).any(0).sum())   # EXACT zeros (see prox_l1)
    sub = rng.integers(0, args.rows, 20000)
    with torch.no_grad():
        zs = net.encode(torch.from_numpy(feats[sub]).to(args.device, torch.float32)).cpu().numpy()
    top = np.argsort(-w)[:5]
    coord_rhos = [(int(c), spearman(zs[:, c], pc[sub].astype(float)),
                   spearman(zs[:, c], ply_all[sub].astype(float))) for c in top]

    print(f"\n  observed-reachable vs cross-game AUC   {sep:.3f}   (can it tell them apart at all)")
    print(f"  spearman(score,  delta_pieces)         {rho_signed:+.3f}   <- SIGNED: the ratchet")
    print(f"  spearman(score, -|delta_pieces|)       {rho_abs:+.3f}   <- magnitude only: the confound")
    print(f"  L1 kept {keep}/{len(w)} input coords of the predictor")
    print(f"  {'coord':>6} {'|w|':>8} {'rho(z,pieces)':>15} {'rho(z,ply)':>12}")
    for c, rp, rl in coord_rhos:
        print(f"  {c:>6} {w[c]:>8.4f} {rp:>+15.3f} {rl:>+12.3f}")

    print(f"\nVERDICT REACH-STRATA paired_ratchet={paired_auc:.4f} "
          f"confounded_ratchet={mean_auc:.4f} "
          f"rho_signed={rho_signed:+.4f} rho_absonly={rho_abs:+.4f} sep_auc={sep:.4f} "
          f"l1_kept={keep}/{len(w)} n_cross={len(ia)} step={payload.get('step')} "
          f"[{time.time()-t0:.0f}s]")
    print("  READ paired_ratchet. > 0.5 => holding the target fixed, the model prefers a source that")
    print("  COULD reach it over one that could not -- the irreversible direction, inferred with no")
    print("  chess programmed anywhere. 0.5 => no source dependence. confounded_ratchet is kept only")
    print("  to document the earlier metric a source-blind model also passed.")


if __name__ == "__main__":
    main()
