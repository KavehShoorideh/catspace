#!/usr/bin/env python
"""interpret_reach.py -- did the model infer the STRATA without being told any chess?

This is the headline measurement of the reach_probability approach (Kaveh 2026-08-05: "key point is
whether we can get strata without programming anything chess specific"). Nothing in the training
pipeline is told about piece count, captures, or legality. Piece count enters HERE, at analysis
time, as a LABEL only -- never as an input. If the structure comes back, it was inferred.

TWO MODEL FAMILIES, ONE EVALUATION PATH. `--ckpt` decides which:
  arch "trunk"  ReachJEPA on frozen lc0 features. The original, kept so the comparison survives.
  arch "vit"    ReachViT: a from-scratch ViT over tokenized boards, region head + IQE head.
Everything below runs identically over both; only load_net/score_pairs dispatch.

WHY THE VIT RUN IS THE ONE THAT CAN SETTLE THIS. The trunk result (paired ratchet 0.570 against a
random-init null of 0.555, flat across the ladder) was NEGATIVE but inconclusive, because a
pretrained chess net already contains the ratchet -- its random-init null is not zero. A
randomly-initialised ViT over raw tokens knows no chess, so its null IS zero and any gap is learned.

THE TEST, and why it is shaped this way.

Total piece count never rises: promotion preserves it, only captures reduce it. So for any pair,

    pieces(b) > pieces(a)   =>   b is UNREACHABLE from a, always.

That is exact ground truth for one side with no learned or hand-written reachability rule. And it is
posed forward, the only direction that matters in a game (Kaveh: "the interesting questions in game
are always about the future"): does the region predicted from a EXCLUDE positions carrying more
material than a?

THE CONFOUND, and the control that kills it. A pair differing by +8 pieces differs in every other way
too, so a model scoring it low may be detecting gross dissimilarity rather than irreversibility. The
PAIRED design holds the target b fixed and varies only the source, so everything depending on b
cancels exactly, and a model that cannot read the source scores 0.500 by construction rather than by
argument.

  >>> THE DIFFERENTIAL, and why the direct ratchet readout alone is worthless. Training pushes
  >>> unobserved reversals apart UNIFORMLY, so "reverses are far" is trained in and proves nothing.
  >>> The strata claim is read on `capture-crossing vs quiet` reversals -- BOTH of them unobserved,
  >>> BOTH given identical repulsion during training, ply-gap matched, differing only in whether
  >>> material fell across the pair. A uniform training signal cannot manufacture a difference
  >>> between two groups it treated identically; only the data can. The repetition-covered
  >>> reversible pairs are printed beside them as the trained-in ~1.0 reference, explicitly LABELLED
  >>> as trained-in, because those ARE excluded from the repulsion by construction.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T
from catspace.research.components.encoder.approaches.reach_probability.src.reach_jepa import ReachJEPA
from catspace.research.components.encoder.approaches.reach_probability.src.reach_vit_jepa import (
    IQE_ARM, REGION, ReachViT)

TRUNK, VIT = "trunk", "vit"


def piece_counts(data_npz, cache):
    """(N,) total piece count per position, for the TRUNK path. ANALYSIS-TIME LABEL ONLY.

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
    """-> (net, payload). Dispatches on cfg['arch']; absent means the original trunk model."""
    p = torch.load(ckpt, map_location=device, weights_only=False)
    c = p["cfg"]
    if c.get("arch", TRUNK) == VIT:
        net = ReachViT(d_model=c["d_model"], layers=c["layers"], heads=c["heads"], d=c["d"],
                       hidden=c["hidden"], components=c["components"],
                       dual=c.get("dual", False), d_cond=c.get("d_cond", 0))
        # Rebuild the POLE hierarchy from the cfg before loading. The pole set is data-derived, so
        # a pole-bearing checkpoint carries weights the bare architecture has no slots for and a
        # strict load_state_dict rejects it outright -- which would have failed EVERY v2 rung.
        # ...but CONTRASTIVE checkpoints (poles mode 'contrastive', 2026-08-07) legitimately
        # carry no pole weights -- attach only when the state_dict actually has them.
        if c.get("pole_parent") and any(k.startswith("poles.") for k in p["state_dict"]):
            net.attach_poles(torch.tensor(c["pole_parent"]), n_sources=1,
                             fixed=not c.get("learned_poles", False),
                             height=c.get("pole_height", 3.0))
    else:
        net = ReachJEPA(in_ch=c["in_ch"], d=c["d"], adapter_ch=c["adapter_ch"], hidden=c["hidden"])
    net.load_state_dict(p["state_dict"])
    return net.to(device).eval(), p


@torch.no_grad()
def score_pairs(net, source, ia, ib, device, batch=4096, arm=REGION):
    """(n,) reachability score, HIGHER = b looks more like a future of a. One path, both families.

    `source` is either the trunk feature array (N,C,8,8) or a (tok, glob) pair of token arrays from
    a TrajectoryStore -- the only thing that differs between the model families is how a row becomes
    a batch, so everything downstream of here is shared and the two are compared like for like.
    """
    out = []
    for s in range(0, len(ia), batch):
        u, v = ia[s:s + batch], ib[s:s + batch]
        if isinstance(source, tuple):
            tok, glob = source
            ta = torch.from_numpy(tok[u].astype(np.int64)).to(device)
            ga = torch.from_numpy(glob[u].astype(np.float32)).to(device)
            tb = torch.from_numpy(tok[v].astype(np.int64)).to(device)
            gb = torch.from_numpy(glob[v].astype(np.float32)).to(device)
            out.append(net.score_rows(ta, ga, tb, gb, arm=arm).float().cpu().numpy())
        else:
            fa = torch.from_numpy(source[u]).to(device, torch.float32)
            fb = torch.from_numpy(source[v]).to(device, torch.float32)
            out.append(net.score(net.encode(fa), net.encode_target(fb)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


@torch.no_grad()
def directed_distance(net, source, ia, ib, device, batch=4096):
    """(n,) arm-B d(a -> b). The quasimetric read directly, rather than through a score."""
    tok, glob = source
    out = []
    for s in range(0, len(ia), batch):
        u, v = ia[s:s + batch], ib[s:s + batch]
        za = net.encode_q(torch.from_numpy(tok[u].astype(np.int64)).to(device),
                          torch.from_numpy(glob[u].astype(np.float32)).to(device))
        zb = net.encode_q(torch.from_numpy(tok[v].astype(np.int64)).to(device),
                          torch.from_numpy(glob[v].astype(np.float32)).to(device))
        out.append(net.distance(za, zb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def paired_ratchet(net, source, pc, ply_all, game_all, k, n, rng, device, ply_tol=8, arm=REGION):
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
    s_up = score_pairs(net, source, up_idx, b_idx, device, arm=arm)
    s_dn = score_pairs(net, source, dn_idx, b_idx, device, arm=arm)
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


def match_on_gap(gap_a, gap_b, rng, n):
    """Index pairs (ia, ib) drawn so the two groups have the SAME ply-gap distribution.

    Without this, "capture-crossing reversals are further" could be nothing but "capture-crossing
    reversals span more plies", which is true and uninteresting -- captures accumulate with time.
    Matching is exact per gap value, so no residual gap difference survives to be misread.
    """
    ia, ib = [], []
    for g in np.intersect1d(np.unique(gap_a), np.unique(gap_b)):
        pa, pb = np.flatnonzero(gap_a == g), np.flatnonzero(gap_b == g)
        m = min(len(pa), len(pb), max(1, n // 20))
        ia.append(rng.choice(pa, m, replace=False)); ib.append(rng.choice(pb, m, replace=False))
    if not ia:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(ia), np.concatenate(ib)


def differential(net, source, tr, rows, rng, device, n=40_000, max_gap=40, pop=None):
    """THE strata verdict for the ViT arms: capture-crossing vs quiet reversals, ply-gap matched.

    Three groups of BACKWARD pairs (b -> a with a earlier), all within a game:
      capture    UNOBSERVED (no repetition covers it) and material FELL across the pair
      quiet      UNOBSERVED and material is UNCHANGED across the pair
      revers.    OBSERVED reversible (a repetition covers it), material unchanged

    capture and quiet received byte-identical treatment in training -- both are unobserved
    reversals, both carried the same repulsion, and neither was ever labelled. So a difference
    between them cannot have been installed by the objective; it can only come from the data. That
    is the whole claim. `revers.` is the trained-in reference: those pairs were EXCLUDED from the
    repulsion by construction, so their ~1.0 is a property of the objective and is labelled as such.
    """
    ply, game, pc, cov = tr.ply_of_row(), tr.game_of_row(), tr.piece_count(), tr.coverage()
    rows = np.asarray(rows)
    if pop is not None:
        # SOURCE-STRATIFIED. The differential is computed over POOLED test games, so if captures
        # correlate with population -- SF games are draw-heavy and plausibly quieter -- a
        # population imbalance between the capture and quiet groups could contribute to the gap.
        # Ply-matching is exact and the ratio form cancels symmetric dissimilarity, but neither
        # touches population. Restricting to ONE population removes it entirely: if the effect
        # holds separately inside human games and inside SF games, it is not a mix artifact.
        src_row = np.repeat(tr.source, tr.length)
        rows = rows[src_row[rows] == pop]
    # draw forward pairs (i < j) inside the eval games, then read them BACKWARD
    i = rng.choice(rows, n)
    g = game[i]
    end = tr.start[g] + tr.length[g] - 1
    j = i + 1 + (rng.random(n) * np.minimum(max_gap, end - i)).astype(np.int64)
    ok = j <= end
    i, j = i[ok], j[ok]
    gap = (ply[j] - ply[i]).astype(np.int64)
    unobs = j > cov[i]                                   # no repetition spans [i, j]
    dpc = pc[i].astype(int) - pc[j].astype(int)          # material LOST from i to j (>= 0)

    groups = {"capture": unobs & (dpc >= 1), "quiet": unobs & (dpc == 0),
              "revers.": ~unobs & (dpc == 0)}
    out = {}
    ref_gap = gap[groups["quiet"]]
    for name, m in groups.items():
        sel = np.flatnonzero(m)
        if len(sel) < 200:
            out[name] = (float("nan"), float("nan"), 0)
            continue
        ma, _ = match_on_gap(gap[sel], ref_gap, rng, n)   # every group matched to the SAME reference
        sel = sel[ma] if len(ma) else sel
        a, b = i[sel], j[sel]
        d_fwd = directed_distance(net, source, a, b, device)
        d_rev = directed_distance(net, source, b, a, device)
        per_pair = d_rev / np.maximum(d_fwd, 1e-6)          # kept for the bootstrap
        ratio = float(np.median(per_pair))
        s_fwd = score_pairs(net, source, a, b, device, arm=REGION)
        s_rev = score_pairs(net, source, b, a, device, arm=REGION)
        out[name] = (ratio, float(np.mean(s_fwd - s_rev)), len(sel), per_pair)
    return out


def bootstrap_diff(per_pair_a, per_pair_b, rng, n_boot=2000):
    """(point, lo, hi) for median(a) - median(b), percentile bootstrap.

    The differential is THE strata claim, so it does not get to be a bare number. Resampling both
    groups independently is right here because they are independently drawn pair sets, not a paired
    design -- and a CI that straddles 0 means the effect is not established however good the point
    estimate looks. This is the guard against reporting the previous attempt's +0.015 as a finding.
    """
    if per_pair_a is None or per_pair_b is None or len(per_pair_a) < 50 or len(per_pair_b) < 50:
        return float("nan"), float("nan"), float("nan")
    pt = float(np.median(per_pair_a) - np.median(per_pair_b))
    na, nb = len(per_pair_a), len(per_pair_b)
    d = np.empty(n_boot)
    for k in range(n_boot):
        d[k] = (np.median(per_pair_a[rng.integers(0, na, na)])
                - np.median(per_pair_b[rng.integers(0, nb, nb)]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return pt, float(lo), float(hi)


def build_eval_context(net, payload, args):
    """-> (source, pc, ply, game, tr, rows). One context, whichever family the ckpt is."""
    c = payload["cfg"]
    if c.get("arch", TRUNK) != VIT:
        feats = np.ascontiguousarray(np.load(args.feats, mmap_mode="r")[:args.rows])
        zd = np.load(args.data, allow_pickle=True)
        n = args.rows
        pc = piece_counts(args.data, args.pc_cache)[:n]
        return feats, pc, zd["ply"][:n], zd["game"][:n], None, np.arange(n)

    # ViT: rebuild the exact trajectory store the run trained on (cache hit), keep TEST games only.
    tr = T.build(n_human=c["games"] // 2, n_sf=c["games"] // 2, seed=c["traj_seed"],
                 max_plies=c["max_plies"])
    from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
        split_by_game)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    keep = np.flatnonzero(split == 2)                     # TEST games: never trained, never calibrated
    game_row = tr.game_of_row()
    rows = np.flatnonzero(np.isin(game_row, keep))
    if args.rows and args.rows < len(rows):
        rows = rows[:args.rows]
    print(f"[interpret] vit | {len(keep):,} test games | {len(rows):,} positions", flush=True)
    return (tr.tok, tr.glob), tr.piece_count(), tr.ply_of_row(), game_row, tr, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=paths.experiment("reach_vit_v1_latest.pt"))
    ap.add_argument("--pairs", default=paths.derived("reach_pairs_v1.npz"),
                    help="trunk path only; the ViT path samples pairs from the trajectory store")
    ap.add_argument("--data", default=paths.derived("field_std_v2.npz"))
    ap.add_argument("--feats", default=paths.derived("trunk_feats/t1-256x10__field_std_v2.npy"))
    ap.add_argument("--pc-cache", default=paths.derived("field_std_v2_piececount.npz"))
    ap.add_argument("--rows", type=int, default=0, help="cap on eval positions (0 = all test rows)")
    ap.add_argument("--n-cross", type=int, default=200_000, help="cross-game probe pairs")
    ap.add_argument("--ply-tol", type=int, default=5, help="match cross-game partners on ply")
    ap.add_argument("--n-paired", type=int, default=60_000, help="targets sampled for the paired test")
    ap.add_argument("--n-diff", type=int, default=40_000, help="pairs for the capture/quiet differential")
    ap.add_argument("--by-source", action="store_true",
                    help="also report the differential WITHIN each population separately, to rule "
                         "out a human/SF mix artifact in the pooled number")
    ap.add_argument("--source-blind", action="store_true",
                    help="THE control: zero the predictor input layer AND constant the source "
                         "embedding. Must score EXACTLY 0.500 on the paired ratchet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    net, payload = load_net(args.ckpt, args.device)
    is_vit = payload["cfg"].get("arch", TRUNK) == VIT
    if args.source_blind:
        if not hasattr(net, "blind_source"):
            raise SystemExit("--source-blind needs a ViT checkpoint (ReachViT.blind_source)")
        net.blind_source()
        print("[interpret] SOURCE-BLIND control active -- paired ratchet must be exactly 0.5000")
    source, pc, ply_all, game_all, tr, rows = build_eval_context(net, payload, args)
    print(f"[interpret] ckpt step {payload.get('step')} | arch {'vit' if is_vit else 'trunk'} | "
          f"piece count {pc.min()}..{pc.max()} [{time.time()-t0:.0f}s]", flush=True)

    arms = [REGION, IQE_ARM] if is_vit else [REGION]

    # ---- 1. THE RATCHET TEST (paired: same target, sources differ) -------------------------------
    print(f"\n  PAIRED ratchet test -- same target b, source varies. A model that ignores the")
    print(f"  source scores exactly 0.500 here by construction.")
    paired_by_arm = {}
    for arm in arms:
        print(f"  [{arm}] {'k':>3} {'n pairs':>9} {'P(plausible > impossible)':>26}")
        vals = []
        for k in (1, 2, 3, 4, 6, 8):
            r, nn = paired_ratchet(net, source, pc[rows], ply_all[rows], game_all[rows],
                                   k, args.n_paired, rng, args.device, arm=arm)
            if nn:
                vals.append(r)
                print(f"  {'':>{len(arm)+2}} {k:>3} {nn:>9,} {r:>26.4f}")
        paired_by_arm[arm] = float(np.mean(vals)) if vals else float("nan")

    # ---- 2. cross-game probe pairs, matched on ply -----------------------------------------------
    n = len(rows)
    order = rows[np.argsort(ply_all[rows], kind="stable")]
    ply_sorted = ply_all[order]
    ia = rows[rng.integers(0, n, args.n_cross)]
    lo = np.searchsorted(ply_sorted, ply_all[ia] - args.ply_tol, "left")
    hi = np.searchsorted(ply_sorted, ply_all[ia] + args.ply_tol, "right")
    pick = (lo + (rng.random(args.n_cross) * np.maximum(hi - lo, 1))).astype(np.int64)
    ib = order[np.clip(pick, 0, n - 1)]
    ok = game_all[ia] != game_all[ib]
    ia, ib = ia[ok], ib[ok]
    s_cross = score_pairs(net, source, ia, ib, args.device)
    dpc = (pc[ib].astype(int) - pc[ia].astype(int))
    dply = np.abs(ply_all[ia].astype(int) - ply_all[ib].astype(int))
    print(f"\n[interpret] cross-game pairs {len(ia):,} | |dply| median {np.median(dply):.0f} "
          f"| delta-pieces p5/50/95 {np.percentile(dpc,[5,50,95]).astype(int).tolist()}", flush=True)

    # ---- 3. THE DIFFERENTIAL (ViT only): capture-crossing vs quiet, both UNOBSERVED --------------
    diff = {}
    if is_vit and tr is not None:
        diff = differential(net, source, tr, rows, rng, args.device, n=args.n_diff)
        print(f"\n  DIFFERENTIAL -- the strata verdict. `capture` and `quiet` are BOTH unobserved")
        print(f"  reversals given IDENTICAL repulsion in training and matched on ply gap, so any")
        print(f"  difference between them came from the data, not the objective.")
        print(f"  {'group':>9} {'n':>8} {'d(b->a)/d(a->b)':>17} {'score(a->b)-score(b->a)':>25}")
        for name in ("capture", "quiet", "revers."):
            r, s, nn = diff.get(name, (float("nan"), float("nan"), 0, None))[:3]
            tag = "  <- trained-in reference, NOT evidence" if name == "revers." else ""
            print(f"  {name:>9} {nn:>8,} {r:>17.3f} {s:>25.3f}{tag}")

    if args.by_source and is_vit and tr is not None:
        print(f"\n  SOURCE-STRATIFIED differential -- the pooled number must survive INSIDE each")
        print(f"  population, or it is a mix artifact rather than a property of chess.")
        for pop, lab in ((T.HUMAN, "human"), (T.SF, "sf-vs-sf")):
            dsub = differential(net, source, tr, rows, rng, args.device, n=args.n_diff, pop=pop)
            c = dsub.get("capture", (0, 0, 0, None)); q = dsub.get("quiet", (0, 0, 0, None))
            pt, lo, hi = bootstrap_diff(c[3], q[3], rng)
            verdict = "holds" if (lo > 0) else ("REVERSED" if hi < 0 else "not established")
            print(f"  {lab:>9}  capture {c[0]:.3f} (n={c[2]:,})  quiet {q[0]:.3f} (n={q[2]:,})  "
                  f"diff {pt:+.3f} CI [{lo:+.3f},{hi:+.3f}]  <- {verdict}")

    # ---- 4. the OLD matched-magnitude test, retained and labelled CONFOUNDED ---------------------
    print(f"\n  [CONFOUNDED -- retained for comparison only] matched-|delta| test. Its +k group has")
    print(f"  piece-RICH targets and its -k group piece-POOR ones, so a target-only model passes it.")
    print(f"  {'|dpc|':>5} {'n(+k)':>8} {'n(-k)':>8} {'mean s(+k)':>11} {'mean s(-k)':>11} "
          f"{'gap':>8} {'AUC(-k>+k)':>11}")
    aucs = []
    for k in range(1, 9):
        p_up, p_dn = s_cross[dpc == k], s_cross[dpc == -k]
        if len(p_up) < 50 or len(p_dn) < 50:
            continue
        a = auc(p_dn, p_up)
        aucs.append(a)
        print(f"  {k:>5} {len(p_up):>8,} {len(p_dn):>8,} {p_up.mean():>11.3f} {p_dn.mean():>11.3f} "
              f"{p_dn.mean()-p_up.mean():>8.3f} {a:>11.3f}")
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")

    # ---- 5. the confound it must beat, and basic separation --------------------------------------
    rho_signed = spearman(s_cross, dpc.astype(float))
    rho_abs = spearman(s_cross, -np.abs(dpc).astype(float))
    if is_vit:
        ri = rows[rng.integers(0, n, 50_000)]
        rg = game_all[ri]
        rj = np.minimum(ri + 1 + rng.integers(1, 30, len(ri)),
                        tr.start[rg] + tr.length[rg] - 1)
        keep = rj > ri
        real = score_pairs(net, source, ri[keep], rj[keep], args.device)
    else:
        zp = np.load(args.pairs, allow_pickle=True)
        m = (zp["i"] < len(rows)) & (zp["j"] < len(rows)) & (zp["split"] == 2)
        real = score_pairs(net, source, zp["i"][m], zp["j"][m], args.device)[zp["gap"][m] > 8]
    sep = auc(real, s_cross)

    # ---- 6. what the L1 kept ---------------------------------------------------------------------
    w = net.head_in.weight.detach().abs().mean(0).cpu().numpy()
    keep_n = int((net.head_in.weight.detach().abs() > 0).any(0).sum())

    print(f"\n  observed-reachable vs cross-game AUC   {sep:.3f}   (can it tell them apart at all)")
    print(f"  spearman(score,  delta_pieces)         {rho_signed:+.3f}   <- SIGNED: the ratchet")
    print(f"  spearman(score, -|delta_pieces|)       {rho_abs:+.3f}   <- magnitude only: the confound")
    print(f"  L1 kept {keep_n}/{len(w)} input coords of the predictor")

    cap = diff.get("capture", (float("nan"),))[0]
    qui = diff.get("quiet", (float("nan"),))[0]
    rev = diff.get("revers.", (float("nan"),))[0]
    cm_pt, cm_lo, cm_hi = bootstrap_diff(diff.get("capture", (0, 0, 0, None))[3],
                                         diff.get("quiet", (0, 0, 0, None))[3], rng) if diff \
        else (float("nan"),) * 3
    if diff:
        print(f"\n  capture - quiet = {cm_pt:+.3f}  95% CI [{cm_lo:+.3f}, {cm_hi:+.3f}]"
              f"  {'ESTABLISHED (excludes 0)' if (cm_lo > 0 or cm_hi < 0) else 'NOT established (CI spans 0)'}")
    print(f"\nVERDICT REACH-STRATA-VIT arch={'vit' if is_vit else 'trunk'} "
          f"paired_ratchet_region={paired_by_arm.get(REGION, float('nan')):.4f} "
          f"paired_ratchet_iqe={paired_by_arm.get(IQE_ARM, float('nan')):.4f} "
          f"diff_capture={cap:.3f} diff_quiet={qui:.3f} diff_reversible={rev:.3f} "
          f"capture_minus_quiet={cm_pt:+.3f} ci=[{cm_lo:+.3f},{cm_hi:+.3f}] "
          f"confounded_ratchet={mean_auc:.4f} rho_signed={rho_signed:+.4f} rho_absonly={rho_abs:+.4f} "
          f"sep_auc={sep:.4f} l1_kept={keep_n}/{len(w)} "
          f"source_blind={int(args.source_blind)} step={payload.get('step')} [{time.time()-t0:.0f}s]")
    print("  READ, in order: (1) paired_ratchet against the SAME number from the --random-init ckpt;")
    print("  (2) capture_minus_quiet, which the uniform repulsion cannot have produced; (3) the")
    print("  source-blind run, which must print exactly 0.5000. confounded_ratchet is kept only to")
    print("  document the earlier metric that a source-blind model also passed.")


if __name__ == "__main__":
    main()
