#!/usr/bin/env python
"""
experiments/train_lichess_fb.py — train TorchFB (full-board Forward-Backward
embedding) on Lichess position shards built by build_lichess_shards.py.

Holdout: every game with game_id % 50 == 0 is never trained on; validation
retrieval and the reach-slope verdicts use only those games.

Verdicts printed at the end:
  VAL_TOP1     in-batch retrieval acc at batch size --batch (chance ~1/batch)
  REACH_SLOPE_WON / _LOST  mean per-game spearman(ply, F(s)@zMATE_W) on
               held-out won/lost games -- the cone must tighten toward the
               mate as a WON game progresses; lost games should not.

Resumable: if --ckpt exists it is loaded (model+optimizer+step) and training
continues to --steps total.

LR schedule: cosine decay from --lr down to --lr-min, spanning THIS
invocation's remaining steps (resume step -> --steps), not the whole
training history -- each `--steps` extension gets its own fresh decay.
2026-07-11 finding: a 30k-step extension at a CONSTANT lr=3e-4 (no decay)
measurably hurt downstream planner quality (decompose FRAC_IMPROVED 0.833
-> 0.617, MEAN_GAIN 0.417 -> 0.310) even though raw retrieval loss looked
fine -- consistent with the literature on InfoNCE-style contrastive training
being prone to representation drift/dimensional collapse under a
non-decaying LR (SimCLR/CLIP both decay to ~1/10 of peak). See JOURNAL.md.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from catspace.audit import build_provenance, is_provenance_clean
from catspace.data.encode import board_from_packed
from catspace.data.shards import LichessPairSource, MixedPairSource
from catspace.io.paths import derived_dir, newest_shard_dir
from catspace.nn.fb import TorchFB, load_ckpt, pick_device, save_ckpt
from catspace.nn.features import feature_planes, omega_ids

HOLDOUT_MOD = 50


def batch_tensors(batch, device):
    """PairBatch -> (planes_s, omega_s, planes_g, ply_gap) on device, holdout
    rows dropped.

    2026-07-12: the --winner-pov-only filter (round 11) is REMOVED, per
    Kaveh. Rationale: (a) the good/bad information is already IN the data
    -- a goal position that is a mate for the mover is a good future, a
    mate against them is a bad one; the loss should see both and learn the
    geometry of each, not have losing trajectories censored out; (b) the
    ply-gap calibration term specifically NEEDS unrecoverable losing
    trajectories to learn what "no way back" looks like as a distance --
    filtering them out deletes exactly that training signal; (c) the
    filter interacted pathologically with the batch-size skip guard in
    main() (keeping ~48% of a 512-row batch lands right under the
    batch//2=256 skip threshold, so most batches were built then thrown
    away -- the round-13 first launch spun for 35 min without completing
    100 steps because of this)."""
    train_mask = (batch.meta["game_id"] % HOLDOUT_MOD) != 0
    if not train_mask.any():
        return None
    idx = np.flatnonzero(train_mask)
    planes_s = feature_planes(batch.anchors[idx], batch.meta["board_meta"][idx])
    planes_g = feature_planes(batch.goals[idx], batch.meta["board_meta_g"][idx])
    om = omega_ids(batch.meta["white_elo"][idx], batch.meta["black_elo"][idx],
                   batch.meta["clock"][idx])
    ply_gap = (batch.meta["ply_g"][idx].astype(np.float32)
               - batch.meta["ply"][idx].astype(np.float32))
    # material strictly decreased from anchor to goal: the pair crossed a
    # capture, so the REVERSE hop (goal -> anchor) is impossible in real
    # chess -- the asymmetry-margin term (loss_fn) trains on exactly these
    material_drop = (np.bitwise_count(batch.anchors[idx]).sum(axis=1)
                     > np.bitwise_count(batch.goals[idx]).sum(axis=1))
    return (torch.from_numpy(planes_s).to(device),
            torch.from_numpy(om).to(device),
            torch.from_numpy(planes_g).to(device),
            torch.from_numpy(ply_gap).to(device),
            torch.from_numpy(material_drop).to(device))


def collect_holdout(src: LichessPairSource, n_batches: int, batch_size: int, seed: int):
    """Fixed holdout pair batches (packed+meta kept small; planes built lazily)."""
    out = []
    buf_a, buf_g, buf_ma, buf_mg, buf_om = [], [], [], [], []
    for batch in src.batches(batch_size, seed):
        held = np.flatnonzero(batch.meta["game_id"] % HOLDOUT_MOD == 0)
        if held.size == 0:
            continue
        buf_a.append(batch.anchors[held]); buf_g.append(batch.goals[held])
        buf_ma.append(batch.meta["board_meta"][held]); buf_mg.append(batch.meta["board_meta_g"][held])
        buf_om.append(omega_ids(batch.meta["white_elo"][held], batch.meta["black_elo"][held],
                                batch.meta["clock"][held]))
        if sum(len(a) for a in buf_a) >= n_batches * batch_size:
            break
    a = np.concatenate(buf_a); g = np.concatenate(buf_g)
    ma = np.concatenate(buf_ma); mg = np.concatenate(buf_mg); om = np.concatenate(buf_om)
    for i in range(0, min(len(a), n_batches * batch_size), batch_size):
        sl = slice(i, i + batch_size)
        if len(a[sl]) == batch_size:
            out.append((a[sl], ma[sl], om[sl], g[sl], mg[sl]))
    return out


@torch.no_grad()
def val_metrics(fb, holdout, device):
    fb.eval()
    top1s, top8s, losses = [], [], []
    for a, ma, om, g, mg in holdout:
        ps = torch.from_numpy(feature_planes(a, ma)).to(device)
        pg = torch.from_numpy(feature_planes(g, mg)).to(device)
        o = torch.from_numpy(om).to(device)
        f = fb.embed_F(ps, o); b = fb.embed_B(pg)
        logits = fb.score_matrix(f, b) / fb.tau
        target = torch.arange(len(f), device=device)
        losses.append(float(torch.nn.functional.cross_entropy(logits, target)))
        ranks = (logits >= logits.gather(1, target[:, None])).sum(1)
        top1s.append(float((ranks <= 1).float().mean()))
        top8s.append(float((ranks <= 8).float().mean()))
    fb.train()
    return float(np.mean(losses)), float(np.mean(top1s)), float(np.mean(top8s))


def collect_mate_finals(shard_dir: Path, cap: int = 2048) -> dict:
    """(packed, meta) of final CHECKMATE positions of decisive games
    (include_final=True stored them), keyed by white-POV result (MATE_W =
    result +1). The shard scan is the expensive half of zgoal building —
    do it ONCE, re-embed at every save."""
    finals = {1: [], -1: []}
    for path in sorted(shard_dir.glob("shard_*.npz")):
        npz = np.load(path)
        gid, result = npz["game_id"], npz["result"]      # bind once (NpzFile re-reads per access)
        packed, meta = npz["packed"], npz["meta"]
        last = np.flatnonzero(np.r_[np.diff(gid) != 0, True])
        for row in last:
            res = int(result[row])
            if res == 0 or len(finals[res]) >= cap:
                continue
            board = board_from_packed(packed[row], meta[row])
            if board.is_checkmate():
                finals[res].append((packed[row], meta[row]))
        if all(len(v) >= cap for v in finals.values()):
            break
    return finals


def embed_zgoals(fb, finals: dict, device, verbose: bool = False) -> dict:
    """zMATE_W / zMATE_B = mean B over the collected mate finals under the
    CURRENT weights, plus MATE_DIFF (the outcome direction: cancels the
    "generic finality" component the two mate goals share — diagnosed
    2026-07-11, it made raw reach slopes identical for won and lost games).
    Attached at every periodic save so an interrupted run still leaves a
    planner-usable checkpoint (the 2026-07-11 zgoals-less checkpoint bug)."""
    was_training = fb.training
    fb.eval()
    zgoals = {}
    with torch.no_grad():
        for res, name in ((1, "MATE_W"), (-1, "MATE_B")):
            rows = finals[res]
            packed = np.stack([r[0] for r in rows]); meta = np.stack([r[1] for r in rows])
            planes = torch.from_numpy(feature_planes(packed, meta)).to(device)
            zgoals[name] = fb.embed_B(planes).mean(dim=0).cpu()   # FAR centroid
            if getattr(fb, "two_horizon", False):
                # NEAR centroid too -- the near readout's default goal (the
                # policy may swap in a near exemplar BANK at eval time, since
                # centroids are flat; centroid kept as a cheap baseline)
                zgoals[name + "_NEAR"] = fb.embed_B_near(planes).mean(dim=0).cpu()
            if verbose:
                print(f"zgoal {name}: {len(rows)} checkmate finals")
    zgoals["MATE_DIFF"] = zgoals["MATE_W"] - zgoals["MATE_B"]
    if was_training:
        fb.train()
    return zgoals


def build_zgoals(shard_dir: Path, fb, device, cap: int = 2048) -> dict:
    return embed_zgoals(fb, collect_mate_finals(shard_dir, cap), device, verbose=True)


def reach_slope(shard_dir: Path, fb, z, device, want_result: int, n_games: int = 200):
    """Mean per-game spearman(ply, reach) over held-out games with the given
    result -- the trajectory-level sanity check of the cone."""
    from scipy.stats import spearmanr
    rhos = []
    for path in sorted(shard_dir.glob("shard_*.npz")):
        npz = np.load(path)
        data = {k: npz[k] for k in npz.files}            # bind once (NpzFile re-reads per access)
        gid = data["game_id"]
        held = (gid % HOLDOUT_MOD == 0) & (data["result"] == want_result)
        for g in np.unique(gid[held]):
            lo, hi = np.searchsorted(gid, [g, g + 1])    # gid non-decreasing within a shard
            rows = np.arange(lo, hi)
            if len(rows) < 10:
                continue
            planes = torch.from_numpy(feature_planes(data["packed"][rows], data["meta"][rows])).to(device)
            om = torch.from_numpy(omega_ids(data["white_elo"][rows], data["black_elo"][rows],
                                            data["clock"][rows])).to(device)
            with torch.no_grad():
                reach = fb.score(fb.embed_F(planes, om), z.to(device)).cpu().numpy()
            rho = spearmanr(data["ply"][rows], reach).statistic
            if np.isfinite(rho):
                rhos.append(rho)
            if len(rhos) >= n_games:
                return float(np.mean(rhos)), len(rhos)
    return float(np.mean(rhos)) if rhos else float("nan"), len(rhos)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default=None, help="shard dir (default: newest under data/shards)")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--quasimetric", action="store_true",
                    help="score(f,g) = -d(f,g)+r(f,g), d a real (triangle-inequality-"
                         "respecting) metric, instead of a plain cosine dot product -- "
                         "see nn/fb.py module docstring. Only meaningful with --fresh "
                         "(resuming inherits quasimetric from the checkpoint's own config)")
    ap.add_argument("--ply-gap-weight", type=float, default=0.05,
                    help="quasimetric-only: weight of the MSE(d(f,g), ply_gap/scale) term "
                         "that calibrates the metric's ABSOLUTE scale to real move-distance "
                         "-- in-batch retrieval alone only enforces relative ranking. 0 disables.")
    ap.add_argument("--ply-gap-scale", type=float, default=50.0,
                    help="normalizer for the ply-gap target (roughly the pairing horizon's "
                         "mean, so the regression target starts near O(1))")
    ap.add_argument("--asym-weight", type=float, default=0.0,
                    help="quasimetric-only: weight of the asymmetry-margin hinge -- pairs "
                         "whose material dropped anchor->goal train d(reverse) > d(forward) "
                         "+ margin (you can't un-capture; derived from trajectory direction "
                         "only). 0 disables (default).")
    ap.add_argument("--asym-margin", type=float, default=0.2)
    ap.add_argument("--two-horizon", action="store_true",
                    help="two-horizon architecture (TWO_HORIZON_DESIGN.md): shared trunk + "
                         "separate near/far heads; near trained on short-gap pairs (cosine), "
                         "far on long-gap pairs (quasimetric + ply-gap). Forces quasimetric.")
    ap.add_argument("--near-max", type=int, default=8, help="two-horizon: near head sees gap <= this")
    ap.add_argument("--far-min", type=int, default=16, help="two-horizon: far head sees gap >= this")
    ap.add_argument("--near-weight", type=float, default=1.0, help="two-horizon: weight on the near loss")
    ap.add_argument("--selfplay-shards", default=None,
                    help="dir of experiments/selfplay_generate.py output shards to MIX into "
                         "training (holdout/val stay human-only for a stable reference)")
    ap.add_argument("--selfplay-frac", type=float, default=0.3,
                    help="fraction of TRAINING batches drawn from --selfplay-shards vs human data")
    ap.add_argument("--gamma", type=float, default=0.98,
                    help="pairing horizon: mean k=1+1/(1-gamma)=51 plies, on par with "
                         "typical stored game length (0.99's 101 snapped ~all goals to "
                         "final positions)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=None,
                    help="cosine-decay floor for THIS invocation's remaining steps "
                         "(resume step -> --steps); default lr/10 (SimCLR/CLIP convention)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt", default=None, help="default: data/derived/lichess_fb.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    args = ap.parse_args()

    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    ckpt_path = Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt"
    device = pick_device(args.device)
    print(f"shards={shard_dir.name} device={device} steps={args.steps} batch={args.batch} d={args.d} "
          f"quasimetric={args.quasimetric} ply_gap_weight={args.ply_gap_weight} "
          f"asym_weight={args.asym_weight} two_horizon={args.two_horizon}"
          + (f" (near<={args.near_max}/far>={args.far_min})" if args.two_horizon else ""),
          flush=True)

    step = 0
    if ckpt_path.exists() and not args.fresh:
        fb, payload = load_ckpt(ckpt_path, device)
        step = payload["step"]
        print(f"resumed {ckpt_path.name} at step {step}")
    else:
        fb = TorchFB(d=args.d, seed=args.seed, quasimetric=args.quasimetric,
                     two_horizon=args.two_horizon)
        fb.to(device)
    start_step = step                      # cosine decay spans [start_step, args.steps)
    lr_min = args.lr_min if args.lr_min is not None else args.lr / 10
    opt = torch.optim.AdamW(fb.parameters(), lr=args.lr)
    if ckpt_path.exists() and not args.fresh:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "opt_state" in payload:
            opt.load_state_dict(payload["opt_state"])

    human_src = LichessPairSource(shard_dir, gamma=args.gamma)
    src = human_src
    if args.selfplay_shards:
        selfplay_src = LichessPairSource(Path(args.selfplay_shards), gamma=args.gamma)
        src = MixedPairSource(human_src, selfplay_src, args.selfplay_frac)
        print(f"mixing self-play data from {args.selfplay_shards} at frac={args.selfplay_frac}",
              flush=True)
    holdout = collect_holdout(human_src, n_batches=8, batch_size=args.batch, seed=999)
    print(f"holdout: {len(holdout)} batches of {args.batch}", flush=True)
    finals = collect_mate_finals(shard_dir)

    provenance = build_provenance(
        script="train_lichess_fb.py", args=vars(args),
        data_columns_used=["packed", "meta", "game_id", "white_elo", "black_elo", "clock"],
        train_batch_fn=batch_tensors, train_main_fn=main)
    if not is_provenance_clean(provenance):
        raise SystemExit(f"static_purity_check found forbidden references, refusing to train: "
                         f"{provenance['static_check']['hits']}")

    fb.train()
    epoch = 0
    it = iter(src.batches(args.batch, seed=args.seed))
    t0 = time.time()
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            it = iter(src.batches(args.batch, seed=args.seed + epoch))
            continue
        tensors = batch_tensors(batch, device)
        if tensors is None or len(tensors[0]) < args.batch // 2:
            continue
        frac = min(1.0, (step - start_step) / max(1, args.steps - start_step))
        lr_now = lr_min + 0.5 * (args.lr - lr_min) * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr_now
        if args.two_horizon:
            ps, om, pg, gap, _mdrop = tensors
            loss, top1 = fb.two_horizon_loss(ps, om, pg, gap, near_max=args.near_max,
                                             far_min=args.far_min, near_weight=args.near_weight,
                                             ply_gap_weight=args.ply_gap_weight,
                                             ply_gap_scale=args.ply_gap_scale)
        else:
            loss, top1 = fb.loss_fn(*tensors, ply_gap_weight=args.ply_gap_weight,
                                    ply_gap_scale=args.ply_gap_scale,
                                    asym_weight=args.asym_weight, asym_margin=args.asym_margin)
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % 100 == 0:
            rate = 100 / (time.time() - t0); t0 = time.time()
            print(f"step {step}  loss {float(loss):.4f}  train_top1 {float(top1):.3f}  "
                  f"lr {lr_now:.2e}  ({rate:.1f} it/s)", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            vloss, vtop1, vtop8 = val_metrics(fb, holdout, device)
            print(f"  VAL step {step}  loss {vloss:.4f}  top1 {vtop1:.3f}  top8 {vtop8:.3f}", flush=True)
            save_ckpt(fb, ckpt_path, step=step, opt=opt,
                      zgoals=embed_zgoals(fb, finals, device), provenance=provenance)

    zgoals = embed_zgoals(fb, finals, device, verbose=True)
    save_ckpt(fb, ckpt_path, step=step, opt=opt, zgoals=zgoals, provenance=provenance)
    print(f"saved {ckpt_path}")

    vloss, vtop1, vtop8 = val_metrics(fb, holdout, device)
    slope_w, nw = reach_slope(shard_dir, fb, zgoals["MATE_W"], device, want_result=1)
    slope_l, nl = reach_slope(shard_dir, fb, zgoals["MATE_W"], device, want_result=-1)
    dslope_w, _ = reach_slope(shard_dir, fb, zgoals["MATE_DIFF"], device, want_result=1)
    dslope_l, _ = reach_slope(shard_dir, fb, zgoals["MATE_DIFF"], device, want_result=-1)
    print(f"VERDICT VAL_TOP1={vtop1:.3f} VAL_TOP8={vtop8:.3f} (chance {1/args.batch:.4f})")
    print(f"VERDICT REACH_SLOPE_WON={slope_w:.3f} (n={nw}) REACH_SLOPE_LOST={slope_l:.3f} (n={nl})")
    print(f"VERDICT DIFF_SLOPE_WON={dslope_w:.3f} DIFF_SLOPE_LOST={dslope_l:.3f} "
          f"(won-lost separation is the outcome signal; both were negative at step 2000)")


if __name__ == "__main__":
    main()
