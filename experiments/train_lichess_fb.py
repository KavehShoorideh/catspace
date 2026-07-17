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
    # game result per anchor (+1 White win / 0 draw / -1 Black win) for the
    # outcome-poles/axis losses. Already used for zgoals -- not an eval-leak signal.
    result = batch.meta["result"][idx].astype(np.float32)
    pte = batch.meta.get("plies_to_end")
    plies_to_end = (pte[idx].astype(np.float32) if pte is not None
                    else np.full(len(idx), 1e6, dtype=np.float32))
    out = [torch.from_numpy(planes_s).to(device),
           torch.from_numpy(om).to(device),
           torch.from_numpy(planes_g).to(device),
           torch.from_numpy(ply_gap).to(device),
           torch.from_numpy(material_drop).to(device),
           torch.from_numpy(result).to(device),
           torch.from_numpy(plies_to_end).to(device)]
    # QRL successor (1-ply transition s->s') + valid mask, when the source
    # provides it. valid = successor is a real distinct position (not a game's
    # last row, where succ==self would give a trivial d(s,s)=0 constraint).
    if "packed_succ" in batch.meta:
        planes_succ = feature_planes(batch.meta["packed_succ"][idx],
                                     batch.meta["board_meta_succ"][idx])
        valid = ~batch.meta["succ_is_last"][idx]
        out.append(torch.from_numpy(planes_succ).to(device))
        out.append(torch.from_numpy(valid).to(device))
    return tuple(out)


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
    if getattr(fb, "outcome_poles", False):
        # unify the goal: the planner should navigate toward the LEARNED win pole
        # (the vector F was organised around), not the checkmate-centroid
        # side-channel. Keep the centroids under *_CENTROID for reference.
        with torch.no_grad():
            p = torch.nn.functional.normalize(fb.poles.detach(), dim=1).cpu()
        zgoals["MATE_W_CENTROID"], zgoals["MATE_B_CENTROID"] = zgoals["MATE_W"], zgoals["MATE_B"]
        zgoals["POLE_B"], zgoals["POLE_D"], zgoals["POLE_W"] = p[0], p[1], p[2]
        zgoals["MATE_W"], zgoals["MATE_B"] = p[2], p[0]           # win / loss pole
        zgoals["MATE_DIFF"] = p[2] - p[0]
        if verbose:
            print("zgoals: MATE_W/B overridden with learned win/loss POLES")
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
    ap.add_argument("--cert-base", action="store_true",
                    help="certainty in the BASE objective (2026-07-15, toy-validated): "
                         "win-prob head on F trained on game results (outcome-conditioned, "
                         "no oracle), and for won games regress d(F(s), zgoal) to "
                         "(plies_to_end + lam*(-ln P_head))/scale -- the promoted toy "
                         "target with the head standing in for rollout counts")
    ap.add_argument("--cert-base-weight", type=float, default=1.0)
    ap.add_argument("--phead-weight", type=float, default=0.3)
    ap.add_argument("--committor-base", action="store_true",
                    help="committor-in-base-objective (2026-07-16): train the 3-class "
                         "outcome head (= multinomial committor over the W/D/L "
                         "boundary surfaces) at --phead-weight alongside NCE+ply-gap; "
                         "NO pole-distance cert term, no goal vectors -- play reads "
                         "out via playout_ab --phead-b. The zero-training transfer "
                         "result (cert_base's by-product phead beating toy-trained "
                         "fields at depth) is this mode's floor.")
    ap.add_argument("--cert-lam", type=float, default=8.0)
    ap.add_argument("--cert-scale", type=float, default=50.0)
    ap.add_argument("--zgoal-refresh", type=int, default=2000,
                    help="re-embed MATE_W/B goal centroids every N steps (they drift as F/B train)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--iqe", action="store_true",
                    help="Interval Quasimetric Embedding distance head (merged paper): "
                         "valid+universal quasimetric BY CONSTRUCTION, replaces the "
                         "MRN metric_scale/W score. Right geometry for the field.")
    ap.add_argument("--iqe-components", type=int, default=32,
                    help="IQE component count (d must divide it); k=d/components "
                         "is the per-component interval-union dim")
    ap.add_argument("--iqe-embed-scale", type=float, default=50.0,
                    help="fixed scale on IQE embeddings (un-normalized). 50 bootstraps "
                         "InfoNCE; use ~1 with --qrl-objective (QRL sets its own scale "
                         "via the push offset + unit-step constraint).")
    ap.add_argument("--qrl-objective", action="store_true",
                    help="train the IQE/quasimetric with the QRL objective (Wang et al. "
                         "2023) it was DESIGNED for -- global push softplus(offset-d) on "
                         "random pairs + local d(s,s')<=1 constraint (dual-ascent lambda), "
                         "NO InfoNCE. Fixes the interval collapse InfoNCE leaves.")
    ap.add_argument("--qrl-lambda-lr", type=float, default=0.01,
                    help="dedicated LR for the QRL Lagrange multiplier (dual ascent), "
                         "excluded from the cosine schedule. Higher = constraint tracks "
                         "faster so the global push can't inflate the unit-step distance.")
    ap.add_argument("--qrl-push-offset", type=float, default=40.0,
                    help="QRL global-push target distance (plies). Set WELL beyond the "
                         "longest forcing line so reachable long lines (chained to their "
                         "true ply length) stay closer than unreachable random pairs -- "
                         "it is a saturating prior, NOT a horizon cap.")
    ap.add_argument("--channels", type=int, default=64, help="trunk conv width")
    ap.add_argument("--blocks", type=int, default=6, help="trunk residual blocks")
    ap.add_argument("--enc-out", type=int, default=256, help="encoder output dim")
    ap.add_argument("--dh", type=int, default=512, help="head hidden dim")
    ap.add_argument("--l1-metric-scale", type=float, default=0.0,
                    help="L1 tax on the per-dimension quasimetric metric_scale "
                         "(2026-07-16): prices DISTANCE dimensions so a wide "
                         "embedding allocates them sparsely/per-pattern, letting "
                         "rare regimes decouple from the frequent one instead of "
                         "being dragged as undefended collateral (effective rank "
                         "was ~7/64 regardless of width; drift ratio 1.71). The "
                         "representation stays free; only the metric is taxed.")
    ap.add_argument("--l1-warmup", type=int, default=10000,
                    help="steps of 0 L1 before the tax ramps in (explore wide "
                         "first, then sparsify)")
    ap.add_argument("--unreach-weight", type=float, default=0.0,
                    help="monotonicity hard-negative repulsion (2026-07-16): "
                         "each step, add one piece to every anchor -> a provably "
                         "unreachable goal (count strictly up), and push its "
                         "d(F(s),B(neg)) above a batch-relative margin. Exact, "
                         "free, directional hard negatives to speed separation; "
                         "quasimetric-safe (inf is allowed). GATE on not taxing "
                         "short-horizon sharpness (asymmetry-hinge precedent).")
    ap.add_argument("--unreach-margin-q", type=float, default=0.9,
                    help="margin = this quantile of the batch's positive-pair "
                         "distances + 0.25 (push negatives beyond reachable mass)")
    ap.add_argument("--horizon-k", type=float, default=0.0,
                    help="bound the quasimetric at k plies: calibrate distance to "
                         "min(k, ply_gap)/scale (Kaveh 2026-07-16). k~=10 makes the "
                         "measured ~10-ply retrieval horizon explicit; beyond-k "
                         "positions become the natural contrast class.")
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
    ap.add_argument("--distributional", action="store_true",
                    help="option B (UNCERTAINTY_DESIGN.md): add a CATEGORICAL head predicting "
                         "distance-to-goal over ply-gap bins (cross-entropy); quasimetric d stays "
                         "the planning distance, categorical entropy = the uncertainty signal. "
                         "Forces quasimetric.")
    ap.add_argument("--n-bins", type=int, default=12, help="distributional: number of ply-gap bins")
    ap.add_argument("--dist-weight", type=float, default=0.5, help="distributional: weight on the categorical loss")
    ap.add_argument("--competence", action="store_true",
                    help="Method 2 (training-integrated): add a head predicting the model's own "
                         "per-anchor retrieval error (epistemic 'where I fit poorly'). Native, "
                         "always-current competence signal for reliability-gated search.")
    ap.add_argument("--competence-weight", type=float, default=0.1, help="weight on the competence head loss")
    ap.add_argument("--outcome-poles", action="store_true",
                    help="add 3 learnable terminal poles (loss/draw/win) and a loss that repels "
                         "them and hinges each state's quasimetric HOPS so its own-outcome pole is "
                         "closer than the others -- outcome-conditioned region separation. Forces "
                         "quasimetric.")
    ap.add_argument("--outcome-weight", type=float, default=0.3, help="weight on the outcome-poles loss")
    ap.add_argument("--pole-tau", type=float, default=1.0,
                    help="softmax temperature for the soft outcome-pole cross-entropy (higher = "
                         "softer/gentler pull, less region compression)")
    ap.add_argument("--pole-margin", type=float, default=3.0,
                    help="minimum (scaled) distance kept between the three poles")
    ap.add_argument("--repel-weight", type=float, default=0.0,
                    help="cross-outcome repulsion (t-SNE-style, no attractor): push DIFFERENT-"
                         "outcome anchor pairs apart in hops up to --repel-margin. Needs quasimetric; "
                         "no new params (uses embed_B on anchors).")
    ap.add_argument("--repel-margin", type=float, default=1.5,
                    help="hops margin cross-outcome pairs are repelled up to (then force saturates)")
    ap.add_argument("--concept-axes", type=int, default=0,
                    help="number of learnable concept DIRECTIONS (slot 0 = outcome axis). Each "
                         "concept separates from its opposite along its own axis; other dims stay "
                         "free so different concepts' regions can overlap (multi-concept superposition).")
    ap.add_argument("--axis-weight", type=float, default=0.5, help="outcome-axis hinge weight")
    ap.add_argument("--axis-margin", type=float, default=1.0)
    ap.add_argument("--axis-gate-plies", type=float, default=8.0,
                    help="proximity gate: pull strength ~ exp(-plies_to_end/this)")
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
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="also save a step-tagged LADDER checkpoint every N steps (kept, not "
                         "overwritten) for early-stopping: eval the downstream metric across "
                         "the ladder and pick/stop at the peak instead of a fixed --steps budget")
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
        fb = TorchFB(d=args.d, channels=args.channels, blocks=args.blocks,
                     enc_out=args.enc_out, dh=args.dh,
                     seed=args.seed, quasimetric=args.quasimetric,
                     two_horizon=args.two_horizon, distributional=args.distributional,
                     n_bins=args.n_bins, competence=args.competence,
                     outcome_poles=args.outcome_poles, concept_axes=args.concept_axes,
                     iqe=args.iqe, iqe_components=args.iqe_components,
                     iqe_embed_scale=args.iqe_embed_scale)
        print(f"model params: {sum(p.numel() for p in fb.parameters())/1e6:.1f}M "
              f"(d={args.d} channels={args.channels} blocks={args.blocks} enc_out={args.enc_out})")
        fb.to(device)
    start_step = step                      # cosine decay spans [start_step, args.steps)
    lr_min = args.lr_min if args.lr_min is not None else args.lr / 10
    if args.qrl_objective and getattr(fb, "quasimetric", False):
        # the Lagrange multiplier gets its OWN, higher LR and is excluded from
        # the cosine schedule: dual ascent must track the constraint fast enough
        # to keep the global push from inflating the 1-ply step distance.
        main_params = [p for n, p in fb.named_parameters() if n != "qrl_raw_lambda"]
        opt = torch.optim.AdamW(main_params, lr=args.lr)
        opt.add_param_group({"params": [fb.qrl_raw_lambda], "lr": args.qrl_lambda_lr,
                             "is_lambda": True})
    else:
        opt = torch.optim.AdamW(fb.parameters(), lr=args.lr)
    if ckpt_path.exists() and not args.fresh:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "opt_state" in payload:
            try:
                opt.load_state_dict(payload["opt_state"])
            except ValueError as e:
                # e.g. a prior round added the pole param group (2 groups) but this
                # resume constructs poles as part of the model (1 group). Momentum
                # is disposable for a fine-tune -> continue with a fresh optimizer.
                print(f"opt_state not restored ({e}); fresh optimizer", flush=True)
    # resuming a checkpoint that predates outcome-poles: bolt the 3 poles onto
    # the loaded model AFTER opt_state restore, as a fresh param group (so the
    # restored optimizer state for the existing params still lines up).
    if args.outcome_poles and not getattr(fb, "outcome_poles", False):
        assert fb.quasimetric, "--outcome-poles needs a quasimetric checkpoint"
        import torch.nn as nn
        fb.poles = nn.Parameter(nn.functional.normalize(
            torch.randn(3, fb.d), dim=1).to(device))
        fb.outcome_poles = True
        fb.config["outcome_poles"] = True
        opt.add_param_group({"params": [fb.poles]})
        print("added outcome poles to resumed model", flush=True)
    # same bolt-on for concept axes on a pre-axis checkpoint
    if args.concept_axes > 0 and getattr(fb, "n_concept_axes", 0) == 0:
        import torch.nn as nn
        fb.concept_axes = nn.Parameter(nn.functional.normalize(
            torch.randn(args.concept_axes, fb.d), dim=1).to(device))
        fb.n_concept_axes = args.concept_axes
        fb.config["concept_axes"] = args.concept_axes
        opt.add_param_group({"params": [fb.concept_axes]})
        print(f"added {args.concept_axes} concept axes to resumed model", flush=True)

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

    phead, zW, zB = None, None, None
    if args.cert_base or args.committor_base:
        from catspace.nn.eval_head import EvalHead, descriptive_loss
        phead = EvalHead(d_in=args.d, seed=args.seed).to(device)
        opt.add_param_group({"params": phead.parameters()})
        mode = "cert-base" if args.cert_base else "committor-base"
        print(f"{mode} ON: phead {sum(q.numel() for q in phead.parameters())} params, "
              f"phead-weight={args.phead_weight}"
              + (f" lam={args.cert_lam} scale={args.cert_scale}" if args.cert_base else
                 " (no pole term, no goal vectors)"))
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
            if g.get("is_lambda"):
                continue                      # QRL multiplier keeps its fixed LR
            g["lr"] = lr_now
        core, result_t, pte_t = tensors[:5], tensors[5], tensors[6]
        if args.qrl_objective:
            if len(tensors) < 9:
                raise SystemExit("--qrl-objective needs the 1-ply successor from the "
                                 "shard source (packed_succ); re-run on shards built by "
                                 "the updated LichessPairSource.")
            planes_succ, valid = tensors[7], tensors[8]
            loss, qstats = fb.qrl_loss(core[0], core[1], planes_succ, core[2], valid,
                                       push_offset=args.qrl_push_offset)
            top1 = torch.zeros(())      # QRL has no in-batch retrieval term; VAL still tracks it
            if step % 100 == 0:
                print(f"    qrl push {qstats['push']:.3f} sq_dev {qstats['sq_dev']:.4f} "
                      f"lam {qstats['lam']:.3f} d_step {qstats['d_step']:.3f} "
                      f"d_rand {qstats['d_rand']:.3f}", flush=True)
        elif args.two_horizon:
            ps, om, pg, gap, _mdrop = core
            loss, top1 = fb.two_horizon_loss(ps, om, pg, gap, near_max=args.near_max,
                                             far_min=args.far_min, near_weight=args.near_weight,
                                             ply_gap_weight=args.ply_gap_weight,
                                             ply_gap_scale=args.ply_gap_scale)
        else:
            loss, top1 = fb.loss_fn(*core, ply_gap_weight=args.ply_gap_weight,
                                    ply_gap_scale=args.ply_gap_scale,
                                    asym_weight=args.asym_weight, asym_margin=args.asym_margin,
                                    dist_weight=args.dist_weight,
                                    competence_weight=args.competence_weight,
                                    result=result_t, outcome_weight=args.outcome_weight,
                                    pole_tau=args.pole_tau, pole_margin=args.pole_margin,
                                    repel_weight=args.repel_weight, repel_margin=args.repel_margin,
                                    plies_to_end=pte_t, axis_weight=args.axis_weight,
                                    axis_margin=args.axis_margin,
                                    axis_gate_plies=args.axis_gate_plies,
                                    horizon_k=args.horizon_k)
        if args.committor_base:
            ps_c, om_c = core[0], core[1]
            f_s = fb.embed_F(ps_c, om_c)
            p_loss = descriptive_loss(phead, f_s, result_t.long())
            loss = loss + args.phead_weight * p_loss
            if step % 100 == 0:
                print(f"    phead {float(p_loss):.4f}", flush=True)
        if args.cert_base:
            if zW is None or step % args.zgoal_refresh == 0:
                zg = embed_zgoals(fb, finals, device)
                zW = zg["MATE_W"].to(device).float().detach()
                zB = zg["MATE_B"].to(device).float().detach()
            ps_c, om_c = core[0], core[1]
            f_s = fb.embed_F(ps_c, om_c)
            p_loss = descriptive_loss(phead, f_s, result_t.long())
            probs = torch.softmax(phead(f_s), dim=1).detach()
            cert = torch.zeros((), device=device)
            ok = torch.isfinite(pte_t)
            for res_val, z, cls in ((1, zW, 0), (-1, zB, 2)):
                m = (result_t == res_val) & ok
                if int(m.sum()) >= 8:
                    d = fb.distance_matrix(f_s[m], z[None, :])[:, 0]
                    ph = probs[m, cls].clamp_min(1e-3)
                    tgt = (pte_t[m] + args.cert_lam * (-torch.log(ph))) / args.cert_scale
                    cert = cert + ((d - tgt) ** 2).mean()
            loss = loss + args.phead_weight * p_loss + args.cert_base_weight * cert
            if step % 100 == 0:
                print(f"    cert {float(cert):.4f}  phead {float(p_loss):.4f}", flush=True)
        if args.unreach_weight > 0 and getattr(fb, "quasimetric", False):
            from catspace.nn.hard_negatives import repel_loss, unreachable_goals
            neg_packed = unreachable_goals(batch.anchors[np.flatnonzero(
                (batch.meta["game_id"] % HOLDOUT_MOD) != 0)], seed=step)
            om_neg = omega_ids(np.zeros(len(neg_packed)), np.zeros(len(neg_packed)),
                               np.full(len(neg_packed), np.nan))  # goal side: omega unused
            neg_planes = feature_planes(neg_packed, batch.meta["board_meta"][np.flatnonzero(
                (batch.meta["game_id"] % HOLDOUT_MOD) != 0)])
            b_neg = fb.embed_B(torch.from_numpy(neg_planes).to(device))
            f_s = fb.embed_F(core[0], core[1])
            d_neg = fb.distance_matrix(f_s, b_neg).diagonal()
            d_pos = fb.distance_matrix(f_s, fb.embed_B(core[2])).diagonal().detach()
            margin = torch.quantile(d_pos, args.unreach_margin_q) + 0.25
            loss = loss + args.unreach_weight * repel_loss(d_neg, margin)
        if args.l1_metric_scale > 0 and getattr(fb, "quasimetric", False):
            # tax on the DISTANCE dimensions (metric_scale), ramped after warmup:
            # a wide embedding is free; using many dims in the METRIC costs.
            ramp = min(1.0, max(0.0, (step - args.l1_warmup) / max(1, args.l1_warmup)))
            l1 = args.l1_metric_scale * ramp * fb.metric_scale.abs().sum()
            loss = loss + l1
            if step % 100 == 0 and ramp > 0:
                nz = int((fb.metric_scale.abs() > 1e-3).sum())
                print(f"    l1 {float(l1):.4f}  active_dims {nz}/{fb.d}", flush=True)
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
        if args.ckpt_every and step % args.ckpt_every == 0 and step < args.steps:
            # step-tagged LADDER checkpoint (kept, not overwritten) so early
            # stopping can pick the peak of the real downstream metric across
            # steps instead of trusting a fixed budget (2026-07-13, Kaveh)
            ladder = ckpt_path.with_name(f"{ckpt_path.stem}_step{step}{ckpt_path.suffix}")
            save_ckpt(fb, ladder, step=step, opt=opt,
                      zgoals=embed_zgoals(fb, finals, device), provenance=provenance)
            if phead is not None:
                # phead saves WITH each snapshot (2026-07-16: without it, the
                # 5k-mates/155k-shuffles regression couldn't be localized)
                torch.save({"state": phead.state_dict(), "d_in": args.d},
                           ladder.with_name(ladder.stem + "_phead.pt"))
            print(f"  ladder checkpoint -> {ladder.name}", flush=True)

    zgoals = embed_zgoals(fb, finals, device, verbose=True)
    save_ckpt(fb, ckpt_path, step=step, opt=opt, zgoals=zgoals, provenance=provenance)
    print(f"saved {ckpt_path}")
    # NOTE 2026-07-16: periodic --ckpt-every snapshots historically saved
    # WITHOUT their phead, making step-wise play forensics impossible (the
    # 5k-mates / 155k-shuffles rook regression could not be localized).
    # Snapshot pheads now save alongside (see periodic-save block).
    if phead is not None:
        hp = ckpt_path.with_name(ckpt_path.stem + "_phead.pt")
        torch.save({"state": phead.state_dict(), "d_in": args.d}, hp)
        print(f"saved {hp}")

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
