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
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from latentchess.data.encode import board_from_packed
from latentchess.data.shards import LichessPairSource
from latentchess.io.paths import derived_dir, shards_dir
from latentchess.nn.fb import TorchFB, load_ckpt, pick_device, save_ckpt
from latentchess.nn.features import feature_planes, omega_ids

HOLDOUT_MOD = 50


def newest_shard_dir() -> Path:
    dirs = [p for p in shards_dir().iterdir() if p.is_dir() and list(p.glob("shard_*.npz"))]
    if not dirs:
        raise SystemExit("no shard dirs under data/shards -- run experiments/build_lichess_shards.py first")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def batch_tensors(batch, device):
    """PairBatch -> (planes_s, omega_s, planes_g) on device, holdout rows dropped."""
    train_mask = (batch.meta["game_id"] % HOLDOUT_MOD) != 0
    if not train_mask.any():
        return None
    idx = np.flatnonzero(train_mask)
    planes_s = feature_planes(batch.anchors[idx], batch.meta["board_meta"][idx])
    planes_g = feature_planes(batch.goals[idx], batch.meta["board_meta_g"][idx])
    om = omega_ids(batch.meta["white_elo"][idx], batch.meta["black_elo"][idx],
                   batch.meta["clock"][idx])
    return (torch.from_numpy(planes_s).to(device),
            torch.from_numpy(om).to(device),
            torch.from_numpy(planes_g).to(device))


def collect_holdout(src: LichessPairSource, n_batches: int, batch_size: int, seed: int):
    """Fixed holdout pair batches (packed+meta kept small; planes built lazily)."""
    out = []
    buf_a, buf_g, buf_ma, buf_mg, buf_om = [], [], [], [], []
    for batch in src.batches(batch_size, seed):
        held = np.flatnonzero((batch.meta["game_id"] % HOLDOUT_MOD) == 0)
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
        logits = (f @ b.T) / fb.tau
        target = torch.arange(len(f), device=device)
        losses.append(float(torch.nn.functional.cross_entropy(logits, target)))
        ranks = (logits >= logits.gather(1, target[:, None])).sum(1)
        top1s.append(float((ranks <= 1).float().mean()))
        top8s.append(float((ranks <= 8).float().mean()))
    fb.train()
    return float(np.mean(losses)), float(np.mean(top1s)), float(np.mean(top8s))


def build_zgoals(shard_dir: Path, fb, device, cap: int = 2048):
    """zMATE_W / zMATE_B = mean B over final CHECKMATE positions of decisive
    games (include_final=True stored them). White-POV: MATE_W = white mated
    black (result +1)."""
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
    zgoals = {}
    with torch.no_grad():
        for res, name in ((1, "MATE_W"), (-1, "MATE_B")):
            rows = finals[res]
            packed = np.stack([r[0] for r in rows]); meta = np.stack([r[1] for r in rows])
            planes = torch.from_numpy(feature_planes(packed, meta)).to(device)
            zgoals[name] = fb.embed_B(planes).mean(dim=0).cpu()
            print(f"zgoal {name}: {len(rows)} checkmate finals")
    return zgoals


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
                reach = (fb.embed_F(planes, om) @ z.to(device)).cpu().numpy()
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
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt", default=None, help="default: data/derived/lichess_fb.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    args = ap.parse_args()

    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    ckpt_path = Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt"
    device = pick_device(args.device)
    print(f"shards={shard_dir.name} device={device} steps={args.steps} batch={args.batch} d={args.d}")

    step = 0
    if ckpt_path.exists() and not args.fresh:
        fb, payload = load_ckpt(ckpt_path, device)
        step = payload["step"]
        print(f"resumed {ckpt_path.name} at step {step}")
    else:
        fb = TorchFB(d=args.d, seed=args.seed)
        fb.to(device)
    opt = torch.optim.AdamW(fb.parameters(), lr=args.lr)
    if ckpt_path.exists() and not args.fresh:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "opt_state" in payload:
            opt.load_state_dict(payload["opt_state"])

    src = LichessPairSource(shard_dir, gamma=args.gamma)
    holdout = collect_holdout(src, n_batches=8, batch_size=args.batch, seed=999)
    print(f"holdout: {len(holdout)} batches of {args.batch}")

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
        loss, top1 = fb.loss_fn(*tensors)
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % 100 == 0:
            rate = 100 / (time.time() - t0); t0 = time.time()
            print(f"step {step}  loss {float(loss):.4f}  train_top1 {float(top1):.3f}  ({rate:.1f} it/s)", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            vloss, vtop1, vtop8 = val_metrics(fb, holdout, device)
            print(f"  VAL step {step}  loss {vloss:.4f}  top1 {vtop1:.3f}  top8 {vtop8:.3f}", flush=True)
            save_ckpt(fb, ckpt_path, step=step, opt=opt)

    zgoals = build_zgoals(shard_dir, fb, device)
    save_ckpt(fb, ckpt_path, step=step, opt=opt, zgoals=zgoals)
    print(f"saved {ckpt_path}")

    vloss, vtop1, vtop8 = val_metrics(fb, holdout, device)
    slope_w, nw = reach_slope(shard_dir, fb, zgoals["MATE_W"], device, want_result=1)
    slope_l, nl = reach_slope(shard_dir, fb, zgoals["MATE_W"], device, want_result=-1)
    print(f"VERDICT VAL_TOP1={vtop1:.3f} VAL_TOP8={vtop8:.3f} (chance {1/args.batch:.4f})")
    print(f"VERDICT REACH_SLOPE_WON={slope_w:.3f} (n={nw}) REACH_SLOPE_LOST={slope_l:.3f} (n={nl})")


if __name__ == "__main__":
    main()
