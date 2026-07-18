#!/usr/bin/env python
"""
experiments/phead_calibration.py — the phead calibration gate + surface viz
(ARCHITECTURE_REVIEW.md top action item, 2026-07-17).

The committor P feeds coherence, the obvious-region soft-terminal, and (post-
MVP) resign/draw; an overconfident phead poisons all three silently. Two
instruments, both on HELD-OUT games (game_id % 50 == 0, never trained on):

  RELIABILITY / ECE   binned P(win) vs realized outcome frequency. Terminal
                      calibration: among positions with P(win)~0.8, ~80% must
                      actually be won.
  MARTINGALE RESIDUAL the committor is a Doob martingale of the terminal event
                      under the play measure: E[P(s_{t+1}) | s_t] = P(s_t)
                      (tower property). Systematic per-ply drift on held-out
                      play = the phead is NOT a conditional expectation under
                      mu. Doubles as a LEAKAGE detector (adaptedness). And it
                      is necessary-not-sufficient (constants pass), so ECE +
                      sharpness are reported alongside.

Also renders the SURFACES (Kaveh): PCA of F(s) over holdout positions colored
by W/D/L class and by continuous P(win) -- the outcome regions in embedding
space -- plus committor-vs-ply traces for won/lost games (the "touchdown
approach" curves).

Usage:
  .venv/bin/python experiments/phead_calibration.py \
      --ckpt data/derived/sep/cert_base_full.pt \
      --phead data/derived/sep/cert_base_full_phead.pt \
      --shards data/shards/lichess_db_standard_rated_2019-01.prefix4gb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

HOLDOUT_MOD = 50


def load_holdout_games(shard_dir: Path, max_games: int, seed: int):
    """held-out games as ordered position sequences:
    list of dicts(packed, meta, ply, result, white_elo, black_elo, clock)."""
    rng = np.random.default_rng(seed)
    games = []
    paths = sorted(shard_dir.glob("shard_*.npz"))
    rng.shuffle(paths)
    for p in paths:
        if len(games) >= max_games:
            break
        d = np.load(p)
        gid = d["game_id"]
        held = np.flatnonzero(gid % HOLDOUT_MOD == 0)
        if held.size == 0:
            continue
        packed, meta, ply = d["packed"][held], d["meta"][held], d["ply"][held]
        res = d["result"][held]
        we, be, ck = d["white_elo"][held], d["black_elo"][held], d["clock"][held]
        hgid = gid[held]
        change = np.flatnonzero(np.diff(hgid)) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(hgid)]])
        for s, e in zip(starts, ends):
            if e - s < 8:                       # too short to trace
                continue
            order = np.argsort(ply[s:e], kind="stable")
            games.append(dict(packed=packed[s:e][order], meta=meta[s:e][order],
                              ply=ply[s:e][order], result=int(res[s]),
                              white_elo=we[s:e][order], black_elo=be[s:e][order],
                              clock=ck[s:e][order]))
            if len(games) >= max_games:
                break
    return games


@torch.no_grad()
def embed_games(fb, phead, games, device, batch=256):
    """-> per-game arrays of P (n_i, 3) softmax W/D/L, plus flat F embeddings."""
    from catspace.nn.features import feature_planes, omega_ids
    all_p, all_f = [], []
    for g in games:
        om = omega_ids(g["white_elo"], g["black_elo"], g["clock"])
        ps, fs = [], []
        for i in range(0, len(g["packed"]), batch):
            sl = slice(i, i + batch)
            planes = torch.from_numpy(
                feature_planes(g["packed"][sl], g["meta"][sl])).to(device)
            o = torch.from_numpy(om[sl]).to(device)
            f = fb.embed_F(planes, o)
            ps.append(torch.softmax(phead(f), dim=1).cpu().numpy())
            fs.append(f.cpu().numpy())
        all_p.append(np.concatenate(ps))
        all_f.append(np.concatenate(fs))
    return all_p, all_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--phead", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--max-games", type=int, default=400)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-png", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from catspace.nn.eval_head import EvalHead
    from catspace.nn.fb import load_ckpt, pick_device
    dev = pick_device(args.device)
    fb, _ = load_ckpt(Path(args.ckpt), dev)
    fb.eval()
    hp = torch.load(args.phead, map_location=dev, weights_only=False)
    phead = EvalHead(d_in=hp["d_in"]).to(dev)
    phead.load_state_dict(hp["state"]); phead.eval()

    games = load_holdout_games(Path(args.shards), args.max_games, args.seed)
    print(f"holdout games: {len(games)}  positions: {sum(len(g['ply']) for g in games)}")
    all_p, all_f = embed_games(fb, phead, games, dev)

    # ---- reliability / ECE on P(win), per-game bootstrap CI ------------------
    # top bin CLOSED (float32 softmax saturates to exactly 1.0 -- the most
    # overconfident predictions must not fall out of the histogram) and the CI
    # resamples GAMES, since all positions of a game share one Bernoulli
    # outcome (MATH_AUDIT: effective sample size ~ games, not positions).
    pw = np.concatenate([p[:, 0] for p in all_p])
    won = np.concatenate([np.full(len(p), g["result"] == 1)
                          for p, g in zip(all_p, games)]).astype(float)
    gidx = np.concatenate([np.full(len(p), i) for i, p in enumerate(all_p)])
    edges = np.linspace(0, 1, args.bins + 1)

    def ece_of(mask):
        e, rows = 0.0, []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = mask & (pw >= lo) & ((pw < hi) | (hi == edges[-1]))
            if m.sum() == 0:
                continue
            conf, acc = pw[m].mean(), won[m].mean()
            e += (m.sum() / mask.sum()) * abs(conf - acc)
            rows.append((float(lo), float(hi), float(conf), float(acc), int(m.sum())))
        return e, rows

    ece, rel_rows = ece_of(np.ones(len(pw), bool))
    rng = np.random.default_rng(1)
    eboots = []
    for _ in range(min(args.boot, 500)):
        gs = rng.integers(0, len(games), len(games))
        cnt = np.bincount(gs, minlength=len(games)).astype(float)
        # weighted ECE via per-game weights (games resampled with replacement)
        w = cnt[gidx]
        e, tot = 0.0, w.sum()
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pw >= lo) & ((pw < hi) | (hi == edges[-1]))
            wm = w[m].sum()
            if wm == 0:
                continue
            conf = (pw[m] * w[m]).sum() / wm
            acc = (won[m] * w[m]).sum() / wm
            e += (wm / tot) * abs(conf - acc)
        eboots.append(e)
    elo_, ehi_ = np.percentile(eboots, [2.5, 97.5])
    sharp = float(pw.std())

    # ---- martingale residuals, phase-binned (per-ply, not just endpoints) ----
    # mean(diff) telescopes to (P_T - P_0)/(n-1) -- an ENDPOINTS-only test
    # (MATH_AUDIT). Keep it (valid for the endpoint null) AND bin the per-ply
    # residuals by game phase (early/mid/late thirds), per-game means with a
    # bootstrap-over-games CI, so within-game structure is actually tested.
    per_game_drift, phase_drift = [], ([], [], [])
    for p in all_p:
        if len(p) < 2:
            continue
        d = np.diff(p[:, 0])
        per_game_drift.append(float(d.mean()))
        t = np.linspace(0, 1, len(d))
        for k, (lo_t, hi_t) in enumerate(((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.001))):
            m = (t >= lo_t) & (t < hi_t)
            if m.any():
                phase_drift[k].append(float(d[m].mean()))
    per_game_drift = np.array(per_game_drift)
    idx = rng.integers(0, len(per_game_drift), (args.boot, len(per_game_drift)))
    boots = per_game_drift[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    drift = float(per_game_drift.mean())

    print(f"VERDICT PHEAD_ECE={ece:.4f} CI=[{elo_:.4f},{ehi_:.4f}] (bins={args.bins}, "
          f"per-game bootstrap) SHARPNESS={sharp:.3f} n_pos={len(pw)} n_games={len(games)}")
    print(f"VERDICT MARTINGALE_ENDPOINT_DRIFT={drift:+.5f}/ply CI=[{lo:+.5f},{hi:+.5f}] "
          f"(n_games={len(per_game_drift)}; 0 in CI => endpoint-consistent)")
    for k, name in enumerate(("early", "mid", "late")):
        pd_ = np.array(phase_drift[k])
        if len(pd_) == 0:
            continue
        bi = rng.integers(0, len(pd_), (args.boot, len(pd_)))
        bl, bh = np.percentile(pd_[bi].mean(axis=1), [2.5, 97.5])
        print(f"VERDICT MARTINGALE_{name.upper()}_DRIFT={pd_.mean():+.5f}/ply "
              f"CI=[{bl:+.5f},{bh:+.5f}] (n={len(pd_)})")
    for r in rel_rows:
        print(f"  bin [{r[0]:.1f},{r[1]:.1f}) conf={r[2]:.3f} realized={r[3]:.3f} n={r[4]}")

    # ---- surfaces PNG -------------------------------------------------------
    out = args.out_png or f"artifacts/experiments/phead_surfaces_{Path(args.ckpt).stem}.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = np.concatenate(all_f)
    P = np.concatenate(all_p)
    sub = np.random.default_rng(2).permutation(len(F))[:8000]
    Fc = F[sub] - F[sub].mean(0)
    _, _, Vt = np.linalg.svd(Fc, full_matrices=False)
    xy = Fc @ Vt[:2].T
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cls = P[sub].argmax(1)
    for c, color, lab in ((0, "#2166ac", "win"), (1, "#999999", "draw"), (2, "#b2182b", "loss")):
        m = cls == c
        axes[0].scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.4, c=color, label=lab)
    axes[0].legend(markerscale=4); axes[0].set_title("W/D/L regions (argmax class), PCA of F(s)")
    sc = axes[1].scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.5, c=P[sub, 0],
                         cmap="RdYlBu_r", vmin=0, vmax=1)
    plt.colorbar(sc, ax=axes[1], label="P(win)")
    axes[1].set_title("committor field P(win) over the same projection")
    for g, p in zip(games, all_p):
        if len(p) < 8:
            continue
        t = np.linspace(0, 1, len(p))
        col = {1: "#2166ac", 0: "#999999", -1: "#b2182b"}[g["result"]]
        axes[2].plot(t, p[:, 0], color=col, alpha=0.08, lw=0.8)
    axes[2].set_xlabel("normalized ply"); axes[2].set_ylabel("P(win)")
    axes[2].set_title("committor traces (blue=won, red=lost, grey=drawn)")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"SURFACES_PNG={out}")
    Path(out).with_suffix(".json").write_text(json.dumps(
        dict(ece=ece, sharpness=sharp, drift=drift, drift_ci=[float(lo), float(hi)],
             n_pos=int(len(pw)), n_games=len(games), rel_rows=rel_rows)))


if __name__ == "__main__":
    main()
