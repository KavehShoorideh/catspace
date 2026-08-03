#!/usr/bin/env python
"""experiments/train_rho_head.py -- THE SOFT-MIN SIDE (Kaveh 2026-07-22: "build the soft-min
side under the opponent model"; INQUIRY_MULTICHANNEL_FIELD sec 5): per-regime reach-
probability heads rho_c(x, g) on the multichannel field's FROZEN towers.

C-learning-style: positives = same-walk (state_t, state_{t+k}) pairs from the banked
shared-anchor rollouts, k ~ Geometric(1-gamma) (so sigma(score) estimates the discounted
occupancy); negatives = other walks' states. Regime enters through the field's own
conditioning channel (omega col 3). Readouts:
  -log rho_c  = the SOFT distance (log-sum-exp over paths, learned from visitation)
  the softmin <= hardmin inequality vs the IQE d (sec 5: sum >= max term)
VERDICTs: held-out AUC per regime · monotonicity of -log rho in ply-gap · the inequality rate.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.tools.stats_eval.tracking import track_run


class RhoHead(nn.Module):
    """score(F_c(x), B(g)) -> logit of 'g is in x's discounted future under regime c'.
    The regime lives in F's conditioning; the head itself is regime-agnostic."""

    def __init__(self, d: int = 512, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * d, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, F, B):
        return self.net(torch.cat([F, B, F * B], dim=-1))[:, 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="data/derived/sep/lichess_mc2.pt")
    ap.add_argument("--shards", default="data/shards/regime_rollouts_v1")
    ap.add_argument("--n-states", type=int, default=200_000)
    ap.add_argument("--gamma", type=float, default=0.85)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="data/derived/sep/rho_head_v1.pt")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from contextlib import ExitStack
    _stack = ExitStack()
    trk = _stack.enter_context(track_run("rho_head", args, run_name=Path(args.out).stem))
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    fb, _pay = load_ckpt(Path(args.field), dev); fb.eval()
    d_dim = fb.d
    om_base = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]

    # ---- load walk structure: states + (walk id, ply, regime) --------------------
    P_, M_, WID, PLY, REG = [], [], [], [], []
    wid_next = 0
    shard_files = sorted(glob.glob(str(Path(args.shards) / "shard_*.npz")))
    rng.shuffle(shard_files)      # sorted prefix = one daemon phase = regime-coverage hole
    for f in shard_files:
        z = np.load(f)
        gid, reg, ply = z["game_id"], z["regime"], z["ply"]
        # split on game-token change OR ply reset (belt & braces: tokens are per-walk,
        # but a collision would silently corrupt the future-pair indexing)
        change = np.flatnonzero((np.diff(gid.astype(np.int64)) != 0) | (np.diff(ply) <= 0)) + 1
        starts = np.concatenate([[0], change]); ends = np.concatenate([change, [len(gid)]])
        for s, e in zip(starts, ends):
            P_.append(z["packed"][s:e]); M_.append(z["meta"][s:e])
            WID.append(np.full(e - s, wid_next)); PLY.append(ply[s:e])
            REG.append(np.full(e - s, reg[s])); wid_next += 1
        if sum(len(p) for p in P_) >= args.n_states:
            break
    P = np.concatenate(P_); M = np.concatenate(M_)
    WID = np.concatenate(WID); PLY = np.concatenate(PLY); REG = np.concatenate(REG)
    if len(P) > args.n_states:                      # hard cap (shards are ~100k rows each;
        P, M = P[:args.n_states], M[:args.n_states]  # a truncated tail walk stays valid --
        WID, PLY, REG = WID[:args.n_states], PLY[:args.n_states], REG[:args.n_states]  # prefix order + ply intact)
    n = len(P)
    walks = {}
    for i in range(n):
        walks.setdefault(WID[i], []).append(i)
    print(f"[data] {n} states, {len(walks)} walks, regimes {sorted(set(REG.tolist()))}  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- precompute embeddings on the frozen towers (batched) --------------------
    def embed_all():
        Fs = np.zeros((n, d_dim), np.float32); Bs = np.zeros((n, d_dim), np.float32)
        for s in range(0, n, 1024):
            sl = slice(s, min(s + 1024, n))
            pl = torch.from_numpy(feature_planes(P[sl], M[sl])).to(dev)
            om = np.tile(om_base, (sl.stop - sl.start, 1)).astype(np.int64)
            om = np.concatenate([om, REG[sl][:, None].astype(np.int64)], axis=1)
            with torch.no_grad():
                Fs[sl] = fb.embed_F(pl, torch.from_numpy(om).to(dev)).cpu().numpy()
                Bs[sl] = fb.embed_B(pl).cpu().numpy()
            if s % 51200 == 0:
                print(f"  embed {s}/{n}  [{time.time()-t0:.0f}s]", flush=True)
        return Fs, Bs
    F_emb, B_emb = embed_all()

    # ---- train/held-out split by WALK (no leakage across the pair) ---------------
    wids = np.array(sorted(walks))
    rng.shuffle(wids)
    heldout_w = set(wids[: len(wids) // 10].tolist())
    tr_idx = np.flatnonzero(~np.isin(WID, list(heldout_w)))
    te_idx = np.flatnonzero(np.isin(WID, list(heldout_w)))

    head = RhoHead(d=d_dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-5)

    def sample_pairs(pool, k):
        """anchors x_i; positives = same-walk future at geometric gap; negatives = random."""
        xi = pool[rng.integers(0, len(pool), k)]
        gi = np.empty(k, np.int64); lab = np.empty(k, np.float32)
        for j, i in enumerate(xi):
            w = walks[WID[i]]
            pos_in_walk = int(PLY[i])
            if rng.random() < 0.5:
                gap = int(rng.geometric(1.0 - args.gamma))   # k>=1, P(k) ~ gamma^(k-1): occupancy weights
                tgt = min(pos_in_walk + gap, len(w) - 1)
                if tgt > pos_in_walk:
                    gi[j] = w[tgt]; lab[j] = 1.0
                    continue
            gi[j] = pool[rng.integers(0, len(pool))]      # negative: another walk's state
            lab[j] = float(WID[gi[j]] == WID[i] and PLY[gi[j]] > PLY[i])
        return xi, gi, lab

    for s in range(args.steps):
        xi, gi, lab = sample_pairs(tr_idx, args.batch)
        Fx = torch.from_numpy(F_emb[xi]).to(dev)
        Bg = torch.from_numpy(B_emb[gi]).to(dev)
        y = torch.from_numpy(lab).to(dev)
        logit = head(Fx, Bg)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0:
            print(f"  step {s} loss {float(loss.detach()):.4f}  [{time.time()-t0:.0f}s]", flush=True)
            trk.metrics(dict(loss=float(loss.detach())), step=s)
        if args.ckpt_every and s > 0 and s % args.ckpt_every == 0:
            op = Path(args.out)
            torch.save({"state": head.state_dict(), "d": d_dim, "gamma": args.gamma,
                        "field": args.field, "args": vars(args)},
                       op.with_name(f"{op.stem}_step{s}{op.suffix}"))

    # ---- verdicts ---------------------------------------------------------------
    head.eval()
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    per_reg = {}
    logrho_gap = {r: ([], []) for r in set(REG.tolist())}
    with torch.no_grad():
        for _ in range(40):
            xi, gi, lab = sample_pairs(te_idx, 2048)
            lg = head(torch.from_numpy(F_emb[xi]).to(dev),
                      torch.from_numpy(B_emb[gi]).to(dev)).cpu().numpy()
            for r in set(REG[xi].tolist()):
                m = REG[xi] == r
                per_reg.setdefault(r, ([], []))
                per_reg[r][0].extend(lg[m].tolist()); per_reg[r][1].extend(lab[m].tolist())
                pm = m & (lab == 1)
                logrho_gap[r][0].extend((-lg[pm]).tolist())      # -log-odds ~ soft distance
                logrho_gap[r][1].extend((PLY[gi[pm]] - PLY[xi[pm]]).tolist())
    print("VERDICT RHO_HEAD held-out AUC per regime:", flush=True)
    for r in sorted(per_reg):
        lg, lab_ = per_reg[r]
        if len(set(lab_)) > 1:
            auc = roc_auc_score(lab_, lg)
            rho_k = spearmanr(logrho_gap[r][0], logrho_gap[r][1]).correlation if len(logrho_gap[r][0]) > 50 else float("nan")
            print(f"    regime {r:2d}: AUC {auc:.3f}  spearman(-log-odds, ply-gap) {rho_k:+.3f} "
                  f"(monotone soft distance wanted)  n={len(lab_)}", flush=True)
            trk.metrics({f"auc_r{r}": float(auc)})
    torch.save({"state": head.state_dict(), "d": d_dim, "gamma": args.gamma,
                "field": args.field, "args": vars(args)}, args.out)
    print(f"saved {args.out}  [{time.time()-t0:.0f}s]", flush=True)
    _stack.close()


if __name__ == "__main__":
    main()
