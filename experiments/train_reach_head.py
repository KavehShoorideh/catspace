#!/usr/bin/env python
"""experiments/train_reach_head.py -- v1 z-conditioned first-hit reachability head
(REACHABILITY_FOUNDATIONS §4.1; the two-evaluator measure field).

    P(first-reach g within game | s, z_self, elos) = sigma( <phi_r(s,z,elo), psi_r(g)> / sqrt(k) + b )
    E[log1p plies | hit]                          =        <phi_t(...),    psi_t(g)> / sqrt(k) + b_t

Retrieval-factored: goal tower is z-FREE (bank embeds once; queries sweep it by dot product).
z = FROZEN M2b per-player delta (16-d), pidx=-1 (unseen player) -> z=0 population fallback.
Losses imported from experiments/losses.py (first_hit_bce, censored_plies_loss -- tested).
Data: experiments/build_reach_data.py output (materialized labels; in_now pairs excluded).

PRE-REGISTERED SPLITS: eval = split 'train' & game_id%10==0 (seen players, UNSEEN games; z-lift
measured here); trainset = split 'train' & game_id%10!=0; unseen = split 'heldout' (45 players
w/o z -> z=0 fallback; BCE+calibration only, no z-lift possible).

PRE-REGISTERED ACCEPTANCE (printed as VERDICT lines; the smoke must show the SIGN):
  1. z-lift nats/pair on eval: BCE(z=0) - BCE(full) and BCE(z permuted within ±100 Elo) - BCE(full),
     paired bootstrap CI clustered by player (catspace.stats.paired_nll_ci).
  2. Calibration: 10-bin reliability (pred vs realized hit freq) on eval + unseen; max |gap|.
  3. Plies head: MAE(log1p) on hit pairs vs global-train-median baseline.
  4. eff_rank of state hit-embeddings (collapse gate), 3 bootstrap draws.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.diagnostics import eff_rank                              # noqa: E402
from catspace.stats import paired_nll_ci                               # noqa: E402
from catspace.train.scaffold import (TrainConfig, resolve_device,      # noqa: E402
                                     save_torch_ckpt, standard_train)
from experiments.losses import censored_plies_loss, first_hit_bce      # noqa: E402


class ReachHead(nn.Module):
    """Shared-trunk two-tower head; separate output projections per quantity (STANDARDS 10)."""

    def __init__(self, d_phi=64, d_z=16, d_emb=64, width=128):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(d_phi + d_z + 2, width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())
        self.goal = nn.Sequential(nn.Linear(d_phi, width), nn.ReLU(),
                                  nn.Linear(width, width), nn.ReLU())
        self.s_hit = nn.Linear(width, d_emb); self.s_time = nn.Linear(width, d_emb)
        self.g_hit = nn.Linear(width, d_emb); self.g_time = nn.Linear(width, d_emb)
        self.b_hit = nn.Parameter(torch.tensor(0.0)); self.b_time = nn.Parameter(torch.tensor(0.0))
        self.scale = d_emb ** -0.5

    def state_embs(self, phi, z, elos):
        h = self.state(torch.cat([phi, z, elos], -1))
        return self.s_hit(h), self.s_time(h)

    def goal_embs(self, bank):
        g = self.goal(bank)
        return self.g_hit(g), self.g_time(g)

    def forward(self, phi, z, elos, bank):
        sh, st = self.state_embs(phi, z, elos)
        gh, gt = self.goal_embs(bank)
        logit = sh @ gh.T * self.scale + self.b_hit          # (B,G)
        plies_log = st @ gt.T * self.scale + self.b_time     # (B,G)
        return logit, plies_log


def reliability(p, y, bins=10):
    """(pred_mean, realized, count) per equal-width probability bin."""
    edges = np.linspace(0, 1, bins + 1); out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum() > 0:
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


def band_permute(pidx, elo, rng):
    """Wrong-z placebo: remap each PLAYER to a different player's z within ±100 Elo (roll by 1
    in Elo order -> nearest-rating neighbor, never identity when >1 player in band)."""
    ply = np.unique(pidx)
    pelo = np.array([elo[pidx == p].mean() for p in ply])
    order = np.argsort(pelo)
    rolled = np.roll(ply[order], 1)
    mapping = dict(zip(ply[order].tolist(), rolled.tolist()))
    return np.array([mapping[p] for p in pidx], dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/derived/reach/reach_v1.npz")
    ap.add_argument("--zckpt", default="artifacts/experiments/m2b_style_3k.pt")
    ap.add_argument("--out", default="artifacts/experiments/reach_v1")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam-time", type=float, default=1.0)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    dev = resolve_device(args.device)

    d = dict(np.load(args.data, allow_pickle=True))
    zt = torch.load(args.zckpt, map_location="cpu", weights_only=False)
    ztab = zt["state_dict"]["delta.weight"].float()                     # (2975,16) FROZEN
    ztab = torch.cat([ztab, torch.zeros(1, ztab.shape[1])])             # slot -1 -> zeros
    N, G = d["hit"].shape
    split = d["split"].astype(str); gid = d["game_id"]
    is_train = (split == "train") & (gid % 10 != 0)
    is_eval = (split == "train") & (gid % 10 == 0)
    is_unseen = split == "heldout"
    print(f"rows: train {is_train.sum()} | eval(seen-player unseen-game) {is_eval.sum()} | "
          f"unseen-player {is_unseen.sum()} | goals {G}")

    t = lambda x, ty=torch.float32: torch.as_tensor(x, dtype=ty, device=dev)
    phi, bank = t(d["phi"]), t(d["bank"])
    hit, plies, in_now = t(d["hit"]), t(d["plies"]), t(d["in_now"])
    elos = torch.stack([(t(d["elo_self"]) - 1500) / 400, (t(d["elo_oppo"]) - 1500) / 400], -1)
    pidx = d["pidx"].astype(np.int64)                                  # -1 -> last (zero) slot
    zvec = ztab.to(dev)[torch.as_tensor(np.where(pidx < 0, ztab.shape[0] - 1, pidx), device=dev)]

    model = ReachHead(d_phi=phi.shape[1], d_z=ztab.shape[1]).to(dev)
    base = float(hit[t(is_train, torch.bool)][in_now[t(is_train, torch.bool)] == 0].mean())
    with torch.no_grad():
        model.b_hit.fill_(float(np.log(base / (1 - base))))            # base-rate init
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tr_idx = np.flatnonzero(is_train)

    def batch_loss(rows):
        r = torch.as_tensor(rows, device=dev)
        logit, ptime = model(phi[r], zvec[r], elos[r], bank)
        mask = in_now[r] == 0
        l_hit = first_hit_bce(logit[mask], hit[r][mask])
        l_t = censored_plies_loss(ptime[mask], plies[r][mask], hit[r][mask])
        return l_hit, l_t

    def step_fn(model, step):
        model.train(); opt.zero_grad()
        l_hit, l_t = batch_loss(rng.choice(tr_idx, args.batch, replace=False))
        (l_hit + args.lam_time * l_t).backward(); opt.step()
        return {"loss_hit": l_hit.item(), "loss_time": l_t.item()}

    ev_probe = rng.choice(np.flatnonzero(is_eval), min(2048, int(is_eval.sum())), replace=False)

    def gates_fn(model):
        model.eval()
        with torch.no_grad():
            l_hit, l_t = batch_loss(ev_probe)
            sh, _ = model.state_embs(phi[torch.as_tensor(ev_probe, device=dev)],
                                     zvec[torch.as_tensor(ev_probe, device=dev)],
                                     elos[torch.as_tensor(ev_probe, device=dev)])
        return {"eval_hit": l_hit.item(), "eval_time": l_t.item(),
                "eff_rank": eff_rank(sh.cpu().numpy())}

    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      eval_every=args.eval_every, experiment="reach_v1",
                      run_name=Path(args.out).name, device=str(dev))
    standard_train(step_fn, model, cfg, args=args, gates_fn=gates_fn)

    # ---------------- pre-registered acceptance instrument ----------------
    model.eval()

    def per_pair(rows, z):
        """per-pair NLL + preds on non-in_now pairs, chunked."""
        nll, ps, ys, pls, plt, own = [], [], [], [], [], []
        for i in range(0, len(rows), 1024):
            r = torch.as_tensor(rows[i:i + 1024], device=dev)
            logit, ptime = model(phi[r], z[r], elos[r], bank)
            m = in_now[r] == 0
            y = hit[r][m]; lg = logit[m]
            nll.append(torch.nn.functional.binary_cross_entropy_with_logits(
                lg, y.float(), reduction="none").cpu().numpy())
            ps.append(torch.sigmoid(lg).cpu().numpy()); ys.append(y.cpu().numpy())
            hm = m & (hit[r] == 1)
            pls.append(ptime[hm].cpu().numpy()); plt.append(plies[r][hm].cpu().numpy())
            own.append(np.repeat(d["player_id"][rows[i:i + 1024]], m.sum(1).cpu().numpy()))
        return (np.concatenate(nll), np.concatenate(ps), np.concatenate(ys),
                np.concatenate(pls), np.concatenate(plt), np.concatenate(own))

    ev = np.flatnonzero(is_eval)
    with torch.no_grad():
        n_full, p_full, y_ev, tp, tt, players = per_pair(ev, zvec)
        z0 = torch.zeros_like(zvec)
        n_z0, *_ = per_pair(ev, z0)
        perm = band_permute(pidx[ev], d["elo_self"][ev], rng)
        zperm = zvec.clone(); zperm[torch.as_tensor(ev, device=dev)] = \
            ztab.to(dev)[torch.as_tensor(perm, device=dev)]
        n_zp, *_ = per_pair(ev, zperm)

    lift0 = paired_nll_ci(n_full, n_z0, clusters=players)
    liftp = paired_nll_ci(n_full, n_zp, clusters=players)
    print(f"VERDICT z-lift vs z=0     : {lift0[0]*1e3:+.3f} mnats/pair  CI[{lift0[1]*1e3:+.3f},"
          f"{lift0[2]*1e3:+.3f}]  p(better)={lift0[3]:.3f}   (eval: seen players, unseen games)")
    print(f"VERDICT z-lift vs wrong-z : {liftp[0]*1e3:+.3f} mnats/pair  CI[{liftp[1]*1e3:+.3f},"
          f"{liftp[2]*1e3:+.3f}]  p(better)={liftp[3]:.3f}   (±100-Elo band permutation)")

    for name, rows in (("eval", ev), ("unseen", np.flatnonzero(is_unseen))):
        with torch.no_grad():
            _, p, y, tp2, tt2, _ = per_pair(rows, zvec)
        rel = reliability(p, y)
        gap = max(abs(a - b) for a, b, _ in rel)
        print(f"VERDICT calibration [{name}]: max|pred-realized| {gap:.4f} over {len(rel)} bins | "
              f"mean pred {p.mean():.4f} vs realized {y.mean():.4f}")
        for pm, rz, nb in rel:
            print(f"    bin pred {pm:.3f} realized {rz:.3f} n {nb}")
        med = float(np.median(np.log1p(np.maximum(tt if name == 'eval' else tt2, 0))))
        mae = float(np.mean(np.abs((tp if name == 'eval' else tp2)
                                   - np.log1p(np.maximum(tt if name == 'eval' else tt2, 0)))))
        mae_base = float(np.mean(np.abs(med - np.log1p(np.maximum(
            tt if name == 'eval' else tt2, 0)))))
        print(f"VERDICT plies [{name}]: MAE(log1p) {mae:.4f} vs median-baseline {mae_base:.4f}")

    ranks = []
    for s in range(3):
        rr = np.random.default_rng(s).choice(ev, min(2048, len(ev)), replace=False)
        with torch.no_grad():
            sh, _ = model.state_embs(phi[torch.as_tensor(rr, device=dev)],
                                     zvec[torch.as_tensor(rr, device=dev)],
                                     elos[torch.as_tensor(rr, device=dev)])
        ranks.append(eff_rank(sh.cpu().numpy()))
    print(f"VERDICT eff_rank(state hit-emb, d=64): {np.mean(ranks):.1f} "
          f"[{min(ranks):.1f},{max(ranks):.1f}] over 3 bootstrap draws")
    save_torch_ckpt(model, args.out + "_final", args.steps, args=args)


if __name__ == "__main__":
    main()
