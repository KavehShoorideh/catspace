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

v1b (--zopp): adds the CAUSAL opponent-style slot -- z_opp_t (16) + n_obs (1, log-normalized)
from build_zopp_causal.py (train-time == play-time conditioning; z=0 cold start). Placebo for the
slot: permute z_opp among eval rows within (elo_oppo ±100 band x n_obs bucket) -- rating and
observation count can't leak through it.

PRE-REGISTERED ACCEPTANCE (printed as VERDICT lines; the smoke must show the SIGN):
  1. z-lift nats/pair on eval: BCE(z=0) - BCE(full) and BCE(z permuted within ±100 Elo) - BCE(full),
     paired bootstrap CI clustered by player (catspace.research.tools.stats_eval.stats.paired_nll_ci).
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
from catspace.research.tools.chess_specific.diagnostics import eff_rank                              # noqa: E402
from catspace.research.tools.stats_eval.stats import paired_nll_ci                               # noqa: E402
from catspace.research.tools.training_infra.train.scaffold import (TrainConfig, resolve_device,      # noqa: E402
                                     save_torch_ckpt, standard_train)
from experiments.losses import censored_plies_loss, first_hit_bce      # noqa: E402


from catspace.research.components.planner.approaches.reach_field.src.head import ReachHead  # component home (refactor 2026-07-30)


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
    ap.add_argument("--zopp", default="", help="zopp_causal_v1.npz path; empty = v1 (no opponent slot)")
    ap.add_argument("--out", default="artifacts/experiments/reach_v1")
    ap.add_argument("--init", default="", help="warm-start ckpt (fine-tune mode)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam-time", type=float, default=1.0)
    ap.add_argument("--elo-dropout", type=float, default=0.1,
                    help="per-side train-time Elo masking prob (teaches the full-pop fallback)")
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
    hit = torch.as_tensor(d["hit"], device=dev)                    # uint8 (native -- v2 is 4M rows)
    plies = torch.as_tensor(d["plies"], device=dev)                # int16
    in_now = torch.as_tensor(d["in_now"], device=dev)              # uint8
    # Elo block: [elo_self_n, elo_oppo_n, known_self, known_oppo]. Data always has Elo (flags=1);
    # train-time MASKING (below) teaches the full-population fallback (Kaveh 2026-07-28: absent z
    # -> Elo population; absent/provisional Elo -> full population).
    elos = torch.stack([(t(d["elo_self"]) - 1500) / 400, (t(d["elo_oppo"]) - 1500) / 400,
                        torch.ones(N, device=dev), torch.ones(N, device=dev)], -1)
    pidx = d["pidx"].astype(np.int64)                                  # -1 -> last (zero) slot
    zvec = ztab.to(dev)[torch.as_tensor(np.where(pidx < 0, ztab.shape[0] - 1, pidx), device=dev)]

    if args.zopp:
        zo = dict(np.load(args.zopp, allow_pickle=True))
        assert (zo["game_id"] == d["game_id"]).all() and (zo["ply"] == d["ply"]).all(), \
            "zopp file not row-aligned with reach data"
        nobs_norm = np.log1p(zo["n_obs"].astype(np.float32)) / np.log1p(64.0)
        zopp = torch.cat([t(zo["z_opp_t"]), t(nobs_norm).unsqueeze(1)], 1)     # (N,17)
        d_opp = zopp.shape[1]
    else:
        zopp, d_opp = None, 0
    model = ReachHead(d_phi=phi.shape[1], d_z=ztab.shape[1], d_opp=d_opp).to(dev)
    if args.init:
        # FINE-TUNE (agentive labels, Kaveh 2026-07-30): warm-start from a trained
        # field; keep its calibration, let the agentive data move it.
        ck = torch.load(args.init, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print(f"[init] warm-started from {args.init}")
    else:
        base = float(hit[t(is_train, torch.bool)][in_now[t(is_train, torch.bool)] == 0]
                     .float().mean())
        with torch.no_grad():
            model.b_hit.fill_(float(np.log(base / (1 - base))))        # base-rate init
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tr_idx = np.flatnonzero(is_train)

    def ctx(r):
        return torch.cat([elos[r], zopp[r]], 1) if zopp is not None else elos[r]

    def batch_loss(rows, train=False):
        r = torch.as_tensor(rows, device=dev)
        logit, ptime = model(phi[r], zvec[r], train_ctx(r) if train else ctx(r), bank)
        mask = in_now[r] == 0
        l_hit = first_hit_bce(logit[mask], hit[r][mask])
        l_t = censored_plies_loss(ptime[mask], plies[r][mask], hit[r][mask])
        return l_hit, l_t

    def train_ctx(r):
        c = ctx(r).clone()
        for side in (0, 1):                                        # mask elo value + its flag
            m = torch.rand(len(r), device=dev) < args.elo_dropout
            c[m, side] = 0.0; c[m, 2 + side] = 0.0
        return c

    def step_fn(model, step):
        model.train(); opt.zero_grad()
        l_hit, l_t = batch_loss(rng.choice(tr_idx, args.batch, replace=False), train=True)
        (l_hit + args.lam_time * l_t).backward(); opt.step()
        return {"loss_hit": l_hit.item(), "loss_time": l_t.item()}

    ev_probe = rng.choice(np.flatnonzero(is_eval), min(2048, int(is_eval.sum())), replace=False)

    def gates_fn(model):
        model.eval()
        with torch.no_grad():
            l_hit, l_t = batch_loss(ev_probe)
            rp = torch.as_tensor(ev_probe, device=dev)
            sh, _ = model.state_embs(phi[rp], zvec[rp], ctx(rp))
        return {"eval_hit": l_hit.item(), "eval_time": l_t.item(),
                "eff_rank": eff_rank(sh.cpu().numpy())}

    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      eval_every=args.eval_every, experiment="reach_v1",
                      run_name=Path(args.out).name, device=str(dev))
    standard_train(step_fn, model, cfg, args=args, gates_fn=gates_fn)

    # ---------------- pre-registered acceptance instrument ----------------
    model.eval()

    def per_pair(rows, z, zo_override=None):
        """per-pair NLL + preds on non-in_now pairs, chunked. zo_override: (N,17) replacement
        for the opponent slot (v1b ablations/placebos)."""
        nll, ps, ys, pls, plt, own = [], [], [], [], [], []
        for i in range(0, len(rows), 1024):
            r = torch.as_tensor(rows[i:i + 1024], device=dev)
            if zopp is None:
                cx = elos[r]
            else:
                zz = zopp if zo_override is None else zo_override
                cx = torch.cat([elos[r], zz[r]], 1)
            logit, ptime = model(phi[r], z[r], cx, bank)
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

    if zopp is not None:
        with torch.no_grad():
            n_zo0, *_ = per_pair(ev, zvec, zo_override=torch.cat(
                [torch.zeros_like(zopp[:, :16]), zopp[:, 16:]], 1))   # kill style, keep n_obs
            # placebo: permute z_opp among eval rows within (elo_oppo band x n_obs bucket)
            zperm2 = zopp.clone()
            eo = d["elo_oppo"][ev]; nb = d["ply"][ev] * 0  # placeholder init
            nobs_ev = zo["n_obs"][ev]
            bucket = np.digitize(nobs_ev, [1, 5, 10, 20, 40])
            band = (eo // 100).astype(int)
            rng2 = np.random.default_rng(1)
            for key in set(zip(band.tolist(), bucket.tolist())):
                grp = np.flatnonzero((band == key[0]) & (bucket == key[1]))
                if len(grp) > 1:
                    zperm2[torch.as_tensor(ev[grp], device=dev)] = \
                        zperm2[torch.as_tensor(ev[rng2.permutation(grp)], device=dev)]
            n_zop, *_ = per_pair(ev, zvec, zo_override=zperm2)
        lo0 = paired_nll_ci(n_full, n_zo0, clusters=players)
        lop = paired_nll_ci(n_full, n_zop, clusters=players)
        print(f"VERDICT z_opp-lift vs z_opp=0 : {lo0[0]*1e3:+.3f} mnats/pair  CI[{lo0[1]*1e3:+.3f},"
              f"{lo0[2]*1e3:+.3f}]  p(better)={lo0[3]:.3f}   (causal opponent style, n_obs kept)")
        print(f"VERDICT z_opp wrong-opp placebo: {lop[0]*1e3:+.3f} mnats/pair  CI[{lop[1]*1e3:+.3f},"
              f"{lop[2]*1e3:+.3f}]  p(better)={lop[3]:.3f}   (permuted within Elo-band x n_obs bucket)")
        # STRATIFIED z_opp readout (pre-registered): the style effect can only live where the
        # causal estimate has information -- M2c break-even ~10 (identity) / ~40 (beats prior).
        # Unstratified nulls over the 93% low-info rows are uninformative.
        pair_nobs = np.repeat(nobs_ev, (in_now[torch.as_tensor(ev, device=dev)] == 0)
                              .sum(1).cpu().numpy())
        for thr in (10, 40):
            m = pair_nobs >= thr
            if m.sum() > 1000:
                ls = paired_nll_ci(n_full[m], n_zo0[m], clusters=players[m])
                print(f"VERDICT z_opp-lift@n_obs>={thr}: {ls[0]*1e3:+.3f} mnats/pair  "
                      f"CI[{ls[1]*1e3:+.3f},{ls[2]*1e3:+.3f}]  p(better)={ls[3]:.3f}  "
                      f"({m.sum():,} pairs, {100*m.mean():.0f}% of eval)")

    for name, rows in (("eval", ev), ("unseen", np.flatnonzero(is_unseen))):
        if len(rows) == 0:                       # agentive data has no heldout split
            print(f"VERDICT calibration [{name}]: skipped (0 rows)")
            continue
        with torch.no_grad():
            _, p, y, tp2, tt2, _ = per_pair(rows, zvec)
        rel = reliability(p, y)
        gap = max(abs(a - b) for a, b, _ in rel)
        tot = sum(nb for _, _, nb in rel)
        ece = sum(nb * abs(a - b) for a, b, nb in rel) / tot
        print(f"VERDICT calibration [{name}]: ECE {ece:.5f} (count-weighted) | max|gap| {gap:.4f} "
              f"over {len(rel)} bins | mean pred {p.mean():.4f} vs realized {y.mean():.4f}")
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
            rt = torch.as_tensor(rr, device=dev)
            sh, _ = model.state_embs(phi[rt], zvec[rt], ctx(rt))
        ranks.append(eff_rank(sh.cpu().numpy()))
    print(f"VERDICT eff_rank(state hit-emb, d=64): {np.mean(ranks):.1f} "
          f"[{min(ranks):.1f},{max(ranks):.1f}] over 3 bootstrap draws")
    save_torch_ckpt(model, args.out + "_final", args.steps, args=args)


if __name__ == "__main__":
    main()
