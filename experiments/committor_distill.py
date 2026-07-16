#!/usr/bin/env python
"""
experiments/committor_distill.py — Stage 1 of the probability-first-class
architecture (Kaveh, 2026-07-15): the mate goal is a SURFACE, not a pole.

Trains a committor head d_W(s) = -ln P(hit the White-mate boundary first) on
top of F, jointly fine-tuning F, replacing BOTH pieces of the old certainty
distill: no goal vector z_W (touchdown semantics -- a rollout that converted
crossed the surface SOMEWHERE, that's all the target encodes), and no
plies/lambda fusion (pure nats; length enters only through what it actually
costs, which the rollout outcomes already contain).

Loss (v2, default --loss nll): Beta(1,1)-smoothed binomial NLL per state --
each row is k wins of n rollouts, so
    loss = -[(k+1) ln P + (n-k+1) ln(1-P)],  P = exp(-d_head)
A proper scoring rule: its optimum IS the true probability, so calibration
comes with rank (the v1 MSE-on--lnP head ranked well but compressed to
[0.19,0.37]); n carries the evidence weight naturally (no sqrt(n) hack);
the pseudo-counts are the epistemic floor on BOTH ends (finite evidence
never certifies P=0 or P=1). --loss mse keeps the v1 objective for
reference. Early stop on held-out NLL (rewards rank AND scale).

Gates printed as VERDICTs:
  1. held-out Spearman(head, t) vs the incumbent POLE-distance baseline
     Spearman(d(F,zW), t) on the same rows (apples-to-apples on the new target)
  2. RIM resolution: same comparison restricted to near-mate rows
     (plies <= --rim-plies) -- the flat-rim failure this reformulation targets.
A standard InfoNCE batch is mixed in EVERY step (global-field protection).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids
from experiments.certainty_distill import spearman_ci


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-in", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--ckpt-out", default="data/derived/sep/committor.pt")
    ap.add_argument("--table", default="artifacts/experiments/certainty_table_r2_K16.json")
    ap.add_argument("--shards", default="data/shards/lichess_db_standard_rated_2019-01.prefix4gb")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cert-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--rim-plies", type=float, default=8.0,
                    help="near-mate subset: rows with observed plies <= this")
    ap.add_argument("--train-min-n", type=int, default=None,
                    help="train only on rows with >= this many visits (holdout "
                         "keeps everything). Cumulative single-root tables grow "
                         "a long noisy tail of n~4-6 deep states whose P-hat is "
                         "mostly sampling noise; distilling on it corrupts "
                         "late-game F regions (root-loop r10 diagnosis)")
    ap.add_argument("--head-init", default=None,
                    help="warm-start the W head (and _dhead sibling if present) "
                         "from an existing *_whead.pt -- continual training of "
                         "the champion's heads instead of fresh random init "
                         "each loop round")
    ap.add_argument("--weight-cap", type=float, default=8.0,
                    help="cap on the sqrt(n) evidence weight. On cumulative "
                         "single-root tables, near-root states reach n in the "
                         "thousands -- uncapped sqrt(n) concentrates the loss "
                         "~100x on the opening shell and warps F exactly where "
                         "every game passes (root-loop r7/r8: field-better "
                         "candidates with play crashed to 0.61/0.125)")
    ap.add_argument("--no-dhead", action="store_true",
                    help="skip the d_D draw head even on v2 tables (attribution: "
                         "isolate table change from joint-head change)")
    ap.add_argument("--loss", choices=("mse", "nll"), default="mse",
                    help="mse = regression on -lnP targets (DEFAULT -- rank +0.603); "
                         "nll = end-to-end smoothed binomial likelihood, FALSIFIED "
                         "2026-07-15 (head collapsed to base rate, rank +0.05; kept "
                         "for reference). Calibrate scale post-hoc with "
                         "committor_recalibrate.py instead (monotone, rank-exact).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    from catspace.data.shards import LichessPairSource
    from catspace.nn.fb import load_ckpt, pick_device, save_ckpt
    from experiments.train_lichess_fb import batch_tensors, build_zgoals

    dev = pick_device(args.device)
    fb, pay = load_ckpt(Path(args.ckpt_in), dev)
    zW = pay["zgoals"]["MATE_W"]
    zW_t = (zW.to(dev).float() if torch.is_tensor(zW)
            else torch.as_tensor(np.asarray(zW), dtype=torch.float32, device=dev))

    rows = json.loads(Path(args.table).read_text())["rows"]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    n_hold = int(len(rows) * args.holdout_frac)
    hold, train = [rows[i] for i in order[:n_hold]], [rows[i] for i in order[n_hold:]]
    if args.train_min_n:
        before = len(train)
        train = [r for r in train if r["n"] >= args.train_min_n]
        print(f"train-min-n {args.train_min_n}: {before} -> {len(train)} train rows")
    print(f"{len(train)} train / {len(hold)} held-out states ({args.table})")

    def target(r):
        return -np.log(max(r["p_hat"], 1.0 / (r["n"] + 2)))

    # per-boundary DRAW committor (v2 tables with `outcomes` only): the draw
    # surfaces are "out of bounds" a losing player steers TOWARD and a winning
    # player needs clearance from (Kaveh 2026-07-15)
    has_outcomes = all("outcomes" in r for r in rows[:20]) and not args.no_dhead

    def target_draw(r):
        n_draw = sum(v for k, v in r["outcomes"].items() if k.startswith("DRAW"))
        return -np.log(max(n_draw / r["n"], 1.0 / (r["n"] + 2)))

    def encode(rs):
        boards = [chess.Board(r["fen"]) for r in rs]
        packed = np.stack([encode_packed(b) for b in boards])
        meta = np.stack([encode_meta(b) for b in boards])
        om = omega_ids(np.full(len(rs), 1800), np.full(len(rs), 1800),
                       np.full(len(rs), np.nan))
        return (torch.from_numpy(feature_planes(packed, meta)).to(dev),
                torch.from_numpy(om).to(dev))

    d_in = zW_t.shape[-1]

    def make_head():
        return torch.nn.Sequential(torch.nn.Linear(d_in, 128), torch.nn.ReLU(),
                                   torch.nn.Linear(128, 1), torch.nn.Softplus()).to(dev)

    head = make_head()
    dhead = make_head() if has_outcomes else None
    if args.head_init:
        wp = torch.load(args.head_init, map_location=dev, weights_only=False)
        head.load_state_dict(wp["state"])
        print(f"W head warm-started from {args.head_init}")
        dinit = Path(args.head_init.replace("_whead", "_dhead"))
        if dhead is not None and dinit.exists():
            dp = torch.load(dinit, map_location=dev, weights_only=False)
            dhead.load_state_dict(dp["state"])
            print(f"D head warm-started from {dinit}")
    if dhead is not None:
        td_tr = torch.tensor([target_draw(r) for r in train], dtype=torch.float32, device=dev)
        td_ho = np.array([target_draw(r) for r in hold])
        print("v2 table (per-boundary outcomes): training d_D draw-committor head too")

    t_tr = torch.tensor([target(r) for r in train], dtype=torch.float32, device=dev)
    w_tr = torch.tensor([min(np.sqrt(r["n"]), args.weight_cap) for r in train],
                        dtype=torch.float32, device=dev)
    w_tr = w_tr / w_tr.mean()
    # binomial evidence (v2 NLL loss): k wins of n rollouts per state
    n_tr = torch.tensor([r["n"] for r in train], dtype=torch.float32, device=dev)
    k_tr = torch.tensor([r["p_hat"] * r["n"] for r in train], dtype=torch.float32, device=dev)
    if dhead is not None:
        kd_tr = torch.tensor([sum(v for c, v in r["outcomes"].items() if c.startswith("DRAW"))
                              for r in train], dtype=torch.float32, device=dev)
    k_ho = np.array([r["p_hat"] * r["n"] for r in hold])
    n_ho = np.array([r["n"] for r in hold])

    def nll(d, k, n):
        """Beta(1,1)-smoothed binomial NLL of P = exp(-d), normalized per unit
        of evidence (sum/sum keeps between-state n-weighting while holding the
        overall scale ~E[d], so the NCE mixing term keeps its protective
        weight). ln(1-P) = log1p(-exp(-d)); d floored for numerics."""
        d = d.clamp(min=1e-4)
        per = (k + 1) * d - (n - k + 1) * torch.log1p(-torch.exp(-d))
        return per.sum() / (n + 2).sum()

    def nll_np(d, k, n):
        d = np.maximum(d, 1e-4)
        per = (k + 1) * d - (n - k + 1) * np.log1p(-np.exp(-d))
        return float(per.sum() / (n + 2).sum())

    pl_tr, om_tr = encode(train)
    pl_ho, om_ho = encode(hold)
    t_ho = np.array([target(r) for r in hold])
    rim_ho = np.array([r["plies"] is not None and r["plies"] <= args.rim_plies
                       for r in hold])
    print(f"rim subset (plies<={args.rim_plies:g}): {int(rim_ho.sum())} held-out rows")

    def readouts():
        fb.eval(); head.eval()
        with torch.no_grad():
            f = fb.embed_F(pl_ho, om_ho)
            d_head = head(f).squeeze(-1).cpu().numpy()
            d_pole = fb.distance_matrix(f, zW_t[None, :])[:, 0].cpu().numpy()
        fb.train(); head.train()
        return d_head, d_pole

    d_head0, d_pole0 = readouts()
    b0 = spearman_ci(d_pole0, t_ho)
    print(f"POLE baseline held-out Spearman(d(F,zW), -lnP) = {b0[0]:+.3f} "
          f"CI[{b0[1]:+.3f},{b0[2]:+.3f}] (n={len(hold)})")

    src = LichessPairSource(Path(args.shards), gamma=0.95)
    it = iter(src.batches(args.batch, seed=args.seed))
    params = list(fb.parameters()) + list(head.parameters())
    if dhead is not None:
        params += list(dhead.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    fb.train(); head.train()
    ntr = len(train)
    best_r, best_state, stale = -np.inf, None, 0
    for step in range(1, args.steps + 1):
        idx = torch.from_numpy(rng.integers(0, ntr, size=args.batch)).to(dev)
        f = fb.embed_F(pl_tr[idx], om_tr[idx])
        pred = head(f).squeeze(-1)
        if args.loss == "nll":
            cert = nll(pred, k_tr[idx], n_tr[idx])
            if dhead is not None:
                cert = cert + nll(dhead(f).squeeze(-1), kd_tr[idx], n_tr[idx])
        else:
            cert = (w_tr[idx] * (pred - t_tr[idx]) ** 2).mean()
            if dhead is not None:
                pred_d = dhead(f).squeeze(-1)
                cert = cert + (w_tr[idx] * (pred_d - td_tr[idx]) ** 2).mean()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(src.batches(args.batch, seed=args.seed + step)); batch = next(it)
        tens = batch_tensors(batch, dev)
        if tens is None:
            continue
        nce, _ = fb.loss_fn(*tens[:5], ply_gap_weight=0.05)
        loss = args.cert_weight * cert + nce
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0:
            d_head, _ = readouts()
            rho_now, _, _ = spearman_ci(d_head, t_ho)
            if args.loss == "nll":
                r = -nll_np(d_head, k_ho, n_ho)      # maximize -NLL (proper score)
                print(f"step {step}  cert {float(cert):.4f}  nce {float(nce):.3f}  "
                      f"held-out NLL {-r:.4f}  rho {rho_now:+.3f}", flush=True)
            else:
                r = rho_now
                print(f"step {step}  cert {float(cert):.4f}  nce {float(nce):.3f}  "
                      f"held-out rho {rho_now:+.3f}", flush=True)
            if r > best_r:
                best_r, stale = r, 0
                best_state = ({k: v.detach().cpu().clone() for k, v in fb.state_dict().items()},
                              {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                              ({k: v.detach().cpu().clone() for k, v in dhead.state_dict().items()}
                               if dhead is not None else None))
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early stop at step {step} (best rho {best_r:+.3f})")
                    break
    if best_state is not None:
        fb.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
        if dhead is not None and best_state[2] is not None:
            dhead.load_state_dict(best_state[2])

    d_head, d_pole = readouts()
    h1 = spearman_ci(d_head, t_ho)
    print(f"VERDICT COMMITTOR_SPEARMAN pole-baseline {b0[0]:+.3f}[{b0[1]:+.3f},{b0[2]:+.3f}] "
          f"-> head {h1[0]:+.3f}[{h1[1]:+.3f},{h1[2]:+.3f}] (n={len(hold)})")
    # calibration verdict (the v1 failure mode: rank without scale)
    P_pred = np.exp(-np.maximum(d_head, 1e-4))
    p_emp_ho = k_ho / n_ho
    order = np.argsort(P_pred)
    bins = np.array_split(order, 10)
    ece = float(np.mean([abs(P_pred[b].mean() - p_emp_ho[b].mean()) for b in bins if len(b)]))
    print(f"VERDICT COMMITTOR_CALIBRATION span [{P_pred.min():.2f},{P_pred.max():.2f}] "
          f"(empirical [0,1])  ECE(10 bins) {ece:.3f}  held-out NLL {nll_np(d_head, k_ho, n_ho):.4f}")
    if rim_ho.sum() >= 30:
        rb = spearman_ci(d_pole[rim_ho], t_ho[rim_ho])
        rh = spearman_ci(d_head[rim_ho], t_ho[rim_ho])
        print(f"VERDICT RIM_RESOLUTION (plies<={args.rim_plies:g}, n={int(rim_ho.sum())}) "
              f"pole {rb[0]:+.3f}[{rb[1]:+.3f},{rb[2]:+.3f}] -> "
              f"head {rh[0]:+.3f}[{rh[1]:+.3f},{rh[2]:+.3f}]")
    else:
        print(f"RIM_RESOLUTION skipped (only {int(rim_ho.sum())} rim rows held out)")

    if dhead is not None:
        with torch.no_grad():
            fb.eval()
            f_ho = fb.embed_F(pl_ho, om_ho)
            dd = dhead(f_ho).squeeze(-1).cpu().numpy()
        rd = spearman_ci(dd, td_ho)
        print(f"VERDICT DRAW_COMMITTOR_SPEARMAN head {rd[0]:+.3f}[{rd[1]:+.3f},{rd[2]:+.3f}] "
              f"(n={len(hold)})")

    zg = build_zgoals(Path(args.shards), fb, dev)
    zg["MATE_W"] = zW_t.detach().cpu()   # pole kept for reference readouts only
    save_ckpt(fb, Path(args.ckpt_out), step=pay.get("step", 0), zgoals=zg)
    hp = Path(args.ckpt_out).with_name(Path(args.ckpt_out).stem + "_whead.pt")
    torch.save({"state": head.state_dict(), "d_in": d_in}, hp)
    print(f"saved {args.ckpt_out} + {hp}")
    if dhead is not None:
        dp = Path(args.ckpt_out).with_name(Path(args.ckpt_out).stem + "_dhead.pt")
        torch.save({"state": dhead.state_dict(), "d_in": d_in}, dp)
        print(f"saved {dp}")


if __name__ == "__main__":
    main()
