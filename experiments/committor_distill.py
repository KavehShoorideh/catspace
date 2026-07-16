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

Target per table row: t(s) = -ln max(p_hat, 1/(n+2)) -- the Laplace floor is
the epistemic hazard (finite evidence can never certify P=1 or P=0).

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
    print(f"{len(train)} train / {len(hold)} held-out states ({args.table})")

    def target(r):
        return -np.log(max(r["p_hat"], 1.0 / (r["n"] + 2)))

    def encode(rs):
        boards = [chess.Board(r["fen"]) for r in rs]
        packed = np.stack([encode_packed(b) for b in boards])
        meta = np.stack([encode_meta(b) for b in boards])
        om = omega_ids(np.full(len(rs), 1800), np.full(len(rs), 1800),
                       np.full(len(rs), np.nan))
        return (torch.from_numpy(feature_planes(packed, meta)).to(dev),
                torch.from_numpy(om).to(dev))

    d_in = zW_t.shape[-1]
    head = torch.nn.Sequential(torch.nn.Linear(d_in, 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, 1), torch.nn.Softplus()).to(dev)

    t_tr = torch.tensor([target(r) for r in train], dtype=torch.float32, device=dev)
    w_tr = torch.tensor([np.sqrt(r["n"]) for r in train], dtype=torch.float32, device=dev)
    w_tr = w_tr / w_tr.mean()
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
    opt = torch.optim.AdamW(list(fb.parameters()) + list(head.parameters()), lr=args.lr)
    fb.train(); head.train()
    ntr = len(train)
    best_r, best_state, stale = -np.inf, None, 0
    for step in range(1, args.steps + 1):
        idx = torch.from_numpy(rng.integers(0, ntr, size=args.batch)).to(dev)
        f = fb.embed_F(pl_tr[idx], om_tr[idx])
        pred = head(f).squeeze(-1)
        cert = (w_tr[idx] * (pred - t_tr[idx]) ** 2).mean()
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
            r, lo, hi = spearman_ci(d_head, t_ho)
            print(f"step {step}  cert {float(cert):.4f}  nce {float(nce):.3f}  "
                  f"held-out rho {r:+.3f}", flush=True)
            if r > best_r:
                best_r, stale = r, 0
                best_state = ({k: v.detach().cpu().clone() for k, v in fb.state_dict().items()},
                              {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early stop at step {step} (best rho {best_r:+.3f})")
                    break
    if best_state is not None:
        fb.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])

    d_head, d_pole = readouts()
    h1 = spearman_ci(d_head, t_ho)
    print(f"VERDICT COMMITTOR_SPEARMAN pole-baseline {b0[0]:+.3f}[{b0[1]:+.3f},{b0[2]:+.3f}] "
          f"-> head {h1[0]:+.3f}[{h1[1]:+.3f},{h1[2]:+.3f}] (n={len(hold)})")
    if rim_ho.sum() >= 30:
        rb = spearman_ci(d_pole[rim_ho], t_ho[rim_ho])
        rh = spearman_ci(d_head[rim_ho], t_ho[rim_ho])
        print(f"VERDICT RIM_RESOLUTION (plies<={args.rim_plies:g}, n={int(rim_ho.sum())}) "
              f"pole {rb[0]:+.3f}[{rb[1]:+.3f},{rb[2]:+.3f}] -> "
              f"head {rh[0]:+.3f}[{rh[1]:+.3f},{rh[2]:+.3f}]")
    else:
        print(f"RIM_RESOLUTION skipped (only {int(rim_ho.sum())} rim rows held out)")

    zg = build_zgoals(Path(args.shards), fb, dev)
    zg["MATE_W"] = zW_t.detach().cpu()   # pole kept for reference readouts only
    save_ckpt(fb, Path(args.ckpt_out), step=pay.get("step", 0), zgoals=zg)
    hp = Path(args.ckpt_out).with_name(Path(args.ckpt_out).stem + "_whead.pt")
    torch.save({"state": head.state_dict(), "d_in": d_in}, hp)
    print(f"saved {args.ckpt_out} + {hp}")


if __name__ == "__main__":
    main()
