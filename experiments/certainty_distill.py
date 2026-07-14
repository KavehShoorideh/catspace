#!/usr/bin/env python
"""
experiments/certainty_distill.py — distill the certainty field into the slow embedding.

Fine-tunes the incumbent so its quasimetric distance to the mate goal matches the
CERTAINTY-WEIGHTED target on rollout-estimated toy states:
    d_target(s) = (plies(s) + lambda * (-ln P_clip(s))) / scale
    P_clip = max(p_hat, 1/(n+2))   (Laplace floor -- kills -ln 0)
Rows with p_hat=0 have no observed plies -> plies := horizon cap (they are "at
least a horizon away" in certainty terms). Regression is weighted by sqrt(n)
(visit-count confidence). A standard InfoNCE batch from human shards is mixed in
EVERY step so the global field isn't forgotten (the certainty MSE touches only the
toy region).

EVAL (hardened, no point estimates): 20% of rows are HELD OUT before training;
after training we report Spearman(learned d, d_target) on them with a bootstrap CI,
for both the tuned model and the incumbent baseline. Play eval (paired 200-node
playout on the disjoint test_n200) is run separately via playout_ab.py.
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


def spearman_ci(a, b, boot=2000, seed=0):
    def rho(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])
    r = rho(a, b)
    rng = np.random.default_rng(seed)
    n = len(a)
    bs = [rho(a[i], b[i]) for i in (rng.integers(0, n, size=(boot, n)))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return r, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-in", default="data/derived/lichess_fb_4gb_qm_plygap_only.pt")
    ap.add_argument("--ckpt-out", default="data/derived/sep/certainty_distill.pt")
    ap.add_argument("--table", default="artifacts/experiments/certainty_table.json")
    ap.add_argument("--shards", default="data/shards/lichess_db_standard_rated_2019-01.prefix4gb")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lam", type=float, default=8.0, help="plies per nat of -lnP")
    ap.add_argument("--scale", type=float, default=50.0, help="same normaliser as ply-gap term")
    ap.add_argument("--horizon", type=float, default=100.0, help="plies cap for p_hat=0 rows")
    ap.add_argument("--cert-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
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

    def target(r):
        p = max(r["p_hat"], 1.0 / (r["n"] + 2))
        plies = r["plies"] if r["plies"] is not None else args.horizon
        return (plies + args.lam * (-np.log(p))) / args.scale

    def encode(rs):
        boards = [chess.Board(r["fen"]) for r in rs]
        packed = np.stack([encode_packed(b) for b in boards])
        meta = np.stack([encode_meta(b) for b in boards])
        om = omega_ids(np.full(len(rs), 1800), np.full(len(rs), 1800),
                       np.full(len(rs), np.nan))
        return (torch.from_numpy(feature_planes(packed, meta)).to(dev),
                torch.from_numpy(om).to(dev))

    d_tr = torch.tensor([target(r) for r in train], dtype=torch.float32, device=dev)
    w_tr = torch.tensor([np.sqrt(r["n"]) for r in train], dtype=torch.float32, device=dev)
    w_tr = w_tr / w_tr.mean()
    pl_tr, om_tr = encode(train)

    def heldout_spearman(model):
        model.eval()
        with torch.no_grad():
            pl, om = encode(hold)
            f = model.embed_F(pl, om)
            d = model.distance_matrix(f, zW_t[None, :])[:, 0].cpu().numpy()
        model.train()
        t = np.array([target(r) for r in hold])
        return spearman_ci(d, t)

    r0, lo0, hi0 = heldout_spearman(fb)
    print(f"BASELINE held-out Spearman(d, target) = {r0:+.3f} CI[{lo0:+.3f},{hi0:+.3f}] (n={len(hold)})")

    src = LichessPairSource(Path(args.shards), gamma=0.95)
    it = iter(src.batches(args.batch, seed=args.seed))
    opt = torch.optim.AdamW(fb.parameters(), lr=args.lr)
    fb.train()
    ntr = len(train)
    for step in range(1, args.steps + 1):
        idx = torch.from_numpy(rng.integers(0, ntr, size=args.batch)).to(dev)
        f = fb.embed_F(pl_tr[idx], om_tr[idx])
        d = fb.distance_matrix(f, zW_t[None, :])[:, 0]
        cert = (w_tr[idx] * (d - d_tr[idx]) ** 2).mean()
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
        if step % 200 == 0:
            print(f"step {step}  cert {float(cert):.4f}  nce {float(nce):.3f}", flush=True)

    r1, lo1, hi1 = heldout_spearman(fb)
    print(f"TUNED    held-out Spearman(d, target) = {r1:+.3f} CI[{lo1:+.3f},{hi1:+.3f}]")
    print(f"VERDICT CERT_SPEARMAN baseline {r0:+.3f}[{lo0:+.3f},{hi0:+.3f}] -> "
          f"tuned {r1:+.3f}[{lo1:+.3f},{hi1:+.3f}]")
    zg = build_zgoals(Path(args.shards), fb, dev)
    # the cert loss calibrated d(F(s), zW_in) -- saving a rebuilt MATE_W would
    # point playout at a goal the distances were never fit to (measured: ~0.05
    # of held-out rho lost, JOURNAL 2026-07-14 correction)
    zg["MATE_W"] = zW_t.detach().cpu()
    save_ckpt(fb, Path(args.ckpt_out), step=pay.get("step", 0), zgoals=zg)
    print(f"saved {args.ckpt_out}")


if __name__ == "__main__":
    main()
