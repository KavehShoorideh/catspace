#!/usr/bin/env python
"""
experiments/two_channel_distill.py — split the field into TWO channels
(Kaveh 2026-07-15, after the multi-eps identification): plies and sharpness
are ~orthogonal (S-vs-|dtz| rho +0.036), so no constant-lambda fusion can be
right. Phase 1 re-distills the quasimetric d to PURE plies (tb-White eps=0.05
table: near-optimal conversion lengths); phase 2 trains a separate S-head
(frozen F -> softplus scalar) on the identified per-state sharpness. Risk
enters only at READOUT: reach_eff = reach - g_sharp * S(F(s)), with g_sharp
the fallibility weight (omega-dependent later).
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
    ap.add_argument("--ckpt-out", default="data/derived/sep/two_channel.pt")
    ap.add_argument("--plies-table", default="artifacts/experiments/certainty_table_eps05.json")
    ap.add_argument("--sharp-table", default="artifacts/experiments/sharpness_table.json")
    ap.add_argument("--shards", default="data/shards/lichess_db_standard_rated_2019-01.prefix4gb")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--scale", type=float, default=50.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--s-steps", type=int, default=3000)
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

    def encode(fens):
        boards = [chess.Board(f) for f in fens]
        packed = np.stack([encode_packed(b) for b in boards])
        meta = np.stack([encode_meta(b) for b in boards])
        om = omega_ids(np.full(len(fens), 1800), np.full(len(fens), 1800),
                      np.full(len(fens), np.nan))
        return (torch.from_numpy(feature_planes(packed, meta)).to(dev),
                torch.from_numpy(om).to(dev))

    # ---------------- phase 1: d -> PURE plies ----------------
    rows = [r for r in json.loads(Path(args.plies_table).read_text())["rows"]
            if r["plies"] is not None and r["n"] >= 6]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    n_hold = len(rows) // 5
    hold, train = [rows[i] for i in order[:n_hold]], [rows[i] for i in order[n_hold:]]
    print(f"phase 1 (plies): {len(train)} train / {len(hold)} held-out states")
    tgt_tr = torch.tensor([r["plies"] / args.scale for r in train],
                          dtype=torch.float32, device=dev)
    w_tr = torch.tensor([np.sqrt(r["n"]) for r in train], dtype=torch.float32, device=dev)
    w_tr = w_tr / w_tr.mean()
    pl_tr, om_tr = encode([r["fen"] for r in train])
    pl_ho, om_ho = encode([r["fen"] for r in hold])
    tgt_ho = np.array([r["plies"] / args.scale for r in hold])

    def ho_spear(model):
        model.eval()
        with torch.no_grad():
            d = model.distance_matrix(model.embed_F(pl_ho, om_ho), zW_t[None, :])[:, 0].cpu().numpy()
        model.train()
        return spearman_ci(d, tgt_ho)

    r0 = ho_spear(fb)
    print(f"BASELINE held-out Spearman(d, plies) = {r0[0]:+.3f} CI[{r0[1]:+.3f},{r0[2]:+.3f}]")
    src = LichessPairSource(Path(args.shards), gamma=0.95)
    it = iter(src.batches(args.batch, seed=args.seed))
    opt = torch.optim.AdamW(fb.parameters(), lr=args.lr)
    fb.train()
    best_r, best_state, stale = -np.inf, None, 0
    for step in range(1, args.steps + 1):
        idx = torch.from_numpy(rng.integers(0, len(train), size=args.batch)).to(dev)
        f = fb.embed_F(pl_tr[idx], om_tr[idx])
        d = fb.distance_matrix(f, zW_t[None, :])[:, 0]
        loss_p = (w_tr[idx] * (d - tgt_tr[idx]) ** 2).mean()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(src.batches(args.batch, seed=args.seed + step)); batch = next(it)
        tens = batch_tensors(batch, dev)
        if tens is None:
            continue
        nce, _ = fb.loss_fn(*tens[:5], ply_gap_weight=0.05)
        (loss_p + nce).backward()
        opt.step(); opt.zero_grad()
        if step % args.eval_every == 0:
            r, lo, hi = ho_spear(fb)
            print(f"step {step}  plies {float(loss_p):.4f}  nce {float(nce):.3f}  ho rho {r:+.3f}", flush=True)
            if r > best_r:
                best_r, stale = r, 0
                best_state = {k: v.detach().cpu().clone() for k, v in fb.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early stop at {step}")
                    break
    if best_state is not None:
        fb.load_state_dict(best_state)
    r1 = ho_spear(fb)
    print(f"VERDICT PLIES_CHANNEL baseline {r0[0]:+.3f} -> tuned {r1[0]:+.3f}[{r1[1]:+.3f},{r1[2]:+.3f}]")

    # ---------------- phase 2: S-head on frozen F ----------------
    fb.eval()
    srows = json.loads(Path(args.sharp_table).read_text())["rows"]
    s_order = rng.permutation(len(srows))
    s_hold = [srows[i] for i in s_order[:len(srows) // 5]]
    s_train = [srows[i] for i in s_order[len(srows) // 5:]]
    print(f"phase 2 (S-head): {len(s_train)} train / {len(s_hold)} held-out")

    def embF(fens):
        pl, om = encode(fens)
        with torch.no_grad():
            return fb.embed_F(pl, om)
    F_tr, F_ho = embF([r["fen"] for r in s_train]), embF([r["fen"] for r in s_hold])
    S_tr = torch.tensor([max(r["S"], 0.0) for r in s_train], dtype=torch.float32, device=dev)
    S_ho = np.array([max(r["S"], 0.0) for r in s_hold])
    shead = torch.nn.Sequential(torch.nn.Linear(F_tr.shape[1], 128), torch.nn.ReLU(),
                                torch.nn.Linear(128, 1), torch.nn.Softplus()).to(dev)
    sopt = torch.optim.AdamW(shead.parameters(), lr=3e-4)
    for step in range(1, args.s_steps + 1):
        idx = torch.from_numpy(rng.integers(0, len(s_train), size=256)).to(dev)
        pred = shead(F_tr[idx]).squeeze(-1)
        loss = ((pred - S_tr[idx]) ** 2).mean()
        loss.backward(); sopt.step(); sopt.zero_grad()
    with torch.no_grad():
        pred_ho = shead(F_ho).squeeze(-1).cpu().numpy()
    rs = spearman_ci(pred_ho, S_ho)
    print(f"VERDICT S_HEAD held-out Spearman {rs[0]:+.3f} CI[{rs[1]:+.3f},{rs[2]:+.3f}]  "
          f"RMSE {float(np.sqrt(np.mean((pred_ho - S_ho)**2))):.2f} (S sd {S_ho.std():.2f})")

    zg = build_zgoals(Path(args.shards), fb, dev)
    zg["MATE_W"] = zW_t.detach().cpu()
    save_ckpt(fb, Path(args.ckpt_out), step=pay.get("step", 0), zgoals=zg)
    sp = Path(args.ckpt_out).with_name(Path(args.ckpt_out).stem + "_shead.pt")
    torch.save({"state": shead.state_dict(), "d_in": F_tr.shape[1]}, sp)
    print(f"saved {args.ckpt_out} + {sp}")


if __name__ == "__main__":
    main()
