#!/usr/bin/env python
"""experiments/train_field_fullgame.py -- the PROPER FULL-BOARD field (Kaveh 2026-07-26: "train a
proper field"). Single-space (shared phi) IQE, ClockField arch, on REAL full games (Stage C data,
gen_field_data_fullgame.py). Built on catspace/train/scaffold.py (MLflow + ckpt ladders + health
gates; Ray Tune sweep optional). All losses are the tested ones (experiments/losses.py).

SIGNALS (decisions, ARCHITECTURE 8/11):
  * COMMITTOR / ending head (the VALUE) -- categorical over {WIN_MATE, DRAW*, LOSS_MATE}, trained on
    CLASS-BALANCED W/D/L game-result labels (Monte-Carlo outcome under the human play measure). This
    is the metastability committor c(s)=P(win). ALWAYS ON, the centerpiece.
  * MULTI-GOAL quasimetric -- same-game pairs d(phi(s_i),phi(s_j)) -> ply gap (reachability geometry,
    triangulation/composability). ALWAYS ON.
  * REPULSION -- anti-collapse (eff_rank health gate). ALWAYS ON.
  * MATE readout + WDL hinge -- GROUNDED on the <=7-piece tablebase-won subset (exact DTZ). GATED:
    trains only if that subset is non-empty in the batch (full games are mostly off-tablebase).

GATES logged every eval: eff_rank(phi) (collapse), committor-MAE (value calibration vs actual W/D/L),
multi-goal pair-order (Spearman). Health-gated per TRAINING_STANDARDS.
"""
from __future__ import annotations

import argparse, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_clock_field import ClockField
from experiments.losses import quasimetric_regression, wdl_hinge, categorical_ending_loss
from experiments.arch_bakeoff import eff_rank
from catspace.train.scaffold import standard_train, TrainConfig, resolve_device


def load_field_data(path):
    z = np.load(path)
    planes = z["planes"]; dtz = z["dtz"].astype(np.int32); ending = z["ending"].astype(np.int64)
    game = z["game"]; ply = z["ply"]
    # class indices for BALANCED committor training (score buckets: win / draw / loss)
    idx_win = np.flatnonzero(ending == 0)
    idx_loss = np.flatnonzero(ending == 5)
    idx_draw = np.flatnonzero((ending >= 1) & (ending <= 4))
    idx_tbwon = np.flatnonzero(dtz >= 0)                          # tablebase-grounded won subset
    # multi-goal same-line pairs (s before g -> d = ply gap)
    g2 = defaultdict(list)
    for i in range(len(dtz)):
        g2[int(game[i])].append(i)
    rng = np.random.default_rng(0)
    MG_s, MG_g, MG_d = [], [], []
    for rows in g2.values():
        rows = sorted(rows, key=lambda i: ply[i])
        if len(rows) < 2:
            continue
        for _ in range(min(10, len(rows))):
            a, b = sorted(rng.integers(0, len(rows), 2))
            if a == b:
                continue
            si, gj = rows[a], rows[b]; delta = int(ply[gj] - ply[si])
            if delta <= 0:
                continue
            MG_s.append(si); MG_g.append(gj); MG_d.append(np.log1p(delta))
    return dict(planes=planes, dtz=dtz, ending=ending,
                idx_win=idx_win, idx_draw=idx_draw, idx_loss=idx_loss, idx_tbwon=idx_tbwon,
                MG_s=np.array(MG_s), MG_g=np.array(MG_g), MG_d=np.array(MG_d, np.float32))


def make_step(net, opt, D, dev, args, rng):
    planes = D["planes"]
    tgt_mate = np.where(D["dtz"] >= 0, np.log1p(np.clip(D["dtz"], 0, None)), 0.0).astype(np.float32)
    end_t = torch.from_numpy(D["ending"]).to(dev)
    tgt_mate_t = torch.from_numpy(tgt_mate).to(dev)
    won_all = torch.from_numpy((D["dtz"] >= 0).astype(np.float32)).to(dev)
    logM = float(np.log1p(args.margin))

    def fp(idx):
        return torch.from_numpy(planes[idx].astype(np.float32)).to(dev)

    def balanced_committor_batch(nb):
        per = max(1, nb // 3)
        parts = []
        for idx in (D["idx_win"], D["idx_draw"], D["idx_loss"]):
            if len(idx):
                parts.append(idx[rng.integers(0, len(idx), per)])
        return np.concatenate(parts)

    def step(_net, s):
        # multi-goal + repulsion (shared phi)
        pi = rng.integers(0, len(D["MG_s"]), args.pairs)
        es = net.phi(fp(D["MG_s"][pi])); eg = net.phi(fp(D["MG_g"][pi]))
        L_multi = quasimetric_regression(net.d_pair_emb(es, eg),
                                         torch.from_numpy(D["MG_d"][pi]).to(dev))
        perm = torch.randperm(len(pi), device=dev)
        L_repel = F.relu(args.repel_margin - torch.log1p(net.d_pair_emb(es, eg[perm]).clamp(min=0))).mean()
        # committor / ending head on CLASS-BALANCED W/D/L
        cb = balanced_committor_batch(args.batch)
        _, catlog = net.d_mate_and_end(fp(cb))
        L_cat = categorical_ending_loss(catlog, end_t[cb])
        loss = args.w_multi * L_multi + args.w_repel * L_repel + args.w_cat * L_cat
        # mate + WDL hinge GATED on tablebase-grounded subset availability
        L_mate = torch.zeros((), device=dev); L_hinge = torch.zeros((), device=dev)
        if len(D["idx_tbwon"]) >= 8:
            hb = args.batch // 2
            bw = D["idx_tbwon"][rng.integers(0, len(D["idx_tbwon"]), hb)]
            bi = D["idx_loss"] if len(D["idx_loss"]) else D["idx_draw"]
            bi = bi[rng.integers(0, len(bi), hb)]
            bb = np.concatenate([bw, bi])
            dm, _ = net.d_mate_and_end(fp(bb))
            wmask = won_all[bb].bool()
            if wmask.any():
                L_mate = quasimetric_regression(dm[wmask], tgt_mate_t[bb][wmask])
            L_hinge = wdl_hinge(dm, won_all[bb], logM)
            loss = loss + args.w_mate * L_mate + args.w_hinge * L_hinge
        opt.zero_grad(); loss.backward(); opt.step()
        return {k: float(v.detach()) for k, v in
                {"loss": loss, "multi": L_multi, "repel": L_repel,
                 "cat": L_cat, "mate": L_mate, "hinge": L_hinge}.items()}

    return step, fp


def make_gates(net, D, dev, fp, rng):
    from scipy.stats import spearmanr
    planes = D["planes"]

    def gates(_net):
        te = rng.integers(0, len(D["MG_s"]), min(4000, len(D["MG_s"])))
        dp = net.d_pair(fp(D["MG_s"][te]), fp(D["MG_g"][te])).cpu().numpy()
        pair_order = float(spearmanr(dp, np.expm1(D["MG_d"][te])).correlation)
        sub = rng.integers(0, len(planes), 3000)
        er = float(eff_rank(net.phi(fp(sub)).cpu().numpy()))
        # committor-MAE vs actual W/D/L score
        actual = np.where(D["ending"][sub] == 0, 1.0, np.where(D["ending"][sub] == 5, 0.0, 0.5)).astype(np.float32)
        comm = net.committor(fp(sub)).cpu().numpy()
        comm_mae = float(np.abs(comm - actual).mean())
        return {"pair_order": pair_order, "eff_rank": er, "committor_mae": comm_mae}

    return gates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/field_fullgame.npz")
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8); ap.add_argument("--margin", type=float, default=400.0)
    ap.add_argument("--w-multi", type=float, default=1.0); ap.add_argument("--w-mate", type=float, default=1.0)
    ap.add_argument("--w-hinge", type=float, default=1.0); ap.add_argument("--w-repel", type=float, default=0.3)
    ap.add_argument("--w-cat", type=float, default=1.0); ap.add_argument("--repel-margin", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pairs", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="artifacts/experiments/field_fullgame")
    ap.add_argument("--ckpt-every", type=int, default=2000); ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device(args.device); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    D = load_field_data(args.data)
    print(f"[field-fullgame] positions {len(D['ending'])} | win {len(D['idx_win'])} draw {len(D['idx_draw'])} "
          f"loss {len(D['idx_loss'])} tb-won {len(D['idx_tbwon'])} | multi-goal pairs {len(D['MG_s'])} "
          f"| device {dev}", flush=True)

    net = ClockField(args.d, ch=args.ch, blocks=args.blocks, in_planes=112).to(dev)
    print(f"  {sum(p.numel() for p in net.parameters())/1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    step, fp = make_step(net, opt, D, dev, args, rng)
    gates = make_gates(net, D, dev, fp, rng)

    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      eval_every=args.eval_every, experiment="catspace_field_fullgame",
                      run_name=Path(args.out).name,
                      extra={"cfg": {"d": args.d, "ch": args.ch, "blocks": args.blocks, "in_planes": 112}})
    last = standard_train(step, net, cfg, args=args, gates_fn=gates)
    print(f"VERDICT FIELD-FULLGAME d{args.d}: pair-order {last.get('pair_order', float('nan')):+.3f} | "
          f"eff_rank {last.get('eff_rank', float('nan')):.1f} | committor-MAE {last.get('committor_mae', float('nan')):.3f} "
          f"| loss {last.get('loss', float('nan')):.3f} | [{time.time()-t0:.0f}s]", flush=True)
    print(f"  saved ladder -> {args.out}_latest.pt (+ step ladder)", flush=True)


if __name__ == "__main__":
    main()
