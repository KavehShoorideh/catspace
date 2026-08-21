#!/usr/bin/env python
"""gko_train.py -- gko0: the flywheel's consumer (docs/GKO1_PLAN.md, first slice).

Trains on the oracle shards as the converter writes them (pipelined; no fixed corpus):
    MAIA rows   the move head learns P(move | position, elo band) -- KL to the policy
                the band's Maia produced, elo-conditioned via the existing machinery
                (one model, all bands: the dial becomes real)
    SF rows     a NEW per-move action-value head learns HL-Gauss-binned win-probs for
                ALL legal moves (the full-ordering supervision Kaveh authorized)
Vocabulary protection: VQ commitment + cross-head decorrelation stay on; the rest of the
field stack is frozen-by-weight-zero for this slice (operators/intervals join as their
unit gates pass -- gko1 proper).

    .venv/bin/python -u -m ...gko_train --resume-field artifacts/experiments/gcf4h_latest.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .graph_move_policy import MV_CAP


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume-field", default="artifacts/experiments/gcf4h_latest.pt")
    ap.add_argument("--resume-self", default="", help="a prior gko checkpoint (wrapper "
                    "state: field.* + sf_av.*) to continue from")
    ap.add_argument("--shards", default="data/derived/gko1_shards")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--k-bins", type=int, default=64)
    ap.add_argument("--w-maia", type=float, default=1.0)
    ap.add_argument("--w-sf", type=float, default=1.0)
    ap.add_argument("--w-decor", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--rescan-every", type=int, default=200)
    ap.add_argument("--max-shards", type=int, default=400)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/gko0")
    args = ap.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    dev = args.device

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_concept_field import GraphConceptField
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_move_policy import featurize
    from catspace.research.tools.training_infra.hl_gauss import HLGauss
    from catspace.research.tools.training_infra.losses import cross_head_decorrelation
    from catspace.research.components.encoder.approaches.reach_probability.src \
        .trajectories import normalise_elo
    from catspace.research.tools.training_infra.train.scaffold import (
        TrainConfig, standard_train)

    ck = torch.load(args.resume_field, map_location="cpu", weights_only=False)
    A = ck["args"]
    field = GraphConceptField(
        d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"], n_res=3, n_eg=1,
        cls_reach=np.zeros(3, np.int64), cls_eg=np.zeros(3, np.int64),
        cls_out=np.arange(3, dtype=np.int64), edge_codes=A["edge_codes"],
        move_codes=A["move_codes"], gps_parallel=bool(A["gps_parallel"]),
        attn_bias=bool(A["attn_bias"]), agg_max=bool(A["agg_max"]), kin=bool(A["kin"]),
        file_nodes=bool(A["file_nodes"]), phase_head=bool(A["phase_head"]),
        threat_head=bool(A["threat_head"]), elo_cond=True,
        g_heads=A["g_heads"], g_codes=A["g_codes"], d_code=A["d_code"]).to(dev)
    field.load_state_dict(ck["state_dict"])

    class GKO(nn.Module):
        def __init__(self):
            super().__init__()
            self.field = field
            self.sf_av = nn.Sequential(
                nn.Linear(field.d_state + A["d_model"], 512), nn.GELU(),
                nn.Linear(512, args.k_bins))
    model = GKO().to(dev)
    if args.resume_self:
        ck2 = torch.load(args.resume_self, map_location="cpu", weights_only=False)
        model.load_state_dict(ck2["state_dict"])
        print(f"[gko0] resumed self from {args.resume_self} (step {ck2.get('step')})",
              flush=True)
    hl = HLGauss(k=args.k_bins).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    KEYS = ("sq", "gl", "mv", "mut", "vac", "dsc", "att", "n_mv", "n_att",
            "kind", "band", "tgt")
    store = {k: [] for k in KEYS}
    seen = set()

    def rescan():
        files = sorted(glob.glob(os.path.join(args.shards, "shard_*.npz")))
        new = [f for f in files if f not in seen][-args.max_shards:]
        for f in new:
            try:
                z = np.load(f, allow_pickle=True)
                for k in KEYS:
                    store[k].append(np.ascontiguousarray(z[k]))
                seen.add(f)
            except Exception:
                pass
        return len(new)

    def flat(k):
        return np.concatenate(store[k]) if store[k] else None

    while rescan() == 0 or sum(len(a) for a in store["kind"]) < 4096:
        print("[gko0] waiting for shards...", flush=True)
        time.sleep(30)
        rescan()
    D = {k: torch.from_numpy(flat(k)) for k in KEYS}
    N = len(D["kind"])
    hold = (np.frombuffer(np.ascontiguousarray(
        D["band"].numpy() * 0), dtype=np.int16, count=N) * 0)  # placeholder
    hold = rng.random(N) < 0.08
    tr_i = np.flatnonzero(~hold); te_i = np.flatnonzero(hold)
    print(f"[gko0] start: {N:,} rows ({len(te_i):,} held out) | "
          f"sf {int((D['kind']==0).sum()):,} maia {int((D['kind']==1).sum()):,}",
          flush=True)

    def refresh_store():
        nonlocal D, N, tr_i, te_i, hold
        if rescan():
            D = {k: torch.from_numpy(flat(k)) for k in KEYS}
            oldN = N; N = len(D["kind"])
            extra = rng.random(N - oldN) < 0.08
            hold = np.concatenate([hold, extra])
            tr_i = np.flatnonzero(~hold); te_i = np.flatnonzero(hold)

    def batch_of(idx):
        b = {k: D[k][idx].to(dev) for k in ("sq", "gl", "mv", "mut", "att",
                                            "n_mv", "n_att")}
        sqf, mvf, edges, mask = featurize(b, dev, n_nodes=field.n_nodes)
        off = (torch.arange(len(idx), device=dev) * field.n_nodes)[:, None]
        jj = torch.arange(MV_CAP, device=dev)[None, :]
        for name in ("vac", "dsc"):
            a2 = D[name][idx].to(dev).long()
            val = (a2 != 255) & mask[:, :, None]
            edges[name] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val],
                           (a2 + off[:, :, None])[val])
        band = D["band"][idx].float()
        elo_v = torch.where(band > 0, band, torch.full_like(band, 2750.0))
        elo = torch.from_numpy(
            normalise_elo(elo_v.numpy().astype(np.float32)))[:, None].repeat(1, 2).to(dev)
        return sqf, mvf, edges, mask, elo

    def losses_on(idx, train=True):
        sqf, mvf, edges, mask, elo = batch_of(idx)
        zq, g_ids, _, vl, h_mv, pre_g = model.field.encode(sqf, mvf, edges, mask,
                                                          elo=elo)[:6]
        lg = model.field.move_logits(zq, h_mv, mask)
        tgt = D["tgt"][idx].to(dev)
        kind = D["kind"][idx].to(dev)
        fin = torch.isfinite(tgt) & mask
        l_maia = torch.zeros((), device=dev)
        km = kind == 1
        if km.any():
            lp = F.log_softmax(lg[km].masked_fill(~mask[km], -1e9), -1)
            t2 = torch.where(fin[km], tgt[km], torch.zeros_like(tgt[km]))
            t2 = t2 / t2.sum(-1, keepdim=True).clamp(min=1e-8)
            l_maia = -(t2 * lp).masked_fill(~fin[km], 0.0).sum(-1).mean()
        l_sf = torch.zeros((), device=dev)
        ks = kind == 0
        if ks.any():
            x = torch.cat([zq[ks][:, None, :].expand(-1, MV_CAP, -1), h_mv[ks]], -1)
            logits = model.sf_av(x)
            m2 = fin[ks]
            if m2.any():
                l_sf = hl.ce(logits[m2], tgt[ks][m2])
        l_decor = cross_head_decorrelation(pre_g)
        loss = args.w_maia * l_maia + args.w_sf * l_sf + args.w_decor * l_decor + vl
        return loss, l_maia, l_sf, l_decor, (lg, zq, h_mv, fin, tgt, kind, mask)

    def step_fn(m_, step):
        model.train()
        if step % args.rescan_every == 0:
            refresh_store()
        idx = torch.from_numpy(tr_i[rng.integers(0, len(tr_i), args.batch)])
        loss, l_maia, l_sf, l_decor, _ = losses_on(idx)
        opt.zero_grad(); loss.backward(); opt.step()
        out = {"loss": float(loss.detach())}
        if step % args.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                kl_s = sp_s = kl_n = sp_n = 0.0
                ag_s = ag_n = 0
                for a in range(0, min(len(te_i), 2048), 256):
                    ii = torch.from_numpy(te_i[a:a + 256])
                    _, lm, ls, _, (lg, zq, h_mv, fin, tgt, kind, mask) = \
                        losses_on(ii, train=False)
                    km = kind == 1
                    if km.any():
                        kl_s += float(lm) * int(km.sum()); kl_n += int(km.sum())
                        # per-eval maia top-1 agreement (argmax match with the oracle)
                        pm = lg[km].masked_fill(~mask[km], -1e9).argmax(-1)
                        tt = torch.where(fin[km], tgt[km],
                                         torch.full_like(tgt[km], -1e9)).argmax(-1)
                        ag_s += int((pm == tt).sum()); ag_n += int(km.sum())
                    ks = kind == 0
                    if ks.any():
                        x = torch.cat([zq[ks][:, None, :].expand(-1, MV_CAP, -1),
                                       h_mv[ks]], -1)
                        pred = hl.decode(model.sf_av(x))
                        for r in range(len(pred)):
                            f2 = fin[ks][r]
                            if f2.sum() > 3:
                                p2 = pred[r][f2].cpu().numpy()
                                t3 = tgt[ks][r][f2].cpu().numpy()
                                pr = np.argsort(np.argsort(p2))
                                tr2 = np.argsort(np.argsort(t3))
                                c = np.corrcoef(pr, tr2)[0, 1]
                                if np.isfinite(c):
                                    sp_s += c; sp_n += 1
            el = time.time() - t0
            print(f"[gko0] step {step:>6}/{args.steps} {el/60:>6.1f}m "
                  f"({el/max(1,step):.3f}s/step) | maia {float(l_maia):.3f} "
                  f"sf {float(l_sf):.3f} decor {float(l_decor):.3f} | "
                  f"HELD-OUT maiaKL {kl_s/max(1,kl_n):.3f} "
                  f"maia-top1 {ag_s/max(1,ag_n):.3f} "
                  f"sf-spearman {sp_s/max(1,sp_n):.3f} | rows {N:,}", flush=True)
            out.update({"maia_kl": kl_s / max(1, kl_n),
                        "sf_spearman": sp_s / max(1, sp_n), "rows": N})
            model.train()
        return out

    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      start_step=0, eval_every=args.eval_every, experiment="catspace",
                      run_name=args.out.split("/")[-1], device=dev)
    standard_train(step_fn, model, cfg, args)


if __name__ == "__main__":
    main()
