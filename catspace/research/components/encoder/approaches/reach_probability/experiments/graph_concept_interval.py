#!/usr/bin/env python
"""graph_concept_interval.py -- alpha/beta bounds FROM THE CONCEPTS (Kaveh 2026-08-20).

"I would actually like us to use our own concepts to predict this. Basically predict
alpha and beta. Let's see if our concepts do a good job of this."

The field (gcf4h) is FROZEN; the interval head reads ONLY the quantized concept state zq
plus the move's VQ-code embedding -- a pure readout probe. If [dV_lo, dV_hi] from this
input is sharp and calibrated, the discrete vocabulary carries the bound information; the
reference is gdvq1 (same losses, same CF pair data, CONTINUOUS move-node features).
Training is head-only over precomputed features: the encode sweep runs once and caches.

    .venv/bin/python -m ...graph_concept_interval --field artifacts/experiments/gcf4h_latest.pt
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .graph_move_policy import MV_CAP


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="artifacts/experiments/gcf4h_latest.pt")
    ap.add_argument("--cf-graphs", default="data/derived/move_graphs_v2_cf_24696_6545146.npz")
    ap.add_argument("--cf", default="data/derived/counterfactual_45142_6545146.npz")
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--tau-lo", type=float, default=0.1)
    ap.add_argument("--tau-hi", type=float, default=0.9)
    ap.add_argument("--fixed-elo", type=float, default=2750)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/gciv1")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_concept_field import GraphConceptField
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_move_policy import featurize
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.research.components.encoder.approaches.reach_probability.src \
        .trajectories import normalise_elo

    t0 = time.time()
    ck = torch.load(args.field, map_location="cpu", weights_only=False)
    A = ck["args"]
    field = GraphConceptField(
        d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"], n_res=3, n_eg=1,
        cls_reach=np.zeros(3, np.int64), cls_eg=np.zeros(3, np.int64),
        cls_out=np.arange(3, dtype=np.int64), edge_codes=A["edge_codes"],
        move_codes=A["move_codes"], gps_parallel=bool(A["gps_parallel"]),
        attn_bias=bool(A["attn_bias"]), agg_max=bool(A["agg_max"]), kin=bool(A["kin"]),
        file_nodes=bool(A["file_nodes"]), phase_head=bool(A["phase_head"]),
        threat_head=bool(A["threat_head"]), elo_cond=bool(A.get("elo", 0)),
        g_heads=A["g_heads"], g_codes=A["g_codes"], d_code=A["d_code"]).to(dev)
    field.load_state_dict(ck["state_dict"]); field.eval()
    for p2 in field.parameters():
        p2.requires_grad_(False)
    assert A["move_codes"] > 0, "concept-move codes required for the pure-concept probe"

    C = np.load(args.cf_graphs)
    CD = {k: torch.from_numpy(np.ascontiguousarray(C[k])) for k in
          ("sq", "gl", "mv", "mut", "vac", "dsc", "att", "n_mv", "n_att", "p_row")}
    cf = np.load(args.cf)
    rowmap = {int(r): i for i, r in enumerate(CD["p_row"].numpy())}
    mvC, n_mvC = CD["mv"].numpy(), CD["n_mv"].numpy()
    played = C["played"]
    gi_l, ja_l, jp_l, dl_l = [], [], [], []
    for row, rf, rt, tr_, ct in zip(cf["row"], cf["rnd_from"], cf["rnd_to"],
                                    cf["treat"], cf["ctrl"]):
        gidx = rowmap.get(int(row))
        if gidx is None:
            continue
        k = n_mvC[gidx]
        hit = np.flatnonzero((mvC[gidx, :k, 1] == rf) & (mvC[gidx, :k, 2] == rt))
        if not len(hit) or int(hit[0]) == int(played[gidx]):
            continue
        gi_l.append(gidx); ja_l.append(int(hit[0])); jp_l.append(int(played[gidx]))
        dl_l.append(float(int(tr_) - int(ct)))
    cgi = np.array(gi_l); cja = np.array(ja_l); cjp = np.array(jp_l)
    cdl = np.array(dl_l, np.float32)

    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    gor = tr.game_of_row()
    rs = np.random.default_rng(args.seed); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    is_te = test_g[gor[CD["p_row"].numpy()[cgi]]]
    ctr = np.flatnonzero(~is_te); cte = np.flatnonzero(is_te)
    print(f"[gciv] {len(cgi):,} CF pairs ({len(ctr):,} train / {len(cte):,} held-out) "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- PRECOMPUTE the frozen concept features once: zq + per-slot move-code embeds --
    Np = len(CD["p_row"])
    ZQ = torch.zeros(Np, field.d_state)
    QMV = torch.zeros(Np, MV_CAP, 32)
    e_norm = float(normalise_elo(np.float32(args.fixed_elo)))
    with torch.no_grad():
        for a in range(0, Np, 192):
            idx = torch.arange(a, min(a + 192, Np))
            b = {k: CD[k][idx].to(dev) for k in ("sq", "gl", "mv", "mut", "att",
                                                 "n_mv", "n_att")}
            sqf, mvf, edges, mask = featurize(b, dev, n_nodes=field.n_nodes)
            off = (torch.arange(len(idx), device=dev) * field.n_nodes)[:, None]
            jj = torch.arange(MV_CAP, device=dev)[None, :]
            for name in ("vac", "dsc"):
                arr = CD[name][idx].to(dev).long()
                val = (arr != 255) & mask[:, :, None]
                edges[name] = (((jj + 64 + off)[:, :, None].expand_as(arr))[val],
                               (arr + off[:, :, None])[val])
            elo = torch.full((len(idx), 2), e_norm, device=dev)
            zq, _, _, _, h_mv, _ = field.encode(sqf, mvf, edges, mask, elo=elo)
            q_mv, _, _ = field.mv_vq(field.mv_vq_proj(h_mv))
            ZQ[idx] = zq.cpu()
            QMV[idx] = q_mv.cpu()
            if a % (192 * 20) == 0:
                print(f"  precompute {a:,}/{Np:,} [{time.time()-t0:.0f}s]", flush=True)
    print(f"[gciv] features cached [{time.time()-t0:.0f}s]", flush=True)

    head = nn.Sequential(nn.Linear(field.d_state + 32, args.hidden), nn.GELU(),
                         nn.Linear(args.hidden, 3)).to(dev)     # (dv, off_lo, off_hi)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    def fwd(rec_idx, slot_arr):
        gi = cgi[rec_idx]
        x = torch.cat([ZQ[gi], QMV[gi, slot_arr]], -1).to(dev)
        o = head(x)
        dv = o[:, 0]
        lo = dv - F.softplus(o[:, 1])
        hi = dv + F.softplus(o[:, 2])
        return dv, lo, hi

    def pinball(err, tau):
        return torch.maximum(tau * err, (tau - 1) * err).mean()

    def eval_split(split):
        with torch.no_grad():
            cov = mae = n = 0
            sgn = sgnn = 0
            wid = 0.0
            for a in range(0, len(split), 4096):
                ii = split[a:a + 4096]
                dva, loa, hia = fwd(ii, cja[ii])
                dvp, lop, hip = fwd(ii, cjp[ii])
                dl = torch.from_numpy(cdl[ii]).to(dev)
                d_lo, d_hi = loa - hip, hia - lop
                cov += int(((dl >= d_lo) & (dl <= d_hi)).sum()); n += len(ii)
                dh = dva - dvp
                mae += float((dh - dl).abs().sum())
                nz = dl != 0
                sgn += int(((dh[nz] > 0) == (dl[nz] > 0)).sum()); sgnn += int(nz.sum())
                wid += float((hia - loa).sum())
        return cov / n, mae / n, sgn / max(sgnn, 1), wid / n

    for step in range(1, args.steps + 1):
        ii = ctr[rng.integers(0, len(ctr), args.batch)]
        dva, loa, hia = fwd(ii, cja[ii])
        dvp, lop, hip = fwd(ii, cjp[ii])
        dl = torch.from_numpy(cdl[ii]).to(dev)
        l_pt = F.huber_loss(dva - dvp, dl) + 0.2 * (dvp ** 2).mean()
        d_lo, d_hi = loa - hip, hia - lop
        l_iv = pinball(dl - d_lo, args.tau_lo) + pinball(dl - d_hi, args.tau_hi)
        loss = l_pt + l_iv
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == 1:
            cov, mae, sgn, wid = eval_split(cte)
            print(f"[gciv] step {step:>5}/{args.steps} | loss {float(loss.detach()):.3f} | "
                  f"HELD-OUT sign {sgn:.3f} mae {mae:.3f} | IV coverage {cov:.3f} "
                  f"(target {args.tau_hi - args.tau_lo:.1f}) width {wid:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    torch.save({"head": head.state_dict(), "args": vars(args), "field": args.field},
               args.out + "_latest.pt")
    print(f"[gciv] wrote {args.out}_latest.pt")


if __name__ == "__main__":
    main()
