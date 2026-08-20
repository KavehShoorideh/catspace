#!/usr/bin/env python
"""gen_concept_stats.py -- per-(head, code) corpus outcome statistics for the concepts panel.

Kaveh 2026-08-20: "note which concepts help me and which help my opponent if we know."
The honest version of "if we know": sweep the 300k graph corpus through the frozen gcf3
encoder, and for every (global head, code) record the MEAN WHITE-POV GAME OUTCOME of
positions carrying that code, plus the count. At play time a position's active codes look
up their historical lean; codes with small n or |mean| get shown as neutral -- the panel
never claims more than the corpus supports.

    .venv/bin/python -m ...gen_concept_stats --ckpt artifacts/experiments/gcf3_latest.pt
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/gcf3_latest.pt")
    ap.add_argument("--graphs", default="data/derived/move_graphs_v2_300000_6545146.npz")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="data/derived/concept_stats_gcf3.npz")
    ap.add_argument("--standard-only", type=int, default=1,
                    help="exclude the piece-down games: their engineered imbalances leak "
                         "huge fake outcome leans into early-game codes (the move-1 "
                         "'material balance improves on a knight hop' artifact)")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_concept_field import GraphConceptField
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_move_policy import featurize, MV_CAP
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import factor_vocab
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    dev = args.device
    t0 = time.time()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
    z = np.load(args.graphs)
    keys = ("sq", "gl", "mv", "mut", "vac", "dsc", "att", "n_mv", "n_att", "p_row")
    D = {k: torch.from_numpy(np.ascontiguousarray(z[k])) for k in keys}
    om = z["outcome_mover"].astype(np.float32)
    stm = z["gl"][:, 0].astype(bool)
    ow = np.where(stm, om, -om)                      # White-POV outcome per row
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{len(tr.tok)}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    cr, ce, co, egs = factor_vocab(vocab)
    net = GraphConceptField(
        d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"], n_res=len(vocab),
        n_eg=len(egs), cls_reach=cr, cls_eg=ce, cls_out=co, edge_codes=A["edge_codes"],
        move_codes=A["move_codes"], gps_parallel=bool(A["gps_parallel"]),
        attn_bias=bool(A["attn_bias"]), agg_max=bool(A["agg_max"]), kin=bool(A["kin"]),
        file_nodes=bool(A["file_nodes"]), phase_head=bool(A["phase_head"]),
        threat_head=bool(A["threat_head"])).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()

    H, C = net.g_heads, 32
    s_sum = np.zeros((H, C)); s_cnt = np.zeros((H, C), np.int64)
    s_sq = np.zeros((H, C))
    # per-(head, code, GAME) partial sums for CLUSTER-ROBUST CIs (Kaveh 2026-08-20
    # "where's the ci coming from": positions share their game's outcome, so iid SEs are
    # up to ~10x too narrow for codes that fire all game long; SE must be over games)
    gor = tr.game_of_row()
    gids = gor[z["p_row"]]
    keep = np.ones(N, bool)
    if args.standard_only:
        keep = gids < args.games                      # piece-down games are appended AFTER
        print(f"  standard-only: {int(keep.sum()):,}/{N:,} rows "
              f"({N - int(keep.sum()):,} piece-down rows excluded)", flush=True)
    g_part = {}                                       # (h, c, gid) -> [sum, n]
    N = len(D["p_row"])
    with torch.no_grad():
        for a in range(0, N, args.batch):
            ii = torch.arange(a, min(a + args.batch, N))
            b = {k: D[k][ii].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
            sqf, mvf, edges, mask = featurize(b, dev, n_nodes=net.n_nodes)
            off = (torch.arange(len(ii), device=dev) * net.n_nodes)[:, None]
            jj = torch.arange(MV_CAP, device=dev)[None, :]
            for name in ("vac", "dsc"):
                arr = D[name][ii].to(dev).long()
                val = (arr != 255) & mask[:, :, None]
                edges[name] = (((jj + 64 + off)[:, :, None].expand_as(arr))[val],
                               (arr + off[:, :, None])[val])
            _, g_ids, _, _, _, _ = net.encode(sqf, mvf, edges, mask)
            gi = g_ids.cpu().numpy()
            w = ow[ii.numpy()]
            gb = gids[ii.numpy()]
            kp = keep[ii.numpy()]
            gi, w, gb = gi[kp], w[kp], gb[kp]
            if not len(w):
                continue
            for h in range(H):
                np.add.at(s_sum, (h, gi[:, h]), w)
                np.add.at(s_sq, (h, gi[:, h]), w * w)
                np.add.at(s_cnt, (h, gi[:, h]), 1)
                for j in range(len(w)):
                    k = (h, int(gi[j, h]), int(gb[j]))
                    e = g_part.get(k)
                    if e is None:
                        g_part[k] = [float(w[j]), 1]
                    else:
                        e[0] += float(w[j]); e[1] += 1
            if (a // args.batch) % 100 == 0:
                print(f"  {a:,}/{N:,} [{time.time()-t0:.0f}s]", flush=True)
    mean = s_sum / np.maximum(s_cnt, 1)
    ex2 = s_sq / np.maximum(s_cnt, 1)
    # cluster-robust (CR0) 95 percent CI: var_cluster = sum_g (S_g - n_g*mu)^2 / N^2
    cr = np.zeros((H, C)); ngames = np.zeros((H, C), np.int64)
    for (h, c, _gid), (sg, ng) in g_part.items():
        cr[h, c] += (sg - ng * mean[h, c]) ** 2
        ngames[h, c] += 1
    ci_cluster = 1.96 * np.sqrt(cr) / np.maximum(s_cnt, 1)
    np.savez_compressed(args.out, mean_wpov=mean, count=s_cnt, ex2=ex2,
                        ci_cluster=ci_cluster, n_games=ngames, ckpt=args.ckpt)
    big = (s_cnt > 1000)
    iid_ci = 1.96 * np.sqrt(np.maximum(ex2 - mean ** 2, 0) / np.maximum(s_cnt, 1))
    infl = np.where(iid_ci > 0, ci_cluster / np.maximum(iid_ci, 1e-9), 1.0)
    print(f"  cluster-CI inflation vs iid on well-sampled codes: "
          f"median {np.median(infl[big]):.1f}x, max {infl[big].max():.1f}x")
    live = (s_cnt > 0).sum()
    print(f"\n  wrote {args.out} | {live}/{H*C} codes seen | "
          f"strongest leans: {np.abs(mean)[s_cnt>300].max():.3f} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
