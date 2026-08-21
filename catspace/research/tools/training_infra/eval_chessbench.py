#!/usr/bin/env python
"""eval_chessbench.py -- the gko1 H3 yardstick: score a gko checkpoint against the
DeepMind ChessBench test action-value bag (an EXTERNAL, fixed target our flywheel
never trains on -- U11 excised the 2,257 lichess-shared positions from the pool and
walks filter against the full test key set; the clean/seen split is reported anyway).

Per position: encode once, sf_av head scores every legal-move slot, compare with the
bag's win-probs over the same moves -- per-position Spearman and top-1 agreement.
Moves past the MV_CAP=48 slot cap are skipped and counted.

    .venv/bin/python -m catspace.research.tools.training_infra.eval_chessbench \
        --ckpt artifacts/experiments/gko0c_latest.pt --n 2000
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/gko0c_latest.pt")
    ap.add_argument("--field-args", default="artifacts/experiments/gcf4h_latest.pt")
    ap.add_argument("--bag", default="data/external/chessbench/test_action_value.bag")
    ap.add_argument("--n", type=int, default=2000, help="positions (0 = all)")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    import chess
    from scipy.stats import spearmanr
    from catspace.research.tools.training_infra.bagz import BagReader, decode_action_value
    from catspace.research.tools.training_infra.hl_gauss import HLGauss
    from catspace.research.tools.training_infra.gen_move_graphs import (
        parent_graph, glob_of_board)
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_move_policy import featurize, MV_CAP
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_concept_field import GraphConceptField
    from catspace.research.components.encoder.approaches.reach_probability.src \
        .trajectories import normalise_elo

    dev = args.device
    ckf = torch.load(args.field_args, map_location="cpu", weights_only=False)
    A = ckf["args"]
    field = GraphConceptField(
        d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"], n_res=3, n_eg=1,
        cls_reach=np.zeros(3, np.int64), cls_eg=np.zeros(3, np.int64),
        cls_out=np.arange(3, dtype=np.int64), edge_codes=A["edge_codes"],
        move_codes=A["move_codes"], gps_parallel=bool(A["gps_parallel"]),
        attn_bias=bool(A["attn_bias"]), agg_max=bool(A["agg_max"]), kin=bool(A["kin"]),
        file_nodes=bool(A["file_nodes"]), phase_head=bool(A["phase_head"]),
        threat_head=bool(A["threat_head"]), elo_cond=True,
        g_heads=A["g_heads"], g_codes=A["g_codes"], d_code=A["d_code"])
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    field.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("field.")})
    k_bins = sd["sf_av.2.bias"].shape[0]
    sf_av = torch.nn.Sequential(
        torch.nn.Linear(field.d_state + A["d_model"], 512), torch.nn.GELU(),
        torch.nn.Linear(512, k_bins))
    sf_av.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("sf_av.")})
    field.to(dev).eval(); sf_av.to(dev).eval()
    hl = HLGauss(k=k_bins).to(dev)

    # the bag is NOT fen-sorted (verified: no adjacent-equal fens) -- group by dict
    r = BagReader(args.bag)
    seen_keys = set(np.load("data/derived/chessbench_overlap_keys.npz")["keys"].tolist())
    bymap = {}
    for i in range(len(r)):
        fen, mv, wp = decode_action_value(r[i])
        bymap.setdefault(fen, []).append((mv, wp))
    groups = list(bymap.items())
    rng = np.random.default_rng(args.seed)
    if args.n and args.n < len(groups):
        groups = [groups[j] for j in rng.choice(len(groups), args.n, replace=False)]
    print(f"[cb] {len(groups):,} positions ({time.time()-t0:.0f}s to index)", flush=True)

    elo1 = torch.from_numpy(normalise_elo(np.full((1, 2), 2750.0, np.float32)))
    sps, top1, trunc, n_mv_tot = [], [], 0, 0
    sps_clean, top1_clean = [], []
    for a0 in range(0, len(groups), args.batch):
        gb = groups[a0:a0 + args.batch]
        bs, uci_maps = {"sq": [], "gl": [], "mv": [], "mut": [], "vac": [], "dsc": [],
                        "att": [], "n_mv": [], "n_att": []}, []
        for fen, _ in gb:
            b = chess.Board(fen)
            g, _ = parent_graph(b)
            for k in ("sq", "mv", "mut", "vac", "dsc", "att", "n_mv", "n_att"):
                bs[k].append(g[k])
            bs["gl"].append(glob_of_board(b))
            uci_maps.append({m.uci(): j for j, m in
                             enumerate(list(b.legal_moves)[:int(g["n_mv"])])})
        bt = {k: torch.from_numpy(np.stack(v)).to(dev) for k, v in bs.items()}
        sqf, mvf, edges, mask = featurize(bt, dev, n_nodes=field.n_nodes)
        off = (torch.arange(len(gb), device=dev) * field.n_nodes)[:, None]
        jj = torch.arange(MV_CAP, device=dev)[None, :]
        for name in ("vac", "dsc"):
            a2 = bt[name].long()
            val = (a2 != 255) & mask[:, :, None]
            edges[name] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val],
                           (a2 + off[:, :, None])[val])
        with torch.no_grad():
            elo = elo1.repeat(len(gb), 1).to(dev)
            zq, _, _, _, h_mv, _ = field.encode(sqf, mvf, edges, mask, elo=elo)[:6]
            pred = hl.decode(sf_av(torch.cat(
                [zq[:, None, :].expand(-1, MV_CAP, -1), h_mv], -1)))
        for bi, (fen, mvwp) in enumerate(gb):
            um = uci_maps[bi]
            ys, ps = [], []
            for mv, wp in mvwp:
                j = um.get(mv)
                if j is None:
                    trunc += 1
                    continue
                ys.append(wp); ps.append(float(pred[bi, j]))
            n_mv_tot += len(mvwp)
            if len(ys) > 2:
                sp = spearmanr(ys, ps).statistic
                t1 = float(np.argmax(ys) == np.argmax(ps))
                if np.isfinite(sp):
                    sps.append(sp); top1.append(t1)
                    if " ".join(fen.split()[:4]) not in seen_keys:
                        sps_clean.append(sp); top1_clean.append(t1)
    print(f"[cb] ckpt {args.ckpt.split('/')[-1]} | positions {len(sps):,} "
          f"(clean {len(sps_clean):,}) | move-slot truncation {trunc}/{n_mv_tot} "
          f"({100*trunc/max(1,n_mv_tot):.2f}%)")
    print(f"[cb] action-value spearman {np.mean(sps):.4f} (clean {np.mean(sps_clean):.4f}) "
          f"| top1-vs-SF {np.mean(top1):.4f} (clean {np.mean(top1_clean):.4f}) "
          f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
