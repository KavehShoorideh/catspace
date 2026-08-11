#!/usr/bin/env python
"""subgoal_vq_align.py -- do the MINED subgoal clusters land in single VQ code cells?

The miner finds invariance classes top-down (game events); the quantizer finds them bottom-up
(the evaluation's own level sets). Alignment = per subgoal cluster, the concentration of its
members' codes per VQ head (top-code share vs that code's base rate).

    .venv/bin/python -m ...subgoal_vq_align --ckpt <field.pt>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.experiments.concept_vq import ConceptVQ
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import load_net
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    pay_vq = torch.load(base + "_vq.pt", map_location=args.device, weights_only=False)
    vq = ConceptVQ(d_in=pay_vq["d_in"], heads=pay_vq["heads"], codes=pay_vq["codes"]).to(args.device)
    vq.load_state_dict(pay_vq["state_dict"]); vq.eval()
    net, pay = load_net(args.ckpt, args.device)
    c = pay["cfg"]
    tr = T.build(n_human=0, n_sf=c["games"], seed=c["traj_seed"], max_plies=c["max_plies"],
                 n_piecedown=c.get("n_piecedown", 0), verbose=False)
    clusters = [json.loads(l) for l in open(paths.experiment("subgoal_candidates.jsonl"))]
    # base rates over a random sample
    rng = np.random.default_rng(0)
    rs = rng.choice(len(tr.tok), 30000, replace=False)
    def codes_of(rows):
        with torch.no_grad():
            phi = net.backbone(torch.from_numpy(tr.tok[rows].astype(np.int64)).to(args.device),
                               torch.from_numpy(tr.glob[rows].astype(np.float32)).to(args.device))
            _, ids, _ = vq(phi)
        return ids.cpu().numpy()
    base_ids = codes_of(rs)
    base_rate = [np.bincount(base_ids[:, h], minlength=pay_vq["codes"]) / len(base_ids)
                 for h in range(pay_vq["heads"])]
    print("cluster | invariant | per-head top-code share (base rate) -> concentration lift")
    for cl in clusters:
        if not cl.get("rows"):
            continue
        ids = codes_of(np.array(cl["rows"]))
        parts = []
        for h in range(pay_vq["heads"]):
            cnt = np.bincount(ids[:, h], minlength=pay_vq["codes"])
            top = int(np.argmax(cnt))
            share = cnt[top] / len(ids)
            parts.append(f"h{h}:c{top} {share:.0%}({base_rate[h][top]:.0%})")
        print(f"  c{cl['cluster']:>3} {'INV' if cl['invariant'] else '   '}  " + "  ".join(parts))


if __name__ == "__main__":
    main()
