#!/usr/bin/env python
"""name_jqt_concepts.py -- predicate naming for a JQT codebook (Kaveh 2026-08-12: "if there's
a known correlation with python-chess concepts, I want it known"). Same battery and map format
as concept_vq's naming pass (predicates are NAMING ONLY, never training), applied to the
jointly-trained quantizer in the _jqt sidecar.

    .venv/bin/python -m ...name_jqt_concepts --ckpt <field.pt> [--rows 120000]
writes <stem>_conceptmap.json
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows", type=int, default=120_000)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.experiments.concept_vq import (
        predicates_from_tok)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
        load_net)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.jqt import (
        JQTModule)
    from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T

    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    stem = re.sub(r"_(latest|step\d+)$", "", base)
    net, pay = load_net(args.ckpt, args.device)
    c = pay["cfg"]
    pj = torch.load(stem + "_jqt.pt", map_location=args.device, weights_only=False)
    jm = JQTModule(d_model=pj["d_in"], heads=pj["heads"], codes=pj["codes"], d=pj["d"],
                   square_codes=pj.get("square_codes", 0),
                   piece_codes=pj.get("piece_codes", 0)).to(args.device)
    jm.load_state_dict(pj["state_dict"]); jm.eval()
    tr = T.build(n_human=0, n_sf=c["games"], seed=c["traj_seed"], max_plies=c["max_plies"],
                 n_piecedown=c.get("n_piecedown", 0), verbose=False)
    rng = np.random.default_rng(0)
    rows = rng.choice(tr.n_positions, min(args.rows, tr.n_positions), replace=False)
    ids_all = np.empty((len(rows), pj["heads"]), np.int64)
    with torch.no_grad():
        for a in range(0, len(rows), 4096):
            rr = rows[a:a + 4096]
            phi = net.backbone(
                torch.from_numpy(tr.tok[rr].astype(np.int64)).to(args.device),
                torch.from_numpy(tr.glob[rr].astype(np.float32)).to(args.device))
            _, ids = jm.target_codes(phi)
            ids_all[a:a + 4096] = ids.cpu().numpy()
    preds = predicates_from_tok(tr.tok[rows])
    cmap = {}
    for k, pv in preds.items():
        b0 = float(np.mean(pv))
        if b0 < 0.01 or b0 > 0.99:
            continue
        best = None
        for h in range(pj["heads"]):
            for code in range(pj["codes"]):
                m = ids_all[:, h] == code
                if m.sum() < 200:
                    continue
                hit = float(np.mean(pv[m]))
                lift = hit - b0
                if best is None or abs(lift) > abs(best[3]):
                    best = (h, code, hit, lift)
        if best:
            h, code, hit, lift = best
            cmap[k] = {"head": h, "code": int(code), "p_given_code": round(hit, 3),
                       "base": round(b0, 3), "lift": round(lift, 3),
                       "anti": lift < 0}
            print(f"  {k:22s} -> h{h}/c{code:<3} P {hit:.0%} (base {b0:.0%}, lift {lift:+.0%})")
    json.dump(cmap, open(stem + "_conceptmap.json", "w"), indent=1)
    print(f"[name] {len(cmap)} named -> {stem}_conceptmap.json")


if __name__ == "__main__":
    main()
